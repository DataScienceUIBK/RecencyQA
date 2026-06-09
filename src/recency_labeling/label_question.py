import json
import os
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from termcolor import colored
import traceback
import random

NUM_PROMPTS = 13  # Number of times to prompt for each question
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE = 3
MAX_RETRIES = 2  # Number of retries for failed parses
PROGRESS_LOG_INTERVAL = 10  # Log progress every N questions

# Valid labels for validation
VALID_LABELS = {
    "An-Hour",
    "A-Few-Hours", 
    "A-Day",
    "A-Few-Days",
    "A-Week",
    "A-Few-Weeks",
    "A-Month",
    "A-Few-Months",
    "A-Year",
    "A-Few-Years",
    "Many-Years",
    "Never"
}

system_prompt = """You are an analyst. Your task is to assign a label to a question. The label should reflect when you expect the answer to this question to change for the first time, based on the nature of the information it requires."""

def build_user_prompt(question):
    return f"""
Based on the question provided below:
Question: {question}

(a) Assign a label to the question based on when you expect the answer to change for the first time.

Consider how soon you expect the answer to change for the first time.

- An-Hour — The answer is likely to change within an hour
- A-Few-Hours — The answer is likely to change within a few hours
- A-Day — The answer is likely to change within a day
- A-Few-Days — The answer is likely to change within a few days
- A-Week — The answer is likely to change within a week
- A-Few-Weeks — The answer is likely to change within a few weeks
- A-Month — The answer is likely to change within a month
- A-Few-Months — The answer is likely to change within a few months
- A-Year — The answer is likely to change within a year
- A-Few-Years — The answer is likely to change within a few years 
- Many-Years — The answer is likely to change within many years  
- Never — The answer is not likely to ever change.

(b) Provide your reasoning for the label you assigned.
Explain when and why you expect this information to change, or state why you believe it will never change.
Please limit your explanation to 2-3 concise sentences.

Format your response ONLY as a JSON object with this structure:
{{
"Label": "<label>",
"Reasoning": "<reasoning>"
}}

Output ONLY the JSON object, nothing else.
"""


class QuestionLabeler:
    """Class to handle question labeling with batched inference and retries"""
    
    def __init__(self, model_name, batch_size=DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer"""
        print(colored("Loading model and tokenizer...", "cyan", attrs=["bold"]))
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Set padding side for decoder-only models
        self.tokenizer.padding_side = "left"
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(colored(f"Model loaded on {DEVICE}\n", "green"))
    
    def build_model_prompt(self, user_prompt):
        """Build prompt using tokenizer's chat template"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            return f"{system_prompt}\n\n{user_prompt}\n\n"
    
    def extract_json_from_response(self, response_text):
        """Extract and validate JSON from model response"""
        # Try to find JSON object in the response
        match = re.search(r'\{[\s\S]*?\}', response_text)
        if match:
            json_str = match.group()
            try:
                parsed = json.loads(json_str)
                
                # Validate label
                label = parsed.get("Label")
                if label not in VALID_LABELS:
                    # Try case-insensitive match
                    label_lower = label.lower() if label else ""
                    for valid_label in VALID_LABELS:
                        if valid_label.lower() == label_lower:
                            parsed["Label"] = valid_label
                            return parsed
                    
                    tqdm.write(colored(f"  ⚠️ Invalid label: {label}", "yellow"))
                    return None
                
                return parsed
            except json.JSONDecodeError as e:
                tqdm.write(colored(f"  ⚠️ JSON decode error: {e}", "yellow"))
                return None
        return None
    
    def generate_batch(self, prompts, temperature=0.7):
        """Generate responses for a batch of prompts"""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        responses = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[-1]
            raw_response = self.tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            ).strip()
            responses.append(raw_response)
        
        return responses
    
    def get_labels_batch(self, question, attempt_ids):
        """Generate labels for multiple independent prompts with retry logic"""
        
        # Build prompts for all attempts
        prompts = []
        for _ in attempt_ids:
            user_prompt = build_user_prompt(question)
            full_prompt = self.build_model_prompt(user_prompt)
            prompts.append(full_prompt)

        # Generate initial responses
        raw_responses = self.generate_batch(prompts, temperature=0.7)
        
        results = []
        failed_indices = []
        
        # Parse responses and identify failures
        for i, raw_response in enumerate(raw_responses):
            parsed = self.extract_json_from_response(raw_response)
            if parsed:
                results.append({
                    "Label": parsed.get("Label"),
                    "Reasoning": parsed.get("Reasoning"),
                    "attempt_num": attempt_ids[i],
                    "retried": False
                })
            else:
                failed_indices.append(i)
                results.append(None)  # Placeholder
        
        # Retry failed parses with lower temperature
        if failed_indices and MAX_RETRIES > 0:
            for retry_num in range(MAX_RETRIES):
                if not failed_indices:
                    break
                    
                tqdm.write(colored(f"  🔄 Retrying {len(failed_indices)} failed parse(s) (attempt {retry_num + 1}/{MAX_RETRIES})...", "yellow"))
                
                retry_prompts = [prompts[i] for i in failed_indices]
                # Use lower temperature for retries
                retry_temp = 0.3 if retry_num == 0 else 0.1
                retry_responses = self.generate_batch(retry_prompts, temperature=retry_temp)
                
                still_failed = []
                for j, idx in enumerate(failed_indices):
                    parsed = self.extract_json_from_response(retry_responses[j])
                    if parsed:
                        results[idx] = {
                            "Label": parsed.get("Label"),
                            "Reasoning": parsed.get("Reasoning"),
                            "attempt_num": attempt_ids[idx],
                            "retried": True,
                            "retry_count": retry_num + 1
                        }
                    else:
                        still_failed.append(idx)
                
                failed_indices = still_failed
        
        # Fill in any remaining failures
        for idx in failed_indices:
            results[idx] = {
                "Label": "[ERROR]",
                "Reasoning": "Failed to parse JSON after retries",
                "attempt_num": attempt_ids[idx],
                "retried": True,
                "retry_count": MAX_RETRIES
            }
        
        return results
    
    def get_majority_label(self, labels):
        """Determine the majority label from a list of labels"""
        # Filter out None and [ERROR] values
        valid_labels = [label for label in labels if label is not None and label != "[ERROR]"]
        
        if not valid_labels:
            return None, {}
        
        # Count occurrences
        label_counts = Counter(valid_labels)
        
        # Get the most common label
        majority_label = label_counts.most_common(1)[0][0]
        
        # Calculate distribution
        total = len(valid_labels)
        distribution = {label: count/total for label, count in label_counts.items()}
        
        return majority_label, distribution
    
    def process_question(self, question_data, question_idx, pbar=None):
        """Process a single question with batched generation"""
        question = question_data["question"]
        q_id = question_data["q_id"]

        msg = f"Processing Q{q_id}: {question[:60]}{'...' if len(question) > 60 else ''}"
        if pbar:
            pbar.set_description(msg)

        all_responses = []
        all_labels = []
        retry_count = 0

        attempt_counter = 1

        while attempt_counter <= NUM_PROMPTS:
            current_batch_size = min(self.batch_size, NUM_PROMPTS - attempt_counter + 1)
            attempt_ids = list(range(attempt_counter, attempt_counter + current_batch_size))

            batch_results = self.get_labels_batch(question, attempt_ids)

            all_responses.extend(batch_results)
            all_labels.extend([r["Label"] for r in batch_results])
            retry_count += sum(1 for r in batch_results if r.get("retried", False))

            attempt_counter += current_batch_size

        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Majority vote
        majority_label, distribution = self.get_majority_label(all_labels)
        
        # Count errors
        error_count = sum(1 for label in all_labels if label == "[ERROR]")

        result = {
            "q_id": q_id,
            "question": question,
            "source": question_data.get("source"),
            "hop_type": question_data.get("hop_type"),
            "majority_label": majority_label,
            "label_distribution": distribution,
            "all_labels": all_labels,
            "all_responses": all_responses,
            "num_prompts": NUM_PROMPTS,
            "error_count": error_count,
            "retry_count": retry_count
        }

        # Log results
        tqdm.write(colored(f"  Q{q_id} → ", "white") + 
                   colored(majority_label or "[NO CONSENSUS]", "green", attrs=["bold"]) +
                   colored(f" | Errors: {error_count}, Retries: {retry_count}", "cyan"))

        return result


# ============================================================================
# File I/O Functions
# ============================================================================

def append_to_jsonl(data, filepath):
    """Append a single JSON object to a JSONL file"""
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        tqdm.write(colored(f"❌ Failed to write to {filepath}: {e}", "red"))
        return False


def load_jsonl(filepath):
    """Load all entries from a JSONL file"""
    entries = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def load_processed_ids(output_file):
    """Load already processed question IDs from JSONL file"""
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            for result in load_jsonl(output_file):
                processed_ids.add(result["q_id"])
            print(colored(f"📂 Found {len(processed_ids)} already processed questions", "yellow"))
        except Exception as e:
            print(colored(f"⚠️ Error reading existing results: {e}", "red"))
    return processed_ids


def load_failed_ids(failed_file):
    """Load failed question IDs from JSONL file"""
    failed_ids = set()
    if os.path.exists(failed_file):
        try:
            for entry in load_jsonl(failed_file):
                failed_ids.add(entry["q_id"])
            print(colored(f"📂 Found {len(failed_ids)} previously failed questions", "yellow"))
        except Exception as e:
            print(colored(f"⚠️ Error reading failed questions: {e}", "red"))
    return failed_ids


def convert_jsonl_to_json(jsonl_file, json_file):
    """Convert JSONL output to standard JSON array format"""
    results = load_jsonl(jsonl_file)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    return len(results)


def save_failed_question(question_data, failed_file, reason, error_details=None):
    """Save a failed question to the failed questions file for reprocessing"""
    failed_entry = {
        "q_id": question_data.get("q_id"),
        "question": question_data.get("question"),
        "source": question_data.get("source"),
        "hop_type": question_data.get("hop_type"),
        "failure_reason": reason,
        "error_details": error_details,
        "failed_at": datetime.now().isoformat(),
        "gpu_device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    }
    append_to_jsonl(failed_entry, failed_file)
    tqdm.write(colored(f"  ❌ Q{question_data.get('q_id')} saved to failed questions", "red"))


# ============================================================================
# Main Processing Function
# ============================================================================

def process_dataset(input_file, output_file, batch_size, source_name, model_name, output_dir):
    """Process entire dataset with incremental writing and failed question tracking"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize labeler
    labeler = QuestionLabeler(model_name, batch_size=batch_size)
    labeler.load_model()
    
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # Setup output files
    base_name = output_file.replace('.json', '')
    jsonl_output = os.path.join(output_dir, f"{base_name}.jsonl")
    final_json_output = os.path.join(output_dir, output_file)
    failed_file = os.path.join(output_dir, f"{base_name}_failed.jsonl")
    
    # Load already processed IDs (resume support)
    processed_ids = load_processed_ids(jsonl_output)
    
    # Filter out already processed
    remaining = [q for q in questions_data if q["q_id"] not in processed_ids]
    
    if not remaining:
        print(colored("All questions already processed!", "green"))
    else:
        print(colored(f"Processing {len(remaining)} remaining questions...\n", "cyan"))
    
    # Track stats incrementally (lightweight - just counts)
    processed_count = len(processed_ids)
    total_errors = 0
    total_retries = 0
    failed_count = 0
    
    # Process each question with progress bar
    with tqdm(remaining, desc="Processing", unit="question") as pbar:
        for idx, question_data in enumerate(pbar):
            try:
                # Process question with batching
                result = labeler.process_question(question_data, idx, pbar)
                result["source"] = source_name  # Add source to result
                
                # Check if question completely failed (no valid labels at all)
                if result.get("majority_label") is None and result.get("error_count", 0) == NUM_PROMPTS:
                    # All attempts failed - save to failed file
                    save_failed_question(
                        question_data, 
                        failed_file, 
                        reason="all_attempts_failed",
                        error_details=f"All {NUM_PROMPTS} labeling attempts failed"
                    )
                    failed_count += 1
                else:
                    # Write successful result to JSONL
                    append_to_jsonl(result, jsonl_output)
                    processed_count += 1
                
                # Update stats
                total_errors += result.get("error_count", 0)
                total_retries += result.get("retry_count", 0)
                
                # Periodic status
                if processed_count % PROGRESS_LOG_INTERVAL == 0:
                    tqdm.write(colored(f"💾 Progress: {processed_count} saved, {failed_count} failed", "green"))
                    
            except torch.cuda.OutOfMemoryError as e:
                # GPU OOM - save to failed and continue
                tqdm.write(colored(f"  ⚠️ CUDA OOM for Q{question_data.get('q_id')}", "red"))
                save_failed_question(
                    question_data,
                    failed_file,
                    reason="cuda_oom",
                    error_details=str(e)
                )
                failed_count += 1
                torch.cuda.empty_cache()
                
            except Exception as e:
                # Other errors - save to failed and continue
                tqdm.write(colored(f"  ⚠️ Error processing Q{question_data.get('q_id')}: {e}", "red"))
                save_failed_question(
                    question_data,
                    failed_file,
                    reason="processing_error",
                    error_details=traceback.format_exc()
                )
                failed_count += 1
    
    # Convert JSONL to JSON for final output
    print(colored("\n📄 Converting to final JSON format...", "cyan"))
    total_count = convert_jsonl_to_json(jsonl_output, final_json_output)
    
    # Load results only for final stats (one-time read)
    results = load_jsonl(jsonl_output)
    
    # Print summary
    print(colored("\n" + "=" * 60, "magenta"))
    print(colored("✅ PROCESSING COMPLETE", "green", attrs=["bold"]))
    print(colored("=" * 60, "magenta"))
    print(colored(f"  Total processed: {total_count}", "cyan"))
    print(colored(f"  Total failed: {failed_count}", "red" if failed_count > 0 else "green"))
    print(colored(f"  JSONL output: {jsonl_output}", "cyan"))
    print(colored(f"  JSON output: {final_json_output}", "cyan"))
    if failed_count > 0:
        print(colored(f"  Failed questions: {failed_file}", "yellow"))
    
    # Calculate final stats from loaded results
    total_errors = sum(r.get("error_count", 0) for r in results)
    total_retries = sum(r.get("retry_count", 0) for r in results)
    no_consensus = sum(1 for r in results if r.get("majority_label") is None)
    
    print(colored(f"\n  Parse errors: {total_errors}", "yellow" if total_errors > 0 else "green"))
    print(colored(f"  Total retries: {total_retries}", "yellow" if total_retries > 0 else "green"))
    print(colored(f"  No consensus: {no_consensus}", "yellow" if no_consensus > 0 else "green"))
    
    # Label distribution summary
    if results:
        label_counts = Counter(r.get("majority_label") for r in results if r.get("majority_label"))
        print(colored("\n  Label Distribution:", "cyan", attrs=["bold"]))
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = count / total_count * 100
            print(colored(f"    {label}: {count} ({pct:.1f}%)", "white"))
    
    # Reminder about failed questions
    if failed_count > 0:
        print(colored(f"\n⚠️  {failed_count} questions failed and saved to: {failed_file}", "yellow"))
        print(colored(f"   Reprocess with: python label_questions.py --input {failed_file} --output {base_name}_retry.json", "yellow"))

    
# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Label questions with majority voting (batched)")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file name")
    parser.add_argument("--source", type=str, default="unknown", help="Source name for logging")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="HF model name"  )
    parser.add_argument("--output-dir", type=str, default="outputs/recency_labels", help="Directory where outputs will be saved")
    parser.add_argument("--seed", type=int, default=None, help="Random seed" )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for parallel processing")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS, help="Number of prompts per question")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Max retries for failed parses")
    
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
          
    # Update globals based on arguments
    NUM_PROMPTS = args.num_prompts
    MAX_RETRIES = args.max_retries
    
    print(colored("\n" + "=" * 60, "magenta"))
    print(colored("  QUESTION LABELING WITH MAJORITY VOTING", "magenta", attrs=["bold"]))
    print(colored("=" * 60, "magenta"))
    print(colored(f"  Source: {args.source}", "cyan"))
    print(colored(f"  Model: {args.model}", "cyan"))
    print(colored(f"  Prompts per question: {NUM_PROMPTS}", "cyan"))
    print(colored(f"  Batch size: {args.batch_size}", "cyan"))
    print(colored(f"  Max retries: {MAX_RETRIES}", "cyan"))
    print(colored(f"  Input: {args.input}", "cyan"))
    print(colored(f"  Output: {args.output}", "cyan"))
    print(colored(f"  Seed: {args.seed}", "cyan"))
    print(colored(f"  Output Dir: {args.output_dir}", "cyan"))  
    print(colored(f"  Writing: Incremental (JSONL) → Final (JSON)", "cyan"))
    print(colored(f"  Failed questions: Saved separately for reprocessing", "cyan"))
    print(colored("=" * 60 + "\n", "magenta"))
    
    process_dataset(args.input, args.output, args.batch_size, args.source, args.model, args.output_dir)
