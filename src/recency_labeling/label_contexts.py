import json
import os
from termcolor import colored
import argparse
from tqdm import tqdm
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


system_prompt = """You are an analyst. Your task is to assign a label to a question, given a specific context. The label should reflect when you expect the answer to this question to change for the first time, based on the nature of the information it requires and the context in which it is asked."""


def build_user_prompt(question, context):
    return f"""
Based on the question and the context provided below:
Question: {question}
Context: {context}

(a) Assign a label to the question based on when you expect the answer to change for the first time, given the context.

Consider **how soon** you expect the answer to change **for the first time**

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

(b) Provide your justification for the label you assigned.
Explain when and why you expect this information to change, or state why you believe it will never change.
Please limit your explanation to 2-3 concise sentences.

Format your response as a JSON list, where each output is represented as:
[
{{
"Label": "<label>",
"Justification": "<justification>"
}}
]
"""


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


def load_model(model_name="meta-llama/Llama-3.3-70B-Instruct"):
    """Load model from HuggingFace."""
    print(colored(f"\n🔄 Loading model: {model_name}", "cyan"))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(colored(f"✓ Model loaded", "green"))
    return tokenizer, model


def validate_label(label):
    """Check if label is valid."""
    if label in VALID_LABELS:
        return True
    
    # Try case-insensitive match
    label_lower = label.lower() if label else ""
    for valid_label in VALID_LABELS:
        if valid_label.lower() == label_lower:
            return True
    
    return False


def label_contexts_batch(questions_batch, contexts_batch, tokenizer, model, num_prompts=13, max_retries=3):
    """
    Generate labels for a batch of (question, context) pairs, prompting each NUM_PROMPTS times.
    
    Args:
        questions_batch: List of question texts
        contexts_batch: List of context texts
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        num_prompts: Number of times to prompt for each (question, context) pair
        max_retries: Number of retry attempts for failed parses
    
    Returns:
        List of dicts with all labels, majority label, distribution, etc.
    """
    from collections import Counter
    import math
    
    # Expand batch to repeat each (question, context) pair num_prompts times
    expanded_questions = []
    expanded_contexts = []
    original_indices = []
    
    for idx, (question, context) in enumerate(zip(questions_batch, contexts_batch)):
        for attempt in range(num_prompts):
            expanded_questions.append(question)
            expanded_contexts.append(context)
            original_indices.append(idx)
    
    # Prepare all prompts
    all_messages = []
    for question, context in zip(expanded_questions, expanded_contexts):
        user_prompt = build_user_prompt(question, context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        all_messages.append(messages)
    
    # Store results for all attempts
    attempt_results = [None] * len(expanded_questions)
    indices_to_retry = list(range(len(expanded_questions)))
    
    for retry_attempt in range(max_retries):
        if not indices_to_retry:
            break
        
        # Get prompts for items that need (re)trying
        current_messages = [all_messages[i] for i in indices_to_retry]
        
        try:
            texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for messages in current_messages
            ]
            
            model_inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=1536
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            new_failures = []
            for i, response in enumerate(responses):
                original_idx = indices_to_retry[i]
                
                try:
                    # Extract JSON
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0].strip()
                    elif "```" in response:
                        response = response.split("```")[1].split("```")[0].strip()
                    
                    match = re.search(r"\[[\s\S]*\]", response)

                    if not match:
                        raise json.JSONDecodeError(
                            "No JSON list found",
                            response,
                            0
                        )
                    
                    parsed = json.loads(match.group())
                    
                    # Handle both list and dict formats
                    if isinstance(parsed, list) and len(parsed) > 0:
                        label = parsed[0].get("Label")
                        justification = parsed[0].get("Justification")
                    elif isinstance(parsed, dict):
                        label = parsed.get("Label")
                        justification = parsed.get("Justification")
                    else:
                        new_failures.append(original_idx)
                        continue
                    
                    if not validate_label(label):
                        print(colored(f"  ⚠️ Invalid label: {label}", "yellow"))
                        new_failures.append(original_idx)
                        continue
                    
                    attempt_results[original_idx] = {
                        "label": label,
                        "justification": justification,
                        "retried": retry_attempt > 0,
                        "retry_count": retry_attempt
                    }
                
                except json.JSONDecodeError:
                    new_failures.append(original_idx)
            
            indices_to_retry = new_failures
            
            if indices_to_retry and retry_attempt < max_retries - 1:
                print(colored(f"  🔄 Retrying {len(indices_to_retry)} failed parse(s) (attempt {retry_attempt + 2}/{max_retries})", "yellow"))
        
        except Exception as e:
            print(colored(f"  ⚠️ Batch error (attempt {retry_attempt + 1}/{max_retries}): {e}", "yellow"))
            if retry_attempt == max_retries - 1:
                # Fill failures with error
                for idx in indices_to_retry:
                    attempt_results[idx] = {
                        "label": "[ERROR]",
                        "justification": f"Batch error: {str(e)}",
                        "retried": True,
                        "retry_count": max_retries
                    }
                indices_to_retry = []
    
    # Fill any remaining failures
    for idx in indices_to_retry:
        if attempt_results[idx] is None:
            attempt_results[idx] = {
                "label": "[ERROR]",
                "justification": f"Failed after {max_retries} attempts",
                "retried": True,
                "retry_count": max_retries
            }
    
    # Aggregate results by original (question, context) pair
    final_results = []
    for orig_idx in range(len(questions_batch)):
        # Get all attempts for this (question, context) pair
        pair_attempts = []
        for i, orig_i in enumerate(original_indices):
            if orig_i == orig_idx:
                pair_attempts.append(attempt_results[i])
        
        # Collect all labels and responses
        all_labels = [attempt["label"] for attempt in pair_attempts]
        all_responses = pair_attempts
        
        # Calculate statistics
        valid_labels = [label for label in all_labels if label != "[ERROR]"]
        error_count = sum(1 for label in all_labels if label == "[ERROR]")
        retry_count = sum(1 for attempt in pair_attempts if attempt.get("retried", False))

        final_results.append({
            "all_labels": all_labels,
            "all_responses": all_responses
        })
    
    return final_results


def load_processed_questions(output_file):
    """Load already processed question-context pairs."""
    if not os.path.exists(output_file):
        return set(), []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Track which (q_id, label, context_idx) tuples are processed
        processed_tuples = set()
        for item in data:
            if 'context_labels' in item:
                q_id = item['q_id']
                for label_key, contexts_with_labels in item['context_labels'].items():
                    for idx, context_data in enumerate(contexts_with_labels):
                        if 'contextual_all_labels' in context_data:
                            processed_tuples.add((q_id, label_key, idx))
        
        print(colored(f"✓ Found {len(processed_tuples)} already processed (question, context) pairs", "yellow"))
        return processed_tuples, data
    except Exception as e:
        print(colored(f"⚠️  Error reading output file: {e}", "red"))
        return set(), []


def label_contexts_for_dataset(input_file, output_file, model_name="meta-llama/Llama-3.3-70B-Instruct", 
                                batch_size=8, num_prompts=13):
    
    try:
        # Load model
        # tokenizer, model = load_model(model_name)
        
        # Load input data
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        print(colored(f"✓ Loaded {len(input_data)} questions from {input_file}", "green"))
        print(colored(f"📦 Batch size: {batch_size}", "cyan"))
        print(colored(f"🔢 Prompts per context: {num_prompts}", "cyan"))

        output_dir = os.path.dirname(output_file)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load existing output
        processed_tuples, output_data = load_processed_questions(output_file)
          
        # Create output map
        output_map = {item['q_id']: item for item in output_data}
        
        # Prepare all (question, context) pairs that need processing
        batch_items = []
        for question_data in input_data:
            q_id = question_data['q_id']
            question = question_data['question']
            label_contexts = question_data.get('label_contexts', {})
            
            for original_label, contexts in label_contexts.items():
                for ctx_idx, context_obj in enumerate(contexts):
                    # Check if already processed
                    if (q_id, original_label, ctx_idx) in processed_tuples:
                        continue
                    
                    context_text = context_obj.get('Context', '')
                    if not context_text or '[ERROR' in context_text:
                        continue
                    
                    batch_items.append({
                        'q_id': q_id,
                        'question': question,
                        'original_label': original_label,
                        'context_idx': ctx_idx,
                        'context': context_text,
                        'question_data': question_data
                    })
        
        if not batch_items:
            print(colored("\n✅ All contexts already labeled!", "green"))
            return output_data
        
        print(colored(f"\n📊 Total (question, context) pairs to label: {len(batch_items)}", "cyan"))
        
        tokenizer, model = load_model(model_name)

        num_batches = (len(batch_items) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(batch_items))
            batch = batch_items[start_idx:end_idx]
            
            # Prepare batch
            questions_batch = [item['question'] for item in batch]
            contexts_batch = [item['context'] for item in batch]
            
            # Generate labels (13 times per context)
            labels_batch = label_contexts_batch(questions_batch, contexts_batch, tokenizer, model, num_prompts)
            
            # Store results and save after each question is complete
            questions_to_save = set()
            
            for item, label_result in zip(batch, labels_batch):
                q_id = item['q_id']
                original_label = item['original_label']
                ctx_idx = item['context_idx']
                
                # Get or create question in output
                if q_id not in output_map:
                    output_map[q_id] = item['question_data'].copy()
                    output_data.append(output_map[q_id])
                
                question_output = output_map[q_id]
                
                # Initialize context_labels if not present
                if 'context_labels' not in question_output:
                    question_output['context_labels'] = {}
                
                if original_label not in question_output['context_labels']:
                    # Copy contexts from label_contexts
                    label_contexts = question_output.get('label_contexts', {}).get(original_label, [])
                    question_output['context_labels'][original_label] = [ctx.copy() for ctx in label_contexts]
                
                # Add labels, statistics, and all responses to the specific context
                context_list = question_output['context_labels'][original_label]
                if ctx_idx < len(context_list):
                    context_list[ctx_idx]['contextual_all_labels'] = label_result['all_labels']
                    context_list[ctx_idx]['contextual_all_responses'] = label_result['all_responses']
                questions_to_save.add(q_id)
            
            try:
                temp_file = output_file + '.tmp'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                os.replace(temp_file, output_file)
            except Exception as save_error:
                print(colored(f"  ⚠️ Save error: {save_error}", "red"))
        
        # Final save
        print(colored(f"\n💾 Writing final output...", "cyan"))
        try:
            temp_file = output_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, output_file)
            print(colored(f"✓ Final save successful", "green"))
        except Exception as e:
            print(colored(f"❌ Final save error: {e}", "red"))
        
        print(colored(f"\n✅ Context labeling complete!", "green", attrs=["bold"]))
        print(colored(f"💾 Saved to: {output_file}", "green"))
        
        # Show statistics
        print_statistics(output_data)
        
        return output_data
    
    except Exception as e:
        print(colored(f"❌ Error: {e}", "red"))
        import traceback
        traceback.print_exc()
        return None


def print_statistics(data):
    """Print statistics about labeled contexts."""
    print(colored("\n📊 Context Labeling Statistics:", "cyan", attrs=["bold"]))
    
    questions_with_labels = sum(1 for item in data if 'context_labels' in item)
    total_contexts = 0
    total_labeled = 0
    total_errors = 0
    
    for item in data:
        context_labels = item.get('context_labels', {})
        for label_key, contexts in context_labels.items():
            total_contexts += len(contexts)
            total_labeled += sum(1 for ctx in contexts if 'contextual_all_labels' in ctx)
            total_errors += sum(1 for ctx in contexts if ctx.get('contextual_error_count', 0) > 0)
    
    print(colored(f"\n  Questions with labeled contexts: {questions_with_labels}/{len(data)}", "green"))
    print(colored(f"  Total contexts labeled: {total_labeled}/{total_contexts}", "cyan"))
    print(colored(f"  Errors: {total_errors}", "red" if total_errors > 0 else "green"))
    
    # Show example
    example = next((item for item in data if 'context_labels' in item), None)
    if example:
        print(colored("\n📋 Example:", "cyan", attrs=["bold"]))
        print(colored(f"\n  Q{example['q_id']}: {example['question']}", "white"))
        
        for original_label, contexts in list(example.get('context_labels', {}).items())[:1]:
            print(colored(f"\n  Original Label: {original_label}", "yellow"))
            for i, ctx in enumerate(contexts[:2], 1):
                print(colored(f"       All labels: {ctx.get('contextual_all_labels', [])}", "white"))
                if ctx.get('contextual_all_responses'):
                    first_response = ctx['contextual_all_responses'][0]
                    print(colored(f"       First justification: {first_response.get('justification', 'N/A')[:100]}...", "white"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label each context with temporal labels"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file with label_contexts"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of (question, context) pairs per batch (default: 8)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=13,
        help="Number of times to prompt each context (default: 13)"
    )
    
    args = parser.parse_args()
    
    print(colored("\n" + "="*80, "magenta"))
    print(colored("  CONTEXT LABELING", "magenta", attrs=["bold"]))
    print(colored("="*80 + "\n", "magenta"))
    
    label_contexts_for_dataset(
        args.input,
        args.output,
        args.model,
        args.batch_size,
        args.num_prompts
    )
