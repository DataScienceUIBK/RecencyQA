import json
import os
import re
from termcolor import colored
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


system_prompt = """You are helping construct a temporal QA dataset where each question has a label indicating how soon its answer is expected to change. Some questions only make sense when asked in a specific situation. Your task is to generate short, realistic contexts that make the given label appropriate."""


def build_user_prompt(question, label):
    return f"""
You are given a question and its label

Question: {question}
Label: {label}

Label represents how soon the answer to that question is likely to change.

Generate 3 different, very brief but clear contextual sentences that give information about when this question is asked, and this label makes sense.
The context should ground the question in an event
Do **not** include specific years — simply create a moment where the question arises and the label makes sense.


Format your response as a JSON list, where each context is represented as:
[
{{
"Context": "<context>"
}}
]
Output ONLY the JSON list.
"""


def load_model(model_name="meta-llama/Llama-3.3-70B-Instruct"):
    print(colored(f"\n🔄 Loading model: {model_name}", "cyan"))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    print(colored(f"✓ Model loaded", "green"))
    return tokenizer, model


def generate_contexts_batch(questions_batch, labels_batch, tokenizer, model, max_retries=3):
    """
    Generate contexts for a batch of (question, label) pairs.
    
    Returns:
        List of lists, where each inner list contains 3 context dicts
    """
    # Prepare all prompts
    all_messages = []
    for question, label in zip(questions_batch, labels_batch):
        user_prompt = build_user_prompt(question, label)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        all_messages.append(messages)
    
    results = [None] * len(questions_batch)
    indices_to_retry = list(range(len(questions_batch)))
    
    for attempt in range(max_retries):
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
                truncation=True
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=2048,
                    temperature=0.1,
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
                    
                    # Validate: should be a list of 3 context objects
                    if isinstance(parsed, list) and len(parsed) == 3:
                        # Check each has "Context" key
                        if all("Context" in item for item in parsed):
                            results[original_idx] = parsed
                        else:
                            new_failures.append(original_idx)
                    else:
                        new_failures.append(original_idx)
                
                except json.JSONDecodeError:
                    new_failures.append(original_idx)
            
            indices_to_retry = new_failures
            
            if indices_to_retry and attempt < max_retries - 1:
                print(colored(f"  🔄 Retrying {len(indices_to_retry)} failed parse(s) (attempt {attempt + 2}/{max_retries})", "yellow"))
        
        except Exception as e:
            print(colored(f"  ⚠️ Batch error (attempt {attempt + 1}/{max_retries}): {e}", "yellow"))
            if attempt == max_retries - 1:
                # Fill failures with error
                for idx in indices_to_retry:
                    results[idx] = [
                        {"Context": "[ERROR: Failed to generate context]"},
                        {"Context": "[ERROR: Failed to generate context]"},
                        {"Context": "[ERROR: Failed to generate context]"}
                    ]
                indices_to_retry = []
    
    # Fill any remaining failures
    for idx in indices_to_retry:
        if results[idx] is None:
            results[idx] = [
                {"Context": "[ERROR: Failed after retries]"},
                {"Context": "[ERROR: Failed after retries]"},
                {"Context": "[ERROR: Failed after retries]"}
            ]
    
    return results


def load_processed_questions(output_file):
    if not os.path.exists(output_file):
        return set(), []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if question has contexts generated for all unique labels
        processed_ids = set()
        for item in data:
            if 'label_contexts' in item:
                # Check if all unique labels have contexts
                unique_labels = item.get('unique_labels', [])
                label_contexts = item.get('label_contexts', {})
                if all(label in label_contexts for label in unique_labels):
                    processed_ids.add(item['q_id'])
        
        print(colored(f"✓ Found {len(processed_ids)} already processed questions", "yellow"))
        return processed_ids, data
    except Exception as e:
        print(colored(f"⚠️  Error reading output file: {e}", "red"))
        return set(), []


def generate_contexts_for_dataset(input_file, output_file, model_name="meta-llama/Llama-3.3-70B-Instruct", 
                                   batch_size=8):
    """
    Generate contexts for each unique label in each question.
    
    For each question with unique_labels = ["A-Year", "Never"], this will:
    1. Generate 3 contexts for "A-Year" 
    2. Generate 3 contexts for "Never"
    3. Store as: label_contexts = {"A-Year": [...], "Never": [...]}
    """
    
    try:

        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        print(colored(f"✓ Loaded {len(input_data)} questions from {input_file}", "green"))
        print(colored(f"📦 Batch size: {batch_size}", "cyan"))
        

        output_dir = os.path.dirname(output_file)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        processed_ids, output_data = load_processed_questions(output_file)
        
        output_map = {item['q_id']: item for item in output_data}
        
        remaining = [q for q in input_data if q['q_id'] not in processed_ids]
        question_lookup = {q["q_id"]: q for q in remaining}

        print(colored(f"✅ Already processed: {len(processed_ids)}", "yellow"))
                
        if not remaining:
            print(colored("\n✅ All questions already processed!", "green"))
            return output_data
        
        print(colored(f"\n🔄 Processing {len(remaining)} remaining questions...\n", "cyan"))
        
        tokenizer, model = load_model(model_name)
        
        # Flatten: create (question, label) pairs for batching
        batch_items = []
        for question_data in remaining:
            unique_labels = question_data.get('unique_labels', [])
            for label in unique_labels:
                batch_items.append({
                    'q_id': question_data['q_id'],
                    'question': question_data['question'],
                    'label': label,
                    'question_data': question_data
                })
        
        print(colored(f"📊 Total (question, label) pairs to process: {len(batch_items)}", "cyan"))
        
        num_batches = (len(batch_items) + batch_size - 1) // batch_size
        
        # Store contexts temporarily by q_id
        temp_contexts = {}
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(batch_items))
            batch = batch_items[start_idx:end_idx]
            
            questions_batch = [item['question'] for item in batch]
            labels_batch = [item['label'] for item in batch]
            
            contexts_batch = generate_contexts_batch(questions_batch, labels_batch, tokenizer, model)
            
            for item, contexts in zip(batch, contexts_batch):
                q_id = item['q_id']
                label = item['label']
                
                if q_id not in temp_contexts:
                    temp_contexts[q_id] = {}
                
                temp_contexts[q_id][label] = contexts

            completed_questions = []
            
            for q_id, label_contexts in temp_contexts.items():
                question_data = question_lookup.get(q_id)
                if question_data is None:
                    continue
                                
                unique_labels = question_data.get('unique_labels', [])
                
                if all(label in label_contexts for label in unique_labels):
                    # Add contexts to question data
                    question_data['label_contexts'] = label_contexts
                    
                    if q_id in output_map:
                        output_map[q_id].update(question_data)
                    else:
                        output_data.append(question_data)
                        output_map[q_id] = question_data
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)

                    completed_questions.append(q_id)

            for q_id in completed_questions:
                del temp_contexts[q_id]
        
        print(colored(f"\n✅ Context generation complete!", "green", attrs=["bold"]))
        print(colored(f"💾 Saved to: {output_file}", "green"))
        print(colored(f"📊 Processed {len(output_data)} questions", "cyan"))
        
        print_statistics(output_data)
        
        return output_data
    
    except Exception as e:
        print(colored(f"❌ Error: {e}", "red"))
        import traceback
        traceback.print_exc()
        return None


def print_statistics(data):

    print(colored("\n📊 Context Generation Statistics:", "cyan", attrs=["bold"]))
    
    questions_with_contexts = sum(1 for item in data if 'label_contexts' in item)
    total_labels = sum(len(item.get('label_contexts', {})) for item in data)
    total_contexts = sum(
        sum(len(contexts) for contexts in item.get('label_contexts', {}).values())
        for item in data
    )
    
    print(colored(f"\n  Questions with contexts: {questions_with_contexts}/{len(data)}", "green"))
    print(colored(f"  Total labels with contexts: {total_labels}", "cyan"))
    print(colored(f"  Total contexts generated: {total_contexts}", "cyan"))
    
    example = next((item for item in data if 'label_contexts' in item), None)
    if example:
        print(colored("\n📋 Example:", "cyan", attrs=["bold"]))
        print(colored(f"\n  Q{example['q_id']}: {example['question']}", "white"))
        print(colored(f"  Unique labels: {example['unique_labels']}", "yellow"))
        
        for label, contexts in example.get('label_contexts', {}).items():
            print(colored(f"\n  Label: {label}", "green", attrs=["bold"]))
            for i, ctx in enumerate(contexts, 1):
                print(colored(f"    {i}. {ctx.get('Context', 'N/A')}", "white"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate contexts for each unique label in questions"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file"
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
        help="Number of (question, label) pairs per batch (default: 8)"
    )
    
    args = parser.parse_args()
    
    print(colored("\n" + "="*80, "magenta"))
    print(colored("  CONTEXT GENERATION FOR UNIQUE LABELS", "magenta", attrs=["bold"]))
    print(colored("="*80 + "\n", "magenta"))
    
    generate_contexts_for_dataset(
        args.input,
        args.output,
        args.model,
        args.batch_size
    )
    
