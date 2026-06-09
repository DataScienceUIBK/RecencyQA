import json
import os
import re
import traceback
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from termcolor import colored


DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_BATCH_SIZE = 2
MAX_RETRIES = 2


SYSTEM_PROMPT = """You are helping construct a temporal QA dataset where each question has a label indicating how soon its answer is expected to change. Some questions only make sense when asked in a specific situation. Your task is to generate short, realistic contexts that make the given label appropriate."""


def build_user_prompt(question, label):
    return f"""
You are given a question and its label.

Question: {question}
Label: {label}

Label represents how soon the answer to that question is likely to change.

Generate 3 different very brief but clear contextual sentences that give information about when this question is asked and make this label appropriate.
The context should ground the question in an event.
Do not include specific years — simply create a moment where the question arises and the label makes sense.

Format your response as a JSON list:
[
  {{
    "Context": "<context>"
  }}
]

Output ONLY the JSON list, nothing else.
"""


class ContextGenerator:
    def __init__(self, model_name, batch_size=DEFAULT_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def load_model(self):
        print(colored("Loading model and tokenizer...", "cyan", attrs=["bold"]))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print(colored("Model loaded.\n", "green"))

    def build_model_prompt(self, user_prompt):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        return f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\n"

    def generate_batch(self, prompts, temperature=0.7):
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
            response = self.tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            ).strip()
            responses.append(response)

        return responses

    def extract_contexts(self, response_text):
        match = re.search(r"\[[\s\S]*\]", response_text)

        if not match:
            return None

        try:
            parsed = json.loads(match.group())

            contexts = []
            for item in parsed:
                context = item.get("Context", "").strip()
                if context:
                    contexts.append({"Context": context})

            if len(contexts) == 0:
                return None

            return contexts

        except Exception:
            return None

    def generate_contexts_for_pair(self, question, label):
        user_prompt = build_user_prompt(question, label)
        full_prompt = self.build_model_prompt(user_prompt)

        for retry in range(MAX_RETRIES + 1):
            temperature = 0.7 if retry == 0 else 0.3
            response = self.generate_batch([full_prompt], temperature=temperature)[0]
            contexts = self.extract_contexts(response)

            if contexts:
                return contexts, response, retry

        return None, response, MAX_RETRIES


def load_json_or_jsonl(path):
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(item, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_unique_labels(record):
    """
    Uses every unique recency label assigned to a question.
    For non-stationary questions, this means all labels in label_distribution.
    For stationary questions, this usually means one label.
    """
    label_distribution = record.get("label_distribution", {})

    if label_distribution:
        return list(label_distribution.keys())

    if record.get("majority_label"):
        return [record["majority_label"]]

    return []


def load_processed_pairs(output_path):
    processed = set()

    if not os.path.exists(output_path):
        return processed

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                processed.add((item["q_id"], item["label"]))

    return processed


def process_dataset(input_file, output_dir, output_file, model_name, batch_size):
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_file)
    failed_path = os.path.join(
        output_dir,
        output_file.replace(".json", "_failed.jsonl")
    )

    data = load_json_or_jsonl(input_file)

    generator = ContextGenerator(model_name, batch_size=batch_size)
    generator.load_model()

    final_results = []

    for record in tqdm(data, desc="Generating contexts"):
        q_id = record["q_id"]
        question = record["question"]

        unique_labels = record.get("unique_labels")
        if unique_labels is None:
            unique_labels = list(record.get("label_distribution", {}).keys())

        label_contexts = {}

        for label in unique_labels:
            try:
                contexts, raw_response, retry_count = generator.generate_contexts_for_pair(
                    question,
                    label
                )

                if contexts is None:
                    failed = {
                        "q_id": q_id,
                        "question": question,
                        "label": label,
                        "failure_reason": "failed_to_parse_contexts",
                        "raw_response": raw_response,
                        "failed_at": datetime.now().isoformat()
                    }
                    append_jsonl(failed, failed_path)
                    label_contexts[label] = []
                else:
                    label_contexts[label] = contexts

            except Exception:
                failed = {
                    "q_id": q_id,
                    "question": question,
                    "label": label,
                    "failure_reason": "processing_error",
                    "error_details": traceback.format_exc(),
                    "failed_at": datetime.now().isoformat()
                }
                append_jsonl(failed, failed_path)
                label_contexts[label] = []

        output_item = {
            "q_id": q_id,
            "question": question,
            "source": record.get("source"),
            "hop_type": record.get("hop_type"),
            "all_labels": record.get("all_labels"),
            "unique_labels": unique_labels,
            "label_distribution": record.get("label_distribution"),
            "majority_label": record.get("majority_label"),
            "confidence": record.get("confidence"),
            "entropy": record.get("entropy"),
            "label_contexts": label_contexts
        }

        final_results.append(output_item)

        # save incrementally after every question
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(colored("\nContext generation complete.", "green", attrs=["bold"]))
    print(colored(f"Output: {output_path}", "cyan"))
    print(colored(f"Failed: {failed_path}", "yellow"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate temporal contexts for each question-label pair"
    )

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/context_generation")
    parser.add_argument("--output", type=str, default="generated_contexts.jsonl")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    args = parser.parse_args()

    process_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        output_file=args.output,
        model_name=args.model,
        batch_size=args.batch_size
    )
