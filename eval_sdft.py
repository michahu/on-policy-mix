#!/usr/bin/env python3
"""LMC evaluation for Qwen/SDFT models with JSON domain data.

Interpolates between two models at multiple alphas, evaluates each
interpolant on each domain's eval set.

Usage:
    python eval_sdft.py \
        --model_a Qwen/Qwen2.5-7B-Instruct \
        --model_b runs/sdft/checkpoint-100 \
        --eval_domains medical_data science_data tooluse_data \
        --alphas 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --output_dir runs/lmc_sdft/test

    # With LoRA adapter on model_a:
    python eval_sdft.py \
        --model_a Qwen/Qwen2.5-7B-Instruct \
        --model_b Qwen/Qwen2.5-7B-Instruct \
        --lora_dir_a runs/sdft_lora/checkpoint-100 \
        --eval_domains medical_data science_data tooluse_data \
        --output_dir runs/lmc_sdft/test
"""

import argparse
import json
import logging
import os
import re

import torch
from datasets import Dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.merge import linear_interpolation_merge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_ROOT = os.environ.get("SDFT_DATA_ROOT", "data")


def _eval_data_path(domain, full_eval=False):
    """Return the eval data path for a domain, using the small subset when available."""
    if not full_eval:
        small = os.path.join(DATA_ROOT, domain, "eval_data_small.json")
        if os.path.exists(small):
            return small
    return os.path.join(DATA_ROOT, domain, "eval_data.json")


def load_eval_domain(domain, tokenizer, max_samples=500, max_length=1024, full_eval=False):
    """Load eval data for a domain, tokenize as chat-formatted sequences.

    Each example is formatted as: user prompt + assistant golden_answer,
    then tokenized to compute cross-entropy loss.
    """
    eval_path = _eval_data_path(domain, full_eval=full_eval)
    with open(eval_path) as f:
        raw = json.load(f)

    if max_samples and len(raw) > max_samples:
        raw = raw[:max_samples]

    input_ids_list = []
    labels_list = []

    for ex in raw:
        answer = ex["golden_answer"]
        if not isinstance(answer, str):
            answer = json.dumps(answer)
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": answer},
        ]
        # Tokenize the full conversation
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")

        # Build labels: mask everything except the assistant response
        prompt_messages = [{"role": "user", "content": ex["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"])

        ids = encoded["input_ids"].squeeze(0)
        labels = ids.clone()
        labels[:prompt_len] = -100  # mask prompt tokens

        input_ids_list.append(ids)
        labels_list.append(labels)

    return input_ids_list, labels_list


def eval_loss(model, tokenizer, input_ids_list, labels_list, batch_size=4):
    """Compute average cross-entropy loss over tokenized examples."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(input_ids_list), batch_size):
        batch_ids = input_ids_list[i:i + batch_size]
        batch_labels = labels_list[i:i + batch_size]

        # Pad to same length within batch
        max_len = max(ids.size(0) for ids in batch_ids)
        padded_ids = torch.full((len(batch_ids), max_len), tokenizer.pad_token_id or 0, dtype=torch.long)
        padded_labels = torch.full((len(batch_ids), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.long)

        for j, (ids, labs) in enumerate(zip(batch_ids, batch_labels)):
            padded_ids[j, :ids.size(0)] = ids
            padded_labels[j, :labs.size(0)] = labs
            attention_mask[j, :ids.size(0)] = 1

        padded_ids = padded_ids.to(device)
        padded_labels = padded_labels.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=padded_ids, attention_mask=attention_mask, labels=padded_labels)

        # Count non-masked tokens for proper averaging
        n_tokens = (padded_labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def load_eval_raw(domain, max_samples=500, full_eval=False):
    """Load raw eval examples (prompt + golden_answer) for a domain."""
    eval_path = _eval_data_path(domain, full_eval=full_eval)
    with open(eval_path) as f:
        raw = json.load(f)
    if max_samples and len(raw) > max_samples:
        raw = raw[:max_samples]
    return raw


def generate_responses(model, tokenizer, raw_examples, max_new_tokens=256, batch_size=4):
    """Generate model responses for a list of raw examples."""
    model.eval()
    device = next(model.parameters()).device
    responses = []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in range(0, len(raw_examples), batch_size):
        batch = raw_examples[i:i + batch_size]
        prompts = []
        for ex in batch:
            messages = [{"role": "user", "content": ex["prompt"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)

        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, ids in enumerate(output_ids):
            input_len = encoded["input_ids"][j].shape[0]
            generated = tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            responses.append(generated.strip())

    tokenizer.padding_side = original_padding_side
    return responses


def extract_mcq_answer(text):
    """Extract A/B/C/D answer letter from model response."""
    # Look for patterns like "final answer" or standalone letter
    patterns = [
        r"(?:final answer|answer is|answer:)\s*[:\s]*([A-D])\b",
        r"\b([A-D])\.\s",  # "A. " at word boundary
        r"^([A-D])$",  # just the letter
        r"\b([A-D])\b",  # any standalone letter (last resort)
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def score_mcq(responses, raw_examples):
    """Score multiple-choice responses by exact letter match."""
    correct = 0
    for resp, ex in zip(responses, raw_examples):
        predicted = extract_mcq_answer(resp)
        golden = ex["golden_answer"].strip().upper()
        if predicted == golden:
            correct += 1
    return correct / len(raw_examples) if raw_examples else 0.0


def extract_tool_call(text):
    """Extract Action and Action_Input from a tool-use response."""
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
    input_match = re.search(r"Action\s*Input:\s*(.+?)(?:\n|$)", text)
    if action_match:
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else "{}"
        return action, action_input
    return None, None


def fuzzy_match_action_input(predicted_input_str, golden_input_str):
    """Fuzzy match predicted Action_Input against golden Action_Input.

    Parses both as JSON dicts and checks that all non-empty golden values
    appear in the predicted input. Returns True if they match.
    """
    try:
        golden = json.loads(golden_input_str)
    except (json.JSONDecodeError, TypeError):
        golden = {}
    try:
        predicted = json.loads(predicted_input_str)
    except (json.JSONDecodeError, TypeError):
        predicted = {}

    if not golden:
        return True  # no constraints to check

    # Check each non-empty golden value appears in predicted
    for key, golden_val in golden.items():
        # Skip empty/default golden values — these are optional
        if golden_val == "" or golden_val is None:
            continue
        predicted_val = predicted.get(key)
        # Coerce both to strings for comparison (handles int vs str mismatches)
        if str(predicted_val).lower().strip() != str(golden_val).lower().strip():
            return False
    return True


def score_tooluse(responses, raw_examples):
    """Score tool-use responses by action name match AND fuzzy input match."""
    correct = 0
    for resp, ex in zip(responses, raw_examples):
        golden_calls = ex["golden_answer"]
        if not isinstance(golden_calls, list):
            golden_calls = [golden_calls]
        golden_action = golden_calls[0]["Action"]
        golden_input = golden_calls[0].get("Action_Input", "{}")

        predicted_action, predicted_input = extract_tool_call(resp)
        if predicted_action == golden_action and fuzzy_match_action_input(predicted_input, golden_input):
            correct += 1
    return correct / len(raw_examples) if raw_examples else 0.0


def _read_openai_key_from_env_file(env_path):
    if not env_path or not os.path.exists(env_path):
        return None
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() in {"OPENAI_API_KEY", "OPENAI_KEY"}:
                value = value.strip().strip("'\"")
                if value:
                    return value
    return None


def _openai_env_file_candidates():
    """Return likely .env files without assuming the eval repo owns the data root."""
    candidates = []
    explicit = os.environ.get("SDFT_OPENAI_ENV_FILE") or os.environ.get("OPENAI_ENV_FILE")
    if explicit:
        candidates.append(explicit)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, ".env"))

    data_root = os.environ.get("SDFT_DATA_ROOT")
    if data_root:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(data_root)), ".env"))

    seen = set()
    unique = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def load_openai_key():
    """Load OpenAI API key from the environment or known .env locations."""
    for name in ("OPENAI_API_KEY", "OPENAI_KEY"):
        value = os.environ.get(name)
        if value:
            return value

    for env_path in _openai_env_file_candidates():
        value = _read_openai_key_from_env_file(env_path)
        if value:
            return value

    searched = ", ".join(_openai_env_file_candidates())
    raise ValueError(
        "OpenAI key not found; set OPENAI_API_KEY/OPENAI_KEY, "
        "set SDFT_OPENAI_ENV_FILE/OPENAI_ENV_FILE, or add one to .env. "
        f"Searched: {searched}"
    )


def score_medical_fuzzy(responses, raw_examples):
    """Score medical free-response answers using GPT-4 fuzzy matching."""
    from openai import OpenAI

    client = OpenAI(api_key=load_openai_key())
    correct = 0

    for resp, ex in zip(responses, raw_examples):
        golden = ex["golden_answer"]
        prompt = (
            "You are an evaluator. Determine if the predicted answer is semantically "
            "equivalent to the reference answer for a medical question. "
            "Minor wording differences are acceptable. "
            "Respond with ONLY 'correct' or 'incorrect'.\n\n"
            f"Question: {ex['prompt']}\n"
            f"Reference answer: {golden}\n"
            f"Predicted answer: {resp}\n\n"
            "Verdict:"
        )
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            verdict = completion.choices[0].message.content.strip().lower()
            if "correct" in verdict and "incorrect" not in verdict:
                correct += 1
        except Exception as e:
            logger.warning(f"GPT-4 scoring failed for an example: {e}")

    return correct / len(raw_examples) if raw_examples else 0.0


DOMAIN_SCORERS = {
    "science_data": score_mcq,
    "medical_data": score_medical_fuzzy,
    "tooluse_data": score_tooluse,
}


def eval_accuracy(model, tokenizer, domain, raw_examples, max_new_tokens=256, batch_size=4):
    """Generate responses and compute accuracy for a domain."""
    responses = generate_responses(model, tokenizer, raw_examples,
                                   max_new_tokens=max_new_tokens, batch_size=batch_size)
    scorer = DOMAIN_SCORERS.get(domain)
    if scorer is None:
        logger.warning(f"No accuracy scorer for domain '{domain}', skipping accuracy.")
        return None, responses
    accuracy = scorer(responses, raw_examples)
    return accuracy, responses


def load_model(model_path, lora_dir=None, merge_lora=False):
    """Load a causal LM, optionally with a LoRA adapter merged in."""
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    if lora_dir:
        model = PeftModel.from_pretrained(model, lora_dir)
        if merge_lora:
            model = model.merge_and_unload()
    return model


@torch.inference_mode()
def merge_and_eval(args):
    """Main LMC evaluation: interpolate models at each alpha, eval on each domain."""
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_a)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval data for all domains
    logger.info("Loading evaluation datasets...")
    eval_data = {}
    eval_raw = {}
    for domain in args.eval_domains:
        logger.info(f"  Loading {domain}...")
        ids, labels = load_eval_domain(domain, tokenizer, max_samples=args.max_eval_samples,
                                        full_eval=args.full_eval)
        eval_data[domain] = (ids, labels)
        if args.eval_accuracy:
            eval_raw[domain] = load_eval_raw(domain, max_samples=args.max_eval_samples,
                                             full_eval=args.full_eval)
        logger.info(f"  {domain}: {len(ids)} examples")

    # Load models
    logger.info(f"Loading model A: {args.model_a}")
    model_a = load_model(args.model_a, lora_dir=args.lora_dir_a, merge_lora=True)

    logger.info(f"Loading model B: {args.model_b}")
    model_b = load_model(args.model_b, lora_dir=args.lora_dir_b, merge_lora=True)

    all_results = {}

    for alpha in args.alphas:
        logger.info(f"Evaluating alpha={alpha:.3f} ...")

        if alpha == 1.0:
            merged = model_a
        elif alpha == 0.0:
            merged = model_b
        else:
            merged = linear_interpolation_merge(model_a, model_b, alpha)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        merged.to(device)

        per_dataset_results = {}
        for domain in args.eval_domains:
            ids, labels = eval_data[domain]
            loss = eval_loss(merged, tokenizer, ids, labels, batch_size=args.batch_size)
            per_dataset_results[domain] = {"eval_loss": loss}
            logger.info(f"  {domain}: loss={loss:.4f}")

            if args.eval_accuracy and domain in eval_raw:
                acc, responses = eval_accuracy(
                    merged, tokenizer, domain, eval_raw[domain],
                    max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                )
                if acc is not None:
                    per_dataset_results[domain]["accuracy"] = acc
                    logger.info(f"  {domain}: accuracy={acc:.4f}")
                # Save generated responses
                resp_file = os.path.join(
                    args.output_dir, f"responses_alpha{alpha:.3f}_{domain}.json"
                )
                with open(resp_file, "w") as f:
                    json.dump(
                        [{"prompt": ex["prompt"], "golden_answer": ex["golden_answer"],
                          "predicted": resp}
                         for ex, resp in zip(eval_raw[domain], responses)],
                        f, indent=2,
                    )

        all_results[f"alpha_{alpha:.3f}"] = {
            "alpha": alpha,
            "per_dataset_results": per_dataset_results,
        }

        if alpha != 0.0 and alpha != 1.0:
            del merged
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save results
    results_file = os.path.join(args.output_dir, "linear_connectivity_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Print summary - losses
    logger.info("=" * 70)
    logger.info("LOSSES:")
    header = f"{'Alpha':<10} " + " ".join(f"{d:<20}" for d in args.eval_domains)
    logger.info(header)
    for ak in sorted(all_results.keys()):
        a = all_results[ak]["alpha"]
        losses = [f"{all_results[ak]['per_dataset_results'].get(d, {}).get('eval_loss', float('nan')):<20.4f}"
                  for d in args.eval_domains]
        logger.info(f"{a:<10.3f} " + " ".join(losses))

    # Print summary - accuracy (if computed)
    if args.eval_accuracy:
        logger.info("")
        logger.info("ACCURACY:")
        logger.info(header)
        for ak in sorted(all_results.keys()):
            a = all_results[ak]["alpha"]
            accs = []
            for d in args.eval_domains:
                acc = all_results[ak]["per_dataset_results"].get(d, {}).get("accuracy")
                accs.append(f"{acc:<20.4f}" if acc is not None else f"{'N/A':<20}")
            logger.info(f"{a:<10.3f} " + " ".join(accs))
    logger.info("=" * 70)

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="LMC evaluation for SDFT models")
    parser.add_argument("--model_a", type=str, required=True, help="Path/name of model A")
    parser.add_argument("--model_b", type=str, required=True, help="Path/name of model B")
    parser.add_argument("--lora_dir_a", type=str, default=None, help="LoRA adapter dir for model A")
    parser.add_argument("--lora_dir_b", type=str, default=None, help="LoRA adapter dir for model B")
    parser.add_argument("--eval_domains", type=str, nargs="+", required=True,
                        help="Domain names under data/ to evaluate on")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="Interpolation alpha values")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--max_eval_samples", type=int, default=500, help="Max eval samples per domain")
    parser.add_argument("--eval_accuracy", action="store_true",
                        help="Also compute accuracy via generation (slower)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per example for accuracy eval")
    parser.add_argument("--full_eval", action="store_true",
                        help="Use full eval_data.json instead of eval_data_small.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_and_eval(args)
