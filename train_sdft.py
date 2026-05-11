import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from string import Template
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_ROOT = os.environ.get("SDFT_DATA_ROOT", "data")


def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1,
        help="Number of training epochs (supports fractional)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum optimizer steps. If the requested epochs finish earlier, training stops at the epoch boundary.",
    )
    parser.add_argument(
        "--num_prompts_per_batch",
        type=int,
        default=32,
        help="Number of prompts per batch",
    )
    parser.add_argument(
        "--ref_model_mixup_alpha",
        type=float,
        default=0.01,
        help="Reference model mixup alpha",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL coefficient for distillation. If 0.0, reference model is not loaded separately.",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--train_domains",
        type=str,
        nargs="+",
        default=["tooluse_data"],
        help="Domain directory names under data/ (e.g., tooluse_data medical_data science_data)",
    )
    parser.add_argument(
        "--train_weights",
        type=float,
        nargs="+",
        default=None,
        help="Mixing weights for each domain (e.g., 0.5 0.3 0.2)",
    )
    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA training")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="LoRA target modules",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--save_steps", type=int, default=100, help="Save checkpoint every N steps"
    )
    # SFT mode
    parser.add_argument(
        "--sft", action="store_true", help="Use standard SFT instead of SDFT"
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    # Interleaved accuracy evaluation
    parser.add_argument(
        "--eval_domains",
        type=str,
        nargs="+",
        default=["medical_data", "science_data", "tooluse_data"],
        help="Domains to evaluate accuracy on at each checkpoint save (set empty to disable)",
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=500, help="Max eval samples per domain"
    )
    parser.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=2048,
        help="Max tokens to generate for eval",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4, help="Batch size for eval generation"
    )
    parser.add_argument(
        "--no_eval", action="store_true", help="Disable interleaved accuracy evaluation"
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Max training samples per domain (shuffle + truncate if exceeded)"
    )
    return parser.parse_args()


def format_example(example):
    teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "teacher_prompt": [
            {
                "role": "user",
                "content": teacher_prompt.substitute(
                    orig_content=example["prompt"],
                    output_text="\n".join(example["golden_response"]),
                ),
            }
        ],
    }


def format_sft_example(example):
    """Format example as chat messages for standard SFT (no teacher prompt)."""
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": "\n".join(example["golden_response"])},
        ],
    }


def tokenize_sft_dataset(dataset, tokenizer, max_seq_length):
    """Tokenize a chat-formatted dataset with prompt masking for SFT."""

    def _tokenize(example):
        # Apply chat template to full conversation
        full_text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        # Apply chat template to prompt only (to compute prompt length)
        prompt_text = tokenizer.apply_chat_template(
            example["messages"][:1], tokenize=False, add_generation_prompt=True
        )

        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        prompt_enc = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        prompt_len = len(prompt_enc["input_ids"])

        # Mask prompt tokens with -100 so loss is only on completion
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = dataset.map(_tokenize, remove_columns=dataset.column_names)
    return dataset


@dataclass
class SFTDataCollator:
    """Pads input_ids, attention_mask, labels to max length in batch."""

    tokenizer: Any
    max_seq_length: int = 2048

    def __call__(self, features):
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_seq_length,
        )

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(
                f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)

        return {k: torch.tensor(v) for k, v in batch.items()}


def load_domain_dataset(domain_name, seed=42, format_fn=format_example, max_samples=None) -> Dataset:
    """Load and prepare a single domain's dataset with formatted prompts."""
    train_path = os.path.join(DATA_ROOT, domain_name, "train_data.json")
    train_dataset = Dataset.from_json(train_path)
    train_dataset = train_dataset.map(
        format_fn, remove_columns=train_dataset.column_names
    )
    if max_samples is not None and len(train_dataset) > max_samples:
        train_dataset = train_dataset.shuffle(seed=seed).select(range(max_samples))
    return train_dataset


def load_mixed_dataset(domains, weights=None, seed=42, format_fn=format_example, max_samples=None):
    """Load and mix multiple domain datasets according to weights.

    Uses the final domain as the current-stage anchor. The anchor is included
    exactly once; earlier domains are sampled or repeated to match the target
    proportions.
    """
    if weights is None:
        weights = [1.0 / len(domains)] * len(domains)
    else:
        assert len(weights) == len(domains), (
            f"Number of weights ({len(weights)}) must match number of domains ({len(domains)})"
        )
        total = sum(weights)
        weights = [w / total for w in weights]

    domain_datasets = []
    for domain in domains:
        ds = load_domain_dataset(domain, seed=seed, format_fn=format_fn, max_samples=max_samples)
        ds = ds.shuffle(seed=seed)
        domain_datasets.append(ds)

    # Continual callers pass previous domains first and the new/current
    # domain last. Train one epoch over the new domain, then replay prior
    # domains according to their target proportions.
    anchor_idx = len(domain_datasets) - 1
    anchor_size = len(domain_datasets[anchor_idx])
    anchor_weight = weights[anchor_idx]
    if anchor_weight <= 0:
        raise ValueError(
            f"Anchor domain {domains[anchor_idx]!r} must have positive weight"
        )

    mixed_parts = []
    print("\n=== Dataset Composition ===")
    for i, (domain, ds, w) in enumerate(zip(domains, domain_datasets, weights)):
        if i == anchor_idx:
            target = anchor_size
        else:
            # e.g. if anchor weight=0.4 and this weight=0.6,
            # target = anchor_size * 0.6/0.4.
            target = int(anchor_size * w / anchor_weight)

        full_copies = target // len(ds)
        remainder = target % len(ds)

        parts = [ds] * full_copies
        if remainder > 0:
            parts.append(ds.select(range(remainder)))

        if not parts:
            combined = ds.select(range(0))
        elif len(parts) == 1:
            combined = parts[0]
        else:
            combined = concatenate_datasets(parts)
        mixed_parts.append(combined)
        print(
            f"  {domain}: {len(ds)} original -> {len(combined)} samples (weight={w:.3f}, {full_copies} full copies + {remainder} extra)"
        )

    final_dataset = concatenate_datasets(mixed_parts)
    final_dataset = final_dataset.shuffle(seed=seed)
    print(f"  Total: {len(final_dataset)} samples")
    print("===========================\n")
    return final_dataset, None


class AccuracyEvalCallback(TrainerCallback):
    """Evaluates accuracy at each checkpoint save and at end of training."""

    def __init__(
        self,
        tokenizer,
        eval_domains,
        max_eval_samples=500,
        max_new_tokens=256,
        eval_batch_size=4,
    ):
        from eval_sdft import DOMAIN_SCORERS, generate_responses, load_eval_raw

        self.tokenizer = tokenizer
        self.eval_domains = eval_domains
        self.max_new_tokens = max_new_tokens
        self.eval_batch_size = eval_batch_size
        self._generate_responses = generate_responses
        self._domain_scorers = DOMAIN_SCORERS
        self.last_eval_step = None
        self.last_eval_acc = None
        self.best_eval_acc = None
        self.best_eval_step = None
        self.best_checkpoint_dir = None

        # Pre-load eval data
        self.eval_raw = {}
        for domain in eval_domains:
            self.eval_raw[domain] = load_eval_raw(
                domain, max_samples=max_eval_samples, full_eval=False
            )
            logger.info(
                f"Loaded {len(self.eval_raw[domain])} eval examples for {domain}"
            )

    def _evaluate(self, model, state, save_dir=None):
        """Run accuracy evaluation and return (results_dict, mean_accuracy)."""
        logger.info(f"Running accuracy evaluation at step {state.global_step}...")
        was_training = model.training
        model.eval()

        results = {}
        try:
            for domain in self.eval_domains:
                raw_examples = self.eval_raw[domain]
                responses = self._generate_responses(
                    model,
                    self.tokenizer,
                    raw_examples,
                    max_new_tokens=self.max_new_tokens,
                    batch_size=self.eval_batch_size,
                )

                scorer = self._domain_scorers.get(domain)
                if scorer is None:
                    logger.warning(f"No scorer for {domain}, skipping.")
                    continue

                accuracy = scorer(responses, raw_examples)
                results[domain] = {
                    "accuracy": accuracy,
                    "n_examples": len(raw_examples),
                }
                logger.info(
                    f"  {domain}: accuracy={accuracy:.4f} ({len(raw_examples)} examples)"
                )

                # Save per-domain responses
                if save_dir is not None:
                    resp_file = os.path.join(save_dir, f"responses_{domain}.json")
                    with open(resp_file, "w") as f:
                        json.dump(
                            [
                                {
                                    "prompt": ex["prompt"],
                                    "golden_answer": ex["golden_answer"],
                                    "predicted": resp,
                                }
                                for ex, resp in zip(raw_examples, responses)
                            ],
                            f,
                            indent=2,
                        )

            # Compute mean accuracy across domains
            accs = [res["accuracy"] for res in results.values()]
            mean_acc = sum(accs) / len(accs) if accs else 0.0

            # Save results json with step number
            if save_dir is not None:
                results_with_step = {
                    "step": state.global_step,
                    "mean_accuracy": mean_acc,
                    **results,
                }
                results_file = os.path.join(
                    save_dir, f"accuracy_results_step{state.global_step}.json"
                )
                with open(results_file, "w") as f:
                    json.dump(results_with_step, f, indent=2)
                logger.info(f"Accuracy results saved to {results_file}")

            # Log to wandb if available
            try:
                import wandb

                if wandb.run is not None:
                    log_dict = {
                        f"eval_accuracy/{domain}": res["accuracy"]
                        for domain, res in results.items()
                    }
                    wandb.log(log_dict, step=state.global_step)
            except ImportError:
                pass

            # Append to log_history so it persists in trainer_state.json
            state.log_history.append(
                {
                    "step": state.global_step,
                    "mean_accuracy": mean_acc,
                    **{
                        f"eval_accuracy/{domain}": res["accuracy"]
                        for domain, res in results.items()
                    },
                }
            )

            return results, mean_acc

        finally:
            if was_training:
                model.train()

    def on_save(self, args, state, control, **kwargs):
        """Evaluate at each checkpoint save."""
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if not os.path.isdir(checkpoint_dir):
            logger.warning(f"Checkpoint dir {checkpoint_dir} not found, skipping eval.")
            return

        model = kwargs.get("model")
        if model is None:
            logger.warning("Model not available in callback kwargs, skipping eval.")
            return

        _, mean_acc = self._evaluate(model, state, save_dir=checkpoint_dir)
        self.last_eval_step = state.global_step
        self.last_eval_acc = mean_acc

        if self.best_eval_acc is None or mean_acc > self.best_eval_acc:
            # Delete previous best checkpoint since this one is better
            if self.best_checkpoint_dir is not None and os.path.isdir(self.best_checkpoint_dir):
                import shutil
                shutil.rmtree(self.best_checkpoint_dir)
                logger.info(f"Removed previous best checkpoint: {self.best_checkpoint_dir}")
            self.best_eval_acc = mean_acc
            self.best_eval_step = state.global_step
            self.best_checkpoint_dir = checkpoint_dir
            logger.info(
                f"Step {state.global_step} mean accuracy: {mean_acc:.4f} ** new best **"
            )
        else:
            # Not the best — delete this checkpoint immediately
            import shutil
            shutil.rmtree(checkpoint_dir)
            logger.info(
                f"Step {state.global_step} mean accuracy: {mean_acc:.4f} (best: {self.best_eval_acc:.4f} @ step {self.best_eval_step}) — checkpoint removed"
            )

    def _load_checkpoint_weights(self, model, checkpoint_dir):
        """Load model weights from a checkpoint directory into the current model."""
        import glob

        from safetensors.torch import load_file

        safetensor_files = sorted(
            f
            for f in glob.glob(os.path.join(checkpoint_dir, "model*.safetensors"))
            if not f.endswith(".index.json")
        )
        if not safetensor_files:
            logger.warning(f"No safetensor files found in {checkpoint_dir}")
            return False

        state_dict = {}
        for sf in safetensor_files:
            state_dict.update(load_file(sf, device="cpu"))
        model.load_state_dict(state_dict)
        del state_dict
        logger.info(f"Loaded weights from {checkpoint_dir}")
        return True

    def on_train_end(self, args, state, control, **kwargs):
        """Final eval + save best model to base output_dir."""
        model = kwargs.get("model")
        if model is None:
            logger.warning("Model not available in on_train_end, skipping final eval.")
            return

        output_dir = args.output_dir

        # Eval the final model (skip if already evaluated at this step via on_save)
        if self.last_eval_step == state.global_step:
            logger.info(
                f"Skipping redundant eval at step {state.global_step} (already evaluated in on_save)"
            )
            mean_acc = self.last_eval_acc
        else:
            _, mean_acc = self._evaluate(model, state, save_dir=output_dir)
            logger.info(f"Final model accuracy: {mean_acc:.4f}")

        # Update best tracking with final model result
        if mean_acc is not None and (
            self.best_eval_acc is None or mean_acc > self.best_eval_acc
        ):
            self.best_eval_acc = mean_acc
            self.best_eval_step = state.global_step
            self.best_checkpoint_dir = None  # final model is already in memory

        # Load best checkpoint if it's from an earlier step
        if (
            self.best_checkpoint_dir is not None
            and self.best_eval_step != state.global_step
        ):
            logger.info(
                f"Loading best model from step {self.best_eval_step} "
                f"(accuracy={self.best_eval_acc:.4f}) instead of final step {state.global_step}"
            )
            self._load_checkpoint_weights(model, self.best_checkpoint_dir)
        else:
            logger.info(
                f"Saving final model (step {state.global_step}, accuracy={mean_acc:.4f})"
            )

        # Save model
        tokenizer = kwargs.get("processing_class") or self.tokenizer
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Record which step was selected
        best_info = {
            "best_step": self.best_eval_step,
            "best_mean_accuracy": self.best_eval_acc,
            "final_step": state.global_step,
            "final_mean_accuracy": mean_acc,
        }
        with open(os.path.join(output_dir, "best_model_info.json"), "w") as f:
            json.dump(best_info, f, indent=2)

        # Clean up non-best checkpoint directories to save disk
        import shutil

        for entry in os.listdir(output_dir):
            ckpt_path = os.path.join(output_dir, entry)
            if entry.startswith("checkpoint-") and os.path.isdir(ckpt_path):
                shutil.rmtree(ckpt_path)
                logger.info(f"Removed checkpoint dir: {ckpt_path}")


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set up eval callback
    eval_callbacks = []
    if not args.no_eval and not args.use_lora and args.eval_domains:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        eval_callbacks.append(
            AccuracyEvalCallback(
                tokenizer=tokenizer,
                eval_domains=args.eval_domains,
                max_eval_samples=args.max_eval_samples,
                max_new_tokens=args.eval_max_new_tokens,
                eval_batch_size=args.eval_batch_size,
            )
        )

    if args.sft:
        # ── SFT path: standard cross-entropy on ground-truth completions ──
        from transformers import Trainer, TrainingArguments

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

        dataset, _ = load_mixed_dataset(
            args.train_domains,
            args.train_weights,
            args.seed,
            format_fn=format_sft_example,
            max_samples=args.max_train_samples,
        )
        dataset = tokenize_sft_dataset(dataset, tokenizer, args.max_seq_length)

        if args.use_lora:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            model.enable_input_require_grads()
            model.print_trainable_parameters()

        total_steps_per_epoch = math.ceil(
            len(dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        )
        total_steps = math.ceil(total_steps_per_epoch * args.num_train_epochs)
        effective_max_steps = (
            args.max_steps
            if args.max_steps is not None and total_steps > args.max_steps
            else -1
        )
        logger.info(
            f"SFT: {len(dataset)} samples -> {total_steps_per_epoch} steps/epoch"
            f" x {args.num_train_epochs} epochs = {total_steps} steps"
        )
        if effective_max_steps > 0:
            logger.info(f"SFT: capping training at {effective_max_steps} steps")

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            max_steps=effective_max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=args.save_steps,
            max_grad_norm=1.0,
            report_to="wandb",
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=SFTDataCollator(
                tokenizer=tokenizer, max_seq_length=args.max_seq_length
            ),
            callbacks=eval_callbacks,
        )
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        if not eval_callbacks:
            trainer.save_model(args.output_dir)
            trainer.state.save_to_json(
                os.path.join(args.output_dir, "trainer_state.json")
            )

    else:
        # ── SDFT path: distillation with teacher model ──
        from distil_config import DistilConfig
        from distil_trainer import DistilTrainer

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )
        dataset, _ = load_mixed_dataset(
            args.train_domains, args.train_weights, args.seed,
            max_samples=args.max_train_samples,
        )

        peft_config = None
        if args.use_lora:
            from peft import LoraConfig

            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                task_type="CAUSAL_LM",
            )

        num_generations = 8  # DistilConfig default
        total_steps_per_epoch = math.ceil(
            len(dataset) * num_generations / args.num_prompts_per_batch
        )
        total_steps = math.ceil(total_steps_per_epoch * args.num_train_epochs)
        effective_max_steps = (
            args.max_steps
            if args.max_steps is not None and total_steps > args.max_steps
            else -1
        )
        logger.info(
            f"SDFT: {len(dataset)} samples -> {total_steps_per_epoch} steps/epoch"
            f" x {args.num_train_epochs} epochs = {total_steps} steps"
        )
        if effective_max_steps > 0:
            logger.info(f"SDFT: capping training at {effective_max_steps} steps")

        config = DistilConfig(
            seed=args.seed,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.3,
            vllm_enable_sleep_mode=True,
            learning_rate=args.learning_rate,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=1,
            bf16=True,
            fp16=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.num_prompts_per_batch,
            max_prompt_length=1024,
            max_completion_length=2048,
            num_train_epochs=args.num_train_epochs,
            max_steps=effective_max_steps,
            save_steps=args.save_steps,
            max_grad_norm=1,
            report_to="wandb",
            output_dir=args.output_dir,
            log_completions=False,  # True for debugging
            sync_ref_model=True,
            ref_model_sync_steps=1,
            ref_model_mixup_alpha=args.ref_model_mixup_alpha,
            beta=args.beta,
            vllm_importance_sampling_correction=True,
            num_loss_tokens_to_skip=3,
        )
        trainer = DistilTrainer(
            model=model,
            ref_model=teacher_model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=eval_callbacks,
        )
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        if not eval_callbacks:
            trainer.save_model(args.output_dir)
            trainer.state.save_to_json(
                os.path.join(args.output_dir, "trainer_state.json")
            )
