#!/usr/bin/env python3
"""
Simple finetuning script for language models.
"""

import logging
import os

import torch
from hf_olmo import OLMoConfig, OLMoForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.argparser import get_argument_parser
from src.data import (
    DEFAULT_DATA_ROOT,
    CustomDataCollator,
    get_dataset,
    get_mixed_dataset,
)
from src.utils import LimitedDataset, extract_dataset_name


def _enable_loss_only_training_forward(model):
    """Avoid returning giant logits from training steps.

    Accelerate converts model outputs to fp32 after forward. For causal LM
    training, returning logits can require many extra GB even though Trainer
    only needs the loss.
    """
    original_forward = model.forward

    def loss_only_training_forward(*args, **kwargs):
        kwargs.pop("num_items_in_batch", None)

        if not model.training or kwargs.get("labels") is None:
            return original_forward(*args, **kwargs)

        kwargs["return_dict"] = True
        outputs = original_forward(*args, **kwargs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        del outputs
        return {"loss": loss}

    model.forward = loss_only_training_forward


def _main(args):
    """Load model and run finetuning with periodic evaluation."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Resolve data_root: CLI arg > env var > default
    data_root = args.data_root or os.environ.get("DATA_ROOT") or DEFAULT_DATA_ROOT
    logger.info(f"Using data root: {data_root}")

    # Set default dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    torch.set_default_dtype(dtype)
    logger.info(f"Set default dtype to {args.dtype}")

    # Set seed for reproducibility
    if args.seed is not None:
        logger.info(f"Setting seed to: {args.seed}")
        set_seed(args.seed)

    # Load model and tokenizer
    logger.info(f"Loading model from: {args.model_dir}")
    if args.model_init_mode == "config":
        logger.info("Initializing model from config only (random weights)")
        config = OLMoConfig.from_pretrained(args.model_dir)
        model = OLMoForCausalLM(config, init_params=True)
    else:
        model = OLMoForCausalLM.from_pretrained(args.model_dir)

    if args.lora_checkpoint_dir:
        # Load pre-initialized LoRA checkpoint (e.g., created using svd_init.py with PiSSA, MiLoRA, etc.)
        logger.info(
            f"Loading pre-initialized LoRA checkpoint from: {args.lora_checkpoint_dir}"
        )
        model = PeftModel.from_pretrained(
            model, args.lora_checkpoint_dir, subfolder="./lora", is_trainable=True
        )
    elif args.use_lora:
        # Initialize LoRA from scratch with specified hyperparameters
        logger.info("Initializing LoRA from scratch with standard configuration")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules
            if args.lora_target_modules
            else None,
            exclude_modules=args.lora_exclude_modules
            if args.lora_exclude_modules
            else None,
            init_lora_weights=True,
        )
        model = get_peft_model(model, peft_config)

    _enable_loss_only_training_forward(model)

    # Load datasets (always use mixed dataset wrapper for consistency)
    logger.info(f"Loading training dataset from {len(args.train_data_file)} source(s)")

    # For single dataset, use uniform weight [1.0]
    train_weights = args.train_weights if len(args.train_data_file) > 1 else [1.0]

    train_dataset = get_mixed_dataset(
        data_files=args.train_data_file,
        weights=train_weights,
        total_steps=args.max_steps * args.batch_size
        if args.max_steps > 0
        else 100000 * args.batch_size,
        data_work_dir=args.output_dir,
        sequence_length=args.seq_len,
        data_root=data_root,
    )

    eval_dataset = None
    if args.eval_data_file:
        logger.info(
            f"Loading evaluation dataset from {len(args.eval_data_file)} source(s)"
        )
        logger.info(f"Max eval samples: {args.max_eval_samples}")

        if len(args.eval_data_file) > 1:
            # Build individual eval datasets for per-dataset loss tracking
            samples_per_dataset = args.max_eval_samples // len(args.eval_data_file)
            eval_dataset = {}
            for data_file in args.eval_data_file:
                name = extract_dataset_name(data_file)
                ds = get_dataset(
                    data_file=data_file,
                    data_work_dir=args.output_dir,
                    sequence_length=args.seq_len,
                    data_root=data_root,
                )
                eval_dataset[name] = LimitedDataset(ds, samples_per_dataset)
                logger.info(
                    f"  Eval dataset '{name}': {len(eval_dataset[name])} samples"
                )
        else:
            eval_dataset = get_mixed_dataset(
                data_files=args.eval_data_file,
                weights=[1.0],
                total_steps=args.max_eval_samples,
                data_work_dir=args.output_dir,
                sequence_length=args.seq_len,
                data_root=data_root,
            )
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

        logger.info(f"Eval frequency: every {args.eval_steps} training steps")

    # Determine metric for best model selection
    if eval_dataset and isinstance(eval_dataset, dict):
        first_name = list(eval_dataset.keys())[0]
        metric_for_best = f"eval_{first_name}_loss"
    elif eval_dataset:
        metric_for_best = "eval_loss"
    else:
        metric_for_best = None

    # For WSD, we override scheduler creation after trainer init,
    # so pass "constant" as a placeholder to TrainingArguments.
    use_wsd = args.lr_scheduler_type == "wsd"
    hf_scheduler_type = "constant" if use_wsd else args.lr_scheduler_type
    hf_scheduler_kwargs = {} if use_wsd else {"min_lr_rate": args.min_lr_rate}

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs if args.max_steps < 0 else 1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=hf_scheduler_type,
        lr_scheduler_kwargs=hf_scheduler_kwargs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if (args.save_steps > 0 and (not eval_dataset or args.save_steps % args.eval_steps == 0)) else (args.eval_steps if eval_dataset else None),
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        eval_on_start=True if eval_dataset else False,
        save_strategy="steps" if (args.save_steps > 0 or eval_dataset) else "no",
        save_total_limit=1,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model=metric_for_best,
        bf16=args.dtype == "bfloat16" and torch.cuda.is_available(),
        fp16=args.dtype == "float16" and torch.cuda.is_available(),
        dataloader_pin_memory=True,
        dataloader_num_workers=args.num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed if args.seed is not None else 42,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": CustomDataCollator(),
    }

    trainer = Trainer(**trainer_kwargs)

    # Override scheduler with warmup-stable-decay
    if use_wsd:
        from torch.optim.lr_scheduler import LambdaLR

        _warmup = args.warmup_steps
        _decay = args.decay_steps
        _min_lr = args.min_lr_rate

        def _create_wsd_scheduler(num_training_steps, optimizer=None):
            if trainer.lr_scheduler is not None:
                return trainer.lr_scheduler
            opt = trainer.optimizer if optimizer is None else optimizer

            def lr_lambda(step):
                # Warmup
                if step < _warmup:
                    return step / max(1, _warmup)
                # Decay
                decay_start = num_training_steps - _decay
                if step >= decay_start:
                    progress = (step - decay_start) / max(1, _decay)
                    return max(_min_lr, 1.0 - progress * (1.0 - _min_lr))
                # Stable
                return 1.0

            trainer.lr_scheduler = LambdaLR(opt, lr_lambda)
            return trainer.lr_scheduler

        trainer.create_scheduler = _create_wsd_scheduler

    # Run training
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving final model to: {args.output_dir}")
    trainer.save_model(args.output_dir)

    # Copy trainer_state.json from latest checkpoint to output dir
    trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))
    logger.info(f"Saved trainer_state.json to: {args.output_dir}")

    logger.info("Training complete!")


def main():
    """Parse arguments and run training."""
    parser = get_argument_parser(description="Finetune a language model")
    args = parser.parse_args()

    # Validate that train_data_file is provided for training
    if not args.train_data_file:
        raise ValueError("--train_data_file is required for training")

    # Validate weights for training data (only required for multiple datasets)
    if len(args.train_data_file) > 1:
        if args.train_weights is None:
            raise ValueError(
                f"--train_weights must be provided when using multiple training datasets. "
                f"Got {len(args.train_data_file)} datasets but no weights."
            )
        if len(args.train_weights) != len(args.train_data_file):
            raise ValueError(
                f"Number of train_weights ({len(args.train_weights)}) must match "
                f"number of train_data_file arguments ({len(args.train_data_file)})"
            )

    # For single training dataset, train_weights is not required (will use [1.0])
    if len(args.train_data_file) == 1 and args.train_weights is not None:
        logger = logging.getLogger(__name__)
        logger.warning(
            "train_weights provided for single training dataset will be ignored"
        )

    # eval_weights are no longer used (each eval dataset is evaluated individually),
    # but we still accept the argument for backward compatibility with launch scripts
    if args.eval_weights is not None:
        logger = logging.getLogger(__name__)
        logger.warning(
            "eval_weights are ignored: each eval dataset is now evaluated individually"
        )

    _main(args)


if __name__ == "__main__":
    main()
