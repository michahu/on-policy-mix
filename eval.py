#!/usr/bin/env python3
"""
Evaluation script for language models.

Supports two modes:
  1. Standard eval: evaluate a single model on (mixed) datasets
  2. Merge eval (LMC): interpolate between two models at multiple alphas,
     evaluate each interpolant on each dataset separately.
     Activated by passing --model_b.
"""

import json
import logging
import os
from typing import List, Optional

import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
)

from src.argparser import get_argument_parser
from src.data import CustomDataCollator, get_dataset, get_mixed_dataset
from src.merge import linear_interpolation_merge
from src.utils import (
    LimitedDataset,
    clean_gpu_memory,
    extract_dataset_name,
    load_model_with_optional_lora,
)


def eval(
    model,
    eval_dataset,
    output_dir,
    batch_size,
    logger=None,
):
    """
    Run evaluation on a model with the given dataset.

    Args:
        model: The model to evaluate
        eval_dataset: The dataset to evaluate on
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        logger: Optional logger instance

    Returns:
        dict: Evaluation results
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Calculate max eval steps based on dataset size
    eval_dataset_length = len(eval_dataset)
    max_eval_steps = eval_dataset_length // batch_size

    logger.info(f"Evaluation dataset size: {eval_dataset_length}")
    logger.info(f"Max eval steps: {max_eval_steps}")
    logger.info(f"Will evaluate on {max_eval_steps * batch_size} samples")

    # Setup trainer for evaluation
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=CustomDataCollator(),
    )

    # Run evaluation
    logger.info("Running evaluation...")
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")

    # Write results to output directory
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return results


@torch.inference_mode()
def merge_and_eval(
    model_a_path: str,
    model_b_path: str,
    eval_data_files: List[str],
    alphas: List[float],
    output_dir: str,
    batch_size: int = 16,
    max_eval_samples: int = 5000,
    seq_len: int = 1024,
    lora_checkpoint_dir_a: Optional[str] = None,
    lora_checkpoint_dir_b: Optional[str] = None,
    logger=None,
):
    """
    Check linear mode connectivity by interpolating between two models and evaluating.

    Evaluates each dataset separately for post-hoc analysis.

    Args:
        model_a_path: Path to first model checkpoint
        model_b_path: Path to second model checkpoint
        eval_data_files: List of evaluation data file paths
        alphas: List of interpolation coefficients (e.g., [0.0, 0.25, 0.5, 0.75, 1.0])
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        max_eval_samples: Maximum number of samples for evaluation per dataset
        seq_len: Sequence length
        lora_checkpoint_dir_a: Optional path to LoRA checkpoint directory for model A.
        lora_checkpoint_dir_b: Optional path to LoRA checkpoint directory for model B.
        logger: Optional logger instance

    Returns:
        dict: Results for each alpha value, with per-dataset losses
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Linear Mode Connectivity Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model A: {model_a_path}")
    logger.info(f"Model B: {model_b_path}")
    logger.info(f"Alpha values: {alphas}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Load the two models (with optional LoRA merge)
    logger.info("Loading model A...")
    model_a = load_model_with_optional_lora(model_a_path, lora_checkpoint_dir_a, merge_lora=True, logger=logger)

    logger.info("Loading model B...")
    model_b = load_model_with_optional_lora(model_b_path, lora_checkpoint_dir_b, merge_lora=True, logger=logger)

    # Load evaluation datasets separately
    logger.info(f"Loading {len(eval_data_files)} evaluation dataset(s) separately")
    eval_datasets = {}
    for data_file in eval_data_files:
        dataset_name = extract_dataset_name(data_file)
        logger.info(f"Loading dataset '{dataset_name}' from {data_file}")
        dataset = get_dataset(
            data_file=data_file,
            data_work_dir=output_dir,
            sequence_length=seq_len,
        )
        dataset_size = len(dataset)
        effective_size = min(dataset_size, max_eval_samples)
        eval_datasets[dataset_name] = LimitedDataset(dataset, effective_size)
        logger.info(f"  Dataset '{dataset_name}' size: {dataset_size}, using {effective_size} samples")

    # Store results for each alpha
    all_results = {}

    for alpha in alphas:
        logger.info("=" * 80)
        logger.info(f"Evaluating with alpha = {alpha:.3f}")
        logger.info(f"  (model = {alpha:.3f} * model_a + {1 - alpha:.3f} * model_b)")
        logger.info("=" * 80)

        if alpha == 1.0:
            merged_model = model_a
        elif alpha == 0.0:
            merged_model = model_b
        else:
            merged_model = linear_interpolation_merge(model_a, model_b, alpha)

        alpha_output_dir = os.path.join(output_dir, f"alpha_{alpha:.3f}")
        os.makedirs(alpha_output_dir, exist_ok=True)

        per_dataset_results = {}
        for dataset_name, eval_dataset in eval_datasets.items():
            logger.info(f"Evaluating on dataset '{dataset_name}'...")
            dataset_output_dir = os.path.join(alpha_output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            results = eval(
                model=merged_model,
                eval_dataset=eval_dataset,
                output_dir=dataset_output_dir,
                batch_size=batch_size,
                logger=logger,
            )
            per_dataset_results[dataset_name] = results
            logger.info(f"  {dataset_name} loss: {results.get('eval_loss', 'N/A')}")

        all_results[f"alpha_{alpha:.3f}"] = {
            "alpha": alpha,
            "per_dataset_results": per_dataset_results,
        }

        if alpha != 0.0 and alpha != 1.0:
            del merged_model
            clean_gpu_memory()

        logger.info(f"Results for alpha = {alpha:.3f}:")
        for dataset_name, results in per_dataset_results.items():
            logger.info(f"  {dataset_name}: {results}")

    # Save all results
    results_file = os.path.join(output_dir, "linear_connectivity_results.json")
    logger.info(f"Saving all results to {results_file}")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    logger.info("=" * 80)
    logger.info("Summary of Results")
    logger.info("=" * 80)
    dataset_names = list(eval_datasets.keys())
    header = f"{'Alpha':<10} " + " ".join(f"{name:<15}" for name in dataset_names)
    logger.info(header)
    logger.info("-" * len(header))
    for alpha_key in sorted(all_results.keys()):
        alpha_val = all_results[alpha_key]["alpha"]
        per_dataset = all_results[alpha_key]["per_dataset_results"]
        losses = []
        for name in dataset_names:
            loss = per_dataset.get(name, {}).get("eval_loss", "N/A")
            if isinstance(loss, float):
                losses.append(f"{loss:<15.4f}")
            else:
                losses.append(f"{loss:<15}")
        logger.info(f"{alpha_val:<10.3f} " + " ".join(losses))
    logger.info("=" * 80)

    return all_results


def main():
    """Parse arguments and run evaluation."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = get_argument_parser(description="Evaluate a language model")

    # Merge/LMC arguments (activates merge mode when --model_b is provided)
    merge_group = parser.add_argument_group("merge eval (LMC)", "Linear mode connectivity evaluation between two models")
    merge_group.add_argument(
        "--model_b",
        default=None,
        help="Path to second model checkpoint. When provided, runs merge eval: "
             "interpolates between --model_dir (model A) and --model_b at multiple alphas.",
    )
    merge_group.add_argument(
        "--lora_checkpoint_dir_b",
        default=None,
        help="Path to LoRA checkpoint directory for model B.",
    )
    merge_group.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="Explicit alpha values for interpolation (e.g., 0.0 0.25 0.5 0.75 1.0).",
    )
    merge_group.add_argument(
        "--num_alphas",
        type=int,
        default=11,
        help="Number of equally-spaced alpha values (default: 11). Ignored if --alphas is provided.",
    )

    args = parser.parse_args()

    # --- Merge eval mode ---
    if args.model_b is not None:
        # Resolve eval data files
        eval_data_files = args.data_file or args.eval_data_file
        if not eval_data_files:
            raise ValueError("Either --data_file or --eval_data_file must be provided")

        # Determine alpha values
        if args.alphas is not None:
            alphas = sorted(args.alphas)
        else:
            alphas = np.linspace(0.0, 1.0, args.num_alphas).tolist()
        logger.info(f"Alpha values: {alphas}")

        merge_and_eval(
            model_a_path=args.model_dir,
            model_b_path=args.model_b,
            eval_data_files=eval_data_files,
            alphas=alphas,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_eval_samples=args.max_eval_samples,
            seq_len=args.seq_len,
            lora_checkpoint_dir_a=args.lora_checkpoint_dir,
            lora_checkpoint_dir_b=args.lora_checkpoint_dir_b,
            logger=logger,
        )
        return

    # --- Standard eval mode ---
    if args.data_file:
        eval_data_file = args.data_file
        eval_weights = args.eval_weights if len(args.data_file) > 1 else [1.0]
    elif args.eval_data_file:
        eval_data_file = args.eval_data_file
        eval_weights = args.eval_weights if len(args.eval_data_file) > 1 else [1.0]
    else:
        raise ValueError("Either --data_file or --eval_data_file must be provided")

    if len(eval_data_file) > 1:
        if eval_weights is None:
            raise ValueError(
                f"--eval_weights must be provided when using multiple evaluation datasets. "
                f"Got {len(eval_data_file)} datasets but no weights."
            )
        if len(eval_weights) != len(eval_data_file):
            raise ValueError(
                f"Number of eval_weights ({len(eval_weights)}) must match "
                f"number of data_file arguments ({len(eval_data_file)})"
            )

    if len(eval_data_file) == 1 and eval_weights is not None and eval_weights != [1.0]:
        logger.warning(
            "eval_weights provided for single evaluation dataset will be ignored"
        )

    logger.info(f"Loading model from: {args.model_dir}")
    model = load_model_with_optional_lora(args.model_dir, args.lora_checkpoint_dir, logger=logger)

    logger.info(f"Loading evaluation dataset from {len(eval_data_file)} source(s)")
    logger.info(f"Max eval samples: {args.max_eval_samples}")

    eval_weights = eval_weights if len(eval_data_file) > 1 else [1.0]

    eval_dataset = get_mixed_dataset(
        data_files=eval_data_file,
        weights=eval_weights,
        total_steps=args.max_eval_samples,
        data_work_dir=args.output_dir,
        sequence_length=args.seq_len,
    )

    eval(
        model=model,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        logger=logger,
    )


if __name__ == "__main__":
    main()
