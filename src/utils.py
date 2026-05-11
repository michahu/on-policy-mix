"""
Shared utilities for model merging and evaluation scripts.
"""

import logging
import os
from typing import Any, Optional

import torch
from hf_olmo import OLMoForCausalLM
from peft import PeftModel


class LimitedDataset:
    """
    Wrapper to limit dataset size for evaluation.

    Useful when you want to evaluate on a subset of a larger dataset
    without loading the entire thing into memory differently.
    """

    def __init__(self, dataset: Any, max_samples: int):
        self.dataset = dataset
        self.max_samples = min(max_samples, len(dataset))

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int) -> Any:
        if idx >= self.max_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.max_samples})")
        return self.dataset[idx]


def load_model_with_optional_lora(
    model_path: str,
    lora_checkpoint_dir: Optional[str] = None,
    merge_lora: bool = True,
    is_trainable: bool = False,
    logger: Optional[logging.Logger] = None,
) -> OLMoForCausalLM:
    """
    Load a model, optionally with LoRA weights.

    Args:
        model_path: Path to base model checkpoint
        lora_checkpoint_dir: Optional path to LoRA checkpoint directory
        merge_lora: If True, merge LoRA weights into base model (for inference).
                   If False, keep as PeftModel (for continued training).
        is_trainable: Whether the LoRA weights should be trainable (only relevant if merge_lora=False)
        logger: Optional logger instance

    Returns:
        Model with LoRA weights applied (merged or as PeftModel)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Loading model from {model_path}")
    model = OLMoForCausalLM.from_pretrained(model_path)

    if lora_checkpoint_dir is not None:
        logger.info(f"Loading LoRA checkpoint from {lora_checkpoint_dir}")
        model = PeftModel.from_pretrained(
            model, lora_checkpoint_dir, is_trainable=is_trainable
        )

        if merge_lora:
            logger.info("Merging LoRA weights into model...")
            model = model.merge_and_unload()
            logger.info("LoRA weights merged successfully")

    return model


def compute_objective(
    losses: dict,
    objective_type: str = "sum",
    task_weights: Optional[dict] = None,
) -> float:
    """
    Compute a scalar objective from per-task losses.

    Args:
        losses: Dictionary mapping task names to their losses
        objective_type: How to combine losses:
            - "sum": Sum of all losses
            - "max": Maximum loss (minimax objective)
            - "weighted_sum": Weighted sum using task_weights
            - "mean": Mean of all losses
        task_weights: Weights for each task (required for weighted_sum)

    Returns:
        Scalar objective value
    """
    loss_values = list(losses.values())

    if objective_type == "sum":
        return sum(loss_values)
    elif objective_type == "mean":
        return sum(loss_values) / len(loss_values)
    elif objective_type == "max":
        return max(loss_values)
    elif objective_type == "weighted_sum":
        if task_weights is None:
            raise ValueError("task_weights required for weighted_sum objective")
        return sum(task_weights.get(k, 1.0) * v for k, v in losses.items())
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_dataset_name(data_file: str) -> str:
    """
    Extract a clean dataset name from a data file path.

    Examples:
        "data-mixes/arxiv_eval.txt" -> "arxiv"
        "/path/to/stackexchange_train.txt" -> "stackexchange"
    """
    basename = os.path.basename(data_file)
    name = basename.replace(".txt", "")
    name = name.replace("_eval", "").replace("_train", "").replace("_test", "")
    return name
