"""Merge a LoRA adapter into its base model via scaled linear interpolation.

Given a base model and a LoRA checkpoint trained on top of it, produces:
    merged = alpha_new * (base + LoRA) + (1 - alpha_new) * base

This is equivalent to scaling the LoRA delta by alpha_new and adding it to the base.

Usage:
    python -m pipeline.merge_lora \
        --base_model runs/continual_lora_merge/ord0/150M/stage_1_algebraic_stack \
        --lora_checkpoint_dir runs/.../lora_train/checkpoint-10000 \
        --alpha_new 0.7 \
        --output_dir runs/.../stage_2/merged
"""

import argparse

from src.merge import linear_interpolation_merge
from src.utils import load_model_with_optional_lora


def merge_lora_into_base(base_model, lora_checkpoint_dir, alpha_new, output_dir):
    """Merge LoRA adapter into base model with interpolation weight.

    Args:
        base_model: Path to base model checkpoint.
        lora_checkpoint_dir: Path to LoRA checkpoint (contains adapter_model.safetensors).
        alpha_new: Weight for the LoRA-merged model (0-1).
        output_dir: Directory to save the merged model.
    """
    from pathlib import Path

    print(f"Loading base + LoRA merged from {lora_checkpoint_dir}...")
    model_a = load_model_with_optional_lora(base_model, lora_checkpoint_dir, merge_lora=True)

    print(f"Loading plain base model from {base_model}...")
    model_b = load_model_with_optional_lora(base_model)

    print(f"Interpolating: {alpha_new} * (base+LoRA) + {1 - alpha_new} * base ...")
    merged = linear_interpolation_merge(model_a, model_b, alpha=alpha_new)

    del model_a, model_b
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out}...")
    merged.save_pretrained(str(out))
    print(f"Done. Sentinel: {out / 'model.safetensors'}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model via scaled interpolation.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_checkpoint_dir", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--alpha_new", type=float, required=True, help="Interpolation weight for LoRA-merged model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged model")
    args = parser.parse_args()

    merge_lora_into_base(args.base_model, args.lora_checkpoint_dir, args.alpha_new, args.output_dir)


if __name__ == "__main__":
    main()
