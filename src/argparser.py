"""
Shared argument parser for training and evaluation scripts.
"""

import argparse


def get_argument_parser(description="Finetune a language model"):
    """Create and return an argument parser for training/evaluation."""
    parser = argparse.ArgumentParser(description=description)

    # Model arguments
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to model directory (local path or HuggingFace model name)",
    )
    parser.add_argument(
        "--model_init_mode",
        type=str,
        default="pretrained",
        choices=["pretrained", "config"],
        help="How to initialize the model from --model_dir: "
        "'pretrained' loads weights, 'config' loads only the architecture config and initializes fresh weights.",
    )

    # PEFT arguments
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        default=None,
        help="Path to pre-initialized LoRA checkpoint directory (created using svd_init.py). "
        "This supports various initialization methods like PiSSA, MiLoRA, etc. "
        "If provided, this takes precedence over --use_lora.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA (Low-Rank Adaptation) fine-tuning with standard initialization. "
        "Ignored if --lora_checkpoint_dir is provided.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling parameter (default: 16)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability (default: 0.05)",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["attn_out", "ff_out", "att_proj", "ff_proj"],
        help="Space-separated list of target modules for LoRA (e.g., 'attn_out att_proj ff_proj').",
    )
    parser.add_argument(
        "--lora_exclude_modules",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of module name suffixes to exclude from LoRA (e.g., 'transformer.ff_out' "
        "to skip the LM head while keeping per-block ff_out).",
    )

    # Data arguments
    parser.add_argument(
        "--train_data_file",
        required=False,
        nargs="+",
        help="Path(s) to the file(s) of paths of tokenized training data. Can specify multiple files for dataset mixture.",
    )
    parser.add_argument(
        "--train_weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each training dataset (will be normalized). Required when using multiple train_data_file arguments.",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        nargs="+",
        help="Path(s) to the file(s) of paths of tokenized evaluation data (optional). Can specify multiple files for dataset mixture.",
    )
    parser.add_argument(
        "--eval_weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each evaluation dataset (will be normalized). Required when using multiple eval_data_file arguments.",
    )
    parser.add_argument(
        "--data_file",
        default=None,
        nargs="+",
        help="Path(s) to the file(s) of paths of tokenized data (for evaluation script). Can specify multiple files for dataset mixture.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="Sequence length (default: 1024)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory for data files. Overrides DEFAULT_DATA_ROOT in src/data.py. "
        "Can also be set via DATA_ROOT environment variable.",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save model checkpoints and results",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum number of training steps (default: 10000). Set to -1 to use num_epochs instead.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3). Only used if max_steps is -1.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training and evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps (default: 100)",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="wsd",
        help="Learning rate scheduler type (default: 'wsd' = warmup-stable-decay)",
    )
    parser.add_argument(
        "--min_lr_rate",
        type=float,
        default=0.1,
        help="Minimum learning rate as a ratio of the initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--decay_steps",
        type=int,
        default=1000,
        help="Number of LR decay steps at the end of training (default: 1000). Used with 'wsd' scheduler.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    # Optimizer arguments
    parser.add_argument(
        "--use_custom_optimizer",
        action="store_true",
        help="Use custom optimizer from src/custom_optimizer.py",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="Optimizer type (default: 'adamw')",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--lora_lr_multiplier",
        type=float,
        default=1.0,
        help="Learning rate multiplier for LoRA parameters (default: 1.0)",
    )

    # L2 initialization regularization
    parser.add_argument(
        "--use_l2_init",
        action="store_true",
        help="Enable L2 regularization towards initial pretrained parameters",
    )
    parser.add_argument(
        "--l2_init_weight",
        type=float,
        default=0.01,
        help="Weight for L2 regularization towards initial parameters (default: 0.01)",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps (default: 10)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=0,
        help="Save checkpoint every N steps (default: 0). Set to 0 to disable checkpoint saving.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=2500,
        help="Evaluate every N steps (default: 2000)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=5000,
        help="Maximum number of samples to use for evaluation (default: 5000)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    # Precision arguments
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Default dtype for training (default: bfloat16)",
    )

    # Dataloader arguments
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers (default: 0). Set to number of CPUs for better GPU utilization.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to simulate larger batch sizes (default: 1)",
    )

    return parser
