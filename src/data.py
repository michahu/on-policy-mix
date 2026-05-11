import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from olmo_core.data import NumpyFSLDatasetConfig
from olmo_core.data.tokenizer import TokenizerConfig

log = logging.getLogger(__name__)


DEFAULT_DATA_ROOT = "/scratch/myh2014/data/"


def _read_data_mix_file(data_mix_path: str, data_root: str = DEFAULT_DATA_ROOT) -> List[str]:
    """Read URLs from a data mix file in the data_mixes folder."""

    data_mix_path = Path(data_mix_path)

    if not data_mix_path.exists():
        log.warning(f"Data mix file not found: {data_mix_path}")
        return []

    paths = []
    with open(data_mix_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(os.path.join(data_root, line))

    return paths


def get_dataset(data_file, data_work_dir, sequence_length, data_root: str = DEFAULT_DATA_ROOT):
    data_paths = _read_data_mix_file(data_file, data_root=data_root)
    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()

    return NumpyFSLDatasetConfig(
        # @willm might be called data_paths
        paths=data_paths,
        work_dir=data_work_dir,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        max_target_sequence_length=8192,
    ).build()


class CustomDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids from features, ignoring metadata
        input_ids = [f["input_ids"] for f in features]

        # Stack input_ids into a batch
        batch = {
            "input_ids": torch.stack(input_ids),
        }

        # For language modeling, labels are the same as input_ids
        batch["labels"] = batch["input_ids"].clone()

        return batch


class MixedDatasetWrapper:
    """
    Wrapper for mixing multiple datasets with specified proportions.

    Given K datasets and weights, this wrapper distributes total_steps across
    datasets according to their weights and provides a unified __getitem__ interface
    that maps global indices to dataset-specific indices.

    Example:
        For 10k steps with 3 datasets and uniform weights [1.0, 1.0, 1.0]:
        - Dataset 1 gets indices 0-3333 (maps to dataset 1, indices 0-3333)
        - Dataset 2 gets indices 3334-6666 (maps to dataset 2, indices 0-3332)
        - Dataset 3 gets indices 6667-9999 (maps to dataset 3, indices 0-3332)
    """

    def __init__(self, datasets: List[Any], weights: List[float], total_steps: int):
        """
        Initialize the mixed dataset wrapper.

        Args:
            datasets: List of K datasets to mix
            weights: List of K weights (will be normalized to sum to 1.0)
            total_steps: Total number of training steps (batches)
        """
        if len(datasets) != len(weights):
            raise ValueError(
                f"Number of datasets ({len(datasets)}) must match number of weights ({len(weights)})"
            )

        if len(datasets) == 0:
            raise ValueError("Must provide at least one dataset")

        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")

        self.datasets = datasets
        self.total_steps = total_steps

        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

        # Calculate samples per dataset
        self.samples_per_dataset = []
        allocated_samples = 0

        for i, weight in enumerate(self.weights[:-1]):
            samples = int(total_steps * weight)
            self.samples_per_dataset.append(samples)
            allocated_samples += samples

        # Give remaining samples to last dataset to ensure total = total_steps
        self.samples_per_dataset.append(total_steps - allocated_samples)

        # Assert each dataset has enough samples
        for i, (dataset, samples_needed) in enumerate(
            zip(self.datasets, self.samples_per_dataset)
        ):
            dataset_size = len(dataset)
            assert dataset_size >= samples_needed, (
                f"Dataset {i} has {dataset_size} samples but needs {samples_needed}. "
                f"Please ensure each dataset is large enough for the requested allocation."
            )

        # Build cumulative index mapping
        # cumulative_indices[i] is the starting global index for dataset i
        self.cumulative_indices = [0]
        for samples in self.samples_per_dataset:
            self.cumulative_indices.append(self.cumulative_indices[-1] + samples)

        log.info(f"MixedDatasetWrapper initialized with {len(datasets)} datasets:")
        for i, (weight, samples) in enumerate(
            zip(self.weights, self.samples_per_dataset)
        ):
            log.info(f"  Dataset {i}: weight={weight:.4f}, samples={samples}")

    def __len__(self) -> int:
        """Return total number of samples across all datasets."""
        return self.total_steps

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item at global index by mapping to appropriate dataset.

        Args:
            idx: Global index (0 to total_steps - 1)

        Returns:
            Sample from the appropriate dataset
        """
        if idx < 0 or idx >= self.total_steps:
            raise IndexError(f"Index {idx} out of range [0, {self.total_steps})")

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i in range(len(self.datasets)):
            if idx < self.cumulative_indices[i + 1]:
                dataset_idx = i
                break

        # Calculate local index within the dataset
        local_idx = idx - self.cumulative_indices[dataset_idx]

        # Get the sample from the dataset
        dataset = self.datasets[dataset_idx]
        return dataset[local_idx]


def get_mixed_dataset(
    data_files: List[str],
    weights: List[float],
    total_steps: int,
    data_work_dir: str,
    sequence_length: int,
    data_root: str = DEFAULT_DATA_ROOT,
) -> MixedDatasetWrapper:
    """
    Create a mixed dataset from multiple data files with specified proportions.

    Args:
        data_files: List of paths to data mix files
        weights: List of weights for each dataset (will be normalized)
        total_steps: Total number of training steps (batches)
        data_work_dir: Working directory for dataset creation
        sequence_length: Sequence length for each dataset

    Returns:
        MixedDatasetWrapper that mixes the datasets according to weights

    Example:
        # Mix 3 datasets uniformly for 10k steps
        mixed_dataset = get_mixed_dataset(
            data_files=["data-mixes/arxiv_train.txt",
                       "data-mixes/stackexchange_train.txt",
                       "data-mixes/reddit_train.txt"],
            weights=[1.0, 1.0, 1.0],
            total_steps=10000,
            data_work_dir="./work_dir",
            sequence_length=1024,
        )
    """
    if len(data_files) != len(weights):
        raise ValueError(
            f"Number of data files ({len(data_files)}) must match number of weights ({len(weights)})"
        )

    log.info(
        f"Creating mixed dataset from {len(data_files)} data sources for {total_steps} steps"
    )

    # Build individual datasets
    datasets = []
    for i, data_file in enumerate(data_files):
        log.info(f"Loading dataset {i} from: {data_file}")
        dataset = get_dataset(
            data_file=data_file,
            data_work_dir=data_work_dir,
            sequence_length=sequence_length,
            data_root=data_root,
        )
        datasets.append(dataset)
        log.info(f"  Dataset {i} size: {len(dataset)}")

    # Create mixed dataset wrapper
    return MixedDatasetWrapper(
        datasets=datasets, weights=weights, total_steps=total_steps
    )
