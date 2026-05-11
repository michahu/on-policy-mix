"""Continual learning pipeline: naive, mix, OnPolicyMix, lora_merge, and mix_sweep conditions.

Usage:
    python -m pipeline.continual naive [--ordering_name ord0] [--model_size 150M] [--dry_run]
    python -m pipeline.continual mix [--ordering_name ord0] [--model_size 150M] [--dry_run]
    python -m pipeline.continual olmix [--ordering_name ord0] [--model_size 150M] [--dry_run]
    python -m pipeline.continual lora_merge [--ordering_name ord0] [--model_size 150M] [--dry_run]
    python -m pipeline.continual mix_sweep [--ordering_name ord0] [--model_size 150M] [--stage 2]
    python -m pipeline.continual status [--condition all]
    python -m pipeline.continual retry --condition naive --ordering_name ord2 --stage 3
"""

import json
import os
import subprocess
from pathlib import Path

import fire

from pipeline.olmix import (
    get_final_eval_losses,
    get_merge_rows,
    run_olmix_fit,
    write_csvs,
    write_olmix_config,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = ["algebraic_stack", "arxiv", "open_web_math", "reddit", "stackexchange"]

ORDERINGS = {
    "ord0": ["algebraic_stack", "arxiv", "open_web_math", "reddit", "stackexchange"],
    "ord1": ["arxiv", "open_web_math", "reddit", "stackexchange", "algebraic_stack"],
    "ord2": ["open_web_math", "reddit", "stackexchange", "algebraic_stack", "arxiv"],
    "ord3": ["reddit", "stackexchange", "algebraic_stack", "arxiv", "open_web_math"],
    "ord4": ["stackexchange", "algebraic_stack", "arxiv", "open_web_math", "reddit"],
    # Branching variants: same stage-1 dataset (open_web_math) as ord2 but a
    # different stage-2 target. Used to run OnPolicyMix for the
    # 12 branching_sweep conditions; only stage 2 is meaningful, the tail
    # datasets are placeholders.
    # "ord_owm_alg": ["open_web_math", "algebraic_stack", "arxiv", "reddit", "stackexchange"],
    # "ord_owm_arx": ["open_web_math", "arxiv", "algebraic_stack", "reddit", "stackexchange"],
    # "ord_owm_se":  ["open_web_math", "stackexchange", "algebraic_stack", "arxiv", "reddit"],
}

MODEL_CONFIGS = {
    "150M": {
        "base_model": "allenai/DataDecide-c4-150M",
        "learning_rate": "5e-4",
        "batch_size": 32,
        "max_steps": 10000,
        "warmup_steps": 1000,
        "sbatch_opts": ["--constraint=a100|h100|l40s"],
    },
    "300M": {
        "base_model": "allenai/DataDecide-c4-300M",
        "learning_rate": "4e-4",
        "batch_size": 32,
        "max_steps": 12500,
        "warmup_steps": 1250,
        "sbatch_opts": ["--constraint=a100|h100"],
    },
    "530M": {
        "base_model": "allenai/DataDecide-c4-530M",
        "learning_rate": "3e-4",
        "batch_size": 32,
        "max_steps": 15000,
        "warmup_steps": 1500,
        "sbatch_opts": ["--constraint=a100|h100|h200"],
    },
    "1B": {
        "base_model": "allenai/DataDecide-c4-1B",
        "learning_rate": "1e-4",
        "batch_size": 16,
        "max_steps": 60000,
        "warmup_steps": 6000,
        "sbatch_opts": ["--constraint=h100|h200"],
    },
}


def _get_lora_lr(model_size):
    """LoRA learning rate: 2x the base learning rate."""
    base_lr = float(MODEL_CONFIGS[model_size]["learning_rate"])
    return str(base_lr * 2)


EVAL_FILES = [f"data-mixes/{ds}_eval.txt" for ds in DATASETS]
OLD_MIX_KEY = "old"

RUNS_ROOT = Path("runs")
SCRIPTS_DIR = Path("scripts")


# ---------------------------------------------------------------------------
# StateFile: atomic JSON state persistence
# ---------------------------------------------------------------------------


class StateFile:
    """Read/write JSON state with atomic saves."""

    def __init__(self, path):
        self.path = Path(path)
        self._data = {}
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)

    @property
    def data(self):
        return self._data

    def get(self, *keys, default=None):
        d = self._data
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    def set(self, *keys_and_value):
        """set('stages', '1', 'status', 'COMPLETED') sets nested path."""
        *keys, value = keys_and_value
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# SlurmHelper
# ---------------------------------------------------------------------------


class SlurmHelper:
    """Submit and check Slurm jobs."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    def submit(self, script, args, job_name, dependency=None, sbatch_opts=None):
        """Submit a Slurm job. Returns job_id string, or 'DRY_RUN'."""
        cmd = ["sbatch", f"--job-name={job_name}"]
        if dependency:
            cmd.append(f"--dependency=afterok:{dependency}")
        if sbatch_opts:
            cmd.extend(sbatch_opts)
        cmd.append(str(script))
        cmd.extend(args)

        if self.dry_run:
            print(f"  [DRY RUN] {' '.join(cmd)}")
            return "DRY_RUN"

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  sbatch failed: {result.stderr.strip()}")
            return None
        # "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted job {job_id} ({job_name})")
        return job_id

    def check_status(self, job_id):
        """Check job status via sacct. Returns PENDING/RUNNING/COMPLETED/FAILED/UNKNOWN."""
        if job_id == "DRY_RUN":
            return "COMPLETED"
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return "UNKNOWN"
            states = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            if not states:
                return "UNKNOWN"
            # First line is the main job state
            state = states[0].split()[0] if states[0] else "UNKNOWN"
            if state in ("COMPLETED",):
                return "COMPLETED"
            elif state in ("PENDING", "REQUEUED"):
                return "PENDING"
            elif state in ("RUNNING", "COMPLETING"):
                return "RUNNING"
            elif state in (
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "CANCELLED+",
            ):
                return "FAILED"
            return "UNKNOWN"
        except Exception:
            return "UNKNOWN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sentinel_exists(output_dir, sentinel="model.safetensors"):
    """Check if a sentinel file exists in output_dir (training completion)."""
    p = Path(output_dir)
    # Single-file, sharded safetensors, or LoRA adapter
    return (
        (p / sentinel).exists()
        or (p / "model.safetensors.index.json").exists()
        or (p / "adapter_model.safetensors").exists()
    )


def _lmc_sentinel_exists(output_dir):
    """Check if LMC results exist."""
    p = Path(output_dir)
    # Look for linear_connectivity_results.json in any checkpoint subdir
    for ckpt in sorted(p.iterdir()) if p.exists() else []:
        if ckpt.is_dir() and ckpt.name.startswith("checkpoint-"):
            if (ckpt / "linear_connectivity_results.json").exists():
                return True
    return False


def distribute_weights_proportional(alpha_new, new_dataset, prev_weights):
    """Distribute weights for k-way training mix.

    alpha_new: olmix weight for D_k (the new dataset).
    prev_weights: {dataset: weight} from stage k-1's training (sums to 1.0).
    Returns: {dataset: weight} for all k datasets, summing to 1.0.
    """
    result = {new_dataset: alpha_new}
    prior_sum = sum(prev_weights.values())
    for d, w in prev_weights.items():
        result[d] = (1 - alpha_new) * w / prior_sum
    return result


def distribute_weights_uniform(alpha_new, new_dataset, prev_datasets):
    """Distribute weights for k-way training mix (ODEN).

    Assigns alpha_new to new dataset, distributes (1-alpha_new)
    uniformly among all previous datasets.
    """
    result = {new_dataset: alpha_new}
    n_old = len(prev_datasets)
    old_weight = (1 - alpha_new) / n_old
    for d in prev_datasets:
        result[d] = old_weight
    return result


def normalize_weights(weights):
    """Normalize a positive mixture dict."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {d: w / total for d, w in weights.items()}


def expand_on_policy_mix_weights(
    opt_weights,
    prev_weights,
    old_key=OLD_MIX_KEY,
):
    """Expand a collapsed-old fit into full dataset weights."""
    w_old = opt_weights.get(old_key, 0.0)
    old_weights = normalize_weights(prev_weights)
    expanded = {d: w_old * w for d, w in old_weights.items()}
    for dataset, weight in opt_weights.items():
        if dataset == old_key:
            continue
        expanded[dataset] = expanded.get(dataset, 0.0) + weight
    return expanded


def _train_args(
    model_dir,
    train_data_files,
    output_dir,
    train_weights=None,
    max_steps=10000,
    learning_rate="5e-4",
    batch_size=32,
    warmup_steps=0,
    lr_scheduler_type="wsd",
    use_lora=False,
    lora_r=None,
    lora_alpha=None,
    save_steps=None,
    eval_steps=None,
    seed=None,
):
    """Build argument list for train.py via slurm_launch_train.sh."""
    args = [
        "--model_dir",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--max_steps",
        str(max_steps),
        "--learning_rate",
        str(learning_rate),
        "--batch_size",
        str(batch_size),
        "--warmup_steps",
        str(warmup_steps),
        "--lr_scheduler_type",
        lr_scheduler_type,
    ]
    # training data
    args.append("--train_data_file")
    args.extend(str(f) for f in train_data_files)
    # eval data
    args.append("--eval_data_file")
    args.extend(EVAL_FILES)
    # weights
    if train_weights:
        args.append("--train_weights")
        args.extend(str(w) for w in train_weights)
    # LoRA
    if use_lora:
        args.extend(
            ["--use_lora", "--lora_r", str(lora_r), "--lora_alpha", str(lora_alpha)]
        )
    # checkpointing
    if save_steps is not None:
        args.extend(["--save_steps", str(save_steps)])
    if eval_steps is not None:
        args.extend(["--eval_steps", str(eval_steps)])
    if seed is not None:
        args.extend(["--seed", str(seed)])
    # wandb
    args.append("--use_wandb")
    return args


def _lmc_args(model_dir, model_b, lora_checkpoint_dir, output_dir, batch_size=64):
    """Build argument list for eval.py merge mode via slurm_launch_lmc.sh."""
    args = [
        "--model_dir",
        str(model_dir),
        "--model_b",
        str(model_b),
        "--lora_checkpoint_dir",
        str(lora_checkpoint_dir),
        "--output_dir",
        str(output_dir),
        "--batch_size",
        str(batch_size),
        "--alphas",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
    ]
    args.append("--eval_data_file")
    args.extend(EVAL_FILES)
    return args


# ---------------------------------------------------------------------------
# NaiveExperiment
# ---------------------------------------------------------------------------


class NaiveExperiment:
    """Manages one ordering for the naive sequential condition."""

    def __init__(self, ordering_name, model_size="150M", dry_run=False, seed=None):
        self.ordering_name = ordering_name
        self.ordering = ORDERINGS[ordering_name]
        self.model_size = model_size
        self.seed = seed
        cfg = MODEL_CONFIGS[model_size]
        self.base_model = cfg["base_model"]
        self.learning_rate = cfg["learning_rate"]
        self.batch_size = cfg["batch_size"]
        self.max_steps = cfg["max_steps"]
        self.warmup_steps = cfg["warmup_steps"]
        self.sbatch_opts = cfg["sbatch_opts"]
        self.run_dir = RUNS_ROOT / "continual_naive" / ordering_name / model_size
        self.slurm = SlurmHelper(dry_run=dry_run)
        self.dry_run = dry_run

        self.state = StateFile(self.run_dir / "state.json")
        if not self.state.data:
            self.state._data = {
                "condition": "naive",
                "ordering_name": ordering_name,
                "ordering": self.ordering,
                "model_size": model_size,
                "base_model": self.base_model,
                "stages": {},
            }
            self.state.save()

    def _stage_output_dir(self, k):
        ds = self.ordering[k - 1]
        return self.run_dir / f"stage_{k}_{ds}"

    def _prev_checkpoint(self, k):
        if k == 1:
            return self.base_model
        return str(self._stage_output_dir(k - 1))

    def run(self):
        """Iterate stages 1-5, advancing the state machine."""
        print(f"=== Naive {self.ordering_name}: {' -> '.join(self.ordering)} ===\n")

        for k in range(1, 6):
            ds = self.ordering[k - 1]
            stage_key = str(k)
            stage = self.state.get("stages", stage_key) or {}
            status = stage.get("status", "PENDING")

            print(f"Stage {k} ({ds}): {status}")

            if status == "COMPLETED":
                continue

            if status == "SUBMITTED":
                # Check Slurm job status
                job_id = stage.get("job_id")
                slurm_status = self.slurm.check_status(job_id)
                output_dir = stage["output_dir"]

                if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                    self.state.set("stages", stage_key, "status", "COMPLETED")
                    self.state.save()
                    print(f"  -> COMPLETED (job {job_id})")
                    continue
                elif slurm_status in ("RUNNING", "PENDING"):
                    print(f"  -> Still {slurm_status} (job {job_id})")
                    return
                elif slurm_status == "COMPLETED" and not _sentinel_exists(output_dir):
                    print(f"  -> Job completed but sentinel missing in {output_dir}")
                    self.state.set("stages", stage_key, "status", "FAILED")
                    self.state.save()
                    return
                else:
                    print(f"  -> FAILED (job {job_id}, sacct={slurm_status})")
                    self.state.set("stages", stage_key, "status", "FAILED")
                    self.state.save()
                    return

            if status == "FAILED":
                print(f"  -> Stage failed. Use 'retry' to reset.")
                return

            # PENDING: submit job
            output_dir = str(self._stage_output_dir(k))
            train_file = f"data-mixes/{ds}_train_s{k}.txt"
            warmup = self.warmup_steps if k == 1 else 0

            args = _train_args(
                model_dir=self._prev_checkpoint(k),
                train_data_files=[train_file],
                output_dir=output_dir,
                max_steps=self.max_steps,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                warmup_steps=warmup,
                seed=self.seed,
            )

            job_id = self.slurm.submit(
                SCRIPTS_DIR / "slurm_launch_train.sh",
                args,
                job_name=f"naive-{self.ordering_name}-s{k}",
                sbatch_opts=self.sbatch_opts,
            )

            if job_id is None:
                print(f"  -> Failed to submit job")
                return

            if self.dry_run:
                continue

            self.state.set(
                "stages",
                stage_key,
                {
                    "status": "SUBMITTED",
                    "job_id": job_id,
                    "output_dir": output_dir,
                    "dataset": ds,
                    "train_weights": {ds: 1.0},
                },
            )
            self.state.save()
            print(f"  -> Submitted, stopping to wait for completion")
            return

        print("\nAll stages complete!")


# ---------------------------------------------------------------------------
# MixExperiment
# ---------------------------------------------------------------------------


class MixExperiment:
    """Manages one ordering for the fixed-mix baseline condition.

    Stage 1: 100% D1 (identical to naive).
    Stage k>=2: 90% D_k, remaining 10% split equally across D_1...D_{k-1}.
    """

    def __init__(self, ordering_name, model_size="150M", dry_run=False, seed=None):
        self.ordering_name = ordering_name
        self.ordering = ORDERINGS[ordering_name]
        self.model_size = model_size
        self.seed = seed
        cfg = MODEL_CONFIGS[model_size]
        self.base_model = cfg["base_model"]
        self.learning_rate = cfg["learning_rate"]
        self.batch_size = cfg["batch_size"]
        self.max_steps = cfg["max_steps"]
        self.warmup_steps = cfg["warmup_steps"]
        self.sbatch_opts = cfg["sbatch_opts"]
        self.run_dir = RUNS_ROOT / "continual_mix" / ordering_name / model_size
        self.slurm = SlurmHelper(dry_run=dry_run)
        self.dry_run = dry_run

        self.state = StateFile(self.run_dir / "state.json")
        if not self.state.data:
            self.state._data = {
                "condition": "mix",
                "ordering_name": ordering_name,
                "ordering": self.ordering,
                "model_size": model_size,
                "base_model": self.base_model,
                "stages": {},
            }
            self.state.save()

    def _stage_output_dir(self, k):
        ds = self.ordering[k - 1]
        return self.run_dir / f"stage_{k}_{ds}"

    def _prev_checkpoint(self, k):
        if k == 1:
            return self.base_model
        return str(self._stage_output_dir(k - 1))

    def _mix_weights(self, k):
        """Compute fixed 90/10 mix weights for stage k.

        Returns: {dataset: weight} dict summing to 1.0.
        """
        ds = self.ordering[k - 1]
        if k == 1:
            return {ds: 1.0}
        prior_datasets = self.ordering[: k - 1]
        old_per_dataset = 0.1 / len(prior_datasets)
        weights = {ds: 0.9}
        for d in prior_datasets:
            weights[d] = old_per_dataset
        return weights

    def run(self):
        """Iterate stages 1-5, advancing the state machine."""
        print(f"=== Mix {self.ordering_name}: {' -> '.join(self.ordering)} ===\n")

        for k in range(1, 6):
            ds = self.ordering[k - 1]
            stage_key = str(k)
            stage = self.state.get("stages", stage_key) or {}
            status = stage.get("status", "PENDING")

            print(f"Stage {k} ({ds}): {status}")

            if status == "COMPLETED":
                continue

            if status == "SUBMITTED":
                job_id = stage.get("job_id")
                slurm_status = self.slurm.check_status(job_id)
                output_dir = stage["output_dir"]

                if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                    self.state.set("stages", stage_key, "status", "COMPLETED")
                    self.state.save()
                    print(f"  -> COMPLETED (job {job_id})")
                    continue
                elif slurm_status in ("RUNNING", "PENDING"):
                    print(f"  -> Still {slurm_status} (job {job_id})")
                    return
                elif slurm_status == "COMPLETED" and not _sentinel_exists(output_dir):
                    print(f"  -> Job completed but sentinel missing in {output_dir}")
                    self.state.set("stages", stage_key, "status", "FAILED")
                    self.state.save()
                    return
                else:
                    print(f"  -> FAILED (job {job_id}, sacct={slurm_status})")
                    self.state.set("stages", stage_key, "status", "FAILED")
                    self.state.save()
                    return

            if status == "FAILED":
                print(f"  -> Stage failed. Use 'retry' to reset.")
                return

            # PENDING: submit job
            output_dir = str(self._stage_output_dir(k))
            weights = self._mix_weights(k)
            train_datasets = list(weights.keys())
            train_files = [f"data-mixes/{d}_train_s{k}.txt" for d in train_datasets]
            train_weight_values = [weights[d] for d in train_datasets]
            warmup = self.warmup_steps if k == 1 else 0

            print(f"  weights: {weights}")

            args = _train_args(
                model_dir=self._prev_checkpoint(k),
                train_data_files=train_files,
                output_dir=output_dir,
                train_weights=train_weight_values if k > 1 else None,
                max_steps=self.max_steps,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                warmup_steps=warmup,
                seed=self.seed,
            )

            job_id = self.slurm.submit(
                SCRIPTS_DIR / "slurm_launch_train.sh",
                args,
                job_name=f"mix-{self.ordering_name}-s{k}",
                sbatch_opts=self.sbatch_opts,
            )

            if job_id is None:
                print(f"  -> Failed to submit job")
                return

            if self.dry_run:
                continue

            self.state.set(
                "stages",
                stage_key,
                {
                    "status": "SUBMITTED",
                    "job_id": job_id,
                    "output_dir": output_dir,
                    "dataset": ds,
                    "train_weights": weights,
                },
            )
            self.state.save()
            print(f"  -> Submitted, stopping to wait for completion")
            return

        print("\nAll stages complete!")


# ---------------------------------------------------------------------------
# OnPolicyMixExperiment
# ---------------------------------------------------------------------------


class OnPolicyMixExperiment:
    """Manages one ordering for the OnPolicyMix continual condition."""

    def __init__(
        self,
        ordering_name,
        model_size="150M",
        dry_run=False,
        prior_mode="uniform_2",
        seed=None,
        stop_after=None,
    ):
        self.ordering_name = ordering_name
        self.ordering = ORDERINGS[ordering_name]
        self.model_size = model_size
        self.prior_mode = prior_mode
        self.seed = seed
        # If set to a substage name (e.g. "olmix_fit"), stop after that substage
        # advances. Used to get the OnPolicyMix recommendation without paying
        # for full_train or any later stages.
        self.stop_after = stop_after
        cfg = MODEL_CONFIGS[model_size]
        self.base_model = cfg["base_model"]
        self.learning_rate = cfg["learning_rate"]
        self.batch_size = cfg["batch_size"]
        self.max_steps = cfg["max_steps"]
        self.warmup_steps = cfg["warmup_steps"]
        self.sbatch_opts = cfg["sbatch_opts"]
        suffix = "_oden" if prior_mode == "oden" else ""
        self.run_dir = (
            RUNS_ROOT / f"continual_olmix{suffix}" / ordering_name / model_size
        )
        self.slurm = SlurmHelper(dry_run=dry_run)
        self.dry_run = dry_run

        self.state = StateFile(self.run_dir / "state.json")
        if not self.state.data:
            self.state._data = {
                "condition": "olmix",
                "ordering_name": ordering_name,
                "ordering": self.ordering,
                "model_size": model_size,
                "base_model": self.base_model,
                "stages": {},
            }
            self.state.save()

    def _stage_dir(self, k):
        if k == 1:
            ds = self.ordering[0]
            return self.run_dir / f"stage_1_{ds}"
        return self.run_dir / f"stage_{k}"

    def _prev_checkpoint(self, k):
        if k == 1:
            return self.base_model
        if k == 2:
            return str(self._stage_dir(1))
        return str(self._stage_dir(k - 1) / "full_train")

    def run(self):
        """Iterate stages 1-5 with substages for k>=2."""
        print(
            f"=== OnPolicyMix {self.ordering_name}: {' -> '.join(self.ordering)} ===\n"
        )

        for k in range(1, 6):
            ds = self.ordering[k - 1]
            stage_key = str(k)
            stage = self.state.get("stages", stage_key) or {}
            status = stage.get("status", "PENDING")

            print(f"Stage {k} ({ds}): {status}")

            if status == "COMPLETED":
                continue

            if k == 1:
                # Stage 1 is identical to naive
                result = self._run_stage_1(stage, stage_key, ds)
            else:
                result = self._run_stage_k(k, stage, stage_key, ds)

            if result == "STOP":
                return
            # result == "CONTINUE" means advance to next stage

        print("\nAll stages complete!")

    def _run_stage_1(self, stage, stage_key, ds):
        """Handle stage 1 (simple training, same as naive)."""
        status = stage.get("status", "PENDING")

        if status == "SUBMITTED":
            job_id = stage.get("job_id")
            slurm_status = self.slurm.check_status(job_id)
            output_dir = stage["output_dir"]

            if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                self.state.set("stages", stage_key, "status", "COMPLETED")
                self.state.save()
                print(f"  -> COMPLETED")
                return "CONTINUE"
            elif slurm_status in ("RUNNING", "PENDING"):
                print(f"  -> Still {slurm_status} (job {job_id})")
                return "STOP"
            elif slurm_status == "COMPLETED" and not _sentinel_exists(output_dir):
                self.state.set("stages", stage_key, "status", "FAILED")
                self.state.save()
                print(f"  -> Job completed but sentinel missing")
                return "STOP"
            else:
                self.state.set("stages", stage_key, "status", "FAILED")
                self.state.save()
                print(f"  -> FAILED (sacct={slurm_status})")
                return "STOP"

        if status == "FAILED":
            print(f"  -> Stage failed. Use 'retry' to reset.")
            return "STOP"

        # PENDING: submit
        output_dir = str(self._stage_dir(1))
        train_file = f"data-mixes/{ds}_train_s1.txt"

        args = _train_args(
            model_dir=self.base_model,
            train_data_files=[train_file],
            output_dir=output_dir,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            seed=self.seed,
        )

        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"olmix-{self.ordering_name}-s1",
            sbatch_opts=self.sbatch_opts,
        )

        if job_id is None:
            return "STOP"

        if self.dry_run:
            return "CONTINUE"

        self.state.set(
            "stages",
            stage_key,
            {
                "status": "SUBMITTED",
                "job_id": job_id,
                "output_dir": output_dir,
                "dataset": ds,
                "train_weights": {ds: 1.0},
            },
        )
        self.state.save()
        print(f"  -> Submitted, stopping to wait")
        return "STOP"

    def _run_stage_k(self, k, stage, stage_key, ds):
        """Handle stage k>=2: lora_probe -> lmc -> olmix_fit -> full_train."""
        substage = stage.get("substage", "lora_probe")
        stage_dir = self._stage_dir(k)
        prev_checkpoint = self._prev_checkpoint(k)
        prev_ds = self.ordering[k - 2]

        print(f"  substage: {substage}")

        # --- lora_probe ---
        if substage == "lora_probe":
            result = self._run_lora_probe(
                k, stage, stage_key, stage_dir, prev_checkpoint, ds, prev_ds
            )
            if result != "ADVANCE":
                return result
            if self.stop_after == "lora_probe":
                print(f"  -> stop_after=lora_probe, halting")
                return "STOP"
            substage = "lmc"

        # --- lmc ---
        if substage == "lmc":
            result = self._run_lmc(k, stage, stage_key, stage_dir, prev_checkpoint)
            if result != "ADVANCE":
                return result
            if self.stop_after == "lmc":
                print(f"  -> stop_after=lmc, halting")
                return "STOP"
            substage = "olmix_fit"

        # --- olmix_fit ---
        if substage == "olmix_fit":
            result = self._run_olmix_fit(k, stage, stage_key, stage_dir, ds, prev_ds)
            if result != "ADVANCE":
                return result
            if self.stop_after == "olmix_fit":
                print(f"  -> stop_after=olmix_fit, halting")
                return "STOP"
            substage = "full_train"

        # --- full_train ---
        if substage == "full_train":
            result = self._run_full_train(
                k, stage, stage_key, stage_dir, prev_checkpoint, ds
            )
            return result

        return "STOP"

    def _run_lora_probe(
        self, k, stage, stage_key, stage_dir, prev_checkpoint, ds, prev_ds
    ):
        """Submit/check LoRA probe job on the newly introduced dataset."""
        lora_state = stage.get("lora_probe", {})
        lp_status = lora_state.get("status", "PENDING")

        if lp_status == "COMPLETED":
            return "ADVANCE"

        if lp_status == "SUBMITTED":
            job_id = lora_state["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            output_dir = lora_state["output_dir"]

            if slurm_status == "COMPLETED" and _sentinel_exists(
                output_dir, sentinel="adapter_model.safetensors"
            ):
                self.state.set("stages", stage_key, "lora_probe", "status", "COMPLETED")
                self.state.set("stages", stage_key, "substage", "lmc")
                self.state.save()
                print(f"    lora_probe -> COMPLETED")
                return "ADVANCE"
            elif slurm_status in ("RUNNING", "PENDING"):
                print(f"    lora_probe -> {slurm_status} (job {job_id})")
                return "STOP"
            else:
                self.state.set("stages", stage_key, "lora_probe", "status", "FAILED")
                self.state.set("stages", stage_key, "status", "FAILED")
                self.state.save()
                print(f"    lora_probe -> FAILED")
                return "STOP"

        if lp_status == "FAILED":
            print(f"    lora_probe -> FAILED. Use 'retry' to reset.")
            return "STOP"

        # PENDING: submit a 90/10 new-vs-old probe mix to avoid overstating
        # forgetting relative to the later full-train mix expansion.
        output_dir = str(stage_dir / "lora_probe")
        prev_stage = self.state.get("stages", str(k - 1)) or {}
        prev_weights = prev_stage.get("train_weights", {self.ordering[k - 2]: 1.0})
        probe_weights = distribute_weights_proportional(0.9, ds, prev_weights)
        train_datasets = list(probe_weights.keys())
        train_files = [f"data-mixes/{d}_train_s{k}.txt" for d in train_datasets]
        train_weight_values = [probe_weights[d] for d in train_datasets]

        args = _train_args(
            model_dir=prev_checkpoint,
            train_data_files=train_files,
            output_dir=output_dir,
            train_weights=train_weight_values,
            max_steps=int(self.max_steps * 0.4),
            learning_rate=_get_lora_lr(self.model_size),
            batch_size=self.batch_size,
            warmup_steps=0,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            save_steps=999999,
            seed=self.seed,
        )

        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"olmix-{self.ordering_name}-s{k}-lora",
            sbatch_opts=self.sbatch_opts,
        )

        if job_id is None:
            return "STOP"

        if self.dry_run:
            return "ADVANCE"

        self.state.set(
            "stages",
            stage_key,
            "lora_probe",
            {
                "status": "SUBMITTED",
                "job_id": job_id,
                "output_dir": output_dir,
            },
        )
        self.state.set("stages", stage_key, "substage", "lora_probe")
        self.state.set("stages", stage_key, "status", "IN_PROGRESS")
        self.state.save()
        print(f"    lora_probe -> Submitted, stopping to wait")
        return "STOP"

    def _run_lmc(self, k, stage, stage_key, stage_dir, prev_checkpoint):
        """Submit/check LMC eval job."""
        lmc_state = stage.get("lmc", {})
        lmc_status = lmc_state.get("status", "PENDING")

        if lmc_status == "COMPLETED":
            return "ADVANCE"

        lmc_output_dir = str(stage_dir / "lmc" / f"checkpoint-{self.max_steps}")

        if lmc_status == "SUBMITTED":
            job_id = lmc_state["job_id"]
            slurm_status = self.slurm.check_status(job_id)

            if slurm_status == "COMPLETED" and _lmc_sentinel_exists(stage_dir / "lmc"):
                self.state.set("stages", stage_key, "lmc", "status", "COMPLETED")
                self.state.set("stages", stage_key, "substage", "olmix_fit")
                self.state.save()
                print(f"    lmc -> COMPLETED")
                return "ADVANCE"
            elif slurm_status in ("RUNNING", "PENDING"):
                print(f"    lmc -> {slurm_status} (job {job_id})")
                return "STOP"
            else:
                self.state.set("stages", stage_key, "lmc", "status", "FAILED")
                self.state.set("stages", stage_key, "status", "FAILED")
                self.state.save()
                print(f"    lmc -> FAILED")
                return "STOP"

        if lmc_status == "FAILED":
            print(f"    lmc -> FAILED. Use 'retry' to reset.")
            return "STOP"

        # PENDING: submit
        lora_probe_dir = str(stage_dir / "lora_probe")

        args = _lmc_args(
            model_dir=prev_checkpoint,
            model_b=prev_checkpoint,
            lora_checkpoint_dir=lora_probe_dir,
            output_dir=lmc_output_dir,
            batch_size=self.batch_size * 2,
        )

        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_lmc.sh",
            args,
            job_name=f"olmix-{self.ordering_name}-s{k}-lmc",
            sbatch_opts=self.sbatch_opts,
        )

        if job_id is None:
            return "STOP"

        if self.dry_run:
            return "ADVANCE"

        self.state.set(
            "stages",
            stage_key,
            "lmc",
            {
                "status": "SUBMITTED",
                "job_id": job_id,
                "output_dir": lmc_output_dir,
            },
        )
        self.state.set("stages", stage_key, "substage", "lmc")
        self.state.save()
        print(f"    lmc -> Submitted, stopping to wait")
        return "STOP"

    def _run_olmix_fit(self, k, stage, stage_key, stage_dir, ds, prev_ds):
        """Run OnPolicyMix fit locally (no Slurm)."""
        olmix_state = stage.get("olmix", {})
        olmix_status = olmix_state.get("status", "PENDING")

        if olmix_status == "COMPLETED":
            return "ADVANCE"

        if olmix_status == "FAILED":
            print(f"    olmix_fit -> FAILED. Use 'retry' to reset.")
            return "STOP"

        # Fit over the collapsed old mix plus the newly introduced dataset.
        dataset_names = [ds, OLD_MIX_KEY]
        eval_dataset_names = self.ordering[:k]  # all datasets seen through stage k

        if self.dry_run:
            lmc_dir = stage_dir / "lmc"
            print(
                f"    [DRY RUN] olmix fit -> get_merge_rows({lmc_dir}, {dataset_names}, eval={eval_dataset_names})"
            )
            print(f"    [DRY RUN] -> write_csvs + write_olmix_config + run_olmix_fit")
            return "ADVANCE"

        # Run olmix fit locally
        lmc_dir = stage_dir / "lmc"
        olmix_dir = stage_dir / "olmix"

        print(
            f"    olmix_fit -> Running locally (eval datasets: {eval_dataset_names})..."
        )
        ratios_rows, metrics_rows = get_merge_rows(
            lmc_dir, dataset_names, eval_dataset_names
        )

        if not ratios_rows:
            print(f"    olmix_fit -> No merge data found in {lmc_dir}")
            self.state.set("stages", stage_key, "olmix", {"status": "FAILED"})
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            return "STOP"

        ratios_file, metrics_file = write_csvs(
            ratios_rows, metrics_rows, dataset_names, olmix_dir, eval_dataset_names
        )
        config_path = olmix_dir / "fit_config.yaml"
        reference_weights = (
            {ds: 1.0 / k, OLD_MIX_KEY: (k - 1.0) / k}
            if self.prior_mode == "oden"
            else {name: 1.0 / len(dataset_names) for name in dataset_names}
        )
        previous_train_weights = (self.state.get("stages", str(k - 1)) or {}).get(
            "train_weights",
            {prev_ds: 1.0},
        )
        old_source_mixture = normalize_weights(previous_train_weights)
        relative_sizes = (
            {name: 1.0 / k for name in eval_dataset_names}
            if self.prior_mode == "oden"
            else {
                **{
                    name: reference_weights[OLD_MIX_KEY] * old_source_mixture[name]
                    for name in old_source_mixture
                },
                ds: reference_weights[ds],
            }
        )
        source_mixtures = {OLD_MIX_KEY: old_source_mixture}
        write_olmix_config(
            ratios_file,
            metrics_file,
            dataset_names,
            config_path,
            relative_sizes=relative_sizes,
            source_mixtures=source_mixtures,
        )
        opt_weights = run_olmix_fit(config_path, olmix_dir / "olmix_out")

        if opt_weights is None:
            print(f"    olmix_fit -> FAILED")
            self.state.set("stages", stage_key, "olmix", {"status": "FAILED"})
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            return "STOP"

        result_file = olmix_dir / "olmix_best_mix.json"
        olmix_dir.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(
                {
                    "dataset_names": dataset_names,
                    "eval_dataset_names": eval_dataset_names,
                    "optimal_weights": opt_weights,
                    "reference_weights": reference_weights,
                    "relative_sizes": relative_sizes,
                    "source_mixtures": source_mixtures,
                },
                f,
                indent=2,
            )

        alpha_new = opt_weights.get(ds, 0.5)
        print(f"    olmix_fit -> alpha({ds}) = {alpha_new:.4f}")

        self.state.set(
            "stages",
            stage_key,
            "olmix",
            {
                "status": "COMPLETED",
                "optimal_weights": opt_weights,
                "alpha_new": alpha_new,
                "old_key": OLD_MIX_KEY,
                "reference_weights": reference_weights,
                "relative_sizes": relative_sizes,
                "source_mixtures": source_mixtures,
                "previous_train_weights": previous_train_weights,
            },
        )
        self.state.set("stages", stage_key, "substage", "full_train")
        self.state.save()
        return "ADVANCE"

    def _run_full_train(self, k, stage, stage_key, stage_dir, prev_checkpoint, ds):
        """Submit/check full training job with the expanded OnPolicyMix weights."""
        ft_state = stage.get("full_train", {})
        ft_status = ft_state.get("status", "PENDING")

        if ft_status == "COMPLETED":
            self.state.set("stages", stage_key, "status", "COMPLETED")
            self.state.save()
            return "CONTINUE"

        output_dir = str(stage_dir / "full_train")

        if ft_status == "SUBMITTED":
            job_id = ft_state["job_id"]
            slurm_status = self.slurm.check_status(job_id)

            if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                self.state.set("stages", stage_key, "full_train", "status", "COMPLETED")
                self.state.set("stages", stage_key, "status", "COMPLETED")
                self.state.save()
                print(f"    full_train -> COMPLETED")
                return "CONTINUE"
            elif slurm_status in ("RUNNING", "PENDING"):
                print(f"    full_train -> {slurm_status} (job {job_id})")
                return "STOP"
            else:
                self.state.set("stages", stage_key, "full_train", "status", "FAILED")
                self.state.set("stages", stage_key, "status", "FAILED")
                self.state.save()
                print(f"    full_train -> FAILED")
                return "STOP"

        if ft_status == "FAILED":
            print(f"    full_train -> FAILED. Use 'retry' to reset.")
            return "STOP"

        olmix_state = stage.get("olmix", {})
        alpha_new = olmix_state.get("alpha_new", 0.5)

        prev_stage = self.state.get("stages", str(k - 1)) or {}
        prev_weights = prev_stage.get("train_weights", {self.ordering[k - 2]: 1.0})
        compact_weights = olmix_state.get(
            "optimal_weights",
            {ds: alpha_new, OLD_MIX_KEY: 1.0 - alpha_new},
        )
        weights = expand_on_policy_mix_weights(
            compact_weights,
            prev_weights,
        )

        # Build train files and weight values in consistent order
        train_datasets = list(weights.keys())
        train_files = [f"data-mixes/{d}_train_s{k}.txt" for d in train_datasets]
        train_weight_values = [weights[d] for d in train_datasets]

        print(f"    full_train weights: {weights}")

        args = _train_args(
            model_dir=prev_checkpoint,
            train_data_files=train_files,
            output_dir=output_dir,
            train_weights=train_weight_values,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            warmup_steps=0,
            seed=self.seed,
        )

        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"olmix-{self.ordering_name}-s{k}-train",
            sbatch_opts=self.sbatch_opts,
        )

        if job_id is None:
            return "STOP"

        if self.dry_run:
            return "CONTINUE"

        self.state.set(
            "stages",
            stage_key,
            "full_train",
            {
                "status": "SUBMITTED",
                "job_id": job_id,
                "output_dir": output_dir,
            },
        )
        self.state.set("stages", stage_key, "train_weights", weights)
        self.state.save()
        print(f"    full_train -> Submitted, stopping to wait")
        return "STOP"


def naive(ordering_name="all", model_size="150M", dry_run=False, seed=None):
    """Run naive sequential continual learning."""
    orderings = list(ORDERINGS.keys()) if ordering_name == "all" else [ordering_name]
    for name in orderings:
        exp = NaiveExperiment(name, model_size=model_size, dry_run=dry_run, seed=seed)
        exp.run()
        print()


def mix(ordering_name="all", model_size="150M", dry_run=False, seed=None):
    """Run fixed 90/10 mix continual learning."""
    orderings = list(ORDERINGS.keys()) if ordering_name == "all" else [ordering_name]
    for name in orderings:
        exp = MixExperiment(name, model_size=model_size, dry_run=dry_run, seed=seed)
        exp.run()
        print()


def olmix(
    ordering_name="all",
    model_size="150M",
    dry_run=False,
    prior_mode="oden",
    seed=None,
    stop_after=None,
):
    """Legacy wrapper for the continual OnPolicyMix pipeline.

    stop_after: optional substage name ("lora_probe", "lmc", "olmix_fit") to
    halt after. Useful for getting the OnPolicyMix recommendation without
    paying for full_train or any later stages.
    """
    orderings = list(ORDERINGS.keys()) if ordering_name == "all" else [ordering_name]
    for name in orderings:
        exp = OnPolicyMixExperiment(
            name,
            model_size=model_size,
            dry_run=dry_run,
            prior_mode=prior_mode,
            seed=seed,
            stop_after=stop_after,
        )
        exp.run()
        print()


def on_policy_mix(
    ordering_name="all",
    model_size="150M",
    dry_run=False,
    prior_mode="oden",
    seed=None,
    stop_after=None,
):
    """Run continual OnPolicyMix."""
    return olmix(
        ordering_name=ordering_name,
        model_size=model_size,
        dry_run=dry_run,
        prior_mode=prior_mode,
        seed=seed,
        stop_after=stop_after,
    )


def status(condition="all"):
    """Print status of all experiments."""
    conditions = (
        ["naive", "mix", "olmix", "olmix_oden", "lora_merge", "lora_merge_oden"]
        if condition == "all"
        else [condition]
    )

    for cond in conditions:
        root = RUNS_ROOT / f"continual_{cond}"
        if not root.exists():
            print(f"No {cond} experiments found.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {cond.upper()} experiments")
        print(f"{'=' * 60}")

        for ord_name in sorted(ORDERINGS.keys()):
            state_files = list(root.glob(f"{ord_name}/*/state.json"))
            for sf in state_files:
                state = StateFile(sf)
                model_size = state.get("model_size", default="?")
                stages = state.get("stages") or {}

                stage_strs = []
                for k in range(1, 6):
                    sk = stages.get(str(k), {})
                    st = sk.get("status", "PENDING")
                    substage = sk.get("substage", "")
                    abbrev = {
                        "COMPLETED": "OK",
                        "SUBMITTED": "RUN",
                        "FAILED": "FAIL",
                        "IN_PROGRESS": "WIP",
                        "PENDING": "--",
                    }.get(st, st[:4])
                    if substage and st not in ("COMPLETED", "PENDING"):
                        abbrev += f"({substage[:4]})"
                    stage_strs.append(abbrev)

                ordering = state.get("ordering") or ORDERINGS.get(ord_name, [])
                ds_abbrevs = [d[:4] for d in ordering]
                print(f"  {ord_name} ({model_size}) [{' -> '.join(ds_abbrevs)}]")
                print(f"    Stages: {' | '.join(stage_strs)}")


def retry(condition, ordering_name, stage, substage=None, model_size="150M"):
    """Reset a FAILED stage to PENDING so the next run() resubmits."""
    root = RUNS_ROOT / f"continual_{condition}" / ordering_name / model_size
    state = StateFile(root / "state.json")

    stage_key = str(stage)
    stage_data = state.get("stages", stage_key)
    if stage_data is None:
        print(f"Stage {stage} not found in state.")
        return

    if substage:
        state.set("stages", stage_key, substage, "status", "PENDING")
        state.set("stages", stage_key, "status", "IN_PROGRESS")
        state.set("stages", stage_key, "substage", substage)
    else:
        state.set("stages", stage_key, "status", "PENDING")
        # Clear job info so it resubmits
        for field in ("job_id",):
            if field in (state.get("stages", stage_key) or {}):
                del state._data["stages"][stage_key][field]
        # Also reset any substage statuses back to PENDING
        for key, val in (state.get("stages", stage_key) or {}).items():
            if isinstance(val, dict) and "status" in val:
                state.set("stages", stage_key, key, "status", "PENDING")
                for f in ("job_id",):
                    if f in val:
                        del state._data["stages"][stage_key][key][f]
    state.save()
    print(
        f"Reset {condition}/{ordering_name}/stage_{stage}"
        + (f"/{substage}" if substage else "")
        + " to PENDING"
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "naive": naive,
            "mix": mix,
            "olmix": olmix,
            "on_policy_mix": on_policy_mix,
            "status": status,
            "retry": retry,
        }
    )
