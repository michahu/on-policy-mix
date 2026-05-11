"""Pretraining pipelines: OnPolicyMix, MergeMix, and swarm baselines.

Usage:
    python -m pipeline.pretrain filo [--model_size 150M] [--max_steps 50000]
    python -m pipeline.pretrain mergemix [--model_size 150M] [--max_steps 50000]
    python -m pipeline.pretrain swarm [--model_size 150M] [--max_steps 50000]
    python -m pipeline.pretrain status [--algorithm all]
    python -m pipeline.pretrain retry --algorithm pretrain_filo_mix --model_size 150M --max_steps 50000 --substage prefix_train
"""

import json
from copy import deepcopy
from pathlib import Path

import fire
import numpy as np

from pipeline.continual import (
    RUNS_ROOT,
    SCRIPTS_DIR,
    SlurmHelper,
    StateFile,
    _sentinel_exists,
    normalize_weights,
)
from pipeline.olmix import (
    get_final_eval_losses,
    run_olmix_fit,
    write_csvs,
    write_olmix_config,
)
from src.utils import load_model_with_optional_lora

DEFAULT_DATASETS = [
    "arxiv",
    "c4",
    "open_web_math",
    "reddit",
    "stackexchange",
]

MODEL_CONFIGS = {
    "150M": {
        "base_model": "allenai/DataDecide-c4-150M",
        "learning_rate": "5e-4",
        "batch_size": 32,
        "max_steps": 50000,
        "warmup_steps": 1000,
        "sbatch_opts": ["--constraint=a100|h100|l40s"],
        "lora_sbatch_opts": ["--constraint=a100|h100|l40s"],
    },
    "300M": {
        "base_model": "allenai/DataDecide-c4-300M",
        "learning_rate": "4e-4",
        "batch_size": 64,
        "max_steps": 50000,
        "warmup_steps": 5000,
        "sbatch_opts": ["--constraint=h200"],
        "lora_sbatch_opts": ["--constraint=a100|h100"],
        "lora_batch_size": 32,
        "lora_gradient_accumulation_steps": 2,
    },
    "530M": {
        "base_model": "allenai/DataDecide-c4-530M",
        "learning_rate": "3e-4",
        "batch_size": 64,
        "max_steps": 80000,
        "warmup_steps": 8000,
        "eval_steps": 5000,
        "sbatch_opts": ["--constraint=h200"],
        "lora_sbatch_opts": ["--constraint=a100|h100"],
        "lora_batch_size": 32,
        "lora_gradient_accumulation_steps": 2,
    },
}


PROXY_MODEL_CONFIGS = {
    "20M": {
        "base_model": "allenai/DataDecide-c4-20M",
        "learning_rate": "1e-3",
        "batch_size": 32,
        "max_steps": 10000,
        "warmup_steps": 1000,
        "sbatch_opts": ["--constraint=a100|h100|l40s"],
    },
}


def _default_train_files(datasets):
    return {ds: f"data-mixes/{ds}_train.txt" for ds in datasets}


def _default_eval_files(datasets):
    return {ds: f"data-mixes/{ds}_eval.txt" for ds in datasets}


def _scale_model_config(config, max_steps=None):
    scaled = dict(config)
    if max_steps is not None:
        default_steps = config["max_steps"]
        warmup_ratio = config["warmup_steps"] / default_steps
        scaled["max_steps"] = max_steps
        scaled["warmup_steps"] = int(round(max_steps * warmup_ratio))
    return scaled


def _resolve_model_config(model_size, max_steps=None, use_proxy_defaults=False):
    source = PROXY_MODEL_CONFIGS if use_proxy_defaults else MODEL_CONFIGS
    if model_size not in source:
        valid = ", ".join(sorted(source))
        raise ValueError(f"Unknown model size {model_size}. Valid: {valid}")
    return _scale_model_config(source[model_size], max_steps=max_steps)


def _normalize_weights(weights):
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {k: v / total for k, v in weights.items()}


def _probe_weights_with_old_mix(new_weight, target_dataset, old_weights):
    """Build a probe mix with ``new_weight`` on the target and the rest on old mix.

    Unlike the continual helper, this supports overlap between ``target_dataset``
    and ``old_weights`` so it works for both rebalance (target already in old
    mix) and expansion (target is brand new).
    """
    prior_sum = sum(old_weights.values())
    if prior_sum <= 0:
        raise ValueError("old_weights must sum to a positive value")
    probe_weights = {
        d: (1.0 - new_weight) * w / prior_sum for d, w in old_weights.items()
    }
    probe_weights[target_dataset] = probe_weights.get(target_dataset, 0.0) + new_weight
    return probe_weights


def _sample_proxy_weights(datasets, proxy_count, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(proxy_count):
        vals = rng.dirichlet(np.ones(len(datasets)))
        weights = {ds: float(v) for ds, v in zip(datasets, vals)}
        samples.append(
            {"merge_id": f"proxy_{i:03d}", "weights": _normalize_weights(weights)}
        )
    return samples


def _build_train_args(
    model_dir,
    train_data_files,
    output_dir,
    eval_data_files,
    train_weights=None,
    max_steps=10000,
    learning_rate="5e-4",
    batch_size=32,
    warmup_steps=0,
    lr_scheduler_type="wsd",
    model_init_mode="pretrained",
    use_lora=False,
    lora_r=None,
    lora_alpha=None,
    lora_target_modules=None,
    lora_exclude_modules=None,
    save_steps=None,
    eval_steps=None,
    seed=None,
    gradient_accumulation_steps=None,
):
    args = [
        "--model_dir",
        str(model_dir),
        "--model_init_mode",
        model_init_mode,
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
    args.append("--train_data_file")
    args.extend(str(f) for f in train_data_files)
    args.append("--eval_data_file")
    args.extend(str(f) for f in eval_data_files)
    if train_weights:
        args.append("--train_weights")
        args.extend(str(w) for w in train_weights)
    if use_lora:
        args.extend(
            ["--use_lora", "--lora_r", str(lora_r), "--lora_alpha", str(lora_alpha)]
        )
        if lora_target_modules:
            args.append("--lora_target_modules")
            args.extend(str(m) for m in lora_target_modules)
        if lora_exclude_modules:
            args.append("--lora_exclude_modules")
            args.extend(str(m) for m in lora_exclude_modules)
    if save_steps is not None:
        args.extend(["--save_steps", str(save_steps)])
    if eval_steps is not None:
        args.extend(["--eval_steps", str(eval_steps)])
    if seed is not None:
        args.extend(["--seed", str(seed)])
    if gradient_accumulation_steps is not None:
        args.extend(["--gradient_accumulation_steps", str(gradient_accumulation_steps)])
    args.append("--use_wandb")
    return args


def _proxy_eval_sentinel_exists(output_dir):
    return (Path(output_dir) / "linear_connectivity_results.json").exists()


def _round_steps(total_steps, fraction, minimum=1):
    return max(minimum, int(round(total_steps * fraction)))


def _save_proxy_model(base_model_dir, lora_dirs, weights, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = load_model_with_optional_lora(base_model_dir, merge_lora=False)
    base_state = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    merged_state = {k: v.float().clone() for k, v in base_state.items()}

    for dataset, lora_dir in lora_dirs.items():
        alpha = weights.get(dataset, 0.0)
        if alpha == 0.0:
            continue
        lora_model = load_model_with_optional_lora(
            base_model_dir,
            lora_checkpoint_dir=lora_dir,
            merge_lora=True,
        )
        lora_state = lora_model.state_dict()
        for key, merged_tensor in merged_state.items():
            merged_tensor.add_(
                alpha
                * (lora_state[key].detach().cpu().float() - base_state[key].float())
            )
        del lora_model

    final_state = {
        key: tensor.to(dtype=base_state[key].dtype)
        for key, tensor in merged_state.items()
    }
    merged_model = deepcopy(base_model)
    merged_model.load_state_dict(final_state)
    merged_model.save_pretrained(str(output_dir))


def _save_multi_merge_model(model_dirs, weights, output_dir):
    """Merge K full models with given weights: merged = sum(w_i * model_i).

    Args:
        model_dirs: dict mapping dataset name -> model directory path
        weights: dict mapping dataset name -> merge weight (should sum to 1)
        output_dir: path to save the merged model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(model_dirs.keys())
    first_ds = datasets[0]
    first_model = load_model_with_optional_lora(model_dirs[first_ds], merge_lora=False)
    first_state = {
        k: v.detach().cpu().float() for k, v in first_model.state_dict().items()
    }
    w0 = weights.get(first_ds, 0.0)

    merged_state = {k: w0 * v for k, v in first_state.items()}

    for dataset in datasets[1:]:
        w = weights.get(dataset, 0.0)
        if w == 0.0:
            continue
        model = load_model_with_optional_lora(model_dirs[dataset], merge_lora=False)
        for key, merged_tensor in merged_state.items():
            merged_tensor.add_(w * model.state_dict()[key].detach().cpu().float())
        del model

    final_state = {
        key: tensor.to(dtype=first_state[key].dtype)
        for key, tensor in merged_state.items()
    }
    merged_model = deepcopy(first_model)
    merged_model.load_state_dict(final_state)
    merged_model.save_pretrained(str(output_dir))


def _get_proxy_eval_losses(eval_dir, eval_dataset_names):
    results_file = Path(eval_dir) / "linear_connectivity_results.json"
    if not results_file.exists():
        return None
    with open(results_file) as f:
        results = json.load(f)
    if not results:
        return None
    entry = next(iter(results.values()))
    per_dataset = entry.get("per_dataset_results", {})
    losses = {}
    for ds in eval_dataset_names:
        ds_results = per_dataset.get(ds)
        if ds_results is None or "eval_loss" not in ds_results:
            return None
        losses[ds] = ds_results["eval_loss"]
    return losses


def get_proxy_rows(proxy_eval_dir, proxy_specs, dataset_names, eval_dataset_names=None):
    if eval_dataset_names is None:
        eval_dataset_names = dataset_names
    proxy_eval_dir = Path(proxy_eval_dir)
    if not proxy_eval_dir.exists():
        return [], []

    specs_by_id = {
        spec["merge_id"]: spec.get("ratios", spec["weights"]) for spec in proxy_specs
    }
    ratios_rows = []
    metrics_rows = []

    for merge_dir in sorted(d for d in proxy_eval_dir.iterdir() if d.is_dir()):
        merge_id = merge_dir.name
        if merge_id not in specs_by_id:
            continue
        losses = _get_proxy_eval_losses(merge_dir, eval_dataset_names)
        if losses is None:
            continue
        weights = specs_by_id[merge_id]
        ratios_rows.append(
            {"run": merge_id, **{ds: weights[ds] for ds in dataset_names}}
        )
        metrics_rows.append(
            {
                "run": merge_id,
                **{f"eval_{ds}_loss": losses[ds] for ds in eval_dataset_names},
            }
        )
    return ratios_rows, metrics_rows


def get_training_rows(
    proxy_runs_dir, proxy_specs, dataset_names, eval_dataset_names=None
):
    if eval_dataset_names is None:
        eval_dataset_names = dataset_names
    proxy_runs_dir = Path(proxy_runs_dir)
    if not proxy_runs_dir.exists():
        return [], []

    specs_by_id = {spec["merge_id"]: spec["weights"] for spec in proxy_specs}
    ratios_rows = []
    metrics_rows = []

    for run_dir in sorted(d for d in proxy_runs_dir.iterdir() if d.is_dir()):
        run_id = run_dir.name
        if run_id not in specs_by_id:
            continue
        losses = get_final_eval_losses(run_dir, eval_dataset_names)
        if not losses:
            continue
        weights = specs_by_id[run_id]
        ratios_rows.append({"run": run_id, **{ds: weights[ds] for ds in dataset_names}})
        metrics_rows.append(
            {
                "run": run_id,
                **{
                    f"eval_{ds}_loss": losses[ds]
                    for ds in eval_dataset_names
                    if ds in losses
                },
            }
        )
    return ratios_rows, metrics_rows


class ExperimentBase:
    algorithm_name = None

    def __init__(
        self,
        model_size="150M",
        max_steps=None,
        datasets=None,
        train_files=None,
        eval_files=None,
        proxy_count=20,
        proxy_fraction=0.2,
        seed=None,
        dry_run=False,
    ):
        self.model_size = model_size
        self.datasets = list(datasets) if datasets else list(DEFAULT_DATASETS)
        self.train_files = train_files or _default_train_files(self.datasets)
        self.eval_files = eval_files or _default_eval_files(self.datasets)
        self.proxy_count = proxy_count
        self.proxy_fraction = proxy_fraction
        self.seed = seed
        self.dry_run = dry_run
        self.slurm = SlurmHelper(dry_run=dry_run)

        cfg = _resolve_model_config(model_size, max_steps=max_steps)
        self.model_id = cfg["base_model"]
        self.learning_rate = cfg["learning_rate"]
        self.batch_size = cfg["batch_size"]
        self.max_steps = cfg["max_steps"]
        self.warmup_steps = cfg["warmup_steps"]
        self.sbatch_opts = cfg["sbatch_opts"]
        self.lora_sbatch_opts = cfg.get("lora_sbatch_opts", self.sbatch_opts)
        self.lora_batch_size = cfg.get("lora_batch_size", self.batch_size)
        self.lora_gradient_accumulation_steps = cfg.get(
            "lora_gradient_accumulation_steps", None
        )
        self.eval_steps = cfg.get("eval_steps")
        seed_suffix = f"seed_{self.seed}" if self.seed is not None else "seed_none"
        self.run_dir = (
            RUNS_ROOT
            / self.algorithm_name
            / model_size
            / f"{self.max_steps}_steps"
            / seed_suffix
        )
        self.state = StateFile(self.run_dir / "state.json")

    @property
    def proxy_steps(self):
        return _round_steps(self.max_steps, self.proxy_fraction)

    def _train_file_list(self, datasets=None):
        if datasets is None:
            datasets = self.datasets
        return [self.train_files[ds] for ds in datasets]

    def _eval_file_list(self):
        return [self.eval_files[ds] for ds in self.datasets]

    @property
    def proxy_specs(self):
        return self.state.get("proxy_specs", default=[])

    def _run_batch_proxy_eval(self, job_name_prefix):
        """Evaluate every proxy in a single batched Slurm job.

        Subclasses must implement ``_proxy_model_dir(merge_id)`` and
        ``_proxy_eval_dir(merge_id)``. Per-spec status is tracked in
        ``state["proxy_eval"]`` (matching the old format so ``get_proxy_rows``
        keeps working). The batch job's id is tracked in
        ``state["proxy_eval_batch"]``.
        """
        # Recompute per-spec completion from disk first — sentinels are the
        # source of truth, the state dict just mirrors them.
        for spec in self.proxy_specs:
            merge_id = spec["merge_id"]
            output_dir = str(self._proxy_eval_dir(merge_id))
            if _proxy_eval_sentinel_exists(output_dir):
                cur = self.state.get("proxy_eval", merge_id, default={})
                if cur.get("status") != "COMPLETED":
                    self.state.set(
                        "proxy_eval",
                        merge_id,
                        {"status": "COMPLETED", "output_dir": output_dir},
                    )
        self.state.save()

        pending_specs = [
            spec
            for spec in self.proxy_specs
            if not _proxy_eval_sentinel_exists(
                str(self._proxy_eval_dir(spec["merge_id"]))
            )
        ]

        if not pending_specs:
            self.state.set("substage", "olmix_fit")
            self.state.save()
            for spec in self.proxy_specs:
                print(f"  proxy_eval[{spec['merge_id']}] -> COMPLETED")
            return "ADVANCE"

        batch = self.state.get("proxy_eval_batch", default={})
        batch_status = batch.get("status", "PENDING")

        if batch_status == "SUBMITTED":
            job_id = batch["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status in ("RUNNING", "PENDING"):
                print(
                    f"  proxy_eval[batch] -> {slurm_status} (job {job_id}, "
                    f"{len(self.proxy_specs) - len(pending_specs)}/{len(self.proxy_specs)} done)"
                )
                return "STOP"
            # Job has finished (COMPLETED or FAILED). Mark per-spec status from
            # sentinels and clear the batch slot so a follow-up run can resubmit
            # any stragglers.
            n_failed = 0
            for spec in pending_specs:
                merge_id = spec["merge_id"]
                self.state.set("proxy_eval", merge_id, "status", "FAILED")
                n_failed += 1
                print(
                    f"  proxy_eval[{merge_id}] -> FAILED (no sentinel after job {job_id})"
                )
            self.state.set("proxy_eval_batch", {"status": "FAILED", "job_id": job_id})
            self.state.set("status", "FAILED")
            self.state.save()
            print(
                f"  proxy_eval[batch] -> job {job_id} finished but {n_failed} "
                f"specs are missing sentinels. Use retry to reset."
            )
            return "STOP"

        if batch_status == "FAILED":
            print("  proxy_eval[batch] -> FAILED. Use retry to reset.")
            return "STOP"

        # PENDING — submit a single batch job covering all pending specs.
        manifest_dir = self.run_dir / "proxy_eval"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "batch_manifest.tsv"
        with open(manifest_path, "w") as f:
            for spec in pending_specs:
                merge_id = spec["merge_id"]
                model_dir = str(self._proxy_model_dir(merge_id))
                output_dir = str(self._proxy_eval_dir(merge_id))
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                f.write(f"{model_dir}\t{output_dir}\n")

        eval_args = [
            str(manifest_path),
            "--batch_size",
            str(self.batch_size * 2),
            "--eval_data_file",
            *[str(p) for p in self._eval_file_list()],
        ]
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_batch_eval.sh",
            eval_args,
            job_name=f"{job_name_prefix}-batch-{self.model_size}-{self.max_steps}",
            sbatch_opts=self.sbatch_opts,
        )
        if job_id is None:
            return "STOP"
        if self.dry_run:
            print(
                f"  [DRY RUN] proxy_eval -> would launch 1 batch job for "
                f"{len(pending_specs)} proxies"
            )
            return "ADVANCE"

        for spec in pending_specs:
            self.state.set(
                "proxy_eval",
                spec["merge_id"],
                {
                    "status": "SUBMITTED",
                    "job_id": job_id,
                    "output_dir": str(self._proxy_eval_dir(spec["merge_id"])),
                },
            )
        self.state.set(
            "proxy_eval_batch",
            {
                "status": "SUBMITTED",
                "job_id": job_id,
                "manifest": str(manifest_path),
                "n_specs": len(pending_specs),
            },
        )
        self.state.set("substage", "proxy_eval")
        self.state.set("status", "IN_PROGRESS")
        self.state.save()
        print(
            f"  proxy_eval[batch] -> Submitted job {job_id} for "
            f"{len(pending_specs)} proxies"
        )
        return "STOP"


class OnPolicyMixExperiment(ExperimentBase):
    """Unified OnPolicyMix pipeline.

    Collapses an "old" model (either an ERM prefix trained here, or a pretrain
    endpoint passed in via ``base_model_dir``) into a single coefficient, trains
    one LoRA probe per ``new_dataset``, does K Dirichlet-sampled proxy merges
    over the ``(n_new + 1)``-simplex ``{old, d_1, …, d_n_new}``, fits optimal
    weights via olmix with an ODEN prior, and finally continues training from
    the base model using the expanded mix

        old_centered = 0.8 * old_weights + 0.2 * uniform_old
        final[d] = w_old * old_centered[d] + opt_weights.get(d, 0)

    Pretrain-from-scratch mode (default): ``base_model_dir`` is None, the class
    runs ``prefix_train`` on all ``datasets`` with ``old_weights`` (defaults to
    uniform 1/K) to build the ERM prefix, then ``new_datasets`` defaults to the
    same K datasets so the fit rebalances them.

    Continual-from-pretrain mode: instantiate via ``from_pretrain_run``, which
    loads ``datasets``/``old_weights``/``base_model_dir`` from an existing
    ``runs/pretrain_*`` state.json and lets you pass brand-new ``new_datasets``.
    """

    algorithm_name = "pretrain_filo_mix"
    OLD_KEY = "old"

    def __init__(
        self,
        model_size="150M",
        max_steps=None,
        datasets=None,
        train_files=None,
        eval_files=None,
        old_weights=None,
        base_model_dir=None,
        new_datasets=None,
        new_train_files=None,
        new_eval_files=None,
        proxy_count=20,
        prefix_fraction=0.2,
        lora_steps_fraction=0.2,
        seed=None,
        dry_run=False,
        run_dir=None,
        stage_id=None,
        stop_after=None,
    ):
        self.prefix_fraction = prefix_fraction
        self.lora_steps_fraction = lora_steps_fraction
        self._base_model_dir_override = base_model_dir
        self.continual_mode = base_model_dir is not None
        self.stage_id = stage_id
        self.stop_after = stop_after
        super().__init__(
            model_size=model_size,
            max_steps=max_steps,
            datasets=datasets,
            train_files=train_files,
            eval_files=eval_files,
            proxy_count=proxy_count,
            proxy_fraction=prefix_fraction,
            seed=seed,
            dry_run=dry_run,
        )

        # Old weights default: uniform 1/K over self.datasets
        if old_weights is None:
            u = 1.0 / len(self.datasets)
            self.old_weights = {d: u for d in self.datasets}
        else:
            self.old_weights = {d: float(old_weights[d]) for d in self.datasets}

        # New datasets default: same as old (pure rebalance, pretrain mode)
        if new_datasets is None:
            self.new_datasets = list(self.datasets)
            self.new_train_files = dict(self.train_files)
            self.new_eval_files = dict(self.eval_files)
        else:
            self.new_datasets = list(new_datasets)
            self.new_train_files = new_train_files or {
                d: f"data-mixes/{d}.txt" for d in self.new_datasets
            }
            self.new_eval_files = new_eval_files or {
                d: f"data-mixes/{d}_eval.txt" for d in self.new_datasets
            }

        # Optional run_dir override (used by from_pretrain_run for continual mode)
        if run_dir is not None:
            self.run_dir = Path(run_dir)
            self.state = StateFile(self.run_dir / "state.json")

        if not self.state.data:
            init_substage = "lora_probes" if self.continual_mode else "prefix_train"
            self.state._data = {
                "condition": self.algorithm_name,
                "continual_mode": self.continual_mode,
                "base_model_dir": self._base_model_dir_override,
                "model_size": model_size,
                "model_id": self.model_id,
                "max_steps": self.max_steps,
                "datasets": self.datasets,
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "old_weights": self.old_weights,
                "new_datasets": self.new_datasets,
                "new_train_files": self.new_train_files,
                "new_eval_files": self.new_eval_files,
                "proxy_count": proxy_count,
                "prefix_fraction": prefix_fraction,
                "lora_steps_fraction": lora_steps_fraction,
                "seed": seed,
                "proxy_specs": _sample_proxy_weights(
                    [self.OLD_KEY] + self.new_datasets,
                    proxy_count,
                    0 if seed is None else seed,
                ),
                "substage": init_substage,
                "status": "PENDING",
            }
            self.state.save()

    @classmethod
    def from_pretrain_run(
        cls,
        pretrain_run_dir,
        new_datasets=None,
        new_train_files=None,
        new_eval_files=None,
        **kwargs,
    ):
        """Build a continual-pretrain OnPolicyMixExperiment from a finished pretrain run.

        Reads ``datasets``/``train_files``/``eval_files`` from the pretrain
        run's ``state.json`` and uses its ``olmix.optimal_weights`` (if present,
        else uniform) as ``old_weights``. The pretrain endpoint's
        ``final_train`` directory becomes the ``base_model_dir``.
        """
        p = Path(pretrain_run_dir)
        pstate = json.loads((p / "state.json").read_text())
        base_model_dir = str(p / "final_train")
        datasets = pstate["datasets"]
        train_files = pstate["train_files"]
        eval_files = pstate["eval_files"]
        opt = (pstate.get("olmix") or {}).get("optimal_weights")
        old_weights = {d: opt[d] for d in datasets} if opt else None
        if new_datasets is None:
            new_datasets = ["tulu_flan_v0", "tulu_flan_v2"]
        if new_train_files is None:
            new_train_files = {d: f"data-mixes/{d}.txt" for d in new_datasets}
        if new_eval_files is None:
            new_eval_files = {d: f"data-mixes/{d}_eval.txt" for d in new_datasets}
        tag = p.relative_to(RUNS_ROOT).as_posix().replace("/", "__")
        run_dir = RUNS_ROOT / "continual_pretrain_filo_mix" / tag
        return cls(
            datasets=datasets,
            train_files=train_files,
            eval_files=eval_files,
            old_weights=old_weights,
            base_model_dir=base_model_dir,
            new_datasets=new_datasets,
            new_train_files=new_train_files,
            new_eval_files=new_eval_files,
            run_dir=run_dir,
            **kwargs,
        )

    @property
    def prefix_steps(self):
        return _round_steps(self.max_steps, self.prefix_fraction)

    @property
    def final_steps(self):
        if self.continual_mode:
            return self.max_steps
        return max(1, self.max_steps - self.prefix_steps)

    @property
    def lora_probe_steps(self):
        return _round_steps(self.max_steps, self.lora_steps_fraction)

    @property
    def prefix_dir(self):
        return self.run_dir / "prefix_train"

    @property
    def base_model_dir(self):
        """The model LoRA probes and final_train start from.

        In pretrain mode: the ERM prefix directory (populated after
        ``prefix_train``). In continual mode: the ``base_model_dir`` passed
        into ``__init__`` (e.g. a pretrain endpoint).
        """
        if self._base_model_dir_override is not None:
            return self._base_model_dir_override
        return str(self.prefix_dir)

    def _lora_probe_dir(self, dataset):
        return self.run_dir / "lora_probes" / dataset

    def _proxy_model_dir(self, merge_id):
        return self.run_dir / "proxy_merges" / merge_id

    def _proxy_eval_dir(self, merge_id):
        return self.run_dir / "proxy_eval" / merge_id

    def _union_datasets(self):
        """Old datasets followed by any new datasets not already in old (order-preserving)."""
        seen = set()
        out = []
        for d in (*self.datasets, *self.new_datasets):
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def _union_train_file(self, d):
        return self.train_files.get(d) or self.new_train_files[d]

    def _union_eval_file(self, d):
        return self.eval_files.get(d) or self.new_eval_files[d]

    def _eval_file_list(self):
        # Override ExperimentBase._eval_file_list: eval on union of old + new.
        return [self._union_eval_file(d) for d in self._union_datasets()]

    def compute_expanded_weights(self, opt_weights=None):
        """Expand the (n_new+1)-way olmix result into the full mix over union.

            final[d] = w_old * old_weights.get(d, 0) + opt_weights.get(d, 0)

        Returns a dict keyed by ``_union_datasets()``. Returns None if olmix
        hasn't produced optimal_weights yet (and no override was passed in).
        """
        if opt_weights is None:
            opt_weights = self.state.get("olmix", "optimal_weights", default=None)
        if opt_weights is None:
            return None
        w_old = opt_weights[self.OLD_KEY]
        old_weights = normalize_weights(self.old_weights)
        return {
            d: w_old * old_weights.get(d, 0.0) + opt_weights.get(d, 0.0)
            for d in self._union_datasets()
        }

    def run(self):
        mode = "continual" if self.continual_mode else "pretrain"
        stage_tag = f" {self.stage_id}" if self.stage_id else ""
        print(
            f"=== OnPolicyMix{stage_tag} [{mode}] ({self.model_size}, {self.max_steps} steps) ==="
        )
        default_substage = "lora_probes" if self.continual_mode else "prefix_train"
        substage = self.state.get("substage", default=default_substage)

        if substage == "prefix_train":
            if self.continual_mode:
                substage = "lora_probes"
            else:
                result = self._run_prefix_train()
                if result != "ADVANCE":
                    return result
                if self.stop_after == "prefix_train":
                    return "ADVANCE"
                substage = "lora_probes"
        if substage == "lora_probes":
            result = self._run_lora_probes()
            if result != "ADVANCE":
                return result
            if self.stop_after == "lora_probes":
                return "ADVANCE"
            substage = "proxy_merge"
        if substage == "proxy_merge":
            result = self._run_proxy_merge()
            if result != "ADVANCE":
                return result
            substage = "proxy_eval"
        if substage == "proxy_eval":
            result = self._run_proxy_eval()
            if result != "ADVANCE":
                return result
            substage = "olmix_fit"
        if substage == "olmix_fit":
            result = self._run_olmix_fit()
            if result != "ADVANCE":
                return result
            substage = "final_train"
        if substage == "final_train":
            return self._run_final_train()
        return "ADVANCE"

    def _run_prefix_train(self):
        stage = self.state.get("prefix_train", default={})
        status = stage.get("status", "PENDING")
        output_dir = str(self.prefix_dir)

        if status == "COMPLETED":
            return "ADVANCE"
        if status == "SUBMITTED":
            job_id = stage["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                self.state.set("prefix_train", "status", "COMPLETED")
                self.state.set("substage", "lora_probes")
                self.state.set("status", "IN_PROGRESS")
                self.state.save()
                print("  prefix_train -> COMPLETED")
                return "ADVANCE"
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"  prefix_train -> {slurm_status} (job {job_id})")
                return "STOP"
            self.state.set("prefix_train", "status", "FAILED")
            self.state.set("status", "FAILED")
            self.state.save()
            print("  prefix_train -> FAILED")
            return "STOP"
        if status == "FAILED":
            print("  prefix_train -> FAILED. Use retry to reset.")
            return "STOP"

        weights = [self.old_weights[d] for d in self.datasets]
        args = _build_train_args(
            model_dir=self.model_id,
            model_init_mode="config",
            train_data_files=self._train_file_list(),
            output_dir=output_dir,
            eval_data_files=self._eval_file_list(),
            train_weights=weights,
            max_steps=self.prefix_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            warmup_steps=int(round(self.warmup_steps * self.prefix_fraction)),
            seed=self.seed,
        )
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"filo-prefix-{self.model_size}-{self.max_steps}",
            sbatch_opts=self.sbatch_opts,
        )
        if job_id is None:
            return "STOP"
        if self.dry_run:
            return "ADVANCE"

        self.state.set(
            "prefix_train",
            {
                "status": "SUBMITTED",
                "job_id": job_id,
                "output_dir": output_dir,
            },
        )
        self.state.set("substage", "prefix_train")
        self.state.set("status", "IN_PROGRESS")
        self.state.save()
        print("  prefix_train -> Submitted")
        return "STOP"

    def _run_lora_probes(self):
        probes = self.state.get("lora_probes", default={})
        any_running = False

        for dataset in self.new_datasets:
            probe = probes.get(dataset, {})
            status = probe.get("status", "PENDING")
            output_dir = str(self._lora_probe_dir(dataset))

            if status == "COMPLETED":
                continue

            if status == "SUBMITTED":
                job_id = probe["job_id"]
                slurm_status = self.slurm.check_status(job_id)
                if slurm_status == "COMPLETED" and _sentinel_exists(
                    output_dir, sentinel="adapter_model.safetensors"
                ):
                    self.state.set("lora_probes", dataset, "status", "COMPLETED")
                    self.state.save()
                    print(f"  lora_probe[{dataset}] -> COMPLETED")
                    continue
                if slurm_status in ("RUNNING", "PENDING"):
                    any_running = True
                    print(f"  lora_probe[{dataset}] -> {slurm_status} (job {job_id})")
                    continue
                self.state.set("lora_probes", dataset, "status", "FAILED")
                self.state.set("status", "FAILED")
                self.state.save()
                print(f"  lora_probe[{dataset}] -> FAILED")
                return "STOP"

            if status == "FAILED":
                print(f"  lora_probe[{dataset}] -> FAILED. Use retry to reset.")
                return "STOP"

            probe_weights = _probe_weights_with_old_mix(0.9, dataset, self.old_weights)
            train_datasets = list(probe_weights.keys())
            train_files = [self._union_train_file(d) for d in train_datasets]
            train_weight_values = [probe_weights[d] for d in train_datasets]

            args = _build_train_args(
                model_dir=self.base_model_dir,
                model_init_mode="pretrained",
                train_data_files=train_files,
                output_dir=output_dir,
                eval_data_files=self._eval_file_list(),
                train_weights=train_weight_values,
                max_steps=self.lora_probe_steps,
                learning_rate=str(float(self.learning_rate) * 2),
                batch_size=self.lora_batch_size,
                gradient_accumulation_steps=self.lora_gradient_accumulation_steps,
                warmup_steps=0,
                use_lora=True,
                lora_r=16,
                lora_alpha=32,
                lora_exclude_modules=["transformer.ff_out"],
                save_steps=999999,
                seed=self.seed,
            )
            job_id = self.slurm.submit(
                SCRIPTS_DIR / "slurm_launch_train.sh",
                args,
                job_name=f"filo-lora-{dataset}-{self.model_size}-{self.max_steps}",
                sbatch_opts=self.lora_sbatch_opts,
            )
            if job_id is None:
                return "STOP"
            if self.dry_run:
                continue
            self.state.set(
                "lora_probes",
                dataset,
                {
                    "status": "SUBMITTED",
                    "job_id": job_id,
                    "output_dir": output_dir,
                },
            )
            self.state.set("substage", "lora_probes")
            self.state.set("status", "IN_PROGRESS")
            self.state.save()
            print(f"  lora_probe[{dataset}] -> Submitted")

        if self.dry_run:
            print(
                f"  [DRY RUN] lora_probes -> would launch {len(self.new_datasets)} LoRA jobs"
            )
            return "ADVANCE"
        probes = self.state.get("lora_probes", default={})
        all_done = all(
            probes.get(ds, {}).get("status") == "COMPLETED" for ds in self.new_datasets
        )
        if all_done:
            self.state.set("substage", "proxy_merge")
            self.state.save()
            return "ADVANCE"
        return "STOP"

    def _run_proxy_merge(self):
        proxy_state = self.state.get("proxy_merge", default={})
        status = proxy_state.get("status", "PENDING")
        if status == "COMPLETED":
            return "ADVANCE"
        if status == "FAILED":
            print("  proxy_merge -> FAILED. Use retry to reset.")
            return "STOP"
        if self.dry_run:
            print(
                f"  [DRY RUN] proxy_merge -> would build {len(self.proxy_specs)} merged proxies"
            )
            return "ADVANCE"

        lora_dirs = {}
        for dataset in self.new_datasets:
            probe = self.state.get("lora_probes", dataset)
            if probe is None or probe.get("status") != "COMPLETED":
                print(f"  proxy_merge -> missing completed LoRA probe for {dataset}")
                self.state.set("proxy_merge", {"status": "FAILED"})
                self.state.set("status", "FAILED")
                self.state.save()
                return "STOP"
            lora_dirs[dataset] = probe["output_dir"]

        try:
            for spec in self.proxy_specs:
                out = self._proxy_model_dir(spec["merge_id"])
                if _sentinel_exists(out):
                    continue
                # spec["weights"] contains an OLD_KEY entry that _save_proxy_model
                # silently skips (it's not in lora_dirs), giving:
                #   merged = base + Σ_{d in new_datasets} w_d · ΔLoRA_d
                _save_proxy_model(self.base_model_dir, lora_dirs, spec["weights"], out)
        except Exception as exc:
            self.state.set("proxy_merge", {"status": "FAILED", "error": str(exc)})
            self.state.set("status", "FAILED")
            self.state.save()
            print(f"  proxy_merge -> FAILED: {exc}")
            return "STOP"

        self.state.set("proxy_merge", {"status": "COMPLETED"})
        self.state.set("substage", "proxy_eval")
        self.state.save()
        print("  proxy_merge -> COMPLETED")
        return "ADVANCE"

    def _run_proxy_eval(self):
        return self._run_batch_proxy_eval(job_name_prefix="filo-eval")

    def _run_olmix_fit(self):
        olmix_state = self.state.get("olmix", default={})
        status = olmix_state.get("status", "PENDING")
        if status == "COMPLETED":
            return "ADVANCE"
        if status == "FAILED":
            print("  olmix_fit -> FAILED. Use retry to reset.")
            return "STOP"
        if self.dry_run:
            print("  [DRY RUN] olmix_fit -> would fit from proxy eval rows")
            return "ADVANCE"

        fit_dataset_names = [self.OLD_KEY] + self.new_datasets
        eval_dataset_names = self._union_datasets()
        ratios_rows, metrics_rows = get_proxy_rows(
            self.run_dir / "proxy_eval",
            self.proxy_specs,
            fit_dataset_names,
            eval_dataset_names,
        )
        if not ratios_rows:
            self.state.set("olmix", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  olmix_fit -> No valid proxy rows")
            return "STOP"

        olmix_dir = self.run_dir / "olmix"
        ratios_file, metrics_file = write_csvs(
            ratios_rows, metrics_rows, fit_dataset_names, olmix_dir, eval_dataset_names
        )
        config_path = olmix_dir / "fit_config.yaml"
        # ODEN prior: old coefficient carries the full mass of K old datasets;
        # each new dataset contributes 1 unit. Total = K + n_new.
        K = len(self.datasets)
        n_new = len(self.new_datasets)
        total = K + n_new
        reference_weights = {
            self.OLD_KEY: K / total,
            **{d: 1.0 / total for d in self.new_datasets},
        }
        old_source_mixture = normalize_weights(self.old_weights)
        relative_sizes = {
            d: 1.0 / total for d in self._union_datasets()
        }
        source_mixtures = {self.OLD_KEY: old_source_mixture}
        write_olmix_config(
            ratios_file,
            metrics_file,
            fit_dataset_names,
            config_path,
            relative_sizes=relative_sizes,
            source_mixtures=source_mixtures,
        )
        opt_weights = run_olmix_fit(config_path, olmix_dir / "olmix_out")
        if opt_weights is None:
            self.state.set("olmix", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  olmix_fit -> FAILED")
            return "STOP"

        with open(olmix_dir / "olmix_best_mix.json", "w") as f:
            json.dump(
                {
                    "dataset_names": fit_dataset_names,
                    "eval_dataset_names": eval_dataset_names,
                    "optimal_weights": opt_weights,
                    "reference_weights": reference_weights,
                    "relative_sizes": relative_sizes,
                    "source_mixtures": source_mixtures,
                },
                f,
                indent=2,
            )
        self.state.set(
            "olmix",
            {
                "status": "COMPLETED",
                "optimal_weights": opt_weights,
                "reference_weights": reference_weights,
                "relative_sizes": relative_sizes,
                "source_mixtures": source_mixtures,
            },
        )
        self.state.set("substage", "final_train")
        self.state.save()
        print(f"  olmix_fit -> COMPLETED: {opt_weights}")
        return "ADVANCE"

    def _run_final_train(self):
        final_state = self.state.get("final_train", default={})
        status = final_state.get("status", "PENDING")
        output_dir = str(self.run_dir / "final_train")

        if self.dry_run:
            print(
                f"  [DRY RUN] final_train -> would continue from {self.base_model_dir} for {self.final_steps} steps"
            )
            return "CONTINUE"
        if status == "COMPLETED":
            self.state.set("status", "COMPLETED")
            self.state.save()
            print("  final_train -> COMPLETED")
            return "CONTINUE"
        if status == "SUBMITTED":
            job_id = final_state["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                self.state.set("final_train", "status", "COMPLETED")
                self.state.set("status", "COMPLETED")
                self.state.save()
                print("  final_train -> COMPLETED")
                return "CONTINUE"
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"  final_train -> {slurm_status} (job {job_id})")
                return "STOP"
            self.state.set("final_train", "status", "FAILED")
            self.state.set("status", "FAILED")
            self.state.save()
            print("  final_train -> FAILED")
            return "STOP"
        if status == "FAILED":
            print("  final_train -> FAILED. Use retry to reset.")
            return "STOP"

        opt_weights = self.state.get("olmix", "optimal_weights", default=None)
        if opt_weights is None:
            self.state.set("final_train", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  final_train -> missing optimized weights")
            return "STOP"

        # Expand the (n_new+1)-way optimal weights into the full mix via the
        # shared helper (see compute_expanded_weights docstring). In both the
        # pretrain rebalance and continual expand cases the sum equals
        # w_old + Σ w_new = 1.
        expanded = self.compute_expanded_weights(opt_weights)
        all_datasets = self._union_datasets()
        all_train_files = [self._union_train_file(d) for d in all_datasets]
        all_eval_files = [self._union_eval_file(d) for d in all_datasets]
        train_weights = [expanded[d] for d in all_datasets]
        args = _build_train_args(
            model_dir=self.base_model_dir,
            model_init_mode="pretrained",
            train_data_files=all_train_files,
            output_dir=output_dir,
            eval_data_files=all_eval_files,
            train_weights=train_weights,
            max_steps=self.final_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            warmup_steps=0,
            seed=self.seed,
        )
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"filo-final-{self.model_size}-{self.max_steps}",
            sbatch_opts=self.sbatch_opts,
        )
        if job_id is None:
            return "STOP"
        self.state.set(
            "final_train",
            {
                "status": "SUBMITTED",
                "job_id": job_id,
                "output_dir": output_dir,
            },
        )
        self.state.set("substage", "final_train")
        self.state.set("status", "IN_PROGRESS")
        self.state.save()
        print("  final_train -> Submitted")
        return "STOP"


class MergeMixExperiment(ExperimentBase):
    """MergeMix baseline: train one model per dataset, then evaluate randomly
    weighted merges of those K models as proxies for data mixes.

    Stages:
      1. dataset_train  - train K models, one per dataset
      2. proxy_merge    - merge K models with proxy_count random weight combos
      3. proxy_eval     - evaluate each merged model
      4. olmix_fit      - fit optimal mix from proxy eval results
      5. final_train    - train from scratch with optimal mix
    """

    algorithm_name = "pretrain_mergemix"

    def __init__(
        self,
        model_size="150M",
        max_steps=None,
        datasets=None,
        train_files=None,
        eval_files=None,
        proxy_count=12,
        proxy_fraction=0.2,
        seed=None,
        dry_run=False,
    ):
        super().__init__(
            model_size=model_size,
            max_steps=max_steps,
            datasets=datasets,
            train_files=train_files,
            eval_files=eval_files,
            proxy_count=proxy_count,
            proxy_fraction=proxy_fraction,
            seed=seed,
            dry_run=dry_run,
        )
        if not self.state.data:
            self.state._data = {
                "condition": self.algorithm_name,
                "model_size": model_size,
                "model_id": self.model_id,
                "max_steps": self.max_steps,
                "datasets": self.datasets,
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "proxy_count": proxy_count,
                "proxy_fraction": proxy_fraction,
                "seed": seed,
                "proxy_specs": _sample_proxy_weights(
                    self.datasets, proxy_count, 0 if seed is None else seed
                ),
                "substage": "dataset_train",
                "status": "PENDING",
            }
            self.state.save()

    def _dataset_run_dir(self, dataset):
        return self.run_dir / "dataset_runs" / dataset

    def _proxy_model_dir(self, merge_id):
        return self.run_dir / "proxy_merges" / merge_id

    def _proxy_eval_dir(self, merge_id):
        return self.run_dir / "proxy_eval" / merge_id

    def run(self):
        print(f"=== MergeMix ({self.model_size}, {self.max_steps} steps) ===")
        substage = self.state.get("substage", default="dataset_train")
        if substage == "dataset_train":
            result = self._run_dataset_train()
            if result != "ADVANCE":
                return
            substage = "proxy_merge"
        if substage == "proxy_merge":
            result = self._run_proxy_merge()
            if result != "ADVANCE":
                return
            substage = "proxy_eval"
        if substage == "proxy_eval":
            result = self._run_proxy_eval()
            if result != "ADVANCE":
                return
            substage = "olmix_fit"
        if substage == "olmix_fit":
            result = self._run_olmix_fit()
            if result != "ADVANCE":
                return
            substage = "final_train"
        if substage == "final_train":
            self._run_final_train()

    def _run_dataset_train(self):
        """Train one model per dataset."""
        train_state = self.state.get("dataset_train", default={})
        any_running = False

        for dataset in self.datasets:
            item = train_state.get(dataset, {})
            status = item.get("status", "PENDING")
            output_dir = str(self._dataset_run_dir(dataset))

            if status == "COMPLETED":
                continue

            if status == "SUBMITTED":
                job_id = item["job_id"]
                slurm_status = self.slurm.check_status(job_id)
                if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                    self.state.set("dataset_train", dataset, "status", "COMPLETED")
                    self.state.save()
                    print(f"  dataset_train[{dataset}] -> COMPLETED")
                    continue
                if slurm_status in ("RUNNING", "PENDING"):
                    any_running = True
                    print(
                        f"  dataset_train[{dataset}] -> {slurm_status} (job {job_id})"
                    )
                    continue
                self.state.set("dataset_train", dataset, "status", "FAILED")
                self.state.set("status", "FAILED")
                self.state.save()
                print(f"  dataset_train[{dataset}] -> FAILED")
                return "STOP"

            if status == "FAILED":
                print(f"  dataset_train[{dataset}] -> FAILED. Use retry to reset.")
                return "STOP"

            args = _build_train_args(
                model_dir=self.model_id,
                model_init_mode="config",
                train_data_files=[self.train_files[dataset]],
                output_dir=output_dir,
                eval_data_files=self._eval_file_list(),
                max_steps=self.proxy_steps,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                warmup_steps=int(round(self.warmup_steps * self.proxy_fraction)),
                seed=self.seed,
            )
            job_id = self.slurm.submit(
                SCRIPTS_DIR / "slurm_launch_train.sh",
                args,
                job_name=f"mergemix-ds-{dataset}-{self.model_size}-{self.max_steps}",
                sbatch_opts=self.sbatch_opts,
            )
            if job_id is None:
                return "STOP"
            if self.dry_run:
                continue
            self.state.set(
                "dataset_train",
                dataset,
                {
                    "status": "SUBMITTED",
                    "job_id": job_id,
                    "output_dir": output_dir,
                },
            )
            self.state.set("substage", "dataset_train")
            self.state.set("status", "IN_PROGRESS")
            self.state.save()
            print(f"  dataset_train[{dataset}] -> Submitted")

        if self.dry_run:
            print(
                f"  [DRY RUN] dataset_train -> would launch {len(self.datasets)} per-dataset jobs"
            )
            return "ADVANCE"
        train_state = self.state.get("dataset_train", default={})
        all_done = all(
            train_state.get(ds, {}).get("status") == "COMPLETED" for ds in self.datasets
        )
        if all_done:
            self.state.set("substage", "proxy_merge")
            self.state.save()
            return "ADVANCE"
        return "STOP"

    def _run_proxy_merge(self):
        """Merge K per-dataset models with each proxy_spec's weights."""
        proxy_state = self.state.get("proxy_merge", default={})
        status = proxy_state.get("status", "PENDING")
        if status == "COMPLETED":
            return "ADVANCE"
        if status == "FAILED":
            print("  proxy_merge -> FAILED. Use retry to reset.")
            return "STOP"
        if self.dry_run:
            print(
                f"  [DRY RUN] proxy_merge -> would build {len(self.proxy_specs)} merged proxies"
            )
            return "ADVANCE"

        model_dirs = {}
        for dataset in self.datasets:
            item = self.state.get("dataset_train", dataset)
            if item is None or item.get("status") != "COMPLETED":
                print(f"  proxy_merge -> missing completed training for {dataset}")
                self.state.set("proxy_merge", {"status": "FAILED"})
                self.state.set("status", "FAILED")
                self.state.save()
                return "STOP"
            model_dirs[dataset] = item["output_dir"]

        try:
            for spec in self.proxy_specs:
                out = self._proxy_model_dir(spec["merge_id"])
                if _sentinel_exists(out):
                    continue
                _save_multi_merge_model(model_dirs, spec["weights"], out)
        except Exception as exc:
            self.state.set("proxy_merge", {"status": "FAILED", "error": str(exc)})
            self.state.set("status", "FAILED")
            self.state.save()
            print(f"  proxy_merge -> FAILED: {exc}")
            return "STOP"

        self.state.set("proxy_merge", {"status": "COMPLETED"})
        self.state.set("substage", "proxy_eval")
        self.state.save()
        print("  proxy_merge -> COMPLETED")
        return "ADVANCE"

    def _run_proxy_eval(self):
        """Evaluate each merged proxy model on all eval datasets (one batched job)."""
        return self._run_batch_proxy_eval(job_name_prefix="mergemix-eval")

    def _run_olmix_fit(self):
        olmix_state = self.state.get("olmix", default={})
        status = olmix_state.get("status", "PENDING")
        if status == "COMPLETED":
            return "ADVANCE"
        if status == "FAILED":
            print("  olmix_fit -> FAILED. Use retry to reset.")
            return "STOP"
        if self.dry_run:
            print("  [DRY RUN] olmix_fit -> would fit from proxy eval rows")
            return "ADVANCE"

        ratios_rows, metrics_rows = get_proxy_rows(
            self.run_dir / "proxy_eval", self.proxy_specs, self.datasets, self.datasets
        )
        if not ratios_rows:
            self.state.set("olmix", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  olmix_fit -> No valid proxy rows")
            return "STOP"

        olmix_dir = self.run_dir / "olmix"
        ratios_file, metrics_file = write_csvs(
            ratios_rows, metrics_rows, self.datasets, olmix_dir, self.datasets
        )
        config_path = olmix_dir / "fit_config.yaml"
        write_olmix_config(ratios_file, metrics_file, self.datasets, config_path)
        opt_weights = run_olmix_fit(config_path, olmix_dir / "olmix_out")
        if opt_weights is None:
            self.state.set("olmix", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  olmix_fit -> FAILED")
            return "STOP"
        with open(olmix_dir / "olmix_best_mix.json", "w") as f:
            json.dump(
                {
                    "dataset_names": self.datasets,
                    "eval_dataset_names": self.datasets,
                    "optimal_weights": opt_weights,
                },
                f,
                indent=2,
            )
        self.state.set("olmix", {"status": "COMPLETED", "optimal_weights": opt_weights})
        self.state.set("substage", "final_train")
        self.state.save()
        print(f"  olmix_fit -> COMPLETED: {opt_weights}")
        return "ADVANCE"

    def _run_final_train(self):
        final_state = self.state.get("final_train", default={})
        status = final_state.get("status", "PENDING")
        output_dir = str(self.run_dir / "final_train")

        if self.dry_run:
            print(
                f"  [DRY RUN] final_train -> would train {self.model_size} from config for {self.max_steps} steps"
            )
            return "CONTINUE"
        if status == "COMPLETED":
            self.state.set("status", "COMPLETED")
            self.state.save()
            print("  final_train -> COMPLETED")
            return "CONTINUE"
        if status == "SUBMITTED":
            job_id = final_state["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                self.state.set("final_train", "status", "COMPLETED")
                self.state.set("status", "COMPLETED")
                self.state.save()
                print("  final_train -> COMPLETED")
                return "CONTINUE"
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"  final_train -> {slurm_status} (job {job_id})")
                return "STOP"
            self.state.set("final_train", "status", "FAILED")
            self.state.set("status", "FAILED")
            self.state.save()
            print("  final_train -> FAILED")
            return "STOP"
        if status == "FAILED":
            print("  final_train -> FAILED. Use retry to reset.")
            return "STOP"

        opt_weights = self.state.get("olmix", "optimal_weights", default=None)
        if opt_weights is None:
            self.state.set("final_train", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  final_train -> missing optimized weights")
            return "STOP"

        args = _build_train_args(
            model_dir=self.model_id,
            model_init_mode="config",
            train_data_files=self._train_file_list(),
            output_dir=output_dir,
            eval_data_files=self._eval_file_list(),
            train_weights=[opt_weights[ds] for ds in self.datasets],
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            seed=self.seed,
        )
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"mergemix-final-{self.model_size}-{self.max_steps}",
            sbatch_opts=self.sbatch_opts,
        )
        if job_id is None:
            return "STOP"
        self.state.set(
            "final_train",
            {"status": "SUBMITTED", "job_id": job_id, "output_dir": output_dir},
        )
        self.state.set("substage", "final_train")
        self.state.set("status", "IN_PROGRESS")
        self.state.save()
        print("  final_train -> Submitted")
        return "STOP"


class SwarmExperiment(ExperimentBase):
    algorithm_name = "pretrain_swarm"

    def __init__(
        self,
        model_size="150M",
        max_steps=None,
        datasets=None,
        train_files=None,
        eval_files=None,
        proxy_count=12,
        proxy_fraction=0.2,
        proxy_model_size="20M",
        proxy_model_id=None,
        reuse_150m_mix=False,
        seed=None,
        dry_run=False,
    ):
        self.proxy_model_size = proxy_model_size
        self.reuse_150m_mix = reuse_150m_mix
        self.proxy_model_cfg = _resolve_model_config(
            proxy_model_size, max_steps=None, use_proxy_defaults=True
        )
        if proxy_model_id is not None:
            self.proxy_model_cfg = dict(self.proxy_model_cfg)
            self.proxy_model_cfg["base_model"] = proxy_model_id

        super().__init__(
            model_size=model_size,
            max_steps=max_steps,
            datasets=datasets,
            train_files=train_files,
            eval_files=eval_files,
            proxy_count=proxy_count,
            proxy_fraction=proxy_fraction,
            seed=seed,
            dry_run=dry_run,
        )
        if not self.state.data:
            self.state._data = {
                "condition": self.algorithm_name,
                "model_size": model_size,
                "model_id": self.model_id,
                "max_steps": self.max_steps,
                "datasets": self.datasets,
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "proxy_count": proxy_count,
                "proxy_fraction": proxy_fraction,
                "proxy_model_size": proxy_model_size,
                "proxy_model_id": self.proxy_model_cfg["base_model"],
                "reuse_150m_mix": reuse_150m_mix,
                "seed": seed,
                "proxy_specs": _sample_proxy_weights(
                    self.datasets, proxy_count, 0 if seed is None else seed
                ),
                "substage": "proxy_train",
                "status": "PENDING",
            }
            self.state.save()

    def _proxy_run_dir(self, merge_id):
        return self.run_dir / "proxy_runs" / merge_id

    def _find_reusable_150m_mix(self):
        if not self.reuse_150m_mix or self.model_size == "150M":
            return None

        source_root = RUNS_ROOT / self.algorithm_name / "150M"
        if not source_root.exists():
            return None

        expected = {
            "datasets": self.datasets,
            "train_files": self.train_files,
            "eval_files": self.eval_files,
            "proxy_count": self.proxy_count,
            "proxy_fraction": self.proxy_fraction,
            "proxy_model_size": self.proxy_model_size,
            "proxy_model_id": self.proxy_model_cfg["base_model"],
            "seed": self.seed,
            "proxy_specs": self.proxy_specs,
        }

        for state_path in sorted(source_root.glob("**/state.json")):
            if state_path == self.state.path:
                continue
            with open(state_path) as f:
                source_state = json.load(f)
            if source_state.get("olmix", {}).get("status") != "COMPLETED":
                continue
            opt_weights = source_state.get("olmix", {}).get("optimal_weights")
            if opt_weights is None:
                continue
            if any(source_state.get(key) != value for key, value in expected.items()):
                continue
            return state_path.parent, opt_weights

        return None

    def _maybe_reuse_150m_mix(self):
        source = self._find_reusable_150m_mix()
        if source is None:
            return False

        source_run_dir, opt_weights = source
        print(f"  reuse_150m_mix -> using olmix weights from {source_run_dir}")
        if self.dry_run:
            return True

        self.state.set(
            "proxy_reuse",
            {"status": "COMPLETED", "source_run": str(source_run_dir)},
        )
        self.state.set(
            "olmix",
            {
                "status": "COMPLETED",
                "optimal_weights": opt_weights,
                "reused_from": str(source_run_dir),
            },
        )
        self.state.set("substage", "final_train")
        self.state.set("status", "IN_PROGRESS")
        self.state.save()
        return True

    def run(self):
        print(
            f"=== Swarm ({self.model_size}, {self.max_steps} steps; proxy={self.proxy_model_size}) ==="
        )
        substage = self.state.get("substage", default="proxy_train")
        if substage in {"proxy_train", "olmix_fit"} and self._maybe_reuse_150m_mix():
            substage = "final_train"
        if substage == "proxy_train":
            result = self._run_proxy_train()
            if result != "ADVANCE":
                return
            substage = "olmix_fit"
        if substage == "olmix_fit":
            result = self._run_olmix_fit()
            if result != "ADVANCE":
                return
            substage = "final_train"
        if substage == "final_train":
            self._run_final_train()

    def _run_proxy_train(self):
        proxy_state = self.state.get("proxy_train", default={})
        any_running = False
        proxy_lr = self.proxy_model_cfg["learning_rate"]
        proxy_batch = self.proxy_model_cfg["batch_size"]
        proxy_warmup = int(
            round(self.proxy_model_cfg["warmup_steps"] * self.proxy_fraction)
        )

        for spec in self.proxy_specs:
            merge_id = spec["merge_id"]
            item = proxy_state.get(merge_id, {})
            status = item.get("status", "PENDING")
            output_dir = str(self._proxy_run_dir(merge_id))

            if status == "COMPLETED":
                continue

            if status == "SUBMITTED":
                job_id = item["job_id"]
                slurm_status = self.slurm.check_status(job_id)
                if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                    self.state.set("proxy_train", merge_id, "status", "COMPLETED")
                    self.state.save()
                    print(f"  proxy_train[{merge_id}] -> COMPLETED")
                    continue
                if slurm_status in ("RUNNING", "PENDING"):
                    any_running = True
                    print(f"  proxy_train[{merge_id}] -> {slurm_status} (job {job_id})")
                    continue
                # Tolerate failed proxies — mark FAILED and continue to next.
                self.state.set("proxy_train", merge_id, "status", "FAILED")
                self.state.save()
                print(f"  proxy_train[{merge_id}] -> FAILED (skipping)")
                continue

            if status == "FAILED":
                # Skip failed proxies; olmix_fit can proceed on the survivors.
                print(f"  proxy_train[{merge_id}] -> FAILED (skipping)")
                continue

            args = _build_train_args(
                model_dir=self.proxy_model_cfg["base_model"],
                model_init_mode="config",
                train_data_files=self._train_file_list(),
                output_dir=output_dir,
                eval_data_files=self._eval_file_list(),
                train_weights=[spec["weights"][ds] for ds in self.datasets],
                max_steps=self.proxy_steps,
                learning_rate=proxy_lr,
                batch_size=proxy_batch,
                warmup_steps=proxy_warmup,
                seed=self.seed,
            )
            job_id = self.slurm.submit(
                SCRIPTS_DIR / "slurm_launch_train.sh",
                args,
                job_name=f"swarm-proxy-{merge_id}-{self.model_size}-{self.max_steps}",
                sbatch_opts=self.proxy_model_cfg["sbatch_opts"],
            )
            if job_id is None:
                return "STOP"
            if self.dry_run:
                continue
            self.state.set(
                "proxy_train",
                merge_id,
                {
                    "status": "SUBMITTED",
                    "job_id": job_id,
                    "output_dir": output_dir,
                },
            )
            self.state.set("substage", "proxy_train")
            self.state.set("status", "IN_PROGRESS")
            self.state.save()
            print(f"  proxy_train[{merge_id}] -> Submitted")

        if self.dry_run:
            print(
                f"  [DRY RUN] proxy_train -> would launch {len(self.proxy_specs)} {self.proxy_model_size} proxy jobs"
            )
            return "ADVANCE"
        proxy_state = self.state.get("proxy_train", default={})
        n_pending = sum(
            1
            for spec in self.proxy_specs
            if proxy_state.get(spec["merge_id"], {}).get("status") == "SUBMITTED"
        )
        if any_running or n_pending:
            return "STOP"
        n_completed = sum(
            1
            for spec in self.proxy_specs
            if proxy_state.get(spec["merge_id"], {}).get("status") == "COMPLETED"
        )
        n_failed = sum(
            1
            for spec in self.proxy_specs
            if proxy_state.get(spec["merge_id"], {}).get("status") == "FAILED"
        )
        n_total = len(self.proxy_specs)
        print(
            f"  proxy_train -> {n_completed}/{n_total} completed "
            f"({n_failed} failed, skipping)"
        )
        if n_completed == 0:
            self.state.set("status", "FAILED")
            self.state.save()
            print("  proxy_train -> no successful proxies; cannot fit olmix")
            return "STOP"
        self.state.set("substage", "olmix_fit")
        self.state.save()
        return "ADVANCE"

    def _run_olmix_fit(self):
        olmix_state = self.state.get("olmix", default={})
        status = olmix_state.get("status", "PENDING")
        if status == "COMPLETED":
            return "ADVANCE"
        if status == "FAILED":
            print("  olmix_fit -> FAILED. Use retry to reset.")
            return "STOP"
        if self.dry_run:
            print("  [DRY RUN] olmix_fit -> would fit from swarm proxy rows")
            return "ADVANCE"

        ratios_rows, metrics_rows = get_training_rows(
            self.run_dir / "proxy_runs", self.proxy_specs, self.datasets, self.datasets
        )
        if not ratios_rows:
            self.state.set("olmix", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  olmix_fit -> No valid swarm rows")
            return "STOP"

        olmix_dir = self.run_dir / "olmix"
        ratios_file, metrics_file = write_csvs(
            ratios_rows, metrics_rows, self.datasets, olmix_dir, self.datasets
        )
        config_path = olmix_dir / "fit_config.yaml"
        write_olmix_config(ratios_file, metrics_file, self.datasets, config_path)
        opt_weights = run_olmix_fit(config_path, olmix_dir / "olmix_out")
        if opt_weights is None:
            self.state.set("olmix", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  olmix_fit -> FAILED")
            return "STOP"
        with open(olmix_dir / "olmix_best_mix.json", "w") as f:
            json.dump(
                {
                    "dataset_names": self.datasets,
                    "eval_dataset_names": self.datasets,
                    "optimal_weights": opt_weights,
                },
                f,
                indent=2,
            )
        self.state.set("olmix", {"status": "COMPLETED", "optimal_weights": opt_weights})
        self.state.set("substage", "final_train")
        self.state.save()
        print(f"  olmix_fit -> COMPLETED: {opt_weights}")
        return "ADVANCE"

    def _run_final_train(self):
        final_state = self.state.get("final_train", default={})
        status = final_state.get("status", "PENDING")
        output_dir = str(self.run_dir / "final_train")

        if self.dry_run:
            print(
                f"  [DRY RUN] final_train -> would train {self.model_size} from config for {self.max_steps} steps"
            )
            return "CONTINUE"
        if status == "COMPLETED":
            self.state.set("status", "COMPLETED")
            self.state.save()
            print("  final_train -> COMPLETED")
            return "CONTINUE"
        if status == "SUBMITTED":
            job_id = final_state["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                self.state.set("final_train", "status", "COMPLETED")
                self.state.set("status", "COMPLETED")
                self.state.save()
                print("  final_train -> COMPLETED")
                return "CONTINUE"
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"  final_train -> {slurm_status} (job {job_id})")
                return "STOP"
            self.state.set("final_train", "status", "FAILED")
            self.state.set("status", "FAILED")
            self.state.save()
            print("  final_train -> FAILED")
            return "STOP"
        if status == "FAILED":
            print("  final_train -> FAILED. Use retry to reset.")
            return "STOP"

        opt_weights = self.state.get("olmix", "optimal_weights", default=None)
        if opt_weights is None:
            self.state.set("final_train", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  final_train -> missing optimized weights")
            return "STOP"

        args = _build_train_args(
            model_dir=self.model_id,
            model_init_mode="config",
            train_data_files=self._train_file_list(),
            output_dir=output_dir,
            eval_data_files=self._eval_file_list(),
            train_weights=[opt_weights[ds] for ds in self.datasets],
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            seed=self.seed,
        )
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"swarm-final-{self.model_size}-{self.max_steps}",
            sbatch_opts=self.sbatch_opts,
        )
        if job_id is None:
            return "STOP"
        self.state.set(
            "final_train",
            {"status": "SUBMITTED", "job_id": job_id, "output_dir": output_dir},
        )
        self.state.set("substage", "final_train")
        self.state.set("status", "IN_PROGRESS")
        self.state.save()
        print("  final_train -> Submitted")
        return "STOP"


class OnPolicyMixChain:
    """Three-stage OnPolicyMix orchestrator:

    Stage 1: ERM prefix on ``datasets`` (uniform) — no LoRA probes.
    Stage 2: 5-LoRA rebalance on top of stage 1's prefix (old = uniform 1/K).
    Stage 3: 2-LoRA tulu expand on top of stage 2's final (old = stage 2's
             expanded weights over ``datasets``).

    All stages share ``runs/pretrain_filo_mix/{size}/{total_steps}_steps/seed_{seed}/``
    and nest their own state under ``stage_{1,2,3}/``.
    """

    algorithm_name = "pretrain_filo_mix"

    def __init__(
        self,
        model_size="150M",
        stage_1_steps=10000,
        stage_2_steps=20000,
        stage_3_steps=10000,
        datasets=None,
        train_files=None,
        eval_files=None,
        new_datasets=None,
        new_train_files=None,
        new_eval_files=None,
        lora_steps_fraction=0.2,
        proxy_count=12,
        seed=None,
        dry_run=False,
    ):
        self.model_size = model_size
        self.stage_1_steps = stage_1_steps
        self.stage_2_steps = stage_2_steps
        self.stage_3_steps = stage_3_steps
        self.total_steps = stage_1_steps + stage_2_steps + stage_3_steps
        self.datasets = list(datasets) if datasets else list(DEFAULT_DATASETS)
        self.train_files = train_files or _default_train_files(self.datasets)
        self.eval_files = eval_files or _default_eval_files(self.datasets)
        if new_datasets is None:
            new_datasets = ["tulu_flan_v0", "tulu_flan_v2"]
        self.new_datasets = list(new_datasets)
        self.new_train_files = new_train_files or {
            d: f"data-mixes/{d}.txt" for d in self.new_datasets
        }
        self.new_eval_files = new_eval_files or {
            d: f"data-mixes/{d}_eval.txt" for d in self.new_datasets
        }
        self.lora_steps_fraction = lora_steps_fraction
        self.proxy_count = proxy_count
        self.seed = seed
        self.dry_run = dry_run

        seed_suffix = f"seed_{seed}" if seed is not None else "seed_none"
        self.run_dir = (
            RUNS_ROOT
            / self.algorithm_name
            / model_size
            / f"{self.total_steps}_steps"
            / seed_suffix
        )
        self.state = StateFile(self.run_dir / "state.json")
        if not self.state.data:
            self.state._data = {
                "condition": self.algorithm_name,
                "chain": True,
                "model_size": model_size,
                "max_steps": self.total_steps,
                "stage_1_steps": stage_1_steps,
                "stage_2_steps": stage_2_steps,
                "stage_3_steps": stage_3_steps,
                "datasets": self.datasets,
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "new_datasets": self.new_datasets,
                "new_train_files": self.new_train_files,
                "new_eval_files": self.new_eval_files,
                "lora_steps_fraction": lora_steps_fraction,
                "proxy_count": proxy_count,
                "seed": seed,
                "active_stage": 1,
                "stages": {},
                "status": "PENDING",
            }
            self.state.save()

    def _stage_1(self):
        return OnPolicyMixExperiment(
            model_size=self.model_size,
            max_steps=self.stage_1_steps,
            datasets=self.datasets,
            train_files=self.train_files,
            eval_files=self.eval_files,
            proxy_count=0,
            prefix_fraction=1.0,
            lora_steps_fraction=0.0,
            seed=self.seed,
            dry_run=self.dry_run,
            run_dir=self.run_dir / "stage_1",
            stage_id="stage_1",
            stop_after="prefix_train",
        )

    def _stage_2(self):
        K = len(self.datasets)
        uniform = {d: 1.0 / K for d in self.datasets}
        return OnPolicyMixExperiment(
            model_size=self.model_size,
            max_steps=self.stage_2_steps,
            datasets=self.datasets,
            train_files=self.train_files,
            eval_files=self.eval_files,
            old_weights=uniform,
            base_model_dir=str(self.run_dir / "stage_1" / "prefix_train"),
            new_datasets=self.datasets,
            new_train_files=self.train_files,
            new_eval_files=self.eval_files,
            proxy_count=self.proxy_count,
            prefix_fraction=0.0,
            lora_steps_fraction=self.lora_steps_fraction,
            seed=self.seed,
            dry_run=self.dry_run,
            run_dir=self.run_dir / "stage_2",
            stage_id="stage_2",
        )

    def _stage_3(self, old_weights):
        return OnPolicyMixExperiment(
            model_size=self.model_size,
            max_steps=self.stage_3_steps,
            datasets=self.datasets,
            train_files=self.train_files,
            eval_files=self.eval_files,
            old_weights=old_weights,
            base_model_dir=str(self.run_dir / "stage_2" / "final_train"),
            new_datasets=self.new_datasets,
            new_train_files=self.new_train_files,
            new_eval_files=self.new_eval_files,
            proxy_count=self.proxy_count,
            prefix_fraction=0.0,
            lora_steps_fraction=self.lora_steps_fraction,
            seed=self.seed,
            dry_run=self.dry_run,
            run_dir=self.run_dir / "stage_3",
            stage_id="stage_3",
        )

    def _save_chain(self):
        if not self.dry_run:
            self.state.save()

    def run(self):
        print(
            f"=== OnPolicyMix Chain ({self.model_size}, "
            f"{self.stage_1_steps}+{self.stage_2_steps}+{self.stage_3_steps} steps) ==="
        )
        K = len(self.datasets)

        # Stage 1 — ERM prefix.
        self.state.set("active_stage", 1)
        self._save_chain()
        s1 = self._stage_1()
        s1.run()
        s1_done = (
            s1.state.get("prefix_train", "status", default="PENDING") == "COMPLETED"
        )
        if not s1_done and not self.dry_run:
            return
        self.state.set(
            "stages",
            "stage_1",
            {
                "status": "COMPLETED" if s1_done else "PENDING",
                "base_model_dir": str(self.run_dir / "stage_1" / "prefix_train"),
                "final_weights": {d: 1.0 / K for d in self.datasets},
            },
        )
        self._save_chain()

        # Stage 2 — 5-LoRA rebalance.
        self.state.set("active_stage", 2)
        self._save_chain()
        s2 = self._stage_2()
        s2.run()
        s2_done = (
            s2.state.get("final_train", "status", default="PENDING") == "COMPLETED"
        )
        if not s2_done and not self.dry_run:
            return
        s2_final = (
            s2.compute_expanded_weights()
            if s2_done
            else {d: 1.0 / K for d in self.datasets}  # dry-run placeholder
        )
        self.state.set(
            "stages",
            "stage_2",
            {
                "status": "COMPLETED" if s2_done else "PENDING",
                "base_model_dir": str(self.run_dir / "stage_2" / "final_train"),
                "final_weights": s2_final,
            },
        )
        self._save_chain()

        # Stage 3 — 2-LoRA tulu expand.
        self.state.set("active_stage", 3)
        self._save_chain()
        s3 = self._stage_3(old_weights=s2_final)
        s3.run()
        s3_done = (
            s3.state.get("final_train", "status", default="PENDING") == "COMPLETED"
        )
        if not s3_done and not self.dry_run:
            return
        s3_final = s3.compute_expanded_weights() if s3_done else None
        self.state.set(
            "stages",
            "stage_3",
            {
                "status": "COMPLETED" if s3_done else "PENDING",
                "base_model_dir": str(self.run_dir / "stage_3" / "final_train"),
                "final_weights": s3_final,
            },
        )
        if s3_done:
            self.state.set("status", "COMPLETED")
        self._save_chain()


FiLoMixExperiment = OnPolicyMixExperiment
FiLoMixChain = OnPolicyMixChain


def on_policy_mix(
    model_size="150M",
    stage_1_steps=10000,
    stage_2_steps=20000,
    stage_3_steps=10000,
    datasets=None,
    train_files=None,
    eval_files=None,
    new_datasets=None,
    new_train_files=None,
    new_eval_files=None,
    proxy_count=12,
    lora_steps_fraction=0.2,
    dry_run=False,
    seed=None,
):
    """Three-stage OnPolicyMix: ERM -> 5-dataset rebalance -> 2-dataset expansion."""
    if isinstance(datasets, (list, tuple)):
        datasets = list(datasets)
    if isinstance(new_datasets, (list, tuple)):
        new_datasets = list(new_datasets)
    chain = OnPolicyMixChain(
        model_size=model_size,
        stage_1_steps=stage_1_steps,
        stage_2_steps=stage_2_steps,
        stage_3_steps=stage_3_steps,
        datasets=datasets,
        train_files=train_files,
        eval_files=eval_files,
        new_datasets=new_datasets,
        new_train_files=new_train_files,
        new_eval_files=new_eval_files,
        proxy_count=proxy_count,
        lora_steps_fraction=lora_steps_fraction,
        dry_run=dry_run,
        seed=seed,
    )
    chain.run()


def filo(**kwargs):
    """Legacy wrapper for the pretraining OnPolicyMix pipeline."""
    return on_policy_mix(**kwargs)


def run(**kwargs):
    return on_policy_mix(**kwargs)


def cpt(
    pretrain_run_dir,
    model_size="150M",
    max_steps=None,
    new_datasets=None,
    new_train_files=None,
    new_eval_files=None,
    proxy_count=12,
    lora_steps_fraction=0.2,
    dry_run=False,
    seed=None,
):
    """Continual pretraining: continue a pretrain endpoint on new datasets
    via OnPolicyMix (old pretrain mix collapsed into one
    coefficient, LoRA probes on the new datasets, proxy-merge+fit in the
    ``{old, *new_datasets}`` simplex, final full-parameter train with the
    expanded mix).
    """
    if isinstance(new_datasets, (list, tuple)):
        new_datasets = list(new_datasets)
    exp = OnPolicyMixExperiment.from_pretrain_run(
        pretrain_run_dir=pretrain_run_dir,
        new_datasets=new_datasets,
        new_train_files=new_train_files,
        new_eval_files=new_eval_files,
        model_size=model_size,
        max_steps=max_steps,
        proxy_count=proxy_count,
        lora_steps_fraction=lora_steps_fraction,
        dry_run=dry_run,
        seed=seed,
    )
    exp.run()


def mergemix(
    model_size="150M",
    max_steps=None,
    datasets=None,
    train_files=None,
    eval_files=None,
    proxy_count=12,
    proxy_fraction=0.2,
    dry_run=False,
    seed=None,
):
    exp = MergeMixExperiment(
        model_size=model_size,
        max_steps=max_steps,
        datasets=datasets,
        train_files=train_files,
        eval_files=eval_files,
        proxy_count=proxy_count,
        proxy_fraction=proxy_fraction,
        dry_run=dry_run,
        seed=seed,
    )
    exp.run()


def swarm(
    model_size="150M",
    max_steps=None,
    datasets=None,
    train_files=None,
    eval_files=None,
    proxy_count=12,
    proxy_fraction=0.2,
    proxy_model_size="20M",
    proxy_model_id="allenai/DataDecide-c4-20M",
    reuse_150m_mix=False,
    dry_run=False,
    seed=None,
):
    exp = SwarmExperiment(
        model_size=model_size,
        max_steps=max_steps,
        datasets=datasets,
        train_files=train_files,
        eval_files=eval_files,
        proxy_count=proxy_count,
        proxy_fraction=proxy_fraction,
        proxy_model_size=proxy_model_size,
        proxy_model_id=proxy_model_id,
        reuse_150m_mix=reuse_150m_mix,
        dry_run=dry_run,
        seed=seed,
    )
    exp.run()


def status(algorithm="all", model_size="all", max_steps="all"):
    algorithms = (
        [
            "pretrain_filo_mix",
            "pretrain_mergemix",
            "pretrain_swarm",
            "continual_pretrain_filo_mix",
        ]
        if algorithm == "all"
        else [algorithm]
    )
    for algo in algorithms:
        root = RUNS_ROOT / algo
        if not root.exists():
            print(f"No {algo} experiments found.")
            continue
        print(f"\n{algo}:")
        all_state_paths = sorted(root.glob("**/state.json"))
        # Identify chain runs so we can render their stage children inline and
        # skip printing those stage state.json files again at the top level.
        chain_dirs = set()
        for sp in all_state_paths:
            try:
                data = StateFile(sp).data
            except Exception:
                continue
            if data.get("chain"):
                chain_dirs.add(sp.parent)
        for state_path in all_state_paths:
            # Skip stage state files nested under a chain (rendered below).
            if any(
                cd in state_path.parents and state_path.parent != cd
                for cd in chain_dirs
            ):
                continue
            state = StateFile(state_path)
            size = state.get("model_size", default="?")
            steps = state.get("max_steps", default="?")
            seed = state.get("seed", default="?")
            if model_size != "all" and size != model_size:
                continue
            if max_steps != "all" and str(steps) != str(max_steps):
                continue
            overall_status = state.get("status", default="PENDING")
            if state.get("chain"):
                active = state.get("active_stage", default="?")
                print(
                    f"  {size}/{steps}/seed_{seed}: {overall_status} (chain, active=stage_{active})"
                )
                for stage_name in ("stage_1", "stage_2", "stage_3"):
                    inner_path = state_path.parent / stage_name / "state.json"
                    if not inner_path.exists():
                        print(f"      {stage_name}: PENDING")
                        continue
                    inner = StateFile(inner_path)
                    inner_status = inner.get("status", default="PENDING")
                    inner_sub = inner.get("substage", default="unknown")
                    print(f"      {stage_name}: {inner_status} ({inner_sub})")
            else:
                substage = state.get("substage", default="unknown")
                print(f"  {size}/{steps}/seed_{seed}: {overall_status} ({substage})")


def retry(
    algorithm,
    model_size="150M",
    max_steps=10000,
    seed=0,
    substage="prefix_train",
    item=None,
):
    seed_suffix = f"seed_{seed}"
    root = RUNS_ROOT / algorithm / model_size / f"{max_steps}_steps" / seed_suffix
    state = StateFile(root / "state.json")
    if not state.data:
        print(f"No state found for {algorithm}/{model_size}/{max_steps}/{seed_suffix}")
        return

    if substage in {"prefix_train", "proxy_merge", "olmix", "final_train"}:
        state.set(substage, "status", "PENDING")
        state.set("substage", substage if substage != "olmix" else "olmix_fit")
    elif substage == "proxy_eval":
        # Batch eval: clear the batch slot and any per-spec entries so the
        # next run resubmits whatever is still missing a sentinel.
        if item is None:
            state.set("proxy_eval", {})
            state.set("proxy_eval_batch", {})
        else:
            state.set("proxy_eval", item, "status", "PENDING")
            state.set("proxy_eval_batch", {})
        state.set("substage", "proxy_eval")
    elif substage in {"lora_probes", "proxy_train", "dataset_train"}:
        if item is None:
            print(f"retry for {substage} requires --item")
            return
        state.set(substage, item, "status", "PENDING")
        state.set("substage", substage)
    else:
        print(f"Unknown substage: {substage}")
        return
    state.set("status", "IN_PROGRESS")
    state.save()
    suffix = f"/{item}" if item else ""
    print(f"Reset {algorithm}/{model_size}/{max_steps}/{substage}{suffix} to PENDING")


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run,
            "on_policy_mix": on_policy_mix,
            "filo": filo,
            "cpt": cpt,
            "mergemix": mergemix,
            "swarm": swarm,
            "status": status,
            "retry": retry,
        }
    )
