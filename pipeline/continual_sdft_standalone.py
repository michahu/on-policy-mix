"""Continual SDFT/SFT pipeline with explicit old/new LoRA probes.

Standalone variant of ``continual_sdft.py`` that uses the in-tree
``pipeline.olmix_fit`` reimplementation instead of the OLMix submodule,
so the op-mix code release does not depend on the external ``olmix``
package. Runs are written under
``runs/continual_(sft|sdft)_standalone_<base>*/`` to avoid colliding
with the original pipeline.

Usage:
    python -m pipeline.continual_sdft_standalone opm [--dry_run] [--seed 42]
    python -m pipeline.continual_sdft_standalone naive [--dry_run] [--seed 42]
    python -m pipeline.continual_sdft_standalone mix [--dry_run] [--seed 42]
    python -m pipeline.continual_sdft_standalone sft_opm [--dry_run] [--seed 42]
    python -m pipeline.continual_sdft_standalone status [--condition all]
    python -m pipeline.continual_sdft_standalone retry --condition opm --stage 2 --substage old_probe
"""

import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path

import fire
import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.continual import (
    OLD_MIX_KEY,
    SlurmHelper,
    StateFile,
    _sentinel_exists,
    expand_on_policy_mix_weights,
    normalize_weights,
)
from pipeline.olmix_fit import fit_and_propose, write_csvs


ORDERING = ["tooluse_data", "science_data", "medical_data"]

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
RUNS_ROOT = Path("runs")
SCRIPTS_DIR = Path("scripts")
FULL_TRAIN_MAX_STEPS = 10000

DATASET_SIZES = {
    "medical_data": 10000,
    "science_data": 1233,
    "tooluse_data": 4046,
}

OLD_PROBE_NEW_WEIGHT = 0.1
NEW_PROBE_NEW_WEIGHT = 0.9
SDFT_PROXY_VERSION = "sdft_10_90_v1"


def _run_suffix():
    return os.environ.get("SDFT_RUN_SUFFIX", "")


def _hp_suffix(ref_model_mixup_alpha, beta, use_sft):
    if use_sft:
        return ""
    parts = []
    if ref_model_mixup_alpha != 0.01:
        parts.append(f"rma{ref_model_mixup_alpha:g}")
    if beta != 0.0:
        parts.append(f"beta{beta:g}")
    return "_".join(parts)


def _condition_dir(base, use_sft, ref_model_mixup_alpha, beta, seed):
    prefix = "continual_sft_standalone" if use_sft else "continual_sdft_standalone"
    name = f"{prefix}_{base}"
    suffix = _hp_suffix(ref_model_mixup_alpha, beta, use_sft)
    if suffix:
        name += f"_{suffix}"
    run_suffix = _run_suffix()
    if run_suffix:
        name += run_suffix
    if seed is not None:
        name += f"_s{seed}"
    return name


def _stage_seed(base_seed, k):
    if base_seed is None:
        return None
    return base_seed + (k - 1)


def _get_stage_hps(state, stage_key, default_rma, default_beta):
    stage = state.get("stages", stage_key) or {}
    overrides = stage.get("hyperparam_overrides", {})
    return (
        overrides.get("ref_model_mixup_alpha", default_rma),
        overrides.get("beta", default_beta),
    )


def _sdft_sentinel_exists(output_dir, sentinel="model.safetensors"):
    return _sentinel_exists(output_dir, sentinel=sentinel)


def _sdft_proxy_eval_sentinel_exists(output_dir):
    return (Path(output_dir) / "linear_connectivity_results.json").exists()


def _sdft_train_args(
    model_name,
    train_domains,
    output_dir,
    train_weights=None,
    use_lora=False,
    lora_r=16,
    lora_alpha=32,
    learning_rate=2e-5,
    num_prompts_per_batch=32,
    ref_model_mixup_alpha=0.01,
    beta=0.0,
    use_sft=False,
    num_train_epochs=1,
    max_steps=None,
    save_steps=None,
    seed=None,
    eval_domains=None,
    max_train_samples=10000,
):
    args = [
        "--model_name",
        str(model_name),
        "--output_dir",
        str(output_dir),
        "--learning_rate",
        str(learning_rate),
        "--num_train_epochs",
        str(num_train_epochs),
    ]
    if use_sft:
        args.append("--sft")
    else:
        args.extend(
            [
                "--num_prompts_per_batch",
                str(num_prompts_per_batch),
                "--ref_model_mixup_alpha",
                str(ref_model_mixup_alpha),
                "--beta",
                str(beta),
            ]
        )
    args.append("--train_domains")
    args.extend(train_domains)
    if train_weights:
        args.append("--train_weights")
        args.extend(str(w) for w in train_weights)
    if use_lora:
        args.extend(
            [
                "--use_lora",
                "--lora_r",
                str(lora_r),
                "--lora_alpha",
                str(lora_alpha),
            ]
        )
    if save_steps is not None:
        args.extend(["--save_steps", str(save_steps)])
    if max_steps is not None:
        args.extend(["--max_steps", str(max_steps)])
    if seed is not None:
        args.extend(["--seed", str(seed)])
    if eval_domains is not None:
        args.append("--eval_domains")
        args.extend(eval_domains)
    if max_train_samples is not None:
        args.extend(["--max_train_samples", str(max_train_samples)])
    return args


def _stage_proxy_specs(ds):
    specs = []
    span = NEW_PROBE_NEW_WEIGHT - OLD_PROBE_NEW_WEIGHT
    for pct in range(1, 10):
        effective_new = pct / 10.0
        component_alpha = (effective_new - OLD_PROBE_NEW_WEIGHT) / span
        specs.append(
            {
                "merge_id": f"new_{effective_new:.1f}",
                "version": SDFT_PROXY_VERSION,
                "component_alpha": component_alpha,
                "weights": {
                    OLD_MIX_KEY: 1.0 - component_alpha,
                    ds: component_alpha,
                },
                "ratios": {
                    OLD_MIX_KEY: 1.0 - effective_new,
                    ds: effective_new,
                },
            }
        )
    return specs


def _sdft_probe_weights(new_weight, new_dataset, prev_weights):
    result = {}
    old_weights = normalize_weights(prev_weights)
    for dataset, weight in old_weights.items():
        result[dataset] = (1.0 - new_weight) * weight
    result[new_dataset] = result.get(new_dataset, 0.0) + new_weight
    return result


def _load_sdft_model(model_dir, lora_dir=None):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if lora_dir is not None:
        model = PeftModel.from_pretrained(model, str(lora_dir))
        model = model.merge_and_unload()
    return model


def _save_tokenizer_like(model_dir, output_dir):
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(str(output_dir))


def _safetensors_key_map(model_dir):
    """Map state-dict key → on-disk safetensors path for `model_dir`.

    Handles both single-file (`model.safetensors`) and sharded
    (`model.safetensors.index.json` + per-shard files) layouts.
    """
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.exists():
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]
        return {k: model_dir / v for k, v in weight_map.items()}
    if single_path.exists():
        with safe_open(str(single_path), framework="pt") as f:
            keys = list(f.keys())
        return {k: single_path for k in keys}
    raise FileNotFoundError(f"no safetensors weights found in {model_dir}")


def _copy_model_config(src_dir, dst_dir):
    """Copy config.json / generation_config.json (everything `save_pretrained`
    writes besides weights) from `src_dir` to `dst_dir`."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    for name in ("config.json", "generation_config.json"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def _save_sdft_proxy_model(base_model_dir, lora_dirs, weights, output_dir):
    """Save `base + Σ_c α_c · (lora_c − base)` to `output_dir`.

    Pipeline always passes a single LoRA component at α=1.0, which collapses
    to the LoRA-merged base model. Holding only the bf16 model (≈base size)
    avoids the previous fp32 double-buffer that OOM'd at >64G.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nonzero = [
        (component, lora_dirs[component])
        for component in lora_dirs
        if float(weights.get(component, 0.0)) != 0.0
    ]

    if len(nonzero) == 1 and abs(float(weights[nonzero[0][0]]) - 1.0) < 1e-12:
        _, lora_dir = nonzero[0]
        model = _load_sdft_model(base_model_dir, lora_dir=lora_dir)
        model.save_pretrained(str(output_dir))
        del model
        _save_tokenizer_like(base_model_dir, output_dir)
        return

    raise NotImplementedError(
        "_save_sdft_proxy_model only supports a single LoRA component at α=1.0; "
        "the multi-component path was unused. Reintroduce streaming logic if needed."
    )


def _save_sdft_multi_merge_model(model_dirs, weights, output_dir):
    """Linearly interpolate state dicts from `model_dirs` and save to `output_dir`.

    Streams tensors key-by-key via safetensors lazy access. Peak RSS is the
    output dict in bf16 (≈one model) plus a single per-key fp32 scratch
    tensor — instead of the previous `O(n_models)` fp32 dicts, which blew
    past 64G for a 7B base.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    components = list(model_dirs.keys())
    weight_vals = {c: float(weights.get(c, 0.0)) for c in components}
    key_maps = {c: _safetensors_key_map(model_dirs[c]) for c in components}

    first = components[0]
    keys = list(key_maps[first].keys())

    with ExitStack() as stack:
        shard_handles = {}

        def _get(component, key):
            shard = key_maps[component][key]
            cache_key = (component, shard)
            if cache_key not in shard_handles:
                shard_handles[cache_key] = stack.enter_context(
                    safe_open(str(shard), framework="pt", device="cpu")
                )
            return shard_handles[cache_key].get_tensor(key)

        merged = {}
        for k in keys:
            t0 = _get(first, k)
            out_dtype = t0.dtype
            acc = t0.float() * weight_vals[first]
            del t0
            for c in components[1:]:
                w = weight_vals[c]
                if w == 0.0:
                    continue
                t = _get(c, k)
                acc.add_(t.float() * w)
                del t
            merged[k] = acc.to(out_dtype)
            del acc

    save_safetensors(merged, str(output_dir / "model.safetensors"))
    del merged
    _copy_model_config(model_dirs[first], output_dir)
    _save_tokenizer_like(model_dirs[first], output_dir)


def _get_accuracy_proxy_rows(proxy_eval_dir, proxy_specs, dataset_names, eval_dataset_names):
    proxy_eval_dir = Path(proxy_eval_dir)
    specs_by_id = {
        spec["merge_id"]: spec.get("ratios", spec["weights"]) for spec in proxy_specs
    }
    ratios_rows = []
    metrics_rows = []
    if not proxy_eval_dir.exists():
        return ratios_rows, metrics_rows

    for merge_dir in sorted(d for d in proxy_eval_dir.iterdir() if d.is_dir()):
        merge_id = merge_dir.name
        if merge_id not in specs_by_id:
            continue
        results_file = merge_dir / "linear_connectivity_results.json"
        if not results_file.exists():
            continue
        with open(results_file) as handle:
            results = json.load(handle)
        entry = results.get("alpha_1.000") or next(iter(results.values()), None)
        if not entry:
            continue
        per_ds = entry.get("per_dataset_results", {})
        error_rates = {}
        for ds in eval_dataset_names:
            if ds in per_ds and "accuracy" in per_ds[ds]:
                error_rates[ds] = 1.0 - per_ds[ds]["accuracy"]
        if not error_rates:
            continue
        weights = specs_by_id[merge_id]
        ratios_rows.append({"run": merge_id, **{ds: weights[ds] for ds in dataset_names}})
        metrics_rows.append(
            {
                "run": merge_id,
                **{
                    f"eval_{ds}_loss": error_rates[ds]
                    for ds in eval_dataset_names
                    if ds in error_rates
                },
            }
        )
    return ratios_rows, metrics_rows


class SequentialSdftExperiment:
    def __init__(
        self,
        dry_run=False,
        use_sft=False,
        mode="naive",
        ref_model_mixup_alpha=0.01,
        beta=0.0,
        seed=None,
    ):
        self.ordering = list(ORDERING)
        self.dry_run = dry_run
        self.use_sft = use_sft
        self.mode = mode
        self.ref_model_mixup_alpha = ref_model_mixup_alpha
        self.beta = beta
        self.seed = seed
        self.run_dir = RUNS_ROOT / _condition_dir(
            mode, use_sft, ref_model_mixup_alpha, beta, seed
        )
        self.slurm = SlurmHelper(dry_run=dry_run)
        self.state = StateFile(self.run_dir / "state.json")
        if not self.state.data:
            self.state._data = {
                "condition": f"{'sft' if use_sft else 'sdft'}_{mode}",
                "ordering": self.ordering,
                "base_model": BASE_MODEL,
                "stages": {},
            }
            self.state.save()

    def _stage_dir(self, k):
        return self.run_dir / f"stage_{k}_{self.ordering[k - 1]}"

    def _prev_model(self, k):
        if k == 1:
            return BASE_MODEL
        return str(self._stage_dir(k - 1))

    def _weights(self, k, ds):
        if k == 1 or self.mode == "naive":
            return {ds: 1.0}
        old = self.ordering[: k - 1]
        return {ds: 0.9, **{d: 0.1 / len(old) for d in old}}

    def run(self):
        label = "SFT" if self.use_sft else "SDFT"
        print(f"=== {self.mode} {label}: {' -> '.join(self.ordering)} ===")
        for k, ds in enumerate(self.ordering, start=1):
            stage_key = str(k)
            stage = self.state.get("stages", stage_key) or {}
            status = stage.get("status", "PENDING")
            print(f"Stage {k} ({ds}): {status}")
            if status == "COMPLETED":
                continue
            if status != "COMPLETED" and _sdft_sentinel_exists(self._stage_dir(k)):
                weights = self._weights(k, ds)
                self.state.set(
                    "stages",
                    stage_key,
                    {
                        "status": "COMPLETED",
                        "output_dir": str(self._stage_dir(k)),
                        "dataset": ds,
                        "train_weights": weights,
                    },
                )
                self.state.save()
                print("  -> COMPLETED (existing checkpoint)")
                continue
            if status == "SUBMITTED":
                job_id = stage["job_id"]
                slurm_status = self.slurm.check_status(job_id)
                if slurm_status in ("RUNNING", "PENDING"):
                    print(f"  -> {slurm_status} (job {job_id})")
                    return
                self.state.set("stages", stage_key, "status", "FAILED")
                self.state.save()
                print("  -> FAILED")
                return
            if status == "FAILED":
                print("  -> Stage failed. Use retry to reset.")
                return

            weights = self._weights(k, ds)
            domains = list(weights.keys())
            rma, beta = _get_stage_hps(self.state, stage_key, self.ref_model_mixup_alpha, self.beta)
            args = _sdft_train_args(
                model_name=self._prev_model(k),
                train_domains=domains,
                output_dir=str(self._stage_dir(k)),
                train_weights=[weights[d] for d in domains] if len(domains) > 1 else None,
                use_sft=self.use_sft,
                ref_model_mixup_alpha=rma,
                beta=beta,
                max_steps=FULL_TRAIN_MAX_STEPS,
                seed=_stage_seed(self.seed, k),
                eval_domains=self.ordering[:k],
            )
            prefix = "sft" if self.use_sft else "sdft"
            job_id = self.slurm.submit(
                SCRIPTS_DIR / "slurm_launch_sdft.sh",
                args,
                job_name=f"{prefix}-{self.mode}-s{k}",
                sbatch_opts=["--constraint=h200"],
            )
            if job_id is None:
                return
            if self.dry_run:
                continue
            self.state.set(
                "stages",
                stage_key,
                {
                    "status": "SUBMITTED",
                    "job_id": job_id,
                    "output_dir": str(self._stage_dir(k)),
                    "dataset": ds,
                    "train_weights": weights,
                },
            )
            self.state.save()
            print("  -> Submitted, stopping to wait")
            return
        print("\nAll stages complete!")


class ExplicitOldProbeSdftExperiment:
    def __init__(
        self,
        dry_run=False,
        use_sft=False,
        ref_model_mixup_alpha=0.01,
        beta=0.0,
        seed=None,
        stop_after=None,
        kl_reg=0.05,
        lora_probe_epochs=0.4,
    ):
        self.ordering = list(ORDERING)
        self.dry_run = dry_run
        self.use_sft = use_sft
        self.ref_model_mixup_alpha = ref_model_mixup_alpha
        self.beta = beta
        self.seed = seed
        self.stop_after = stop_after
        self.kl_reg = kl_reg
        self.lora_probe_epochs = lora_probe_epochs
        self.run_dir = RUNS_ROOT / _condition_dir(
            "opm", use_sft, ref_model_mixup_alpha, beta, seed
        )
        self.slurm = SlurmHelper(dry_run=dry_run)
        self.state = StateFile(self.run_dir / "state.json")
        if not self.state.data:
            self.state._data = {
                "condition": "sft_opm" if use_sft else "sdft_opm",
                "ordering": self.ordering,
                "base_model": BASE_MODEL,
                "version": SDFT_PROXY_VERSION,
                "kl_reg": kl_reg,
                "stages": {},
            }
            self.state.save()

    def _stage_dir(self, k):
        if k == 1:
            return self.run_dir / f"stage_1_{self.ordering[0]}"
        return self.run_dir / f"stage_{k}"

    def _prev_model(self, k):
        if k == 1:
            return BASE_MODEL
        if k == 2:
            return str(self._stage_dir(1))
        prev_stage = self.state.get("stages", str(k - 1)) or {}
        prev_full = prev_stage.get("full_train", {})
        return prev_full.get("output_dir") or str(self._stage_dir(k - 1) / "full_train")

    def _proxy_specs(self, stage_key, ds):
        specs = self.state.get("stages", stage_key, "proxy_specs")
        if specs is None or any(spec.get("version") != SDFT_PROXY_VERSION for spec in specs):
            specs = _stage_proxy_specs(ds)
            self.state.set("stages", stage_key, "proxy_specs", specs)
            self.state.save()
        return specs

    def _component_model_dir(self, stage_dir, component):
        return stage_dir / "component_models" / component

    def _proxy_model_dir(self, stage_dir, merge_id):
        return stage_dir / "proxy_models" / merge_id

    def _proxy_eval_dir(self, stage_dir, merge_id):
        return stage_dir / "proxy_eval" / merge_id

    def run(self):
        label = "SFT" if self.use_sft else "SDFT"
        print(f"=== explicit-probe {label}: {' -> '.join(self.ordering)} ===")
        for k, ds in enumerate(self.ordering, start=1):
            stage_key = str(k)
            stage = self.state.get("stages", stage_key) or {}
            status = stage.get("status", "PENDING")
            print(f"Stage {k} ({ds}): {status}")
            if status == "COMPLETED" and (
                k == 1 or stage.get("full_train", {}).get("version") == SDFT_PROXY_VERSION
            ):
                continue
            if k == 1:
                result = self._run_stage_1(stage, stage_key, ds)
            else:
                result = self._run_stage_k(k, stage, stage_key, ds)
            if result == "STOP":
                return
        print("\nAll stages complete!")

    def _run_stage_1(self, stage, stage_key, ds):
        status = stage.get("status", "PENDING")
        output_dir = str(self._stage_dir(1))
        if status != "COMPLETED" and _sdft_sentinel_exists(output_dir):
            self.state.set(
                "stages",
                stage_key,
                {
                    "status": "COMPLETED",
                    "output_dir": output_dir,
                    "dataset": ds,
                    "train_weights": {ds: 1.0},
                },
            )
            self.state.save()
            print("  -> COMPLETED (existing checkpoint)")
            return "CONTINUE"
        if status == "SUBMITTED":
            job_id = stage["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"  -> {slurm_status} (job {job_id})")
                return "STOP"
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print("  -> FAILED")
            return "STOP"
        if status == "FAILED":
            print("  -> Stage failed. Use retry to reset.")
            return "STOP"

        rma, beta = _get_stage_hps(self.state, stage_key, self.ref_model_mixup_alpha, self.beta)
        args = _sdft_train_args(
            model_name=BASE_MODEL,
            train_domains=[ds],
            output_dir=output_dir,
            use_sft=self.use_sft,
            ref_model_mixup_alpha=rma,
            beta=beta,
            max_steps=FULL_TRAIN_MAX_STEPS,
            seed=_stage_seed(self.seed, 1),
            eval_domains=[ds],
        )
        prefix = "sft" if self.use_sft else "sdft"
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_sdft.sh",
            args,
            job_name=f"{prefix}-opm-s1",
            sbatch_opts=["--constraint=h200"],
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
        print("  -> Submitted, stopping to wait")
        return "STOP"

    def _run_stage_k(self, k, stage, stage_key, ds):
        substage = stage.get("substage", "old_probe")
        stage_dir = self._stage_dir(k)
        prev_model = self._prev_model(k)
        print(f"  substage: {substage}")

        if substage in ("old_probe", "new_probe"):
            results = []
            for probe_name, new_weight in (
                ("old_probe", OLD_PROBE_NEW_WEIGHT),
                ("new_probe", NEW_PROBE_NEW_WEIGHT),
            ):
                results.append(
                    self._run_probe(
                        k, stage_key, stage_dir, prev_model, ds,
                        probe_name, new_weight,
                    )
                )
            if any(r == "FAILED" for r in results):
                return "STOP"
            if not all(r == "COMPLETED" for r in results):
                return "STOP"
            self.state.set("stages", stage_key, "substage", "proxy_merge")
            self.state.save()
            if self.stop_after in ("old_probe", "new_probe"):
                return "STOP"
            substage = "proxy_merge"
        if substage == "proxy_merge":
            result = self._run_proxy_merge(stage_key, stage_dir, prev_model, ds)
            if result != "ADVANCE":
                return result
            substage = "proxy_eval"
        if substage == "proxy_eval":
            result = self._run_proxy_eval(stage_key, stage_dir, ds)
            if result != "ADVANCE":
                return result
            if self.stop_after == "proxy_eval":
                return "STOP"
            substage = "olmix_fit"
        if substage == "olmix_fit":
            result = self._run_olmix_fit(k, stage_key, stage_dir, ds)
            if result != "ADVANCE":
                return result
            if self.stop_after == "olmix_fit":
                return "STOP"
            substage = "full_train"
        if substage == "full_train":
            return self._run_full_train(k, stage_key, stage_dir, prev_model, ds)
        return "STOP"

    def _run_probe(self, k, stage_key, stage_dir, prev_model, ds, substage, new_weight):
        state = self.state.get("stages", stage_key, substage, default={})
        status = state.get("status", "PENDING")
        output_dir = str(stage_dir / substage)
        if state.get("version") != SDFT_PROXY_VERSION and status in {"COMPLETED", "SUBMITTED", "FAILED"}:
            status = "PENDING"
        # vLLM/NCCL teardown sometimes segfaults after the adapter is already
        # written to disk. Trust the on-disk artifact over the slurm exit code.
        if status != "COMPLETED" and _sdft_sentinel_exists(
            output_dir, sentinel="adapter_model.safetensors"
        ):
            recovered = dict(state) if isinstance(state, dict) else {}
            recovered["status"] = "COMPLETED"
            recovered.setdefault("output_dir", output_dir)
            recovered["version"] = SDFT_PROXY_VERSION
            self.state.set("stages", stage_key, substage, recovered)
            self.state.save()
            print(f"    {substage} -> COMPLETED (recovered from disk)")
            return "COMPLETED"
        if status == "COMPLETED":
            return "COMPLETED"
        if status == "SUBMITTED":
            job_id = state["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"    {substage} -> {slurm_status} (job {job_id})")
                return "RUNNING"
            self.state.set("stages", stage_key, substage, "status", "FAILED")
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print(f"    {substage} -> FAILED")
            return "FAILED"
        if status == "FAILED":
            print(f"    {substage} -> FAILED. Use retry to reset.")
            return "FAILED"

        prev_stage = self.state.get("stages", str(k - 1)) or {}
        prev_weights = prev_stage.get("train_weights", {self.ordering[k - 2]: 1.0})
        probe_weights = _sdft_probe_weights(new_weight, ds, prev_weights)
        train_domains = list(probe_weights.keys())
        rma, beta = _get_stage_hps(self.state, stage_key, self.ref_model_mixup_alpha, self.beta)
        args = _sdft_train_args(
            model_name=prev_model,
            train_domains=train_domains,
            output_dir=output_dir,
            train_weights=[probe_weights[d] for d in train_domains],
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            learning_rate=4e-5,
            num_train_epochs=self.lora_probe_epochs,
            max_steps=256,
            use_sft=self.use_sft,
            ref_model_mixup_alpha=rma,
            beta=beta,
            save_steps=999999,
            seed=_stage_seed(self.seed, k),
            eval_domains=self.ordering[:k],
        )
        prefix = "sft" if self.use_sft else "sdft"
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_sdft.sh",
            args,
            job_name=f"{prefix}-opm-s{k}-{substage.replace('_', '-')}",
            sbatch_opts=["--constraint=h200"],
        )
        if job_id is None:
            return "FAILED"
        if self.dry_run:
            return "COMPLETED"
        self.state.set(
            "stages",
            stage_key,
            substage,
            {
                "status": "SUBMITTED",
                "job_id": job_id,
                "output_dir": output_dir,
                "probe_weights": probe_weights,
                "new_weight": new_weight,
                "version": SDFT_PROXY_VERSION,
            },
        )
        self.state.set("stages", stage_key, "status", "IN_PROGRESS")
        self.state.save()
        print(f"    {substage} weights: {probe_weights}")
        print(f"    {substage} -> Submitted")
        return "SUBMITTED"

    def _run_proxy_merge(self, stage_key, stage_dir, prev_model, ds):
        state = self.state.get("stages", stage_key, "proxy_merge", default={})
        status = state.get("status", "PENDING")
        if state.get("version") != SDFT_PROXY_VERSION and status in {"COMPLETED", "FAILED"}:
            status = "PENDING"
        if status == "COMPLETED":
            return "ADVANCE"
        if status == "FAILED":
            print("    proxy_merge -> FAILED. Use retry to reset.")
            return "STOP"
        specs = self._proxy_specs(stage_key, ds)
        if self.dry_run:
            print(f"    [DRY RUN] proxy_merge -> would build {len(specs)} proxies")
            return "ADVANCE"

        old_probe = self.state.get("stages", stage_key, "old_probe")
        new_probe = self.state.get("stages", stage_key, "new_probe")
        if old_probe is None or old_probe.get("status") != "COMPLETED":
            print("    proxy_merge -> missing completed old_probe")
            return "STOP"
        if new_probe is None or new_probe.get("status") != "COMPLETED":
            print("    proxy_merge -> missing completed new_probe")
            return "STOP"
        try:
            component_models = {}
            for component, probe_dir in (
                (OLD_MIX_KEY, old_probe["output_dir"]),
                (ds, new_probe["output_dir"]),
            ):
                model_dir = self._component_model_dir(stage_dir, component)
                if not _sdft_sentinel_exists(model_dir):
                    _save_sdft_proxy_model(
                        prev_model,
                        {component: probe_dir},
                        {component: 1.0},
                        model_dir,
                    )
                component_models[component] = model_dir
            for spec in specs:
                out = self._proxy_model_dir(stage_dir, spec["merge_id"])
                if _sdft_sentinel_exists(out):
                    continue
                _save_sdft_multi_merge_model(component_models, spec["weights"], out)
        except Exception as exc:
            self.state.set("stages", stage_key, "proxy_merge", {"status": "FAILED", "error": str(exc)})
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print(f"    proxy_merge -> FAILED: {exc}")
            return "STOP"

        self.state.set("stages", stage_key, "proxy_merge", {"status": "COMPLETED", "version": SDFT_PROXY_VERSION})
        self.state.set("stages", stage_key, "substage", "proxy_eval")
        self.state.save()
        print("    proxy_merge -> COMPLETED")
        return "ADVANCE"

    def _run_proxy_eval(self, stage_key, stage_dir, ds):
        specs = self._proxy_specs(stage_key, ds)
        for spec in specs:
            merge_id = spec["merge_id"]
            output_dir = str(self._proxy_eval_dir(stage_dir, merge_id))
            if _sdft_proxy_eval_sentinel_exists(output_dir):
                cur = self.state.get("stages", stage_key, "proxy_eval", merge_id, default={})
                if cur.get("status") != "COMPLETED":
                    self.state.set(
                        "stages",
                        stage_key,
                        "proxy_eval",
                        merge_id,
                        {"status": "COMPLETED", "output_dir": output_dir, "version": SDFT_PROXY_VERSION},
                    )
        self.state.save()

        pending_specs = [
            spec
            for spec in specs
            if not _sdft_proxy_eval_sentinel_exists(str(self._proxy_eval_dir(stage_dir, spec["merge_id"])))
        ]
        if not pending_specs:
            self.state.set("stages", stage_key, "substage", "olmix_fit")
            self.state.save()
            return "ADVANCE"

        batch = self.state.get("stages", stage_key, "proxy_eval_batch", default={})
        batch_status = batch.get("status", "PENDING")
        if batch.get("version") != SDFT_PROXY_VERSION and batch_status in {"SUBMITTED", "FAILED"}:
            batch_status = "PENDING"
        if batch_status == "SUBMITTED":
            job_id = batch["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"    proxy_eval -> {slurm_status} (job {job_id})")
                return "STOP"
            self.state.set(
                "stages",
                stage_key,
                "proxy_eval_batch",
                {"status": "FAILED", "job_id": job_id, "version": SDFT_PROXY_VERSION},
            )
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print("    proxy_eval -> FAILED")
            return "STOP"
        if batch_status == "FAILED":
            print("    proxy_eval -> FAILED. Use retry to reset.")
            return "STOP"

        manifest_dir = stage_dir / "proxy_eval"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "batch_manifest.tsv"
        with open(manifest_path, "w") as handle:
            for spec in pending_specs:
                merge_id = spec["merge_id"]
                model_dir = str(self._proxy_model_dir(stage_dir, merge_id))
                output_dir = str(self._proxy_eval_dir(stage_dir, merge_id))
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                handle.write(f"{model_dir}\t{output_dir}\n")

        eval_args = [
            str(manifest_path),
            "--batch_size",
            "4",
            "--max_eval_samples",
            "500",
            "--eval_accuracy",
            "--eval_domains",
            *self.ordering[: int(stage_key)],
        ]
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_batch_eval_sdft.sh",
            eval_args,
            job_name=f"sdft-opm-s{stage_key}-eval",
            sbatch_opts=["--constraint=h200"],
        )
        if job_id is None:
            return "STOP"
        if self.dry_run:
            return "ADVANCE"
        for spec in pending_specs:
            self.state.set(
                "stages",
                stage_key,
                "proxy_eval",
                spec["merge_id"],
                {
                    "status": "SUBMITTED",
                    "job_id": job_id,
                    "output_dir": str(self._proxy_eval_dir(stage_dir, spec["merge_id"])),
                    "version": SDFT_PROXY_VERSION,
                },
            )
        self.state.set(
            "stages",
            stage_key,
            "proxy_eval_batch",
            {
                "status": "SUBMITTED",
                "job_id": job_id,
                "manifest": str(manifest_path),
                "n_specs": len(pending_specs),
                "version": SDFT_PROXY_VERSION,
            },
        )
        self.state.set("stages", stage_key, "substage", "proxy_eval")
        self.state.save()
        print("    proxy_eval -> Submitted, stopping to wait")
        return "STOP"

    def _run_olmix_fit(self, k, stage_key, stage_dir, ds):
        state = self.state.get("stages", stage_key, "olmix", default={})
        status = state.get("status", "PENDING")
        if state.get("version") != SDFT_PROXY_VERSION and status in {"COMPLETED", "FAILED"}:
            status = "PENDING"
        if state.get("kl_reg", 0.05) != self.kl_reg and status in {"COMPLETED", "FAILED"}:
            status = "PENDING"
        if status == "COMPLETED":
            return "ADVANCE"
        if status == "FAILED":
            print("    olmix_fit -> FAILED. Use retry to reset.")
            return "STOP"
        if self.dry_run:
            print("    [DRY RUN] olmix_fit -> would fit from proxy eval rows")
            return "ADVANCE"

        dataset_names = [ds, OLD_MIX_KEY]
        eval_dataset_names = self.ordering[:k]
        specs = self._proxy_specs(stage_key, ds)
        ratios_rows, metrics_rows = _get_accuracy_proxy_rows(
            stage_dir / "proxy_eval", specs, dataset_names, eval_dataset_names
        )
        if not ratios_rows:
            self.state.set("stages", stage_key, "olmix", {"status": "FAILED"})
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print("    olmix_fit -> No proxy accuracy data found")
            return "STOP"

        olmix_dir = stage_dir / "olmix"
        write_csvs(
            ratios_rows, metrics_rows, dataset_names, olmix_dir, eval_dataset_names
        )
        previous_train_weights = (self.state.get("stages", str(k - 1)) or {}).get(
            "train_weights", {self.ordering[k - 2]: 1.0}
        )
        old_source_mixture = normalize_weights(previous_train_weights)
        reference_weights = {name: 1.0 / len(dataset_names) for name in dataset_names}
        relative_sizes = {
            **{
                name: reference_weights[OLD_MIX_KEY] * old_source_mixture[name]
                for name in old_source_mixture
            },
            ds: reference_weights[ds],
        }
        source_mixtures = {OLD_MIX_KEY: old_source_mixture}
        opt_weights = fit_and_propose(
            ratios_rows,
            metrics_rows,
            dataset_names,
            eval_dataset_names,
            relative_sizes=relative_sizes,
            source_mixtures=source_mixtures,
            kl_reg=self.kl_reg,
        )
        if opt_weights is None:
            self.state.set("stages", stage_key, "olmix", {"status": "FAILED"})
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print("    olmix_fit -> FAILED")
            return "STOP"

        with open(olmix_dir / "olmix_best_mix.json", "w") as handle:
            json.dump(
                {
                    "dataset_names": dataset_names,
                    "eval_dataset_names": eval_dataset_names,
                    "optimal_weights": opt_weights,
                    "reference_weights": reference_weights,
                    "relative_sizes": relative_sizes,
                    "source_mixtures": source_mixtures,
                    "kl_reg": self.kl_reg,
                },
                handle,
                indent=2,
            )
        alpha_new = opt_weights.get(ds, 0.5)
        self.state.set(
            "stages",
            stage_key,
            "olmix",
            {
                "status": "COMPLETED",
                "version": SDFT_PROXY_VERSION,
                "optimal_weights": opt_weights,
                "component_weights": opt_weights,
                "alpha_new": alpha_new,
                "old_key": OLD_MIX_KEY,
                "reference_weights": reference_weights,
                "relative_sizes": relative_sizes,
                "source_mixtures": source_mixtures,
                "kl_reg": self.kl_reg,
                "previous_train_weights": previous_train_weights,
            },
        )
        self.state.set("stages", stage_key, "substage", "full_train")
        self.state.save()
        print(f"    olmix_fit -> alpha({ds}) = {alpha_new:.4f}")
        return "ADVANCE"

    def _run_full_train(self, k, stage_key, stage_dir, prev_model, ds):
        stage = self.state.get("stages", stage_key) or {}
        state = stage.get("full_train", {})
        status = state.get("status", "PENDING")
        output_dir = str(stage_dir / "full_train")
        if state.get("version") != SDFT_PROXY_VERSION and status in {"COMPLETED", "SUBMITTED", "FAILED"}:
            status = "PENDING"
        # Same shutdown-segfault pattern as the LoRA probes: if the merged
        # checkpoint is on disk, treat the substage as COMPLETED.
        if status != "COMPLETED" and _sdft_sentinel_exists(output_dir):
            recovered = dict(state) if isinstance(state, dict) else {}
            recovered["status"] = "COMPLETED"
            recovered.setdefault("output_dir", output_dir)
            recovered["version"] = SDFT_PROXY_VERSION
            self.state.set("stages", stage_key, "full_train", recovered)
            self.state.set("stages", stage_key, "status", "COMPLETED")
            self.state.save()
            print("    full_train -> COMPLETED (recovered from disk)")
            return "CONTINUE"
        if status == "COMPLETED":
            self.state.set("stages", stage_key, "status", "COMPLETED")
            self.state.save()
            return "CONTINUE"
        if status == "SUBMITTED":
            job_id = state["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"    full_train -> {slurm_status} (job {job_id})")
                return "STOP"
            self.state.set("stages", stage_key, "full_train", "status", "FAILED")
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print("    full_train -> FAILED")
            return "STOP"
        if status == "FAILED":
            print("    full_train -> FAILED. Use retry to reset.")
            return "STOP"

        prev_stage = self.state.get("stages", str(k - 1)) or {}
        prev_weights = prev_stage.get("train_weights", {self.ordering[k - 2]: 1.0})
        olmix_state = stage.get("olmix", {})
        weights = expand_on_policy_mix_weights(
            olmix_state.get("optimal_weights", {ds: 0.5, OLD_MIX_KEY: 0.5}),
            prev_weights,
        )
        train_domains = list(weights.keys())
        rma, beta = _get_stage_hps(self.state, stage_key, self.ref_model_mixup_alpha, self.beta)
        args = _sdft_train_args(
            model_name=prev_model,
            train_domains=train_domains,
            output_dir=output_dir,
            train_weights=[weights[d] for d in train_domains],
            use_sft=self.use_sft,
            ref_model_mixup_alpha=rma,
            beta=beta,
            max_steps=FULL_TRAIN_MAX_STEPS,
            seed=_stage_seed(self.seed, k),
            eval_domains=self.ordering[:k],
        )
        prefix = "sft" if self.use_sft else "sdft"
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_sdft.sh",
            args,
            job_name=f"{prefix}-opm-s{k}-train",
            sbatch_opts=["--constraint=h200"],
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
                "version": SDFT_PROXY_VERSION,
            },
        )
        self.state.set("stages", stage_key, "train_weights", weights)
        self.state.save()
        print(f"    full_train weights: {weights}")
        print("    full_train -> Submitted, stopping to wait")
        return "STOP"


def naive(dry_run=False, ref_model_mixup_alpha=0.01, beta=0.0, seed=None):
    SequentialSdftExperiment(
        dry_run=dry_run,
        mode="naive",
        ref_model_mixup_alpha=ref_model_mixup_alpha,
        beta=beta,
        seed=seed,
    ).run()


def mix(dry_run=False, ref_model_mixup_alpha=0.01, beta=0.0, seed=None):
    SequentialSdftExperiment(
        dry_run=dry_run,
        mode="mix",
        ref_model_mixup_alpha=ref_model_mixup_alpha,
        beta=beta,
        seed=seed,
    ).run()


def opm(
    dry_run=False,
    ref_model_mixup_alpha=0.01,
    beta=0.0,
    seed=None,
    stop_after=None,
    kl_reg=0.05,
    lora_probe_epochs=0.4,
):
    ExplicitOldProbeSdftExperiment(
        dry_run=dry_run,
        ref_model_mixup_alpha=ref_model_mixup_alpha,
        beta=beta,
        seed=seed,
        stop_after=stop_after,
        kl_reg=kl_reg,
        lora_probe_epochs=lora_probe_epochs,
    ).run()


def sft_naive(dry_run=False, seed=None):
    SequentialSdftExperiment(
        dry_run=dry_run, use_sft=True, mode="naive", seed=seed
    ).run()


def sft_mix(dry_run=False, seed=None):
    SequentialSdftExperiment(
        dry_run=dry_run, use_sft=True, mode="mix", seed=seed
    ).run()


def sft_opm(dry_run=False, seed=None, stop_after=None, kl_reg=0.05):
    ExplicitOldProbeSdftExperiment(
        dry_run=dry_run,
        use_sft=True,
        seed=seed,
        stop_after=stop_after,
        kl_reg=kl_reg,
    ).run()


def olmix(**kwargs):
    return opm(**kwargs)


def sft_olmix(**kwargs):
    return sft_opm(**kwargs)


CONDITION_DIR_MAP = {
    "naive": "continual_sdft_standalone_naive",
    "mix": "continual_sdft_standalone_mix",
    "opm": "continual_sdft_standalone_opm",
    "olmix": "continual_sdft_standalone_opm",
    "sft_naive": "continual_sft_standalone_naive",
    "sft_mix": "continual_sft_standalone_mix",
    "sft_opm": "continual_sft_standalone_opm",
    "sft_olmix": "continual_sft_standalone_opm",
}


def _find_condition_dirs(condition):
    base = CONDITION_DIR_MAP.get(condition, condition)
    if not RUNS_ROOT.exists():
        return []
    return sorted(
        d for d in RUNS_ROOT.iterdir() if d.is_dir() and (d.name == base or d.name.startswith(base + "_"))
    )


def status(condition="all"):
    conditions = (
        ["naive", "mix", "opm", "sft_naive", "sft_mix", "sft_opm"]
        if condition == "all"
        else [condition]
    )
    for cond in conditions:
        roots = _find_condition_dirs(cond)
        if not roots:
            print(f"No {cond} experiments found.")
            continue
        printed = False
        for root in roots:
            state_path = root / "state.json"
            if not state_path.exists():
                continue
            printed = True
            print(f"\n{'=' * 60}")
            print(f"  {root.name}")
            print(f"{'=' * 60}")
            state = StateFile(state_path)
            ordering = state.get("ordering", default=[])
            stages = state.get("stages", default={})
            stage_strs = []
            for k in range(1, len(ordering) + 1):
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
            print(f"  [{' -> '.join(d[:6] for d in ordering)}]")
            print(f"    Stages: {' | '.join(stage_strs)}")
        if not printed:
            print(f"No {cond} state files found.")


def retry(condition, stage, substage=None, seed=None, ref_model_mixup_alpha=None, beta=None):
    if condition in CONDITION_DIR_MAP:
        dir_name = CONDITION_DIR_MAP[condition]
        if seed is not None:
            dir_name = f"{dir_name}_s{seed}"
    else:
        dir_name = condition
    root = RUNS_ROOT / dir_name
    state = StateFile(root / "state.json")
    stage_key = str(stage)
    if state.get("stages", stage_key) is None:
        print(f"Stage {stage} not found in state.")
        return

    if substage:
        if substage == "proxy_eval":
            state.set("stages", stage_key, "proxy_eval_batch", {"status": "PENDING"})
        else:
            state.set("stages", stage_key, substage, "status", "PENDING")
        state.set("stages", stage_key, "status", "IN_PROGRESS")
        state.set("stages", stage_key, "substage", substage)
    else:
        state.set("stages", stage_key, "status", "PENDING")
        stage_data = state.get("stages", stage_key) or {}
        for field in ("job_id", "output_dir"):
            stage_data.pop(field, None)

    overrides = {}
    if ref_model_mixup_alpha is not None:
        overrides["ref_model_mixup_alpha"] = ref_model_mixup_alpha
    if beta is not None:
        overrides["beta"] = beta
    if overrides:
        state.set("stages", stage_key, "hyperparam_overrides", overrides)

    state.save()
    print(
        f"Reset {condition}/stage_{stage}"
        + (f"/{substage}" if substage else "")
        + " to PENDING"
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "naive": naive,
            "mix": mix,
            "opm": opm,
            "olmix": olmix,
            "sft_naive": sft_naive,
            "sft_mix": sft_mix,
            "sft_opm": sft_opm,
            "sft_olmix": sft_olmix,
            "status": status,
            "retry": retry,
        }
    )
