"""Continual learning pipeline with an explicit old-data probe.

Usage:
    python -m pipeline.continual_opm opm [--ordering_name ord0] [--model_size 150M] [--dry_run]
    python -m pipeline.continual_opm status [--condition all]
    python -m pipeline.continual_opm retry --condition opm --ordering_name ord0 --stage 2 --substage old_probe
"""

import json
import os
from pathlib import Path

import fire

from pipeline.continual import (
    EVAL_FILES,
    MODEL_CONFIGS,
    OLD_MIX_KEY,
    ORDERINGS,
    RUNS_ROOT,
    SCRIPTS_DIR,
    SlurmHelper,
    StateFile,
    _get_lora_lr,
    _sentinel_exists,
    _train_args,
    expand_on_policy_mix_weights,
    normalize_weights,
)
from pipeline.olmix import (
    run_olmix_fit,
    write_csvs,
    write_olmix_config,
)
from pipeline.pretrain import (
    _proxy_eval_sentinel_exists,
    _save_multi_merge_model,
    _save_proxy_model,
    get_proxy_rows,
)

OLD_PROBE_NEW_WEIGHT = 0.1
NEW_PROBE_NEW_WEIGHT = 0.9
OPM_PROXY_VERSION = "opm_10_90_v2"


def _run_suffix():
    return os.environ.get("OPM_RUN_SUFFIX", "")


def _stage_proxy_specs(ds):
    """Build proxy specs for explicit old/new probes.

    ``weights`` are interpolation weights between trained probe models.
    ``ratios`` are the effective data-mixture coordinates passed to OLMix.
    """
    specs = []
    span = NEW_PROBE_NEW_WEIGHT - OLD_PROBE_NEW_WEIGHT
    for pct in range(1, 10):
        effective_new = pct / 10.0
        component_alpha = (effective_new - OLD_PROBE_NEW_WEIGHT) / span
        specs.append(
            {
                "merge_id": f"new_{effective_new:.1f}",
                "version": OPM_PROXY_VERSION,
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


class ExplicitOldProbeExperiment:
    """Continual OnPolicyMix variant with explicit old and new probes."""

    def __init__(
        self,
        ordering_name,
        model_size="150M",
        dry_run=False,
        seed=None,
        stop_after=None,
        kl_reg=0.05,
        lora_steps=2500,
    ):
        self.ordering_name = ordering_name
        self.ordering = ORDERINGS[ordering_name]
        self.model_size = model_size
        self.seed = seed
        self.stop_after = stop_after
        self.kl_reg = kl_reg
        self.lora_steps = lora_steps
        cfg = MODEL_CONFIGS[model_size]
        self.base_model = cfg["base_model"]
        self.learning_rate = cfg["learning_rate"]
        self.batch_size = cfg["batch_size"]
        self.max_steps = cfg["max_steps"]
        self.warmup_steps = cfg["warmup_steps"]
        self.sbatch_opts = cfg["sbatch_opts"]
        suffix = _run_suffix()
        self.run_dir = RUNS_ROOT / f"continual_opm{suffix}" / ordering_name / model_size
        self.slurm = SlurmHelper(dry_run=dry_run)
        self.dry_run = dry_run
        self.state = StateFile(self.run_dir / "state.json")
        if not self.state.data:
            self.state._data = {
                "condition": "opm",
                "ordering_name": ordering_name,
                "ordering": self.ordering,
                "model_size": model_size,
                "base_model": self.base_model,
                "kl_reg": kl_reg,
                "lora_steps": lora_steps,
                "stages": {},
            }
            self.state.save()

    def _stage_dir(self, k):
        if k == 1:
            return self.run_dir / f"stage_1_{self.ordering[0]}"
        return self.run_dir / f"stage_{k}"

    def _prev_checkpoint(self, k):
        if k == 1:
            return self.base_model
        if k == 2:
            return str(self._stage_dir(1))
        return str(self._stage_dir(k - 1) / "full_train")

    def _proxy_specs(self, stage_key, ds):
        specs = self.state.get("stages", stage_key, "proxy_specs")
        if specs is None or any(
            spec.get("version") != OPM_PROXY_VERSION for spec in specs
        ):
            specs = _stage_proxy_specs(ds)
            self.state.set("stages", stage_key, "proxy_specs", specs)
            self.state.save()
        return specs

    def _component_model_dir(self, stage_dir, component):
        return stage_dir / "component_models_v2" / component

    def _proxy_model_dir(self, stage_dir, merge_id):
        return stage_dir / "proxy_models_v2" / merge_id

    def _proxy_eval_dir(self, stage_dir, merge_id):
        return stage_dir / "proxy_eval_v2" / merge_id

    def run(self):
        print(
            f"=== ExplicitOldProbe {self.ordering_name}: {' -> '.join(self.ordering)} ===\n"
        )
        for k in range(1, 6):
            ds = self.ordering[k - 1]
            stage_key = str(k)
            stage = self.state.get("stages", stage_key) or {}
            status = stage.get("status", "PENDING")
            print(f"Stage {k} ({ds}): {status}")
            if status == "COMPLETED" and (
                k == 1
                or stage.get("full_train", {}).get("version") == OPM_PROXY_VERSION
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
        if status == "SUBMITTED":
            job_id = stage.get("job_id")
            slurm_status = self.slurm.check_status(job_id)
            output_dir = stage["output_dir"]
            if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                self.state.set("stages", stage_key, "status", "COMPLETED")
                self.state.save()
                print("  -> COMPLETED")
                return "CONTINUE"
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"  -> Still {slurm_status} (job {job_id})")
                return "STOP"
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print("  -> FAILED")
            return "STOP"
        if status == "FAILED":
            print("  -> Stage failed. Use 'retry' to reset.")
            return "STOP"

        output_dir = str(self._stage_dir(1))
        args = _train_args(
            model_dir=self.base_model,
            train_data_files=[f"data-mixes/{ds}_train_s1.txt"],
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
            job_name=f"opm-{self.ordering_name}-s1",
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
        print("  -> Submitted, stopping to wait")
        return "STOP"

    def _run_stage_k(self, k, stage, stage_key, ds):
        substage = stage.get("substage", "old_probe")
        stage_dir = self._stage_dir(k)
        prev_checkpoint = self._prev_checkpoint(k)
        print(f"  substage: {substage}")

        if substage in ("old_probe", "new_probe"):
            results = []
            for probe_name, new_weight in (
                ("old_probe", OLD_PROBE_NEW_WEIGHT),
                ("new_probe", NEW_PROBE_NEW_WEIGHT),
            ):
                results.append(
                    self._run_probe(
                        k, stage_key, stage_dir, prev_checkpoint, ds,
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
            result = self._run_proxy_merge(k, stage_key, stage_dir, prev_checkpoint, ds)
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
            return self._run_full_train(k, stage_key, stage_dir, prev_checkpoint, ds)
        return "STOP"

    def _run_probe(self, k, stage_key, stage_dir, prev_checkpoint, ds, probe_name, new_weight):
        state = self.state.get("stages", stage_key, probe_name, default={})
        status = state.get("status", "PENDING")
        output_dir = str(stage_dir / f"{probe_name}_v2")
        if state.get("version") != OPM_PROXY_VERSION and status in {
            "COMPLETED",
            "SUBMITTED",
            "FAILED",
        }:
            status = "PENDING"
        if status == "COMPLETED":
            return "COMPLETED"
        if status == "SUBMITTED":
            job_id = state["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status == "COMPLETED" and _sentinel_exists(
                output_dir, sentinel="adapter_model.safetensors"
            ):
                self.state.set("stages", stage_key, probe_name, "status", "COMPLETED")
                self.state.save()
                print(f"    {probe_name} -> COMPLETED")
                return "COMPLETED"
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"    {probe_name} -> {slurm_status} (job {job_id})")
                return "RUNNING"
            self.state.set("stages", stage_key, probe_name, "status", "FAILED")
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print(f"    {probe_name} -> FAILED")
            return "FAILED"
        if status == "FAILED":
            print(f"    {probe_name} -> FAILED. Use 'retry' to reset.")
            return "FAILED"

        prev_stage = self.state.get("stages", str(k - 1)) or {}
        prev_weights = prev_stage.get("train_weights", {self.ordering[k - 2]: 1.0})
        probe_weights = dict(prev_weights)
        total_old = sum(prev_weights.values())
        for dataset in list(probe_weights.keys()):
            probe_weights[dataset] = (
                (1.0 - new_weight) * probe_weights[dataset] / total_old
            )
        probe_weights[ds] = probe_weights.get(ds, 0.0) + new_weight
        train_datasets = list(probe_weights.keys())
        args = _train_args(
            model_dir=prev_checkpoint,
            train_data_files=[f"data-mixes/{d}_train_s{k}.txt" for d in train_datasets],
            output_dir=output_dir,
            train_weights=[probe_weights[d] for d in train_datasets],
            max_steps=self.lora_steps,
            learning_rate=_get_lora_lr(self.model_size),
            batch_size=self.batch_size,
            warmup_steps=0,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            save_steps=999999,
            seed=self.seed,
        )
        short_name = probe_name.replace("_", "-")
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"opm-{self.ordering_name}-s{k}-{short_name}",
            sbatch_opts=self.sbatch_opts,
        )
        if job_id is None:
            return "FAILED"
        if self.dry_run:
            return "COMPLETED"
        self.state.set(
            "stages",
            stage_key,
            probe_name,
            {
                "status": "SUBMITTED",
                "job_id": job_id,
                "output_dir": output_dir,
                "version": OPM_PROXY_VERSION,
                "probe_weights": probe_weights,
            },
        )
        self.state.set("stages", stage_key, "status", "IN_PROGRESS")
        self.state.save()
        print(f"    {probe_name} -> Submitted")
        return "SUBMITTED"

    def _run_proxy_merge(self, k, stage_key, stage_dir, prev_checkpoint, ds):
        state = self.state.get("stages", stage_key, "proxy_merge", default={})
        status = state.get("status", "PENDING")
        if state.get("version") != OPM_PROXY_VERSION and status in {
            "COMPLETED",
            "FAILED",
        }:
            status = "PENDING"
        if status == "COMPLETED" and state.get("version") == OPM_PROXY_VERSION:
            return "ADVANCE"
        if status == "FAILED":
            print("    proxy_merge -> FAILED. Use 'retry' to reset.")
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
                if not _sentinel_exists(model_dir):
                    _save_proxy_model(
                        prev_checkpoint,
                        {component: probe_dir},
                        {component: 1.0},
                        model_dir,
                    )
                component_models[component] = model_dir
            for spec in specs:
                out = self._proxy_model_dir(stage_dir, spec["merge_id"])
                if _sentinel_exists(out):
                    continue
                _save_multi_merge_model(component_models, spec["weights"], out)
        except Exception as exc:
            self.state.set(
                "stages",
                stage_key,
                "proxy_merge",
                {"status": "FAILED", "error": str(exc)},
            )
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print(f"    proxy_merge -> FAILED: {exc}")
            return "STOP"

        self.state.set(
            "stages",
            stage_key,
            "proxy_merge",
            {"status": "COMPLETED", "version": OPM_PROXY_VERSION},
        )
        self.state.set("stages", stage_key, "substage", "proxy_eval")
        self.state.save()
        print("    proxy_merge -> COMPLETED")
        return "ADVANCE"

    def _run_proxy_eval(self, stage_key, stage_dir, ds):
        specs = self._proxy_specs(stage_key, ds)
        for spec in specs:
            merge_id = spec["merge_id"]
            output_dir = str(self._proxy_eval_dir(stage_dir, merge_id))
            if _proxy_eval_sentinel_exists(output_dir):
                cur = self.state.get(
                    "stages", stage_key, "proxy_eval", merge_id, default={}
                )
                if cur.get("status") != "COMPLETED":
                    self.state.set(
                        "stages",
                        stage_key,
                        "proxy_eval",
                        merge_id,
                        {
                            "status": "COMPLETED",
                            "output_dir": output_dir,
                            "version": OPM_PROXY_VERSION,
                        },
                    )
        self.state.save()

        pending_specs = [
            spec
            for spec in specs
            if not _proxy_eval_sentinel_exists(
                str(self._proxy_eval_dir(stage_dir, spec["merge_id"]))
            )
        ]
        if not pending_specs:
            self.state.set("stages", stage_key, "substage", "olmix_fit")
            self.state.save()
            return "ADVANCE"

        batch = self.state.get("stages", stage_key, "proxy_eval_batch", default={})
        batch_status = batch.get("status", "PENDING")
        if batch.get("version") != OPM_PROXY_VERSION and batch_status in {
            "SUBMITTED",
            "FAILED",
        }:
            batch_status = "PENDING"
        if batch_status == "SUBMITTED":
            job_id = batch["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"    proxy_eval -> {slurm_status} (job {job_id})")
                return "STOP"
            for spec in pending_specs:
                self.state.set(
                    "stages",
                    stage_key,
                    "proxy_eval",
                    spec["merge_id"],
                    "status",
                    "FAILED",
                )
            self.state.set(
                "stages",
                stage_key,
                "proxy_eval_batch",
                {"status": "FAILED", "job_id": job_id},
            )
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print("    proxy_eval -> FAILED")
            return "STOP"
        if batch_status == "FAILED":
            print("    proxy_eval -> FAILED. Use 'retry' to reset.")
            return "STOP"

        manifest_dir = stage_dir / "proxy_eval_v2"
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
            str(self.batch_size * 2),
            "--eval_data_file",
            *[str(path) for path in EVAL_FILES],
        ]
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_batch_eval.sh",
            eval_args,
            job_name=f"opm-{self.ordering_name}-{stage_key}-eval",
            sbatch_opts=self.sbatch_opts,
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
                    "output_dir": str(
                        self._proxy_eval_dir(stage_dir, spec["merge_id"])
                    ),
                    "version": OPM_PROXY_VERSION,
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
                "version": OPM_PROXY_VERSION,
            },
        )
        self.state.set("stages", stage_key, "substage", "proxy_eval")
        self.state.save()
        print(f"    proxy_eval -> Submitted, stopping to wait")
        return "STOP"

    def _run_olmix_fit(self, k, stage_key, stage_dir, ds):
        state = self.state.get("stages", stage_key, "olmix", default={})
        status = state.get("status", "PENDING")
        if state.get("version") != OPM_PROXY_VERSION and status in {
            "COMPLETED",
            "FAILED",
        }:
            status = "PENDING"
        if state.get("kl_reg", 0.05) != self.kl_reg and status in {
            "COMPLETED",
            "FAILED",
        }:
            status = "PENDING"
        if status == "COMPLETED" and state.get("version") == OPM_PROXY_VERSION:
            return "ADVANCE"
        if status == "FAILED":
            print("    olmix_fit -> FAILED. Use 'retry' to reset.")
            return "STOP"

        specs = self._proxy_specs(stage_key, ds)
        if self.dry_run:
            print("    [DRY RUN] olmix_fit -> would fit from proxy eval rows")
            return "ADVANCE"

        dataset_names = [ds, OLD_MIX_KEY]
        eval_dataset_names = self.ordering[:k]
        ratios_rows, metrics_rows = get_proxy_rows(
            stage_dir / "proxy_eval_v2",
            specs,
            dataset_names,
            eval_dataset_names,
        )
        if not ratios_rows:
            self.state.set("stages", stage_key, "olmix", {"status": "FAILED"})
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print("    olmix_fit -> No proxy data found")
            return "STOP"

        olmix_dir = stage_dir / "olmix"
        ratios_file, metrics_file = write_csvs(
            ratios_rows, metrics_rows, dataset_names, olmix_dir, eval_dataset_names
        )
        config_path = olmix_dir / "fit_config.yaml"
        reference_weights = {
            name: 1.0 / len(dataset_names) for name in dataset_names
        }
        previous_train_weights = (self.state.get("stages", str(k - 1)) or {}).get(
            "train_weights",
            {self.ordering[k - 2]: 1.0},
        )
        old_source_mixture = normalize_weights(previous_train_weights)
        relative_sizes = {
            **{
                name: reference_weights[OLD_MIX_KEY] * old_source_mixture[name]
                for name in old_source_mixture
            },
            ds: reference_weights[ds],
        }
        source_mixtures = {OLD_MIX_KEY: old_source_mixture}
        write_olmix_config(
            ratios_file,
            metrics_file,
            dataset_names,
            config_path,
            relative_sizes=relative_sizes,
            source_mixtures=source_mixtures,
            kl_reg=self.kl_reg,
        )
        opt_weights = run_olmix_fit(config_path, olmix_dir / "olmix_out")
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
                "version": OPM_PROXY_VERSION,
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

    def _run_full_train(self, k, stage_key, stage_dir, prev_checkpoint, ds):
        stage = self.state.get("stages", stage_key) or {}
        state = stage.get("full_train", {})
        status = state.get("status", "PENDING")
        output_dir = str(stage_dir / "full_train")
        if state.get("version") != OPM_PROXY_VERSION and status in {
            "COMPLETED",
            "SUBMITTED",
            "FAILED",
        }:
            status = "PENDING"
        if status == "COMPLETED":
            self.state.set("stages", stage_key, "status", "COMPLETED")
            self.state.save()
            return "CONTINUE"
        if status == "SUBMITTED":
            job_id = state["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status == "COMPLETED" and _sentinel_exists(output_dir):
                self.state.set("stages", stage_key, "full_train", "status", "COMPLETED")
                self.state.set(
                    "stages", stage_key, "full_train", "version", OPM_PROXY_VERSION
                )
                self.state.set("stages", stage_key, "status", "COMPLETED")
                self.state.save()
                print("    full_train -> COMPLETED")
                return "CONTINUE"
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"    full_train -> {slurm_status} (job {job_id})")
                return "STOP"
            self.state.set("stages", stage_key, "full_train", "status", "FAILED")
            self.state.set("stages", stage_key, "status", "FAILED")
            self.state.save()
            print("    full_train -> FAILED")
            return "STOP"
        if status == "FAILED":
            print("    full_train -> FAILED. Use 'retry' to reset.")
            return "STOP"

        olmix_state = stage.get("olmix", {})
        prev_stage = self.state.get("stages", str(k - 1)) or {}
        prev_weights = prev_stage.get("train_weights", {self.ordering[k - 2]: 1.0})
        weights = expand_on_policy_mix_weights(
            olmix_state.get("optimal_weights", {ds: 0.5, OLD_MIX_KEY: 0.5}),
            prev_weights,
        )
        train_datasets = list(weights.keys())
        args = _train_args(
            model_dir=prev_checkpoint,
            train_data_files=[f"data-mixes/{d}_train_s{k}.txt" for d in train_datasets],
            output_dir=output_dir,
            train_weights=[weights[d] for d in train_datasets],
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            warmup_steps=0,
            seed=self.seed,
        )
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"opm-{self.ordering_name}-s{k}-train",
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
                "version": OPM_PROXY_VERSION,
            },
        )
        self.state.set("stages", stage_key, "train_weights", weights)
        self.state.save()
        print(f"    full_train weights: {weights}")
        print("    full_train -> Submitted, stopping to wait")
        return "STOP"


def opm(
    ordering_name="all",
    model_size="150M",
    dry_run=False,
    seed=None,
    stop_after=None,
    kl_reg=0.05,
    lora_steps=2500,
):
    orderings = list(ORDERINGS.keys()) if ordering_name == "all" else [ordering_name]
    for name in orderings:
        exp = ExplicitOldProbeExperiment(
            name,
            model_size=model_size,
            dry_run=dry_run,
            seed=seed,
            stop_after=stop_after,
            kl_reg=kl_reg,
            lora_steps=lora_steps,
        )
        exp.run()
        print()


def run(**kwargs):
    return opm(**kwargs)


def status(condition="all"):
    conditions = ["opm"] if condition == "all" else [condition]
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
    root = RUNS_ROOT / f"continual_{condition}" / ordering_name / model_size
    state = StateFile(root / "state.json")
    stage_key = str(stage)
    stage_data = state.get("stages", stage_key)
    if stage_data is None:
        print(f"Stage {stage} not found in state.")
        return
    if substage:
        if substage == "proxy_eval":
            state.set("stages", stage_key, "proxy_eval_batch", {"status": "PENDING"})
            for spec in state.get("stages", stage_key, "proxy_specs", default=[]):
                state.set(
                    "stages",
                    stage_key,
                    "proxy_eval",
                    spec["merge_id"],
                    "status",
                    "PENDING",
                )
        else:
            state.set("stages", stage_key, substage, "status", "PENDING")
        state.set("stages", stage_key, "status", "IN_PROGRESS")
        state.set("stages", stage_key, "substage", substage)
    else:
        state.set("stages", stage_key, "status", "PENDING")
        for key, val in (state.get("stages", stage_key) or {}).items():
            if isinstance(val, dict) and "status" in val:
                state.set("stages", stage_key, key, "status", "PENDING")
    state.save()
    print(
        f"Reset {condition}/{ordering_name}/stage_{stage}"
        + (f"/{substage}" if substage else "")
        + " to PENDING"
    )


if __name__ == "__main__":
    fire.Fire({"opm": opm, "run": run, "status": status, "retry": retry})
