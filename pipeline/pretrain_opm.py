"""Pretraining pipeline with explicit old-data probes at each OPM stage.

Usage:
    python -m pipeline.pretrain_opm opm [--model_size 150M] [--stage_1_steps 10000]
    python -m pipeline.pretrain_opm cpt runs/pretrain_opm/150M/10000_steps/seed_none/stage_3
    python -m pipeline.pretrain_opm status [--algorithm all]
    python -m pipeline.pretrain_opm retry --algorithm pretrain_opm --model_size 150M --max_steps 20000 --substage probe_train --item old
"""

import json
import os
from pathlib import Path

import fire

from pipeline.continual import (
    RUNS_ROOT,
    SCRIPTS_DIR,
    StateFile,
    _sentinel_exists,
    normalize_weights,
)
from pipeline.olmix import (
    run_olmix_fit,
    write_csvs,
    write_olmix_config,
)
from pipeline.pretrain import (
    DEFAULT_DATASETS,
    MODEL_CONFIGS,
    ExperimentBase,
    _build_train_args,
    _default_eval_files,
    _default_train_files,
    _probe_weights_with_old_mix,
    _proxy_eval_sentinel_exists,
    _sample_proxy_weights,
    _save_multi_merge_model,
    _save_proxy_model,
    get_proxy_rows,
)


def _run_suffix():
    return os.environ.get("OPM_RUN_SUFFIX", "")


class ExplicitOldProbeExperiment(ExperimentBase):
    """OnPolicyMix variant with an explicit old-data LoRA probe."""

    algorithm_name = "pretrain_opm"
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
        lora_steps=5000,
        seed=None,
        dry_run=False,
        run_dir=None,
        stage_id=None,
        stop_after=None,
        kl_reg=0.05,
    ):
        self.prefix_fraction = prefix_fraction
        self.lora_steps = lora_steps
        self._base_model_dir_override = base_model_dir
        self.continual_mode = base_model_dir is not None
        self.stage_id = stage_id
        self.stop_after = stop_after
        self.kl_reg = kl_reg
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

        if old_weights is None:
            uniform = 1.0 / len(self.datasets)
            self.old_weights = {d: uniform for d in self.datasets}
        else:
            self.old_weights = {d: float(old_weights[d]) for d in self.datasets}

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

        if run_dir is not None:
            self.run_dir = Path(run_dir)
            self.state = StateFile(self.run_dir / "state.json")

        if not self.state.data:
            init_substage = "probe_train" if self.continual_mode else "prefix_train"
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
                "lora_steps": lora_steps,
                "seed": seed,
                "kl_reg": kl_reg,
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
        run_dir = RUNS_ROOT / f"continual_pretrain_opm{_run_suffix()}" / tag
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
        return max(1, int(round(self.max_steps * self.prefix_fraction)))

    @property
    def final_steps(self):
        if self.continual_mode:
            return self.max_steps
        return max(1, self.max_steps - self.prefix_steps)

    @property
    def lora_probe_steps(self):
        return self.lora_steps

    @property
    def prefix_dir(self):
        return self.run_dir / "prefix_train"

    @property
    def base_model_dir(self):
        if self._base_model_dir_override is not None:
            return self._base_model_dir_override
        return str(self.prefix_dir)

    @property
    def proxy_specs(self):
        return self.state.get("proxy_specs", default=[])

    def _old_probe_dir(self):
        return self.run_dir / "probe_train" / self.OLD_KEY

    def _dataset_probe_dir(self, dataset):
        return self.run_dir / "probe_train" / dataset

    def _component_model_dir(self, component):
        return self.run_dir / "component_models" / component

    def _proxy_model_dir(self, merge_id):
        return self.run_dir / "proxy_merges" / merge_id

    def _proxy_eval_dir(self, merge_id):
        return self.run_dir / "proxy_eval" / merge_id

    def _union_datasets(self):
        seen = set()
        out = []
        for dataset in (*self.datasets, *self.new_datasets):
            if dataset not in seen:
                seen.add(dataset)
                out.append(dataset)
        return out

    def _union_train_file(self, dataset):
        return self.train_files.get(dataset) or self.new_train_files[dataset]

    def _union_eval_file(self, dataset):
        return self.eval_files.get(dataset) or self.new_eval_files[dataset]

    def _eval_file_list(self):
        return [self._union_eval_file(d) for d in self._union_datasets()]

    def compute_expanded_weights(self, opt_weights=None):
        if opt_weights is None:
            opt_weights = self.state.get("olmix", "optimal_weights", default=None)
        if opt_weights is None:
            return None
        if self.OLD_KEY not in opt_weights:
            # Weights already expanded by olmix (source_mixtures was used)
            return {d: opt_weights.get(d, 0.0) for d in self._union_datasets()}
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
            f"=== ExplicitOldProbe{stage_tag} [{mode}] ({self.model_size}, {self.max_steps} steps) ==="
        )
        default_substage = "probe_train" if self.continual_mode else "prefix_train"
        substage = self.state.get("substage", default=default_substage)

        if substage == "prefix_train":
            if self.continual_mode:
                substage = "probe_train"
            else:
                result = self._run_prefix_train()
                if result != "ADVANCE":
                    return result
                if self.stop_after == "prefix_train":
                    return "ADVANCE"
                substage = "probe_train"
        if substage == "probe_train":
            result = self._run_probe_train()
            if result != "ADVANCE":
                return result
            if self.stop_after == "probe_train":
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
                self.state.set("substage", "probe_train")
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

        args = _build_train_args(
            model_dir=self.model_id,
            model_init_mode="config",
            train_data_files=[self.train_files[d] for d in self.datasets],
            output_dir=output_dir,
            eval_data_files=self._eval_file_list(),
            train_weights=[self.old_weights[d] for d in self.datasets],
            max_steps=self.prefix_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            warmup_steps=int(round(self.warmup_steps * self.prefix_fraction)),
            seed=self.seed,
        )
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"opm-prefix-{self.model_size}-{self.max_steps}",
            sbatch_opts=self.sbatch_opts,
        )
        if job_id is None:
            return "STOP"
        if self.dry_run:
            return "ADVANCE"
        self.state.set(
            "prefix_train",
            {"status": "SUBMITTED", "job_id": job_id, "output_dir": output_dir},
        )
        self.state.set("substage", "prefix_train")
        self.state.set("status", "IN_PROGRESS")
        self.state.save()
        print("  prefix_train -> Submitted")
        return "STOP"

    def _run_probe_train(self):
        probe_state = self.state.get("probe_train", default={})
        any_running = False

        def submit_probe(component, train_files, train_weights, output_dir):
            args = _build_train_args(
                model_dir=self.base_model_dir,
                model_init_mode="pretrained",
                train_data_files=train_files,
                output_dir=output_dir,
                eval_data_files=self._eval_file_list(),
                train_weights=train_weights,
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
                job_name=f"opm-lora-{component}-{self.model_size}-{self.max_steps}",
                sbatch_opts=self.lora_sbatch_opts,
            )
            if job_id is None:
                return None
            if not self.dry_run:
                self.state.set(
                    "probe_train",
                    component,
                    {"status": "SUBMITTED", "job_id": job_id, "output_dir": output_dir},
                )
                self.state.set("substage", "probe_train")
                self.state.set("status", "IN_PROGRESS")
                self.state.save()
                print(f"  probe_train[{component}] -> Submitted")
            return job_id

        components = [self.OLD_KEY] + self.new_datasets
        for component in components:
            item = probe_state.get(component, {})
            status = item.get("status", "PENDING")
            output_dir = str(
                self._old_probe_dir()
                if component == self.OLD_KEY
                else self._dataset_probe_dir(component)
            )

            if status == "COMPLETED":
                continue

            if status == "SUBMITTED":
                job_id = item["job_id"]
                slurm_status = self.slurm.check_status(job_id)
                if slurm_status == "COMPLETED" and _sentinel_exists(
                    output_dir, sentinel="adapter_model.safetensors"
                ):
                    self.state.set("probe_train", component, "status", "COMPLETED")
                    self.state.save()
                    print(f"  probe_train[{component}] -> COMPLETED")
                    continue
                if slurm_status in ("RUNNING", "PENDING"):
                    any_running = True
                    print(
                        f"  probe_train[{component}] -> {slurm_status} (job {job_id})"
                    )
                    continue
                self.state.set("probe_train", component, "status", "FAILED")
                self.state.set("status", "FAILED")
                self.state.save()
                print(f"  probe_train[{component}] -> FAILED")
                return "STOP"

            if status == "FAILED":
                print(f"  probe_train[{component}] -> FAILED. Use retry to reset.")
                return "STOP"

            if component == self.OLD_KEY:
                train_datasets = list(self.old_weights.keys())
                train_files = [self.train_files[d] for d in train_datasets]
                train_weights = [self.old_weights[d] for d in train_datasets]
            else:
                probe_weights = _probe_weights_with_old_mix(
                    0.9, component, self.old_weights
                )
                train_datasets = list(probe_weights.keys())
                train_files = [self._union_train_file(d) for d in train_datasets]
                train_weights = [probe_weights[d] for d in train_datasets]

            if submit_probe(component, train_files, train_weights, output_dir) is None:
                return "STOP"

        if self.dry_run:
            print(
                f"  [DRY RUN] probe_train -> would launch {len(components)} LoRA jobs"
            )
            return "ADVANCE"

        probe_state = self.state.get("probe_train", default={})
        all_done = all(
            probe_state.get(component, {}).get("status") == "COMPLETED"
            for component in components
        )
        if all_done:
            self.state.set("substage", "proxy_merge")
            self.state.save()
            return "ADVANCE"
        if any_running:
            return "STOP"
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

        probe_dirs = {}
        components = [self.OLD_KEY] + self.new_datasets
        for component in components:
            item = self.state.get("probe_train", component)
            if item is None or item.get("status") != "COMPLETED":
                print(f"  proxy_merge -> missing completed probe for {component}")
                self.state.set("proxy_merge", {"status": "FAILED"})
                self.state.set("status", "FAILED")
                self.state.save()
                return "STOP"
            probe_dirs[component] = item["output_dir"]

        try:
            component_models = {}
            for component in components:
                model_dir = self._component_model_dir(component)
                if not _sentinel_exists(model_dir):
                    _save_proxy_model(
                        self.base_model_dir,
                        {component: probe_dirs[component]},
                        {component: 1.0},
                        model_dir,
                    )
                component_models[component] = model_dir

            for spec in self.proxy_specs:
                out = self._proxy_model_dir(spec["merge_id"])
                if _sentinel_exists(out):
                    continue
                _save_multi_merge_model(component_models, spec["weights"], out)
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
            return "ADVANCE"

        batch = self.state.get("proxy_eval_batch", default={})
        batch_status = batch.get("status", "PENDING")
        if batch_status == "SUBMITTED":
            job_id = batch["job_id"]
            slurm_status = self.slurm.check_status(job_id)
            if slurm_status in ("RUNNING", "PENDING"):
                print(f"  proxy_eval[batch] -> {slurm_status} (job {job_id})")
                return "STOP"
            for spec in pending_specs:
                self.state.set("proxy_eval", spec["merge_id"], "status", "FAILED")
            self.state.set("proxy_eval_batch", {"status": "FAILED", "job_id": job_id})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  proxy_eval[batch] -> FAILED")
            return "STOP"
        if batch_status == "FAILED":
            print("  proxy_eval[batch] -> FAILED. Use retry to reset.")
            return "STOP"

        manifest_dir = self.run_dir / "proxy_eval"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "batch_manifest.tsv"
        with open(manifest_path, "w") as handle:
            for spec in pending_specs:
                merge_id = spec["merge_id"]
                model_dir = str(self._proxy_model_dir(merge_id))
                output_dir = str(self._proxy_eval_dir(merge_id))
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                handle.write(f"{model_dir}\t{output_dir}\n")

        eval_args = [
            str(manifest_path),
            "--batch_size",
            str(self.batch_size),
            "--eval_data_file",
            *[str(p) for p in self._eval_file_list()],
        ]
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_batch_eval.sh",
            eval_args,
            job_name=f"opm-eval-batch-{self.model_size}-{self.max_steps}",
            sbatch_opts=self.sbatch_opts,
        )
        if job_id is None:
            return "STOP"
        if self.dry_run:
            print(
                f"  [DRY RUN] proxy_eval -> would launch 1 batch job for {len(pending_specs)} proxies"
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
        print(f"  proxy_eval[batch] -> Submitted job {job_id}")
        return "STOP"

    def _run_olmix_fit(self):
        olmix_state = self.state.get("olmix", default={})
        status = olmix_state.get("status", "PENDING")
        if olmix_state.get("kl_reg", 0.05) != self.kl_reg and status in {
            "COMPLETED",
            "FAILED",
        }:
            status = "PENDING"
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
        total = len(self.datasets) + len(self.new_datasets)
        reference_weights = {
            self.OLD_KEY: len(self.datasets) / total,
            **{d: 1.0 / total for d in self.new_datasets},
        }
        old_source_mixture = normalize_weights(self.old_weights)
        if self.continual_mode:
            relative_sizes = {d: 1.0 / total for d in self._union_datasets()}
            source_mixtures = {self.OLD_KEY: old_source_mixture}
        else:
            relative_sizes = dict(reference_weights)
            source_mixtures = None
        config_path = olmix_dir / "fit_config.yaml"
        write_olmix_config(
            ratios_file,
            metrics_file,
            fit_dataset_names,
            config_path,
            relative_sizes=relative_sizes,
            source_mixtures=source_mixtures,
            kl_reg=self.kl_reg,
        )
        opt_weights = run_olmix_fit(config_path, olmix_dir / "olmix_out")
        if opt_weights is None:
            self.state.set("olmix", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  olmix_fit -> FAILED")
            return "STOP"

        with open(olmix_dir / "olmix_best_mix.json", "w") as handle:
            json.dump(
                {
                    "dataset_names": fit_dataset_names,
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
        self.state.set(
            "olmix",
            {
                "status": "COMPLETED",
                "optimal_weights": opt_weights,
                "component_weights": opt_weights,
                "reference_weights": reference_weights,
                "relative_sizes": relative_sizes,
                "source_mixtures": source_mixtures,
                "kl_reg": self.kl_reg,
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

        expanded = self.compute_expanded_weights()
        if expanded is None:
            self.state.set("final_train", {"status": "FAILED"})
            self.state.set("status", "FAILED")
            self.state.save()
            print("  final_train -> missing optimized weights")
            return "STOP"

        all_datasets = self._union_datasets()
        args = _build_train_args(
            model_dir=self.base_model_dir,
            model_init_mode="pretrained",
            train_data_files=[self._union_train_file(d) for d in all_datasets],
            output_dir=output_dir,
            eval_data_files=[self._union_eval_file(d) for d in all_datasets],
            train_weights=[expanded[d] for d in all_datasets],
            max_steps=self.final_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            warmup_steps=0,
            eval_steps=self.eval_steps,
            seed=self.seed,
        )
        job_id = self.slurm.submit(
            SCRIPTS_DIR / "slurm_launch_train.sh",
            args,
            job_name=f"opm-final-{self.model_size}-{self.max_steps}",
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


class ExplicitOldProbeChain:
    """Two-stage explicit-old-probe pretraining chain.

    Stage 1: prefix_fraction=0.2 ERM prefix (10k steps at 150M) then probes +
             olmix + optimized final train (40k steps at 150M).
             max_steps=50k → 10k prefix + 40k final.
    Stage 2: prefix_fraction=0.0, continues from stage 1 final_train.
             Adds tulu_flan_v0/v2 via probes + olmix + final train (10k steps).
    """

    def __init__(
        self,
        model_size="150M",
        stage_1_steps=None,
        stage_2_steps=10000,
        datasets=None,
        train_files=None,
        eval_files=None,
        new_datasets=None,
        new_train_files=None,
        new_eval_files=None,
        proxy_count=20,
        lora_steps=5000,
        dry_run=False,
        seed=None,
        kl_reg=0.05,
    ):
        default_steps = MODEL_CONFIGS[model_size]["max_steps"]
        self.model_size = model_size
        self.stage_1_steps = (
            stage_1_steps if stage_1_steps is not None else default_steps
        )
        self.stage_2_steps = stage_2_steps
        self.datasets = list(datasets) if datasets else list(DEFAULT_DATASETS)
        self.train_files = train_files or _default_train_files(self.datasets)
        self.eval_files = eval_files or _default_eval_files(self.datasets)
        self.new_datasets = (
            list(new_datasets) if new_datasets else ["tulu_flan_v0", "tulu_flan_v2"]
        )
        self.new_train_files = new_train_files or {
            d: f"data-mixes/{d}.txt" for d in self.new_datasets
        }
        self.new_eval_files = new_eval_files or {
            d: f"data-mixes/{d}_eval.txt" for d in self.new_datasets
        }
        self.proxy_count = proxy_count
        self.lora_steps = lora_steps
        self.dry_run = dry_run
        self.seed = seed
        self.kl_reg = kl_reg
        seed_suffix = f"seed_{seed}" if seed is not None else "seed_none"
        self.run_dir = (
            RUNS_ROOT
            / f"pretrain_opm{_run_suffix()}"
            / model_size
            / f"{stage_1_steps}_steps"
            / seed_suffix
        )
        self.state = StateFile(self.run_dir / "state.json")
        if not self.state.data:
            self.state._data = {
                "condition": "pretrain_opm",
                "chain": True,
                "model_size": model_size,
                "stage_1_steps": stage_1_steps,
                "stage_2_steps": stage_2_steps,
                "datasets": self.datasets,
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "new_datasets": self.new_datasets,
                "new_train_files": self.new_train_files,
                "new_eval_files": self.new_eval_files,
                "proxy_count": proxy_count,
                "lora_steps": lora_steps,
                "seed": seed,
                "kl_reg": kl_reg,
                "status": "PENDING",
                "stages": {},
            }
            self.state.save()

    def _stage_1(self):
        k = len(self.datasets)
        uniform = {d: 1.0 / k for d in self.datasets}
        return ExplicitOldProbeExperiment(
            model_size=self.model_size,
            max_steps=self.stage_1_steps,
            datasets=self.datasets,
            train_files=self.train_files,
            eval_files=self.eval_files,
            old_weights=uniform,
            new_datasets=self.datasets,
            new_train_files=self.train_files,
            new_eval_files=self.eval_files,
            proxy_count=self.proxy_count,
            prefix_fraction=0.2,
            lora_steps=self.lora_steps,
            seed=self.seed,
            dry_run=self.dry_run,
            run_dir=self.run_dir / "stage_1",
            stage_id="stage_1",
            kl_reg=self.kl_reg,
        )

    def _stage_2(self, old_weights):
        return ExplicitOldProbeExperiment(
            model_size=self.model_size,
            max_steps=self.stage_2_steps,
            datasets=self.datasets,
            train_files=self.train_files,
            eval_files=self.eval_files,
            old_weights=old_weights,
            base_model_dir=str(self.run_dir / "stage_1" / "final_train"),
            new_datasets=self.new_datasets,
            new_train_files=self.new_train_files,
            new_eval_files=self.new_eval_files,
            proxy_count=self.proxy_count,
            prefix_fraction=0.0,
            lora_steps=self.lora_steps,
            seed=self.seed,
            dry_run=self.dry_run,
            run_dir=self.run_dir / "stage_2",
            stage_id="stage_2",
            kl_reg=self.kl_reg,
        )

    def _save_chain(self):
        if not self.dry_run:
            self.state.save()

    def run(self):
        print(
            f"=== ExplicitOldProbe Chain ({self.model_size}, "
            f"{self.stage_1_steps}+{self.stage_2_steps} steps) ==="
        )
        self.state.set("active_stage", 1)
        self._save_chain()
        s1 = self._stage_1()
        s1.run()
        s1_done = (
            s1.state.get("final_train", "status", default="PENDING") == "COMPLETED"
        )
        if not s1_done and not self.dry_run:
            return
        uniform = {d: 1.0 / len(self.datasets) for d in self.datasets}
        s1_final = s1.compute_expanded_weights() if s1_done else uniform
        self.state.set(
            "stages",
            "stage_1",
            {
                "status": "COMPLETED" if s1_done else "PENDING",
                "base_model_dir": str(self.run_dir / "stage_1" / "final_train"),
                "final_weights": s1_final,
            },
        )
        self._save_chain()

        self.state.set("active_stage", 2)
        self._save_chain()
        s2 = self._stage_2(s1_final)
        s2.run()
        s2_done = (
            s2.state.get("final_train", "status", default="PENDING") == "COMPLETED"
        )
        if not s2_done and not self.dry_run:
            return
        s2_final = s2.compute_expanded_weights() if s2_done else None
        self.state.set(
            "stages",
            "stage_2",
            {
                "status": "COMPLETED" if s2_done else "PENDING",
                "base_model_dir": str(self.run_dir / "stage_2" / "final_train"),
                "final_weights": s2_final,
            },
        )
        if s2_done:
            self.state.set("status", "COMPLETED")
        self._save_chain()


def opm(
    model_size="150M",
    stage_1_steps=None,
    stage_2_steps=10000,
    datasets=None,
    train_files=None,
    eval_files=None,
    new_datasets=None,
    new_train_files=None,
    new_eval_files=None,
    proxy_count=12,
    lora_steps=5000,
    dry_run=False,
    seed=None,
    kl_reg=0.05,
):
    if isinstance(datasets, (list, tuple)):
        datasets = list(datasets)
    if isinstance(new_datasets, (list, tuple)):
        new_datasets = list(new_datasets)
    chain = ExplicitOldProbeChain(
        model_size=model_size,
        stage_1_steps=stage_1_steps,
        stage_2_steps=stage_2_steps,
        datasets=datasets,
        train_files=train_files,
        eval_files=eval_files,
        new_datasets=new_datasets,
        new_train_files=new_train_files,
        new_eval_files=new_eval_files,
        proxy_count=proxy_count,
        lora_steps=lora_steps,
        dry_run=dry_run,
        seed=seed,
        kl_reg=kl_reg,
    )
    chain.run()


def cpt(
    pretrain_run_dir,
    model_size="150M",
    max_steps=None,
    new_datasets=None,
    new_train_files=None,
    new_eval_files=None,
    proxy_count=12,
    lora_steps=5000,
    dry_run=False,
    seed=None,
    kl_reg=0.05,
):
    if isinstance(new_datasets, (list, tuple)):
        new_datasets = list(new_datasets)
    exp = ExplicitOldProbeExperiment.from_pretrain_run(
        pretrain_run_dir=pretrain_run_dir,
        new_datasets=new_datasets,
        new_train_files=new_train_files,
        new_eval_files=new_eval_files,
        model_size=model_size,
        max_steps=max_steps,
        proxy_count=proxy_count,
        lora_steps=lora_steps,
        dry_run=dry_run,
        seed=seed,
        kl_reg=kl_reg,
    )
    exp.run()


def status(algorithm="all", model_size="all", max_steps="all"):
    algorithms = (
        ["pretrain_opm", "continual_pretrain_opm"]
        if algorithm == "all"
        else [algorithm]
    )

    def is_chain_state(data):
        return data.get("chain") or (
            "stage_1_steps" in data and "stage_2_steps" in data and "stages" in data
        )

    for algo in algorithms:
        root = RUNS_ROOT / algo
        if not root.exists():
            print(f"No {algo} experiments found.")
            continue
        print(f"\n{algo}:")
        all_state_paths = sorted(root.glob("**/state.json"))
        chain_dirs = set()
        for state_path in all_state_paths:
            try:
                data = StateFile(state_path).data
            except Exception:
                continue
            if is_chain_state(data):
                chain_dirs.add(state_path.parent)
        for state_path in all_state_paths:
            if any(
                chain_dir in state_path.parents and state_path.parent != chain_dir
                for chain_dir in chain_dirs
            ):
                continue
            state = StateFile(state_path)
            size = state.get("model_size", default="?")
            steps = state.get(
                "max_steps", default=state.get("stage_1_steps", default="?")
            )
            seed = state.get("seed", default="?")
            if model_size != "all" and size != model_size:
                continue
            if max_steps != "all" and str(steps) != str(max_steps):
                continue
            overall_status = state.get("status", default="PENDING")
            if is_chain_state(state.data):
                active = state.get("active_stage", default="?")
                print(
                    f"  {size}/{steps}/seed_{seed}: {overall_status} (chain, active=stage_{active})"
                )
                for stage_name in ("stage_1", "stage_2"):
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
    seed=None,
    substage="prefix_train",
    item=None,
):
    seed_suffix = f"seed_{seed}" if seed is not None else "seed_none"
    root = RUNS_ROOT / algorithm / model_size / f"{max_steps}_steps" / seed_suffix
    state = StateFile(root / "state.json")
    if not state.data:
        print(f"No state found for {algorithm}/{model_size}/{max_steps}/{seed_suffix}")
        return

    if substage in {"prefix_train", "proxy_merge", "olmix", "final_train"}:
        state.set(substage, "status", "PENDING")
        state.set("substage", substage if substage != "olmix" else "olmix_fit")
    elif substage == "proxy_eval":
        if item is None:
            state.set("proxy_eval", {})
            state.set("proxy_eval_batch", {})
        else:
            state.set("proxy_eval", item, "status", "PENDING")
            state.set("proxy_eval_batch", {})
        state.set("substage", "proxy_eval")
    elif substage == "probe_train":
        if item is None:
            print("retry for probe_train requires --item")
            return
        state.set("probe_train", item, "status", "PENDING")
        state.set("substage", "probe_train")
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
            "opm": opm,
            "cpt": cpt,
            "status": status,
            "retry": retry,
        }
    )
