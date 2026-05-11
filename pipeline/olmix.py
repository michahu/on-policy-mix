"""olmix driver: convert sweep runs to olmix CSV format and run olmix fit.

Usage (single-sweep, e.g. runs/continual_sft_mix):
    python -m pipeline.olmix main runs/continual_sft_mix \\
        --dataset_names algebraic_stack,arxiv \\
        [--prefix ""] [--model_size 150M]

Run olmix fit directly:
    python -m pipeline.olmix fit config.yaml olmix_out/

Compare merge-proxy vs sweep-proxy:
    python -m pipeline.olmix compare [--merges_dir ...] [--runs_dir ...]
"""

import csv
import json
import math
import os
import pickle
import re
import subprocess
from pathlib import Path

import fire
import yaml

DEFAULT_OLMIX_REPO = os.environ.get(
    "OLMIX_REPO",
    str(Path(__file__).resolve().parents[1] / "olmix"),
)


# ---------------------------------------------------------------------------
# Data collection: runs source
# ---------------------------------------------------------------------------


def parse_weights(dir_name, prefix, dataset_names):
    """Parse mixture weights from a run directory name.

    Examples:
        "lora-w0.6-.4", prefix="lora", dataset_names=["arxiv", "reddit"] -> {"arxiv": 0.6, "reddit": 0.4}
        "lora",         prefix="lora", dataset_names=["arxiv"]            -> {"arxiv": 1.0}
        "arxiv_p30",    prefix="",     dataset_names=["algebraic_stack", "arxiv"] -> {"algebraic_stack": 0.7, "arxiv": 0.3}

    Returns:
        dict mapping dataset_name -> weight, or None if parsing fails.
    """
    weight_str = dir_name.replace(prefix, "").lstrip("-")

    if not weight_str or weight_str == "equal":
        n = len(dataset_names)
        weights_list = [round(1.0 / n, 6)] * n
        return dict(zip(dataset_names, weights_list))

    # Try {dataset}_p{N} pattern (e.g. "arxiv_p30" -> arxiv gets 30%, others split remainder)
    m = re.match(r"^(.+)_p(\d+)$", weight_str)
    if m:
        ds_name, pct = m.group(1), int(m.group(2)) / 100.0
        if ds_name in dataset_names:
            n_others = len(dataset_names) - 1
            other_weight = (1.0 - pct) / n_others if n_others > 0 else 0.0
            return {
                ds: (pct if ds == ds_name else other_weight) for ds in dataset_names
            }

    # e.g. "w0.6-.4" or "w0.5-.5"
    raw = weight_str.lstrip("w").split("-")
    try:
        weights_list = [float(w) for w in raw]
    except ValueError:
        return None
    total = sum(weights_list)
    if total > 0:
        weights_list = [w / total for w in weights_list]

    if len(weights_list) != len(dataset_names):
        return None

    return dict(zip(dataset_names, weights_list))


def get_eval_losses_at_step(run_dir, dataset_names, step):
    """Extract per-dataset eval losses at a specific step from the last checkpoint's trainer_state.json.

    Returns:
        dict mapping dataset_name -> eval_loss, or None if not available.
    """
    ckpts = sorted(
        [
            d
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpts:
        return None

    state_file = ckpts[-1] / "trainer_state.json"
    if not state_file.exists():
        return None

    with open(state_file) as f:
        state = json.load(f)

    losses = {}
    for ds in dataset_names:
        key = f"eval_{ds}_loss"
        for entry in state["log_history"]:
            if entry.get("step") == step and key in entry:
                losses[ds] = entry[key]
                break

    return losses if losses else None


def get_final_eval_losses(run_dir, dataset_names):
    """Extract per-dataset eval losses from the final checkpoint's trainer_state.json.

    Returns:
        dict mapping dataset_name -> eval_loss, or None if not available.
    """
    ckpts = sorted(
        [
            d
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpts:
        return None

    state_file = ckpts[-1] / "trainer_state.json"
    if not state_file.exists():
        return None

    with open(state_file) as f:
        state = json.load(f)

    final_step = state["global_step"]
    losses = {}
    for ds in dataset_names:
        key = f"eval_{ds}_loss"
        for entry in reversed(state["log_history"]):
            if entry.get("step") == final_step and key in entry:
                losses[ds] = entry[key]
                break

    return losses if losses else None


def get_run_rows(sweep_dir, prefix, dataset_names):
    """Extract ratios/metrics rows from a training-sweep directory.

    Scans all {prefix}*/ subdirs, parses weights from the dir name, and
    reads per-dataset eval losses from the final checkpoint's trainer_state.json.

    Returns:
        (ratios_rows, metrics_rows, n_skipped) — lists of dicts ready for
        csv.DictWriter, plus the number of subdirs skipped.
    """
    run_dirs = sorted(
        [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    )

    ratios_rows = []
    metrics_rows = []
    skipped = 0

    for run_dir in run_dirs:
        run_id = run_dir.name

        weights = parse_weights(run_id, prefix, dataset_names)
        if weights is None:
            skipped += 1
            continue

        losses = get_final_eval_losses(run_dir, dataset_names)
        if not losses:
            skipped += 1
            continue

        ratios_rows.append({"run": run_id, **{ds: weights[ds] for ds in dataset_names}})
        metrics_rows.append(
            {
                "run": run_id,
                **{
                    f"eval_{ds}_loss": losses[ds]
                    for ds in dataset_names
                    if ds in losses
                },
            }
        )

    return ratios_rows, metrics_rows, skipped


# ---------------------------------------------------------------------------
# Data collection: merges source
# ---------------------------------------------------------------------------


def get_merge_rows(merge_pair_dir, dataset_names, eval_dataset_names=None):
    """Extract ratios/metrics rows from a merge-eval directory.

    Reads linear_connectivity_results.json from the latest checkpoint.
    Each alpha value becomes one proxy run. Skips pure single-dataset endpoints.

    Args:
        merge_pair_dir: Directory containing checkpoint-* subdirs.
        dataset_names: 2-element list for the merge components (alpha weights).
        eval_dataset_names: Datasets to extract eval losses for. Defaults to dataset_names.

    Returns:
        (ratios_rows, metrics_rows) lists, each entry a dict ready for csv.DictWriter.
    """
    if eval_dataset_names is None:
        eval_dataset_names = dataset_names
    ckpts = sorted(
        [
            d
            for d in merge_pair_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpts:
        print(f"No checkpoint directories found in {merge_pair_dir}")
        return [], []

    results_file = ckpts[-1] / "linear_connectivity_results.json"
    if not results_file.exists():
        print(f"linear_connectivity_results.json not found in {ckpts[-1]}")
        return [], []

    with open(results_file) as f:
        results = json.load(f)

    ratios_rows = []
    metrics_rows = []

    for alpha_key, data in sorted(results.items()):
        alpha = data["alpha"]
        per_ds = data["per_dataset_results"]

        if alpha < 1e-6 or alpha > 1.0 - 1e-6:
            print(f"  SKIP alpha={alpha:.3f}: single-dataset endpoint")
            continue

        losses = {
            ds: per_ds[ds]["eval_loss"] for ds in eval_dataset_names if ds in per_ds
        }
        if not losses:
            print(f"  SKIP alpha={alpha:.3f}: no eval losses found")
            continue
        missing = set(eval_dataset_names) - set(losses)
        if missing:
            print(f"  WARN alpha={alpha:.3f}: missing eval datasets {missing}")

        weights = {dataset_names[0]: alpha, dataset_names[1]: 1.0 - alpha}
        run_id = f"alpha_{alpha:.3f}"

        weight_strs = [f"{weights[ds]:.3f}" for ds in dataset_names]
        loss_strs = [f"{losses.get(ds, float('nan')):.4f}" for ds in eval_dataset_names]
        print(f"  OK   {run_id}: weights={weight_strs} losses={loss_strs}")

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


# ---------------------------------------------------------------------------
# olmix helpers
# ---------------------------------------------------------------------------


def write_csvs(
    ratios_rows, metrics_rows, dataset_names, output_dir, eval_dataset_names=None
):
    """Write ratios.csv and metrics.csv to output_dir. Returns (ratios_path, metrics_path).

    Args:
        dataset_names: Column names for ratios.csv (merge components).
        eval_dataset_names: Column names for metrics.csv. Defaults to dataset_names.
    """
    if eval_dataset_names is None:
        eval_dataset_names = dataset_names

    output_dir.mkdir(parents=True, exist_ok=True)

    ratios_file = output_dir / "ratios.csv"
    with open(ratios_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run"] + dataset_names)
        writer.writeheader()
        writer.writerows(ratios_rows)

    metrics_file = output_dir / "metrics.csv"
    metric_cols = [f"eval_{ds}_loss" for ds in eval_dataset_names]
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run"] + metric_cols, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    return ratios_file, metrics_file


def write_olmix_config(
    ratios_file,
    metrics_file,
    dataset_names,
    config_path,
    relative_sizes=None,
    source_mixtures=None,
    token_counts=None,
    total_tokens=None,
    repetition_factor=None,
    fit_only=False,
    kl_reg=0.05,
):
    """Write a minimal olmix fit config YAML.

    If relative_sizes is provided with source_mixtures, treats it as the leaf-level
    KL prior and collapses it to the source-level prior expected by OLMix.

    If relative_sizes is provided without source_mixtures, uses it directly as the
    source-level KL prior (must have one entry per dataset_name).

    Otherwise, falls back to a uniform prior over dataset_names.

    If relative_sizes and source_mixtures are provided, writes
    the expanded-KL config fields used by OLMix's exact proposer.

    If token_counts is provided (dict of dataset -> count), uses real dataset
    sizes and enables constraints with the given repetition_factor.
    """
    n = len(dataset_names)
    if source_mixtures is not None and relative_sizes is None:
        raise ValueError("relative_sizes must be provided with source_mixtures")

    leaf_relative_sizes = relative_sizes if source_mixtures is not None else None
    if leaf_relative_sizes is not None:
        relative_sizes = {}
        for source in dataset_names:
            if source in source_mixtures:
                missing_leaves = [
                    leaf
                    for leaf in source_mixtures[source]
                    if leaf not in leaf_relative_sizes
                ]
                if missing_leaves:
                    raise ValueError(
                        f"source_mixtures[{source}] contains leaves missing from "
                        f"relative_sizes: {missing_leaves}"
                    )
                relative_sizes[source] = sum(
                    leaf_relative_sizes[leaf] for leaf in source_mixtures[source]
                )
            else:
                if source not in leaf_relative_sizes:
                    raise ValueError(
                        f"relative_sizes must include unexpanded source {source}"
                    )
                relative_sizes[source] = leaf_relative_sizes[source]
    elif relative_sizes is None:
        relative_sizes = {ds: round(1.0 / n, 6) for ds in dataset_names}

    if token_counts is None:
        token_counts = {ds: 10_485_760_000 for ds in dataset_names}
    if total_tokens is None:
        total_tokens = sum(token_counts.values())

    enable_constraints = repetition_factor is not None
    constraints_config = {"enabled": False}
    if enable_constraints:
        constraints_config = {
            "enabled": True,
            "target_tokens": total_tokens,
            "repetition_factor": repetition_factor,
        }

    config = {
        "swarm": {
            "ratios": str(ratios_file.resolve()),
            "metrics": str(metrics_file.resolve()),
        },
        "priors": {
            "total_tokens": total_tokens,
            "relative_sizes": relative_sizes,
            "token_counts": token_counts,
        },
        "regression": {
            "type": "log_linear",
            "seed": 0,
            "n_test": 0,
            "train_split": 1.0,
            "aggregate_task_families": False,
        },
        "proposer": {
            "type": "exact",
            "temperature": None,
            "kl_reg": kl_reg,
            "fit_only": fit_only,
            "make_worst_mix": False,
        },
        "constraints": constraints_config,
        "filtering": {"drop_metrics": [], "obj_weights": {}},
    }
    if source_mixtures is not None:
        config["priors"]["expanded_relative_sizes"] = leaf_relative_sizes
        config["proposer"]["expanded_kl_source_mixtures"] = source_mixtures
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_olmix_fit(config_path, olmix_output_dir, olmix_repo=DEFAULT_OLMIX_REPO):
    """Run olmix fit and return the optimal weights dict, or None on failure."""
    config_path, olmix_output_dir, olmix_repo = (
        Path(config_path),
        Path(olmix_output_dir),
        Path(olmix_repo),
    )
    olmix_output_dir.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    result = subprocess.run(
        [
            "uv",
            "run",
            "olmix",
            "fit",
            "--config",
            str(config_path.resolve()),
            "--output-dir",
            str(olmix_output_dir.resolve()),
        ],
        cwd=str(olmix_repo),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr)[-800:]
        print(f"  olmix fit failed:\n{output}")
        return None

    subdirs = sorted(
        [d for d in olmix_output_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    if not subdirs:
        print(f"  No output subdir found in {olmix_output_dir}")
        return None

    opt_file = subdirs[-1] / "opt_avg_all_metrics_log_linear_reg_optimal.json"
    if not opt_file.exists():
        print(f"  Optimal weights file not found in {subdirs[-1]}")
        return None

    with open(opt_file) as f:
        weights_list = json.load(f)
    return {entry["domain"]: entry["weight"] for entry in weights_list}


def _run_olmix_command(config_path, olmix_output_dir, olmix_repo=DEFAULT_OLMIX_REPO):
    config_path, olmix_output_dir, olmix_repo = (
        Path(config_path),
        Path(olmix_output_dir),
        Path(olmix_repo),
    )
    olmix_output_dir.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    result = subprocess.run(
        [
            "uv",
            "run",
            "olmix",
            "fit",
            "--config",
            str(config_path.resolve()),
            "--output-dir",
            str(olmix_output_dir.resolve()),
        ],
        cwd=str(olmix_repo),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr)[-800:]
        print(f"  olmix fit failed:\n{output}")
        return None

    subdirs = sorted(
        [d for d in olmix_output_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    if not subdirs:
        print(f"  No output subdir found in {olmix_output_dir}")
        return None
    return subdirs[-1]


def run_olmix_surface_fit(config_path, olmix_output_dir, olmix_repo=DEFAULT_OLMIX_REPO):
    """Run OLMix in fit-only mode and return log-linear params keyed by metric."""
    output_subdir = _run_olmix_command(config_path, olmix_output_dir, olmix_repo)
    if output_subdir is None:
        return None

    link_file = output_subdir / "path_to_regression_model.txt"
    if not link_file.exists():
        print(f"  Regression model link not found in {output_subdir}")
        return None

    params_file = Path(link_file.read_text().strip())
    if not params_file.is_absolute():
        params_file = Path(olmix_repo) / params_file
    if not params_file.exists():
        print(f"  Regression model file not found: {params_file}")
        return None

    with open(params_file, "rb") as handle:
        return pickle.load(handle)


def _normalize_weights(weights, domains):
    vals = [max(0.0, float(weights.get(domain, 0.0))) for domain in domains]
    total = sum(vals)
    if total <= 0.0 or not math.isfinite(total):
        return {domain: 1.0 / len(domains) for domain in domains}
    return {domain: val / total for domain, val in zip(domains, vals)}


def l1_diff(w1, w2, domains):
    return sum(abs(w1.get(d, 0.0) - w2.get(d, 0.0)) for d in domains)


# ---------------------------------------------------------------------------
# Main: single-sweep driver
# ---------------------------------------------------------------------------


def main(
    sweep_dir,
    prefix="",
    model_size="150M",
    dataset_names=None,
    olmix_repo=DEFAULT_OLMIX_REPO,
    output_dir=None,
):
    """Run olmix fit on a single training-sweep directory.

    Args:
        sweep_dir: Directory containing weight-sweep run subdirs (e.g., runs/continual_sft_mix).
        prefix: Prefix for run subdirs ('lora', 'full', or '' for all).
        model_size: Auto-detect {sweep_dir}/{model_size}/ subdir if it exists.
        dataset_names: Dataset names in order. If omitted, inferred from sweep_dir basename.
        olmix_repo: Path to olmix repo with its own uv venv.
        output_dir: Where to write CSVs, config, and olmix output. Default: sweep_dir.
    """
    sweep_dir = Path(sweep_dir)
    if (sweep_dir / model_size).is_dir():
        sweep_dir = sweep_dir / model_size

    output_dir = Path(output_dir) if output_dir else sweep_dir
    olmix_repo = Path(olmix_repo)

    if dataset_names is None:
        dataset_names = sweep_dir.name.split("-")
        print(f"Inferred dataset names from dir: {dataset_names}")
    elif isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    n_candidates = sum(
        1 for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
    )
    if not n_candidates:
        prefix_msg = f"'{prefix}'" if prefix else "(any)"
        print(f"No subdirectories starting with {prefix_msg} found in {sweep_dir}")
        return

    print(f"Found {n_candidates} candidate run(s) in {sweep_dir}\n")
    ratios_rows, metrics_rows, skipped = get_run_rows(sweep_dir, prefix, dataset_names)

    for rr, mr in zip(ratios_rows, metrics_rows):
        ds_weights = [f"{rr[ds]:.3f}" for ds in dataset_names]
        ds_losses = [
            f"{mr.get(f'eval_{ds}_loss', float('nan')):.4f}" for ds in dataset_names
        ]
        print(f"  OK   {rr['run']}: weights={ds_weights} losses={ds_losses}")
    if skipped:
        print(f"  ({skipped} skipped)")

    if not ratios_rows:
        print("\nNo valid runs found. Nothing written.")
        return

    ratios_file, metrics_file = write_csvs(
        ratios_rows, metrics_rows, dataset_names, output_dir
    )
    print(f"\nWrote {len(ratios_rows)} run(s) to:")
    print(f"  {ratios_file}")
    print(f"  {metrics_file}")

    config_path = output_dir / "fit_config.yaml"
    write_olmix_config(ratios_file, metrics_file, dataset_names, config_path)
    print(f"  {config_path}")

    print(f"\nRunning olmix fit ({len(ratios_rows)} proxy points)...")
    opt_weights = run_olmix_fit(config_path, output_dir / "olmix_out", olmix_repo)

    if opt_weights is None:
        print("olmix fit failed.")
        return

    print(f"\nOptimal weights:")
    for ds in dataset_names:
        print(f"  {ds}: {opt_weights.get(ds, 0.0):.4f}")

    result_file = output_dir / "olmix_best_mix.json"
    with open(result_file, "w") as f:
        json.dump(
            {"dataset_names": dataset_names, "optimal_weights": opt_weights},
            f,
            indent=2,
        )
    print(f"\nSaved to {result_file}")


# ---------------------------------------------------------------------------
# compare_main: merge-proxy vs sweep-proxy comparison (not the default CLI)
# ---------------------------------------------------------------------------


def compare_main(
    merges_dir="results/merges/lora",
    runs_dir="runs",
    prefix="lora",
    olmix_repo=DEFAULT_OLMIX_REPO,
    output_dir="olmix_outputs/comparison",
):
    """Compare merge-based vs training-sweep-based olmix proxy methods.

    For each pair present in both merges_dir and runs_dir, runs olmix fit
    on both sources and reports the L1 distance between recommended weights.
    """
    merges_dir = Path(merges_dir)
    runs_dir = Path(runs_dir)
    olmix_repo = Path(olmix_repo)
    output_dir = Path(output_dir)

    merge_pairs = {d.name for d in merges_dir.iterdir() if d.is_dir()}
    run_pairs = {d.name for d in runs_dir.iterdir() if d.is_dir() and "-" in d.name}
    common_pairs = sorted(merge_pairs & run_pairs)

    print(f"Pairs in both {merges_dir} and {runs_dir}: {len(common_pairs)}")
    for p in common_pairs:
        print(f"  {p}")
    print()

    results = []

    for pair in common_pairs:
        print(f"{'=' * 60}")
        print(f"  {pair}")
        print(f"{'=' * 60}")
        dataset_names = pair.split("-")
        pair_out = output_dir / pair

        print("  [merges] collecting...")
        merge_ratios, merge_metrics = get_merge_rows(merges_dir / pair, dataset_names)
        if not merge_ratios:
            print(f"  SKIP {pair}: no merge data\n")
            continue
        mf_r, mf_m = write_csvs(
            merge_ratios, merge_metrics, dataset_names, pair_out / "merges"
        )
        write_olmix_config(
            mf_r, mf_m, dataset_names, pair_out / "merges" / "fit_config.yaml"
        )
        print(f"  [merges] running olmix fit ({len(merge_ratios)} proxy points)...")
        merges_opt = run_olmix_fit(
            pair_out / "merges" / "fit_config.yaml",
            pair_out / "merges" / "olmix_out",
            olmix_repo,
        )

        print("  [runs] collecting...")
        run_ratios, run_metrics, n_skipped = get_run_rows(
            runs_dir / pair, args.prefix, dataset_names
        )
        if not run_ratios:
            print(f"  SKIP {pair}: no runs data\n")
            continue
        rf_r, rf_m = write_csvs(
            run_ratios, run_metrics, dataset_names, pair_out / "runs"
        )
        write_olmix_config(
            rf_r, rf_m, dataset_names, pair_out / "runs" / "fit_config.yaml"
        )
        print(f"  [runs] running olmix fit ({len(run_ratios)} proxy points)...")
        runs_opt = run_olmix_fit(
            pair_out / "runs" / "fit_config.yaml",
            pair_out / "runs" / "olmix_out",
            olmix_repo,
        )

        if merges_opt is None or runs_opt is None:
            print(f"  SKIP {pair}: olmix fit failed\n")
            continue

        diff = l1_diff(merges_opt, runs_opt, dataset_names)
        print(
            f"  merges optimal : { {d: f'{merges_opt[d]:.3f}' for d in dataset_names} }"
        )
        print(
            f"  runs   optimal : { {d: f'{runs_opt[d]:.3f}' for d in dataset_names} }"
        )
        print(f"  L1 diff        : {diff:.4f}\n")

        results.append(
            {
                "pair": pair,
                "dataset_names": dataset_names,
                "merges_optimal": merges_opt,
                "runs_optimal": runs_opt,
                "l1_diff": diff,
                "n_merge_points": len(merge_ratios),
                "n_run_points": len(run_ratios),
            }
        )

    if not results:
        print("No results to summarize.")
        return

    print(f"\n{'=' * 70}")
    print("SUMMARY — merge-proxy vs sweep-proxy (olmix optimal weights)")
    print(f"{'=' * 70}")

    col_pair = 35
    print(
        f"{'Pair':<{col_pair}} {'Source':<8} "
        + "  ".join(f"{'W[' + str(i) + ']':>7}" for i in range(2))
        + f"  {'L1 diff':>8}"
    )
    print("-" * 70)

    for r in results:
        ds = r["dataset_names"]
        mw = r["merges_optimal"]
        rw = r["runs_optimal"]
        w_merges = "  ".join(f"{mw.get(d, 0):>7.3f}" for d in ds)
        w_runs = "  ".join(f"{rw.get(d, 0):>7.3f}" for d in ds)
        print(f"{r['pair']:<{col_pair}} {'merges':<8} {w_merges}")
        print(f"{'':  <{col_pair}} {'runs':<8} {w_runs}  {r['l1_diff']:>8.4f}")
        print()

    avg_diff = sum(r["l1_diff"] for r in results) / len(results)
    print(f"Mean L1 difference across {len(results)} pair(s): {avg_diff:.4f}")

    summary_file = output_dir / "comparison_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nFull results saved to {summary_file}")


# ---------------------------------------------------------------------------
# lora_proxy: single LoRA run as olmix proxy
# ---------------------------------------------------------------------------


def lora_proxy(
    run_dir,
    init_dir="runs/s1",
    output_dir="olmix_inputs",
    olmix_repo=DEFAULT_OLMIX_REPO,
    run_fit=True,
):
    """Use a single LoRA run (90% target, 10% source) as a cheap olmix proxy.

    For each {source}_to_{target} subdir in run_dir:
      - m0: base model losses (step 0 from init_dir/{source})
      - m1: source-trained losses (final step from init_dir/{source})
      - m2: LoRA 90/10 losses (final step from run_dir/{source}_to_{target})
      - Normalized m2: m2 * m0 / m1

    Writes ratios.csv + metrics.csv and runs olmix fit.

    Args:
        run_dir: Directory with {source}_to_{target} subdirs (e.g. runs/lora_no_warmup/150M).
        init_dir: Directory with per-source init runs (e.g. runs/s1).
        output_dir: Where to write CSVs and olmix output.
        olmix_repo: Path to olmix repo.
        run_fit: Whether to run olmix fit after writing CSVs.
    """
    run_dir = Path(run_dir)
    init_dir = Path(init_dir)
    output_dir = Path(output_dir)
    olmix_repo = Path(olmix_repo)

    pair_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and "_to_" in d.name]
    )

    if not pair_dirs:
        print(f"No *_to_* subdirs found in {run_dir}")
        return

    print(f"Found {len(pair_dirs)} pair(s) in {run_dir}\n")

    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        source, target = pair_name.split("_to_")
        dataset_names = [source, target]

        print(f"{'=' * 60}")
        print(f"  {pair_name}")
        print(f"{'=' * 60}")

        source_run = init_dir / source
        if not source_run.is_dir():
            print(f"  SKIP: init dir {source_run} not found\n")
            continue

        # m0: base model losses (step 0)
        m0 = get_eval_losses_at_step(source_run, dataset_names, step=0)
        if m0 is None:
            print(f"  SKIP: no step-0 eval losses in {source_run}\n")
            continue

        # m1: source-trained losses (final step)
        m1 = get_final_eval_losses(source_run, dataset_names)
        if m1 is None:
            print(f"  SKIP: no final eval losses in {source_run}\n")
            continue

        # m2: LoRA 90/10 losses (final step)
        m2 = get_final_eval_losses(pair_dir, dataset_names)
        if m2 is None:
            print(f"  SKIP: no final eval losses in {pair_dir}\n")
            continue

        # Normalize m2: m2 * m0 / m1
        m2_norm = {ds: m2[ds] * m0[ds] / m1[ds] for ds in dataset_names}

        print(f"  m0 (base):       { {ds: f'{m0[ds]:.4f}' for ds in dataset_names} }")
        print(f"  m1 (source):     { {ds: f'{m1[ds]:.4f}' for ds in dataset_names} }")
        print(f"  m2 (lora 90/10): { {ds: f'{m2[ds]:.4f}' for ds in dataset_names} }")
        print(
            f"  m2 (normalized): { {ds: f'{m2_norm[ds]:.4f}' for ds in dataset_names} }"
        )

        # Build rows: m1 = (1.0, 0.0), m2 = (0.1, 0.9)
        ratios_rows = [
            {"run": "m1", source: 1.0, target: 0.0},
            {"run": "m2", source: 0.1, target: 0.9},
        ]
        metrics_rows = [
            {"run": "m1", **{f"eval_{ds}_loss": m1[ds] for ds in dataset_names}},
            {"run": "m2", **{f"eval_{ds}_loss": m2_norm[ds] for ds in dataset_names}},
        ]

        pair_out = output_dir / pair_name
        ratios_file, metrics_file = write_csvs(
            ratios_rows, metrics_rows, dataset_names, pair_out
        )
        print(f"  Wrote: {ratios_file}")
        print(f"         {metrics_file}")

        config_path = pair_out / "fit_config.yaml"
        write_olmix_config(ratios_file, metrics_file, dataset_names, config_path)
        print(f"         {config_path}")

        if run_fit:
            print(f"  Running olmix fit...")
            opt_weights = run_olmix_fit(config_path, pair_out / "olmix_out", olmix_repo)

            if opt_weights is None:
                print(f"  olmix fit failed.\n")
                continue

            print(f"  Optimal weights:")
            for ds in dataset_names:
                print(f"    {ds}: {opt_weights.get(ds, 0.0):.4f}")

            result_file = pair_out / "olmix_best_mix.json"
            with open(result_file, "w") as f:
                json.dump(
                    {"dataset_names": dataset_names, "optimal_weights": opt_weights},
                    f,
                    indent=2,
                )
            print(f"  Saved to {result_file}")

        print()


def merge_proxy(
    merges_dir,
    output_dir=None,
    olmix_repo=DEFAULT_OLMIX_REPO,
    run_fit=True,
):
    """Run olmix on merge-eval results (continual learning setting).

    For each {source}_to_{target} subdir in merges_dir, reads the
    linear_connectivity_results.json and maps alpha to data mixture weights:
      alpha=0 → pure source, alpha=1 → pure target.

    Args:
        merges_dir: Directory with {source}_to_{target} subdirs containing
                    checkpoint-*/linear_connectivity_results.json.
        output_dir: Where to write CSVs and olmix output. Default: merges_dir.
        olmix_repo: Path to olmix repo.
        run_fit: Whether to run olmix fit after writing CSVs.
    """
    merges_dir = Path(merges_dir)
    output_dir = Path(output_dir) if output_dir else merges_dir
    olmix_repo = Path(olmix_repo)

    pair_dirs = sorted(
        [d for d in merges_dir.iterdir() if d.is_dir() and "_to_" in d.name]
    )

    if not pair_dirs:
        print(f"No *_to_* subdirs found in {merges_dir}")
        return

    print(f"Found {len(pair_dirs)} pair(s) in {merges_dir}\n")

    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        source, target = pair_name.split("_to_")

        print(f"{'=' * 60}")
        print(f"  {pair_name}")
        print(f"{'=' * 60}")

        # Pass [target, source] so that get_merge_rows maps:
        #   alpha → target weight, (1-alpha) → source weight
        # This is correct because alpha=1 means full LoRA (target-trained).
        dataset_names_for_merge = [target, source]

        ratios_rows, metrics_rows = get_merge_rows(pair_dir, dataset_names_for_merge)
        if not ratios_rows:
            print(f"  SKIP: no merge data\n")
            continue

        # Reorder to canonical [source, target] for output
        dataset_names = [source, target]
        pair_out = output_dir / pair_name

        ratios_file, metrics_file = write_csvs(
            ratios_rows, metrics_rows, dataset_names_for_merge, pair_out
        )
        print(f"  Wrote {len(ratios_rows)} points to {pair_out}")

        config_path = pair_out / "fit_config.yaml"
        write_olmix_config(
            ratios_file, metrics_file, dataset_names_for_merge, config_path
        )

        if run_fit:
            print(f"  Running olmix fit...")
            opt_weights = run_olmix_fit(config_path, pair_out / "olmix_out", olmix_repo)

            if opt_weights is None:
                print(f"  olmix fit failed.\n")
                continue

            print(f"  Optimal weights:")
            for ds in dataset_names_for_merge:
                print(f"    {ds}: {opt_weights.get(ds, 0.0):.4f}")

            result_file = pair_out / "olmix_best_mix.json"
            with open(result_file, "w") as f:
                json.dump(
                    {
                        "dataset_names": dataset_names_for_merge,
                        "optimal_weights": opt_weights,
                    },
                    f,
                    indent=2,
                )
            print(f"  Saved to {result_file}")

        print()


if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
            "fit": run_olmix_fit,
            "compare": compare_main,
            "lora_proxy": lora_proxy,
            "merge_proxy": merge_proxy,
        }
    )
