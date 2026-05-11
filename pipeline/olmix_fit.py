"""Self-contained reimplementation of the OLMix fit path used by the
continual pipelines.

This module replaces the subprocess call to ``uv run olmix fit`` in
``pipeline.olmix`` with an in-process implementation of just the code
path the standalone pipelines exercise: log-linear ``ScalingLaw``
regression followed by a cvxpy KL-regularized exact proposer, with
optional collapsed-source expansion for the KL prior.

Entry point: ``fit_and_propose(...)``.

Adapted from https://github.com/allenai/olmix (Apache-2.0) and the
upstream mixing-laws code at https://github.com/yegcjs/mixinglaws.
"""

import csv
import logging
import math
import multiprocessing as mp
import random
from functools import partial
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    mp.set_start_method("fork")
except RuntimeError:
    pass


# ---------------------------------------------------------------------------
# Scaling-law fit (log-linear)
# ---------------------------------------------------------------------------


def mixing_law(x, param, **kwargs):
    """y = exp(log_c) + exp(x @ t), with param = [log_c, t_1, ..., t_d]."""
    log_c_i = param[0]
    t_i = param[1:]
    return torch.exp(log_c_i) + torch.exp(torch.matmul(x, t_i))


def init_params_log_linear_law(idx, num_domains=3):
    """Init-grid generator: 10 log_c values × 30 random t-vectors per metric."""
    for log_c_i in np.linspace(-2, 1.5, 10):
        for _ in range(30):
            ts = [-np.random.rand() if i == idx else np.random.rand() * 0.1 for i in range(num_domains)]
            yield [log_c_i, *ts]


def _fit_scaling_laws(func, valid_split, x, y, max_step, eps, delta, init_param):
    param = torch.nn.Parameter(init_param)
    x_t, y_t = torch.tensor(x).to(param), torch.tensor(y).to(param)
    if valid_split == 0:
        train_x, eval_x = x_t, x_t[:0]
        train_y, eval_y = y_t, y_t[:0]
    else:
        train_x, eval_x = x_t[:-valid_split], x_t[-valid_split:]
        train_y, eval_y = y_t[:-valid_split], y_t[-valid_split:]
    optimizer = torch.optim.LBFGS(
        [param], lr=0.01, history_size=10, max_iter=20, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        prediction = func(train_x, param)
        loss = torch.nn.functional.huber_loss(train_y, prediction, delta=delta, reduction="sum")
        loss.backward()
        return loss

    min_loss, best_param = 1e10, None
    for i in range(max_step):
        float(optimizer.step(closure))
        with torch.no_grad():
            if len(eval_x) > 1:
                eval_prediction = func(eval_x, param)
                eval_loss = torch.nn.functional.huber_loss(eval_prediction, eval_y, delta=delta).item()
            elif len(eval_x) == 1:
                eval_prediction = func(eval_x, param)
                eval_loss = torch.nn.functional.mse_loss(eval_prediction, eval_y).item()
            else:
                eval_prediction = func(train_x, param)
                eval_loss = torch.nn.functional.huber_loss(eval_prediction, train_y, delta=delta).item()
        improvement = abs(eval_loss - min_loss)
        if eval_loss <= min_loss:
            min_loss = eval_loss
            best_param = param.detach().clone()
        if improvement < eps:
            break

    assert best_param is not None
    return min_loss, best_param.detach().cpu().numpy()


class ScalingLaw:
    def __init__(self, func):
        self.func = func
        self.params = None

    def fit(self, x, y, init_params, max_step=20, eps=0.0, workers=-1, valid_split=0, delta=0.01):
        if workers == -1:
            workers = min(4, mp.cpu_count())
        init_params = [torch.tensor(p, dtype=torch.float32) for p in init_params]
        minloss, optimal_param = 1e10, None
        _fit = partial(_fit_scaling_laws, self.func, valid_split, x, y, max_step, eps, delta)
        if workers != 1:
            with mp.Pool(workers, maxtasksperchild=20) as pool:
                for loss, param in tqdm(
                    pool.imap_unordered(_fit, init_params, chunksize=2),
                    total=len(init_params),
                ):
                    if loss < minloss:
                        minloss = loss
                        optimal_param = param
        else:
            for ip in tqdm(init_params):
                loss, param = _fit(ip)
                if loss < minloss:
                    minloss = loss
                    optimal_param = param
        assert optimal_param is not None
        self.params = optimal_param.tolist()
        logger.info(f"min loss: {minloss}")
        return self.params


class LogLinearRegressor:
    """Wraps ScalingLaw to fit a per-metric log-linear mixing law."""

    def __init__(self, params=None):
        np.random.seed(42)
        random.seed(42)
        if params is None:
            self.model = ScalingLaw(mixing_law)
        else:
            self.model = params

    def fit(self, X, Y, idx, max_step=100, delta=0.02, early_stopping=0.0, workers=-1):
        target = Y[:, idx]
        self.model = self.model.fit(
            X,
            target,
            init_params_log_linear_law(idx, num_domains=X.shape[-1]),
            max_step=max_step,
            delta=delta,
            eps=early_stopping,
            workers=workers,
        )

    def predict(self, X):
        return mixing_law(
            torch.tensor(X, dtype=torch.float),
            torch.tensor(self.model, dtype=torch.float),
        ).numpy()


# ---------------------------------------------------------------------------
# Exact KL-regularized proposer
# ---------------------------------------------------------------------------


def build_expansion_matrix(collapsed_prior, expanded_prior=None, source_mixtures=None):
    """Build a (n_expanded × n_collapsed) linear map from collapsed-source
    weights to expanded leaf weights.

    Returns (matrix, expanded_keys).
    """
    collapsed_keys = list(collapsed_prior.keys())
    if expanded_prior is None and source_mixtures is None:
        return np.eye(len(collapsed_keys)), collapsed_keys
    if expanded_prior is None or source_mixtures is None:
        raise ValueError("expanded_prior and source_mixtures must be provided together")

    expanded_keys = list(expanded_prior.keys())
    matrix = np.zeros((len(expanded_keys), len(collapsed_keys)), dtype=float)
    expanded_index = {key: idx for idx, key in enumerate(expanded_keys)}
    covered_leaves: dict[str, str] = {}

    for collapsed_idx, source in enumerate(collapsed_keys):
        if source in source_mixtures:
            mixture = source_mixtures[source]
            total = sum(mixture.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Expanded source mixture for {source} must sum to 1.0, got {total}")
            for leaf, weight in mixture.items():
                if leaf not in expanded_index:
                    raise ValueError(f"Expanded source mixture leaf {leaf} missing from expanded prior")
                if leaf in covered_leaves:
                    raise ValueError(
                        f"Expanded prior leaf {leaf} is covered by both {covered_leaves[leaf]} and {source}"
                    )
                covered_leaves[leaf] = source
                matrix[expanded_index[leaf], collapsed_idx] = weight
        else:
            if source not in expanded_index:
                raise ValueError(f"Collapsed source {source} missing from expanded prior")
            if source in covered_leaves:
                raise ValueError(
                    f"Expanded prior leaf {source} is covered by both {covered_leaves[source]} and {source}"
                )
            covered_leaves[source] = source
            matrix[expanded_index[source], collapsed_idx] = 1.0

    column_sums = matrix.sum(axis=0)
    if not np.allclose(column_sums, 1.0):
        raise ValueError(
            f"Expanded source mixtures must define a full partition of each collapsed source, got {column_sums}"
        )

    row_sums = matrix.sum(axis=1)
    if not np.all(row_sums > 0):
        uncovered = [k for k, r in zip(expanded_keys, row_sums) if r == 0]
        raise ValueError(f"Expanded prior contains leaves not covered by the collapsed source mapping: {uncovered}")
    return matrix, expanded_keys


def log_linear_exact_propose(
    predictors,
    prior_distributions,
    expanded_prior_distributions=None,
    expanded_source_mixtures=None,
    kl_reg=0.05,
):
    """Solve min_x sum_i (1/n) exp(A_i @ x) + kl_reg * KL(expand(x) || q)
    s.t. x >= 0, sum(x) = 1, where A_i is the slope vector of the i-th
    fitted LogLinearRegressor.
    """
    if kl_reg is None:
        raise ValueError("kl_reg must be provided")

    A = np.array([p.model[1:] for p in predictors])  # (n_metrics, d)
    n, d = A.shape
    weights = np.ones(n) / n

    if expanded_prior_distributions is None:
        expansion_matrix = np.eye(d, dtype=float)
        q = np.array(list(prior_distributions.values()))
    else:
        expansion_matrix, expanded_keys = build_expansion_matrix(
            prior_distributions,
            expanded_prior=expanded_prior_distributions,
            source_mixtures=expanded_source_mixtures,
        )
        q = np.array([expanded_prior_distributions[key] for key in expanded_keys])

    q = np.asarray(q, dtype=float)
    q = np.maximum(q, 1e-12)
    q = q / q.sum()

    x = cp.Variable(d)
    loss = cp.sum(cp.multiply(weights, cp.exp(A @ x)))
    x_for_kl = expansion_matrix @ x
    kl = cp.sum(cp.rel_entr(x_for_kl, q))
    obj = loss + kl_reg * kl

    constraints = [x >= 0, cp.sum(x) == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver="ECOS", verbose=False)
    if x.value is None:
        raise RuntimeError(f"cvxpy solve failed with status {prob.status}")
    return np.asarray(x.value)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def write_csvs(ratios_rows, metrics_rows, dataset_names, output_dir, eval_dataset_names=None):
    """Persist the proxy ratios and metrics to CSV for reproducibility.

    Mirrors ``pipeline.olmix.write_csvs`` so the standalone pipelines
    produce identical on-disk artifacts.
    """
    if eval_dataset_names is None:
        eval_dataset_names = dataset_names

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios_file = output_dir / "ratios.csv"
    with open(ratios_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run"] + list(dataset_names))
        writer.writeheader()
        writer.writerows(ratios_rows)

    metrics_file = output_dir / "metrics.csv"
    metric_cols = [f"eval_{ds}_loss" for ds in eval_dataset_names]
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run"] + metric_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics_rows)

    return ratios_file, metrics_file


def _collapse_relative_sizes(relative_sizes, dataset_names, source_mixtures):
    """Collapse leaf-level relative sizes into source-level for the given
    ``dataset_names`` order. Returns the collapsed prior (dict).
    """
    if source_mixtures is None:
        return {name: relative_sizes[name] for name in dataset_names}

    collapsed = {}
    for source in dataset_names:
        if source in source_mixtures:
            missing = [leaf for leaf in source_mixtures[source] if leaf not in relative_sizes]
            if missing:
                raise ValueError(
                    f"source_mixtures[{source}] contains leaves missing from relative_sizes: {missing}"
                )
            collapsed[source] = sum(relative_sizes[leaf] for leaf in source_mixtures[source])
        else:
            if source not in relative_sizes:
                raise ValueError(f"relative_sizes must include uncollapsed source {source}")
            collapsed[source] = relative_sizes[source]
    return collapsed


def fit_and_propose(
    ratios_rows,
    metrics_rows,
    dataset_names,
    eval_dataset_names,
    relative_sizes,
    source_mixtures=None,
    kl_reg=0.05,
    workers=-1,
):
    """Fit per-metric log-linear laws on the in-memory proxy rows and
    propose a KL-regularized optimal mixture.

    Args:
        ratios_rows: list of {"run": str, **{domain: weight}} dicts.
        metrics_rows: list of {"run": str, **{f"eval_{ds}_loss": float}} dicts.
        dataset_names: source-level domain order (matches columns in ratios).
        eval_dataset_names: leaf-level eval losses to fit on (one law each).
        relative_sizes: leaf-level KL prior when source_mixtures is given,
            otherwise a source-level prior over ``dataset_names``.
        source_mixtures: optional {collapsed_source: {leaf: weight}}.
        kl_reg: KL strength for the proposer.
        workers: ScalingLaw fit workers (-1 → up to 4).

    Returns:
        dict mapping each name in ``dataset_names`` to its optimal
        source-level weight, or ``None`` if the proxy data is empty or
        the convex solve fails.
    """
    if not ratios_rows or not metrics_rows:
        return None

    runs_in_ratios = [r["run"] for r in ratios_rows]
    metrics_by_run = {m["run"]: m for m in metrics_rows}
    common = [r for r in runs_in_ratios if r in metrics_by_run]
    if not common:
        return None

    metric_cols = [f"eval_{ds}_loss" for ds in eval_dataset_names]

    X_rows, Y_rows = [], []
    for run_id in common:
        ratio = next(r for r in ratios_rows if r["run"] == run_id)
        metric = metrics_by_run[run_id]
        row_y = [metric.get(c, math.nan) for c in metric_cols]
        if any(not math.isfinite(v) for v in row_y):
            continue
        X_rows.append([ratio[d] for d in dataset_names])
        Y_rows.append(row_y)

    if len(X_rows) < len(dataset_names):
        logger.warning(
            f"Fewer proxy points ({len(X_rows)}) than domains ({len(dataset_names)}); fit may be ill-conditioned"
        )
    if not X_rows:
        return None

    X_train = np.asarray(X_rows, dtype=float)
    Y_train = np.asarray(Y_rows, dtype=float)

    predictors = []
    for idx in range(len(metric_cols)):
        reg = LogLinearRegressor()
        reg.fit(X_train, Y_train, idx, workers=workers)
        predictors.append(reg)

    collapsed_prior = _collapse_relative_sizes(relative_sizes, dataset_names, source_mixtures)
    expanded_prior = relative_sizes if source_mixtures is not None else None

    try:
        weights = log_linear_exact_propose(
            predictors,
            prior_distributions=collapsed_prior,
            expanded_prior_distributions=expanded_prior,
            expanded_source_mixtures=source_mixtures,
            kl_reg=kl_reg,
        )
    except Exception as exc:
        logger.error(f"cvxpy proposer failed: {exc}")
        return None

    return {name: float(w) for name, w in zip(dataset_names, weights)}
