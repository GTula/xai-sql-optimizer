"""End-to-end evaluation used by the revised research paper.

The module produces auditable CSV files for workload composition, candidate
plans, out-of-query Ridge fidelity, SHAP stability across seeds, and XAI
scalability.  It intentionally keeps query execution out of the main grid: the
study explains the optimizers' cost models, while runtime validation remains a
separate experiment in :mod:`xai.demo_bayes_runtime_validation`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from sql.optimizer import BayesOptimizer
from sql.parser import Parser
from xai.experimental_workload import (
    DatasetConfig,
    QuerySpec,
    SIZE_PRESETS,
    SUPPORTED_DISTRIBUTIONS,
    build_experimental_db,
    dataset_manifest,
    generate_workload,
)
from xai.optimizer_explainer import build_feature_matrix, generate_candidate_plans


METADATA_COLUMNS = {
    "run_id",
    "seed",
    "size_label",
    "distribution",
    "fact_rows",
    "query_id",
    "family",
    "table_count",
    "join_count",
    "predicate_count",
    "optimizer",
    "plan_id",
    "plan_order",
}
TARGET = "estimated_cost"


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: Tuple[int, ...] = (1, 2, 3, 4, 5)
    sizes: Tuple[str, ...] = ("small", "medium", "large")
    distributions: Tuple[str, ...] = SUPPORTED_DISTRIBUTIONS
    optimizers: Tuple[str, ...] = ("selinger", "bayes")
    max_plans: int = 50
    max_orders_per_table_count: int = 12
    bootstrap_iterations: int = 1_000
    top_k: int = 5
    query_limit: int | None = None

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("Se requiere al menos una semilla")
        if not self.sizes or any(size not in SIZE_PRESETS for size in self.sizes):
            raise ValueError("Tamaños experimentales inválidos")
        if not self.distributions or any(
            item not in SUPPORTED_DISTRIBUTIONS for item in self.distributions
        ):
            raise ValueError("Distribuciones experimentales inválidas")
        if not self.optimizers or any(
            item not in {"selinger", "bayes"} for item in self.optimizers
        ):
            raise ValueError("Optimizadores inválidos")
        if self.max_plans < 2:
            raise ValueError("max_plans debe ser al menos 2")


@dataclass(frozen=True)
class ExperimentOutputs:
    output_dir: str
    raw_features_csv: str
    candidate_plans_csv: str
    ridge_fidelity_csv: str
    ridge_fidelity_across_seeds_csv: str
    query_ranking_csv: str
    shap_per_run_csv: str
    shap_stability_csv: str
    shap_stability_per_query_csv: str
    scalability_csv: str
    workload_summary_csv: str
    optimizer_details_csv: str
    paper_results_md: str
    experiment_manifest_json: str


def _plan_order(plan) -> str:
    tables = [plan.from_table]
    tables.extend(join.table for join in (plan.join_clauses or []))
    return "->".join(tables)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _percentile(values: Sequence[float], q: float) -> float:
    return (
        float(np.percentile(np.asarray(values, dtype=float), q))
        if len(values) > 0
        else float("nan")
    )


def _bootstrap_mean_ci(
    values: Sequence[float], iterations: int, seed: int = 2026
) -> Tuple[float, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(clean) == 0:
        return float("nan"), float("nan")
    if len(clean) == 1 or iterations <= 0:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for index in range(iterations):
        means[index] = float(np.mean(rng.choice(clean, size=len(clean), replace=True)))
    return _percentile(means, 2.5), _percentile(means, 97.5)


def _metric_bundle(y_log: np.ndarray, pred_log: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    y_cost = np.expm1(y_log)
    pred_cost = np.maximum(np.expm1(pred_log), 0.0)
    mae_cost = float(mean_absolute_error(y_cost, pred_cost))
    mean_cost = float(np.mean(np.abs(y_cost)))
    return {
        "r2": float(r2_score(y_log, pred_log)),
        "mae_log": float(mean_absolute_error(y_log, pred_log)),
        "mae_cost": mae_cost,
        "normalized_mae": mae_cost / mean_cost if mean_cost > 0 else 0.0,
    }


def _group_bootstrap_ci(
    y_log: np.ndarray,
    pred_log: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or iterations <= 0:
        metrics = _metric_bundle(y_log, pred_log)
        return {key: (value, value) for key, value in metrics.items()}
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = defaultdict(list)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for _ in range(iterations):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in selected])
        for key, value in _metric_bundle(y_log[indices], pred_log[indices]).items():
            if np.isfinite(value):
                samples[key].append(value)
    return {
        key: (_percentile(values, 2.5), _percentile(values, 97.5))
        for key, values in samples.items()
    }


def _rank_metrics(
    rows: Sequence[Mapping[str, object]], predictions: np.ndarray
) -> List[Dict[str, object]]:
    from scipy.stats import kendalltau, spearmanr

    grouped: Dict[int, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[int(row["query_id"])].append(index)
    metrics: List[Dict[str, object]] = []
    for query_id, indices in sorted(grouped.items()):
        if len(indices) < 2:
            continue
        actual = np.asarray([float(rows[index][TARGET]) for index in indices])
        predicted = np.asarray([predictions[index] for index in indices])
        # When every candidate has exactly the same target cost there is no
        # optimizer ranking to reproduce. Such queries remain in R²/MAE but are
        # excluded from rank agreement instead of being counted as failures.
        if np.ptp(actual) <= 1e-12:
            continue
        if np.ptp(predicted) <= 1e-12:
            spearman = 0.0
            kendall = 0.0
        else:
            spearman = spearmanr(actual, predicted).statistic
            kendall = kendalltau(actual, predicted).statistic
        metrics.append(
            {
                "query_id": query_id,
                "candidate_plans": len(indices),
                "spearman_rho": float(spearman) if np.isfinite(spearman) else 0.0,
                "kendall_tau": float(kendall) if np.isfinite(kendall) else 0.0,
                "top1_agreement": int(np.argmin(actual) == np.argmin(predicted)),
            }
        )
    return metrics


def _feature_names(rows: Sequence[Mapping[str, object]]) -> List[str]:
    names = set()
    for row in rows:
        names.update(row.keys())
    return sorted(names - METADATA_COLUMNS - {TARGET})


def _matrix(
    rows: Sequence[Mapping[str, object]], feature_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(
        [
            [float(row.get(name, 0.0) or 0.0) for name in feature_names]
            for row in rows
        ],
        dtype=float,
    )
    y_cost = np.asarray([float(row[TARGET]) for row in rows], dtype=float)
    groups = np.asarray([int(row["query_id"]) for row in rows], dtype=int)
    return np.sign(X) * np.log1p(np.abs(X)), np.log1p(np.maximum(y_cost, 0.0)), groups


def _fit_grouped_ridge(
    rows: Sequence[Mapping[str, object]],
    bootstrap_iterations: int,
    top_k: int,
    seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], float]:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_names = _feature_names(rows)
    X, y_log, groups = _matrix(rows, feature_names)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("La fidelidad agrupada requiere al menos dos consultas")

    start = time.perf_counter()
    predictions = np.empty(len(rows), dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    for train_indices, test_indices in splitter.split(X, y_log, groups):
        model = make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 33))
        )
        model.fit(X[train_indices], y_log[train_indices])
        predictions[test_indices] = model.predict(X[test_indices])
    metrics = _metric_bundle(y_log, predictions)
    intervals = _group_bootstrap_ci(
        y_log, predictions, groups, bootstrap_iterations, seed
    )
    ranking = _rank_metrics(rows, np.maximum(np.expm1(predictions), 0.0))
    for key in ("spearman_rho", "kendall_tau", "top1_agreement"):
        values = [float(row[key]) for row in ranking]
        metrics[key] = float(np.mean(values)) if values else None
        intervals[key] = (
            _bootstrap_mean_ci(values, bootstrap_iterations, seed + 17)
            if values
            else (None, None)
        )

    # Fit the reporting model on all rows.  For a linear model in standardized
    # space, coefficient * centered value is the exact additive contribution.
    reporting_model = make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 33))
    )
    reporting_model.fit(X, y_log)
    scaler = reporting_model.named_steps["standardscaler"]
    ridge = reporting_model.named_steps["ridgecv"]
    standardized = scaler.transform(X)
    contributions = (standardized - standardized.mean(axis=0)) * ridge.coef_
    fit_ms = (time.perf_counter() - start) * 1_000.0

    shap_rows: List[Dict[str, object]] = []
    for query_id in unique_groups:
        indices = np.flatnonzero(groups == query_id)
        magnitudes = np.mean(np.abs(contributions[indices]), axis=0)
        total = float(np.sum(magnitudes))
        percentages = magnitudes / total * 100.0 if total > 0 else magnitudes
        order = np.argsort(-percentages)
        ranks = np.empty(len(feature_names), dtype=int)
        ranks[order] = np.arange(1, len(feature_names) + 1)
        for feature_index, feature_name in enumerate(feature_names):
            shap_rows.append(
                {
                    "query_id": int(query_id),
                    "feature": feature_name,
                    "importance_pct": float(percentages[feature_index]),
                    "rank": int(ranks[feature_index]),
                    "in_top_k": int(ranks[feature_index] <= top_k),
                }
            )

    summary: Dict[str, object] = {
        "rows": len(rows),
        "queries": len(unique_groups),
        "rank_evaluable_queries": len(ranking),
    }
    for key, value in metrics.items():
        summary[key] = value
        summary[f"{key}_ci95_low"] = intervals[key][0]
        summary[f"{key}_ci95_high"] = intervals[key][1]
    return summary, ranking, shap_rows, fit_ms


def _theoretical_plans(table_count: int, optimizer: str) -> int:
    if table_count <= 1:
        return 1
    if optimizer == "bayes":
        return math.factorial(table_count - 1)
    # Connected left-deep orders for a star: fact first, or one dimension first.
    return 2 * math.factorial(table_count - 1)


def _collect_raw_data(
    config: ExperimentConfig,
    specs: Sequence[QuerySpec],
    verbose: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    feature_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    temporary_root = tempfile.mkdtemp(prefix="research_xai_")
    try:
        total_runs = (
            len(config.seeds)
            * len(config.sizes)
            * len(config.distributions)
            * len(config.optimizers)
        )
        run_number = 0
        for seed in config.seeds:
            for size_label in config.sizes:
                for distribution in config.distributions:
                    dataset_config = DatasetConfig(
                        seed=seed,
                        size_label=size_label,
                        distribution=distribution,
                    )
                    manifest = dataset_manifest(dataset_config)
                    for optimizer_name in config.optimizers:
                        run_number += 1
                        if verbose:
                            print(
                                f"[{run_number}/{total_runs}] {optimizer_name} "
                                f"{dataset_config.experiment_id()}"
                            )
                        db = build_experimental_db(
                            temporary_root, optimizer_name, dataset_config
                        )
                        try:
                            optimizer = db.engine.optimizer
                            run_id = f"{dataset_config.experiment_id()}_{optimizer_name}"
                            for spec in specs:
                                statement = Parser.parse(spec.sql)
                                fix_base = isinstance(optimizer, BayesOptimizer)
                                tracemalloc.start()
                                started = time.perf_counter()
                                candidates = generate_candidate_plans(
                                    statement,
                                    max_plans=config.max_plans,
                                    fix_base_table=fix_base,
                                )
                                candidate_ms = (time.perf_counter() - started) * 1_000.0
                                feature_started = time.perf_counter()
                                _, _, feature_dicts = build_feature_matrix(
                                    candidates, statement, optimizer
                                )
                                feature_ms = (time.perf_counter() - feature_started) * 1_000.0
                                _, peak_bytes = tracemalloc.get_traced_memory()
                                tracemalloc.stop()
                                theoretical = _theoretical_plans(
                                    spec.table_count, optimizer_name
                                )
                                candidate_rows.append(
                                    {
                                        "run_id": run_id,
                                        "seed": seed,
                                        "size_label": size_label,
                                        "distribution": distribution,
                                        "fact_rows": manifest["fact_rows"],
                                        "query_id": spec.query_id,
                                        "family": spec.family,
                                        "table_count": spec.table_count,
                                        "join_count": spec.join_count,
                                        "predicate_count": spec.predicate_count,
                                        "predicate_logic": spec.predicate_logic,
                                        "selectivity_band": spec.selectivity_band,
                                        "optimizer": optimizer_name,
                                        "theoretical_connected_plans": theoretical,
                                        "candidate_plans": len(candidates),
                                        "reached_plan_cap": int(len(candidates) >= config.max_plans),
                                        "candidate_generation_ms": candidate_ms,
                                        "feature_extraction_ms": feature_ms,
                                        "xai_preparation_ms": candidate_ms + feature_ms,
                                        "peak_memory_kib": peak_bytes / 1024.0,
                                    }
                                )
                                for plan_id, (plan, values) in enumerate(
                                    zip(candidates, feature_dicts)
                                ):
                                    row: Dict[str, object] = {
                                        "run_id": run_id,
                                        "seed": seed,
                                        "size_label": size_label,
                                        "distribution": distribution,
                                        "fact_rows": manifest["fact_rows"],
                                        "query_id": spec.query_id,
                                        "family": spec.family,
                                        "table_count": spec.table_count,
                                        "join_count": spec.join_count,
                                        "predicate_count": spec.predicate_count,
                                        "optimizer": optimizer_name,
                                        "plan_id": plan_id,
                                        "plan_order": _plan_order(plan),
                                    }
                                    row.update(values)
                                    feature_rows.append(row)
                        finally:
                            db.close()
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return feature_rows, candidate_rows


def _evaluate_runs(
    feature_rows: Sequence[Mapping[str, object]], config: ExperimentConfig
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in feature_rows:
        grouped[str(row["run_id"])].append(row)
    fidelity_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []
    shap_rows: List[Dict[str, object]] = []
    fit_time_rows: List[Dict[str, object]] = []
    for run_index, (run_id, rows) in enumerate(sorted(grouped.items()), start=1):
        first = rows[0]
        summary, rankings, query_shap, fit_ms = _fit_grouped_ridge(
            rows,
            bootstrap_iterations=config.bootstrap_iterations,
            top_k=config.top_k,
            seed=10_000 + run_index,
        )
        identity = {
            "run_id": run_id,
            "seed": first["seed"],
            "size_label": first["size_label"],
            "distribution": first["distribution"],
            "optimizer": first["optimizer"],
        }
        fidelity_rows.append({**identity, **summary, "ridge_fit_ms": fit_ms})
        ranking_rows.extend({**identity, **row} for row in rankings)
        shap_rows.extend({**identity, **row} for row in query_shap)
        fit_time_rows.append({**identity, "ridge_fit_ms": fit_ms})
    return fidelity_rows, ranking_rows, shap_rows, fit_time_rows


def _aggregate_shap(
    shap_rows: Sequence[Mapping[str, object]], config: ExperimentConfig
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    per_query_groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in shap_rows:
        key = (
            row["optimizer"],
            row["size_label"],
            row["distribution"],
            row["query_id"],
            row["feature"],
        )
        per_query_groups[key].append(row)

    per_query: List[Dict[str, object]] = []
    for key, rows in sorted(per_query_groups.items(), key=lambda item: str(item[0])):
        values = [float(row["importance_pct"]) for row in rows]
        ci_low, ci_high = _bootstrap_mean_ci(
            values, config.bootstrap_iterations, seed=31_337
        )
        per_query.append(
            {
                "optimizer": key[0],
                "size_label": key[1],
                "distribution": key[2],
                "query_id": key[3],
                "feature": key[4],
                "seed_repetitions": len(rows),
                "mean_importance_pct": float(np.mean(values)),
                "std_importance_pct": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "top_k_frequency": float(np.mean([float(row["in_top_k"]) for row in rows])),
                "mean_rank": float(np.mean([float(row["rank"]) for row in rows])),
            }
        )

    global_groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in per_query:
        key = (row["optimizer"], row["size_label"], row["distribution"], row["feature"])
        global_groups[key].append(row)
    global_rows: List[Dict[str, object]] = []
    for key, rows in sorted(global_groups.items(), key=lambda item: str(item[0])):
        means = [float(row["mean_importance_pct"]) for row in rows]
        ci_low, ci_high = _bootstrap_mean_ci(
            means, config.bootstrap_iterations, seed=73_331
        )
        global_rows.append(
            {
                "optimizer": key[0],
                "size_label": key[1],
                "distribution": key[2],
                "feature": key[3],
                "queries": len(rows),
                "mean_importance_pct": float(np.mean(means)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_top_k_frequency": float(
                    np.mean([float(row["top_k_frequency"]) for row in rows])
                ),
            }
        )
    return global_rows, per_query


def _aggregate_fidelity_across_seeds(
    fidelity_rows: Sequence[Mapping[str, object]], config: ExperimentConfig
) -> List[Dict[str, object]]:
    """Summarize model fidelity over independent dataset seeds."""
    metrics = (
        "r2",
        "mae_log",
        "mae_cost",
        "normalized_mae",
        "spearman_rho",
        "kendall_tau",
        "top1_agreement",
    )
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in fidelity_rows:
        grouped[
            (
                str(row["optimizer"]),
                str(row["size_label"]),
                str(row["distribution"]),
            )
        ].append(row)

    result: List[Dict[str, object]] = []
    for group_index, (key, rows) in enumerate(sorted(grouped.items()), start=1):
        summary: Dict[str, object] = {
            "optimizer": key[0],
            "size_label": key[1],
            "distribution": key[2],
            "seed_repetitions": len(rows),
        }
        for metric_index, metric in enumerate(metrics):
            values = [
                float(row[metric])
                for row in rows
                if row.get(metric) not in (None, "")
                and np.isfinite(float(row[metric]))
            ]
            summary[f"{metric}_valid_repetitions"] = len(values)
            if not values:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
                summary[f"{metric}_ci95_low"] = None
                summary[f"{metric}_ci95_high"] = None
                continue
            ci_low, ci_high = _bootstrap_mean_ci(
                values,
                config.bootstrap_iterations,
                seed=90_000 + group_index * 100 + metric_index,
            )
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            summary[f"{metric}_ci95_low"] = ci_low
            summary[f"{metric}_ci95_high"] = ci_high
        result.append(summary)
    return result


def _summaries(
    specs: Sequence[QuerySpec],
    candidate_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[Tuple[int, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(int(row["table_count"]), str(row["optimizer"]))].append(row)
    counts = defaultdict(int)
    for spec in specs:
        counts[spec.table_count] += 1
    workload_rows: List[Dict[str, object]] = []
    for (table_count, optimizer), rows in sorted(grouped.items()):
        candidates = [float(row["candidate_plans"]) for row in rows]
        workload_rows.append(
            {
                "table_count": table_count,
                "join_count": max(0, table_count - 1),
                "query_templates": counts[table_count],
                "optimizer": optimizer,
                "mean_candidate_plans": float(np.mean(candidates)),
                "min_candidate_plans": int(np.min(candidates)),
                "max_candidate_plans": int(np.max(candidates)),
                "cap_frequency": float(
                    np.mean([float(row["reached_plan_cap"]) for row in rows])
                ),
            }
        )

    scalability_rows: List[Dict[str, object]] = []
    for (table_count, optimizer), rows in sorted(grouped.items()):
        elapsed = [float(row["xai_preparation_ms"]) for row in rows]
        memory = [float(row["peak_memory_kib"]) for row in rows]
        scalability_rows.append(
            {
                "table_count": table_count,
                "optimizer": optimizer,
                "observations": len(rows),
                "mean_xai_preparation_ms": float(np.mean(elapsed)),
                "median_xai_preparation_ms": float(np.median(elapsed)),
                "p95_xai_preparation_ms": _percentile(elapsed, 95),
                "mean_peak_memory_kib": float(np.mean(memory)),
                "p95_peak_memory_kib": _percentile(memory, 95),
            }
        )
    return workload_rows, scalability_rows


def _optimizer_details() -> List[Dict[str, object]]:
    return [
        {
            "optimizer": "Selinger",
            "search": "dynamic programming over connected left-deep orders",
            "cardinality_source": "catalog row counts and NDV",
            "base_table": "not fixed",
            "candidate_space": "up to 2*(N-1)! on the star schema",
        },
        {
            "optimizer": "BayesCard",
            "search": "dynamic programming with probabilistic selectivity estimates",
            "cardinality_source": "Bayesian profiles with catalog fallback",
            "base_table": "FROM table fixed",
            "candidate_space": "(N-1)! on the star schema",
        },
    ]


def _paper_markdown(
    config: ExperimentConfig,
    specs: Sequence[QuerySpec],
    fidelity_across_seeds: Sequence[Mapping[str, object]],
    workload_rows: Sequence[Mapping[str, object]],
) -> str:
    def formatted(row: Mapping[str, object], key: str) -> str:
        value = row.get(key)
        return "—" if value in (None, "") else f"{float(value):.3f}"

    def formatted_rank(row: Mapping[str, object], key: str) -> str:
        valid = int(row.get(f"{key.removesuffix('_mean')}_valid_repetitions", 0))
        expected = int(row["seed_repetitions"])
        return formatted(row, key) if valid == expected else "—"

    lines = [
        "# Generated experimental results",
        "",
        "> This file is generated by `xai/research_evaluation.py`. Do not edit results manually.",
        "",
        "## Experimental design",
        "",
        f"- Query templates: {len(specs)}",
        f"- Seeds: {', '.join(map(str, config.seeds))}",
        f"- Sizes: {', '.join(config.sizes)}",
        f"- Distributions: {', '.join(config.distributions)}",
        f"- Candidate-plan cap: {config.max_plans}",
        "",
        "## Candidate plans",
        "",
        "| Tables | Optimizer | Queries | Mean | Min | Max | Cap frequency |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in workload_rows:
        lines.append(
            f"| {row['table_count']} | {row['optimizer']} | {row['query_templates']} | "
            f"{float(row['mean_candidate_plans']):.2f} | {row['min_candidate_plans']} | "
            f"{row['max_candidate_plans']} | {float(row['cap_frequency']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Ridge fidelity (out-of-query GroupKFold)",
            "",
            "Values are means across seeds; brackets show the bootstrap 95% CI across seeds.",
            "",
            "| Optimizer | Size | Distribution | Seeds | R² [95% CI] | MAE log [95% CI] | Kendall τ | Top-1 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fidelity_across_seeds:
        lines.append(
            f"| {row['optimizer']} | {row['size_label']} | {row['distribution']} | "
            f"{row['seed_repetitions']} | {formatted(row, 'r2_mean')} "
            f"[{formatted(row, 'r2_ci95_low')}, {formatted(row, 'r2_ci95_high')}] | "
            f"{formatted(row, 'mae_log_mean')} "
            f"[{formatted(row, 'mae_log_ci95_low')}, {formatted(row, 'mae_log_ci95_high')}] | "
            f"{formatted_rank(row, 'kendall_tau_mean')} | "
            f"{formatted_rank(row, 'top1_agreement_mean')} |"
        )
    lines.extend(
        [
            "",
            "An em dash means rank agreement was not defined in every seed because candidate target costs were tied.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_research_experiment(
    output_dir: str,
    config: ExperimentConfig | None = None,
    verbose: bool = True,
) -> ExperimentOutputs:
    config = config or ExperimentConfig()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = generate_workload(config.max_orders_per_table_count)
    if config.query_limit is not None:
        specs = specs[: config.query_limit]

    feature_rows, candidate_rows = _collect_raw_data(config, specs, verbose)
    fidelity_rows, ranking_rows, shap_rows, fit_time_rows = _evaluate_runs(
        feature_rows, config
    )
    fidelity_across_seeds = _aggregate_fidelity_across_seeds(fidelity_rows, config)
    global_shap, per_query_shap = _aggregate_shap(shap_rows, config)
    workload_rows, scalability_rows = _summaries(specs, candidate_rows)

    paths = {
        "raw_features_csv": output / "raw_plan_features.csv",
        "candidate_plans_csv": output / "candidate_plans.csv",
        "ridge_fidelity_csv": output / "ridge_fidelity.csv",
        "ridge_fidelity_across_seeds_csv": output / "ridge_fidelity_across_seeds.csv",
        "query_ranking_csv": output / "query_ranking_agreement.csv",
        "shap_per_run_csv": output / "shap_per_run.csv",
        "shap_stability_csv": output / "shap_stability.csv",
        "shap_stability_per_query_csv": output / "shap_stability_per_query.csv",
        "scalability_csv": output / "scalability.csv",
        "workload_summary_csv": output / "workload_summary.csv",
        "optimizer_details_csv": output / "optimizer_details.csv",
    }
    _write_csv(paths["raw_features_csv"], feature_rows)
    _write_csv(paths["candidate_plans_csv"], candidate_rows)
    _write_csv(paths["ridge_fidelity_csv"], fidelity_rows)
    _write_csv(paths["ridge_fidelity_across_seeds_csv"], fidelity_across_seeds)
    _write_csv(paths["query_ranking_csv"], ranking_rows)
    _write_csv(paths["shap_per_run_csv"], shap_rows)
    _write_csv(paths["shap_stability_csv"], global_shap)
    _write_csv(paths["shap_stability_per_query_csv"], per_query_shap)
    _write_csv(paths["scalability_csv"], scalability_rows)
    _write_csv(paths["workload_summary_csv"], workload_rows)
    _write_csv(paths["optimizer_details_csv"], _optimizer_details())

    paper_path = output / "PAPER_RESULTS.md"
    paper_path.write_text(
        _paper_markdown(config, specs, fidelity_across_seeds, workload_rows),
        encoding="utf-8",
    )
    manifest_path = output / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "query_count": len(specs),
                "feature_rows": len(feature_rows),
                "candidate_observations": len(candidate_rows),
                "ridge_runs": len(fidelity_rows),
                "fidelity_across_seed_groups": len(fidelity_across_seeds),
                "ridge_fit_times": fit_time_rows,
                "ranking_policy": (
                    "queries with constant target cost are excluded from ranking metrics; "
                    "missing rank metrics are reported as not applicable"
                ),
                "scope": {
                    "claim": "proof_of_concept",
                    "runtime_validation": "separate experiment",
                    "dba_user_study": "not performed",
                    "modern_optimizers": "future work",
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ExperimentOutputs(
        output_dir=str(output),
        paper_results_md=paper_path.name,
        experiment_manifest_json=manifest_path.name,
        **{name: path.name for name, path in paths.items()},
    )


def recompute_from_raw(output_dir: str, bootstrap_iterations: int | None = None) -> ExperimentOutputs:
    """Recalculate derived metrics without regenerating datasets or plans."""
    output = Path(output_dir)
    manifest_path = output / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe el manifiesto: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_config = manifest["config"]
    config = ExperimentConfig(
        seeds=tuple(raw_config["seeds"]),
        sizes=tuple(raw_config["sizes"]),
        distributions=tuple(raw_config["distributions"]),
        optimizers=tuple(raw_config["optimizers"]),
        max_plans=int(raw_config["max_plans"]),
        max_orders_per_table_count=int(raw_config["max_orders_per_table_count"]),
        bootstrap_iterations=(
            int(bootstrap_iterations)
            if bootstrap_iterations is not None
            else int(raw_config["bootstrap_iterations"])
        ),
        top_k=int(raw_config["top_k"]),
        query_limit=raw_config.get("query_limit"),
    )
    config.validate()

    def read_rows(path: Path) -> List[Dict[str, object]]:
        with path.open(newline="", encoding="utf-8") as stream:
            return [dict(row) for row in csv.DictReader(stream)]

    feature_rows = read_rows(output / "raw_plan_features.csv")
    candidate_rows = read_rows(output / "candidate_plans.csv")
    specs = generate_workload(config.max_orders_per_table_count)
    if config.query_limit is not None:
        specs = specs[: int(config.query_limit)]

    fidelity_rows, ranking_rows, shap_rows, fit_time_rows = _evaluate_runs(
        feature_rows, config
    )
    fidelity_across_seeds = _aggregate_fidelity_across_seeds(fidelity_rows, config)
    global_shap, per_query_shap = _aggregate_shap(shap_rows, config)
    workload_rows, scalability_rows = _summaries(specs, candidate_rows)
    derived = {
        "ridge_fidelity_csv": output / "ridge_fidelity.csv",
        "ridge_fidelity_across_seeds_csv": output / "ridge_fidelity_across_seeds.csv",
        "query_ranking_csv": output / "query_ranking_agreement.csv",
        "shap_per_run_csv": output / "shap_per_run.csv",
        "shap_stability_csv": output / "shap_stability.csv",
        "shap_stability_per_query_csv": output / "shap_stability_per_query.csv",
        "scalability_csv": output / "scalability.csv",
        "workload_summary_csv": output / "workload_summary.csv",
        "optimizer_details_csv": output / "optimizer_details.csv",
    }
    _write_csv(derived["ridge_fidelity_csv"], fidelity_rows)
    _write_csv(derived["ridge_fidelity_across_seeds_csv"], fidelity_across_seeds)
    _write_csv(derived["query_ranking_csv"], ranking_rows)
    _write_csv(derived["shap_per_run_csv"], shap_rows)
    _write_csv(derived["shap_stability_csv"], global_shap)
    _write_csv(derived["shap_stability_per_query_csv"], per_query_shap)
    _write_csv(derived["scalability_csv"], scalability_rows)
    _write_csv(derived["workload_summary_csv"], workload_rows)
    _write_csv(derived["optimizer_details_csv"], _optimizer_details())
    paper_path = output / "PAPER_RESULTS.md"
    paper_path.write_text(
        _paper_markdown(config, specs, fidelity_across_seeds, workload_rows),
        encoding="utf-8",
    )
    manifest["config"] = asdict(config)
    manifest["ridge_runs"] = len(fidelity_rows)
    manifest["fidelity_across_seed_groups"] = len(fidelity_across_seeds)
    manifest["ridge_fit_times"] = fit_time_rows
    manifest["ranking_policy"] = (
        "queries with constant target cost are excluded from ranking metrics; "
        "missing rank metrics are reported as not applicable"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ExperimentOutputs(
        output_dir=str(output),
        raw_features_csv="raw_plan_features.csv",
        candidate_plans_csv="candidate_plans.csv",
        paper_results_md=paper_path.name,
        experiment_manifest_json=manifest_path.name,
        **{name: path.name for name, path in derived.items()},
    )


def _csv_tuple(raw: str, cast=str) -> Tuple:
    return tuple(cast(item.strip()) for item in raw.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta la evaluación reproducible del paper")
    parser.add_argument("--output-dir", default="databases/research_evaluation")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--sizes", default="small,medium,large")
    parser.add_argument("--distributions", default="uniform,skewed,correlated")
    parser.add_argument("--optimizers", default="selinger,bayes")
    parser.add_argument("--max-plans", type=int, default=50)
    parser.add_argument("--orders", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Recalcula métricas usando los CSV existentes, sin regenerar planes",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(
        seeds=_csv_tuple(args.seeds, int),
        sizes=_csv_tuple(args.sizes),
        distributions=_csv_tuple(args.distributions),
        optimizers=_csv_tuple(args.optimizers),
        max_plans=args.max_plans,
        max_orders_per_table_count=args.orders,
        bootstrap_iterations=args.bootstrap,
        query_limit=args.query_limit,
    )
    if args.reuse_raw:
        outputs = recompute_from_raw(args.output_dir, args.bootstrap)
    else:
        outputs = run_research_experiment(
            args.output_dir, config=config, verbose=not args.quiet
        )
    print(json.dumps(asdict(outputs), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
