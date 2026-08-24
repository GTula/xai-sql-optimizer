"""Demo de importancia GLOBAL de features del optimizador.

La demo usa todas las consultas de ``docs/CONSULTAS_100.sql`` mediante la
funcion batch existente. Para cada optimizador ajusta un proxy global del
costo y reporta mean(|SHAP|), no el SHAP local de un unico plan.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from xai.batch_explainer import run_batch_analysis


QUERIES_FILE = os.path.join(ROOT, "docs", "CONSULTAS_100.sql")
OUTPUT_DIR = os.path.join(ROOT, "databases", "feature_analysis")
METADATA_COLUMNS = {"query_id", "optimizer", "plan_id", "plan_order"}
TARGET = "estimated_cost"

# Ante features casi equivalentes conservamos primero la que tiene una
# interpretacion mas directa del trabajo efectuado por el plan.
FEATURE_PRIORITY = [
    "sum_intermediate_rows",
    "max_intermediate_rows",
    "avg_intermediate_rows",
    "most_expensive_join_cost",
    "most_expensive_join_position",
    "base_filtered_rows",
    "first_table_rows",
    "max_table_rows",
    "index_usage_count",
    "full_scan_count",
    "join_count",
    "filter_count",
    "predicate_count",
    "aggregation_count",
    "projected_column_count",
    "ast_depth",
    "ast_node_count",
    "table_count",
]


def _read_feature_table(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        names = [
            name for name in (reader.fieldnames or [])
            if name not in METADATA_COLUMNS and name != TARGET
        ]
    return rows, names


def _priority(name: str) -> Tuple[int, str]:
    if name in FEATURE_PRIORITY:
        return FEATURE_PRIORITY.index(name), name
    # Las metricas especificas de cada join quedan despues de sus agregados.
    return len(FEATURE_PRIORITY), name


def _remove_redundant_features(
    X: np.ndarray,
    names: Sequence[str],
    correlation_threshold: float = 0.98,
) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
    """Quita constantes y deja un representante por grupo correlacionado."""
    varying = [i for i in range(X.shape[1]) if np.ptp(X[:, i]) > 1e-12]
    if not varying:
        return np.empty((len(X), 0)), [], {}

    X = X[:, varying]
    names = [names[i] for i in varying]
    order = sorted(range(len(names)), key=lambda i: _priority(names[i]))
    kept: List[int] = []
    aliases: Dict[str, List[str]] = defaultdict(list)

    for idx in order:
        representative = None
        for kept_idx in kept:
            corr = np.corrcoef(X[:, idx], X[:, kept_idx])[0, 1]
            if np.isfinite(corr) and abs(corr) >= correlation_threshold:
                representative = names[kept_idx]
                break
        if representative is None:
            kept.append(idx)
        else:
            aliases[representative].append(names[idx])

    return X[:, kept], [names[i] for i in kept], dict(aliases)


def compute_global_importance(
    rows: Sequence[Dict[str, str]], feature_names: Sequence[str], optimizer: str
) -> Tuple[List[Tuple[str, float]], float, Dict[str, List[str]], int]:
    """Calcula mean absolute SHAP sobre todos los planes de un optimizador."""
    selected = [row for row in rows if row["optimizer"] == optimizer]
    if len(selected) < 2:
        raise ValueError(f"No hay suficientes planes para {optimizer}")

    X = np.asarray(
        [[float(row.get(name, 0.0) or 0.0) for name in feature_names] for row in selected],
        dtype=float,
    )
    y = np.asarray([float(row[TARGET]) for row in selected], dtype=float)
    X, names, aliases = _remove_redundant_features(X, feature_names)
    if not names:
        raise ValueError(f"No hay features variables para {optimizer}")

    # log1p evita que las cardinalidades enormes dominen numericamente el proxy.
    X_model = np.sign(X) * np.log1p(np.abs(X))
    y_model = np.log1p(np.maximum(y, 0.0))

    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 4, 33)),
    )
    model.fit(X_model, y_model)
    prediction = model.predict(X_model)

    # Para un modelo lineal, SHAP en el espacio estandarizado es exactamente
    # coeficiente * desviacion respecto al background medio. La agregacion
    # mean(abs(.)) produce importancia global, siempre no negativa.
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridgecv"]
    Z = scaler.transform(X_model)
    shap_values = (Z - Z.mean(axis=0)) * ridge.coef_
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    total = float(mean_abs_shap.sum())
    percentages = (
        mean_abs_shap / total * 100.0 if total > 0 else np.zeros_like(mean_abs_shap)
    )
    ranking = sorted(zip(names, percentages.tolist()), key=lambda item: item[1], reverse=True)
    return ranking, float(r2_score(y_model, prediction)), aliases, len(selected)


def plot_global_importance(
    ranking: Sequence[Tuple[str, float]], optimizer: str, top_k: int = 12
) -> str:
    import matplotlib.pyplot as plt

    top = list(ranking[:top_k])[::-1]
    names = [name for name, _ in top]
    values = [value for _, value in top]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(names, values, color="#2f6f9f", alpha=0.9)
    ax.set_xlabel("Importancia global: mean(|SHAP|) normalizada (%)")
    ax.set_title(f"Importancia global de features - {optimizer.upper()}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"shap_global_{optimizer}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    print("Analisis GLOBAL de features")
    print(f"Archivo de consultas: {QUERIES_FILE}")

    # Esta es la funcion existente encargada de leer y procesar el archivo.
    summary = run_batch_analysis(
        queries_file=QUERIES_FILE,
        output_dir=OUTPUT_DIR,
        dataset_name="CONSULTAS_100",
        optimizers=["selinger", "bayes"],
        max_plans=50,
        verbose=True,
    )
    if not summary.estimated_features_csv:
        raise RuntimeError("El batch no genero la tabla de features estimadas")

    feature_path = os.path.join(summary.output_directory, summary.estimated_features_csv)
    rows, feature_names = _read_feature_table(feature_path)

    for optimizer in summary.optimizers:
        ranking, r2, aliases, plan_count = compute_global_importance(
            rows, feature_names, optimizer
        )
        print(f"\n{optimizer.upper()}: {plan_count} planes, R2 global={r2:.3f}")
        print("  Importancia global (porcentaje de mean absolute SHAP):")
        for name, percentage in ranking[:12]:
            suffix = ""
            if aliases.get(name):
                suffix = f" [representa tambien: {', '.join(aliases[name])}]"
            print(f"  - {name:<32} {percentage:6.2f}%{suffix}")
        image_path = plot_global_importance(ranking, optimizer)
        print(f"  Grafico: {image_path}")

    print("\nLos porcentajes son magnitudes globales y suman 100%; no expresan direccion local.")


if __name__ == "__main__":
    main()
