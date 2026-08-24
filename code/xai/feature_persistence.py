"""
XAI: Persistencia de features para análisis posterior.

Guarda features estimadas (extract_plan_features) y features reales (RuntimeTrace)
en formato CSV para estudio y validación offline.

Uso típico:
    # Después de generar planes y matriz de features
    save_estimated_features_table(
        plans, X, feature_names, feature_dicts,
        dataset_name="mi_experimento",
        optimizer_name="BayesOptimizer",
    )

    # Después de ejecutar con trace
    save_runtime_features_table(
        runtime_traces, plan_orders,
        dataset_name="mi_experimento",
    )
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sql.parser import SelectStatement

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "databases/feature_analysis"


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia de features estimadas
# ─────────────────────────────────────────────────────────────────────────────

def save_estimated_features_table(
    plans: List[SelectStatement],
    X: np.ndarray,
    feature_names: List[str],
    feature_dicts: List[Dict[str, float]],
    dataset_name: str,
    optimizer_name: str = "Unknown",
    query_id: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Persiste features estimadas (de extract_plan_features) en CSV.

    Genera un archivo CSV con una fila por plan candidato y columnas para:
    - query_id: identificador de la consulta
    - timestamp: momento de generación
    - optimizer: nombre del optimizador usado
    - plan_id: índice del plan (0-based)
    - plan_order: orden de tablas (ej: "users->orders->products")
    - <feature_1>, <feature_2>, ... : valores de cada feature

    Args:
        plans: lista de planes candidatos (SelectStatement).
        X: matriz de features (n_plans × n_features).
        feature_names: nombres de features en el orden de X.
        feature_dicts: lista de dicts de features (uno por plan).
        dataset_name: nombre del dataset/experimento (usado en nombre de archivo).
        optimizer_name: nombre del optimizador usado.
        query_id: identificador opcional de la consulta.
        output_dir: directorio de salida (default: databases/feature_analysis).
        metadata: metadata adicional a guardar en archivo JSON acompañante.

    Returns:
        Ruta absoluta del archivo CSV generado.
    """
    # Crear directorio si no existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if query_id:
        filename = f"estimated_{dataset_name}_q{query_id}_{timestamp}.csv"
        meta_filename = f"estimated_{dataset_name}_q{query_id}_{timestamp}_meta.json"
    else:
        filename = f"estimated_{dataset_name}_{timestamp}.csv"
        meta_filename = f"estimated_{dataset_name}_{timestamp}_meta.json"

    csv_path = output_path / filename
    meta_path = output_path / meta_filename

    # Preparar datos
    rows = []
    for plan_id, (plan, feature_dict) in enumerate(zip(plans, feature_dicts)):
        plan_order = _format_plan_order(plan)
        row = {
            "query_id": query_id or "default",
            "timestamp": timestamp,
            "optimizer": optimizer_name,
            "plan_id": plan_id,
            "plan_order": plan_order,
        }
        # Agregar features en orden
        for fname in feature_names:
            row[fname] = feature_dict.get(fname, 0.0)
        rows.append(row)

    # Escribir CSV
    if rows:
        fieldnames = ["query_id", "timestamp", "optimizer", "plan_id", "plan_order"] + feature_names
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Features estimadas guardadas en: {csv_path}")
        logger.info(f"  Planes: {len(rows)}, Features: {len(feature_names)}")

        # Guardar metadata
        meta = {
            "dataset_name": dataset_name,
            "query_id": query_id,
            "optimizer": optimizer_name,
            "timestamp": timestamp,
            "n_plans": len(rows),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "csv_file": filename,
        }
        if metadata:
            meta["additional"] = metadata

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata guardada en: {meta_path}")
    else:
        logger.warning("No hay planes para guardar.")

    return str(csv_path.absolute())


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia de features reales (runtime)
# ─────────────────────────────────────────────────────────────────────────────

def save_runtime_features_table(
    runtime_traces: List["RuntimeTrace"],  # type: ignore
    plan_orders: List[List[str]],
    dataset_name: str,
    query_id: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Persiste features reales (de RuntimeTrace) en CSV.

    Genera un archivo CSV con una fila por plan ejecutado y columnas para:
    - query_id: identificador de la consulta
    - timestamp: momento de captura
    - plan_id: índice del plan (0-based)
    - plan_order: orden de tablas ejecutado
    - base_filtered_rows, sum_intermediate_rows, max_intermediate_rows, ...
    - join_1_input_rows, join_1_table_rows, join_1_output_rows, ...

    Args:
        runtime_traces: lista de RuntimeTrace (uno por plan ejecutado).
        plan_orders: lista del orden de tablas por plan.
        dataset_name: nombre del dataset/experimento.
        query_id: identificador opcional de la consulta.
        output_dir: directorio de salida.
        metadata: metadata adicional.

    Returns:
        Ruta absoluta del archivo CSV generado.
    """
    # Crear directorio si no existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if query_id:
        filename = f"runtime_{dataset_name}_q{query_id}_{timestamp}.csv"
        meta_filename = f"runtime_{dataset_name}_q{query_id}_{timestamp}_meta.json"
    else:
        filename = f"runtime_{dataset_name}_{timestamp}.csv"
        meta_filename = f"runtime_{dataset_name}_{timestamp}_meta.json"

    csv_path = output_path / filename
    meta_path = output_path / meta_filename

    # Convertir traces a features
    rows = []
    all_feature_names = set()
    for plan_id, (trace, plan_order) in enumerate(zip(runtime_traces, plan_orders)):
        features = trace.to_features()
        all_feature_names.update(features.keys())

        row = {
            "query_id": query_id or "default",
            "timestamp": timestamp,
            "plan_id": plan_id,
            "plan_order": "->".join(plan_order),
            "final_rows": trace.final_rows,
        }
        row.update(features)
        rows.append(row)

    # Escribir CSV
    if rows:
        # Ordenar features para consistencia
        feature_names_sorted = sorted(all_feature_names)
        fieldnames = ["query_id", "timestamp", "plan_id", "plan_order", "final_rows"] + feature_names_sorted

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Features runtime guardadas en: {csv_path}")
        logger.info(f"  Planes ejecutados: {len(rows)}, Features: {len(feature_names_sorted)}")

        # Guardar metadata
        meta = {
            "dataset_name": dataset_name,
            "query_id": query_id,
            "timestamp": timestamp,
            "n_plans": len(rows),
            "n_features": len(feature_names_sorted),
            "feature_names": feature_names_sorted,
            "csv_file": filename,
        }
        if metadata:
            meta["additional"] = metadata

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata guardada en: {meta_path}")
    else:
        logger.warning("No hay traces runtime para guardar.")

    return str(csv_path.absolute())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_plan_order(plan: SelectStatement) -> str:
    """Formatea el orden de tablas de un plan como string legible."""
    tables = [plan.from_table]
    if plan.join_clauses:
        tables.extend([jc.table for jc in plan.join_clauses])
    return "->".join(tables)


# ─────────────────────────────────────────────────────────────────────────────
# Función de conveniencia: guardar ambos tipos
# ─────────────────────────────────────────────────────────────────────────────

def save_complete_analysis(
    plans: List[SelectStatement],
    X: np.ndarray,
    feature_names: List[str],
    feature_dicts: List[Dict[str, float]],
    runtime_traces: Optional[List["RuntimeTrace"]] = None,  # type: ignore
    dataset_name: str = "experiment",
    optimizer_name: str = "Unknown",
    query_id: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Guarda features estimadas y runtime en un solo llamado.

    Returns:
        Dict con rutas de archivos generados: {"estimated": path, "runtime": path}
    """
    result = {}

    # Guardar features estimadas
    estimated_path = save_estimated_features_table(
        plans, X, feature_names, feature_dicts,
        dataset_name, optimizer_name, query_id, output_dir, metadata,
    )
    result["estimated"] = estimated_path

    # Guardar features runtime si están disponibles
    if runtime_traces:
        plan_orders = [
            [plan.from_table] + ([jc.table for jc in plan.join_clauses] if plan.join_clauses else [])
            for plan in plans
        ]
        runtime_path = save_runtime_features_table(
            runtime_traces, plan_orders,
            dataset_name, query_id, output_dir, metadata,
        )
        result["runtime"] = runtime_path

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Función helper para integración con explain_optimizer_decision
# ─────────────────────────────────────────────────────────────────────────────

def explain_and_persist(
    stmt: SelectStatement,
    optimizer: "QueryOptimizer",  # type: ignore
    dataset_name: str,
    query_id: Optional[str] = None,
    max_plans: int = 50,
    top_k_features: int = 5,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple["Explanation", Dict[str, str]]:  # type: ignore
    """
    Wrapper que combina explain_optimizer_decision + persistencia automática.

    Genera la explicación del optimizador y persiste automáticamente las
    features estimadas en CSV. Opcionalmente, si se ejecutan los planes
    con trace, también se pueden persistir las features reales.

    Args:
        stmt: consulta SQL parseada.
        optimizer: instancia del optimizador.
        dataset_name: nombre del dataset/experimento.
        query_id: identificador de la consulta (opcional).
        max_plans: cantidad máxima de planes candidatos.
        top_k_features: cantidad de top features SHAP.
        output_dir: directorio de salida para CSVs.
        metadata: metadata adicional a guardar.

    Returns:
        (explanation, file_paths): tupla con la explicación y dict de rutas
        de archivos generados {"estimated": path}.

    Ejemplo:
        from xai.feature_persistence import explain_and_persist
        
        explanation, paths = explain_and_persist(
            stmt=parsed_query,
            optimizer=db.optimizer,
            dataset_name="imdb_join",
            query_id="q3",
        )
        
        print(explanation.natural_language_summary)
        print(f"Features guardadas en: {paths['estimated']}")
    """
    # Importar aquí para evitar dependencia circular
    from xai.optimizer_explainer import (
        generate_candidate_plans,
        build_feature_matrix,
        explain_optimizer_decision,
    )
    from sql.optimizer import BayesOptimizer

    # Generar features (mismo flujo que explain_optimizer_decision)
    fix_base = isinstance(optimizer, BayesOptimizer)
    candidates = generate_candidate_plans(stmt, max_plans=max_plans, fix_base_table=fix_base)
    X, feature_names, feature_dicts = build_feature_matrix(candidates, stmt, optimizer)

    # Persistir features estimadas
    file_paths = {}
    if len(candidates) > 0:
        estimated_path = save_estimated_features_table(
            plans=candidates,
            X=X,
            feature_names=feature_names,
            feature_dicts=feature_dicts,
            dataset_name=dataset_name,
            optimizer_name=type(optimizer).__name__,
            query_id=query_id,
            output_dir=output_dir,
            metadata=metadata,
        )
        file_paths["estimated"] = estimated_path

    # Generar explicación
    explanation = explain_optimizer_decision(
        stmt=stmt,
        optimizer=optimizer,
        max_plans=max_plans,
        top_k_features=top_k_features,
    )

    return explanation, file_paths

