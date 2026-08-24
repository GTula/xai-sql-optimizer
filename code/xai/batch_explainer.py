"""
XAI: Análisis batch de consultas para evaluación del optimizador.

Ejecuta múltiples consultas desde un archivo, genera explicaciones con ambos
optimizadores (Selinger y Bayes), y persiste features estimadas y reales
en archivos CSV consolidados para análisis posterior.

Uso:
    from xai.batch_explainer import run_batch_analysis
    
    summary = run_batch_analysis(
        queries_file="docs/CONSULTAS_100.sql",
        output_dir="databases/feature_analysis",
        dataset_name="experimento_100q"
    )

Archivos generados:
    - summary_{dataset_name}_{timestamp}.csv: resumen de todas las consultas
    - estimated_{dataset_name}_batch_{timestamp}.csv: features estimadas consolidadas
    - runtime_{dataset_name}_batch_{timestamp}.csv: features reales consolidadas
    - metadata_{dataset_name}_{timestamp}.json: metadata del batch
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── ajustar path para poder importar desde la raíz del proyecto ─────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from database import Database
from sql.parser import Parser
from sql.optimizer import BayesOptimizer
from xai.optimizer_explainer import (
    explain_optimizer_decision,
    generate_candidate_plans,
    build_feature_matrix,
)
from xai.runtime_trace import RuntimeTrace, compare_shap_vs_runtime
from xai.feature_persistence import (
    save_estimated_features_table,
    save_runtime_features_table,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Resultado del análisis de una consulta individual."""
    query_id: int
    query_sql: str
    optimizer: str
    success: bool
    error: Optional[str]
    
    # Métricas de optimización
    chosen_plan: Optional[List[str]]
    estimated_cost: Optional[float]
    num_candidate_plans: Optional[int]
    
    # Métricas de ejecución
    execution_time_ms: Optional[float]
    rows_returned: Optional[int]
    
    # Métricas de explicabilidad
    shap_available: bool
    fidelity_score: Optional[float]
    top_features: Optional[List[str]]
    
    # Timestamp
    timestamp: str


@dataclass
class EstimatedFeaturesBatch:
    """Features estimadas para acumulación batch."""
    query_id: int
    optimizer: str
    plans: List[SelectStatement]
    X: np.ndarray
    feature_names: List[str]
    feature_dicts: List[Dict[str, float]]


@dataclass
class RuntimeFeaturesBatch:
    """Features reales para acumulación batch."""
    query_id: int
    optimizer: str
    trace: RuntimeTrace
    plan_order: List[str]
    fidelity_score: float


@dataclass
class BatchSummary:
    """Resumen del análisis batch completo."""
    dataset_name: str
    queries_file: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    optimizers: List[str]
    start_time: str
    end_time: str
    total_duration_seconds: float
    output_directory: str
    summary_csv: str
    metadata_json: str
    estimated_features_csv: Optional[str]
    runtime_features_csv: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Database setup
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_db(base_dir: str, optimizer_type: str) -> Database:
    """
    Crea una base de datos temporal con 3 tablas y datos de prueba.
    
    Estructura:
    - fact: 3000 filas (id, dept_id, loc_id)
    - departments: 1 fila (id, dept_name)
    - locations: 300 filas (id, city)
    
    Distribución diseñada para crear selectividad variada en JOINs.
    """
    db_path = os.path.join(base_dir, f"{optimizer_type}.db")
    db = Database(db_path, optimizer_type=optimizer_type)

    # Crear tablas
    db.execute("CREATE TABLE fact        (id INTEGER, dept_id INTEGER, loc_id INTEGER)")
    db.execute("CREATE TABLE departments (id INTEGER, dept_name VARCHAR(50))")
    db.execute("CREATE TABLE locations   (id INTEGER, city VARCHAR(50))")

    # Poblar departments (1 fila: baja cardinalidad)
    db.execute("INSERT INTO departments VALUES (1, 'Engineering')")

    # Poblar locations (300 filas: cardinalidad media)
    for loc_id in range(1, 301):
        db.execute(f"INSERT INTO locations VALUES ({loc_id}, 'City {loc_id}')")

    # Poblar fact (3000 filas: alta cardinalidad)
    # dept_id=1 para primeras 3 filas, dept_id=999 para el resto (crea skew)
    for i in range(1, 3001):
        dept_id = 1 if i <= 3 else 999
        loc_id  = ((i - 1) % 300) + 1
        db.execute(f"INSERT INTO fact VALUES ({i}, {dept_id}, {loc_id})")

    # Analizar estadísticas para el optimizador
    db.execute("ANALYSE")
    
    return db


# ─────────────────────────────────────────────────────────────────────────────
# Query processing
# ─────────────────────────────────────────────────────────────────────────────

def process_single_query(
    query_id: int,
    sql: str,
    db: Database,
    optimizer_type: str,
    max_plans: int = 50,
) -> Tuple[QueryResult, Optional[EstimatedFeaturesBatch], Optional[RuntimeFeaturesBatch]]:
    """
    Procesa una consulta individual: parsea, optimiza, explica y retorna datos para persistencia batch.
    
    Returns:
        Tupla (QueryResult, EstimatedFeaturesBatch o None, RuntimeFeaturesBatch o None)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = QueryResult(
        query_id=query_id,
        query_sql=sql.strip(),
        optimizer=optimizer_type,
        success=False,
        error=None,
        chosen_plan=None,
        estimated_cost=None,
        num_candidate_plans=None,
        execution_time_ms=None,
        rows_returned=None,
        shap_available=False,
        fidelity_score=None,
        top_features=None,
        timestamp=timestamp,
    )
    
    estimated_batch = None
    runtime_batch = None
    
    try:
        # 1. Parsear consulta
        stmt = Parser.parse(sql)
        optimizer = db.engine.optimizer
        
        # 2. Generar planes candidatos y features
        fix_base = isinstance(optimizer, BayesOptimizer)
        candidates = generate_candidate_plans(stmt, max_plans=max_plans, fix_base_table=fix_base)
        X, feature_names, feature_dicts = build_feature_matrix(candidates, stmt, optimizer)
        
        result.num_candidate_plans = len(candidates)
        
        # 3. Guardar features estimadas para batch
        estimated_batch = EstimatedFeaturesBatch(
            query_id=query_id,
            optimizer=optimizer_type,
            plans=candidates,
            X=X,
            feature_names=feature_names,
            feature_dicts=feature_dicts,
        )
        
        # 4. Generar explicación
        explanation = explain_optimizer_decision(
            stmt,
            optimizer,
            max_plans=max_plans,
            top_k_features=5,
        )
        
        result.chosen_plan = explanation.chosen_plan
        result.estimated_cost = explanation.selected_plan_score
        result.shap_available = explanation.shap_used
        result.top_features = explanation.top_features[:3]  # Top 3
        
        # 5. Ejecutar con trace y capturar features reales
        if explanation.shap_used:
            try:
                start_time = time.perf_counter()
                planned_stmt = optimizer.optimize(stmt)
                rows, raw_trace = db.engine.execute_plan_with_trace(planned_stmt)
                end_time = time.perf_counter()
                
                result.execution_time_ms = (end_time - start_time) * 1000
                result.rows_returned = len(rows) if rows else 0
                
                trace = RuntimeTrace.from_engine_trace(raw_trace)
                
                # Comparar SHAP vs Runtime para calcular fidelity
                report = compare_shap_vs_runtime(
                    shap_values=explanation.shap_values_for_selected_plan,
                    estimated_features=explanation.feature_values_for_selected_plan,
                    runtime_trace=trace,
                    top_k=5,
                    fidelity_threshold_pct=20.0,
                )
                result.fidelity_score = report.overall_fidelity_score
                
                # Guardar features runtime para batch
                runtime_batch = RuntimeFeaturesBatch(
                    query_id=query_id,
                    optimizer=optimizer_type,
                    trace=trace,
                    plan_order=explanation.chosen_plan,
                    fidelity_score=report.overall_fidelity_score,
                )
                    
            except ValueError as exc:
                # Plan no ejecutable (limitación conocida con Selinger)
                logger.debug(f"[Q{query_id}] Plan no ejecutable: {exc}")
                result.execution_time_ms = None
                result.fidelity_score = None
        
        result.success = True
        
    except Exception as e:
        result.error = str(e)
        logger.error(f"[Q{query_id}] Error procesando consulta: {e}")
    
    return result, estimated_batch, runtime_batch


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_analysis(
    queries_file: str,
    output_dir: str = "databases/feature_analysis",
    dataset_name: Optional[str] = None,
    optimizers: Optional[List[str]] = None,
    max_plans: int = 50,
    verbose: bool = True,
) -> BatchSummary:
    """
    Ejecuta análisis batch de consultas desde un archivo.
    
    Args:
        queries_file: Ruta al archivo con consultas (una por línea).
        output_dir: Directorio donde guardar los resultados.
        dataset_name: Nombre del dataset (default: nombre del archivo).
        optimizers: Lista de optimizadores a usar (default: ["selinger", "bayes"]).
        max_plans: Número máximo de planes candidatos por consulta.
        verbose: Mostrar progreso por consola.
    
    Returns:
        BatchSummary con resumen del análisis.
    
    Genera archivos consolidados:
        - summary_{dataset_name}_{timestamp}.csv: resumen de todas las consultas
        - metadata_{dataset_name}_{timestamp}.json: metadata del batch
        - estimated_{dataset_name}_batch_{timestamp}.csv: features estimadas de todas las consultas
        - runtime_{dataset_name}_batch_{timestamp}.csv: features reales de todas las consultas
    """
    start_time = datetime.now()
    
    # Validar archivo de entrada
    if not os.path.exists(queries_file):
        raise FileNotFoundError(f"Archivo de consultas no encontrado: {queries_file}")
    
    # Configuración por defecto
    if dataset_name is None:
        dataset_name = Path(queries_file).stem
    
    if optimizers is None:
        optimizers = ["selinger", "bayes"]
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Leer consultas
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip() and not line.strip().startswith("--")]
    
    total_queries = len(queries)
    
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Batch Analysis: {dataset_name}")
        print(f"{'─' * 60}")
        print(f"Consultas        : {total_queries}")
        print(f"Optimizadores    : {', '.join(optimizers)}")
        print(f"Max planes/query : {max_plans}")
        print(f"Output directory : {output_dir}")
        print(f"{'─' * 60}\n")
    
    # Procesar consultas
    all_results: List[QueryResult] = []
    all_estimated_features: List[EstimatedFeaturesBatch] = []
    all_runtime_features: List[RuntimeFeaturesBatch] = []
    tmp_dir = tempfile.mkdtemp(prefix="batch_xai_")
    
    try:
        for opt_type in optimizers:
            # Crear DB una vez por optimizador
            if verbose:
                print(f"\n[{opt_type.upper()}] Creando base de datos temporal...")
            
            db = build_demo_db(tmp_dir, opt_type)
            
            try:
                for idx, sql in enumerate(queries, start=1):
                    if verbose:
                        print(f"[{opt_type.upper()}] Procesando query {idx}/{total_queries}...", end=" ")
                    
                    result, estimated_batch, runtime_batch = process_single_query(
                        query_id=idx,
                        sql=sql,
                        db=db,
                        optimizer_type=opt_type,
                        max_plans=max_plans,
                    )
                    
                    all_results.append(result)
                    
                    if estimated_batch:
                        all_estimated_features.append(estimated_batch)
                    
                    if runtime_batch:
                        all_runtime_features.append(runtime_batch)
                    
                    if verbose:
                        status = "✓" if result.success else "✗"
                        print(f"{status}")
                        
            finally:
                db.close()
    
    finally:
        # Limpiar directorio temporal
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Generar archivos de salida
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    summary_csv_path = os.path.join(output_dir, f"summary_{dataset_name}_{timestamp_str}.csv")
    metadata_json_path = os.path.join(output_dir, f"metadata_{dataset_name}_{timestamp_str}.json")
    
    # Escribir features consolidadas
    estimated_features_path = None
    runtime_features_path = None
    
    if all_estimated_features:
        estimated_features_path = os.path.join(
            output_dir, 
            f"estimated_{dataset_name}_batch_{timestamp_str}.csv"
        )
        _write_consolidated_estimated_features(all_estimated_features, estimated_features_path)
    
    if all_runtime_features:
        runtime_features_path = os.path.join(
            output_dir,
            f"runtime_{dataset_name}_batch_{timestamp_str}.csv"
        )
        _write_consolidated_runtime_features(all_runtime_features, runtime_features_path)
    
    # Escribir resumen
    _write_summary_csv(all_results, summary_csv_path)
    
    # Calcular estadísticas
    successful = sum(1 for r in all_results if r.success)
    failed = len(all_results) - successful
    
    summary = BatchSummary(
        dataset_name=dataset_name,
        queries_file=queries_file,
        total_queries=total_queries,
        successful_queries=successful,
        failed_queries=failed,
        optimizers=optimizers,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        total_duration_seconds=duration,
        output_directory=output_dir,
        summary_csv=os.path.basename(summary_csv_path),
        metadata_json=os.path.basename(metadata_json_path),
        estimated_features_csv=os.path.basename(estimated_features_path) if estimated_features_path else None,
        runtime_features_csv=os.path.basename(runtime_features_path) if runtime_features_path else None,
    )
    
    _write_metadata_json(summary, all_results, metadata_json_path)
    
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Resumen")
        print(f"{'─' * 60}")
        print(f"Total queries    : {total_queries}")
        print(f"Exitosas         : {successful}")
        print(f"Fallidas         : {failed}")
        print(f"Duración         : {duration:.2f}s")
        print(f"Archivos salida  :")
        print(f"  - {summary.summary_csv}")
        print(f"  - {summary.metadata_json}")
        if summary.estimated_features_csv:
            print(f"  - {summary.estimated_features_csv}")
        if summary.runtime_features_csv:
            print(f"  - {summary.runtime_features_csv}")
        print(f"{'─' * 60}\n")
    
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Output generation
# ─────────────────────────────────────────────────────────────────────────────

def _write_consolidated_estimated_features(
    batches: List[EstimatedFeaturesBatch],
    output_path: str,
) -> None:
    """
    Escribe CSV consolidado con features estimadas de todas las consultas.
    
    Formato: query_id, optimizer, plan_id, plan_order, feature_1, feature_2, ...
    """
    if not batches:
        return
    
    # Recolectar todos los nombres de features únicas
    all_feature_names = set()
    for batch in batches:
        all_feature_names.update(batch.feature_names)
    
    feature_names_sorted = sorted(all_feature_names)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # Cabeceras: metadata + features
        fieldnames = ["query_id", "optimizer", "plan_id", "plan_order"] + feature_names_sorted
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Escribir una fila por cada plan de cada consulta
        for batch in batches:
            for plan_idx, (plan, feat_dict) in enumerate(zip(batch.plans, batch.feature_dicts)):
                # Extraer orden de tablas del plan
                plan_order = _extract_plan_order(plan)
                
                # Crear fila con metadata
                row = {
                    "query_id": batch.query_id,
                    "optimizer": batch.optimizer,
                    "plan_id": plan_idx,
                    "plan_order": "->".join(plan_order),
                }
                
                # Agregar features (0.0 si no existe)
                for feat_name in feature_names_sorted:
                    row[feat_name] = feat_dict.get(feat_name, 0.0)
                
                writer.writerow(row)


def _write_consolidated_runtime_features(
    batches: List[RuntimeFeaturesBatch],
    output_path: str,
) -> None:
    """
    Escribe CSV consolidado con features reales de todas las consultas.
    
    Formato: query_id, optimizer, plan_order, rows_scanned_*, rows_produced_*, fidelity_score
    """
    if not batches:
        return
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # Recolectar nombres de features del primer trace para cabeceras
        first_trace = batches[0].trace
        feature_names = sorted(first_trace.to_features().keys())
        
        fieldnames = ["query_id", "optimizer", "plan_order", "fidelity_score"] + feature_names
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Escribir una fila por cada consulta ejecutada
        for batch in batches:
            features = batch.trace.to_features()
            
            row = {
                "query_id": batch.query_id,
                "optimizer": batch.optimizer,
                "plan_order": "->".join(batch.plan_order),
                "fidelity_score": batch.fidelity_score,
            }
            
            # Agregar features runtime
            for feat_name in feature_names:
                row[feat_name] = features.get(feat_name, 0.0)
            
            writer.writerow(row)


def _extract_plan_order(plan: SelectStatement) -> List[str]:
    """Extrae orden de tablas de un plan (SelectStatement)."""
    tables = [plan.from_table]
    if plan.join_clauses:
        tables.extend([jc.table for jc in plan.join_clauses])
    return tables


def _write_summary_csv(results: List[QueryResult], output_path: str) -> None:
    """Escribe CSV con resumen de todas las consultas."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_id",
            "optimizer",
            "success",
            "error",
            "chosen_plan",
            "estimated_cost",
            "num_candidate_plans",
            "execution_time_ms",
            "rows_returned",
            "shap_available",
            "fidelity_score",
            "top_features",
            "timestamp",
        ])
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                "query_id": r.query_id,
                "optimizer": r.optimizer,
                "success": r.success,
                "error": r.error or "",
                "chosen_plan": "->".join(r.chosen_plan) if r.chosen_plan else "",
                "estimated_cost": f"{r.estimated_cost:.4f}" if r.estimated_cost else "",
                "num_candidate_plans": r.num_candidate_plans or "",
                "execution_time_ms": f"{r.execution_time_ms:.2f}" if r.execution_time_ms else "",
                "rows_returned": r.rows_returned or "",
                "shap_available": r.shap_available,
                "fidelity_score": f"{r.fidelity_score:.2f}" if r.fidelity_score else "",
                "top_features": "|".join(r.top_features) if r.top_features else "",
                "timestamp": r.timestamp,
            })


def _write_metadata_json(
    summary: BatchSummary,
    results: List[QueryResult],
    output_path: str,
) -> None:
    """Escribe JSON con metadata del batch."""
    metadata = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
        "statistics": {
            "by_optimizer": {},
        }
    }
    
    # Calcular estadísticas por optimizador
    for opt in summary.optimizers:
        opt_results = [r for r in results if r.optimizer == opt]
        successful = sum(1 for r in opt_results if r.success)
        
        avg_cost = np.mean([r.estimated_cost for r in opt_results if r.estimated_cost is not None])
        avg_time = np.mean([r.execution_time_ms for r in opt_results if r.execution_time_ms is not None])
        avg_fidelity = np.mean([r.fidelity_score for r in opt_results if r.fidelity_score is not None])
        
        metadata["statistics"]["by_optimizer"][opt] = {
            "total_queries": len(opt_results),
            "successful": successful,
            "failed": len(opt_results) - successful,
            "avg_estimated_cost": float(avg_cost) if not np.isnan(avg_cost) else None,
            "avg_execution_time_ms": float(avg_time) if not np.isnan(avg_time) else None,
            "avg_fidelity_score": float(avg_fidelity) if not np.isnan(avg_fidelity) else None,
        }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point para ejecución desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Análisis batch de consultas SQL con optimizadores Selinger y Bayes."
    )
    parser.add_argument(
        "queries_file",
        help="Archivo con consultas SQL (una por línea)"
    )
    parser.add_argument(
        "--output-dir",
        default="databases/feature_analysis",
        help="Directorio de salida (default: databases/feature_analysis)"
    )
    parser.add_argument(
        "--dataset-name",
        help="Nombre del dataset (default: nombre del archivo)"
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["selinger", "bayes"],
        choices=["selinger", "bayes"],
        help="Optimizadores a usar (default: selinger bayes)"
    )
    parser.add_argument(
        "--max-plans",
        type=int,
        default=50,
        help="Máximo de planes candidatos por consulta (default: 50)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No mostrar progreso por consola"
    )
    
    args = parser.parse_args()
    
    try:
        summary = run_batch_analysis(
            queries_file=args.queries_file,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            optimizers=args.optimizers,
            max_plans=args.max_plans,
            verbose=not args.quiet,
        )
        
        print(f"\n✓ Análisis completado exitosamente")
        print(f"  Ver resultados en: {summary.output_directory}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
