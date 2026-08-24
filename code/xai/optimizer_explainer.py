"""
XAI: Explicabilidad del optimizador SQL usando SHAP.

Flujo principal:
  1. optimizer.optimize(stmt)  → plan elegido (AST reordenado)
  2. generate_candidate_plans  → hasta max_plans permutaciones del mismo query
  3. extract_plan_features      → vector tabular por plan
  4. build_feature_matrix       → matriz X (n_plans × n_features)
  5. predict_plan_cost          → scoring compatible con SHAP
  6. compute_shap_explanation   → SHAP sobre proxy Ridge (sklearn) o heurística
  7. compute_fallback_explanation → delta vs. promedio si SHAP no disponible
  8. build_natural_language_summary → resumen legible
"""

from __future__ import annotations

import copy
import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sql.optimizer import BayesOptimizer, QueryOptimizer, SelingerOptimizer
from sql.parser import (
    BinaryOp,
    ColumnRef,
    FunctionCall,
    JoinClause,
    SelectStatement,
    UnaryOp,
)
from xai.explanation import Explanation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Plan generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidate_plans(
    stmt: SelectStatement,
    max_plans: int = 50,
    fix_base_table: bool = False,
) -> List[SelectStatement]:
    """
    Enumera hasta max_plans permutaciones del orden de JOIN.

    El plan original (orden tal como llega en stmt) siempre se incluye como
    primera entrada. Las demás se generan iterando sobre itertools.permutations
    hasta alcanzar max_plans.

    fix_base_table=True (usar con BayesOptimizer): mantiene stmt.from_table
    como primera tabla y solo permuta el orden de los JOINs. Esto refleja el
    espacio de búsqueda real de BayesCard, cuyo DP siempre ancla la tabla FROM.
    fix_base_table=False (Selinger, BasicOptimizer): permuta todas las tablas.

    Nota: para consultas con N tablas hay N! permutaciones. Con max_plans=50
    y N≥5 (120 perms) se corta el muestreo. Si N≤4 se cubren todas.
    """
    if not stmt.join_clauses:
        return [copy.deepcopy(stmt)]

    # primero obtenemos todas las tablas de la consulta
    tables = [stmt.from_table] + [jc.table for jc in stmt.join_clauses]
    # condition_map: tabla → condición de join original
    condition_map: Dict[str, object] = {
        jc.table: jc.condition for jc in stmt.join_clauses
    }

    plans: List[SelectStatement] = []
    seen: set = set()

    # Siempre incluir el plan original primero
    _add_permuted_plan(plans, seen, stmt, tables, condition_map)

    if fix_base_table:
        # Solo permuta los JOINs, la tabla FROM queda fija (igual que BayesCard)
        join_tables = [jc.table for jc in stmt.join_clauses]
        for perm in itertools.permutations(join_tables):
            if len(plans) >= max_plans:
                break
            _add_permuted_plan(
                plans, seen, stmt,
                [stmt.from_table] + list(perm),
                condition_map,
            )
    else:
        for perm in itertools.permutations(tables):
            if len(plans) >= max_plans:
                break
            _add_permuted_plan(plans, seen, stmt, list(perm), condition_map)

    return plans


def _add_permuted_plan(
    plans: List[SelectStatement],
    seen: set,
    stmt: SelectStatement,
    table_order: List[str],
    condition_map: Dict[str, object],
) -> None:
    """Agrega un plan (order de tablas) a la lista si no fue visto antes."""
    key = tuple(table_order)
    if key in seen:
        return
    seen.add(key)

    new_stmt = copy.deepcopy(stmt)
    # la tabla principal va a ser la primera del orden dado
    new_stmt.from_table = table_order[0]
    # Asignar condición original de cada tabla; si la tabla era from_table
    # (sin condición propia), condition_map.get devuelve None → JOIN sin ON.
    new_stmt.join_clauses = [
        JoinClause(t, condition_map.get(t)) for t in table_order[1:]
    ]
    plans.append(new_stmt)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

#: Orden canónico de features (IMPORTANTE: estimated_cost al final para facilitar
#: el slicing al construir X_structural en compute_shap_explanation).
FEATURE_NAMES: List[str] = [
    "table_count",
    "join_count",
    "filter_count",
    "index_usage_count",
    "full_scan_count",
    "ast_node_count",
    "ast_depth",
    "projected_column_count",
    "predicate_count",
    "sort_count",           # siempre 0: el parser actual no soporta ORDER BY
    "aggregation_count",
    "first_table_rows",     # varía con el orden de JOIN → útil para SHAP
    "max_table_rows",
    "estimated_cost",       # target del proxy model; excluido de X_structural
]

#: Etiquetas legibles en español para cada feature.
FEATURE_LABELS: Dict[str, str] = {
    "table_count":            "Cantidad de tablas en el plan",
    "join_count":             "Cantidad de joins",
    "filter_count":           "Condiciones de filtro (WHERE)",
    "index_usage_count":      "Joins que aprovechan índices",
    "full_scan_count":        "Accesos sin índice (full scan)",
    "ast_node_count":         "Nodos totales del AST",
    "ast_depth":              "Profundidad estructural del AST",
    "projected_column_count": "Columnas proyectadas",
    "predicate_count":        "Condiciones totales (filtros + joins)",
    "sort_count":             "Operaciones de ordenamiento",
    "aggregation_count":      "Funciones de agregación",
    "first_table_rows":       "Filas estimadas de la primera tabla",
    "max_table_rows":         "Filas de la tabla más grande",
    "estimated_cost":         "Costo estimado del plan",
}

#: Interpretación semántica: dirección "buena" y razón por feature.
#: low_is_good=True  → un valor bajo es bueno para el plan
#: high_is_good=True → un valor alto es bueno para el plan
#: neutral=True      → no hay una dirección preferida clara
FEATURE_INTERPRETATION: Dict[str, Dict] = {
    "first_table_rows":       {"low_is_good": True,  "reason": "Empezar con una tabla más chica reduce el tamaño del resultado intermedio"},
    "max_table_rows":         {"low_is_good": True,  "reason": "Tablas más grandes elevan el costo de acceso"},
    "index_usage_count":      {"high_is_good": True, "reason": "Más uso de índices reduce el costo de acceso a datos"},
    "full_scan_count":        {"low_is_good": True,  "reason": "Cada full scan implica recorrer toda la tabla sin índice"},
    "estimated_cost":         {"low_is_good": True,  "reason": "Un costo estimado menor indica un plan más eficiente"},
    "predicate_count":        {"neutral": True,      "reason": "Más condiciones pueden filtrar más filas, pero también añaden complejidad"},
    "filter_count":           {"neutral": True,      "reason": "Más filtros pueden reducir el resultado, pero añaden evaluación"},
    "ast_depth":              {"low_is_good": True,  "reason": "Mayor profundidad estructural puede indicar mayor complejidad de ejecución"},
    "ast_node_count":         {"low_is_good": True,  "reason": "Más nodos en el AST implican mayor complejidad de procesamiento"},
    "join_count":             {"neutral": True,      "reason": "La cantidad de joins depende de la consulta, no del plan de ejecución"},
    "table_count":            {"neutral": True,      "reason": "La cantidad de tablas depende de la consulta, no del plan"},
    "aggregation_count":      {"neutral": True,      "reason": "Las agregaciones dependen de la consulta, no del orden de joins"},
    "projected_column_count": {"neutral": True,      "reason": "Las columnas proyectadas dependen de la consulta, no del plan"},
    "sort_count":             {"neutral": True,      "reason": "Las ordenaciones dependen de la consulta, no del plan"},
}

#: Agrupación semántica de features por categoría funcional.
FEATURE_CATEGORIES: Dict[str, List[str]] = {
    "Tamaño de datos":    ["first_table_rows", "max_table_rows", "table_count"],
    "Acceso a datos":     ["index_usage_count", "full_scan_count"],
    "Selectividad":       ["filter_count", "predicate_count"],
    "Complejidad lógica": ["ast_depth", "ast_node_count", "join_count"],
    "Proyección":         ["projected_column_count", "aggregation_count", "sort_count"],
    "Costo":              ["estimated_cost"],
}


def _label_for_feature(feature: str) -> str:
    """Etiqueta legible, incluyendo features dinamicas por paso de JOIN."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    aggregate_labels = {
        "base_filtered_rows": "Filas iniciales filtradas",
        "sum_intermediate_rows": "Suma de filas intermedias",
        "max_intermediate_rows": "Mayor resultado intermedio",
        "avg_intermediate_rows": "Promedio de filas intermedias",
        "most_expensive_join_cost": "Costo del JOIN mas caro",
        "most_expensive_join_position": "Posicion del JOIN mas caro",
    }
    if feature in aggregate_labels:
        return aggregate_labels[feature]
    if feature.startswith("join_"):
        parts = feature.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            step = parts[1]
            metric = "_".join(parts[2:])
            metric_labels = {
                "input_rows": "Filas de entrada antes del JOIN",
                "table_rows": "Filas filtradas de la tabla agregada",
                "selectivity": "Selectividad del JOIN",
                "output_rows": "Filas intermedias luego del JOIN",
                "cost": "Costo del paso de JOIN",
            }
            if metric in metric_labels:
                return f"{metric_labels[metric]} {step}"
    return feature


def _interpretation_for_feature(feature: str) -> Dict:
    """Interpretacion semantica, incluyendo features dinamicas por JOIN."""
    if feature in FEATURE_INTERPRETATION:
        return FEATURE_INTERPRETATION[feature]
    if feature.startswith("join_"):
        if feature.endswith("_selectivity"):
            return {
                "low_is_good": True,
                "reason": "Una selectividad menor suele producir menos filas intermedias",
            }
        if feature.endswith(("_input_rows", "_table_rows", "_output_rows", "_cost")):
            return {
                "low_is_good": True,
                "reason": "Menos filas o menor costo en este paso reducen el trabajo acumulado",
            }
    if feature in {
        "base_filtered_rows",
        "sum_intermediate_rows",
        "max_intermediate_rows",
        "avg_intermediate_rows",
        "most_expensive_join_cost",
        "most_expensive_join_position",
    }:
        return {
            "low_is_good": True,
            "reason": "Valores menores indican menos trabajo intermedio acumulado",
        }
    return {}


def extract_plan_features(
    plan: SelectStatement,
    stmt: SelectStatement,
    optimizer: QueryOptimizer,
) -> Dict[str, float]:
    """
    Convierte un AST/plan en un vector de features tabulares.

    Fuentes de datos:
    - estimated_cost:
        REAL para SelingerOptimizer (usa _get_table_cost + _estimate_selectivity).
        REAL para BayesOptimizer    (usa cost_model.table_cardinality + join_selectivity).
        HEURÍSTICO para BasicOptimizer y otros (producto de row counts del catálogo).
    - index_usage_count:
        REAL: cuenta cuántos join-clauses tienen al menos una columna indexada en catalog.
    - first_table_rows / max_table_rows:
        REAL si catalog disponible; HEURÍSTICO (1000.0) si no.
    - sort_count: siempre 0 (TODO: extender cuando se soporte ORDER BY).
    """
    tables = [plan.from_table] + (
        [jc.table for jc in plan.join_clauses] if plan.join_clauses else []
    )
    join_clauses = plan.join_clauses or []
    catalog = optimizer.catalog

    # ── structural ──────────────────────────────────────────────────────────
    table_count = float(len(tables))
    join_count = float(len(join_clauses))
    filter_count = float(_count_predicates(stmt.where_clause))
    predicate_count = filter_count + float(
        sum(_count_predicates(jc.condition) for jc in join_clauses)
    )
    projected_column_count = float(len(stmt.columns))
    sort_count = 0.0  # TODO: extender con ORDER BY
    aggregation_count = float(
        sum(1 for col in stmt.columns if isinstance(col, FunctionCall))
    )

    # ── índices ─────────────────────────────────────────────────────────────
    index_usage_count = 0.0
    if catalog:
        for jc in join_clauses:
            refs = _extract_column_refs(jc.condition)
            for ref in refs:
                tname = ref.table or jc.table
                if _has_index(catalog, tname, ref.column):
                    index_usage_count += 1.0
                    break  # un índice por join clause es suficiente
    full_scan_count = max(table_count - index_usage_count, 0.0)

    # ── AST metrics ──────────────────────────────────────────────────────────
    ast_node_count = float(_count_ast_nodes(plan))
    ast_depth = float(_ast_depth(plan))

    # ── row counts ───────────────────────────────────────────────────────────
    first_table_rows = _get_table_rows(tables[0], optimizer)
    max_table_rows = max(_get_table_rows(t, optimizer) for t in tables)

    # ── costo estimado y pasos left-deep ─────────────────────────────────────
    plan_trace = _estimate_plan_trace(plan, stmt, optimizer)
    estimated_cost = plan_trace["total_cost"]
    step_features = _join_step_features(plan_trace)

    features = {
        "table_count": table_count,
        "join_count": join_count,
        "filter_count": filter_count,
        "index_usage_count": index_usage_count,
        "full_scan_count": full_scan_count,
        "ast_node_count": ast_node_count,
        "ast_depth": ast_depth,
        "projected_column_count": projected_column_count,
        "predicate_count": predicate_count,
        "sort_count": sort_count,
        "aggregation_count": aggregation_count,
        "first_table_rows": first_table_rows,
        "max_table_rows": max_table_rows,
    }
    features.update(step_features)
    features["estimated_cost"] = estimated_cost
    return features


def _get_table_rows(table_name: str, optimizer: QueryOptimizer) -> float:
    """Devuelve el row count de una tabla. REAL cuando el catálogo o cost model
    lo tienen; HEURÍSTICO (1000.0) como fallback."""
    # REAL: BayesOptimizer expone su cost model con cardinality real
    if isinstance(optimizer, BayesOptimizer):
        try:
            return float(optimizer._delegate.cost_model.table_cardinality(table_name))
        except Exception:
            pass
    # REAL: SelingerOptimizer usa statistics.csv vía _get_table_cost
    if isinstance(optimizer, SelingerOptimizer):
        try:
            return float(optimizer._get_table_cost(table_name))
        except Exception:
            pass
    # REAL: catálogo directo
    if optimizer.catalog:
        try:
            return float(optimizer.catalog.get_table_row_count(table_name))
        except Exception:
            pass
    return 1000.0  # HEURÍSTICO: fallback por defecto


def _estimate_plan_trace(
    plan: SelectStatement,
    stmt: SelectStatement,
    optimizer: QueryOptimizer,
) -> Dict[str, Any]:
    """
    Calcula el costo left-deep y expone cada paso del plan.

    Cada paso contiene:
      input_rows  : filas intermedias antes del JOIN
      table_rows  : filas de la tabla que se agrega, luego de filtros locales
      selectivity : selectividad estimada del predicado de JOIN
      output_rows : filas intermedias luego del JOIN
      cost        : costo del paso (igual al output intermedio acumulado)
    """
    tables = [plan.from_table] + (
        [jc.table for jc in plan.join_clauses] if plan.join_clauses else []
    )
    join_clauses = plan.join_clauses or []
    steps: List[Dict[str, float]] = []

    if isinstance(optimizer, SelingerOptimizer):
        intermediate = float(optimizer._get_table_cost(tables[0]))
        total = intermediate
        for step_no, jc in enumerate(join_clauses, start=1):
            input_rows = intermediate
            table_rows = float(optimizer._get_table_cost(jc.table))
            selectivity = float(optimizer._estimate_selectivity(jc.condition))
            intermediate = input_rows * table_rows * selectivity
            total += intermediate
            steps.append({
                "step": float(step_no),
                "input_rows": input_rows,
                "table_rows": table_rows,
                "selectivity": selectivity,
                "output_rows": intermediate,
                "cost": intermediate,
            })
        return {"base_rows": float(optimizer._get_table_cost(tables[0])), "total_cost": float(total), "steps": steps}

    if isinstance(optimizer, BayesOptimizer):
        cm = optimizer._delegate.cost_model
        local_predicates = _extract_local_predicates(stmt.where_clause, tables)
        base_rows = float(
            cm.table_cardinality(tables[0])
            * cm.table_filter_selectivity(tables[0], local_predicates.get(tables[0]))
        )
        intermediate = base_rows
        total = intermediate
        for step_no, jc in enumerate(join_clauses, start=1):
            input_rows = intermediate
            table_rows = float(
                cm.table_cardinality(jc.table)
                * cm.table_filter_selectivity(jc.table, local_predicates.get(jc.table))
            )
            selectivity = float(cm.join_selectivity(jc.condition))
            intermediate = input_rows * table_rows * selectivity
            total += intermediate
            steps.append({
                "step": float(step_no),
                "input_rows": input_rows,
                "table_rows": table_rows,
                "selectivity": selectivity,
                "output_rows": intermediate,
                "cost": intermediate,
            })
        return {"base_rows": base_rows, "total_cost": float(total), "steps": steps}

    base_rows = float(_get_table_rows(tables[0], optimizer))
    intermediate = base_rows
    total = intermediate
    for step_no, table_name in enumerate(tables[1:], start=1):
        input_rows = intermediate
        table_rows = float(_get_table_rows(table_name, optimizer))
        selectivity = 1.0
        intermediate = input_rows * table_rows * selectivity
        total += intermediate
        steps.append({
            "step": float(step_no),
            "input_rows": input_rows,
            "table_rows": table_rows,
            "selectivity": selectivity,
            "output_rows": intermediate,
            "cost": intermediate,
        })
    return {"base_rows": base_rows, "total_cost": float(total), "steps": steps}


def _join_step_features(plan_trace: Dict[str, Any]) -> Dict[str, float]:
    """Convierte el trace del plan en features dinamicas para SHAP."""
    steps = plan_trace.get("steps", [])
    output_rows = [float(s["output_rows"]) for s in steps]
    costs = [float(s["cost"]) for s in steps]

    features: Dict[str, float] = {
        "base_filtered_rows": float(plan_trace.get("base_rows", 0.0)),
        "sum_intermediate_rows": float(sum(output_rows)),
        "max_intermediate_rows": float(max(output_rows) if output_rows else 0.0),
        "avg_intermediate_rows": float(sum(output_rows) / len(output_rows) if output_rows else 0.0),
        "most_expensive_join_cost": float(max(costs) if costs else 0.0),
        "most_expensive_join_position": float(costs.index(max(costs)) + 1 if costs else 0.0),
    }

    for i, step in enumerate(steps, start=1):
        prefix = f"join_{i}"
        features[f"{prefix}_input_rows"] = float(step["input_rows"])
        features[f"{prefix}_table_rows"] = float(step["table_rows"])
        features[f"{prefix}_selectivity"] = float(step["selectivity"])
        features[f"{prefix}_output_rows"] = float(step["output_rows"])
        features[f"{prefix}_cost"] = float(step["cost"])

    return features


def _estimate_plan_cost(
    plan: SelectStatement,
    stmt: SelectStatement,
    optimizer: QueryOptimizer,
) -> float:
    """
    Estima el costo acumulado de un plan left-deep.

    El modelo acumulativo (left-deep) depende del orden:
      intermediate_0 = rows(tabla_0)
      intermediate_i = intermediate_{i-1} * rows(tabla_i) * sel_i

    Esto hace que el costo varíe con el orden de JOIN: empezar por una tabla
    chica produce intermedios menores en los pasos siguientes.

    REAL (SelingerOptimizer): usa _get_table_cost + _estimate_selectivity.
    REAL (BayesOptimizer): usa cost_model.table_cardinality + table_filter_selectivity + join_selectivity.
        La tabla FROM queda fija (igual que el DP real de BayesCard).
    HEURÍSTICO (otros): producto acumulativo de row counts sin selectividades.
    """
    return float(_estimate_plan_trace(plan, stmt, optimizer)["total_cost"])


def _extract_local_predicates(where_clause, tables: List[str]) -> Dict[str, object]:
    """Replica la extracción de predicados locales por tabla usada por BayesCard."""
    if where_clause is None:
        return {}

    predicates: Dict[str, object] = {}
    for table_name in tables:
        table_predicate = _predicate_for_table(where_clause, table_name)
        if table_predicate is not None:
            predicates[table_name] = table_predicate
    return predicates


def _predicate_for_table(expression, table_name: str):
    if expression is None:
        return None

    if isinstance(expression, UnaryOp):
        operand = _predicate_for_table(expression.operand, table_name)
        if operand is None:
            return None
        return UnaryOp(expression.operator, operand)

    if not isinstance(expression, BinaryOp):
        return None

    if expression.operator.upper() in {"AND", "OR"}:
        left = _predicate_for_table(expression.left, table_name)
        right = _predicate_for_table(expression.right, table_name)
        if left is None and right is None:
            return None
        if left is None:
            return right
        if right is None:
            return left
        return BinaryOp(left, expression.operator, right)

    tables_in_expression = _tables_in_expression(expression)
    if tables_in_expression and tables_in_expression.issubset({table_name}):
        return expression

    if not tables_in_expression:
        return expression

    return None


def _tables_in_expression(expression) -> set:
    if expression is None:
        return set()

    if hasattr(expression, "table") and getattr(expression, "table", None):
        return {expression.table}

    if isinstance(expression, UnaryOp):
        return _tables_in_expression(expression.operand)

    if isinstance(expression, BinaryOp):
        return _tables_in_expression(expression.left) | _tables_in_expression(expression.right)

    return set()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    plans: List[SelectStatement],
    stmt: SelectStatement,
    optimizer: QueryOptimizer,
) -> Tuple[np.ndarray, List[str], List[Dict[str, float]]]:
    """
    Construye la matriz de features para todos los planes candidatos.

    Returns:
        X              : ndarray (n_plans × n_features)
        feature_names  : lista de nombres en el mismo orden que las columnas de X
        feature_dicts  : lista de dicts originales (uno por plan)
    """
    feature_dicts = [extract_plan_features(p, stmt, optimizer) for p in plans]
    if not feature_dicts:
        return np.empty((0, 0)), [], []

    feature_names = list(feature_dicts[0].keys())
    X = np.array(
        [[fd[fn] for fn in feature_names] for fd in feature_dicts],
        dtype=float,
    )
    return X, feature_names, feature_dicts


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cost prediction (SHAP-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def predict_plan_cost(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """
    Función de scoring compatible con SHAP.
    Toma una matriz de features y devuelve un vector de costos (menor = mejor).

    REAL: si 'estimated_cost' está en feature_names, lo usa directamente.
    HEURÍSTICO: si no, usa combinación ponderada de features estructurales.
        Los pesos reflejan el impacto relativo esperado de cada feature:
        más full_scans → mayor costo; más índices → menor costo; etc.
    """
    idx = {name: i for i, name in enumerate(feature_names)}

    if "estimated_cost" in idx:
        return X[:, idx["estimated_cost"]].copy()

    # HEURÍSTICO: pesos elegidos para aproximar el comportamiento del optimizador
    weights = {
        "table_count":        100.0,
        "join_count":         500.0,
        "full_scan_count":    800.0,
        "index_usage_count": -300.0,
        "filter_count":        50.0,
        "predicate_count":     30.0,
        "ast_depth":           20.0,
        "first_table_rows":     1.0,
    }
    cost = np.zeros(len(X))
    for fname, weight in weights.items():
        if fname in idx:
            cost += X[:, idx[fname]] * weight
    return np.maximum(cost, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP explanation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_explanation(
    X: np.ndarray,
    feature_names: List[str],
    chosen_idx: int,
    top_k: int,
) -> Tuple[Optional[Dict[str, float]], List[str], bool, List[str]]:
    """
    Aplica SHAP sobre un modelo proxy (Ridge si sklearn disponible) para
    explicar el plan elegido en términos de sus features estructurales.

    Diseño:
    - X_structural: todas las features EXCEPTO estimated_cost (evita circularidad).
    - y_cost      : vector de estimated_cost por plan (target del proxy).
    - proxy       : Ridge regression X_structural → y_cost.
    - SHAP explica el proxy, respondiendo qué features estructurales
      explican las variaciones de costo entre planes.

    Si sklearn no está disponible, el proxy es predict_plan_cost (heurística).
    Si SHAP no está instalado, retorna (None, [], False, warnings) para que
    la capa superior use compute_fallback_explanation.

    Returns:
        shap_values_dict : {feature: shap_value} para el plan elegido, o None
        top_features     : top_k features por |shap_value|
        shap_used        : True si SHAP se ejecutó correctamente
        warnings         : advertencias acumuladas
    """
    warnings_out: List[str] = []

    # ── necesitamos al menos 2 planes para que el proxy tenga sentido ──────
    if len(X) < 2:
        warnings_out.append(
            "Solo hay un plan candidato; se necesitan ≥2 para usar SHAP."
        )
        return None, [], False, warnings_out

    # ── separar features estructurales del target (estimated_cost) ─────────
    structural_idx = [i for i, n in enumerate(feature_names) if n != "estimated_cost"]
    structural_names = [feature_names[i] for i in structural_idx]
    X_structural = X[:, structural_idx]

    if "estimated_cost" in feature_names:
        y_cost = X[:, feature_names.index("estimated_cost")]
    else:
        y_cost = predict_plan_cost(X, feature_names)

    # ── intentar sklearn ────────────────────────────────────────────────────
    try:
        from sklearn.linear_model import Ridge
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        warnings_out.append(
            "sklearn no disponible; SHAP usará función heurística directamente."
        )

    # ── intentar SHAP ───────────────────────────────────────────────────────
    try:
        import shap as shap_lib
    except ImportError:
        warnings_out.append(
            "SHAP no está instalado (pip install shap). Usando explicación fallback."
        )
        return None, [], False, warnings_out

    try:
        if has_sklearn:
            # REAL: entrenar proxy Ridge sobre (X_structural → estimated_cost)
            proxy = Ridge(alpha=1.0)
            proxy.fit(X_structural, y_cost)

            # LinearExplainer es exacto y muy rápido para modelos lineales
            try:
                explainer = shap_lib.LinearExplainer(proxy, X_structural)
                shap_values = explainer.shap_values(
                    X_structural[chosen_idx : chosen_idx + 1]
                )
            except Exception:
                # Fallback a KernelExplainer si LinearExplainer falla
                background = X_structural.mean(axis=0, keepdims=True)
                explainer = shap_lib.KernelExplainer(proxy.predict, background)
                shap_values = explainer.shap_values(
                    X_structural[chosen_idx : chosen_idx + 1], nsamples=100
                )
        else:
            # HEURÍSTICO: usar predict_plan_cost directamente como "modelo"
            _names = structural_names  # closure para la lambda

            def _predict_heuristic(Xp: np.ndarray) -> np.ndarray:
                return predict_plan_cost(Xp, _names)

            background = X_structural.mean(axis=0, keepdims=True)
            explainer = shap_lib.KernelExplainer(_predict_heuristic, background)
            shap_values = explainer.shap_values(
                X_structural[chosen_idx : chosen_idx + 1], nsamples=100
            )
            warnings_out.append(
                "SHAP explica una función heurística (no el costo real del optimizador) "
                "porque sklearn no está instalado."
            )

        sv = np.array(shap_values).flatten()
        shap_dict = {name: float(sv[i]) for i, name in enumerate(structural_names)}

        top_features = _select_diverse_lime_features(
            shap_dict,
            X_structural,
            structural_names,
            top_k,
        )

        return shap_dict, top_features, True, warnings_out

    except Exception as exc:
        warnings_out.append(
            f"SHAP falló con error: {exc}. Usando explicación fallback."
        )
        return None, [], False, warnings_out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fallback explanation
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_linear_fit(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    ridge_alpha: float = 1e-6,
) -> Tuple[float, np.ndarray]:
    """Ajuste lineal ponderado simple con una regularizacion ridge pequena."""
    X_design = np.column_stack([np.ones(len(X)), X])
    if sample_weight is not None:
        w = np.sqrt(np.maximum(sample_weight, 1e-12))
        X_design = X_design * w[:, None]
        y = y * w

    reg = np.eye(X_design.shape[1]) * ridge_alpha
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(X_design.T @ X_design + reg) @ X_design.T @ y
    return float(beta[0]), beta[1:].astype(float)


def _select_diverse_lime_features(
    lime_values: Dict[str, float],
    X_structural: np.ndarray,
    structural_names: List[str],
    top_k: int,
) -> List[str]:
    """
    Elige features LIME evitando columnas constantes o duplicadas entre planes.

    Con pocos candidatos es comun que varias features sean la misma senal
    (por ejemplo join_1_output_rows == join_1_cost). Mostrar todas da la falsa
    impresion de que LIME encontro muchas causas independientes.
    """
    name_to_idx = {name: i for i, name in enumerate(structural_names)}
    semantic_priority = {
        "sum_intermediate_rows": 0,
        "max_intermediate_rows": 1,
        "avg_intermediate_rows": 2,
        "most_expensive_join_cost": 3,
        "join_1_output_rows": 10,
        "join_1_cost": 11,
        "join_2_output_rows": 12,
        "join_2_cost": 13,
        "join_1_table_rows": 20,
        "join_2_table_rows": 21,
    }
    ranked = sorted(
        lime_values.items(),
        key=lambda x: (-round(abs(x[1]), 1), semantic_priority.get(x[0], 1000), x[0]),
    )
    selected: List[str] = []
    selected_columns: List[np.ndarray] = []

    for name, weight in ranked:
        if abs(weight) <= 1e-12:
            continue
        idx = name_to_idx[name]
        column = X_structural[:, idx]
        if np.nanmax(column) - np.nanmin(column) <= 1e-9:
            continue

        duplicate = False
        for previous in selected_columns:
            if np.allclose(column, previous, rtol=1e-9, atol=1e-9):
                duplicate = True
                break
            corr = np.corrcoef(column, previous)[0, 1]
            if np.isfinite(corr) and abs(corr) >= 0.999:
                duplicate = True
                break
        if duplicate:
            continue

        selected.append(name)
        selected_columns.append(column)
        if len(selected) >= top_k:
            break

    if selected:
        return selected

    return [name for name, _ in ranked[:top_k]]


def compute_lime_explanation(
    X: np.ndarray,
    feature_names: List[str],
    chosen_idx: int,
    top_k: int,
    random_state: Optional[int] = 42,
    n_samples: int = 500,
) -> Tuple[Dict[str, float], List[str], bool, List[str]]:
    """
    Explicacion local tipo LIME sobre el modelo de costo estimado.

    Las perturbaciones son artificiales y no necesariamente representan planes
    SQL validos. Se interpreta como sensibilidad local del modelo de costo,
    no como validacion contra runtime.
    """
    warnings_out: List[str] = [
        "LIME perturba features artificialmente; no todas las muestras son planes SQL validos."
    ]

    if len(X) < 2:
        warnings_out.append("Solo hay un plan candidato; LIME necesita al menos 2 planes.")
        return {}, [], False, warnings_out

    structural_idx = [i for i, n in enumerate(feature_names) if n != "estimated_cost"]
    structural_names = [feature_names[i] for i in structural_idx]
    if not structural_idx:
        warnings_out.append("No hay features estructurales para explicar con LIME.")
        return {}, [], False, warnings_out

    X_structural = X[:, structural_idx].astype(float)
    if "estimated_cost" in feature_names:
        y_cost = X[:, feature_names.index("estimated_cost")].astype(float)
    else:
        y_cost = predict_plan_cost(X, feature_names).astype(float)

    x0 = X_structural[chosen_idx]
    rng = np.random.default_rng(random_state)

    feature_scale = X_structural.std(axis=0)
    fallback_scale = np.maximum(np.abs(x0) * 0.10, 1.0)
    feature_scale = np.where(feature_scale > 1e-9, feature_scale, fallback_scale)

    try:
        from sklearn.linear_model import Ridge

        proxy = Ridge(alpha=1.0)
        proxy.fit(X_structural, y_cost)

        def predict_proxy(values: np.ndarray) -> np.ndarray:
            return proxy.predict(values)

    except Exception:
        intercept, coef = _weighted_linear_fit(X_structural, y_cost, ridge_alpha=1.0)

        def predict_proxy(values: np.ndarray) -> np.ndarray:
            return intercept + values @ coef

        warnings_out.append("sklearn no disponible; LIME usa proxy lineal interno.")

    perturbations = rng.normal(loc=x0, scale=feature_scale, size=(n_samples, len(x0)))
    perturbations = np.maximum(perturbations, 0.0)
    perturbations[0] = x0

    predictions = predict_proxy(perturbations)
    normalized_delta = (perturbations - x0) / feature_scale
    distances = np.sqrt(np.sum(normalized_delta ** 2, axis=1))
    kernel_width = max(np.sqrt(len(x0)) * 0.75, 1.0)
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    _, local_coef = _weighted_linear_fit(
        normalized_delta,
        predictions,
        sample_weight=weights,
        ridge_alpha=1e-3,
    )

    lime_dict = {
        name: float(local_coef[i])
        for i, name in enumerate(structural_names)
    }
    top_features = _select_diverse_lime_features(
        lime_dict,
        X_structural,
        structural_names,
        top_k,
    )
    return lime_dict, top_features, True, warnings_out


def compute_fallback_explanation(
    X: np.ndarray,
    feature_names: List[str],
    chosen_idx: int,
    top_k: int,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Explicación heurística cuando SHAP no está disponible.

    Compara las features del plan elegido contra el promedio de los demás
    planes candidatos. La diferencia relativa indica qué tan distinto es
    el plan elegido en cada feature.

    Returns:
        feature_diffs : {feature: relative_diff} — positivo = elegido > promedio
        top_features  : top_k features por |relative_diff|
    """
    chosen = X[chosen_idx]
    others_mask = np.ones(len(X), dtype=bool)
    others_mask[chosen_idx] = False

    if others_mask.sum() == 0:
        # Solo hay un plan: sin comparación posible
        return {n: 0.0 for n in feature_names}, feature_names[:top_k]

    others_mean = X[others_mask].mean(axis=0)
    eps = 1e-9
    relative_diffs = {
        feature_names[i]: float(
            (chosen[i] - others_mean[i]) / (abs(others_mean[i]) + eps)
        )
        for i in range(len(feature_names))
    }

    favorable_feats: List[Tuple[str, float]] = []
    for feat, diff in relative_diffs.items():
        if feat == "estimated_cost":
            continue
        interp = _interpretation_for_feature(feat)
        if interp.get("low_is_good"):
            score = -diff
        elif interp.get("high_is_good"):
            score = diff
        else:
            score = abs(diff)
        if score > 0:
            favorable_feats.append((feat, score))

    if favorable_feats:
        sorted_feats = sorted(favorable_feats, key=lambda x: x[1], reverse=True)
    else:
        sorted_feats = sorted(
            ((f, v) for f, v in relative_diffs.items() if f != "estimated_cost"),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    top_features = [f for f, _ in sorted_feats[:top_k]]
    return relative_diffs, top_features


# ─────────────────────────────────────────────────────────────────────────────
# 7. Natural language summary
# ─────────────────────────────────────────────────────────────────────────────

def build_natural_language_summary(
    chosen_plan: List[str],
    chosen_score: float,
    compared_plans_count: int,
    top_features: List[str],
    shap_values: Optional[Dict[str, float]],
    feature_values: Dict[str, float],
    shap_used: bool,
    warnings: List[str],
) -> str:
    """Genera un resumen en lenguaje natural de la explicación SHAP."""
    join_order = " -> ".join(chosen_plan) if chosen_plan else "(sin tablas)"
    lines = [
        f"El optimizador eligió el orden de JOIN: [{join_order}]",
        f"Costo estimado del plan elegido: {chosen_score:.4f}",
        f"Planes candidatos comparados: {compared_plans_count}",
        "",
    ]

    method = "SHAP (modelo proxy Ridge)" if shap_used else "heurística fallback (delta vs. promedio)"
    lines.append(f"Método de explicación: {method}")

    if top_features:
        lines.append(f"Top {len(top_features)} features más influyentes:")
        for feat in top_features:
            val = feature_values.get(feat, 0.0)
            if shap_values and feat in shap_values:
                sv = shap_values[feat]
                direction = "reduce" if sv < 0 else "aumenta"
                lines.append(
                    f"  - {feat}: {val:.2f}  (SHAP={sv:+.4f}, {direction} el costo)"
                )
            else:
                lines.append(f"  - {feat}: {val:.2f}")

    if warnings:
        lines.append("")
        lines.append("Advertencias:")
        for w in warnings:
            lines.append(f"  ! {w}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 8. AST helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_column_refs(expr) -> List[ColumnRef]:
    """Extrae todas las ColumnRef de una expresión AST recursivamente."""
    if expr is None:
        return []
    if isinstance(expr, ColumnRef):
        return [expr]
    if isinstance(expr, BinaryOp):
        return _extract_column_refs(expr.left) + _extract_column_refs(expr.right)
    if isinstance(expr, UnaryOp):
        return _extract_column_refs(expr.operand)
    if isinstance(expr, FunctionCall):
        refs: List[ColumnRef] = []
        for arg in expr.args:
            refs.extend(_extract_column_refs(arg))
        return refs
    return []


def _count_predicates(expr) -> int:
    """Cuenta predicados atómicos en una expresión (WHERE / JOIN condition)."""
    if expr is None:
        return 0
    if isinstance(expr, BinaryOp):
        op = expr.operator.upper() if isinstance(expr.operator, str) else ""
        if op in {"AND", "OR"}:
            return _count_predicates(expr.left) + _count_predicates(expr.right)
        return 1
    if isinstance(expr, UnaryOp):
        return _count_predicates(expr.operand)
    return 0


def _count_ast_nodes(node) -> int:
    """Cuenta todos los nodos en el AST del plan."""
    if node is None:
        return 0
    count = 1
    if isinstance(node, SelectStatement):
        count += sum(_count_ast_nodes(c) for c in (node.columns or []))
        count += _count_ast_nodes(node.where_clause)
        for jc in node.join_clauses or []:
            count += _count_ast_nodes(jc)
    elif isinstance(node, JoinClause):
        count += _count_ast_nodes(node.condition)
    elif isinstance(node, BinaryOp):
        count += _count_ast_nodes(node.left) + _count_ast_nodes(node.right)
    elif isinstance(node, UnaryOp):
        count += _count_ast_nodes(node.operand)
    elif isinstance(node, FunctionCall):
        count += sum(_count_ast_nodes(a) for a in node.args)
    return count


def _ast_depth(node, depth: int = 0) -> int:
    """Calcula la profundidad máxima del AST."""
    if node is None:
        return depth
    if isinstance(node, SelectStatement):
        sub = (
            [_ast_depth(c, depth + 1) for c in (node.columns or [])]
            + [_ast_depth(node.where_clause, depth + 1)]
            + [_ast_depth(jc, depth + 1) for jc in (node.join_clauses or [])]
        )
        return max(sub, default=depth + 1)
    if isinstance(node, JoinClause):
        return _ast_depth(node.condition, depth + 1)
    if isinstance(node, BinaryOp):
        return max(
            _ast_depth(node.left, depth + 1), _ast_depth(node.right, depth + 1)
        )
    if isinstance(node, UnaryOp):
        return _ast_depth(node.operand, depth + 1)
    if isinstance(node, FunctionCall):
        return max(
            (_ast_depth(a, depth + 1) for a in node.args), default=depth + 1
        )
    return depth + 1


def _has_index(catalog, table: str, column: str) -> bool:
    """True si existe algún índice sobre (tabla, columna) en el catálogo."""
    for idx in catalog.indices.values():
        if idx.table_name == table and idx.column_name == column:
            return True
    return False


def _plan_key(plan: SelectStatement) -> Tuple[str, ...]:
    """Clave que identifica un plan únicamente por su orden de tablas."""
    tables = [plan.from_table] + (
        [jc.table for jc in plan.join_clauses] if plan.join_clauses else []
    )
    return tuple(tables)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Interpretation helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_percentiles(
    X: np.ndarray,
    feature_names: List[str],
    chosen_idx: int,
) -> Dict[str, float]:
    """
    Percentil (0–100) del plan elegido para cada feature respecto a todos los candidatos.
    50 → el plan está en la mediana; 90 → mayor que el 90% de los demás planes.
    """
    chosen = X[chosen_idx]
    result: Dict[str, float] = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        if col.max() == col.min():
            result[name] = 50.0
        else:
            result[name] = float(np.sum(col <= chosen[i]) / len(col) * 100.0)
    return result


def compute_relative_impact(shap_values: Dict[str, float]) -> Dict[str, float]:
    """
    Porcentaje de impacto relativo absoluto de cada feature SHAP sobre el total.
    Permite decir "esta feature explica el 35% del impacto total de la decisión".
    """
    total = sum(abs(v) for v in shap_values.values())
    if total == 0:
        return {k: 0.0 for k in shap_values}
    return {k: abs(v) / total * 100.0 for k, v in shap_values.items()}


def _semantic_impact_label(pct: float) -> str:
    """Etiqueta semántica de impacto: 'alto' (≥30%), 'medio' (10–30%), 'bajo' (<10%)."""
    if pct >= 30:
        return "alto"
    if pct >= 10:
        return "medio"
    return "bajo"


def _feature_direction_for_plan(shap_val: float, feature: str) -> str:
    """
    Devuelve si el SHAP de una feature mejora o perjudica el plan.

    SHAP mide el efecto marginal sobre `estimated_cost` (salida del proxy Ridge).
    La dirección siempre viene del signo del SHAP, independientemente de si la
    feature es low_is_good o high_is_good:
      - SHAP < 0 → reduce el costo estimado → mejora el plan.
      - SHAP > 0 → aumenta el costo estimado → perjudica el plan.
    El campo low_is_good/high_is_good de FEATURE_INTERPRETATION explica *por qué*
    el valor importa, pero no invierte el signo SHAP.
    """
    interp = _interpretation_for_feature(feature)
    if interp.get("neutral"):
        return "neutral"
    if abs(shap_val) < 1e-9:
        return "sin efecto"
    return "mejora el costo" if shap_val < 0 else "perjudica el costo"


def compute_explanation_quality(
    n_plans: int,
    shap_used: bool,
    X: np.ndarray,
    feature_names: List[str],
) -> Tuple[str, List[str]]:
    """
    Calidad de la explicación: 'alta', 'media' o 'baja', con lista de razones.

    Criterios acumulativos:
      - Cantidad de planes candidatos (≥10: buena cobertura).
      - Disponibilidad de SHAP vs. fallback heurístico.
      - Variabilidad de features entre candidatos (discriminación).
    """
    score = 0
    reasons: List[str] = []

    if n_plans >= 10:
        score += 2
        reasons.append(f"{n_plans} planes candidatos comparados (buena cobertura)")
    elif n_plans >= 3:
        score += 1
        reasons.append(f"solo {n_plans} planes candidatos (cobertura limitada)")
    else:
        reasons.append(f"muy pocos candidatos ({n_plans}), explicación débil")

    if shap_used:
        score += 2
        reasons.append("SHAP disponible con proxy lineal (método exacto)")
    else:
        reasons.append("SHAP no disponible — explicación heurística (fallback)")

    if X.shape[0] > 1:
        stds = X.std(axis=0)
        n_stable = int(np.sum(stds < 1e-6))
        pct_stable = n_stable / max(X.shape[1], 1)
        if pct_stable < 0.3:
            score += 1
            reasons.append("features con alta variabilidad entre planes (muy discriminativas)")
        elif pct_stable > 0.7:
            reasons.append("muchas features constantes entre planes (poca discriminación)")

    if score >= 4:
        return "alta", reasons
    if score >= 2:
        return "media", reasons
    return "baja", reasons


def find_runner_up(
    candidates: List[SelectStatement],
    X: np.ndarray,
    feature_names: List[str],
    chosen_idx: int,
) -> Tuple[Optional[List[str]], Optional[float], Optional[int]]:
    """
    Encuentra el segundo plan más barato por costo estimado (runner-up).

    Returns:
        (tabla_order_list, costo, runner_up_idx) o (None, None, None) si no hay alternativas.
    """
    if len(candidates) < 2 or "estimated_cost" not in feature_names:
        return None, None, None

    cost_col = feature_names.index("estimated_cost")
    costs = X[:, cost_col]
    order = np.argsort(costs)
    for idx in order:
        if int(idx) != chosen_idx:
            plan = candidates[int(idx)]
            tables = [plan.from_table] + (
                [jc.table for jc in plan.join_clauses] if plan.join_clauses else []
            )
            return tables, float(costs[int(idx)]), int(idx)
    return None, None, None


def build_structured_explanation(
    chosen_plan: List[str],
    chosen_score: float,
    compared_plans_count: int,
    top_features: List[str],
    shap_values: Optional[Dict[str, float]],
    feature_values: Dict[str, float],
    shap_used: bool,
    warnings: List[str],
    feature_percentiles: Dict[str, float],
    feature_impact_pct: Dict[str, float],
    runner_up_plan: Optional[List[str]],
    runner_up_score: Optional[float],
    explanation_quality: str,
    explanation_quality_reasons: List[str],
) -> Dict[str, object]:
    """
    Construye la explicación estructurada en dos niveles: ejecutivo y técnico.

    Returns un dict con claves:
        executive_summary        : str — 2–3 frases para audiencia no técnica
        factors_positive         : List[str] — factores que favorecieron el plan
        factors_negative         : List[str] — factores que jugaron en contra
        technical_detail         : str — tabla con SHAP, impacto%, percentil, dirección
        natural_language_summary : str — combinación completa de todos los niveles
    """
    join_order = " -> ".join(chosen_plan) if chosen_plan else "(sin tablas)"

    # ── Executive summary ─────────────────────────────────────────────────────
    exec_lines: List[str] = [
        f"El optimizador eligió el orden de JOIN: {join_order}.",
    ]
    if runner_up_plan and runner_up_score is not None and runner_up_score > 0 and chosen_score > 0:
        pct_better = (runner_up_score - chosen_score) / runner_up_score * 100.0
        runner_order = " -> ".join(runner_up_plan)
        if abs(pct_better) < 0.5:
            exec_lines.append(
                f"El plan alternativo más cercano ({runner_order}) tiene el mismo costo estimado "
                f"({runner_up_score:.2f}); el optimizador eligió entre planes equivalentes."
            )
        elif pct_better > 0:
            exec_lines.append(
                f"Este plan es {pct_better:.1f}% más eficiente que la siguiente alternativa "
                f"({runner_order}, costo {runner_up_score:.2f})."
            )
        else:
            # La fórmula XAI asigna más costo al elegido; el optimizador tiene su propio modelo
            pct_worse = abs(pct_better)
            exec_lines.append(
                f"La fórmula XAI estima {pct_worse:.1f}% más costo que el runner-up ({runner_order}, "
                f"costo {runner_up_score:.2f}); el optimizador lo eligió por su modelo interno."
            )
    if top_features:
        top_labels = [_label_for_feature(f) for f in top_features[:3]]
        exec_lines.append(
            f"Los factores principales de la decisión fueron: {', '.join(top_labels)}."
        )
    executive_summary = " ".join(exec_lines)

    # ── Factors positive / negative ──────────────────────────────────────────
    factors_positive: List[str] = []
    factors_negative: List[str] = []

    for feat in top_features:
        label = _label_for_feature(feat)
        val = feature_values.get(feat, 0.0)
        interp = _interpretation_for_feature(feat)
        reason = interp.get("reason", "")
        pct = feature_impact_pct.get(feat, 0.0)
        scale = _semantic_impact_label(pct)
        pct_str = f"{pct:.0f}%" if pct > 0 else ""

        if shap_values and feat in shap_values:
            sv = shap_values[feat]
            direction = _feature_direction_for_plan(sv, feat)
            pct_label = f", impacto {pct_str}" if pct_str else ""
            detail = (
                f"{label} = {val:.2f} | SHAP {sv:+.4f} (escala {scale}{pct_label})"
                f" — {direction}"
            )
            if "mejora" in direction:
                factors_positive.append(f"✓ {detail}. {reason}.")
            elif "perjudica" in direction:
                factors_negative.append(f"✗ {detail}. {reason}.")
            else:
                factors_positive.append(f"~ {detail}. {reason}.")
        else:
            # fallback: usar percentil para determinar si es bueno o malo
            pct_rank = feature_percentiles.get(feat, 50.0)
            low_is_good = interp.get("low_is_good", False)
            high_is_good = interp.get("high_is_good", False)
            if low_is_good:
                if pct_rank <= 50:
                    factors_positive.append(
                        f"✓ {label} = {val:.2f} (percentil {pct_rank:.0f} — valor bajo, favorable). {reason}."
                    )
                else:
                    factors_negative.append(
                        f"✗ {label} = {val:.2f} (percentil {pct_rank:.0f} — valor alto, desfavorable). {reason}."
                    )
            elif high_is_good:
                if pct_rank >= 50:
                    factors_positive.append(
                        f"✓ {label} = {val:.2f} (percentil {pct_rank:.0f} — valor alto, favorable). {reason}."
                    )
                else:
                    factors_negative.append(
                        f"✗ {label} = {val:.2f} (percentil {pct_rank:.0f} — valor bajo, desfavorable). {reason}."
                    )
            else:
                factors_positive.append(
                    f"~ {label} = {val:.2f} (percentil {pct_rank:.0f}, neutral). {reason}."
                )

    # ── Technical detail ──────────────────────────────────────────────────────
    method = "SHAP con proxy Ridge" if shap_used else "heurística fallback (delta vs. promedio)"
    tech_lines: List[str] = [
        f"Método de explicación : {method}",
        f"Planes comparados     : {compared_plans_count}",
        f"Calidad explicación   : {explanation_quality}",
    ]
    for r in explanation_quality_reasons:
        tech_lines.append(f"  • {r}")

    if top_features and shap_values:
        tech_lines.append("")
        header = (
            f"  {'Feature':<28} {'Valor':>10}  {'SHAP':>10}  "
            f"{'Impacto':>8}  {'Escala':<6}  {'Dirección':<22}  {'Percentil':>9}"
        )
        sep = "  " + "─" * (len(header) - 2)
        tech_lines.append(header)
        tech_lines.append(sep)
        for feat in top_features:
            val = feature_values.get(feat, 0.0)
            sv = shap_values.get(feat, 0.0)
            pct = feature_impact_pct.get(feat, 0.0)
            scale = _semantic_impact_label(pct)
            direction = _feature_direction_for_plan(sv, feat)
            pctile = feature_percentiles.get(feat, 50.0)
            tech_lines.append(
                f"  {feat:<28} {val:>10.2f}  {sv:>+10.4f}  "
                f"{pct:>7.1f}%  {scale:<6}  {direction:<22}  {pctile:>8.0f}°"
            )
    elif top_features:
        tech_lines.append("")
        for feat in top_features:
            val = feature_values.get(feat, 0.0)
            pctile = feature_percentiles.get(feat, 50.0)
            tech_lines.append(f"  {feat:<28} {val:>10.2f}   percentil {pctile:.0f}°")

    if warnings:
        tech_lines.append("")
        tech_lines.append("Advertencias:")
        for w in warnings:
            tech_lines.append(f"  ! {w}")

    technical_detail = "\n".join(tech_lines)

    # ── Combined natural_language_summary ─────────────────────────────────────
    SEP60 = "─" * 60
    sections: List[str] = [
        f"PLAN ELEGIDO: {join_order}",
        f"Costo estimado: {chosen_score:.4f} | Candidatos comparados: {compared_plans_count}",
    ]
    if runner_up_plan and runner_up_score is not None:
        pct_better = (
            (runner_up_score - chosen_score) / runner_up_score * 100.0
            if runner_up_score > 0 else 0.0
        )
        if pct_better < 0.5:
            sections.append(
                f"Runner-up: {' -> '.join(runner_up_plan)} "
                f"(costo {runner_up_score:.4f}, equivalente — igual costo estimado)"
            )
        else:
            sections.append(
                f"Runner-up: {' -> '.join(runner_up_plan)} "
                f"(costo {runner_up_score:.4f}, +{pct_better:.1f}% más costoso)"
            )
    sections += [
        "",
        "RESUMEN EJECUTIVO",
        SEP60,
        executive_summary,
    ]
    if factors_positive:
        sections += ["", "FACTORES QUE FAVORECIERON ESTE PLAN"]
        sections += [f"  {f}" for f in factors_positive]
    if factors_negative:
        sections += ["", "FACTORES QUE JUGARON EN CONTRA"]
        sections += [f"  {f}" for f in factors_negative]
    sections += ["", "DETALLE TÉCNICO", SEP60, technical_detail]

    natural_language_summary = "\n".join(sections)

    return {
        "executive_summary": executive_summary,
        "factors_positive": factors_positive,
        "factors_negative": factors_negative,
        "technical_detail": technical_detail,
        "natural_language_summary": natural_language_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11. Contrastive explanation  (elegido vs. runner-up)
# ─────────────────────────────────────────────────────────────────────────────

#: Features que varían entre permutaciones del mismo query → relevantes para
#: la comparación contrastiva. Features fijas (join_count, table_count, etc.)
#: se omiten porque son iguales en todos los planes y no explican la elección.
_CONTRASTIVE_FEATURES: List[str] = [
    "first_table_rows",
    "max_table_rows",
    "base_filtered_rows",
    "sum_intermediate_rows",
    "max_intermediate_rows",
    "avg_intermediate_rows",
    "most_expensive_join_cost",
    "most_expensive_join_position",
    "index_usage_count",
    "full_scan_count",
    "ast_node_count",
    "ast_depth",
    "estimated_cost",
]


def _is_contrastive_feature(feature: str) -> bool:
    return feature in _CONTRASTIVE_FEATURES or feature.startswith("join_")


def _build_join_order_reason(
    chosen_tables: List[str],
    runner_up_tables: List[str],
    chosen_features: Dict[str, float],
    runner_up_features: Dict[str, float],
) -> List[str]:
    """Explica el primer punto donde difieren dos ordenes de JOIN."""
    max_len = min(len(chosen_tables), len(runner_up_tables))
    first_diff: Optional[int] = None
    for idx in range(max_len):
        if chosen_tables[idx] != runner_up_tables[idx]:
            first_diff = idx
            break

    if first_diff is None:
        return []

    if first_diff == 0:
        return [
            f"Los planes ya difieren en la tabla base: {chosen_tables[0]} vs {runner_up_tables[0]}."
        ]

    step = first_diff
    chosen_table = chosen_tables[first_diff]
    runner_table = runner_up_tables[first_diff]
    prefix = " -> ".join(chosen_tables[:first_diff])
    chosen_output = chosen_features.get(f"join_{step}_output_rows", 0.0)
    runner_output = runner_up_features.get(f"join_{step}_output_rows", 0.0)
    chosen_input = chosen_features.get(f"join_{step}_input_rows", 0.0)
    chosen_table_rows = chosen_features.get(f"join_{step}_table_rows", 0.0)
    runner_table_rows = runner_up_features.get(f"join_{step}_table_rows", 0.0)
    chosen_sel = chosen_features.get(f"join_{step}_selectivity", 0.0)
    runner_sel = runner_up_features.get(f"join_{step}_selectivity", 0.0)

    lines = [
        f"Hasta {prefix}, ambos planes venian igual.",
        (
            f"En el paso {step}, el elegido une {chosen_table} antes que {runner_table}: "
            f"{chosen_output:.2f} filas intermedias vs {runner_output:.2f}."
        ),
        (
            f"Calculo del elegido: entrada {chosen_input:.2f} * filas de {chosen_table} "
            f"{chosen_table_rows:.2f} * selectividad {chosen_sel:.6f} = {chosen_output:.2f}."
        ),
        (
            f"Si se unia {runner_table} en ese punto, usaba {runner_table_rows:.2f} filas "
            f"y selectividad {runner_sel:.6f}, generando {runner_output:.2f}."
        ),
    ]

    if runner_output > chosen_output:
        diff = runner_output - chosen_output
        pct = diff / runner_output * 100.0 if runner_output > 0 else 0.0
        lines.append(
            f"Ese primer desvio ahorra {diff:.2f} filas intermedias ({pct:.1f}% menos) y baja el costo acumulado."
        )
    elif abs(runner_output - chosen_output) < 1e-9:
        lines.append(
            "En ese punto los dos producen el mismo intermedio; la ventaja aparece en pasos posteriores."
        )
    else:
        lines.append(
            "En ese punto el elegido no produce menos filas; la ventaja viene de los pasos siguientes o del costo total."
        )

    return lines


def build_contrastive_explanation(
    chosen_idx: int,
    runner_up_idx: Optional[int],
    X: np.ndarray,
    feature_names: List[str],
    feature_dicts: List[Dict[str, float]],
    candidates: List[SelectStatement],
) -> str:
    """
    Genera un bloque de texto que explica la ELECCIÓN (no el costo en aislamiento).

    Responde: ¿por qué el optimizador eligió el plan A y no el plan B?

    Lógica:
    - Si los costos son distintos: la elección se debe a menor costo estimado.
      Muestra qué features difieren y en qué dirección favorecen al elegido.
    - Si los costos son iguales: SHAP no puede explicar la elección porque no
      hay diferencia de costo. La elección fue por orden de enumeración (desempate).
      Muestra igualmente las diferencias de features para contexto.
    """
    chosen_plan = candidates[chosen_idx]
    chosen_tables = [chosen_plan.from_table] + (
        [jc.table for jc in chosen_plan.join_clauses] if chosen_plan.join_clauses else []
    )
    chosen_order = " -> ".join(chosen_tables)
    chosen_cost = feature_dicts[chosen_idx].get("estimated_cost", 0.0)

    if runner_up_idx is None:
        return (
            f"EXPLICACIÓN DE LA ELECCIÓN\n"
            f"  Plan elegido : {chosen_order}\n"
            f"  No hay plan alternativo para comparar (solo un candidato)."
        )

    ru_plan = candidates[runner_up_idx]
    ru_tables = [ru_plan.from_table] + (
        [jc.table for jc in ru_plan.join_clauses] if ru_plan.join_clauses else []
    )
    ru_order = " -> ".join(ru_tables)
    ru_cost = feature_dicts[runner_up_idx].get("estimated_cost", 0.0)

    cost_diff = ru_cost - chosen_cost
    costs_equal = abs(cost_diff) < 0.5

    lines: List[str] = [
        "EXPLICACIÓN DE LA ELECCIÓN",
        f"  Plan elegido : {chosen_order}  (costo {chosen_cost:.2f})",
        f"  Runner-up    : {ru_order}  (costo {ru_cost:.2f})",
        "",
    ]

    if costs_equal:
        lines += [
            "  ► Los dos planes tienen el MISMO costo estimado.",
            "    El optimizador eligió entre alternativas equivalentes.",
            "    Criterio de desempate: primer plan enumerado en la búsqueda.",
            "    SHAP explica el perfil de costo del plan pero NO la elección,",
            "    porque no hubo ventaja de costo que justificar.",
        ]
    elif cost_diff > 0:
        # chosen_cost < ru_cost → el elegido es más barato
        pct = cost_diff / ru_cost * 100.0 if ru_cost > 0 else 0.0
        lines += [
            f"  ► El plan elegido es {pct:.1f}% más barato que el runner-up.",
            "    Regla de decisión: menor costo estimado gana.",
            "",
            "  Diferencias que explican la ventaja de costo:",
        ]
    else:
        # chosen_cost > ru_cost → nuestro cálculo XAI no replica exactamente
        # el modelo interno del optimizador (diferencia de selectividades/estadísticas)
        pct = abs(cost_diff) / chosen_cost * 100.0 if chosen_cost > 0 else 0.0
        lines += [
            f"  ► Nota: nuestra fórmula XAI estima {pct:.1f}% más costo para el plan elegido.",
            "    El optimizador lo eligió igual — su modelo interno difiere de la aproximación XAI.",
            "    (Las selectividades reales del optimizador no son accesibles directamente.)",
            "",
            "  Diferencias estructurales entre los planes:",
        ]

    divergence_lines = _build_join_order_reason(
        chosen_tables=chosen_tables,
        runner_up_tables=ru_tables,
        chosen_features=feature_dicts[chosen_idx],
        runner_up_features=feature_dicts[runner_up_idx],
    )
    if divergence_lines:
        lines += ["", "  Punto clave del orden de JOIN:"]
        lines += [f"    {line}" for line in divergence_lines]

    # ── tabla comparativa de features que varían ──────────────────────────
    diffs_to_show: List[Tuple[str, float, float, float]] = []  # (feat, chosen, ru, diff)
    for feat in feature_names:
        if not _is_contrastive_feature(feat):
            continue
        v_chosen = feature_dicts[chosen_idx].get(feat, 0.0)
        v_ru = feature_dicts[runner_up_idx].get(feat, 0.0)
        diff = v_chosen - v_ru
        if abs(diff) > 1e-9:
            diffs_to_show.append((feat, v_chosen, v_ru, diff))

    if diffs_to_show:
        lines.append("")
        header = (
            f"  {'Feature':<28} {'Elegido':>12}  {'Runner-up':>12}  "
            f"{'Diferencia':>12}  Impacto"
        )
        lines.append(header)
        lines.append("  " + "─" * (len(header) - 2))
        for feat, v_c, v_r, diff in diffs_to_show:
            label = _label_for_feature(feat)
            interp = _interpretation_for_feature(feat)
            # determinar si la diferencia favorece al elegido
            if interp.get("low_is_good"):
                favorable = diff < 0  # elegido tiene menos → bueno
            elif interp.get("high_is_good"):
                favorable = diff > 0  # elegido tiene más → bueno
            else:
                favorable = None

            if feat == "estimated_cost":
                impact_str = "↓ menor costo → ganó" if diff < 0 else "↑ mayor costo"
            elif favorable is True:
                impact_str = "✓ favorece al elegido"
            elif favorable is False:
                impact_str = "✗ juega en contra"
            else:
                impact_str = "~ neutral"

            lines.append(
                f"  {feat:<28} {v_c:>12.2f}  {v_r:>12.2f}  "
                f"{diff:>+12.2f}  {impact_str}"
            )
        lines.append("")
        lines.append(f"  (Etiquetas: {', '.join(f'{f}={_label_for_feature(f)}' for f,*_ in diffs_to_show[:3])})")
    else:
        lines += [
            "",
            "  Las features relevantes son idénticas entre ambos planes.",
            "  La diferencia de costo se debe a la selectividad interna del optimizador,",
            "  no a características estructurales observables.",
        ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def explain_optimizer_decision(
    stmt: SelectStatement,
    optimizer: QueryOptimizer,
    max_plans: int = 50,
    top_k_features: int = 5,
    random_state: Optional[int] = 42,
    use_lime: bool = False,
) -> Explanation:
    """
    Devuelve una Explanation de por qué el optimizador eligió un AST para un SELECT.

    Pasos:
      1. Obtener el plan optimizado real (optimizer.optimize).
      2. Generar hasta max_plans candidatos (permutaciones de join order).
      3. Extraer features tabulares por plan (build_feature_matrix).
      4. Aplicar SHAP sobre modelo proxy (Ridge o heurística) para explicar
         qué features estructurales explican las variaciones de estimated_cost.
      5. Fallback: si SHAP no está disponible, comparar deltas de features.
      6. Construir Explanation con resumen en lenguaje natural.
    """
    if random_state is not None:
        np.random.seed(random_state)

    warnings_acc: List[str] = []

    # ── 1. Plan optimizado (decisión real del optimizador) ──────────────────
    optimized_plan = optimizer.optimize(stmt)

    # ── 2. Candidatos (el plan optimizado debe estar incluido) ──────────────
    # BayesOptimizer ancla la tabla FROM y solo permuta los JOINs.
    # Generar candidatos con el mismo espacio de búsqueda evita comparar
    # el plan elegido contra planes que Bayes nunca consideró.
    fix_base = isinstance(optimizer, BayesOptimizer)
    candidates = generate_candidate_plans(stmt, max_plans=max_plans, fix_base_table=fix_base)

    opt_key = _plan_key(optimized_plan)
    candidate_keys = [_plan_key(p) for p in candidates]

    if opt_key not in candidate_keys:
        # El plan optimizado no está en las permutaciones generadas: insertarlo
        candidates.insert(0, copy.deepcopy(optimized_plan))
        candidate_keys.insert(0, opt_key)
        warnings_acc.append(
            "El plan optimizado no estaba entre los candidatos generados y fue insertado."
        )

    chosen_idx = candidate_keys.index(opt_key)

    # ── 3. Matriz de features ────────────────────────────────────────────────
    X, feature_names, feature_dicts = build_feature_matrix(candidates, stmt, optimizer)

    if X.shape[0] == 0:
        warnings_acc.append("No se pudieron extraer features. Explicación vacía.")
        return Explanation(
            optimizer_name=type(optimizer).__name__,
            chosen_plan=[],
            top_features=[],
            metadata={"warning": "No features extracted"},
            warnings=warnings_acc,
        )

    chosen_features = feature_dicts[chosen_idx]
    chosen_score = float(chosen_features.get("estimated_cost", 0.0))

    # ── 4. Explicación SHAP ──────────────────────────────────────────────────
    shap_dict, top_features, shap_used, shap_warnings = compute_shap_explanation(
        X, feature_names, chosen_idx, top_k_features
    )
    warnings_acc.extend(shap_warnings)

    # ── 5. Fallback si SHAP no disponible ────────────────────────────────────
    if not shap_used:
        _, fallback_top = compute_fallback_explanation(
            X, feature_names, chosen_idx, top_k_features
        )
        top_features = fallback_top
        shap_dict = None

    # â”€â”€ 5b. ExplicaciÃ³n local LIME-like â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lime_dict: Dict[str, float] = {}
    lime_top_features: List[str] = []
    lime_used = False
    if use_lime:
        lime_dict, lime_top_features, lime_used, lime_warnings = compute_lime_explanation(
            X,
            feature_names,
            chosen_idx,
            top_k_features,
            random_state=random_state,
        )
        if not lime_used:
            warnings_acc.extend(lime_warnings)

    # ── 6. Orden de tablas del plan elegido ──────────────────────────────────
    chosen_tables = [optimized_plan.from_table] + (
        [jc.table for jc in optimized_plan.join_clauses]
        if optimized_plan.join_clauses
        else []
    )

    # ── 7. Capa de interpretación ─────────────────────────────────────────────
    feature_percentiles = compute_feature_percentiles(X, feature_names, chosen_idx)
    feature_impact_pct = compute_relative_impact(shap_dict) if shap_dict else {}
    runner_up_tables, runner_up_score, runner_up_idx = find_runner_up(
        candidates, X, feature_names, chosen_idx
    )
    explanation_quality, quality_reasons = compute_explanation_quality(
        n_plans=len(candidates),
        shap_used=shap_used,
        X=X,
        feature_names=feature_names,
    )

    # ── 8. Explicación contrastiva (elegido vs. runner-up) ────────────────────
    contrastive = build_contrastive_explanation(
        chosen_idx=chosen_idx,
        runner_up_idx=runner_up_idx,
        X=X,
        feature_names=feature_names,
        feature_dicts=feature_dicts,
        candidates=candidates,
    )

    # ── 9. Explicación estructurada ───────────────────────────────────────────
    structured = build_structured_explanation(
        chosen_plan=chosen_tables,
        chosen_score=chosen_score,
        compared_plans_count=len(candidates),
        top_features=top_features,
        shap_values=shap_dict,
        feature_values=chosen_features,
        shap_used=shap_used,
        warnings=warnings_acc,
        feature_percentiles=feature_percentiles,
        feature_impact_pct=feature_impact_pct,
        runner_up_plan=runner_up_tables,
        runner_up_score=runner_up_score,
        explanation_quality=explanation_quality,
        explanation_quality_reasons=quality_reasons,
    )
    nl_summary = structured["natural_language_summary"]

    # ── 10. Construir Explanation ─────────────────────────────────────────────
    metadata: Dict[str, str] = {
        "optimizer_type": type(optimizer).__name__,
        "max_plans": str(max_plans),
        "top_k_features": str(top_k_features),
        "random_state": str(random_state),
        "lime_enabled": str(use_lime),
        "natural_language_summary": nl_summary,
    }

    return Explanation(
        optimizer_name=type(optimizer).__name__,
        chosen_plan=chosen_tables,
        top_features=top_features,
        metadata=metadata,
        selected_plan_score=chosen_score,
        compared_plans_count=len(candidates),
        shap_used=shap_used,
        lime_used=lime_used,
        feature_values_for_selected_plan=chosen_features,
        shap_values_for_selected_plan=shap_dict or {},
        lime_values_for_selected_plan=lime_dict,
        lime_top_features=lime_top_features,
        natural_language_summary=nl_summary,
        warnings=warnings_acc,
        executive_summary=structured["executive_summary"],
        factors_positive=structured["factors_positive"],
        factors_negative=structured["factors_negative"],
        runner_up_plan=runner_up_tables or [],
        runner_up_score=runner_up_score,
        explanation_quality=explanation_quality,
        explanation_quality_reasons=quality_reasons,
        feature_impact_pct=feature_impact_pct,
        feature_percentiles=feature_percentiles,
        technical_detail=structured["technical_detail"],
        contrastive_explanation=contrastive,
    )

