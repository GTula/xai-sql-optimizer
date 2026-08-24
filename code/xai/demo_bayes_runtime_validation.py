"""
Demo de validacion Bayes: estimado vs runtime real por plan candidato.

Esta demo es intencionalmente mas exigente que demo_explainer.py:
- usa filtros locales sobre tablas de dimension;
- Bayes los aplica en la estimacion;
- el executor actual aplica WHERE al final.

Si la validacion funciona, deberiamos ver diferencias entre lo estimado
y lo ejecutado realmente. Eso ayuda a detectar cuando la explicacion del
modelo de costo no coincide con el comportamiento del motor.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from database import Database
from sql.optimizer import BayesOptimizer
from sql.parser import Parser, SelectStatement
from xai.optimizer_explainer import (
    _plan_key,
    build_feature_matrix,
    generate_candidate_plans,
)
from xai.runtime_trace import RuntimeTrace


SQL = (
    "SELECT fact.id, departments.priority, locations.zone "
    "FROM fact "
    "JOIN locations ON fact.loc_id = locations.id "
    "JOIN departments ON fact.dept_id = departments.id "
    "WHERE departments.priority = 1 AND locations.zone = 1"
)


def q_error(estimated: float, real: float) -> float:
    est = max(abs(float(estimated)), 1.0)
    obs = max(abs(float(real)), 1.0)
    return max(est / obs, obs / est)


def rel_error_pct(estimated: float, real: float) -> float:
    return abs(float(estimated) - float(real)) / max(abs(float(real)), 1.0) * 100.0


def plan_order(plan: SelectStatement) -> str:
    return " -> ".join(_plan_key(plan))


def build_validation_db(base_dir: str) -> Database:
    db = Database(os.path.join(base_dir, "bayes_runtime_validation.db"), optimizer_type="bayes")

    db.execute("CREATE TABLE fact (id INTEGER, dept_id INTEGER, loc_id INTEGER)")
    db.execute("CREATE TABLE departments (id INTEGER, priority INTEGER)")
    db.execute("CREATE TABLE locations (id INTEGER, zone INTEGER)")

    # 50 departamentos, pero solo 2 pasan el filtro priority = 1.
    for dept_id in range(1, 51):
        priority = 1 if dept_id <= 2 else 0
        db.execute(f"INSERT INTO departments VALUES ({dept_id}, {priority})")

    # 40 ubicaciones, pero solo 4 pasan el filtro zone = 1.
    for loc_id in range(1, 41):
        zone = 1 if loc_id <= 4 else 0
        db.execute(f"INSERT INTO locations VALUES ({loc_id}, {zone})")

    # Fact grande y uniforme. Cada fila matchea un departamento y una ubicacion.
    # El WHERE se evalua al final en el executor actual.
    for row_id in range(1, 2001):
        dept_id = ((row_id - 1) % 50) + 1
        loc_id = ((row_id - 1) % 40) + 1
        db.execute(f"INSERT INTO fact VALUES ({row_id}, {dept_id}, {loc_id})")

    db.execute("ANALYSE")
    return db


def collect_candidate_metrics(db: Database, stmt: SelectStatement) -> List[Dict[str, float]]:
    optimizer = db.engine.optimizer
    candidates = generate_candidate_plans(
        stmt,
        max_plans=50,
        fix_base_table=isinstance(optimizer, BayesOptimizer),
    )
    _, _, estimated_features = build_feature_matrix(candidates, stmt, optimizer)

    rows: List[Dict[str, float]] = []
    for idx, (candidate, estimated) in enumerate(zip(candidates, estimated_features)):
        _, raw_trace = db.engine.execute_plan_with_trace(candidate)
        runtime = RuntimeTrace.from_engine_trace(raw_trace)
        real = runtime.to_features()

        est_work = float(estimated.get("sum_intermediate_rows", 0.0))
        real_work = float(real.get("sum_intermediate_rows", 0.0))

        rows.append({
            "idx": float(idx),
            "plan": plan_order(candidate),
            "estimated_cost": float(estimated.get("estimated_cost", 0.0)),
            "estimated_work": est_work,
            "real_work": real_work,
            "final_rows": float(runtime.final_rows),
            "work_q_error": q_error(est_work, real_work),
            "work_rel_error_pct": rel_error_pct(est_work, real_work),
            "join_1_est": float(estimated.get("join_1_output_rows", 0.0)),
            "join_1_real": float(real.get("join_1_output_rows", 0.0)),
            "join_2_est": float(estimated.get("join_2_output_rows", 0.0)),
            "join_2_real": float(real.get("join_2_output_rows", 0.0)),
        })
    return rows


def print_metrics(rows: List[Dict[str, float]], chosen_plan: str) -> None:
    print("\nQuery:")
    print(f"  {SQL}")
    print("\nComparacion candidato a candidato (Bayes):")
    print(
        f"{'#':>2}  {'plan':<38} {'est_work':>10} {'real_work':>10} "
        f"{'qerr':>8} {'err%':>8} {'final':>8}"
    )
    print("-" * 96)
    for row in rows:
        marker = "*" if row["plan"] == chosen_plan else " "
        print(
            f"{marker}{int(row['idx']):>1}  {row['plan']:<38} "
            f"{row['estimated_work']:>10.2f} {row['real_work']:>10.2f} "
            f"{row['work_q_error']:>8.2f} {row['work_rel_error_pct']:>8.1f} "
            f"{row['final_rows']:>8.0f}"
        )

    print("\nDetalle por paso:")
    for row in rows:
        print(f"- {row['plan']}")
        print(
            f"  join_1_output_rows: estimado={row['join_1_est']:.2f}, "
            f"real={row['join_1_real']:.2f}, qerr={q_error(row['join_1_est'], row['join_1_real']):.2f}"
        )
        print(
            f"  join_2_output_rows: estimado={row['join_2_est']:.2f}, "
            f"real={row['join_2_real']:.2f}, qerr={q_error(row['join_2_est'], row['join_2_real']):.2f}"
        )

    estimated_best = min(rows, key=lambda r: r["estimated_cost"])
    best_real_work = min(row["real_work"] for row in rows)
    real_best_plans = [
        row["plan"]
        for row in rows
        if abs(row["real_work"] - best_real_work) < 1e-9
    ]
    chosen = next(row for row in rows if row["plan"] == chosen_plan)
    regret = chosen["real_work"] - best_real_work

    print("\nRanking:")
    print(f"  Plan elegido por Bayes       : {chosen_plan}")
    print(f"  Mejor por costo estimado     : {estimated_best['plan']}")
    print(f"  Mejor(es) por trabajo real   : {' | '.join(real_best_plans)}")
    print(f"  Regret real del elegido      : {regret:.2f} filas intermedias")

    avg_qerr = sum(row["work_q_error"] for row in rows) / len(rows)
    max_qerr = max(row["work_q_error"] for row in rows)
    print("\nResumen de fidelidad:")
    print(f"  q-error promedio trabajo     : {avg_qerr:.2f}")
    print(f"  q-error maximo trabajo       : {max_qerr:.2f}")
    print(
        "  Lectura                      : valores lejos de 1.0 indican que "
        "la estimacion no coincide con el runtime actual."
    )


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="bayes_runtime_validation_")
    try:
        db = build_validation_db(tmp)
        stmt = Parser.parse(SQL)
        optimized = db.engine.optimizer.optimize(stmt)
        chosen_plan = plan_order(optimized)
        rows = collect_candidate_metrics(db, stmt)
        print_metrics(rows, chosen_plan)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
