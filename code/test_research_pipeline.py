#!/usr/bin/env python3
"""Regression tests for the paper's reproducible evaluation pipeline."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from sql.parser import Parser
from xai.experimental_workload import (
    DatasetConfig,
    build_experimental_db,
    generate_workload,
)
from xai.optimizer_explainer import generate_candidate_plans
from xai.research_evaluation import ExperimentConfig, run_research_experiment


FIVE_TABLE_QUERY = """
SELECT fact.id FROM fact
JOIN departments ON fact.dept_id = departments.id
JOIN locations ON fact.loc_id = locations.id
JOIN employees ON fact.employee_id = employees.id
JOIN projects ON fact.project_id = projects.id;
"""


class ResearchPipelineTests(unittest.TestCase):
    def test_workload_is_structurally_diverse_and_unique(self) -> None:
        workload = generate_workload(max_orders_per_table_count=12)
        self.assertGreaterEqual(len(workload), 250)
        self.assertEqual(len(workload), len({item.sql for item in workload}))
        self.assertEqual({item.table_count for item in workload}, {1, 2, 3, 4, 5, 6})
        self.assertTrue(any(item.predicate_logic == "or" for item in workload))
        self.assertTrue(any(item.predicate_count == 2 for item in workload))

    def test_candidate_plans_are_connected(self) -> None:
        statement = Parser.parse(FIVE_TABLE_QUERY)
        selinger = generate_candidate_plans(statement, max_plans=50, fix_base_table=False)
        bayes = generate_candidate_plans(statement, max_plans=50, fix_base_table=True)
        self.assertEqual(len(selinger), 48)
        self.assertEqual(len(bayes), 24)
        for plan in selinger + bayes:
            visible = {plan.from_table}
            for join in plan.join_clauses or []:
                self.assertIsNotNone(join.condition)
                referenced = self._referenced_tables(join.condition)
                self.assertIn(join.table, referenced)
                self.assertTrue(referenced.intersection(visible))
                visible.add(join.table)

    def test_dataset_generation_is_reproducible(self) -> None:
        root_a = tempfile.mkdtemp(prefix="dataset_a_")
        root_b = tempfile.mkdtemp(prefix="dataset_b_")
        config = DatasetConfig(seed=7, size_label="small", distribution="correlated")
        try:
            db_a = build_experimental_db(root_a, "selinger", config)
            db_b = build_experimental_db(root_b, "selinger", config)
            try:
                fact_a = Path(db_a.catalog.get_table("fact").table_file).read_bytes()
                fact_b = Path(db_b.catalog.get_table("fact").table_file).read_bytes()
                self.assertEqual(fact_a, fact_b)
                self.assertEqual(len(db_a.catalog.tables), 6)
            finally:
                db_a.close()
                db_b.close()
        finally:
            shutil.rmtree(root_a, ignore_errors=True)
            shutil.rmtree(root_b, ignore_errors=True)

    def test_end_to_end_smoke_run_writes_paper_artifacts(self) -> None:
        output = tempfile.mkdtemp(prefix="research_output_")
        try:
            config = ExperimentConfig(
                seeds=(1,),
                sizes=("small",),
                distributions=("uniform",),
                optimizers=("selinger", "bayes"),
                max_plans=20,
                max_orders_per_table_count=2,
                bootstrap_iterations=20,
                query_limit=22,
            )
            result = run_research_experiment(output, config=config, verbose=False)
            for filename in (
                result.ridge_fidelity_csv,
                result.ridge_fidelity_across_seeds_csv,
                result.shap_stability_per_query_csv,
                result.scalability_csv,
                result.paper_results_md,
                result.experiment_manifest_json,
            ):
                target = Path(output, filename)
                self.assertTrue(target.exists(), filename)
                self.assertGreater(target.stat().st_size, 0, filename)
        finally:
            shutil.rmtree(output, ignore_errors=True)

    @classmethod
    def _referenced_tables(cls, expression) -> set[str]:
        table = getattr(expression, "table", None)
        if table:
            return {table}
        result: set[str] = set()
        for attribute in ("left", "right", "operand"):
            child = getattr(expression, attribute, None)
            if child is not None:
                result.update(cls._referenced_tables(child))
        return result


if __name__ == "__main__":
    unittest.main(verbosity=2)
