"""Reproducible synthetic datasets and structurally diverse SQL workloads.

The research workload is deliberately independent from the legacy 100-query
file.  It exposes the experimental factors requested by the paper review:
number of tables/joins/predicates, data size, distribution, and random seed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from database import Database


SIZE_PRESETS: Dict[str, int] = {
    "small": 300,
    "medium": 1_000,
    "large": 3_000,
}
SUPPORTED_DISTRIBUTIONS = ("uniform", "skewed", "correlated")

# table, fact foreign key, human-readable value column
DIMENSIONS: Tuple[Tuple[str, str, str], ...] = (
    ("departments", "dept_id", "dept_name"),
    ("locations", "loc_id", "city"),
    ("employees", "employee_id", "employee_name"),
    ("projects", "project_id", "project_name"),
    ("categories", "category_id", "category_name"),
)


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 0
    size_label: str = "medium"
    distribution: str = "uniform"
    fact_rows: int | None = None

    def resolved_fact_rows(self) -> int:
        if self.fact_rows is not None:
            if self.fact_rows < 30:
                raise ValueError("fact_rows debe ser al menos 30")
            return int(self.fact_rows)
        if self.size_label not in SIZE_PRESETS:
            raise ValueError(f"Tamaño no soportado: {self.size_label}")
        return SIZE_PRESETS[self.size_label]

    def validate(self) -> None:
        self.resolved_fact_rows()
        if self.distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"Distribución no soportada: {self.distribution}")

    def experiment_id(self) -> str:
        return f"{self.size_label}_{self.distribution}_seed{self.seed}"


@dataclass(frozen=True)
class QuerySpec:
    query_id: int
    sql: str
    family: str
    table_count: int
    join_count: int
    predicate_count: int
    predicate_logic: str
    selectivity_band: str
    tables: str


def _dimension_cardinalities(fact_rows: int) -> Dict[str, int]:
    return {
        "departments": max(4, min(20, fact_rows // 75)),
        "locations": max(8, min(100, fact_rows // 20)),
        "employees": max(12, min(200, fact_rows // 10)),
        "projects": max(6, min(80, fact_rows // 30)),
        "categories": max(3, min(20, fact_rows // 100)),
    }


def _sample_key(
    rng: np.random.Generator,
    cardinality: int,
    distribution: str,
    row_index: int,
) -> int:
    if distribution == "uniform":
        return int(rng.integers(1, cardinality + 1))
    if distribution == "skewed":
        return int((rng.zipf(1.7) - 1) % cardinality + 1)
    # Correlated values share a latent group derived from the row position.
    latent_group = row_index % max(2, min(cardinality, 7))
    jitter = int(rng.integers(0, max(1, cardinality // 8 + 1)))
    return int((latent_group + jitter) % cardinality + 1)


def _write_table(db: Database, table_name: str, rows: Iterable[Sequence[object]]) -> None:
    table = db.catalog.get_table(table_name)
    if table is None:
        raise ValueError(f"Tabla no registrada: {table_name}")
    with open(table.table_file, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([column.name for column in table.columns])
        writer.writerows(rows)


def build_experimental_db(
    base_dir: str,
    optimizer_type: str,
    config: DatasetConfig,
) -> Database:
    """Create and analyze a six-table star schema for one experimental run."""
    config.validate()
    fact_rows = config.resolved_fact_rows()
    rng = np.random.default_rng(config.seed)
    cardinalities = _dimension_cardinalities(fact_rows)

    # Every experimental run gets an isolated catalog. Reusing a single catalog
    # for homonymous tables would mix statistics across seeds/configurations and
    # makes catalog scans grow progressively slower during a large grid.
    run_dir = os.path.join(base_dir, f"{optimizer_type}_{config.experiment_id()}")
    os.makedirs(run_dir, exist_ok=True)
    db_path = os.path.join(run_dir, "experiment.db")
    db = Database(db_path, optimizer_type=optimizer_type)
    db.execute(
        "CREATE TABLE fact (id INTEGER, dept_id INTEGER, loc_id INTEGER, "
        "employee_id INTEGER, project_id INTEGER, category_id INTEGER, amount FLOAT)"
    )
    for table_name, _, value_column in DIMENSIONS:
        db.execute(
            f"CREATE TABLE {table_name} (id INTEGER, {value_column} VARCHAR(50))"
        )

    for table_name, _, _ in DIMENSIONS:
        count = cardinalities[table_name]
        _write_table(
            db,
            table_name,
            ((idx, f"{table_name}_{idx}") for idx in range(1, count + 1)),
        )

    fact_data = []
    for row_index in range(1, fact_rows + 1):
        keys = {
            table_name: _sample_key(
                rng, cardinalities[table_name], config.distribution, row_index
            )
            for table_name, _, _ in DIMENSIONS
        }
        if config.distribution == "correlated":
            dept = keys["departments"]
            keys["employees"] = (dept - 1) % cardinalities["employees"] + 1
            keys["projects"] = (dept - 1) % cardinalities["projects"] + 1
            keys["categories"] = (dept - 1) % cardinalities["categories"] + 1
        amount = round(float(rng.lognormal(mean=4.0, sigma=0.7)), 2)
        fact_data.append(
            (
                row_index,
                keys["departments"],
                keys["locations"],
                keys["employees"],
                keys["projects"],
                keys["categories"],
                amount,
            )
        )
    _write_table(db, "fact", fact_data)
    db.execute("ANALYSE")
    return db


def dataset_manifest(config: DatasetConfig) -> Dict[str, object]:
    config.validate()
    fact_rows = config.resolved_fact_rows()
    return {
        **asdict(config),
        "fact_rows": fact_rows,
        "dimension_cardinalities": _dimension_cardinalities(fact_rows),
        "schema": "six_table_star",
    }


def _join_sql(dimensions: Sequence[Tuple[str, str, str]]) -> str:
    return " ".join(
        f"JOIN {table} ON fact.{foreign_key} = {table}.id"
        for table, foreign_key, _ in dimensions
    )


def generate_workload(max_orders_per_table_count: int = 12) -> List[QuerySpec]:
    """Generate deterministic query templates spanning one to six tables."""
    specs: List[QuerySpec] = []

    def add(
        sql: str,
        family: str,
        tables: Sequence[str],
        predicate_count: int,
        predicate_logic: str,
        selectivity_band: str,
    ) -> None:
        specs.append(
            QuerySpec(
                query_id=len(specs) + 1,
                sql=sql.rstrip(";") + ";",
                family=family,
                table_count=len(tables),
                join_count=max(0, len(tables) - 1),
                predicate_count=predicate_count,
                predicate_logic=predicate_logic,
                selectivity_band=selectivity_band,
                tables="|".join(tables),
            )
        )

    add("SELECT fact.id FROM fact", "single_table", ["fact"], 0, "none", "none")
    add(
        "SELECT fact.id FROM fact WHERE fact.id <= 10",
        "single_table",
        ["fact"],
        1,
        "single",
        "low",
    )
    for table_name, _, value_column in DIMENSIONS:
        add(
            f"SELECT {table_name}.id, {table_name}.{value_column} FROM {table_name}",
            "single_table",
            [table_name],
            0,
            "none",
            "none",
        )
        add(
            f"SELECT {table_name}.id FROM {table_name} WHERE {table_name}.id = 1",
            "single_table",
            [table_name],
            1,
            "single",
            "low",
        )

    for table_count in range(2, 7):
        dimension_count = table_count - 1
        orders = list(itertools.permutations(DIMENSIONS, dimension_count))
        for order in orders[:max_orders_per_table_count]:
            first_table = order[0][0]
            first_foreign_key = order[0][1]
            second_foreign_key = order[1][1] if len(order) > 1 else "id"
            tables = ["fact"] + [item[0] for item in order]
            base = (
                f"SELECT fact.id, {first_table}.id FROM fact "
                f"{_join_sql(order)}"
            )
            variants = (
                ("", 0, "none", "none"),
                (" WHERE fact.id <= 10", 1, "single", "low"),
                (
                    f" WHERE fact.{first_foreign_key} = 1",
                    1,
                    "single",
                    "distribution_sensitive",
                ),
                (
                    f" WHERE fact.{first_foreign_key} = 1 "
                    f"AND fact.{second_foreign_key} <= 2",
                    2,
                    "and",
                    "correlation_sensitive",
                ),
                (
                    f" WHERE fact.{first_foreign_key} = 1 "
                    f"OR fact.{second_foreign_key} = 1",
                    2,
                    "or",
                    "distribution_sensitive",
                ),
            )
            for where_sql, predicate_count, logic, selectivity in variants:
                add(
                    base + where_sql,
                    f"join_{table_count}_tables",
                    tables,
                    predicate_count,
                    logic,
                    selectivity,
                )
    return specs


def write_workload(
    sql_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    max_orders_per_table_count: int = 12,
) -> List[QuerySpec]:
    specs = generate_workload(max_orders_per_table_count=max_orders_per_table_count)
    sql_target = Path(sql_path)
    manifest_target = Path(manifest_path)
    sql_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    sql_target.write_text("\n".join(spec.sql for spec in specs) + "\n", encoding="utf-8")
    with manifest_target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(specs[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(spec) for spec in specs)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el workload sintético extendido")
    parser.add_argument("--sql", default="docs/WORKLOAD_EXTENDED.sql")
    parser.add_argument("--manifest", default="docs/WORKLOAD_EXTENDED_manifest.csv")
    parser.add_argument("--orders", type=int, default=12)
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", choices=sorted(SIZE_PRESETS), default="medium")
    parser.add_argument(
        "--distribution", choices=SUPPORTED_DISTRIBUTIONS, default="uniform"
    )
    args = parser.parse_args()
    specs = write_workload(args.sql, args.manifest, args.orders)
    print(f"Workload generado: {len(specs)} consultas")
    if args.dataset_manifest:
        config = DatasetConfig(
            seed=args.seed, size_label=args.size, distribution=args.distribution
        )
        Path(args.dataset_manifest).write_text(
            json.dumps(dataset_manifest(config), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
