#!/usr/bin/env python3
"""Entrena Bayesian Networks por tabla y guarda pickles en bayescard/vendor/Models.

Este trainer usa pgmpy directamente para construir una BN discreta simple por tabla:
- cada columna se trata como variable discreta;
- el modelo aprende marginales desde los datos;
- se serializa un wrapper compatible con el cost model.

Uso:
  python bayescard/train_bn.py --db-dir path/to/csv_tables [--out-dir bayescard/vendor/Models]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

try:
    from pgmpy.models import DiscreteBayesianNetwork
except Exception:  # pragma: no cover - fallback for older pgmpy
    from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork

from pgmpy.inference import VariableElimination

logger = logging.getLogger("train_bn")


@dataclass
class TrainedBN:
    """Wrapper simple y picklable para un BN entrenado por tabla."""

    table_name: str
    model: object
    infer_machine: object
    node_names: List[str]
    domain: Dict[str, List[str]]


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column_name in normalized.columns:
        normalized[column_name] = normalized[column_name].fillna("").astype(str)
    return normalized


def train_table(table_csv: str, out_dir: str, rows_to_use: int = 200000) -> str:
    table_name = os.path.splitext(os.path.basename(table_csv))[0]
    logger.info("Training BN for table %s from %s", table_name, table_csv)

    df = pd.read_csv(table_csv)
    if df.empty:
        logger.warning("Table %s is empty, skipping", table_name)
        return ""

    if rows_to_use > 0 and len(df) > rows_to_use:
        df = df.sample(rows_to_use, random_state=0)

    df = _normalize_dataframe(df)
    node_names = list(df.columns)
    domain = {column_name: sorted(df[column_name].dropna().unique().tolist()) for column_name in node_names}

    # Simple but real BN: all columns are independent variables with learned marginals.
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(node_names)
    model.fit(df)
    infer_machine = VariableElimination(model)

    artifact = TrainedBN(
        table_name=table_name,
        model=model,
        infer_machine=infer_machine,
        node_names=node_names,
        domain=domain,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{table_name}_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f, pickle.HIGHEST_PROTOCOL)

    logger.info("Saved BN pickle to %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True, help="Directory containing table CSV files")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "vendor", "Models"),
        help="Output dir for BN pickles",
    )
    parser.add_argument("--rows", type=int, default=200000, help="Max rows to use per table")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(args.db_dir):
        logger.error("--db-dir must be an existing directory")
        return 2

    csv_files: List[str] = []
    for root, _dirs, files in os.walk(args.db_dir):
        for filename in files:
            if filename.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, filename))

    if not csv_files:
        logger.error("No CSV files found under db-dir")
        return 3

    for csv_path in sorted(csv_files):
        try:
            train_table(csv_path, args.out_dir, rows_to_use=args.rows)
        except Exception:
            logger.exception("Failed training table from %s", csv_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
