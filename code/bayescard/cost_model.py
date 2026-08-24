"""Modelo BayesCard autocontenido para estimar cardinalidades con estadísticas locales."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import csv
import math
import os

from catalog.catalog import Catalog
from sql.parser import BinaryOp, ColumnRef, Literal, UnaryOp
import glob
import logging

logger = logging.getLogger(__name__)

try:
    from bayescard.vendor.Models.BN_single_model import load_BN_single
except Exception:
    load_BN_single = None


@dataclass
class ColumnProfile:
    """Perfil estadístico de una columna."""

    ndv: int
    value_counts: Counter
    min_numeric: Optional[float] = None
    max_numeric: Optional[float] = None


@dataclass
class TableProfile:
    """Perfil estadístico de una tabla."""

    table_name: str
    row_count: int
    columns: Dict[str, ColumnProfile]


class BayesCardCostModel:
    """Estimador BayesCard local basado en perfiles aprendidos desde CSV."""

    def __init__(self, catalog: Optional[Catalog] = None, default_cardinality: float = 1000.0):
        self.catalog = catalog
        self.default_cardinality = float(default_cardinality)
        self._table_profiles: Dict[str, TableProfile] = {}
        self._bn_models: Dict[str, object] = {}
        # cache for BN inference queries: ((table, target_col, frozenset(evidence.items())), tuple(sorted_dist_items))
        self._bn_infer_cache: Dict[Tuple[str, str, frozenset], Dict[str, float]] = {}
        # locate vendored models directory if present
        try:
            vendor_models_dir = os.path.join(os.path.dirname(__file__), "vendor", "Models")
            if os.path.isdir(vendor_models_dir):
                self._vendor_models_dir = vendor_models_dir
            else:
                self._vendor_models_dir = None
        except Exception:
            self._vendor_models_dir = None

    def table_cardinality(self, table_name: str) -> float:
        return float(self._get_table_profile(table_name).row_count)

    def column_ndv(self, table_name: str, column_name: str) -> float:
        profile = self._get_table_profile(table_name)
        column_profile = profile.columns.get(column_name)
        if column_profile is None:
            return 1.0
        return float(max(column_profile.ndv, 1))

    def table_filter_selectivity(self, table_name: str, predicate: Any) -> float:
        if predicate is None:
            return 1.0
        profile = self._get_table_profile(table_name)
        selectivity = self._predicate_selectivity(predicate, table_name, profile)
        return self._clamp_selectivity(selectivity)

    def join_selectivity(self, condition: Any) -> float:
        if condition is None:
            return 1.0

        if isinstance(condition, BinaryOp):
            operator = condition.operator.upper() if isinstance(condition.operator, str) else condition.operator

            if operator == "=":
                left_ref = self._extract_column_ref(condition.left)
                right_ref = self._extract_column_ref(condition.right)

                if left_ref and right_ref:
                    # If both references belong to the same table and a BN is available, try
                    # to compute P(A=B) using conditional inference: P(A=B) = sum_b P(B=b)*P(A=b | B=b)
                    if left_ref[0] == right_ref[0]:
                        table = left_ref[0]
                        # attempt to use BN-based conditional inference
                        right_dist = self._bn_column_distribution(table, right_ref[1])
                        if right_dist is not None:
                            total = 0.0
                            # limit the number of distinct values considered for performance
                            max_vals = 200
                            items = list(right_dist.items())[:max_vals]
                            for v, p_b in items:
                                if p_b <= 0.0:
                                    continue
                                evidence = {right_ref[1]: v}
                                cond_dist = self._bn_conditional_distribution(table, left_ref[1], evidence)
                                p_a_given_b = 0.0
                                if cond_dist is not None:
                                    # probability that left_col equals this specific value v given evidence
                                    p_a_given_b = cond_dist.get(self._normalize_value(v), 0.0)
                                else:
                                    # fallback to marginal of left
                                    left_dist = self._bn_column_distribution(table, left_ref[1])
                                    if left_dist is not None:
                                        p_a_given_b = left_dist.get(self._normalize_value(v), 0.0)
                                total += p_b * p_a_given_b
                            if total > 0.0:
                                return self._clamp_selectivity(total)

                    profile_selectivity = self._profile_join_selectivity(
                        left_ref, right_ref
                    )
                    if profile_selectivity is not None:
                        return profile_selectivity

                    # Cross-table or missing conditional model: if BN marginals are available for both
                    # columns, approximate P(A=B) by dot product of marginals (existing heuristic)
                    left_dist = self._bn_column_distribution(left_ref[0], left_ref[1])
                    right_dist = self._bn_column_distribution(right_ref[0], right_ref[1])
                    if left_dist is not None and right_dist is not None:
                        total = 0.0
                        for v, p1 in left_dist.items():
                            p2 = right_dist.get(v, 0.0)
                            total += p1 * p2
                        return self._clamp_selectivity(total)

                    left_ndv = self.column_ndv(*left_ref)
                    right_ndv = self.column_ndv(*right_ref)
                    return 1.0 / max(left_ndv, right_ndv)

                if left_ref and isinstance(condition.right, Literal):
                    return self._single_column_literal_selectivity(left_ref, condition.right.value)

                if right_ref and isinstance(condition.left, Literal):
                    return self._single_column_literal_selectivity(right_ref, condition.left.value)

            if operator in {"<", "<=", ">", ">="}:
                return 0.33

        return 0.1

    def _profile_join_selectivity(
        self,
        left_ref: Tuple[str, str],
        right_ref: Tuple[str, str],
    ) -> Optional[float]:
        """
        Estima P(left = right) con histogramas de los perfiles CSV.

        Esto captura skew y valores sin match. El fallback 1/max(NDV) asume una
        distribucion uniforme y puede inflar mucho joins con claves raras.
        """
        left_profile = self._get_table_profile(left_ref[0])
        right_profile = self._get_table_profile(right_ref[0])
        left_column = left_profile.columns.get(left_ref[1])
        right_column = right_profile.columns.get(right_ref[1])
        if left_column is None or right_column is None:
            return None

        left_rows = max(left_profile.row_count, 1)
        right_rows = max(right_profile.row_count, 1)
        left_counts = left_column.value_counts
        right_counts = right_column.value_counts
        if not left_counts or not right_counts:
            return None

        if len(left_counts) > len(right_counts):
            left_counts, right_counts = right_counts, left_counts

        matches = sum(
            count * right_counts.get(value, 0)
            for value, count in left_counts.items()
        )
        return self._clamp_selectivity(matches / float(left_rows * right_rows))

    def join_cardinality(self, left_cardinality: float, right_table: str, condition: Any) -> float:
        right_cardinality = self.table_cardinality(right_table)
        return left_cardinality * right_cardinality * self.join_selectivity(condition)

    def _get_table_profile(self, table_name: str) -> TableProfile:
        cached = self._table_profiles.get(table_name)
        if cached is not None:
            return cached

        profile = self._build_table_profile(table_name)
        self._table_profiles[table_name] = profile
        return profile

    def _build_table_profile(self, table_name: str) -> TableProfile:
        table = self.catalog.get_table(table_name) if self.catalog else None
        if table is None or not os.path.exists(table.table_file):
            return TableProfile(table_name=table_name, row_count=int(self.default_cardinality), columns={})

        columns = [column.name for column in table.columns]
        counters = {column_name: Counter() for column_name in columns}
        numeric_values: Dict[str, List[float]] = {column_name: [] for column_name in columns}
        row_count = 0

        with open(table.table_file, "r", encoding="utf-8", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                row_count += 1
                for column_name in columns:
                    raw_value = row.get(column_name, "")
                    counters[column_name][raw_value] += 1
                    numeric_value = self._to_float(raw_value)
                    if numeric_value is not None:
                        numeric_values[column_name].append(numeric_value)

        if row_count == 0:
            return TableProfile(table_name=table_name, row_count=1, columns={})

        column_profiles: Dict[str, ColumnProfile] = {}
        for column_name in columns:
            values = numeric_values[column_name]
            column_profiles[column_name] = ColumnProfile(
                ndv=len(counters[column_name]) if counters[column_name] else 1,
                value_counts=counters[column_name],
                min_numeric=min(values) if values else None,
                max_numeric=max(values) if values else None,
            )

        profile = TableProfile(table_name=table_name, row_count=row_count, columns=column_profiles)

        # Best-effort: try to load a pickled BN model for this table from the vendored Models
        if load_BN_single is not None and self._vendor_models_dir:
            try:
                pattern = os.path.join(self._vendor_models_dir, f"{table_name}*.pkl")
                matches = glob.glob(pattern)
                if matches:
                    path = matches[0]
                    try:
                        bn = load_BN_single(path)
                        self._bn_models[table_name] = bn
                        logger.info(f"Loaded BN model for table {table_name} from {path}")
                    except Exception:
                        logger.exception(f"Failed to load BN model from {path}")
            except Exception:
                logger.exception("Error while searching for vendored BN models")

        return profile

    def _predicate_selectivity(self, predicate: Any, table_name: str, profile: TableProfile) -> float:
        if predicate is None:
            return 1.0

        if isinstance(predicate, UnaryOp):
            if predicate.operator.upper() == "NOT":
                return 1.0 - self._predicate_selectivity(predicate.operand, table_name, profile)
            return 1.0

        if not isinstance(predicate, BinaryOp):
            return 1.0

        operator = predicate.operator.upper() if isinstance(predicate.operator, str) else predicate.operator

        if operator == "AND":
            return self._predicate_selectivity(predicate.left, table_name, profile) * self._predicate_selectivity(
                predicate.right, table_name, profile
            )

        if operator == "OR":
            left = self._predicate_selectivity(predicate.left, table_name, profile)
            right = self._predicate_selectivity(predicate.right, table_name, profile)
            return min(1.0, left + right - left * right)

        left_ref = self._extract_column_ref(predicate.left)
        right_ref = self._extract_column_ref(predicate.right)

        if left_ref and left_ref[0] == table_name and isinstance(predicate.right, Literal):
            return self._literal_selectivity(profile, left_ref[1], predicate.right.value, operator)

        if right_ref and right_ref[0] == table_name and isinstance(predicate.left, Literal):
            return self._literal_selectivity(profile, right_ref[1], predicate.left.value, operator)

        if left_ref and right_ref and left_ref[0] == table_name and right_ref[0] == table_name:
            if operator == "=":
                return 1.0 / max(profile.columns.get(left_ref[1], ColumnProfile(1, Counter())).ndv,
                                 profile.columns.get(right_ref[1], ColumnProfile(1, Counter())).ndv)

        return 1.0

    def _literal_selectivity(self, profile: TableProfile, column_name: str, literal_value: Any, operator: Any) -> float:
        column_profile = profile.columns.get(column_name)
        if column_profile is None:
            return 1.0 / max(profile.row_count, 1)

        if operator == "=":
            literal_key = self._normalize_value(literal_value)
            count = column_profile.value_counts.get(literal_key, 0)
            if count > 0:
                return count / max(profile.row_count, 1)
            # If we have a BN model for this table, try to use it to estimate P(col=val)
            bn_sel = self._bn_selectivity(profile.table_name, column_name, literal_value)
            if bn_sel is not None:
                return bn_sel
            return 1.0 / max(column_profile.ndv, 1)

        if operator == "!=":
            return 1.0 - self._literal_selectivity(profile, column_name, literal_value, "=")

        numeric_literal = self._to_float(literal_value)
        if numeric_literal is None or column_profile.min_numeric is None or column_profile.max_numeric is None:
            return 0.33

        min_value = column_profile.min_numeric
        max_value = column_profile.max_numeric
        if max_value <= min_value:
            return 1.0

        if operator == ">":
            if numeric_literal >= max_value:
                return 0.0
            if numeric_literal <= min_value:
                return 1.0
            return (max_value - numeric_literal) / (max_value - min_value)
        if operator == ">=":
            if numeric_literal > max_value:
                return 0.0
            if numeric_literal <= min_value:
                return 1.0
            return (max_value - numeric_literal) / (max_value - min_value)
        if operator == "<":
            if numeric_literal <= min_value:
                return 0.0
            if numeric_literal >= max_value:
                return 1.0
            return (numeric_literal - min_value) / (max_value - min_value)
        if operator == "<=":
            if numeric_literal < min_value:
                return 0.0
            if numeric_literal >= max_value:
                return 1.0
            return (numeric_literal - min_value) / (max_value - min_value)

        return 0.33

    def _single_column_literal_selectivity(self, column_ref: Tuple[str, str], literal_value: Any) -> float:
        table_name, column_name = column_ref
        profile = self._get_table_profile(table_name)
        return self._literal_selectivity(profile, column_name, literal_value, "=")

    def _bn_column_distribution(self, table_name: str, column_name: str) -> Optional[Dict[str, float]]:
        """Return marginal distribution P(column=value) from a loaded BN model if available.

        Returns a dict mapping stringified values to probabilities, or None if unavailable.
        """
        bn = self._bn_models.get(table_name)
        if bn is None:
            return None

        try:
            # If bn exposes an infer_machine (pgmpy), use it to query the marginal
            if hasattr(bn, "infer_machine") and bn.infer_machine is not None:
                try:
                    res = bn.infer_machine.query([column_name], evidence={})
                    # res may be a DiscreteFactor-like with .values or a dict
                    if hasattr(res, "values"):
                        probs = res.values
                        mapping = None
                        if hasattr(bn, "domain") and column_name in bn.domain:
                            mapping = bn.domain[column_name]
                        dist = {}
                        if mapping is not None:
                            for i, p in enumerate(probs):
                                key = str(mapping[i])
                                dist[key] = float(p)
                        else:
                            for i, p in enumerate(probs):
                                dist[str(i)] = float(p)
                        return dist
                    elif isinstance(res, dict):
                        first = list(res.values())[0]
                        if hasattr(first, "values"):
                            probs = first.values
                            mapping = None
                            if hasattr(bn, "domain") and column_name in bn.domain:
                                mapping = bn.domain[column_name]
                            dist = {}
                            if mapping is not None:
                                for i, p in enumerate(probs):
                                    dist[str(mapping[i])] = float(p)
                            else:
                                for i, p in enumerate(probs):
                                    dist[str(i)] = float(p)
                            return dist
                except Exception:
                    logger.exception("BN infer marginal failed")

            # Fallback: if pomegranate model exists, try predict_proba with empty evidence
            if hasattr(bn, "model") and hasattr(bn.model, "predict_proba"):
                try:
                    probs = bn.model.predict_proba({})
                    if isinstance(probs, list):
                        if hasattr(bn, "node_names") and column_name in bn.node_names:
                            idx = bn.node_names.index(column_name)
                            dist_obj = probs[idx]
                            if hasattr(dist_obj, "parameters"):
                                params = dist_obj.parameters[0]
                                return {str(k): float(v) for k, v in params.items()}
                except Exception:
                    logger.exception("pomegranate predict_proba failed")

        except Exception:
            logger.exception("Unexpected error obtaining BN column distribution")
            return None

        return None

    @staticmethod
    def _extract_column_ref(expression: Any) -> Optional[Tuple[str, str]]:
        if isinstance(expression, ColumnRef) and expression.table:
            return expression.table, expression.column
        return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _normalize_value(value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _clamp_selectivity(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _bn_selectivity(self, table_name: str, column_name: str, literal_value: Any) -> Optional[float]:
        """Best-effort: use a loaded BN model to estimate P(column=literal).

        Returns a float in [0,1] or None if unavailable.
        """
        bn = self._bn_models.get(table_name)
        if bn is None:
            return None

        try:
            dist = self._bn_column_distribution(table_name, column_name)
            if not dist:
                return None

            literal_key = self._normalize_value(literal_value)
            if literal_key in dist:
                return self._clamp_selectivity(dist[literal_key])

            # Allow numeric/string aliasing for values that serialize differently.
            for key, prob in dist.items():
                if self._normalize_value(key) == literal_key:
                    return self._clamp_selectivity(prob)
            return None

        except Exception:
            logger.exception("Unexpected error in _bn_selectivity")
            return None

        return None

    def _bn_conditional_distribution(self, table_name: str, target_column: str, evidence: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Query BN for P(target_column | evidence) and return a dict mapping stringified values to probs.

        Caches results to avoid repeated expensive inference calls. Returns None if inference not available.
        """
        bn = self._bn_models.get(table_name)
        if bn is None:
            return None

        # normalize evidence keys/values to strings
        norm_evidence = {k: self._normalize_value(v) for k, v in (evidence or {}).items()}
        cache_key = (table_name, target_column, frozenset(norm_evidence.items()))
        cached = self._bn_infer_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            if hasattr(bn, "infer_machine") and bn.infer_machine is not None:
                try:
                    res = bn.infer_machine.query([target_column], evidence=norm_evidence)
                    if hasattr(res, "values"):
                        probs = res.values
                        mapping = None
                        if hasattr(bn, "domain") and target_column in bn.domain:
                            mapping = bn.domain[target_column]
                        dist = {}
                        if mapping is not None:
                            for i, p in enumerate(probs):
                                dist[str(mapping[i])] = float(p)
                        else:
                            for i, p in enumerate(probs):
                                dist[str(i)] = float(p)
                        self._bn_infer_cache[cache_key] = dist
                        return dist
                    elif isinstance(res, dict):
                        first = list(res.values())[0]
                        if hasattr(first, "values"):
                            probs = first.values
                            mapping = None
                            if hasattr(bn, "domain") and target_column in bn.domain:
                                mapping = bn.domain[target_column]
                            dist = {}
                            if mapping is not None:
                                for i, p in enumerate(probs):
                                    dist[str(mapping[i])] = float(p)
                            else:
                                for i, p in enumerate(probs):
                                    dist[str(i)] = float(p)
                            self._bn_infer_cache[cache_key] = dist
                            return dist
                except Exception:
                    logger.exception("BN conditional inference failed")

        except Exception:
            logger.exception("Unexpected error obtaining BN conditional distribution")
            return None

        return None
