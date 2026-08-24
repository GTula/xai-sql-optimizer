"""Optimizador BayesCard autocontenido para reordenar JOINs left-deep."""

from __future__ import annotations

import copy
import itertools
from typing import Dict, List, Optional, Tuple

from .cost_model import BayesCardCostModel
from catalog.catalog import Catalog
from sql.parser import BinaryOp, JoinClause, SelectStatement, UnaryOp


class BayesCardQueryOptimizer:
    """Reordena JOINs usando el estimador BayesCard local."""

    def __init__(self, catalog: Optional[Catalog] = None, cost_model: Optional[BayesCardCostModel] = None):
        self.catalog = catalog
        self.cost_model = cost_model or BayesCardCostModel(catalog)

    def optimize(self, stmt: SelectStatement) -> SelectStatement:
        if not stmt.join_clauses or not stmt.from_table:
            return stmt

        tables = [stmt.from_table] + [join_clause.table for join_clause in stmt.join_clauses]
        if len(set(tables)) != len(tables):
            return stmt

        join_condition_by_table = {join_clause.table: join_clause.condition for join_clause in stmt.join_clauses}
        local_predicate_by_table = self._extract_local_predicates(stmt.where_clause, tables)
        base_table = stmt.from_table
        base_cardinality = self.cost_model.table_cardinality(base_table) * self.cost_model.table_filter_selectivity(
            base_table, local_predicate_by_table.get(base_table)
        )

        dp: Dict[frozenset, Tuple[float, List[str]]] = {frozenset([base_table]): (base_cardinality, [base_table])}

        for size in range(2, len(tables) + 1):
            for subset in itertools.combinations(tables, size):
                subset_key = frozenset(subset)
                if base_table not in subset_key:
                    continue

                best_cost = float("inf")
                best_order: Optional[List[str]] = None

                for table_name in subset:
                    if table_name == base_table:
                        continue

                    remaining = frozenset(candidate for candidate in subset if candidate != table_name)
                    if remaining not in dp:
                        continue

                    if not self._condition_is_compatible(join_condition_by_table.get(table_name), remaining | {table_name}):
                        continue

                    prev_cost, prev_order = dp[remaining]
                    table_cardinality = self.cost_model.table_cardinality(table_name) * self.cost_model.table_filter_selectivity(
                        table_name, local_predicate_by_table.get(table_name)
                    )
                    join_condition = join_condition_by_table.get(table_name)
                    join_selectivity = self.cost_model.join_selectivity(join_condition)
                    current_cost = prev_cost * table_cardinality * join_selectivity

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_order = prev_order + [table_name]

                if best_order is not None:
                    dp[subset_key] = (best_cost, best_order)

        final_entry = dp.get(frozenset(tables))
        if final_entry is None:
            return stmt

        _, final_order = final_entry
        reordered = copy.deepcopy(stmt)
        reordered.from_table = final_order[0]

        reordered.join_clauses = [JoinClause(table_name, join_condition_by_table[table_name]) for table_name in final_order[1:]]
        return reordered

    def _condition_is_compatible(self, condition, visible_tables: set) -> bool:
        if condition is None:
            return True

        referenced_tables = self._tables_in_expression(condition)
        if not referenced_tables:
            return True

        return referenced_tables.issubset(visible_tables)

    def _extract_local_predicates(self, where_clause, tables: List[str]) -> Dict[str, object]:
        if where_clause is None:
            return {}

        predicates: Dict[str, object] = {}
        for table_name in tables:
            table_predicate = self._predicate_for_table(where_clause, table_name)
            if table_predicate is not None:
                predicates[table_name] = table_predicate
        return predicates

    def _predicate_for_table(self, expression, table_name: str):
        if expression is None:
            return None

        if isinstance(expression, UnaryOp):
            operand = self._predicate_for_table(expression.operand, table_name)
            if operand is None:
                return None
            return UnaryOp(expression.operator, operand)

        if not isinstance(expression, BinaryOp):
            return None

        if expression.operator.upper() in {"AND", "OR"}:
            left = self._predicate_for_table(expression.left, table_name)
            right = self._predicate_for_table(expression.right, table_name)
            if left is None and right is None:
                return None
            if left is None:
                return right
            if right is None:
                return left
            return BinaryOp(left, expression.operator, right)

        tables_in_expression = self._tables_in_expression(expression)
        if tables_in_expression and tables_in_expression.issubset({table_name}):
            return expression

        if not tables_in_expression:
            return expression

        return None

    def _tables_in_expression(self, expression) -> set:
        if expression is None:
            return set()

        if hasattr(expression, "table") and getattr(expression, "table", None):
            return {expression.table}

        if isinstance(expression, UnaryOp):
            return self._tables_in_expression(expression.operand)

        if isinstance(expression, BinaryOp):
            return self._tables_in_expression(expression.left) | self._tables_in_expression(expression.right)

        return set()