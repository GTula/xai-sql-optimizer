"""
FASE: OPTIMIZATION
Optimizador de consultas (Query Optimizer).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Set
from sql.parser import SelectStatement, JoinClause
from catalog.catalog import Catalog
import itertools
import copy

from bayescard.query_optimizer import BayesCardQueryOptimizer

class QueryOptimizer(ABC):
    """Interfaz base para optimizadores de consultas."""
    
    def __init__(self, catalog: Catalog = None):
        self.catalog = catalog

    
    @abstractmethod
    def optimize(self, stmt: SelectStatement) -> SelectStatement:
        """
        Toma un AST de SELECT, optimiza el orden de los JOINs (y otras operaciones),
        y devuelve un AST modificado listo para ejecución.
        """
        pass


class BasicOptimizer(QueryOptimizer):
    """
    Optimizador Básico: 
    - Agrupa JOINs estrictamente de izquierda a derecha.
    - No requiere estadísticas, asume que el orden escrito es el deseado.
    """
    
    def optimize(self, stmt: SelectStatement) -> SelectStatement:
        # Al no hacer reordenamiento inteligente, simplemente devuelve 
        # el mismo statement. El motor ejecutará iterativamente la lista
        # de join_clauses de izquierda a derecha en el orden original.
        return stmt


class SelingerOptimizer(QueryOptimizer):
    """
    Optimizador Selinger (System R):
    - Usa Programación Dinámica para encontrar el árbol Left-Deep más barato.
    - Utiliza estadísticas del Catalog para estimar el costo.
    """
    
    def optimize(self, stmt: SelectStatement) -> SelectStatement:
        if not stmt.join_clauses:
            return stmt
            
        tables = [stmt.from_table] + [jc.table for jc in stmt.join_clauses]
        original_conditions = {jc.table: jc.condition for jc in stmt.join_clauses}
        
        # dp[frozenset(tables)] = (cost, list_of_tables_in_order)
        dp: Dict[frozenset, Tuple[float, List[str]]] = {}
        
        # Inicialización tamaño 1
        for t in tables:
            dp[frozenset([t])] = (self._get_table_cost(t), [t])
            
        # PD tamaño 2 a N
        for size in range(2, len(tables) + 1):
            for subset in itertools.combinations(tables, size):
                subset_fs = frozenset(subset)
                best_cost = float('inf')
                best_order = None
                
                for t in subset:
                    prev_subset = frozenset(s for s in subset if s != t)
                    prev_cost, prev_order = dp[prev_subset]
                    
                    t_cost = self._get_table_cost(t)
                    
                    # Buscar la condicion que conecta a t con prev_subset
                    cond = None
                    if t in original_conditions:
                        cond = original_conditions[t]
                    else:
                        for oc_t, oc_cond in original_conditions.items():
                            # Simplificación: tomar una condición
                            cond = oc_cond
                            break
                            
                    selectivity = self._estimate_selectivity(cond)
                    
                    current_cost = prev_cost * t_cost * selectivity
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_order = prev_order + [t]
                        
                dp[subset_fs] = (best_cost, best_order)
                
        _, final_order = dp[frozenset(tables)]
        
        # Reconstruir AST
        new_stmt = copy.deepcopy(stmt)
        new_stmt.from_table = final_order[0]
        
        new_joins = []
        used_conditions_ids = set()
        
        for t in final_order[1:]:
            # Asignar condición al JoinClause
            cond = None
            if t in original_conditions and id(original_conditions[t]) not in used_conditions_ids:
                cond = original_conditions[t]
                used_conditions_ids.add(id(cond))
            else:
                # Si no era la tabla original que traía la condición, buscamos una libre
                for oc_t, oc_cond in original_conditions.items():
                    if id(oc_cond) not in used_conditions_ids:
                        cond = oc_cond
                        used_conditions_ids.add(id(cond))
                        break
            new_joins.append(JoinClause(t, cond))
            
        new_stmt.join_clauses = new_joins
        return new_stmt
        
    def _get_table_cost(self, table_name: str) -> float:
        if not self.catalog:
            return 1000.0
            
        try:
            rows = self.catalog._read_rows(self.catalog.statistics_path)
            for r in rows:
                if r["table_name"] == table_name:
                    return float(r["row_count"])
        except Exception:
            pass
        return 1000.0

    def _estimate_selectivity(self, condition) -> float:
        if not self.catalog or not condition:
            return 0.1
            
        if condition.__class__.__name__ == "BinaryOp" and condition.operator == "=":
            def extract_col(expr):
                if expr.__class__.__name__ == "ColumnRef":
                    return expr.table, expr.column
                return None, None
                
            t1, c1 = extract_col(condition.left)
            t2, c2 = extract_col(condition.right)
            
            if t1 and t2:
                ndv1 = self.catalog.get_column_ndv(t1, c1)
                ndv2 = self.catalog.get_column_ndv(t2, c2)
                max_ndv = max(ndv1, ndv2)
                if max_ndv > 0:
                    return 1.0 / max_ndv
                    
        return 0.1


class BayesOptimizer(QueryOptimizer):
    """Optimizador BayesCard: encapsula la logica de ordenamiento de JOINs."""

    def __init__(self, catalog: Catalog = None):
        super().__init__(catalog)
        self._delegate = BayesCardQueryOptimizer(catalog)

    def optimize(self, stmt: SelectStatement) -> SelectStatement:
        return self._delegate.optimize(stmt)


def get_optimizer(optimizer_type: str = "basic", catalog: Catalog = None) -> QueryOptimizer:
    """Fábrica de optimizadores."""
    if optimizer_type.lower() == "basic":
        return BasicOptimizer(catalog)
    elif optimizer_type.lower() == "selinger":
        return SelingerOptimizer(catalog)
    elif optimizer_type.lower() in {"bayes", "bayescard"}:
        return BayesOptimizer(catalog)
    else:
        raise ValueError(f"Optimizador '{optimizer_type}' no soportado.")
