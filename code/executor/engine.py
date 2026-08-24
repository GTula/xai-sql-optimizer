"""
FASE 5: EXECUTOR
Motor de ejecucion: convierte el AST en un plan y lo ejecuta.
Implementacion simple sobre archivos CSV.
"""

from typing import List, Tuple, Any, Dict, Optional
import csv
import os

from sql.parser import (
    SelectStatement,
    InsertStatement,
    UpdateStatement,
    DeleteStatement,
    CreateTableStatement,
    CreateIndexStatement,
    AnalyseStatement,
    ColumnRef,
    BinaryOp,
    Literal,
    UnaryOp,
)
from sql.tokenizer import Tokenizer
from sql.parser import Parser
from sql.validator import Validator
from sql.optimizer import get_optimizer
from catalog.catalog import Catalog, Column, Table
from index.btree import BTree


class ExecutionEngine:
    """
    Motor de ejecucion: ejecuta queries SQL despues de parsear y validar.

    Pipeline:
    1. Tokenizar el SQL
    2. Parsear a AST
    3. Validar semanticamente
    4. Ejecutar
    """

    def __init__(self, catalog: Catalog, db_dir: str, optimizer_type: str = "basic"):
        self.catalog = catalog
        self.db_dir = db_dir
        self.validator = Validator(catalog)
        self.optimizer = get_optimizer(optimizer_type, self.catalog)

    def execute(self, sql: str) -> List[Tuple]:
        """Ejecuta una query SQL y devuelve el resultado."""
        tokenizer = Tokenizer(sql)
        tokens = tokenizer.tokenize()

        parser = Parser(tokens)
        ast = parser.parse_statement()

        self.validator.validate(ast)

        if isinstance(ast, SelectStatement):
            ast = self.optimizer.optimize(ast)
            return self._execute_select(ast)
        if isinstance(ast, InsertStatement):
            return self._execute_insert(ast)
        if isinstance(ast, UpdateStatement):
            return self._execute_update(ast)
        if isinstance(ast, DeleteStatement):
            return self._execute_delete(ast)
        if isinstance(ast, CreateTableStatement):
            return self._execute_create_table(ast)
        if isinstance(ast, CreateIndexStatement):
            return self._execute_create_index(ast)
        if isinstance(ast, AnalyseStatement):
            return self._execute_analyse(ast)

        raise ValueError(f"Tipo de statement desconocido: {type(ast)}")

    def execute_with_trace(self, sql: str) -> "Tuple[List[Tuple], Dict[str, Any]]":
        """
        Ejecuta un SELECT e instrumenta la ejecución para recolectar métricas
        reales por paso de JOIN.

        Returns:
            (results, raw_trace) donde raw_trace contiene:
                base_rows  : int — filas leídas de la tabla base (FROM)
                steps      : list[dict] — por cada JOIN:
                               step, input_rows, table_rows, output_rows
                final_rows : int — filas tras aplicar WHERE
                plan_order : list[str] — tablas en el orden de ejecución
        """
        tokenizer = Tokenizer(sql)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse_statement()
        if not isinstance(ast, SelectStatement):
            raise ValueError("execute_with_trace solo soporta SELECT")
        ast = self.optimizer.optimize(ast)
        # La validación semántica se omite intencionalmente: execute_with_trace
        # se llama sobre queries ya validados por el flujo principal.
        trace: Dict[str, Any] = {
            "base_rows": 0,
            "steps": [],
            "final_rows": 0,
            "plan_order": [],
        }
        results = self._execute_select(ast, _trace=trace)
        return results, trace

    def execute_plan_with_trace(self, stmt: SelectStatement) -> "Tuple[List[Tuple], Dict[str, Any]]":
        """
        Ejecuta un SELECT ya planificado e instrumenta la ejecucion sin volver
        a optimizarlo. Sirve para comparar uno a uno un plan candidato estimado
        contra el runtime real de ese mismo orden de JOIN.
        """
        if not isinstance(stmt, SelectStatement):
            raise ValueError("execute_plan_with_trace solo soporta SelectStatement")

        trace: Dict[str, Any] = {
            "base_rows": 0,
            "steps": [],
            "final_rows": 0,
            "plan_order": [],
        }
        results = self._execute_select(stmt, _trace=trace)
        return results, trace

    def _execute_select(self, stmt: SelectStatement, _trace: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """Ejecuta SELECT. Si se pasa _trace, lo llena con métricas reales de ejecución."""
        if not stmt.from_table:
            return [self._evaluate_select_list(stmt.columns, {})]

        table = self.catalog.get_table(stmt.from_table)
        if not table:
            raise ValueError(f"Tabla '{stmt.from_table}' no existe")

        if not os.path.exists(table.table_file):
            return []

        rows = self._read_table_rows(table)
        tables = [table]

        if _trace is not None:
            _trace["base_rows"] = len(rows)
            _trace["plan_order"] = [stmt.from_table]

        # Nested Loop Join para múltiples JOINs
        if stmt.join_clauses:
            for step_no, jc in enumerate(stmt.join_clauses, start=1):
                join_table = self.catalog.get_table(jc.table)
                if join_table and os.path.exists(join_table.table_file):
                    tables.append(join_table)
                    join_rows = self._read_table_rows(join_table)
                    input_rows = len(rows)
                    combined_rows = []
                    for l_row in rows:
                        for r_row in join_rows:
                            combined = {**l_row, **r_row}
                            if bool(self._evaluate_expression(jc.condition, combined, tables)):
                                combined_rows.append(combined)
                    rows = combined_rows
                    if _trace is not None:
                        _trace["steps"].append({
                            "step": step_no,
                            "input_rows": input_rows,
                            "table_rows": len(join_rows),
                            "output_rows": len(rows),
                        })
                        _trace["plan_order"].append(jc.table)
                else:
                    input_rows = len(rows)
                    rows = []  # Si la tabla de join no existe o está vacía, el inner join es vacío
                    if _trace is not None:
                        _trace["steps"].append({
                            "step": step_no,
                            "input_rows": input_rows,
                            "table_rows": 0,
                            "output_rows": 0,
                        })
                        _trace["plan_order"].append(jc.table)

        if stmt.where_clause:
            rows = [row for row in rows if bool(self._evaluate_expression(stmt.where_clause, row, tables))]

        if _trace is not None:
            _trace["final_rows"] = len(rows)

        return [self._project_row(stmt.columns, row, tables) for row in rows]

    def _execute_insert(self, stmt: InsertStatement) -> List[Tuple]:
        """Ejecuta INSERT."""
        table = self.catalog.get_table(stmt.table)
        if not table:
            raise ValueError(f"Tabla '{stmt.table}' no existe")

        os.makedirs(os.path.dirname(table.table_file), exist_ok=True)

        if stmt.values:
            with open(table.table_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)

                for value_list in stmt.values:
                    row_values = self._build_insert_row(stmt, value_list, table)
                    writer.writerow([self._to_csv_value(v) for v in row_values])

        return []

    def _execute_update(self, stmt: UpdateStatement) -> List[Tuple]:
        """Ejecuta UPDATE (simplificado)."""
        raise NotImplementedError("UPDATE no esta completamente implementado")

    def _execute_delete(self, stmt: DeleteStatement) -> List[Tuple]:
        """Ejecuta DELETE (simplificado)."""
        raise NotImplementedError("DELETE no esta completamente implementado")

    def _execute_create_table(self, stmt: CreateTableStatement) -> List[Tuple]:
        """Ejecuta CREATE TABLE."""
        table_name = stmt.table_name
        columns = [Column(col_name, col_type) for col_name, col_type in stmt.columns]

        os.makedirs(self.db_dir, exist_ok=True)
        table_file = os.path.join(self.db_dir, f"{table_name}.csv")

        if os.path.exists(table_file):
            raise ValueError(f"Archivo de tabla ya existe: {table_file}")

        # Archivo CSV de tabla con header de columnas.
        with open(table_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([col.name for col in columns])

        self.catalog.create_table(table_name, columns, table_file)
        return []

    def _execute_create_index(self, stmt: CreateIndexStatement) -> List[Tuple]:
        """Ejecuta CREATE INDEX (registro en catalogo + archivo de indice)."""
        table = self.catalog.get_table(stmt.table_name)
        if not table:
            raise ValueError(f"Tabla '{stmt.table_name}' no existe")

        if table.get_column(stmt.column_name) is None:
            raise ValueError(
                f"Columna '{stmt.column_name}' no existe en tabla '{stmt.table_name}'"
            )

        index_file = os.path.join(self.db_dir, f"{stmt.index_name}.idx")
        BTree(index_file)

        self.catalog.create_index(
            stmt.index_name,
            stmt.table_name,
            stmt.column_name,
            index_file,
        )

        return []

    def _execute_analyse(self, stmt: AnalyseStatement) -> List[Tuple]:
        """Actualiza estadisticas del catalogo (Filas y NDV) bajo demanda."""
        if stmt.table_name:
            table = self.catalog.get_table(stmt.table_name)
            if not table:
                raise ValueError(f"Tabla '{stmt.table_name}' no existe")

            row_count = self._analyze_table_stats(table)
            return [(table.name, row_count)]

        updated = []
        for table in self.catalog.tables.values():
            row_count = self._analyze_table_stats(table)
            updated.append((table.name, row_count))

        return updated

    def _analyze_table_stats(self, table: Table) -> int:
        """Lee la tabla, cuenta filas y calcula el NDV de cada columna."""
        row_count = 0
        col_sets = {col.name: set() for col in table.columns}
        
        try:
            with open(table.table_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_count += 1
                    for col in table.columns:
                        col_sets[col.name].add(row.get(col.name, ""))
        except FileNotFoundError:
            pass
            
        self.catalog.update_table_row_count(table.name, row_count)
        
        for col_name, unique_vals in col_sets.items():
            self.catalog.update_column_ndv(table.name, col_name, len(unique_vals) if len(unique_vals) > 0 else 1)
            
        return row_count

    def _read_table_rows(self, table: Table) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(table.table_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                row: Dict[str, Any] = {}
                for col in table.columns:
                    val = self._from_csv_value(raw_row.get(col.name, ""), col.type)
                    row[col.name] = val
                    row[f"{table.name}.{col.name}"] = val
                rows.append(row)
        return rows

    def _build_insert_row(self, stmt: InsertStatement, value_list: list, table: Table) -> List[Any]:
        values_by_column: Dict[str, Any] = {}

        if stmt.columns:
            for col_name, value_expr in zip(stmt.columns, value_list):
                values_by_column[col_name] = self._evaluate_expression(value_expr, {}, [table])
        else:
            for col, value_expr in zip(table.columns, value_list):
                values_by_column[col.name] = self._evaluate_expression(value_expr, {}, [table])

        row_values: List[Any] = []
        for col in table.columns:
            if col.name not in values_by_column:
                row_values.append(None)
            else:
                row_values.append(self._coerce_value(values_by_column[col.name], col.type))

        return row_values

    def _project_row(self, select_columns: List[Any], row: Dict[str, Any], tables: List[Table]) -> Tuple:
        if len(select_columns) == 1 and isinstance(select_columns[0], ColumnRef) and select_columns[0].column == "*":
            result = []
            for t in tables:
                for col in t.columns:
                    result.append(row[f"{t.name}.{col.name}"])
            return tuple(result)

        out = []
        for expr in select_columns:
            if isinstance(expr, ColumnRef):
                if expr.table:
                    out.append(row.get(f"{expr.table}.{expr.column}"))
                else:
                    out.append(row.get(expr.column))
            else:
                out.append(self._evaluate_expression(expr, row, tables))
        return tuple(out)

    def _evaluate_expression(self, expr, row: Dict[str, Any], tables: List[Table]):
        if isinstance(expr, Literal):
            return expr.value

        if isinstance(expr, ColumnRef):
            if expr.column == "*":
                raise ValueError("La columna '*' no es valida en este contexto")
            
            if expr.table:
                key = f"{expr.table}.{expr.column}"
                if key in row:
                    return row.get(key)
            
            if expr.column in row:
                return row.get(expr.column)
            
            # Check if column exists in any table
            for t in tables:
                if t.get_column(expr.column):
                    return None
            raise ValueError(f"Columna '{expr.column}' no existe")

        if isinstance(expr, BinaryOp):
            left = self._evaluate_expression(expr.left, row, tables)
            right = self._evaluate_expression(expr.right, row, tables)

            if expr.operator == "=":
                return left == right
            if expr.operator == "<>":
                return left != right
            if expr.operator == "<":
                return left < right
            if expr.operator == "<=":
                return left <= right
            if expr.operator == ">":
                return left > right
            if expr.operator == ">=":
                return left >= right
            if expr.operator == "+":
                return left + right
            if expr.operator == "-":
                return left - right
            if expr.operator == "*":
                return left * right
            if expr.operator == "/":
                return left / right if right != 0 else None
            if expr.operator == "AND":
                return bool(left) and bool(right)
            if expr.operator == "OR":
                return bool(left) or bool(right)
            raise ValueError(f"Operador no soportado: {expr.operator}")

        if isinstance(expr, UnaryOp):
            operand = self._evaluate_expression(expr.operand, row, tables)
            if expr.operator == "NOT":
                return not operand
            raise ValueError(f"Operador unario no soportado: {expr.operator}")

        return None

    def _evaluate_select_list(self, columns: List[Any], row: Dict[str, Any]):
        result = []
        for col_expr in columns:
            if isinstance(col_expr, ColumnRef):
                raise ValueError(
                    "SELECT sin FROM solo soporta literales/expresiones; "
                    "no columnas ni SELECT *"
                )
            result.append(self._evaluate_expression(col_expr, row, []))
        return tuple(result)

    def _coerce_value(self, value: Any, column_type: str) -> Any:
        if value is None:
            return None
        if column_type == "INTEGER":
            return int(value)
        if column_type == "FLOAT":
            return float(value)
        if column_type == "BOOLEAN":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"true", "1", "yes"}
        if column_type in {"VARCHAR", "DATE"}:
            return str(value)
        raise ValueError(f"Tipo no soportado: {column_type}")

    def _to_csv_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _from_csv_value(self, value: str, column_type: str) -> Any:
        if value is None or value == "":
            return None
        if column_type == "INTEGER":
            return int(value)
        if column_type == "FLOAT":
            return float(value)
        if column_type == "BOOLEAN":
            return value.strip().lower() in {"true", "1", "yes"}
        if column_type in {"VARCHAR", "DATE"}:
            return value
        return value

    def _count_data_rows(self, table_file: str) -> int:
        with open(table_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # header
            return sum(1 for _row in reader)
