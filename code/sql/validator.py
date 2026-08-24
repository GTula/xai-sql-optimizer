"""
FASE 4: SQL
Validador: verificar que la query es semánticamente válida
"""

from sql.parser import (
    ASTNode, SelectStatement, InsertStatement, UpdateStatement,
    DeleteStatement, CreateTableStatement, CreateIndexStatement, AnalyseStatement,
    Expression, ColumnRef, BinaryOp, Literal, UnaryOp
)
from catalog.catalog import Catalog


class Validator:
    """
    Validador semántico de queries SQL.
    Verifica que:
    - Las tablas mencionadas existen
    - Las columnas existen en sus tablas
    - Los tipos son compatibles
    """
    
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
    
    def validate(self, ast: ASTNode) -> bool:
        """Valida un AST contra el catálogo"""
        if isinstance(ast, SelectStatement):
            return self._validate_select(ast)
        elif isinstance(ast, InsertStatement):
            return self._validate_insert(ast)
        elif isinstance(ast, UpdateStatement):
            return self._validate_update(ast)
        elif isinstance(ast, DeleteStatement):
            return self._validate_delete(ast)
        elif isinstance(ast, CreateTableStatement):
            return self._validate_create_table(ast)
        elif isinstance(ast, CreateIndexStatement):
            return self._validate_create_index(ast)
        elif isinstance(ast, AnalyseStatement):
            return self._validate_analyse(ast)
        else:
            raise ValueError(f"AST desconocido: {type(ast)}")
    
    def _validate_select(self, stmt: SelectStatement) -> bool:
        """Valida SELECT"""
        tables = []
        if stmt.from_table:
            table = self.catalog.get_table(stmt.from_table)
            if not table:
                raise ValueError(f"Tabla '{stmt.from_table}' no existe")
            tables.append(table)
            
            if stmt.join_clauses:
                for jc in stmt.join_clauses:
                    join_table = self.catalog.get_table(jc.table)
                    if not join_table:
                        raise ValueError(f"Tabla '{jc.table}' no existe (en JOIN)")
                    tables.append(join_table)
                    
                    self._validate_expression(jc.condition, tables)
            
            # Validar columnas en WHERE
            if stmt.where_clause:
                self._validate_expression(stmt.where_clause, tables)
        
        # Validar columnas en SELECT
        for col_expr in stmt.columns:
            if isinstance(col_expr, ColumnRef):
                if col_expr.column != '*' and stmt.from_table:
                    found = False
                    for t in tables:
                        if col_expr.table and col_expr.table != t.name:
                            continue
                        if t.get_column(col_expr.column):
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Columna '{col_expr.column}' no existe en las tablas consultadas")
        
        return True
    
    def _validate_insert(self, stmt: InsertStatement) -> bool:
        """Valida INSERT"""
        table = self.catalog.get_table(stmt.table)
        if not table:
            raise ValueError(f"Tabla '{stmt.table}' no existe")
        
        # Si se especifican columnas, verificar que existen
        if stmt.columns:
            for col in stmt.columns:
                if not table.get_column(col):
                    raise ValueError(f"Columna '{col}' no existe en tabla '{stmt.table}'")
        
        return True
    
    def _validate_update(self, stmt: UpdateStatement) -> bool:
        """Valida UPDATE"""
        table = self.catalog.get_table(stmt.table)
        if not table:
            raise ValueError(f"Tabla '{stmt.table}' no existe")
        
        # Verificar columnas que se actualizan
        for col in stmt.assignments.keys():
            if not table.get_column(col):
                raise ValueError(f"Columna '{col}' no existe en tabla '{stmt.table}'")
        
        # Validar WHERE
        if stmt.where_clause:
            self._validate_expression(stmt.where_clause, [table])
        
        return True
    
    def _validate_delete(self, stmt: DeleteStatement) -> bool:
        """Valida DELETE"""
        table = self.catalog.get_table(stmt.table)
        if not table:
            raise ValueError(f"Tabla '{stmt.table}' no existe")
        
        # Validar WHERE
        if stmt.where_clause:
            self._validate_expression(stmt.where_clause, [table])
        
        return True
    
    def _validate_create_table(self, stmt: CreateTableStatement) -> bool:
        """Valida CREATE TABLE"""
        if self.catalog.get_table(stmt.table_name):
            raise ValueError(f"Tabla '{stmt.table_name}' ya existe")
        
        # Validar tipos
        from config import SUPPORTED_TYPES
        for col_name, col_type in stmt.columns:
            if col_type not in SUPPORTED_TYPES:
                raise ValueError(f"Tipo no soportado: {col_type}")
        
        return True
    
    def _validate_create_index(self, stmt: CreateIndexStatement) -> bool:
        """Valida CREATE INDEX"""
        table = self.catalog.get_table(stmt.table_name)
        if not table:
            raise ValueError(f"Tabla '{stmt.table_name}' no existe")
        
        if not table.get_column(stmt.column_name):
            raise ValueError(f"Columna '{stmt.column_name}' no existe en tabla '{stmt.table_name}'")
        
        if self.catalog.get_index(stmt.index_name):
            raise ValueError(f"Índice '{stmt.index_name}' ya existe")
        
        return True

    def _validate_analyse(self, stmt: AnalyseStatement) -> bool:
        """Valida ANALYSE [tabla]"""
        if stmt.table_name and not self.catalog.get_table(stmt.table_name):
            raise ValueError(f"Tabla '{stmt.table_name}' no existe")
        return True
    
    def _validate_expression(self, expr: Expression, tables: list) -> None:
        """Valida recursivamente una expresión"""
        if isinstance(expr, ColumnRef):
            if expr.column != '*':
                found = False
                for t in tables:
                    if expr.table and expr.table != t.name:
                        continue
                    if t.get_column(expr.column):
                        found = True
                        break
                if not found:
                    table_names = [t.name for t in tables]
                    raise ValueError(f"Columna '{expr.column}' no existe en las tablas {table_names}")
        
        elif isinstance(expr, BinaryOp):
            self._validate_expression(expr.left, tables)
            self._validate_expression(expr.right, tables)
        
        elif isinstance(expr, UnaryOp):
            self._validate_expression(expr.operand, tables)
        
        # Literal no necesita validación
