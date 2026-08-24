"""
FASE 4: SQL
Parser: convierte tokens en un árbol sintáctico (AST)
"""

from dataclasses import dataclass
from typing import List, Optional
from sql.tokenizer import Token, TokenType, Tokenizer


# ============ AST Nodes ============

@dataclass
class ASTNode:
    """Nodo base del árbol sintáctico"""
    pass


@dataclass
class Expression(ASTNode):
    """Expresión genérica"""
    pass


@dataclass
class Literal(Expression):
    """Literal: número, string"""
    value: any
    type: str  # 'number', 'string'


@dataclass
class ColumnRef(Expression):
    """Referencia a columna: columna o tabla.columna"""
    column: str
    table: Optional[str] = None


@dataclass
class BinaryOp(Expression):
    """Operación binaria: a = b, a > 5, etc."""
    left: Expression
    operator: str  # '=', '<', '>', etc.
    right: Expression


@dataclass
class UnaryOp(Expression):
    """Operación unaria: NOT x"""
    operator: str
    operand: Expression


@dataclass
class FunctionCall(Expression):
    """Llamada a función: COUNT(*), SUM(salary)"""
    name: str
    args: List[Expression]


# ============ DML/DDL Statements ============

@dataclass
class JoinClause(ASTNode):
    """JOIN tabla ON condicion"""
    table: str
    condition: Expression


@dataclass
class SelectStatement(ASTNode):
    """SELECT"""
    columns: List[Expression]
    from_table: Optional[str] = None
    where_clause: Optional[Expression] = None
    join_clauses: Optional[List[JoinClause]] = None


@dataclass
class InsertStatement(ASTNode):
    """INSERT INTO tabla VALUES (...)"""
    table: str
    columns: Optional[List[str]] = None
    values: Optional[List[List[Expression]]] = None


@dataclass
class UpdateStatement(ASTNode):
    """UPDATE tabla SET col=val WHERE ..."""
    table: str
    assignments: dict  # {column: value}
    where_clause: Optional[Expression] = None


@dataclass
class DeleteStatement(ASTNode):
    """DELETE FROM tabla WHERE ..."""
    table: str
    where_clause: Optional[Expression] = None


@dataclass
class CreateTableStatement(ASTNode):
    """CREATE TABLE"""
    table_name: str
    columns: List[tuple]  # [(name, type), ...]


@dataclass
class CreateIndexStatement(ASTNode):
    """CREATE INDEX"""
    index_name: str
    table_name: str
    column_name: str


@dataclass
class AnalyseStatement(ASTNode):
    """ANALYSE [tabla]"""
    table_name: Optional[str] = None


# ============ Parser ============

class Parser:
    """
    Parser recursivo descendente para SQL.
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    @staticmethod
    def parse(sql: str) -> ASTNode:
        """Convierte SQL en AST"""
        tokenizer = Tokenizer(sql)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        return parser.parse_statement()
    
    def parse_statement(self) -> ASTNode:
        """Parsea un statement SQL"""
        token = self._current()
        
        if token.type == TokenType.SELECT:
            return self._parse_select()
        elif token.type == TokenType.INSERT:
            return self._parse_insert()
        elif token.type == TokenType.UPDATE:
            return self._parse_update()
        elif token.type == TokenType.DELETE:
            return self._parse_delete()
        elif token.type == TokenType.CREATE:
            return self._parse_create()
        elif token.type == TokenType.ANALYSE:
            return self._parse_analyse()
        else:
            raise SyntaxError(f"Statement inesperado: {token}")

    def _parse_analyse(self) -> AnalyseStatement:
        self._match(TokenType.ANALYSE)

        table_name = None
        if self._current().type == TokenType.IDENTIFIER:
            table_name = self._match(TokenType.IDENTIFIER).value

        return AnalyseStatement(table_name=table_name)
    
    def _parse_select(self) -> SelectStatement:
        self._match(TokenType.SELECT)
        
        # Parsear columnas
        columns = []
        columns.append(self._parse_expression())
        while self._match_type(TokenType.COMMA):
            columns.append(self._parse_expression())
        
        # Parsear FROM
        from_table = None
        if self._match_type(TokenType.FROM):
            from_table = self._match(TokenType.IDENTIFIER).value
        
        # Parsear JOINs
        join_clauses = []
        while self._match_type(TokenType.JOIN):
            join_table = self._match(TokenType.IDENTIFIER).value
            self._match(TokenType.ON)
            join_condition = self._parse_expression()
            join_clauses.append(JoinClause(join_table, join_condition))
        
        # Parsear WHERE
        where_clause = None
        if self._match_type(TokenType.WHERE):
            where_clause = self._parse_expression()
        
        return SelectStatement(columns, from_table, where_clause, join_clauses if join_clauses else None)
    
    def _parse_insert(self) -> InsertStatement:
        self._match(TokenType.INSERT)
        self._match(TokenType.INTO)
        table = self._match(TokenType.IDENTIFIER).value
        
        # Parsear columnas (opcional)
        columns = None
        if self._current().type == TokenType.LPAREN:
            self._advance()
            columns = [self._match(TokenType.IDENTIFIER).value]
            while self._match_type(TokenType.COMMA):
                columns.append(self._match(TokenType.IDENTIFIER).value)
            self._match(TokenType.RPAREN)
        
        # Parsear VALUES
        self._match(TokenType.VALUES)
        values = []
        values.append(self._parse_value_list())
        
        while self._match_type(TokenType.COMMA):
            values.append(self._parse_value_list())
        
        return InsertStatement(table, columns, values)
    
    def _parse_value_list(self) -> List[Expression]:
        """Parsea (val1, val2, ...)"""
        self._match(TokenType.LPAREN)
        values = [self._parse_expression()]
        while self._match_type(TokenType.COMMA):
            values.append(self._parse_expression())
        self._match(TokenType.RPAREN)
        return values
    
    def _parse_update(self) -> UpdateStatement:
        self._match(TokenType.UPDATE)
        table = self._match(TokenType.IDENTIFIER).value
        self._match(TokenType.SET)
        
        assignments = {}
        col = self._match(TokenType.IDENTIFIER).value
        self._match(TokenType.EQUAL)
        assignments[col] = self._parse_expression()
        
        while self._match_type(TokenType.COMMA):
            col = self._match(TokenType.IDENTIFIER).value
            self._match(TokenType.EQUAL)
            assignments[col] = self._parse_expression()
        
        where_clause = None
        if self._match_type(TokenType.WHERE):
            where_clause = self._parse_expression()
        
        return UpdateStatement(table, assignments, where_clause)
    
    def _parse_delete(self) -> DeleteStatement:
        self._match(TokenType.DELETE)
        self._match(TokenType.FROM)
        table = self._match(TokenType.IDENTIFIER).value
        
        where_clause = None
        if self._match_type(TokenType.WHERE):
            where_clause = self._parse_expression()
        
        return DeleteStatement(table, where_clause)
    
    def _parse_create(self) -> ASTNode:
        self._match(TokenType.CREATE)
        
        if self._match_type(TokenType.TABLE):
            return self._parse_create_table()
        elif self._match_type(TokenType.INDEX):
            return self._parse_create_index()
        else:
            raise SyntaxError(f"CREATE inesperado: {self._current()}")
    
    def _parse_create_table(self) -> CreateTableStatement:
        table_name = self._match(TokenType.IDENTIFIER).value
        self._match(TokenType.LPAREN)
        
        columns = []
        col_name = self._match(TokenType.IDENTIFIER).value
        col_type = self._parse_column_type()
        columns.append((col_name, col_type))
        
        while self._match_type(TokenType.COMMA):
            col_name = self._match(TokenType.IDENTIFIER).value
            col_type = self._parse_column_type()
            columns.append((col_name, col_type))
        
        self._match(TokenType.RPAREN)
        return CreateTableStatement(table_name, columns)

    def _parse_column_type(self) -> str:
        """Parsea tipos de columna, incluyendo VARCHAR(n)."""
        token = self._current()
        valid_type_tokens = {
            TokenType.INTEGER,
            TokenType.VARCHAR,
            TokenType.FLOAT,
            TokenType.BOOLEAN,
            TokenType.DATE,
        }

        if token.type not in valid_type_tokens:
            raise SyntaxError(f"Tipo de columna inesperado: {token}")

        type_name = token.value.upper()
        self._advance()

        # Soporte para tipo con precision/longitud, ejemplo: VARCHAR(50)
        if type_name == 'VARCHAR' and self._match_type(TokenType.LPAREN):
            self._match(TokenType.NUMBER)
            self._match(TokenType.RPAREN)

        return type_name
    
    def _parse_create_index(self) -> CreateIndexStatement:
        index_name = self._match(TokenType.IDENTIFIER).value
        self._match(TokenType.ON)
        table_name = self._match(TokenType.IDENTIFIER).value
        self._match(TokenType.LPAREN)
        column_name = self._match(TokenType.IDENTIFIER).value
        self._match(TokenType.RPAREN)
        
        return CreateIndexStatement(index_name, table_name, column_name)
    
    def _parse_expression(self) -> Expression:
        """Parsea una expresión con precedencia"""
        return self._parse_or()
    
    def _parse_or(self) -> Expression:
        left = self._parse_and()
        while self._match_type(TokenType.OR):
            operator = 'OR'
            right = self._parse_and()
            left = BinaryOp(left, operator, right)
        return left
    
    def _parse_and(self) -> Expression:
        left = self._parse_comparison()
        while self._match_type(TokenType.AND):
            operator = 'AND'
            right = self._parse_comparison()
            left = BinaryOp(left, operator, right)
        return left
    
    def _parse_comparison(self) -> Expression:
        left = self._parse_additive()
        
        while True:
            if self._match_type(TokenType.EQUAL):
                operator = '='
            elif self._match_type(TokenType.NOT_EQUAL):
                operator = '<>'
            elif self._match_type(TokenType.LESS_THAN):
                operator = '<'
            elif self._match_type(TokenType.LESS_EQUAL):
                operator = '<='
            elif self._match_type(TokenType.GREATER_THAN):
                operator = '>'
            elif self._match_type(TokenType.GREATER_EQUAL):
                operator = '>='
            else:
                break
            
            right = self._parse_additive()
            left = BinaryOp(left, operator, right)
        
        return left
    
    def _parse_additive(self) -> Expression:
        left = self._parse_multiplicative()
        
        while True:
            if self._match_type(TokenType.PLUS):
                operator = '+'
            elif self._match_type(TokenType.MINUS):
                operator = '-'
            else:
                break
            
            right = self._parse_multiplicative()
            left = BinaryOp(left, operator, right)
        
        return left
    
    def _parse_multiplicative(self) -> Expression:
        left = self._parse_primary()
        
        while True:
            if self._match_type(TokenType.STAR):
                operator = '*'
            elif self._match_type(TokenType.SLASH):
                operator = '/'
            else:
                break
            
            right = self._parse_primary()
            left = BinaryOp(left, operator, right)
        
        return left
    
    def _parse_primary(self) -> Expression:
        token = self._current()
        
        # Literal
        if token.type == TokenType.NUMBER:
            self._advance()
            return Literal(float(token.value) if '.' in token.value else int(token.value), 'number')
        
        # String
        if token.type == TokenType.STRING:
            self._advance()
            return Literal(token.value, 'string')
        
        # Identificador o referencia de columna
        if token.type == TokenType.IDENTIFIER:
            name = token.value
            self._advance()
            
            # Verificar si es tabla.columna
            if self._match_type(TokenType.DOT):
                column = self._match(TokenType.IDENTIFIER).value
                return ColumnRef(column, name)
            
            return ColumnRef(name)
        
        # Star
        if token.type == TokenType.STAR:
            self._advance()
            return ColumnRef('*')
        
        # Paréntesis
        if token.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._match(TokenType.RPAREN)
            return expr
        
        # NOT
        if token.type == TokenType.NOT:
            self._advance()
            operand = self._parse_primary()
            return UnaryOp('NOT', operand)
        
        raise SyntaxError(f"Expresión inesperada: {token}")
    
    def _current(self) -> Token:
        """Token actual"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF
    
    def _advance(self) -> Token:
        """Avanza al siguiente token"""
        token = self._current()
        if token.type != TokenType.EOF:
            self.pos += 1
        return token
    
    def _match(self, token_type: TokenType) -> Token:
        """Verifica y consume un token específico"""
        if self._current().type != token_type:
            raise SyntaxError(f"Esperado {token_type}, encontrado {self._current()}")
        return self._advance()
    
    def _match_type(self, token_type: TokenType) -> bool:
        """Verifica si el token actual es del tipo especificado"""
        if self._current().type == token_type:
            self._advance()
            return True
        return False
