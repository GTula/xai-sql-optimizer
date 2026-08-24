"""
FASE 4: SQL
Tokenizer (Lexer): convierte texto SQL en una lista de tokens
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


class TokenType(Enum):
    # Palabras clave
    SELECT = auto()
    FROM = auto()
    WHERE = auto()
    INSERT = auto()
    INTO = auto()
    VALUES = auto()
    UPDATE = auto()
    SET = auto()
    DELETE = auto()
    CREATE = auto()
    TABLE = auto()
    INDEX = auto()
    ON = auto()
    ANALYSE = auto()
    JOIN = auto()
    
    # Tipos de datos
    INTEGER = auto()
    VARCHAR = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    DATE = auto()
    
    # Operadores
    EQUAL = auto()          # =
    NOT_EQUAL = auto()      # <> o !=
    LESS_THAN = auto()      # <
    LESS_EQUAL = auto()     # <=
    GREATER_THAN = auto()   # >
    GREATER_EQUAL = auto()  # >=
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Símbolos
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    COMMA = auto()          # ,
    SEMICOLON = auto()      # ;
    STAR = auto()           # *
    PLUS = auto()           # +
    MINUS = auto()          # -
    SLASH = auto()          # /
    DOT = auto()            # .
    
    # Literales
    NUMBER = auto()         # 123, 45.67
    STRING = auto()         # 'texto'
    IDENTIFIER = auto()     # nombres_tabla, columnas
    
    # Especiales
    EOF = auto()
    WHITESPACE = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int = 1
    column: int = 1
    
    def __repr__(self):
        return f"Token({self.type.name}, '{self.value}')"


class Tokenizer:
    """
    Lexer que convierte SQL en tokens.
    """
    
    KEYWORDS = {
        'SELECT': TokenType.SELECT,
        'FROM': TokenType.FROM,
        'WHERE': TokenType.WHERE,
        'INSERT': TokenType.INSERT,
        'INTO': TokenType.INTO,
        'VALUES': TokenType.VALUES,
        'UPDATE': TokenType.UPDATE,
        'SET': TokenType.SET,
        'DELETE': TokenType.DELETE,
        'CREATE': TokenType.CREATE,
        'TABLE': TokenType.TABLE,
        'INDEX': TokenType.INDEX,
        'ON': TokenType.ON,
        'ANALYSE': TokenType.ANALYSE,
        'ANALYZE': TokenType.ANALYSE,
        'JOIN': TokenType.JOIN,
        'INTEGER': TokenType.INTEGER,
        'VARCHAR': TokenType.VARCHAR,
        'FLOAT': TokenType.FLOAT,
        'BOOLEAN': TokenType.BOOLEAN,
        'DATE': TokenType.DATE,
        'AND': TokenType.AND,
        'OR': TokenType.OR,
        'NOT': TokenType.NOT,
    }
    
    def __init__(self, sql: str):
        self.sql = sql
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def tokenize(self) -> List[Token]:
        """Convierte el SQL en una lista de tokens"""
        while self.pos < len(self.sql):
            self._skip_whitespace()
            if self.pos >= len(self.sql):
                break
            
            # Intentar cada tipo de token
            if self._try_string():
                continue
            if self._try_identifier_or_keyword():
                continue
            if self._try_number():
                continue
            if self._try_operator():
                continue
            if self._try_symbol():
                continue
            
            # Si nada funcionó, error
            raise SyntaxError(f"Carácter inesperado '{self.sql[self.pos]}' en línea {self.line}")
        
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens
    
    def _current_char(self) -> Optional[str]:
        if self.pos < len(self.sql):
            return self.sql[self.pos]
        return None
    
    def _peek_char(self, offset: int = 1) -> Optional[str]:
        pos = self.pos + offset
        if pos < len(self.sql):
            return self.sql[pos]
        return None
    
    def _advance(self) -> str:
        char = self.sql[self.pos]
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def _skip_whitespace(self):
        while self.pos < len(self.sql) and self.sql[self.pos].isspace():
            self._advance()
    
    def _try_string(self) -> bool:
        if self._current_char() != "'":
            return False
        
        start_col = self.column
        self._advance()  # saltar '
        value = ""
        
        while self._current_char() and self._current_char() != "'":
            if self._current_char() == '\\':
                self._advance()
                value += self._advance()
            else:
                value += self._advance()
        
        if self._current_char() != "'":
            raise SyntaxError(f"String no cerrado en línea {self.line}")
        
        self._advance()  # saltar '
        self.tokens.append(Token(TokenType.STRING, value, self.line, start_col))
        return True
    
    def _try_identifier_or_keyword(self) -> bool:
        if not (self._current_char() and (self._current_char().isalpha() or self._current_char() == '_')):
            return False
        
        start_col = self.column
        value = ""
        
        while self._current_char() and (self._current_char().isalnum() or self._current_char() == '_'):
            value += self._advance()
        
        # Verificar si es keyword
        upper_value = value.upper()
        if upper_value in self.KEYWORDS:
            token_type = self.KEYWORDS[upper_value]
        else:
            token_type = TokenType.IDENTIFIER
        
        self.tokens.append(Token(token_type, value, self.line, start_col))
        return True
    
    def _try_number(self) -> bool:
        if not (self._current_char() and (self._current_char().isdigit() or self._current_char() == '-')):
            return False
        
        # Verificar si el guión es un operador o parte del número
        if self._current_char() == '-' and self._peek_char() and not self._peek_char().isdigit():
            return False
        
        start_col = self.column
        value = ""
        
        while self._current_char() and (self._current_char().isdigit() or self._current_char() == '.'):
            value += self._advance()
        
        self.tokens.append(Token(TokenType.NUMBER, value, self.line, start_col))
        return True
    
    def _try_operator(self) -> bool:
        char = self._current_char()
        next_char = self._peek_char()
        
        two_char_ops = {
            '<>': TokenType.NOT_EQUAL,
            '!=': TokenType.NOT_EQUAL,
            '<=': TokenType.LESS_EQUAL,
            '>=': TokenType.GREATER_EQUAL,
        }
        
        if char and next_char and char + next_char in two_char_ops:
            start_col = self.column
            self._advance()
            self._advance()
            self.tokens.append(Token(two_char_ops[char + next_char], char + next_char, self.line, start_col))
            return True
        
        one_char_ops = {
            '=': TokenType.EQUAL,
            '<': TokenType.LESS_THAN,
            '>': TokenType.GREATER_THAN,
        }
        
        if char and char in one_char_ops:
            start_col = self.column
            self._advance()
            self.tokens.append(Token(one_char_ops[char], char, self.line, start_col))
            return True
        
        return False
    
    def _try_symbol(self) -> bool:
        symbols = {
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            ',': TokenType.COMMA,
            ';': TokenType.SEMICOLON,
            '*': TokenType.STAR,
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '/': TokenType.SLASH,
            '.': TokenType.DOT,
        }
        
        char = self._current_char()
        if char and char in symbols:
            start_col = self.column
            self._advance()
            self.tokens.append(Token(symbols[char], char, self.line, start_col))
            return True
        
        return False
