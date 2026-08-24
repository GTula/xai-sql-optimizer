"""
FASE 5: EXECUTOR
Operadores: SeqScan, Filter, Project, etc.
Base de datos que ejecuta en forma de pipeline de operadores.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional, Iterator
from sql.parser import Expression, ColumnRef, BinaryOp, Literal, UnaryOp
from catalog.catalog import Table
from storage.heap_file import HeapFile
import struct


class Operator(ABC):
    """Operador base con interfaz de iterador"""
    
    @abstractmethod
    def open(self):
        """Abre el operador (inicialización)"""
        pass
    
    @abstractmethod
    def get_next(self) -> Optional[Tuple]:
        """Obtiene la próxima fila"""
        pass
    
    @abstractmethod
    def close(self):
        """Cierra el operador"""
        pass
    
    def to_list(self) -> List[Tuple]:
        """Obtiene todos los resultados como lista"""
        self.open()
        results = []
        while True:
            row = self.get_next()
            if row is None:
                break
            results.append(row)
        self.close()
        return results


class SeqScan(Operator):
    """
    Sequential Scan: escanea todas las filas de una tabla.
    """
    
    def __init__(self, table: Table, heap_file: HeapFile, column_names: List[str]):
        self.table = table
        self.heap_file = heap_file
        self.column_names = column_names
        self.iterator = None
    
    def open(self):
        self.iterator = self.heap_file.scan()
    
    def get_next(self) -> Optional[Tuple]:
        try:
            page_id, offset, record = next(self.iterator)
            # Deserializar registro
            row = self._deserialize_record(record)
            
            # Si se piden columnas específicas, filtrar
            if self.column_names != ['*']:
                result = []
                for col_name in self.column_names:
                    col_idx = self.table.get_column_index(col_name)
                    if col_idx is not None:
                        result.append(row[col_idx])
                return tuple(result)
            
            return tuple(row)
        except StopIteration:
            return None
    
    def close(self):
        self.iterator = None
    
    def _deserialize_record(self, data: bytes) -> List[Any]:
        """Deserializa un registro binario"""
        row = []
        offset = 0
        
        for col in self.table.columns:
            if col.type == 'INTEGER':
                value = struct.unpack('q', data[offset:offset+8])[0]
                offset += 8
            elif col.type == 'VARCHAR':
                length = struct.unpack('I', data[offset:offset+4])[0]
                offset += 4
                value = data[offset:offset+length].decode('utf-8')
                offset += length
            elif col.type == 'FLOAT':
                value = struct.unpack('d', data[offset:offset+8])[0]
                offset += 8
            else:
                value = None
            
            row.append(value)
        
        return row


class Filter(Operator):
    """
    Filter: aplica una condición WHERE a las filas del operador hijo.
    """
    
    def __init__(self, child: Operator, condition: Expression, table: Table):
        self.child = child
        self.condition = condition
        self.table = table
    
    def open(self):
        self.child.open()
    
    def get_next(self) -> Optional[Tuple]:
        while True:
            row = self.child.get_next()
            if row is None:
                return None
            
            # Evaluar condición
            if self._evaluate_condition(row):
                return row
    
    def close(self):
        self.child.close()
    
    def _evaluate_condition(self, row: Tuple) -> bool:
        """Evalúa una expresión para una fila"""
        result = self._evaluate_expression(self.condition, row)
        return bool(result)
    
    def _evaluate_expression(self, expr: Expression, row: Tuple) -> Any:
        """Evalúa recursivamente una expresión"""
        if isinstance(expr, Literal):
            return expr.value
        
        elif isinstance(expr, ColumnRef):
            col_idx = self.table.get_column_index(expr.column)
            if col_idx is not None:
                return row[col_idx]
            raise ValueError(f"Columna '{expr.column}' no existe")
        
        elif isinstance(expr, BinaryOp):
            left = self._evaluate_expression(expr.left, row)
            right = self._evaluate_expression(expr.right, row)
            
            if expr.operator == '=':
                return left == right
            elif expr.operator == '<>':
                return left != right
            elif expr.operator == '<':
                return left < right
            elif expr.operator == '<=':
                return left <= right
            elif expr.operator == '>':
                return left > right
            elif expr.operator == '>=':
                return left >= right
            elif expr.operator == '+':
                return left + right
            elif expr.operator == '-':
                return left - right
            elif expr.operator == '*':
                return left * right
            elif expr.operator == '/':
                return left / right if right != 0 else None
            elif expr.operator == 'AND':
                return left and right
            elif expr.operator == 'OR':
                return left or right
            else:
                raise ValueError(f"Operador desconocido: {expr.operator}")
        
        elif isinstance(expr, UnaryOp):
            operand = self._evaluate_expression(expr.operand, row)
            if expr.operator == 'NOT':
                return not operand
            else:
                raise ValueError(f"Operador unario desconocido: {expr.operator}")
        
        return None


class Project(Operator):
    """
    Project: selecciona columnas específicas (proyección).
    """
    
    def __init__(self, child: Operator, columns: List[str], table: Table):
        self.child = child
        self.columns = columns
        self.table = table
    
    def open(self):
        self.child.open()
    
    def get_next(self) -> Optional[Tuple]:
        row = self.child.get_next()
        if row is None:
            return None
        
        # Proyectar columnas
        if self.columns == ['*']:
            return row
        
        result = []
        # El row ya tiene las columnas en el orden de la tabla
        # Necesitamos seleccionar las que se piden
        for col_name in self.columns:
            col_idx = self.table.get_column_index(col_name)
            if col_idx is not None:
                result.append(row[col_idx])
        
        return tuple(result)
    
    def close(self):
        self.child.close()


class Limit(Operator):
    """
    Limit: restringe el número de filas devueltas.
    """
    
    def __init__(self, child: Operator, limit: int):
        self.child = child
        self.limit = limit
        self.count = 0
    
    def open(self):
        self.child.open()
        self.count = 0
    
    def get_next(self) -> Optional[Tuple]:
        if self.count >= self.limit:
            return None
        
        row = self.child.get_next()
        if row is None:
            return None
        
        self.count += 1
        return row
    
    def close(self):
        self.child.close()


class RowConstructor:
    """
    Ayudante para construir valores en tuplas para insert.
    """
    
    @staticmethod
    def serialize_record(row: List[Any], columns_def) -> bytes:
        """Serializa una fila a bytes"""
        data = bytearray()
        
        for i, value in enumerate(row):
            col = columns_def[i]
            
            if col.type == 'INTEGER':
                data.extend(struct.pack('q', int(value)))
            elif col.type == 'VARCHAR':
                str_value = str(value)
                encoded = str_value.encode('utf-8')
                data.extend(struct.pack('I', len(encoded)))
                data.extend(encoded)
            elif col.type == 'FLOAT':
                data.extend(struct.pack('d', float(value)))
            else:
                raise ValueError(f"Tipo no soportado: {col.type}")
        
        return bytes(data)

class NestedLoopJoin(Operator):
    """
    Nested Loop Join: Combina filas de dos operadores basándose en una condición.
    """
    
    def __init__(self, left_child: Operator, right_child: Operator, condition: Expression, left_table: Table, right_table: Table):
        self.left_child = left_child
        self.right_child = right_child
        self.condition = condition
        self.left_table = left_table
        self.right_table = right_table
        self.left_row = None
    
    def open(self):
        self.left_child.open()
        self.right_child.open()
        self.left_row = self.left_child.get_next()
    
    def get_next(self) -> Optional[Tuple]:
        while self.left_row is not None:
            right_row = self.right_child.get_next()
            
            if right_row is None:
                # Se acabó la tabla derecha, avanzar en la izquierda y reiniciar derecha
                self.left_row = self.left_child.get_next()
                if self.left_row is None:
                    return None
                
                self.right_child.close()
                self.right_child.open()
                right_row = self.right_child.get_next()
                if right_row is None:
                    return None
            
            # Combinar filas
            combined_row = self.left_row + right_row
            
            # En una implementación real completa de operators.py, aquí se evaluaría
            # la condición del JOIN, similar a Filter._evaluate_condition
            return combined_row
            
        return None
    
    def close(self):
        self.left_child.close()
        self.right_child.close()
