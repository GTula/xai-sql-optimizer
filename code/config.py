"""
Constantes globales para el motor SQL minidb
"""

# Tipos de datos soportados
SUPPORTED_TYPES = {
    'INTEGER': 'int',
    'VARCHAR': 'str',
    'FLOAT': 'float',
    'BOOLEAN': 'bool',
    'DATE': 'date'
}

# Tamaño máximo de una cadena VARCHAR (bytes)
MAX_VARCHAR_SIZE = 255

# Número máximo de índices por tabla
MAX_INDEXES_PER_TABLE = 10
