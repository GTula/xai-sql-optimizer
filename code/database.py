"""
Punto de entrada: clase Database que conecta todos los componentes
"""

import os
from catalog.catalog import Catalog
from executor.engine import ExecutionEngine


class Database:
    """
    Interfaz principal de la base de datos minidb.
    
    Ejemplo de uso:
    
    >>> db = Database("mi_base.db")
    >>> db.execute("CREATE TABLE instructor (id INTEGER, name VARCHAR(50), dept VARCHAR(30), salary FLOAT)")
    >>> db.execute("INSERT INTO instructor VALUES (1, 'Einstein', 'Physics', 95000)")
    >>> db.execute("INSERT INTO instructor VALUES (2, 'Mozart', 'Music', 40000)")
    >>> rows = db.execute("SELECT name, salary FROM instructor WHERE dept = 'Music'")
    >>> print(rows)
    [('Mozart', 40000.0)]
    """
    
    def __init__(self, db_file: str, optimizer_type: str = "basic"):
        """
        Inicializa la base de datos.
        
        Args:
            db_file: ruta del archivo de base de datos
            optimizer_type: tipo de optimizador a usar ('basic', 'selinger', etc.)
        """
        self.db_file = db_file
        self.optimizer_type = optimizer_type

        # Estructura de almacenamiento:
        # <base>/databases/
        #   ├─ catalog/    (metadatos en CSV)
        #   └─ <db_name>/  (tablas CSV de la base activa)
        abs_db_path = os.path.abspath(db_file)
        parent_dir = os.path.dirname(abs_db_path)
        db_name = os.path.splitext(os.path.basename(abs_db_path))[0]

        self.databases_root = os.path.join(parent_dir, "databases")
        self.catalog_dir = os.path.join(self.databases_root, "catalog")
        self.db_name = db_name
        self.db_dir = os.path.join(self.databases_root, self.db_name)

        os.makedirs(self.catalog_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)

        # Inicializar componentes
        self.catalog = Catalog(self.catalog_dir, self.db_name)
        self.engine = ExecutionEngine(self.catalog, self.db_dir, self.optimizer_type)
    
    def execute(self, sql: str):
        """
        Ejecuta una query SQL.
        
        Args:
            sql: comando SQL
        
        Returns:
            Lista de tuplas con los resultados (para SELECT)
            o lista vacía (para DDL/DML)
        
        Raises:
            ValueError: si hay error semántico
            SyntaxError: si hay error de sintaxis
        """
        try:
            return self.engine.execute(sql)
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    def close(self):
        """Cierra la base de datos."""
        # La persistencia es incremental en esta rama.
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
