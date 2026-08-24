"""
FASE 7: INTERFACE
REPL: loop interactivo tipo sqlite3
"""

from database import Database
import sys


class REPL:
    """
    Read-Eval-Print Loop para minidb.
    Permite usar el motor SQL de forma interactiva.
    """
    
    def __init__(self, db_file: str):
        self.db = Database(db_file)
        self.running = True
    
    def run(self):
        """Inicia el REPL"""
        print("minidb - Motor SQL educativo")
        print("Digite 'EXIT;' para salir, 'HELP;' para ayuda")
        print()
        
        buffer = ""
        
        while self.running:
            try:
                # Prompt
                if buffer:
                    prompt = "   > "
                else:
                    prompt = "db> "
                
                line = input(prompt)
                
                # Agregar a buffer
                buffer += line + " "
                
                # Si no termina con ;, esperar más lineas
                if not line.strip().endswith(';'):
                    continue
                
                # Procesar comando
                sql = buffer.strip()
                buffer = ""
                
                self._process_command(sql)
            
            except KeyboardInterrupt:
                print("\nCancelado por usuario")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        self.db.close()
        print("\nAdios!")
    
    def _process_command(self, command: str):
        """Procesa un comando SQL"""
        # Remover el punto y coma al final
        command = command.rstrip(';').strip()
        
        # Comandos especiales
        if command.upper() == "EXIT":
            self.running = False
            return
        
        if command.upper() == "HELP":
            self._show_help()
            return
        
        # Mostrar catálogo
        if command.upper() == "SHOW TABLES":
            self._show_tables()
            return
        
        # Ejecutar SQL
        try:
            result = self.db.execute(command)
            
            # Mostrar resultado
            if result:
                for row in result:
                    print(row)
                print(f"\n{len(result)} fila(s)")
            else:
                print("OK")
        
        except SyntaxError as e:
            print(f"Error de sintaxis: {e}")
        except ValueError as e:
            print(f"Error semántico: {e}")
    
    def _show_help(self):
        """Muestra ayuda"""
        print("""
Comandos SQL soportados:
  
  CREATE TABLE nombre (col1 TYPE1, col2 TYPE2, ...)
  CREATE INDEX nombre ON tabla (columna)
  
  SELECT col1, col2, ... FROM tabla [WHERE condición]
  INSERT INTO tabla [(col1, col2, ...)] VALUES (val1, val2, ...)
  UPDATE tabla SET col1=val1, ... [WHERE condición]
  DELETE FROM tabla [WHERE condición]

Tipos de datos:
  - INTEGER
  - VARCHAR(size)
  - FLOAT
  - BOOLEAN
  - DATE

Comandos especiales:
  - SHOW TABLES
  - HELP
  - EXIT

Ejemplo:
  db> CREATE TABLE students (id INTEGER, name VARCHAR(50), gpa FLOAT);
  db> INSERT INTO students VALUES (1, 'Alice', 3.8);
  db> SELECT name, gpa FROM students WHERE gpa > 3.5;
        """)
    
    def _show_tables(self):
        """Muestra todas las tablas"""
        if not self.db.catalog.tables:
            print("No hay tablas")
            return
        
        for name, table in self.db.catalog.tables.items():
            cols = ", ".join([f"{col.name} {col.type}" for col in table.columns])
            print(f"{name}: ({cols})")


def main():
    """Punto de entrada para ejecutar el REPL"""
    import argparse
    
    parser = argparse.ArgumentParser(description='minidb - Motor SQL educativo')
    parser.add_argument('database', nargs='?', default='minidb.db', 
                        help='Archivo de base de datos (por defecto: minidb.db)')
    
    args = parser.parse_args()
    
    repl = REPL(args.database)
    repl.run()


if __name__ == '__main__':
    main()
