#!/usr/bin/env python3
"""
minidb - Motor SQL Educativo
Punto de entrada principal
"""

import sys
import os


def main():
    """Punto de entrada"""
    
    # Argumentos
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'repl':
            # Abrir REPL interactivo
            db_file = sys.argv[2] if len(sys.argv) > 2 else 'minidb.db'
            from interface.repl import REPL
            repl = REPL(db_file)
            repl.run()
        
        elif command == 'example':
            # Ejecutar ejemplo
            from example import main as run_example
            run_example()
        
        elif command in ['-h', '--help', 'help']:
            print_help()
        
        else:
            print(f"Comando desconocido: {command}")
            print_help()
    else:
        # Sin argumentos: mostrar ayuda
        print_help()


def print_help():
    """Muestra mensaje de ayuda"""
    print("""
minidb - Motor SQL Educativo
Uso: python main.py [comando]

Comandos:
  repl [archivo.db]    Abre interactive REPL
                       (por defecto: minidb.db)
  
  example              Ejecuta ejemplo de uso
  
  help, -h, --help     Muestra esta ayuda

Ejemplos:
  python main.py repl                # Inicia REPL con minidb.db
  python main.py repl mi_base.db     # Inicia REPL con mi_base.db
  python main.py example             # Ejecuta script de ejemplo
  python main.py help                # Muestra esta ayuda

Uso Programático:
  from database import Database
  
  db = Database("mi_base.db")
  db.execute("CREATE TABLE ...")
  db.execute("INSERT INTO ...")
  rows = db.execute("SELECT ...")
  db.close()

Para más información, ver README.md y DEVELOPMENT.md
    """)


if __name__ == '__main__':
    main()
