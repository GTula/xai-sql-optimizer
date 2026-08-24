#!/usr/bin/env python3
"""
Ejemplo de uso básico del motor SQL minidb
"""

from database import Database


def main():
    # Crear base de datos
    db = Database("ejemplo.db")
    
    try:
        # Crear tabla
        print("Creating table...")
        db.execute("CREATE TABLE instructor (id INTEGER, name VARCHAR(50), dept VARCHAR(30), salary FLOAT)")
        print("✓ Table created")
        
        # Insertar datos
        print("\nInserting data...")
        db.execute("INSERT INTO instructor VALUES (1, 'Einstein', 'Physics', 95000)")
        db.execute("INSERT INTO instructor VALUES (2, 'Mozart', 'Music', 40000)")
        db.execute("INSERT INTO instructor VALUES (3, 'Darwin', 'Biology', 60000)")
        db.execute("INSERT INTO instructor VALUES (4, 'Tesla', 'Physics', 55000)")
        print("✓ 4 rows inserted")
        
        # Query 1: SELECT sin WHERE
        print("\n--- SELECT all columns ---")
        rows = db.execute("SELECT id, name, dept, salary FROM instructor")
        for row in rows:
            print(row)
        
        # Query 2: SELECT con WHERE
        print("\n--- SELECT with WHERE ---")
        rows = db.execute("SELECT name, salary FROM instructor WHERE dept = 'Physics'")
        print("Physics professors:")
        for row in rows:
            print(f"  {row}")
        
        # Query 3: SELECT específico
        print("\n--- SELECT columns ---")
        rows = db.execute("SELECT name FROM instructor WHERE salary > 50000")
        print("High earners:")
        for row in rows:
            print(f"  {row}")
        
        print("\nExample completed successfully!")
    
    finally:
        db.close()


if __name__ == '__main__':
    main()
