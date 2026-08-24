#!/usr/bin/env python3
"""
Test simple para verificar que el motor funciona
"""

import os
import shutil
from database import Database


def test_basic_functionality():
    """Test básico: crear tabla, insertar, recuperar datos"""
    
    # Limpiar
    if os.path.exists("test_db.db"):
        shutil.rmtree("test_db.db", ignore_errors=True)
    
    print("Testing minidb...")
    
    try:
        # Crear base de datos
        db = Database("test_db.db")
        
        # Test 1: CREATE TABLE
        print("✓ Test 1: CREATE TABLE")
        db.execute("CREATE TABLE users (id INTEGER, name VARCHAR(50), age INTEGER)")
        assert "users" in db.catalog.tables
        print("  ✓ Tabla creada y registrada en catálogo")
        
        # Test 2: INSERT
        print("✓ Test 2: INSERT")
        db.execute("INSERT INTO users VALUES (1, 'Alice', 25)")
        db.execute("INSERT INTO users VALUES (2, 'Bob', 30)")
        db.execute("INSERT INTO users VALUES (3, 'Charlie', 25)")
        print("  ✓ 3 filas insertadas")
        
        # Test 3: SELECT all
        print("✓ Test 3: SELECT all")
        rows = db.execute("SELECT id, name, age FROM users")
        assert len(rows) == 3
        print(f"  ✓ Recuperadas {len(rows)} filas")
        
        # Test 4: SELECT con WHERE
        print("✓ Test 4: SELECT with WHERE")
        rows = db.execute("SELECT name FROM users WHERE age = 25")
        assert len(rows) == 2
        assert rows[0][0] == 'Alice'
        assert rows[1][0] == 'Charlie'
        print(f"  ✓ WHERE funcionando correctamente")
        
        # Test 5: SELECT con múltiples columnas
        print("✓ Test 5: SELECT multiple columns")
        rows = db.execute("SELECT id, name FROM users WHERE age > 25")
        assert len(rows) == 1
        assert rows[0] == (2, 'Bob')
        print(f"  ✓ Proyección de columnas funcionando")
        
        # Test 6: CREATE INDEX
        print("✓ Test 6: CREATE INDEX")
        db.execute("CREATE INDEX idx_age ON users (age)")
        assert "idx_age" in db.catalog.indices
        print("  ✓ Índice creado y registrado")
        
        db.close()
        
        print("\n✅ Todos los tests pasaron!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test falló: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Limpiar
        if os.path.exists("test_db.db"):
            shutil.rmtree("test_db.db", ignore_errors=True)


if __name__ == '__main__':
    success = test_basic_functionality()
    exit(0 if success else 1)
