**Resumen**
- **Qué**: Se portó el código necesario de BayesCard dentro del repositorio `motor_SQL` y se encapsuló bajo el paquete `bayescard`.
- **Objetivo**: Hacer que el motor sea independiente del repo externo BayesCard, dejando solo lo necesario y permitiendo usar el optimizador "bayes" desde el motor.

**Estructura**
- **Paquete principal**: `bayescard` — fachada que expone el optimizador y el modelo de costes.
- **Vendorizado**: `bayescard/vendor/` — subpaquetes importados desde el repositorio original (Models, DataPrepare, DeepDBUtils, etc.) y adaptados para no romper import en tiempo de carga.
- **Entradas clave**: `sql/optimizer.py`, `bayescard/query_optimizer.py`, `bayescard/cost_model.py`, `catalog/catalog.py`.

**Qué se hizo exactamente**
- Se movieron/copiarion los módulos necesarios de BayesCard dentro de `bayescard/vendor/` y se creó una fachada `bayescard` para exponer la API mínima.
- Se implementó `BayesCardCostModel` en `bayescard/cost_model.py` que consulta estadísticas desde el catálogo.
- Se implementó `BayesCardQueryOptimizer` en `bayescard/query_optimizer.py` que aplica un reordenamiento DP (left-deep) seguro, usando el cost model.
- Se agregó `bayescard/train_bn.py` para entrenar BNs por tabla desde CSV y serializarlos como pickles en `bayescard/vendor/Models/`.
- Se agregó una base de ejemplo `demo_bn_db/` para poder entrenar y validar el flujo sin depender de datos externos.
- Se registró un adaptador en `sql/optimizer.py` que devuelve el optimizador cuando se solicita `"bayes"` o `"bayescard"`.
- Se añadieron stubs/alias locales para librerías externas (por ejemplo `pgmpy`, `pomegranate`) y se hizo que `pandas` sea opcional en módulos que solo lo usan para entrenamiento/ETL, para evitar fallos de importación si no están instaladas.

**Cómo funciona el motor hoy con Bayes**
- **Selección del optimizador**: El motor crea un optimizador mediante la fábrica `get_optimizer(...)` en `sql/optimizer.py`. Los valores reconocidos incluyen `"bayes"` y `"bayescard"`.
- **Interfaz**: El optimizador implementa `optimize(stmt: SelectStatement) -> SelectStatement`. Antes de ejecutar una `SELECT`, `ExecutionEngine` llama a `optimize` y ejecuta el plan resultante.
- **Decisiones del optimizador**:
  - Usa `BayesCardCostModel` para estimar cardinalidades y selectividades leyendo estadísticas del catálogo (conteo de filas y NDV por columna).
  - Si hay pickles BN en `bayescard/vendor/Models/`, el cost model intenta usarlos para estimar selectividades más informadas (marginales y joins).
  - El algoritmo de reordenamiento es DP (left-deep) con comprobaciones semánticas que evitan planes que referencian columnas no visibles aún.
  - El optimizador solo reordena `JOIN`s si las condiciones son compatibles con el orden propuesto; se reconstruyen los nodos `Join` del AST de forma segura.

**Archivos importantes**
- [sql/optimizer.py](sql/optimizer.py)
- [bayescard/query_optimizer.py](bayescard/query_optimizer.py)
- [bayescard/cost_model.py](bayescard/cost_model.py)
- [catalog/catalog.py](catalog/catalog.py)
- Vendorizado: [bayescard/vendor/](bayescard/vendor/)

**Ejemplo rápido (uso desde Python)**
```python
from sql.optimizer import get_optimizer
from catalog.catalog import Catalog

catalog = Catalog(...)  # tu catálogo ya cargado
opt = get_optimizer("bayes", catalog)

# `stmt` es un SelectStatement generado por el parser
opt_stmt = opt.optimize(stmt)
# luego ExecutionEngine ejecuta `opt_stmt`
```

O bien, para comprobar que la fachada importa correctamente:
```bash
python -c "import bayescard; import Models.Bayescard_BN as m; import Models.BN_ensemble_model as e; print(bayescard.BayesCardQueryOptimizer.__name__)"
```

**Entrenar modelos BN**
```powershell
python bayescard/train_bn.py --db-dir .\demo_bn_db --out-dir .\bayescard\vendor\Models --rows 50000
```

Ese comando entrena un BN simple por tabla y deja archivos como `departments_model.pkl`, `fact_model.pkl` y `locations_model.pkl` en `bayescard/vendor/Models/`. Después, el optimizador `bayes` puede cargar esos pickles automáticamente.

**Limitaciones y notas**
- Se añadieron stubs para evitar dependencias pesadas en tiempo de importación; para usar la parte de inferencia probabilística (entrenamiento / evaluación con `pgmpy` o `pomegranate`) se deben instalar las librerías reales.
- Actualmente el vendorizado contiene los módulos necesarios para optimización y algunas utilidades; todavía es posible recortar más archivos que no se usen en producción (tests, notebooks, scripts de entrenamiento).
- Recomendación: si necesitas la funcionalidad completa de BayesCard (entrenamiento de BNs, generación de SPNs completos), instala las dependencias enumeradas en el repo original (ver `bayescard/vendor` y los `requirements` correspondientes).

**Siguientes pasos sugeridos**
- Recortar el árbol `bayescard/vendor/` para dejar solo los módulos usados en tiempo de ejecución por el optimizador.
- Agregar pruebas que comparen planes/estimaciones entre el optimizador actual y el optimizador `bayes` sobre un dataset de ejemplo.
- Documentar exactamente qué archivos del vendorizado son críticos y cuáles pueden eliminarse.


