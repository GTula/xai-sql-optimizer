# Documentacion de Archivos - motor_SQL

Este documento inventaria la estructura actual del proyecto y que hace cada pieza.

## Raiz del proyecto

### config.py
- Proposito: constantes globales y tipos soportados.
- Uso real actual: `SUPPORTED_TYPES`.

### database.py
- Proposito: fachada principal del motor.
- Clase: `Database`.
- Metodos:
  - `__init__(db_file)`: crea `databases/catalog/` y `databases/<db_name>/`.
  - `execute(sql)`: delega la ejecucion al engine.
  - `close()`: cierra la base; la persistencia es incremental.
  - `__enter__()` y `__exit__(...)`: soporte para `with`.

### main.py
- Proposito: punto de entrada por consola.
- Funciones:
  - `main()`: procesa `repl`, `example` y ayuda.
  - `print_help()`: muestra uso basico.

### example.py
- Proposito: ejemplo programatico de uso.

### test_minidb.py
- Proposito: prueba basica del flujo principal.

### README.md
- Proposito: guia breve de uso y arranque.

### docs/DOCUMENTACION_ARCHIVOS.md
- Proposito: inventario tecnico del estado actual del proyecto.

## catalog/

### catalog/catalog.py
- Proposito: catalogo persistido como tablas CSV.
- Clases:
  - `Column`: nombre y tipo de columna.
  - `Index`: nombre, tabla, columna y archivo de indice.
  - `Table`: nombre de tabla, columnas y archivo CSV.
  - `Catalog`: administra tablas, indices y estadisticas.
- Tablas de metadatos:
  - `databases.csv`
  - `tables.csv`
  - `columns.csv`
  - `indexes.csv`
  - `statistics.csv`
- Metodos clave:
  - `create_table(...)`
  - `create_index(...)`
  - `load()`
  - `update_table_row_count(...)`

## executor/

### executor/engine.py
- Proposito: convierte el AST en ejecucion real sobre CSV.
- Sentencias soportadas:
  - `SELECT`
  - `INSERT`
  - `CREATE TABLE`
  - `CREATE INDEX`
  - `ANALYSE`
- `SELECT`: lee CSV, filtra en memoria y proyecta columnas.
- `INSERT`: agrega filas al CSV de la tabla.
- `ANALYSE`: actualiza `row_count` en `statistics.csv`.

### executor/operators.py
- Proposito: operadores del modelo iterador/pipeline.
- Clases:
  - `Operator`
  - `SeqScan`
  - `Filter`
  - `Project`
  - `Limit`
  - `RowConstructor`

## index/

### index/btree.py
- Proposito: estructura B-tree persistida para indices.
- Estado actual: el indice se crea y se registra, pero todavia no se usa como plan alternativo en `SELECT`.

## sql/

### sql/tokenizer.py
- Proposito: convierte SQL en tokens.

### sql/parser.py
- Proposito: convierte tokens en AST.
- Incluye `AnalyseStatement` para `ANALYSE [tabla]`.

### sql/validator.py
- Proposito: valida semantica contra el catalogo.
- Tambien valida `ANALYSE`.

## interface/

### interface/repl.py
- Proposito: REPL interactivo para ejecutar SQL.

## storage/

### Estado actual
- En esta rama no hay una implementacion activa de `storage/` de paginas o buffer pool.
- La persistencia de datos se resolvio directamente con CSV.

## Datos y carpetas generadas

- `databases/catalog/`: metadatos CSV globales.
- `databases/<db_name>/`: tablas CSV de cada base.
- `__pycache__/`: bytecode generado por Python.

## Resumen rapido

1. `sql/` entiende la query.
2. `sql/validator.py` la valida contra el catalogo.
3. `executor/engine.py` la ejecuta sobre CSV.
4. `catalog/catalog.py` guarda metadatos y estadisticas.
5. `index/btree.py` crea indices, pero aun no altera el plan de `SELECT`.

## xai/

### xai/explanation.py
- Proposito: contenedor de datos de la explicacion del optimizador.
- Clase: `Explanation`.
- Campos principales:
  - `optimizer_name`, `chosen_plan`, `top_features`, `metadata`
  - `selected_plan_score`, `compared_plans_count`, `shap_used`
  - `feature_values_for_selected_plan`, `shap_values_for_selected_plan`
  - `natural_language_summary`, `warnings`
- Campos de la capa de interpretacion (nuevos):
  - `executive_summary`: 2–3 frases no tecnicas.
  - `factors_positive` / `factors_negative`: listas de factores con label, valor, SHAP, direccion y razon.
  - `runner_up_plan` / `runner_up_score`: segundo plan mas barato por costo estimado.
  - `explanation_quality`: "alta" / "media" / "baja".
  - `explanation_quality_reasons`: lista de razones de la calidad.
  - `feature_impact_pct`: porcentaje del impacto SHAP relativo por feature.
  - `feature_percentiles`: percentil (0–100) del plan elegido por feature respecto a candidatos.
  - `technical_detail`: tabla formateada con SHAP, impacto%, escala y percentil.
- Metodos: `to_dict()`, `__str__()`.

### xai/optimizer_explainer.py
- Proposito: genera explicaciones de las decisiones del optimizador SQL usando SHAP.
- Constantes semanticas:
  - `FEATURE_NAMES`: orden canonico de features.
  - `FEATURE_LABELS`: etiquetas legibles en espanol por feature.
  - `FEATURE_INTERPRETATION`: direccion ("low_is_good" / "high_is_good" / "neutral") y razon por feature.
  - `FEATURE_CATEGORIES`: agrupacion semantica de features por categoria funcional.
  - `_CONTRASTIVE_FEATURES`: features que varían entre permutaciones de JOIN.
- Funciones principales:
  - `generate_candidate_plans(stmt, max_plans, fix_base_table)`: enumera permutaciones del orden de JOIN. Con `fix_base_table=True` (BayesOptimizer) mantiene la tabla FROM fija y solo permuta los JOINs, replicando el espacio de búsqueda real de BayesCard.
  - `extract_plan_features(plan, stmt, optimizer)`: vector tabular de features por plan.
  - `build_feature_matrix(plans, stmt, optimizer)`: matriz X de features para todos los candidatos.
  - `predict_plan_cost(X, feature_names)`: scoring compatible con SHAP.
  - `compute_shap_explanation(X, feature_names, chosen_idx, top_k)`: SHAP sobre proxy Ridge.
  - `compute_fallback_explanation(X, feature_names, chosen_idx, top_k)`: fallback heuristico.
  - `build_natural_language_summary(...)`: resumen combinado (legado, conservado por compat.).
- Modelo de costo (`_estimate_plan_cost`):
  - Modelo acumulativo left-deep: `intermediate_i = intermediate_{i-1} * rows(T_i) * sel_i`.
  - `costo_total = sum(intermediate_0 .. intermediate_n)` (suma de intermedios en cada paso).
  - Esto hace que el orden de JOIN importe: empezar con tabla chica produce intermedios menores.
  - SelingerOptimizer: usa `_get_table_cost` + `_estimate_selectivity`.
  - BayesOptimizer: usa `cost_model.table_cardinality` + `table_filter_selectivity` + `join_selectivity` para alinearse con el planificador real. La tabla FROM queda fija (igual que el DP interno de BayesCard).
  - Otros: suma acumulativa de row counts sin selectividades.
- Helpers de interpretacion (seccion 10):
  - `compute_feature_percentiles(X, feature_names, chosen_idx)`: percentil por feature.
  - `compute_relative_impact(shap_values)`: impacto relativo % por feature SHAP.
  - `_semantic_impact_label(pct)`: "alto" / "medio" / "bajo".
  - `_feature_direction_for_plan(shap_val, feature)`: "mejora el costo" / "perjudica el costo" / "neutral".
  - `compute_explanation_quality(n_plans, shap_used, X, feature_names)`: calidad con razones.
  - `find_runner_up(candidates, X, feature_names, chosen_idx)`: segundo plan mas barato; retorna (tables, cost, idx).
  - `build_structured_explanation(...)`: produce executive_summary, factors_positive/negative, technical_detail y natural_language_summary completo.
- Explicacion contrastiva (seccion 11):
  - `build_contrastive_explanation(chosen_idx, runner_up_idx, X, feature_names, feature_dicts, candidates)`:
    - Si chosen_cost < runner_up_cost: "X% mas barato, regla: menor costo gana."
    - Si chosen_cost > runner_up_cost: nota de discrepancia entre formula XAI y modelo interno.
    - Si costos iguales: "planes equivalentes, desempate por orden de enumeracion."
- Punto de entrada:
  - `explain_optimizer_decision(stmt, optimizer, max_plans, top_k_features)`: orquesta todo y retorna `Explanation`.

### xai/demo_explainer.py
- Proposito: demo interactivo que muestra la explicacion estructurada en dos niveles.
- Muestra: resumen ejecutivo, runner-up, factores positivos/negativos, tabla tecnica con SHAP/impacto/percentil/direccion.

