# REQ_002_Bayes_Cost_Alignment

## Evaluación previa de decisión

### Solución recomendada
- Ajustar `xai/optimizer_explainer.py` para que el costo estimado de `BayesOptimizer` incluya también `table_filter_selectivity`, igual que el optimizador real en `bayescard/query_optimizer.py`.

### Alternativas consideradas
- Dejar la explicación como está y documentar que es una aproximación.
- Reutilizar directamente el optimizador BayesCard para no duplicar lógica.

### Trade-offs
- Complejidad: baja, porque el cambio se limita a la capa explicativa.
- Mantenibilidad: mejora la consistencia entre optimización y explicación, pero sigue habiendo lógica duplicada.
- Performance: impacto despreciable, solo se calcula la selectividad local por tabla.
- Riesgos: si la extracción de predicados locales no coincide con BayesCard, la explicación podría seguir divergiendo en casos límite.

### Decisión final y justificación
- Implementar la alineación del costo XAI con BayesCard incluyendo filtros locales por tabla. Es el cambio mínimo que corrige la inconsistencia visible sin tocar el ranking real del optimizador.

## Revisión en chat (previa a implementación)

### ¿Es la decisión correcta?
- Sí. La discrepancia venía de que XAI omitía `table_filter_selectivity` en Bayes.

### Alternativas mejores (si aplica)
- Reusar el costo del optimizador directamente sería más fiel, pero acopla demasiado la capa explicativa a la lógica de planeamiento.

### Riesgos y edge cases
- Predicados locales complejos pueden no descomponerse exactamente igual que en BayesCard.
- Las dependencias opcionales de XAI (`numpy`, `sklearn`, `shap`) pueden no estar instaladas, aunque eso no afecta la ejecución del cambio.

### Recomendación final (go / adjust)
- Go.

## Resumen
- La explicación de Bayes ahora refleja el mismo modelo de costo que el optimizador usa para ordenar joins.

## Restricciones y supuestos
- No se modifica la lógica de decisión del optimizador, solo la explicación.
- Se preserva el comportamiento de Selinger y del fallback heurístico.

## Prompt final usado
```text
Alinear el costo estimado de la capa XAI con BayesCard para que incluya table_filter_selectivity por tabla, igual que el optimizador real, sin cambiar el planificador ni romper los tests existentes.
```

## Decisiones tomadas
- Se pasó `stmt` a la función de costo para poder extraer predicados locales por tabla.
- Se agregaron helpers locales en XAI para replicar la extracción de predicados de BayesCard.
- Se actualizó la documentación técnica del repositorio.

## Tareas pendientes
- Evaluar si conviene extraer la lógica de predicados locales a una utilidad compartida para evitar duplicación futura.
- Revisar si otras rutas de explicación contrastiva deberían usar el mismo costo alineado.

---

## Fix adicional: espacio de candidatos para BayesOptimizer

### Problema raíz
BayesCard ancla la tabla FROM y solo permuta los JOINs en su DP. La XAI generaba todas las permutaciones posibles (incluyendo órdenes donde la FROM cambia), por lo que comparaba el plan elegido contra candidatos que Bayes nunca evaluó. Esto causaba el mensaje "nuestra fórmula estima más costo pero Bayes lo eligió igual".

### Solución
Se agregó el parámetro `fix_base_table` a `generate_candidate_plans`. `explain_optimizer_decision` lo activa automáticamente para `BayesOptimizer`. Ahora los candidatos de Bayes solo permutan el orden de los JOINs, manteniendo la tabla FROM fija.
