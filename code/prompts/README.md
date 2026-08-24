# prompts/

Esta carpeta documenta la interacción con modelos (Codex, Claude, etc.) por requerimiento.

## Nomenclatura

- `REQ_001_Catalog.md`
- `REQ_002_Optimizer.md`
- `REQ_003_IndexBTree.md`

Formato sugerido: `REQ_###_<Tema>.md`

## Contenido mínimo por archivo

1. Evaluación previa de decisión
2. Resumen del requerimiento
3. Restricciones y supuestos
4. Prompt final usado
5. Decisiones tomadas
6. Tareas pendientes

## Evaluación previa de decisión (obligatoria)

Antes de implementar, cada `REQ_###_*.md` debe incluir:

- solución recomendada y justificación,
- alternativas consideradas,
- trade-offs (complejidad, mantenibilidad, performance, riesgos),
- decisión final y por qué se eligió.

## Revisión senior en chat (obligatoria)

Antes de programar, la discusión en chat debe dejar claro:

- si la decisión actual es correcta o no,
- qué alternativa es mejor (si aplica),
- riesgos principales y edge cases,
- recomendación final (avanzar o ajustar).

Ese resultado debe quedar trazado en el `REQ_###_*.md` correspondiente.
