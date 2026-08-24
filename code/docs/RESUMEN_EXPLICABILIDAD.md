# Resumen explicabilidad

- - Armamos un módulo separado para explicar por que el optimizador elige un plan.
- La idea principal es comparar distintos ordenes posibles de JOIN y ver por que se eligio uno.
- Agregamos una explicacion en texto, no solo numeros, para que sea mas facil de leer.
- Tambien mostramos los factores que mas influyeron en la decision.
- Se usa SHAP para explicar el peso de cada feature.
- Agregamos una comparacion con el segundo mejor plan, para entender mejor la diferencia.
- Ajustamos la explicacion de Bayes para que este mas alineada con lo que realmente usa el optimizador. Lo que se hizo fue usar los mismos estimadores que usa el optimizador para estimar las features del AST generado.
- Agregamos una validacion para comparar lo estimado con lo que pasa al ejecutar. En este caso se ejecuta el AST y se obtienen las features reales tras su ejecución. Esto nos sirve paa asegurarnos que la explicabilidad se haga sobre datos útiles y con ello poder generar valor real en las soluciones.
- Con el ultimo cambio, tambien se guardan features estimadas y reales en CSV (persistencia del ejecutar con trace).
- Esa persistencia sirve para analizar despues los resultados y comparar corridas.

## En resumen

- Ya tenemos una implementación sencilla de explicabilidad funcionando.
- Explica decisiones del optimizador sin cambiar como decide.
- Sirve para entender saber en qué features se basó principalmente el optimizador para elegir un plan por sobre otro; lo cual permite tomar decisiones con información extra para posibles casos de uso en negocios.


## Validación

- Agregamos una demo nueva para validar Bayes contra la ejecucion real: `xai/demo_bayes_runtime_validation.py`.
- Esta demo compara cada plan candidato contra el mismo plan ejecutado, sin volver a optimizar.
- Para eso agregamos una forma de ejecutar un AST ya armado con trace.
- La demo usa un caso mas complejo, con filtros en `departments` y `locations`.
- En ese caso vimos que Bayes estima bastante menos trabajo del que realmente hace el executor.
- Esto pasa porque Bayes estima como si pudiera aplicar filtros locales antes o durante los JOINs.
- Pero el executor actual aplica el `WHERE` al final, despues de hacer los JOINs.
- Entonces SHAP explica bien el modelo estimado, pero ese modelo no siempre coincide con la ejecucion real.
- La validacion ahora sirve porque detecta esa diferencia.
- Usamos `work` como suma de filas intermedias generadas por los JOINs.
- Tambien usamos `q-error` para medir que tan lejos esta lo estimado de lo real.
- En la demo nueva el q-error dio alto, entonces la estimacion no esta alineada con el runtime actual.
- Igual Bayes no eligio un plan peor en ese caso, porque los dos planes terminan con el mismo trabajo real.
- A lo mejor se podría ajustar el executor para hacer pushdown de filtros o ajustar la estimacion para reflejar mejor como ejecuta hoy el motor.
