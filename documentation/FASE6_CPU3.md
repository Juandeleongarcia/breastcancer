# Fase 6 CPU — hasta tres runs de tres épocas

Campaña independiente de las cinco configuraciones de diez épocas, que permanecen intactas. Orden: REGIONAL_2X2, WEIGHT_DECAY, GENTLE_AUG. Se cambian únicamente EPOCHS=3, RUN_NAME y la etiqueta INTERPRETATION; arquitectura, hiperparámetros restantes, preprocesamiento y particiones se heredan exactamente. Todo train (877 pacientes / 8745 cortes) y validación interna (220 / 2200), semilla 42, sin test ni subconjuntos. Paciencia original 5 conservada: con tres épocas el fin viene del límite de épocas, no del early stopping.

```powershell
python -m src.phase6_cpu3 --max-minutes 120
```

Se mide de nuevo cada candidato con cuatro batches reales de train y cuatro de validación, usando pesos descartables. El benchmark es temporal y no proporciona métricas de rendimiento comparables. La estimación de las tres épocas incluye margen del 30%, 15 segundos por época para guardado y gráficas, más reserva global de 120 segundos. Cada admisión vuelve a comprobar el tiempo disponible. No se reduce el número de épocas ni se cambian hiperparámetros tras observar resultados. Si un candidato no cabe, se registra y se sigue al siguiente, sin saltarse la prioridad entre los candidatos admisibles. No se empieza un run sin estimación válida. El límite incluye benchmark, entrenamiento, validación y guardado; no se usan estimaciones CPU anteriores como si fueran mediciones actuales.

Cada run nuevo tiene fecha/hora Madrid, configuración y UUID. En su raíz: config.py, history.csv, epoch_001.pt, epoch_002.pt, epoch_003.pt, best_epoch.pt, loss_curve.png, accuracy_curve.png, auc_curve.png, roc_curve.png. Archivos técnicos y recuperación en internals/. Se conservan optimizador, scheduler, RNG e historial. Los runs anteriores no se mueven ni se modifican.

La comparación guarda las cinco configuraciones: las tres priorizadas y las dos restantes pendientes para GPU. Los resultados completos de tres épocas son PRELIMINARES por duración, aunque usan todos los datos internos. Las métricas se presentan por paciente, con AUC por corte separada. Train loss es online y no equivale a evaluar train con pesos fijos. Se conserva umbral 0,50 y el AUC se calcula con puntuaciones continuas; no se afina el umbral.

Los informes de campaña quedan en models/PHASE6_CPU3_FECHA_HORA_UUID/. El histórico 0,6360 es el dato comunicado por el usuario, sin asumir disponibilidad de su checkpoint o equivalencia exacta de entorno. Una sola semilla y tres épocas no prueban que una variante sea superior. Para GPU, seleccionar candidatos según AUC y estabilidad, repetir la referencia espacial bajo el mismo protocolo, y después ampliar a 10–20 épocas con early stopping por AUC de paciente. No combinar varias intervenciones hasta haberlas contrastado por separado.

No hay commit ni push automáticos. El dataset y los resultados siguen excluidos de Git.

## Resultado de esta campaña

Campaña `models/PHASE6_CPU3_20261022_140326_e422c6093330/`: WEIGHT_DECAY completó 2 de las 3 épocas previstas; se detuvo preventivamente la tercera por falta de margen después de una espera anómala de monitorización. Ningún run llegó a tres épocas completas. Mejor AUC paciente 0,611315, época 2; AUC corte 0,609615, train loss online 0,601081 y validation loss 0,591013. Accuracy 0,704545 con precision/recall/F1 de pCR 0 y especificidad 1: todos los pacientes se clasificaron no pCR al umbral fijo. No se observa sobreajuste creciente entre las dos épocas completas; duración insuficiente para descartarlo. El informe local contiene tabla de los cinco candidatos, evolución, distribuciones y recomendaciones de control espacial + REGIONAL_2X2 + GENTLE_AUG para GPU. Las otras configuraciones permanecen pendientes, sin métricas inventadas.
