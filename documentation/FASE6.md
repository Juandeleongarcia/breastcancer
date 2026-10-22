# Fase 6 — comparaciones controladas y ejecución con presupuesto

## Diagnóstico y límites de la evidencia

El usuario comunica SPATIAL_UNWEIGHT con AUC paciente 0,6360 en época 2, dropout 0,40 y LR 0,0005; LR 0,0001 obtuvo 0,5974 y dropout 0,20 obtuvo 0,5832. Son referencias comunicadas: no se inventan curvas, loss, estados de optimizador ni checkpoints. En este ordenador los runs completos locales de Fases 3–5 llegan a 0,552357; no se ha localizado el run de 0,6360. Los diagnósticos pequeños con AUC 1 no son resultados de generalización.

Train tiene 877 pacientes (620 no pCR / 257 pCR) y 8745 cortes; validación interna tiene 220 pacientes (155 / 65) y 2200 cortes. Predecir siempre no pCR ya produce accuracy de validación 0,704545, recall 0 y AUC 0,5. Por eso se selecciona por AUC continuo de paciente, con media de probabilidades de TODOS sus cortes y umbral de clasificación fijo 0,50.

Se conserva PRE, EARLY−PRE, LATE−PRE en float32, clipping y estandarización ajustados únicamente en train. Los perfiles se verifican contra el SHA256 del CSV y la representación. No se cambia el preprocesamiento ni se usa test. Se comprueban pacientes disjuntos, etiquetas coherentes dentro de paciente, clases, duplicados y procedencia train público. Esto no demuestra la veracidad clínica de las etiquetas originales ni ausencia de confusores de adquisición.

Train loss es online con pesos cambiantes, BatchNorm, dropout y augmentation si corresponde; validación usa pesos fijos al final de época y modo eval. La brecha no mide únicamente sobreajuste. Una caída de train loss junto con empeoramiento de validation loss y estancamiento de AUC es compatible con sobreajuste; el AUC y la calibración pueden evolucionar de forma distinta. No se cambia el umbral para aparentar mejoras de AUC.

El promedio global 1×1 elimina disposición espacial. Retener 2×2 prueba una hipótesis concreta, sin demostrar de antemano que sea la causa del solapamiento. La variante añade 384 parámetros (288033 → 288417); no añade bloques. Las CNN históricas y sus claves de pesos se conservan con SPATIAL_GRID=1 por defecto.

## Cinco experimentos

Referencia técnica congelada: `config/phase6/reference.json`, derivada de la configuración SPATIAL_UNWEIGHT disponible en Fase 5, sin afirmar que reproduce toda la configuración desconocida del run de clase 0,6360.

Protocolo común: semilla 42; todo train y validación interna; batch 16; LR 0,0005; AdamW; dropout 0,40; weight decay 0,0005; dos convoluciones por bloque [32,64,128]; BatchNorm; pérdida no ponderada; clipping y normalización train; máximo 10 épocas; early stopping por AUC paciente con paciencia 5. La paciencia cambia de 4 en el protocolo Fase 5 local a 5 en TODOS los experimentos Fase 6: debe declararse al comparar con históricos.

| ID | Única intervención frente a referencia | Hipótesis / límite |
|---|---|---|
| WEIGHT_DECAY | AdamW weight decay 0,0005 → 0,002 | Regularizar convoluciones y clasificador; puede inducir subajuste. |
| DROPOUT_050 | Dropout 0,40 → 0,50 | Regularizar clasificador sin repetir dropout 0,20. |
| SCHEDULER_FAST | ReduceLROnPlateau patience 2 → 0 | Reducir LR tras la primera época sin mejora; factor 0,5 y mínimo 1e-6 intactos. |
| REGIONAL_2X2 | AdaptiveAvgPool 1×1 → 2×2 y dimensión del Linear coherente | Conservar cuatro regiones; puede aumentar sobreajuste o sensibilidad a posición. |
| GENTLE_AUG | Activar una política gentle_affine | Geometría conjunta PRE/EARLY/LATE, ±5°, traslación/escala ±2%, probabilidad 0,8; sin alteraciones fotométricas. Dos claves JSON describen una sola intervención. |

Los cinco son contrastes contra la misma referencia; no se atribuyen las diferencias entre dos variantes distintas a una sola variable. No se añade un sexto control entrenado a la batería. Sin repetir la referencia, las comparaciones con 0,6360 son observacionales; convendría repetir la referencia en GPU bajo el mismo entorno antes de declarar superioridad. Una semilla no permite establecer diferencias estadísticamente estables.

## Ejecución automática en Windows / VS Code

Seleccionar el Python GPU que ya funciona en el PC de clase. No instalar el lock CPU ni sustituir PyTorch.

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -m src.batch_train --check-device cuda
python -m unittest discover -s tests -v
python -m src.phase6 --device cuda --max-minutes 120
```

El ejecutor mide las cinco configuraciones sobre batches reales, incluidos carga de imágenes y validación, antes de admitir un entrenamiento. Cada estimación contempla las diez épocas completas, margen del 30%, 15 segundos de guardado/gráficas por época y reserva final de 60 segundos. No se supone que early stopping ahorrará tiempo. Si un modelo no cabe, se registra explícitamente y se continúa al siguiente. El límite de reloj incluye benchmark, evaluación y guardado; el deadline de train impide seguir lanzando batches después del límite reservado. No se puede interrumpir de forma segura una operación individual ya en curso.

```powershell
python -m src.phase6 --device cpu --max-minutes 120 --benchmark-only
```

Tras revisar el benchmark puede continuarse la MISMA campaña, presupuesto y dispositivo con:

```powershell
python -m src.phase6 --device cpu --session 'models/PHASE6_IDENTIFICADOR'
```

Sustituir el identificador real. No reutilizar el benchmark CPU para GPU: en GPU se crea una campaña y medición nuevas. Los completados compatibles se omiten. Errores y sus tracebacks quedan registrados; las comparaciones se guardan después de cada decisión. Las interrupciones conservan checkpoints con el sistema de entrenamiento existente. Para reanudar explícitamente un run individual, conservar todo el directorio y utilizar:

```powershell
python -m src.train --resume 'models/RUN_IDENTIFICADOR/epoch_003.pt' --epochs 10 --device cuda
```

Diez significa épocas totales, no diez adicionales. La continuación crea otro RUN; no cambia pesos, optimizador, scheduler, RNG ni resultados del origen. Un cambio de arquitectura, configuración, particiones o fuentes no auditadas se rechaza. Se conserva compatibilidad de evaluación y de continuación para las fuentes exactas auditadas anteriores con grid 1; no se fuerza la carga de pesos de un clasificador 1×1 en uno 2×2. Los checkpoints más antiguos mantienen sus restricciones originales.

## Guardado y resultados

Cada entrenamiento conserva exactamente la raíz sencilla: `config.py`, `history.csv`, todos los `epoch_NNN.pt`, `best_epoch.pt`, `loss_curve.png`, `accuracy_curve.png`, `auc_curve.png`, `roc_curve.png`, y una sola subcarpeta `internals/`. Se mantienen fecha/hora Madrid, configuración y UUID. Las curvas principales accuracy/AUC/ROC son por paciente; loss es por corte y train online está identificado. El AUC por corte y demás métricas quedan en history y el resumen técnico. Los diagnósticos y fuentes comprimidas no añaden carpetas al nivel del RUN.

La campaña `models/PHASE6_.../` guarda `audit.json`, `execution.json`, `comparison.csv`, `INFORME.md` y `errors.jsonl` si hay fallos. La comparación incluye las cinco filas incluso si no se entrenan; métricas ausentes no se sustituyen por cero. Train loss y validation loss de la tabla corresponden a la mejor época de validación; también se registran loss finales, fracción predicha pCR y dispersión de probabilidades. Historial, matrices, distribuciones y errores del mejor checkpoint permiten revisar sobreajuste, discriminación y predominio de clase.

Prioridades provisionales si CPU no admite ningún run: WEIGHT_DECAY, SCHEDULER_FAST, REGIONAL_2X2. Son hipótesis, no ganadores. Ningún resultado parcial de benchmark se compara con validación completa ni con AUC histórico 0,6360. No hay commits ni pushes automáticos.

## Resultado de la campaña CPU del 22 de octubre de 2026

Campaña local: `models/PHASE6_20261022_134527_8473f618e781/`. PyTorch 2.14.0+cpu, CUDA no disponible. Se midieron cuatro batches reales de train y cuatro de validación por configuración, descartando sus pesos y métricas parciales. Ningún entrenamiento completo fue admitido porque ni el más rápido cabe en 120 minutos; no se redujeron retrospectivamente las épocas para producir una comparación distinta de la planificada.

| Experimento | Minutos estimados (10 épocas con margen) | Épocas completas | Mejor época | Train loss | Val loss | AUC paciente / corte | Accuracy / precision / recall / F1 / especificidad | Estado |
|---|---:|---:|---|---|---|---|---|---|
| WEIGHT_DECAY | 329.0 | 0 | — | — | — | — | — | Pendiente GPU: presupuesto CPU insuficiente |
| DROPOUT_050 | 313.5 | 0 | — | — | — | — | — | Pendiente GPU: presupuesto CPU insuficiente |
| SCHEDULER_FAST | 311.5 | 0 | — | — | — | — | — | Pendiente GPU: presupuesto CPU insuficiente |
| REGIONAL_2X2 | 324.1 | 0 | — | — | — | — | — | Pendiente GPU: presupuesto CPU insuficiente |
| GENTLE_AUG | 320.8 | 0 | — | — | — | — | — | Pendiente GPU: presupuesto CPU insuficiente |

Estimación total conservadora: 1598.9 minutos (26.65 horas). Estas cifras son extrapolaciones, no duraciones medidas de entrenamientos completos. El margen de admisión no garantiza una duración exacta, y el deadline sigue protegiendo frente a desviaciones.

No hay un nuevo ganador, ni evidencia de que una intervención haya mejorado o empeorado el AUC 0,6360. La tabla CSV local conserva todas las columnas solicitadas con valores ausentes, sin sustituirlos por métricas inventadas. Para realizar los cinco modelos se necesita GPU con benchmark propio o un presupuesto CPU mayor.

Verificación técnica: 34 pruebas distintas superadas (27 existentes y 7 de Fase 6), incluidas evaluación histórica con pooling ausente, rechazo de cambios incompatibles, reanudación exacta, memorización sintética, cuatro gráficas y guardia que impide entrenar cuando ningún modelo cabe. SHA256 de 38813 archivos originales/históricos sin cambios; configuraciones Fases 3–5 intactas. Las pruebas sintéticas no son resultados clínicos. CUDA no se ha probado en este ordenador CPU.
