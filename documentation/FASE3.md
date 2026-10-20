# Fase 3 — organización y experimentos controlados

Fecha: 20 de octubre de 2026. Entrada exclusivamente de imágenes DCE-MRI; objetivo pCR/no pCR.

## Cambios y preservación

La nueva organización se aplica solo a runs futuros. Los 118 archivos de runs anteriores se comprobaron con SHA-256 y permanecen intactos. No se movieron ni borraron modelos, checkpoints o resultados; tampoco se modificaron dataset, particiones ni .gitignore. No se hicieron commits ni pushes.

La arquitectura baseline permanece disponible con [32,64,128], ReLU, BatchNorm, MaxPooling, AdaptiveAvgPool y clasificador original: 93.825 parámetros. La alternativa configurada [32,64] tiene 19.649 parámetros; no sustituye al baseline.

Archivos ajustados en esta fase: src/experiment.py, src/metrics.py, src/evaluate.py, src/compare_runs.py, calcular_stats.py y README.md. Nuevos: src/phase3.py, tests/test_phase3.py, config/phase3/ y este informe. Backup de entrada: backups/phase3_20261020_165852_debccb9d/. Los cambios de la fase 2 se conservaron.

## Estructura de los nuevos runs

```text
RUN_identificador/
  RESULTADOS.md, analysis.md, history.csv, epoch_summary.csv
  summary.txt, summary.json
  loss_curve.png, loss_history.png
  accuracy_curve_patient.png, accuracy_curve_slice.png
  auc_curve_patient.png, auc_curve_slice.png
  roc_curve_patient.png, roc_curve_slice.png
  confusion_matrix_patient.png, confusion_matrix_slice.png
  probability_distribution_patient.png, probability_distribution_slice.png
  checkpoints/
    best_epoch.pt, epoch_001.pt, epoch_002.pt, epoch_003.pt
  details/
    config.py, config.json, environment.json, sources/
    train_selection.csv, validation_selection.csv
    epoch_001/... (predicciones de train y validación por corte/paciente)
    best_predictions/, errors_patient.csv, errors_slice.csv
    probability_analysis.json, independent_evaluation_check.json
```

Los nombres históricos de algunas gráficas se conservan como compatibilidad. Las figuras con sufijo patient y epoch_summary.csv destacan el nivel principal. El evaluador sigue aceptando checkpoints antiguos con su contrato, y admite la nueva ubicación checkpoints/. Una fecha límite antigua del entrenamiento no bloquea la evaluación posterior.

## Protocolo fijo

Train interno: 877 pacientes, 8.745 cortes. Validación interna completa: 220 pacientes, 2.200 cortes, 155 no pCR y 65 pCR. No se han usado subconjuntos ni test en estos dos entrenamientos.

AdamW, LR=0.0005, weight decay=0.0005, batch=16, dropout=0.40, seed=42, num_workers=0, cuatro hilos CPU, tres épocas. BCEWithLogitsLoss ponderada con pos_weight=2.408028. Entrada baseline [PRE,EARLY−PRE,LATE−PRE], clipping y normalización global originales.

Cada paciente recibe la media de las probabilidades de sus cortes. Selección y early stopping por AUC de paciente, con AUC continuo y umbral de clasificación fijo 0.50. No se optimizó el umbral. Las métricas de train son online y no equivalen a una evaluación del modelo final con pesos fijos.

## Estimación y presupuesto

Benchmark de ocho batches de train y ocho de validación: 12.43 min/época, 37.55 min para tres épocas, 375.45 min (6.26 h) para diez runs al coste del baseline. Por ello no se lanzó la batería completa.

El baseline terminó en 28.22 min. Tras revisar métricas, curvas, errores y distribuciones, se midió la alternativa de dos bloques: estimación 21.96 min, suficiente margen dentro del tiempo restante. Se ejecutó como única modificación CONV_CHANNELS y terminó en 20.69 min.

Campaña completa con benchmarks, verificaciones y ajuste de estadísticas: **58.69 minutos**, por debajo del máximo de 60. Entrenamientos principales: **48.91 minutos**. No quedaron entrenamientos en segundo plano.

Existe un control de fecha límite entre batches. Si interrumpe una época conserva sus pesos en checkpoints/interrupted_epoch_XXX.pt; una validación incompleta no selecciona el mejor modelo. La comprobación de tiempo antes de cada run añade margen.

## Resultados reales del mejor checkpoint de cada run

| Run | CNN | Épocas | Mejor época | AUC paciente | AUC corte | Accuracy paciente | Val loss | Recall | F1 | Tiempo min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3_01 | [32, 64, 128] | 3 | 2 | 0.530521 | 0.529881 | 0.350000 | 0.991894 | 0.938462 | 0.460377 | 28.22 |
| P3_08 | [32, 64] | 3 | 2 | 0.505608 | 0.505599 | 0.404545 | 0.987034 | 0.815385 | 0.447257 | 20.69 |

Los dos son entrenamientos completos y comparables dentro de este protocolo. Los diagnósticos parciales de la fase 2 no se han mezclado con esta tabla. El mejor AUC observado corresponde a P3_01, época 2; no demuestra superioridad estable con una única semilla y solo tres épocas.

Tabla automática con los diez experimentos y estados: models/PHASE3_20261020_170224_34357e40/comparison.csv. Las ocho variantes no ejecutadas no tienen métricas inventadas.

## Evolución y errores

### P3_01

Carpeta: `models/RUN_20261020_170316_P3_01_1156db58ca9d`.

AUC de paciente por época: 0.505211, 0.530521, 0.466501.
Train loss por época: 0.980307, 0.973059, 0.965758.
Val loss por época: 0.993193, 0.991894, 1.025428.

Matriz [[TN,FP],[FN,TP]]: [[16, 139], [4, 61]]. Precision=0.305000; especificidad=0.103226. Fracción predicha pCR=0.9091.
Media de puntuaciones no pCR=0.553348; pCR=0.557938. Distribuciones fuertemente solapadas. Los errores individuales están en details/errors_patient.csv y details/errors_slice.csv.

### P3_08

Carpeta: `models/RUN_20261020_173221_P3_08_48f2dac7c30d`.

AUC de paciente por época: 0.498263, 0.505608, 0.487841.
Train loss por época: 0.980175, 0.977467, 0.973834.
Val loss por época: 0.992743, 0.987034, 0.990422.

Matriz [[TN,FP],[FN,TP]]: [[36, 119], [12, 53]]. Precision=0.308140; especificidad=0.232258. Fracción predicha pCR=0.7818.
Media de puntuaciones no pCR=0.522665; pCR=0.525720. Distribuciones fuertemente solapadas. Los errores individuales están en details/errors_patient.csv y details/errors_slice.csv.

La reducción de train loss no se acompaña de una mejora sostenida de AUC de validación. En ambos runs la tercera época empeora el AUC. Esto es compatible con inestabilidad o aprendizaje poco generalizable, pero tres épocas no identifican por sí solas la causa.

La sensibilidad alta del mejor baseline se explica en parte por predecir pCR en 200/220 pacientes. Un predictor siempre no pCR tendría accuracy 70.45% y AUC 0.50; uno siempre pCR tendría recall 100% y F1 0.4561. Las métricas aisladas pueden ser engañosas. La accuracy del baseline sube en época 3 mientras el AUC baja: por eso se conserva la época 2.

La salida ponderada no debe interpretarse como probabilidad calibrada sin estudiarlo. Cambiar el umbral puede cambiar sensibilidad y especificidad, pero no mejora el AUC de las puntuaciones.

## Preprocesamiento verificado

Las estadísticas se ajustaron con TODO train interno, después de construir canales y aplicar clipping cuando corresponde, y antes de la estandarización. No se utilizaron validación ni test. Todos los perfiles contienen 8.745 muestras, diagnostic_partial=false y la huella SHA-256 del CSV de train.

Perfiles portables en config/phase3/stats/: baseline_clipped.json, enhancement_no_clip.json y raw_no_clip.json.

El baseline reproduce las medias guardadas con diferencia máxima 3.41e-11, y las desviaciones con diferencia máxima 4.72e-11. No se cambiaron sus constantes ni se reescribieron checkpoints.

En los canales de realce, aproximadamente 1% de los píxeles queda en cada límite. En PRE, el 16.47% en el límite inferior incluye ceros ya presentes; no significa que el clipping haya destruido ese porcentaje de tejido. Las fracciones en los límites no cuantifican por sí solas información diagnóstica perdida.

El clipping sigue siendo irreversible: su utilidad requiere la comparación P3_02. La normalización global afín conserva la información salvo el recorte previo; no se añadió normalización por imagen/fase, CLAHE ni ecualización. P3_10 cambia la representación respecto a P3_02 sin clipping y utiliza sus propias estadísticas.

## Diez configuraciones preparadas

| Experimento | Variable principal | Referencia | Estado |
|---|---|---|---|
| P3_01 | Baseline original | — | Ejecutado, 3 épocas completas |
| P3_02 | Sin clipping adicional; estadísticas consistentes | P3_01 | Pendiente GPU |
| P3_03 | LR 0.0001 | P3_01 | Pendiente GPU |
| P3_04 | LR 0.001 | P3_01 | Pendiente GPU |
| P3_05 | Dropout 0.20 | P3_01 | Pendiente GPU |
| P3_06 | Dropout 0.00 | P3_01 | Pendiente GPU |
| P3_07 | BCE sin ponderar | P3_01 | Pendiente GPU |
| P3_08 | Dos bloques [32,64] | P3_01 | Ejecutado, 3 épocas completas |
| P3_09 | Sin z-score global | P3_01 | Pendiente GPU |
| P3_10 | PRE/EARLY/LATE con estadísticas propias | P3_02 | Pendiente GPU |

Las plantillas están en config/phase3/P3_XX.json y pueden viajar con el código. Los perfiles estadísticos son agregados de train, sin imágenes ni pesos. Las configuraciones de run y los checkpoints originales se conservan en models/.

## Verificación y siguiente experimento

15 pruebas unitarias/sintéticas aprobadas. Se verificaron parámetros del baseline, agregación independiente del orden, AUC continuo, etiquetas conflictivas, configuración de una variable principal, nueva estructura, checkpoints y parada por presupuesto.

Reevaluación independiente sobre las 220 pacientes de ambos checkpoints: diferencia máxima de probabilidades 8.33e-17, diferencia de pérdida 0.0 y métricas de paciente idénticas. Se inspeccionaron las curvas y distribuciones; git diff --check pasó y los 118 archivos históricos conservaron sus huellas.

Recomiendo ejecutar P3_07 en GPU como siguiente comparación contra P3_01: estudiar la contribución de la pérdida ponderada al sesgo hacia positivos, sin asumir que quitarla mejorará el AUC. Después P3_03 permitiría estudiar la inestabilidad entre épocas cambiando únicamente LR. Confirmar cualquier candidato con más semillas/folds y un entrenamiento suficiente antes de usar test.

```powershell
python -m src.train --config config/phase3/P3_07.json --device cuda
```

Para clipping o representación, respectivamente:

```powershell
python -m src.train --config config/phase3/P3_02.json --stats config/phase3/stats/enhancement_no_clip.json --device cuda
python -m src.train --config config/phase3/P3_10.json --stats config/phase3/stats/raw_no_clip.json --device cuda
```

El backend CUDA debe existir en el ordenador de clase; estos comandos fallan explícitamente si no está disponible. No se han ejecutado esos experimentos. No se garantiza alcanzar AUC 0.60–0.70.
