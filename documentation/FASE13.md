# Fase 13 — SPATIAL_SMALL multimodal

## Estado y alcance

Implementación experimental aislada de Fases 9–12. Conserva datos, particiones,
CNN original, campañas, checkpoints y Streamlit. No utiliza transfer learning.
No se han realizado entrenamientos reales ni evaluación externa desde Codex.
Los ensayos de entrenamiento, recuperación e inferencia son sintéticos y temporales.

La referencia comunicada por el usuario es AUC CV ≈ 0,5860, DE ≈ 0,0294. Son cifras
aportadas como contexto, no recalculadas aquí ni sustituidas por resultados ficticios.
El comando `compare` requiere la campaña original real y no abre sus checkpoints.

## Auditoría de la CNN original

Verificada con PyTorch sobre `src/model.py` y `config/phase9/SPATIAL_SMALL.json`:

| Etapa | Operaciones | Salida para una imagen |
|---|---|---|
| Entrada | PRE, EARLY − PRE, LATE − PRE | 3 × 256 × 256 |
| Bloque 1 | Conv 3→16, BN, ReLU; Conv 16→16, BN, ReLU; MaxPool 2 | 16 × 128 × 128 |
| Bloque 2 | Conv 16→32, BN, ReLU; Conv 32→32, BN, ReLU; MaxPool 2 | 32 × 64 × 64 |
| Bloque 3 | Conv 32→64, BN, ReLU; Conv 64→64, BN, ReLU; MaxPool 2 | 64 × 32 × 32 |
| Agregación | AdaptiveAvgPool 1 × 1 y Flatten | 64 |
| Clasificador original | Dropout 0,40; Linear 64→1 | Un logit |

Convoluciones 3 × 3, padding 1, con bias; seis BatchNorm con los valores por
defecto originales (eps 1e−5, momentum 0,1), seis ReLU y tres MaxPool.
**72.528 parámetros en el extractor + 65 en el clasificador = 72.593.**
No se ha modificado su implementación ni se presupone que existan pesos locales.

## Nueva arquitectura

`SpatialSmallMultimodal` obtiene `features` de una instancia nueva de la CNN
original. Descarta únicamente su clasificador y concatena:

```text
Imagen [B,3,256,256] → extractor original → [B,64]
Clínica del paciente → transformador guardado → [B,D]
Concatenación [B,64+D] → Dropout(0.40) → Linear(64+D,1) → logit pCR
```

No hay MLP clínica, convoluciones nuevas, capas ocultas añadidas ni pesos precargados.
La rama clínica es una codificación determinista seguida de la fusión lineal.
El Dropout conserva su probabilidad y ahora actúa sobre el vector combinado.

Con las tres categorías presentes en cada train oficial, **D = 7**, vector
concatenado de 71 elementos y **72.600 parámetros totales**: solo siete más que
la CNN de imagen. El código calcula la dimensión por fold; si falta una categoría
en train, no la aprende de validación para mantener artificialmente la dimensión.
El checkpoint conserva su vocabulario y dimensión exactos.

## Auditoría de metadatos

`metadata/patients.csv` contiene 1.273 filas con clave `pid`, sin duplicados.
Las imágenes se enlazan mediante `samples.csv.patient_id = patients.csv.pid`.
Se comprueba la cobertura de desarrollo y la concordancia de sus etiquetas.
No se imprimen ni publican registros médicos individuales.

Desarrollo: **1.097 pacientes, 10.945 cortes**; reservados: **176 pacientes**.
Se ha comprobado que existen todas las imágenes de desarrollo, sin abrir imágenes
externas. El índice oficial se verifica mediante su hash original.

| Fold de validación | Train pacientes / cortes | Validación pacientes / cortes | no pCR / pCR (pacientes) |
|---|---|---|---|
| 0 | 878 / 8.759 | 219 / 2.186 | 155 / 64 |
| 1 | 878 / 8.755 | 219 / 2.190 | 157 / 62 |
| 2 | 877 / 8.753 | 220 / 2.192 | 161 / 59 |
| 3 | 877 / 8.753 | 220 / 2.192 | 145 / 75 |
| 4 | 878 / 8.760 | 219 / 2.185 | 157 / 62 |

Los informes generados por `audit` incluyen tipos y ausencias de cada columna
**solo en desarrollo**. No se usan asociaciones con etiquetas externas para
seleccionar variables, categorías, hiperparámetros o umbral.

| Variable | Decisión y tratamiento |
|---|---|
| `HR_HER2_STATUS` | Única representación del subtipo; en desarrollo hay 453 HRposHER2neg, 354 TripleNeg, 287 HER2pos y 3 ausentes. One-hot ajustado en train; buckets explícitos ausente y desconocido |
| `age` | Edad en años, 3 ausencias en desarrollo. Mediana de train; z-score sobre edades imputadas de pacientes únicos de train; indicador de ausencia |
| `tum_vol` | Excluida. `GUIA.md` la define como «Volumen tumoral», pero no acredita fecha de medición ni carácter basal. No se deduce ese carácter de su nombre ni de su distribución |
| HR/HER2 y otros indicadores de subtipo | Excluidos para evitar redundancia con el subtipo combinado |
| Identificadores, pCR, fold, split, test, dataset y otras columnas | No entran al tensor predictor. Los IDs se utilizan únicamente para unión y control de particiones |

Las categorías se obtienen exclusivamente de pacientes únicos del train del fold.
La media y DE de edad se calculan después de imputar con su mediana. Se usa DE
poblacional y escala 1 si la edad imputada es constante. Un train sin ninguna edad
observada se rechaza. Edades no finitas, no numéricas o fuera de 0–120 requieren
corrección de origen; no se convierten silenciosamente a cero.

El transformador JSON persiste columnas, orden, vocabulario, mediana, media, DE,
dimensión, número de pacientes y hash de sus IDs. Se valida al cargar. Todos los
cortes de cada paciente reciben el mismo vector clínico; no se mezclan filas por
posición ni se pierden pacientes mediante joins internos.

## Preprocesamiento y entrenamiento

Se reutilizan los operadores de `BreastDCEDataset` y el ajuste original
`fit_fold_statistics`: histogramas PNG8 completos, percentiles 1/99 por canal,
clipping y momentos calculados exclusivamente con las imágenes train del fold.
No se utiliza `metadata/statistics.json` global para los folds. No se cambia la
representación, resolución o agregación respecto a SPATIAL_SMALL.

Hiperparámetros de partida comprobados: AdamW, LR 0,0005, weight decay 0,0005,
batch 16, dropout 0,40, BCEWithLogitsLoss sin ponderar, semilla 42, sin augmentation.
Máximo 20 épocas y paciencia 8. ReduceLROnPlateau: modo max, factor 0,5, paciencia 2,
LR mínimo 1e−6, threshold 0. El objetivo es AUC de validación por paciente.

La puntuación de paciente es la media de las probabilidades sigmoid de sus cortes,
igual que en el proyecto original. AUC utiliza probabilidades continuas. La
clasificación usa 0,50 fijo, sin ajustar el umbral para aparentar una mejora.
Los folds se entrenan desde cero; no se reutilizan pesos entre pacientes de CV.

`train_loss` sigue siendo online. `train_eval_loss`, train AUC/accuracy y validación
son evaluaciones completas con los pesos de final de época, sin augmentation.
La evaluación restaura modos de cada módulo y RNG; no cambia pesos, buffers
BatchNorm, optimizador ni el generador de entrenamiento.

## Checkpoints y recuperación

Cada campaña y RUN utiliza hora de Madrid y UUID, con creación exclusiva:

```text
models/phase13/CV13_SPATIAL_SMALL_MULTIMODAL_FECHA_HORA_UUID/
├── config.json
├── campaign.json
├── audit.json
├── fold_inventory.csv
├── fold_0/
│   ├── train.csv
│   ├── validation.csv
│   ├── internals/FIT_FECHA_HORA_UUID/
│   │   ├── config.json
│   │   ├── image_statistics.json
│   │   └── clinical_transform.json
│   └── runs/RUN_FECHA_HORA_F13_FOLD0_UUID/
│       ├── config.py
│       ├── history.csv
│       ├── epoch_001.pt
│       ├── epoch_002.pt
│       ├── best_epoch.pt
│       ├── loss_curve.png
│       ├── accuracy_curve.png
│       ├── auc_curve.png
│       ├── roc_curve.png
│       └── internals/
│           ├── config.json
│           ├── clinical_transform.json
│           ├── training_contract.json
│           ├── inference_contract.json
│           ├── environment.json
│           ├── sources.zip
│           ├── epoch_metrics.json
│           ├── recovery_epoch_start.pt
│           ├── summary.json
│           ├── predictions_patient.csv
│           ├── predictions_slice.csv
│           ├── confusion_matrix_patient.png
│           ├── train_eval_loss_curve.png
│           ├── val_loss_curve.png
│           ├── loss_curve_limited.png
│           └── distribuciones y errores por paciente/corte
├── fold_1/ ... fold_4/
├── reports/SUMMARY_FECHA_HORA_UUID/
├── latest_report.json
└── recommended_final_training.json  # solo al completar CV definitiva
```

Cada checkpoint de época y el mejor contienen pesos, optimizador, scheduler,
configuración, clínica, contrato, RNG, historial y estado de selección/parada.
Se usa `weights_only=True` para cargar artefactos propios fiables. No se cargan
objetos de modelo completos ni se ejecuta el `config.py` de un RUN.

Antes de cada época se conserva su estado inicial. La reanudación crea otra carpeta,
repite solo la época incompleta y conserva el mejor checkpoint completo heredado.
Los contratos comprueban fuentes, hiperparámetros, estadísticas, datos clínicos
relevantes y CSV. Un cambio de CPU/GPU necesita aceptación explícita mediante flag.
Los folds completos se omiten incluso si se solicitan de nuevo; puede entrenarse
0–2 hoy y 3–4 después. Una campaña diagnóstica nunca se promociona a definitiva.
Un bloqueo del sistema operativo impide dos entrenamientos simultáneos sobre
la misma campaña y se libera al terminar el proceso, incluso tras un fallo abrupto.

## Resumen y recomendación final

Cada actualización escribe un informe nuevo con tabla de folds, media/DE de AUC,
resto de métricas, curvas medias y número de folds por época. No se rellenan épocas
tras early stopping. Con un único fold la DE muestral permanece ausente.
Solo con cinco folds completos, predicciones únicas y cobertura exacta se produce
OOF, matriz global y `recommended_final_training.json`.

La recomendación contiene las cinco mejores épocas (primer máximo de AUC por
paciente en cada fold), mediana, media, DE muestral, rango y configuración.
Recomienda la mediana entera de cinco valores, no un óptimo garantizado. La CV
selecciona checkpoints en la misma validación donde informa AUC: no es CV anidada,
y comparar varias versiones añade selección. El test reservado sigue separado.

El informe de comparación original/multimodal usa exclusivamente resultados reales
de campañas completas con las mismas particiones. Registra diferencias de
configuración y complejidad y produce una tabla pareada por fold. Las diferencias
se interpretan de forma descriptiva; cinco folds no son cinco muestras independientes
para probar superioridad. No se usa una accuracy dominada por la clase mayoritaria
como sustituto de discriminación, sensibilidad o especificidad.

## Modelo final e inferencia futura

Solo tras elegir la versión se habilita un ajuste nuevo con todos los pacientes de
desarrollo, estadísticas propias de desarrollo y las épocas fijadas por CV. No
hay validación para escoger checkpoints en este ajuste. El LR se fija previamente
con la mediana observada por época en CV, porque el scheduler adaptativo original
no puede consultar una validación inexistente. Esta política queda persistida.
Se guarda el checkpoint de la época prescrita como `final_epoch.pt`.

El contrato de inferencia incluye arquitectura, estado clínico, operadores y
estadísticas de imagen, formas, umbral y agregación. `load_bundle` y
`predict_patient` permiten una conexión posterior sin tocar Streamlit ahora.
Rechazan pesos CV, `SMOKE_TEST`, perfiles incompletos, claves clínicas extra y
dimensiones incoherentes. Las ausencias y categorías nuevas usan los mismos
transformadores de entrenamiento, sin ajuste sobre el estudio.

El test externo exige un comando y confirmación explícitos, modelo final completo,
checkpoint íntegro y umbral prefijado. Produce sus artefactos por separado y
registra una única evaluación. No modifica pesos, scheduler, umbral o épocas.
Ese comando **no se ha ejecutado sobre datos reales** en esta fase.

## Verificación y archivos nuevos

Las pruebas F13 cubren forward/backward, parámetros, alineación clínica, ausencia
de fuga, normalización, RNG/BatchNorm, checkpoints, reanudación exacta CPU,
interrupciones de época, folds progresivos, gráficos sin alterar datos, OOF,
recomendación, comparación sin pesos y contrato final con datos ficticios.
No establecen rendimiento predictivo ni prueban hardware ROCm físicamente.

Verificación del 29/10/2026: suite completa de 131 pruebas, **127 superadas y 4
aplazadas**, sin fallos. Incluye 16 tests F13 superados; los aplazados corresponden
a la integración futura de arquitecturas/checkpoints de Streamlit. Los 178
archivos previos (proyecto y base Streamlit local) conservan sus hashes exactos.

Archivos nuevos: siete módulos `src/phase13*.py`, una configuración en
`config/phase13/`, `tests/test_phase13.py`, esta documentación y `GPU_CPU_FASE13.md`.
No se editan archivos previos ni se publica la base Streamlit pendiente.
Comandos completos: [GPU_CPU_FASE13.md](../GPU_CPU_FASE13.md).
