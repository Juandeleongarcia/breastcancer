# Fase 2 — mejora controlada

Fecha: 19 de octubre de 2026. Se han ejecutado únicamente diagnósticos cortos en CPU.

## 1. Problemas corregidos

La validación seleccionaba checkpoints por corte, evaluate.py referenciaba componentes eliminados, el cálculo de estadísticas incluía la estandarización activa y faltaba Matplotlib. Se corrigieron estos contratos, el efecto de crear models/ al importar y los imports del test manual. Las predicciones ahora se conservan con sus identificadores.

## 2. Preservación

Backup anterior a cualquier edición: `backups/20261019_185319_00bf8cdd/`, con los nueve archivos originales modificados. No se tocaron las imágenes, CSV de partición, .gitignore ni los objetos Git. No se hicieron commits ni pushes.

La arquitectura baseline conserva Conv2d, BatchNorm, ReLU, MaxPool, AdaptiveAvgPool, Dropout 0.40 y Linear a un logit: 93.825 parámetros. Dos bloques se configuran con [32,64], 19.649 parámetros. El orden y las claves del state_dict permanecen compatibles para la misma arquitectura.

## 3. Archivos modificados y nuevos

Modificados: README.md, config/config.py, src/dataset.py, src/model.py, src/train.py, src/evaluate.py, src/metrics.py, src/test_pipeline.py y calcular_stats.py.

Nuevos: src/configuration.py, src/experiment.py, src/diagnostics.py, src/compare_runs.py, tests/test_metrics.py, requirements.txt, requirements-cpu-lock.txt y este informe. Los originales están en backups/. Los artefactos y pesos nuevos solo están en models/, que sigue excluido de Git.

La orquestación que estaba en train.py se ha movido a experiment.py para compartir el mismo evaluador con evaluate.py. train.py conserva su entrada, train_one_epoch y los helpers públicos; las utilidades correctas de métricas originales se mantienen.

## 4. Configuración

Se mantienen LR=0.0005, weight_decay=0.0005, AdamW, batch=16, seed=42, patience=10, [32,64,128], enhancement, clipping y estadísticas originales. El modo normal sigue en 25 épocas; SMOKE_TEST=False.

Se añaden dispositivo automático, cuatro hilos CPU, augmentation geométrico opcional desactivado, límites de batches, nombre, directorio, frecuencia de guardado, aliases de semilla/paciencia y contrato de estadísticas. La única métrica principal aceptada es patient_auc. Cada run recibe un snapshot resuelto y copias de las fuentes.

## 5. Evaluación por paciente

Cada corte conserva sample_id, patient_id, etiqueta y probabilidad. Se comprueba que no hay duplicados ni etiquetas contradictorias. Se promedian probabilidades por paciente y después se calculan AUC continuo, accuracy, precision, recall/sensibilidad, F1, especificidad y matriz [[TN,FP],[FN,TP]]. La selección y early stopping usan exclusivamente AUC de paciente. Los empates conservan la primera época.

Las columnas históricas val_auc/train_auc siguen siendo POR CORTE; las nuevas columnas val_patient_auc/train_patient_auc son explícitas. Las métricas online de train mezclan modelos de sucesivas actualizaciones y dropout; no equivalen a evaluar el modelo final en modo eval.

## 6. Pruebas ejecutadas

- 13 pruebas unitarias/sintéticas aprobadas: AUC continuo, medias por paciente, orden, conflictos, duplicados, valores no finitos, clase única, confusión, formas de normalización, parámetros y rechazo de estadísticas parciales/incompatibles.
- Forward, loss, backward y actualización real en CPU, con gradientes no nulos y finitos; sin NaN/inf.
- Batch fijo: cuatro muestras de cuatro pacientes de train, dos por clase. 12 actualizaciones; pérdida de evaluación fija 1.194121 a 0.930357 (reducción 22.1%). Pérdida de entrenamiento observada: 1.608627 a 0.267466, con dropout activo. No es un experimento de generalización ni prueba de convergencia a pérdida cero.
- Tres diagnósticos de dos épocas: A, B y D, exactamente los mismos cuatro pacientes/40 cortes de train y cuatro pacientes/40 cortes de validación. Se comprobaron las selecciones idénticas. Test no se ha utilizado.
- Reevaluación independiente de los tres mejores checkpoints: diferencia máxima de probabilidades 5.55e-17 y diferencia de pérdida 0.0.
- Estadísticas de 32 muestras de train: 0.54 segundos; solo diagnóstico del pipeline previo a estandarización. El archivo models/stats_probe_A.json está marcado como parcial y no es admisible para normalizar runs.
- pip check sin conflictos, comprobación de sintaxis, CLI y git diff --check aprobadas.

## 7. Resultados reales

**Todos los resultados de esta tabla son diagnósticos parciales. Se reporta el checkpoint seleccionado, época 1 en los tres casos.**

| Run | CNN | Loss | Clipping | Épocas ejecutadas | AUC paciente | AUC corte | Accuracy paciente | Val loss | Tiempo s |
|---|---|---|---|---:|---:|---:|---:|---:|
| A | [32, 64, 128] | ponderada | True | 2 | 0.5000 | 0.4925 | 0.50 | 1.227184 | 14.11 |
| B | [32, 64] | ponderada | True | 2 | 0.5000 | 0.5000 | 0.25 | 1.196125 | 10.87 |
| D | [32, 64, 128] | normal | True | 2 | 0.5000 | 0.4950 | 0.50 | 0.695082 | 12.17 |

Las pérdidas ponderadas y no ponderadas tienen escalas distintas; no se usan para declarar D superior. En los tres runs precision, recall y F1 de paciente del checkpoint seleccionado son 0. A/D tienen matriz [[2,0],[2,0]], B [[1,1],[2,0]]. No hay evidencia de mejora de generalización.

Identificadores completos:

- `RUN_20261019_185859_A_f95aaf8c5999`
- `RUN_20261019_190008_B_78d6d5ff4500`
- `RUN_20261019_190024_D_96ea11e77f75`

Batch fijo: `RUN_20261019_185815_technical_fixed_batch_ab058719fc43`. Tabla automática: `models/comparison_phase2.csv`. Los cuatro experimentos completos A–D quedan pendientes; los resultados históricos de clase no se han inventado.

## 8. Curvas y capacidad de aprendizaje

Loss de train entre épocas 1 y 2: A 1.41546→1.02295; B 1.24081→1.09954; D 0.73685→0.56735. Val loss: A 1.22718→1.26162; B 1.19612→1.19953; D 0.69508→0.69288. AUC de paciente: 0.50 en ambas épocas de los tres runs.

La red puede actualizar pesos y reducir la pérdida del batch fijo. Esto no demuestra mejora sobre pacientes nuevos: la validación es diminuta y solo hay dos épocas. No se ha establecido qué arquitectura o pérdida generaliza mejor.

## 9. Tiempo y recursos

PyTorch 2.14.0+cpu, CUDA no disponible. Matplotlib 3.11.2 añadido junto con sus dependencias; no se reemplazó PyTorch ni las versiones de numpy/pandas/sklearn. El lock registra el entorno observado; no fuerza el backend del equipo de clase.

Tiempo mediano de actualización del batch de cuatro: 0.2363 s. Extrapolación aproximada: 581.6 s por época completa con validación (9.7 min), 4652 s para cuatro runs de dos épocas (77.5 min). La extrapolación depende de batch, caché y E/S; no es un benchmark GPU. Se comunicó antes de ejecutar los smoke tests y no se lanzaron los runs completos.

Memoria observada al comprobar recursos: aproximadamente 13.84 GiB físicos, 2.36 GiB libres; un proceso Python de diagnóstico llegó a alrededor de 1.10 GiB de working set. Las pruebas se completaron sin OOM. Se usan cargas por batch, no todo el dataset en RAM.

## 10. Estadísticas y contraste

El baseline se conserva con procedencia legacy_unverified. Se ha comprobado el orden de fases, representación y etapa de cálculo con imágenes sintéticas, pero no se han certificado las medias/desviaciones históricas recorriendo todo train. Las constantes son finitas y sus desviaciones positivas.

El clipping es irreversible y puede recortar señal útil; no se afirma que sea inocuo. La estandarización global afín conserva los valores salvo el recorte previo. No se añade CLAHE ni ecualización. El cálculo completo se estima en 2–3 minutos a partir de 32 muestras y no se ejecutó para evitar una operación larga no aprobada. C exige ajuste completo de estadísticas sin clipping; está preparado y bloquea el uso accidental de las constantes incompatibles o parciales.

## 11. Pendientes

Recuperar runs de clase; verificar las estadísticas completas; ejecutar comparativas con train/validación completos en GPU; estudiar varias semillas/folds tras seleccionar pocos candidatos. La reanudación automática sigue pendiente; los checkpoints ya guardan optimizador, pero los comandos actuales inician runs nuevos. La evaluación final de test se reserva para cuando el modelo esté fijado.

## 12. Siguiente modificación recomendada

Primero verificar las estadísticas del baseline sobre TODO train y compararlas con las conservadas, sin reemplazarlas automáticamente. Después ejecutar A completo de dos épocas en clase como referencia. No aumentar complejidad ni cambiar simultáneamente preprocesamiento, pérdida y arquitectura.

## 13. Comandos para continuar

En el equipo con entorno/GPU verificado, desde la raíz:

```powershell
python -m src.train --preset A --device auto
```

Para repetir únicamente el diagnóstico:

```powershell
python -m src.train --preset A --smoke
```

Para preparar C, tras autorizar el recorrido completo de train:

```powershell
python -m calcular_stats --preset C --output models/stats_C_full.json
python -m src.train --preset C --stats models/stats_C_full.json --device auto
```

En este portátil sin activar el entorno, sustituir python por `& .\.env\Scripts\python.exe`. `python -m src.train` sin opciones conserva el entrenamiento normal de 25 épocas y no se ha ejecutado.
