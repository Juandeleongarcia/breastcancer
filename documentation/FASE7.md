# Fase 7 · auditoría y protocolo comparable

Sincronización: `main`, avance rápido a
`6e82ea311c4977e1de74dc0290f51db4694a1dd8`, commit de clase
«Fase 6: ajustes de CNN y configuraciones de entrenamiento». Su único cambio
fue ampliar `config/phase6/WEIGHT_DECAY.json` a 20 épocas / paciencia 8.
Se conserva ese archivo, todos los modelos, particiones y estadísticas.
Backups previos a las modificaciones en `backups/phase7_...` (excluidos de Git).
Fase 7 no hace commit/push ni entrenamiento de pacientes.

## Evidencia histórica y auditoría

| Referencia comunicada por el usuario | AUC paciente | Disponible como reproducción local |
|---|---:|---|
| F5 SPATIAL_UNWEIGHT | 0,6360 | No |
| F6 BASE_SPATIAL | 0,6306 | No |
| F6 REGIONAL_2X2 | 0,6011 | No |
| F6 WEIGHT_DECAY 0,002 | 0,5999 | No |
| SPATIAL_DROPOUT_020 | 0,5832 | No |

La familia espacial de dos convoluciones por bloque y pooling global es el control
más justificado por esas referencias. No prueban que cada diferencia cause una
mejora: son entrenamientos exploratorios con una semilla y sin incertidumbre.
No se asume que los pesos de clase estén aquí.

Sí hay resultados locales anteriores: F4_NO_WEIGHT AUC 0,552357 y
F6_CPU3_WEIGHT_DECAY AUC 0,611315 en época 2 (2 épocas completas, ejecución prevista
de 3 interrumpida por presupuesto). En este último, accuracy 0,704545 pero recall y
F1 cero, especificidad 1: todos los pacientes se clasificaron no pCR al umbral 0,5.
No confundir ese resultado CPU con la referencia de clase 0,5999. No se reevaluaron
sus imágenes ni pesos para esta auditoría. Los diagnósticos de memorización de cuatro
pacientes no son resultados comparables de generalización.

La implementación usa canales `[PRE, EARLY-PRE, LATE-PRE]`, escala inicial /255,
clipping por canal y normalización con estadísticas ajustadas en train. Las
configuraciones F7 conservan `config/phase3/stats/baseline_clipped.json`, comprobando
representación y hash de train (se admiten únicamente diferencias CRLF/LF).
Algunos configs F5 de clase apuntan a `models/stats_baseline_clase.json`, ausente
localmente. No podemos certificar la identidad numérica con ese archivo ausente ni
reproducir exactamente su AUC; el nuevo control fija las estadísticas verificables
del repositorio para las cuatro comparaciones. No se han recalculado ni sustituido
estadísticas. Guardar también las stats originales de clase si se desea esa réplica.

La BCE se calcula como media por corte, ponderando cada batch por su número real
de ejemplos, incluido el último batch incompleto. En estos cuatro modelos es **sin
pérdida ponderada**. El AUC por corte usa sus probabilidades continuas; por paciente,
la **media aritmética de probabilidades** de todos sus cortes. Los pacientes tienen
el mismo peso en el AUC paciente, aunque tengan distinto número de cortes.
La optimización de BCE por corte sí da más peso a pacientes con más cortes.

El mejor checkpoint se selecciona por mejora estricta del AUC paciente de validación
(primer máximo en caso de empate). Early stopping cuenta 8 épocas consecutivas sin
mejora; ni validation loss ni accuracy seleccionan el modelo. Umbral fijo 0,5 para
precision/recall/F1/especificidad; no se ajusta para alterar AUC. Recall = sensibilidad.
Se conservan precisión, recall, F1, sensibilidad y especificidad por corte y paciente.

Se comprueba que train y validación no compartan pacientes y procedan exclusivamente
del train público; no se utiliza test. La normalización firmada verifica procedencia,
pero no sustituye una auditoría externa de duplicados anatómicos entre identificadores.
La selección repetida sobre los mismos 220 pacientes puede sobreajustar la validación;
reservar test para la evaluación final independiente, fuera de este flujo.

## Evaluación al final de época y lectura de curvas

`train_loss` y `train_patient_*` siguen siendo **online**, con pesos cambiantes,
BatchNorm/dropout en train y augmentation si se activó. Se mantienen sus columnas.
Con `EVALUATE_TRAIN=True` se evalúan **todos** los cortes del conjunto de entrenamiento
seleccionado, usando un dataset separado sin augmentation, `model.eval()` y
`torch.inference_mode()`, el mismo criterio BCE y agregación que validación.
No hay pasos del optimizador, gradientes nuevos ni actualización de BatchNorm.
Se restaura RNG de Python/NumPy/PyTorch/loader y el modo del modelo para evitar que
la observación cambie la siguiente época. Los tests verifican también esa trayectoria.

Se añaden `train_eval_loss`, `train_eval_patient_*`, `train_eval_slice_*`,
`train_eval_seconds` y `loss_gap = val_loss - train_eval_loss`. Los estados completos
incluyen `train_eval_metrics`. Los experimentos reales F7 evalúan 877 pacientes,
8.745 cortes; fixtures sintéticas y diagnósticos parciales no acreditan generalización.
Nunca se reconstruyen métricas ausentes de históricos.

`loss_curve.png` muestra online, train eval y validación. `loss_comparison.png`
contiene solo train eval/validación en los mismos ejes. Accuracy y AUC muestran
train eval/validación **por paciente**; las columnas online siguen en history.
La ROC corresponde a validación paciente del mejor checkpoint. Los históricos sin
train eval mantienen sus gráficas claramente etiquetadas online; no se inventa una
línea comparable. No se suavizan curvas ni se eliminan valores difíciles.

Si train eval loss baja mientras val loss sube, aumenta la evidencia de sobreajuste
o de probabilidades mal calibradas; una separación con train online por sí sola no
lo demuestra, porque dropout/BatchNorm y augmentación cambian el protocolo.
AUC puede mejorar con peor BCE si mejora el orden de scores pero aumenta su exceso
de confianza. Revisar histogramas y fracción de predicciones positivas en internals,
comparando accuracy con la prevalencia mayoritaria: 155/220 = 70,45 % en validación.

Coste: cada época añade una pasada forward completa de train: 8.745 cortes, unas
**3,975 veces** la inferencia de validación (2.200 cortes), más carga de imágenes y
agregación. No añade backward. El coste real depende del I/O, batch y GPU; no se
extrapolan segundos de pruebas CPU sintéticas a la RX 6700 XT. Se registra
`train_eval_seconds` y se incluye en `epoch_time_seconds`. Medir la primera época
del control antes de planificar los cuatro runs. Priorizar ejecuciones individuales
si el tiempo de clase no alcanza.

## Cuatro comparaciones controladas

Común: entrada 3×256×256, BN/ReLU, dos convoluciones por bloque, pooling global 1×1,
dropout 0,40, AdamW LR 0,0005, weight decay 0,0005, BCE no ponderada, seed 42,
batch 16, máximo 20 épocas, early stopping 8 por AUC paciente, mismas particiones
y stats. No hay pesos preentrenados ni variables clínicas.

| Configuración | Única diferencia principal frente al control | Hipótesis |
|---|---|---|
| SPATIAL_CONTROL | Ninguna; canales [32,64,128] | Referencia con métricas comparables |
| SPATIAL_GENTLE_AUG | Activar gentle_affine sincronizado | Reducir memorización de posición |
| SPATIAL_SMALL | Canales [16,32,64] | Reducir capacidad sin cambiar estructura |
| SPATIAL_SCHEDULER | Plateau patience 2 → 3 | Evitar reducciones prematuras por AUC ruidoso |

El control tiene 288.033 parámetros y SMALL 72.593. Las dimensiones del
clasificador y gradientes finitos se verifican con entradas sintéticas 3×256×256.

Gentle affine: probabilidad 0,8, rotación ±5°, traslación ±2 %, escala ±2 %,
una misma matriz para los tres canales, interpolación bilineal/reflexión, antes del
clipping/normalización. Sin flips ni transformaciones fotométricas; interpolar puede
modificar valores locales y se considera parte de este experimento. Validación y
train eval permanecen deterministas y sin augmentation.

Scheduler de todos: ReduceLROnPlateau, mode=max sobre AUC paciente, factor 0,5,
min_lr 1e-6, threshold 0. Solo el cuarto aumenta paciencia a 3; puede retrasar una
reducción útil, no se garantiza mayor estabilidad ni AUC. Early stopping 8 permite
observar al menos parte de la respuesta a reducciones antes de detenerse.

Ejecutar **control primero**, luego augmentación y SMALL; scheduler después para
interpretar su trayectoria frente al control. Comparar loss gap y recall además de
AUC y conservar el mejor checkpoint por AUC aunque otra época tenga menor BCE.
La tabla automática incorpora train eval loss y gap de la mejor época; el historial
permite estudiar todas las épocas. No hay nuevas métricas clínicas F7 todavía.

## Compatibilidad y pruebas

Cada época completa guarda modelo/optimizador/scheduler/RNG/historial y estados de
selección. Reanudar crea otra carpeta y conserva los históricos. La migración admite
solo hashes de fuentes anteriores auditados, sin relajar contratos de arquitectura,
normalización o particiones. Un checkpoint anterior sin `EVALUATE_TRAIN` conserva
False; activar True requiere un experimento nuevo desde cero.

Comandos y organización de RUN en [GPU_FASE7.md](../GPU_FASE7.md).
Las pruebas sintéticas cubren invariancia de pesos, gradientes, BN y RNG, BCE
ponderada con último batch incompleto, cinco gráficos, estados por época y
reanudación exacta. La ejecución nativa ROCm/RX 6700 XT queda pendiente en clase;
no se ha reinstalado PyTorch ni alterado el entorno.

Verificación local final: **46 tests superados**, preflight de las cuatro CNN
sintéticas superado y prueba persistente de dos épocas sintéticas con las cinco
gráficas y checkpoints completos. No son métricas de pacientes. También se comprobó
la compatibilidad del contrato del checkpoint local real CPU3 de época 2, sin
reanudarlo ni reevaluar pacientes. Los 766 archivos protegidos de runs, metadatos y
configs F5/F6 conservan sus hashes. Los logs y la evidencia quedan en el backup
local excluido, no en nuevos runs clínicos.

Archivos de esta entrega: `src/experiment.py`, `src/metrics.py`,
`src/configuration.py`, `src/checkpointing.py`, `src/layout_compatibility.json`,
`src/check_suite.py`, `src/compare_runs.py`, los cuatro JSON y suite de
`config/phase7/`, `tests/test_phase7.py`, esta auditoría, `GPU_FASE7.md` y el enlace
de README. Se ajustaron además `src/phase6.py`, `tests/test_phase6.py` y
`tests/test_phase6_cpu3.py` para reconocer el protocolo WEIGHT_DECAY del commit de
clase; no se editaron arquitecturas, preprocesamiento ni configuraciones F5/F6.
