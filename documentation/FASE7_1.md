# Fase 7.1 · auditoría y diagnóstico, 27 de octubre de 2026

## Sincronización y referencia disponible

`main` estaba limpio; fetch y pull --ff-only confirmaron que no había commits
pendientes. HEAD: `dd91c802bd03fdce9d84f90552fd4de0c4f9d58c`. Las cuatro configuraciones
Fase 7 están disponibles e intactas. Se hicieron backups antes de trabajar.
El diagnóstico se realizó sin commit/push; posteriormente se autorizó publicar
la documentación para cerrar Fase 7.1.

Referencia de clase comunicada: SPATIAL_CONTROL completó 10 épocas de un máximo de
20, mejor AUC paciente 0,6306 en época 2, early stopping 8, train loss online bajando
y validation loss inestable/empeorando. No hay historial ni checkpoint de ese RUN
aquí: no se ha reproducido, no se conocen sus train eval loss/LR por época y no
se reconstruyen esos números. Mejorar frente a 0,6306 requerirá validación completa
y el mismo contrato; un diagnóstico de ocho pacientes no establece superioridad.

## Hallazgos verificables del pipeline

- Train: 877 pacientes (620 no pCR / 257 pCR), 8.745 cortes (6.179 / 2.566).
  Entre 5 y 10 cortes por paciente. Validación: 220 pacientes (155 / 65), 2.200
  cortes, 10 por paciente. Sin sample_id duplicados, pacientes comunes ni rutas
  de imágenes compartidas entre las dos particiones. No se utilizó test para
  entrenamiento, evaluación o selección.
- Las rutas de PRE/EARLY/LATE conservan identificador de paciente y fase. No se
  acredita con ello registro anatómico perfecto ni ausencia de duplicados de píxeles
  entre identificadores distintos; eso requeriría una auditoría adicional.
- Se cargan PNG en gris, exactamente 256×256 (se rechazan otros tamaños), /255,
  canales `[PRE, EARLY-PRE, LATE-PRE]`. Antes de clipping, la representación permite
  recuperar EARLY/LATE sumando PRE: no elimina algebraicamente las dos diferencias
  temporales. No aporta tiempos de adquisición ni señal cuantitativa DICOM original.
- Clipping P01/P99 por canal y z-score se mantienen. Las estadísticas de
  `config/phase3/stats/baseline_clipped.json` tienen hash de train y representación
  verificados; se aplican también a validación, sin ajustar estadísticas allí.
  Los límites superiores y los dos límites de enhancement saturan aproximadamente
  el 1 % de valores según el perfil guardado; la masa PRE=0 no significa que se haya
  eliminado ese porcentaje de señal. No hay evidencia nueva para cambiar clipping.
- Gentle affine aplica una misma matriz a los tres canales, antes del clipping/
  estandarización: bilineal, borde por reflexión, ±5°, ±2 % traslación/escala,
  probabilidad 0,8. No flips ni modificaciones fotométricas. La interpolación cambia
  valores locales; no garantiza invariancia física del contraste, pero preserva
  la correspondencia geométrica. Solo se activa en entrenamiento online.
- CNN propia espacial: dos Conv+BN+ReLU por bloque, MaxPool, AdaptiveAvgPool 1×1,
  Dropout 0,4 y una salida logit, sin sigmoid antes de BCEWithLogitsLoss. Control/
  gentle: 288.033 parámetros; SMALL [16,32,64]: 72.593. Máximo 128 filtros.
- BCE sin ponderar, reducción media por corte; batches se acumulan ponderando por
  su número real de cortes. Pacientes con más cortes pesan más en la optimización;
  son muestras correlacionadas, 8.745 cortes no equivalen a 8.745 pacientes.
- AUC corte: probabilidades continuas. AUC paciente: media de probabilidades de
  todos sus cortes, un registro por paciente. No se promedian AUC de batches ni
  se calcula AUC a partir de etiquetas umbralizadas.
- AdamW LR 0,0005 / WD 0,0005. ReduceLROnPlateau observa AUC paciente, mode=max,
  factor 0,5, patience 2 (3 solo en SCHEDULER), min_lr 1e-6. Historial conserva LR
  usado y siguiente LR. No se puede atribuir la oscilación de clase a LR sin su
  historial: un AUC ruidoso con 65 positivos también puede producir oscilaciones.
- Mejor checkpoint por mejora estricta de AUC paciente. Early stopping cuenta ocho
  épocas sin mejora. Mejor época 2 y parada en época 10 son compatibles con ese
  protocolo; no prueban por sí solos un fallo de entrenamiento.

## Curvas: ya correctas, sin reescritura

Train online usa pesos cambiantes, dropout/BN en train y augmentation opcional.
Train eval y validación utilizan los mismos pesos finales, criterio y reducción,
modo eval, sin augmentation ni actualizaciones de BatchNorm. Train eval usa
inference_mode y restaura estados aleatorios; los tests verifican pesos, gradientes,
BN, RNG y equivalencia de la trayectoria con/sin esa observación.

`loss_curve.png`: online + train eval + validación. `loss_comparison.png`: solo
train eval y validación en un mismo eje vertical, sin suavizado. Accuracy/AUC:
train eval y validación por paciente; las columnas online se conservan en history.
ROC: validación paciente del mejor checkpoint. No se reconstruyen train eval
inexistentes en los históricos. Se conserva el código de Fase 7 sin cambios.

## Interpretación científica y prioridades

No se ha identificado un error metodológico concreto que explique por sí solo la
baja discriminación. La alta accuracy puede corresponder al predominio no pCR:
70,45 % sin detectar ningún pCR en validación completa. Examinar siempre recall,
F1, especificidad y distribuciones por clase. En el CPU3 histórico local, por
ejemplo, AUC 0,611315 coexistió con recall/F1 cero; ese resultado es distinto del
SPATIAL_CONTROL de clase.

El sobreajuste se estudia con train eval frente a validación, no con online frente
a validación. Un AUC razonable y BCE creciente pueden coexistir: el orden mejora
mientras las probabilidades se vuelven excesivamente confiadas. No demuestra que
la calibración sea la única causa. Otros riesgos: etiquetas por paciente aplicadas
a cortes con señal tumoral variable, dilución al promediar cortes, pocos pacientes
efectivos y selección repetida sobre la misma validación. Son hipótesis, no hallazgos
anatómicos confirmados. No hay anotaciones de tumor verificadas en esta auditoría.

Prioridad para clase: GENTLE_AUG y luego SMALL. El primero contrasta sensibilidad
a posición manteniendo capacidad; el segundo capacidad manteniendo geometría y
regularización. SCHEDULER después, si esas trayectorias no mejoran. No añadir otra
CNN ni cambiar varias variables sin evidencia. Un AUC mayor aislado tampoco basta:
revisar pérdidas, recall, confianza y persistencia a lo largo de las épocas.

Las configuraciones originales conservan máximo 20 épocas, paciencia 8, semilla 42,
batch 16 y selección por AUC paciente, sin test. No se elige automáticamente un
ganador a partir de las pruebas pequeñas. Comandos exactos:
[GPU_FASE7_1.md](../GPU_FASE7_1.md).

## Evidencias locales

Backups, auditoría de particiones, logs de tests y ejecutor local de diagnóstico:
`backups/phase71_20261027_965688cc/` (excluido de Git). Los diagnósticos quedan dentro
de una campaña `models/DIAGNOSTICS_PHASE71_.../`, no mezclados con RUNs completos.
Cada run se marca `diagnostic_partial=True` y conserva selección exacta de pacientes,
config, semilla, checkpoints completos y cinco gráficas.

## Pruebas CPU ejecutadas: resultados diagnósticos, NO validación completa

Entorno local: PyTorch 2.14.0+cpu, sin GPU accesible por PyTorch. Preflight de las
cuatro configuraciones superado con forward/backward sintético. Las imágenes y CSV
están disponibles. Los subconjuntos son los mismos para los tres modelos: **8
pacientes / 80 cortes train y 8 / 80 validación**, cuatro no pCR y cuatro pCR en
cada partición, seleccionados con seed 42. Se conservaron todos los cortes de esos
pacientes y las particiones originales. Las stats siguen siendo las del train
completo, sin ajuste en la pequeña validación. No se balancea la validación real.

Benchmark descartable: dos batches online, dos de validación y todo train eval;
extrapolación al subconjunto con margen del 50 % +12 s por época para admitir runs.
No se usan sus métricas como resultados. Se admitieron los tres ensayos de tres
épocas con margen respecto al límite global de 20 minutos.

| Modelo | Estimación por época del subconjunto | Reserva estimada para 3 épocas | Épocas realizadas |
|---|---:|---:|---:|
| CONTROL | 25,79 s | 152,05 s | 3 |
| GENTLE_AUG | 22,46 s | 137,07 s | 3 |
| SMALL | 9,59 s | 79,14 s | 3 |

Campaña de benchmark + entrenamientos + guardado: **195,15 s (3 min 15 s)**. Tiempo
sumado de las épocas (sin gráficos posteriores): 58,86 / 58,79 / 26,54 s. Train eval
añadió 12,60 / 12,61 / 5,93 s en las tres épocas respectivas, aproximadamente el 21–22 %
del tiempo de época en estos subconjuntos. No extrapolar estos segundos a ROCm ni a
una comparación clínica completa. No se iniciaron entrenamientos completos CPU.

Métricas del **checkpoint seleccionado por AUC de esos ocho pacientes**:

| Modelo | Mejor época / realizadas | Train online BCE | Train eval BCE | Val BCE | AUC paciente | AUC corte | Accuracy | Precision | Recall | F1 | Especificidad |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CONTROL | 3 / 3 | 0,2527 | 0,6778 | 0,7730 | 0,5000 | 0,5006 | 0,5000 | 0 | 0 | 0 | 1 |
| GENTLE_AUG | 1 / 3 | 0,5512 | 0,6880 | 0,6947 | 0,3125 | 0,3575 | 0,5000 | 0 | 0 | 0 | 1 |
| SMALL | 1 / 3 | 0,5848 | 0,6921 | 0,6948 | 0,5000 | 0,4563 | 0,5000 | 0 | 0 | 0 | 1 |

El primer máximo se conserva en empates; por eso GENTLE_AUG y SMALL seleccionan
época 1. No mezclar pérdidas de última época con métricas del mejor checkpoint.
Recall y sensibilidad son la misma métrica. Al umbral fijo 0,5 todos los mejores
checkpoints predicen no pCR en validación diagnóstica: TN=4, FP=0, FN=4, TP=0.
En un subconjunto equilibrado esto da 50 %, no el 70,45 % de la validación completa.

Evolución completa, sin suavizado:

| Modelo | Época | Online BCE | Train eval BCE | Val BCE | AUC paciente |
|---|---:|---:|---:|---:|---:|
| CONTROL | 1 | 0,5574 | 0,6902 | 0,6951 | 0,3750 |
| CONTROL | 2 | 0,3650 | 0,6797 | 0,7219 | 0,3750 |
| CONTROL | 3 | 0,2527 | 0,6778 | 0,7730 | 0,5000 |
| GENTLE_AUG | 1 | 0,5512 | 0,6880 | 0,6947 | 0,3125 |
| GENTLE_AUG | 2 | 0,3689 | 0,6545 | 0,7133 | 0,3125 |
| GENTLE_AUG | 3 | 0,2577 | 0,6322 | 0,7838 | 0,3125 |
| SMALL | 1 | 0,5848 | 0,6921 | 0,6948 | 0,5000 |
| SMALL | 2 | 0,4595 | 0,6839 | 0,7028 | 0,3750 |
| SMALL | 3 | 0,3704 | 0,6688 | 0,7261 | 0,3750 |

En todos baja train eval BCE y sube val BCE: separación compatible con memorización
temprana de estos pacientes. LR se mantuvo en 0,0005 durante las nueve épocas:
la separación observada aquí no necesita un salto del scheduler para aparecer.
La validación tiene solo cuatro positivos y cuatro negativos: 16 pares, AUC con
pasos de 0,0625 salvo empates. No permite ordenar arquitecturas de forma fiable.

Las probabilidades de ambas clases se solapan. En los checkpoints seleccionados,
medias no pCR / pCR: CONTROL 0,3210 / 0,3185; GENTLE 0,4939 / 0,4925; SMALL
0,4804 / 0,4796. Histogramas, predicciones y errores se guardan en internals.
No se cambió el umbral ni se eligió un ganador automático. SMALL es más barato
aquí; no se ha demostrado que generalice mejor. GENTLE no muestra ventaja en estos
ocho pacientes; tampoco queda descartado para entrenamientos completos.

## Sonda controlada de BatchNorm, sin modificar checkpoints

Se tomó una **copia en memoria** del control diagnóstico de época 3. Sus seis BN
habían acumulado solo 15 batches de entrenamiento. Se recalibraron únicamente los
buffers de esa copia con una pasada de los 80 cortes train, sin augmentation ni
dropout, sin optimizador ni etiquetas usadas para actualizar BN; reset de buffers
y promedio acumulativo, mismo método del diagnóstico Fase 5. No se guardaron pesos
recalibrados ni se cambió la receta del entrenamiento. Se verificó igualdad de
parámetros aprendidos y hash del checkpoint original.

| Estado del control diagnóstico | Train eval BCE | Val BCE | AUC val paciente |
|---|---:|---:|---:|
| Checkpoint original | 0,6778 | 0,7730 | 0,5000 |
| Copia con buffers BN recalibrados en train | 0,1921 | 0,8390 | 0,4375 |

La copia obtiene AUC **de train** 1,0 y accuracy train 1,0: evidencia de que este
modelo sí aprendió a memorizar esos ocho pacientes y de que la estimación temprana
de BN afecta la loss observada en eval. **No es AUC de validación ni una mejora
clínica.** La recalibración empeora BCE/AUC de la pequeña validación; recall sube a
0,75 pero especificidad baja a 0,25 (TN=1, FP=3, FN=1, TP=3). No se adopta como
mejora ni se prepara otra arquitectura a partir de ello. BN funciona según su
protocolo; no se demuestra que sea la causa del resultado de clase con 877 pacientes.
Una sola pasada recalibrada sigue siendo una estimación aproximada de sus buffers.

La sonda finalizó 297,43 s después del inicio de la campaña, todavía muy por debajo
de 20 minutos incluyendo los intervalos de revisión. Evidencia técnica:
`bn_probe_DIAGNOSTIC_ONLY.json` dentro de la campaña.

Resultados completos locales:
`models/DIAGNOSTICS_PHASE71_20261027_223710_002d76043530/`.
No se comparan los AUC diagnósticos anteriores con el 0,6306 de clase.

## Validación final y alcance de los cambios

Suite existente: **46/46 tests al inicio y 46/46 al final**, sin fallos. Cubren
configuraciones, reducción de pérdidas, no actualización de pesos/BN/RNG durante
train eval, curvas, compatibilidad histórica y reanudación exacta sintética.
Además se verificaron los tres RUNs reales diagnósticos: tres checkpoints íntegros
por modelo, best_epoch, cinco PNG, history de tres épocas, 80 cortes/8 pacientes
en cada evaluación y contrato reanudable desde época 1.

No se modificó código, tests, configuraciones, stats, arquitectura ni preprocesamiento.
Se añaden este informe y GPU_FASE7_1.md; se corrigen las indicaciones obsoletas de
publicación/prioridad en GPU_FASE7.md. Los resultados y backups son exclusivamente
locales e ignorados por Git. La publicación autorizada contiene solo documentación;
no se instalaron dependencias ni se publicaron pesos o resultados pesados.
El soporte efectivo ROCm en Linux queda pendiente del preflight de clase.
Integridad final: los hashes de **799 archivos históricos/configuraciones/metadatos**
y **38.109 archivos originales de imágenes** permanecen iguales. El hash de imágenes
es únicamente una comprobación de integridad; no se usaron datos de test para
evaluación ni decisiones experimentales.
