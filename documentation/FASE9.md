# Fase 9 · validación cruzada oficial, sin tocar el test reservado

## Origen y verificación de las asignaciones

Las asignaciones **ya estaban** en `metadata/samples.csv`, columna `fold`.
`utils_caso.py:particion(samples, fold_val)` selecciona `split=train`, usa
`fold != fold_val` para train y `fold == fold_val` para validación; test se excluye.
GUIA.md, secciones B3 y C1, confirma esa intención. No se genera ningún KFold nuevo.

Se comprobó la copia pública original referenciada por descargar_datos.py:
[metadata/samples.csv del dataset](https://storage.googleapis.com/usecasesf-breastdcedl-alumnos-8264/breastdcedl/metadata/samples.csv).
Respuesta HTTP 200; SHA256 de bytes original/local:
`7ddd0ecc8272e6460211bd63c1479513bf815e3ea518d1b6bafdcd69f9a75c03`.
Los índices, pacientes, etiquetas, rutas y folds coinciden exactamente. La copia
descargada de verificación está solo en el backup, no reemplaza archivos originales.
El protocolo fija el hash canónico CSV con CRLF→LF:
`db54c844573a764fa94a9439115b21f1a9989d7967057aace61809dc02330dca`, portable a Linux.

**La tabla orientativa de folds en GUIA.md no coincide con este CSV oficial.**
No se han editado los folds para obtener sus proporciones aproximadas. Prevalecen
las filas verificadas del índice público correspondiente a esta versión del dataset.

| Fold validación | Train pacientes | Train cortes | Val pacientes | Val cortes | Val no pCR | Val pCR | Cortes no pCR / pCR |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 878 | 8.759 | 219 | 2.186 | 155 | 64 | 1.546 / 640 |
| 1 | 878 | 8.755 | 219 | 2.190 | 157 | 62 | 1.570 / 620 |
| 2 | 877 | 8.753 | 220 | 2.192 | 161 | 59 | 1.606 / 586 |
| 3 | 877 | 8.753 | 220 | 2.192 | 145 | 75 | 1.442 / 750 |
| 4 | 878 | 8.760 | 219 | 2.185 | 157 | 62 | 1.565 / 620 |

Unión de validaciones: 1.097 pacientes, 10.945 cortes, 775 no pCR / 322 pCR.
Los 176 pacientes / 1.758 cortes de `split=test`, `fold=-1`, permanecen reservados:
123 no pCR / 53 pCR. No se interpretan como un sexto fold ni se usan para ajustar
normalización, arquitectura, pesos, umbrales o épocas. Las columnas clínicas de
patients.csv no son entradas de la CNN; solo se cotejan pid, etiqueta y split/test.

## Protocolo nuevo y experimentos anteriores

Las particiones `internal_train.csv` / `internal_val.csv` anteriores no representan
estos cinco folds. Se conservan intactas para interpretar los experimentos F2–F8.
Fase 9 usa CSV nuevos, derivados de samples.csv dentro de una campaña independiente.
No se comparará directamente una media CV con el AUC de aquel único holdout como si
fuera una mejora bajo la misma validación.

Inicialmente: SPATIAL_SMALL [16,32,64], dos convoluciones por bloque, BN/ReLU,
MaxPool, AdaptiveAvgPool 1×1, dropout 0,40. También se prepara SPATIAL_CONTROL
[32,64,128]. Se conservan sus arquitecturas propias, sin transfer learning y sin
superar 128 filtros. Cada fold crea un modelo y optimizador **nuevos**, sin compartir
pesos; semilla 42 común, AdamW LR 0,0005 / WD 0,0005, BCE sin ponderación, máximo
20 épocas, early stopping 8 por AUC paciente, scheduler plateau original. Son dos
campañas diferentes, no diez folds mezclados en un solo promedio.

Una paciente puede ser train en cuatro modelos y validación en el quinto, que es
la estructura oficial. Dentro de cada modelo, todos sus cortes permanecen juntos;
ninguna predicción OOF procede de un modelo que la incluyera en su train.

## Estadísticas sin fuga de información

No se reutilizan los perfiles Fase 3/7/8: se ajustaron sobre otro conjunto de train
y podrían incorporar pacientes del fold que ahora es validación. Las plantillas
F9 tienen `requires_fit` y no pueden entrenarse directamente sin ajustar un perfil.

Cada fold ajusta P01/P99 de clipping y medias/stds exclusivamente en sus cuatro
folds de train. Método: una pasada de todos los PNG de ese train, histogramas
exactos de 511 niveles para intensidades/diferencias de enteros 8 bit, cuantiles
lineales y momentos poblacionales float64 tras clipping. No se muestrean píxeles,
pacientes ni cortes en los perfiles completos, ni se abren imágenes de validación
o test durante su ajuste. Memoria acotada a histogramas y un corte.

La representación sigue siendo `[PRE, EARLY-PRE, LATE-PRE]` en escala /255. El
redondeo de la resta float32 del loader puede diferir en la última cifra de los
niveles expresados en float64; no se cambia el loader ni las imágenes. Tests
contrastan cuantiles/momentos contra una referencia completa de píxeles sintéticos.

Cada perfil guarda hash del CSV train, fold, número de pacientes/cortes, método y
preprocesamiento. Se verifica antes de aplicarlo. No se permite reutilizar un perfil
de otro fold aunque tenga el mismo número de pacientes. Los CSV derivados, stats,
configuración y fuentes se protegen con hashes. Una edición obliga a revisar o crear
otra campaña; no se continúa silenciosamente con otro protocolo.

La adaptación de calcular_stats.py no se hace debilitando su restricción histórica
al train interno. Se usa un ajustador CV separado, autorizado por el índice oficial
y por el fold retenido. Los scripts y estadísticas anteriores siguen intactos.

## Runs, métricas y reanudación

Cada campaña conserva índice de folds, auditoría, plantilla y ZIP de las fuentes.
Cada fold contiene CSV derivados, perfil y configuración efectiva, además de RUNs
con fecha/hora/UUID. Se mantiene el formato sencillo de F7: config.py, history.csv,
checkpoints completos por época, best_epoch.pt y cinco curvas; técnicos en internals.

Por época se conservan pérdidas online, train eval y val; AUC/accuracy por paciente
y corte; sensibilidad, especificidad y TN/FP/FN/TP. Se añade **balanced_accuracy** al
diccionario de métricas, por lo que pasa automáticamente al history y checkpoints.
Las pérdidas conservan la BCE media por corte del entrenamiento; no se etiquetan
como una loss por paciente. La evaluación principal de discriminación/clasificación
es por paciente después de agregar todos sus cortes.
No se eliminan aliases ni se reconstruyen columnas ausentes de los runs históricos.
La matriz por paciente de cada época queda en su checkpoint y se representa por
sus cuatro conteos en history; el PNG del mejor checkpoint está en internals.

Las curvas de AUC/accuracy comparan train eval y val paciente con pesos finales,
sin augmentation ni actualización de BN; la loss online permanece identificada
como protocolo distinto. Normalización idéntica en train eval y validación de ese
fold; AUC con scores continuos y clasificación al umbral fijo 0,5.

La única modificación al cálculo general de métricas es añadir balanced accuracy
(sensibilidad+especificidad)/2; loss, AUC, pesos, optimizador y selección no cambian.
La compatibilidad de checkpoints se amplía únicamente para el hash exacto de las
fuentes anteriores auditadas. No se relajan contratos de arquitectura, estadísticas,
particiones ni RNG. Un checkpoint de otro fold se rechaza por CSV/perfil distintos.

Un fold completo se omite al continuar. Un fold interrumpido exige reanudación
explícita desde un checkpoint íntegro o el recovery de inicio de época; los pesos
de media época no se usan como continuación válida. Se conservan optimizador,
scheduler, historial, semilla/RNG y contador de early stopping, en otro RUN. No se
resetea el contador de un fold que ya terminó por early stopping.

## Resumen de cinco folds y OOF

Tras cada fold se genera otra tabla y resumen con nombres únicos, conservando los
previos. La media es no ponderada entre AUC de folds completos de la misma campaña;
la desviación es muestral (`ddof=1`). Antes de completar los cinco, se marca provisional.
Esa dispersión describe variación entre folds; no es por sí sola un intervalo de
confianza y los conjuntos de entrenamiento se solapan entre folds.

El AUC OOF se calcula solo con cinco folds completos, no diagnósticos, y cobertura
exacta de las 1.097 pacientes de train. Cada paciente aporta su probabilidad media
de cortes en el fold que la excluyó, una sola vez. Se verifican fold y etiqueta;
duplicados, ausencias o pacientes de test bloquean el cálculo. Se guardan CSV,
métricas/matriz OOF y ROC OOF. No se mezclan scores de SMALL con los de CONTROL.

El promedio de AUC y el AUC pooled OOF son distintos: el segundo compara también
scores entre modelos con posibles escalas de probabilidad diferentes. Informar ambos,
sin exigir igualdad. Además, el mejor checkpoint de cada fold se eligió con etiquetas
de esa misma validación. Por tanto son **resultados de desarrollo CV con posible
optimismo por selección**, no una estimación externa independiente. El test reservado
sigue sin tocarse para una evaluación final de un procedimiento previamente fijado.

## Uso y validación

Comandos exactos de Linux/ROCm y conda cancer: [GPU_FASE9.md](../GPU_FASE9.md).
La preparación de CSV no entrena ni ajusta estadísticas completas; el run GPU ajusta
el perfil justo antes de entrenar cada fold. Entrenamientos CV completos se bloquean
en CPU. La disponibilidad real de ROCm se comprobará en clase, sin reinstalar PyTorch.

Los tests incluyen auditoría de folds/test, pertenencia de pacientes, cuantiles,
estadísticas solo train, incompatibilidad de perfiles, OOF incompleto/duplicado,
media/desviación y agregación con datos sintéticos, bloqueo CPU y reanudación tras
interrupción. Los fixtures numéricos fabricados para probar el agregador no son
resultados de CNN ni se guardan como resultados clínicos.

El smoke real usa únicamente fold 0, una paciente de cada clase en train y en val,
todos sus cortes y dos épocas de SMALL. Su campaña lleva SMOKE y diagnostic=True;
sus métricas no participan en medias CV ni AUC OOF clínico. No se ejecutan los cinco
entrenamientos completos en el ordenador personal. La publicación del código,
configuraciones, tests y documentación se autoriza tras la verificación de cierre;
los datos, pesos y resultados pesados permanecen locales.

Verificación realizada: **68 tests superados**, sin fallos, y smoke real completado
con **2 pacientes/20 cortes de train y 2 pacientes/20 cortes de validación**, dos
épocas. Checkpoints completos, cinco curvas y balanced accuracy por época verificados.
El contrato de reanudación de un checkpoint histórico real también se validó sin
entrenarlo. Permanecen iguales los hashes de **894 archivos históricos/configs/
metadatos** y **38.109 imágenes originales**. Logs y comprobación del índice público
en backups/phase9_20261028_1b282f8b/ (excluido de Git). ROCm nativo queda pendiente
de verificar en clase. El AUC del smoke no es un resultado clínico ni de cinco folds.

Comprobación de ejecución progresiva sin entrenar: el entrenador ya admitía
`--folds 0 1 2` y posteriormente `--folds 3 4`. Una simulación aislada con la llamada
a entrenamiento sustituida verificó que solicitar de nuevo 0–4 solo ejecuta 3–4,
y repetir 3–4 no lanza nada; se preservan las rutas de los completados. No se
modificó el entrenador para añadir una función que ya existía.
