# Fase 5 — capacidad de aprendizaje y generalización

20 de octubre de 2026. Campaña cerrada con entrenamientos completos y diagnósticos técnicos separados.

## Alcance y conservación

Solo imágenes DCE-MRI y etiquetas pCR/no pCR, sin variables clínicas ni modelos preentrenados. Se conservan los 877 pacientes/8.745 cortes de train y 220 pacientes/2.200 cortes de validación interna, sin utilizar test ni cambiar particiones. Los modelos anteriores no se renombran ni sobrescriben. Backup previo: `backups/phase5_20261020_205310_a5fa9f33/`. Campaña: `models/PHASE5_20261020_205310_a5fa9f33/`.

La CNN original [32,64,128] mantiene sus capas, claves y 93.825 parámetros. La variante espacial, ya disponible desde Fase 4, tiene dos convoluciones por bloque y 288.033 parámetros. No se modifica ninguna de estas arquitecturas en Fase 5.

## Diagnóstico controlado de memorización

Se fijaron mediante semilla 42 dos pacientes de cada clase, exclusivamente de train. Se incluyeron TODOS sus cortes: cuatro pacientes, 40 cortes. Los mismos identificadores y cortes se utilizaron en las tres pruebas. Se mantuvieron dropout 0,4, LR 0,0005 y BCE sin ponderación en todas; no se introdujo augmentation. No son entrenamientos completos ni métricas de validación.

El criterio técnico exige AUC paciente ≥0,99, accuracy por corte ≥0,90 y BCE de evaluación <0,25 en dos comprobaciones consecutivas. Un AUC alto en cuatro pacientes no es evidencia de generalización: la reducción de BCE y la clasificación correcta de todos sus cortes son la comprobación de capacidad de aprendizaje.

| Prueba de train | Variable frente al diagnóstico baseline | Épocas | BCE inicial | BCE final | Accuracy paciente/corte | AUC paciente/corte |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Referencia | 15 | 0,693266 | 0,140883 | 1,0 / 1,0 | 1,0 / 1,0 |
| Sin BatchNorm | Solo elimina BatchNorm | 25 | 0,693266 | 0,067503 | 1,0 / 1,0 | 1,0 / 1,0 |
| Espacial | Solo dos convoluciones por bloque | 15 | 0,693276 | 0,045677 | 1,0 / 1,0 | 1,0 / 1,0 |

La regla de parada no iguala el número de épocas ni la BCE final, por lo que no se presenta esta tabla como un ranking de arquitecturas. La espacial consume más CPU. Las tres redes son capaces de aprender esos ejemplos; esto no demuestra que las etiquetas clínicas sean correctas ni que puedan generalizar a pacientes nuevos.

Los gradientes registrados por capa son finitos y no nulos. Los rangos de norma de pesos convolucionales/BatchNorm registrados son aproximadamente 0,0051–1,1538 en baseline, 0,0199–0,7922 sin BN y 0,0020–1,8956 en espacial. No se interpretan los gradientes casi nulos de ciertos sesgos antes de BN como un fallo: BN puede hacerlos redundantes.

En baseline y espacial, ninguno de los 128 canales globales tiene varianza inferior a 1e-6 en esos cortes; sin BN, aproximadamente 14,84% no varía en ese conjunto. No significa que sean canales muertos en todo el dataset. No aparece un colapso total de características que impida memorizar.

El rango efectivo de la matriz de características globales del baseline es 1,27; con pooling espacial 4×4 para inspección es 2,63. La dimensión y la operación cambian, por lo que esto no prueba que el pooling sea la causa del mal AUC. La separación binaria puede concentrarse en una sola dirección. El pooling global no impide la memorización, aunque podría limitar información espacial útil para generalizar.

En el baseline memorizado, usar estadísticas de batch con dropout desactivado da BCE 0,117984 y recalibrar BN solo con los mismos pacientes de train da 0,113454; AUC y accuracy siguen siendo 1,0. Esta prueba pequeña no justifica eliminar BN ni descarta un desajuste de sus estadísticas en la población completa.

## Etiquetas y particiones

La auditoría encontró cero pacientes compartidos entre train/validación, cero etiquetas contradictorias dentro de un paciente, cero tripletas de rutas compartidas entre las particiones y cero tripletas de rutas con etiquetas contradictorias. No se han auditado píxeles idénticos bajo rutas diferentes ni la veracidad de la respuesta clínica original.

También se comprobó la correspondencia paciente/corte/fase en 32.835 rutas de los 10.945 ejemplos internos: cero discrepancias entre carpeta de paciente, sample_id y sufijo PRE/EARLY/LATE. Esto comprueba el enlace por nombres, no el registro espacial de las imágenes.

Train contiene 620 pacientes no pCR y 257 pCR; validación, 155 y 65. En train, 871 pacientes tienen diez cortes, dos tienen cinco, tres tienen seis y uno tiene siete; los 220 pacientes de validación tienen diez. Esta pequeña desigualdad de cortes no basta para explicar por sí sola el AUC casi aleatorio.

## Preprocesamiento y augmentation

El nuevo modo `gentle_affine` aplica una sola geometría y una misma interpolación bilineal a los tres canales: probabilidad 0,8, rotación ±5°, traslación hasta 2% y escala ±2%, con borde reflejado. No introduce cambios fotométricos ni volteos verticales. Se aplica antes del clipping y el z-score; el dataset de validación no consume RNG al cargar una muestra ni aplica geometría. La opción histórica de volteos se conserva para compatibilidad y no es el protocolo nuevo.

Las pruebas verifican que la transformación común conserva las relaciones lineales entre fases y realces antes del clipping, que validación es determinista y que una continuación con augmentation reproduce exactamente pesos y estado RNG en CPU. La interpolación cambia valores locales al mover la imagen; no se promete identidad píxel a píxel.

La comparación RAW conserva [PRE,EARLY,LATE] frente a [PRE,EARLY−PRE,LATE−PRE], sin clipping en ambos casos. Son representaciones invertibles de la misma información antes de las transformaciones afines globales. La comparación estudia el condicionamiento y el aprendizaje, no una ganancia artificial de información. Cada representación usa su perfil de medias/desviaciones ajustado exclusivamente con todo train en Fase 3 y verificado por hash.

## Entrenamiento, ejecutor y pruebas

Los runs completos usan dos épocas, semilla 42, batch 16 y cuatro hilos CPU. AUC principal: media de probabilidades de TODOS los cortes de cada paciente de validación interna. Umbral fijo 0,5; no se optimiza para modificar el AUC. Las métricas online de train no equivalen a una reevaluación con pesos finales.

La suite inicial de 24 pruebas unitarias y de compatibilidad pasó antes de entrenar. Tras revisar el ejecutor, se añadió una regresión para lotes completamente pendientes: hay 25 pruebas distintas verificadas. La revisión final vuelve a comprobar la suite Fase 5 y el caso de errores/omisión/continuación del lote. Las salidas están en `unittest.log`, `unittest_phase5_final.log` y `unittest_batch_final.log` de la campaña. Incluyen carga estricta de la CNN histórica, gradientes espaciales, métricas continuas y agrupación por paciente, contratos de estadísticas, reanudación exacta con/sin augmentation y validación determinista. El ejecutor rechaza combinar inicio forzado desde cero con reanudación y conserva también resultados y comparación vacía cuando todo queda pendiente.

La incorporación del código de augmentation amplía las huellas y parámetros del contrato de continuación. No se fuerza la reanudación de checkpoints Fase 4 con código diferente: se conservan sus pesos y sus fuentes congeladas para evaluación o continuación con el entorno original. Para mañana se preparan experimentos nuevos Fase 5; no se sustituye el PyTorch GPU de clase.

Guía de comandos actual: [GPU_FASE5.md](GPU_FASE5.md). Batería principal de tres configuraciones y pareja opcional RAW en `config/phase5/gpu_suite.json`. Todas usan hasta diez épocas y un protocolo GPU común de scheduler y early stopping por AUC de paciente.

## Resultados completos verificados

Todos los nuevos entrenamientos de la tabla usan dos épocas completas, toda la validación interna y una semilla. Accuracy, recall, especificidad y F1 son por paciente; loss es BCE por corte.

| Experimento | Referencia | Mejor época | AUC paciente | AUC corte | Accuracy | Val loss | Recall | Especificidad | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AUG_UNWEIGHT | F4_NO_WEIGHT_PC | 2 | 0.545806 | 0.546556 | 0.704545 | 0.606774 | 0.000000 | 1.000000 | 0.000000 |
| RAW_WEIGHTED_NOCLIP | F4_NO_CLIP_PC | 2 | 0.534591 | 0.533105 | 0.331818 | 1.003108 | 0.907692 | 0.090323 | 0.445283 |

Los dos nuevos runs pertenecen a familias distintas: no se atribuyen sus diferencias entre sí únicamente a augmentation o representación. Cada uno se compara descriptivamente con su referencia de Fase 4. Las loss ponderada y no ponderada tienen distinta escala.

### Evolución por época

| Experimento | Época | Train loss online | Val loss | AUC paciente | AUC corte | Accuracy | Recall | Especificidad | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUG_UNWEIGHT | 1 | 0.606499 | 0.614944 | 0.505310 | 0.506542 | 0.704545 | 0.000000 | 1.000000 | 0.000000 |
| AUG_UNWEIGHT | 2 | 0.603497 | 0.606774 | 0.545806 | 0.546556 | 0.704545 | 0.000000 | 1.000000 | 0.000000 |
| RAW_WEIGHTED_NOCLIP | 1 | 0.981781 | 0.997775 | 0.494491 | 0.493704 | 0.381818 | 0.846154 | 0.187097 | 0.447154 |
| RAW_WEIGHTED_NOCLIP | 2 | 0.972826 | 1.003108 | 0.534591 | 0.533105 | 0.331818 | 0.907692 | 0.090323 | 0.445283 |

### Distribuciones y errores

**AUG_UNWEIGHT**: AUC paciente 0.545806, diferencia observada -0.006551 frente a su referencia. Matriz [[TN,FP],[FN,TP]]=[[155, 0], [65, 0]]. Puntuaciones no pCR: media 0.323547, desviación 0.042922; pCR: media 0.330629, desviación 0.042375. Fracción predicha pCR 0.0000. 65 errores de paciente guardados con identificadores.

**RAW_WEIGHTED_NOCLIP**: AUC paciente 0.534591, diferencia observada -0.006650 frente a su referencia. Matriz [[TN,FP],[FN,TP]]=[[14, 141], [6, 59]]. Puntuaciones no pCR: media 0.571352, desviación 0.059013; pCR: media 0.574620, desviación 0.066828. Fracción predicha pCR 0.9091. 147 errores de paciente guardados con identificadores.

### Referencias constantes y aprendizaje más allá de la prevalencia

Un predictor que devuelve en todos los cortes la prevalencia de train (0.293425) obtiene AUC 0,5, accuracy 70,45%, recall 0 y BCE de validación 0.606974. No se ha ajustado esa constante con validación. El run con augmentation tiene BCE 0,606774: una mejora de solo 0,000199 frente a esa referencia. El mejor no ponderado de Fase 4 tenía 0,605551, también cercano a la referencia constante.

Con la ponderación original, el óptimo constante de train es 0,5; su loss ponderada de validación es 0.981502, con AUC 0,5. La loss absoluta de una familia no se compara con la de la otra. Estos controles muestran por qué bajar loss o aumentar accuracy puede aportar muy poca discriminación.

### Evaluación comparable con pesos congelados

Se reevaluó el mejor checkpoint no ponderado de Fase 4, sin dropout ni augmentation, con los mismos pesos sobre TODO train y TODA validación. Train: loss 0.596585, AUC paciente 0.615470, accuracy 0.705815. Validación: loss 0.605551, AUC paciente 0.552357, accuracy 0.704545. Esta comparación utiliza pesos finales iguales, a diferencia de las métricas online de entrenamiento.

### Auditoría de BatchNorm sobre la población completa

Se mantuvieron EXACTAMENTE todos los parámetros aprendidos del checkpoint Fase 4 y se recalibraron solo los buffers BN con los 8.745 cortes/877 pacientes de train, sin dropout y sin gradientes. La misma validación completa pasó de AUC paciente 0.552357 a 0.543127; loss 0.605551 → 0.604539. Recall 0.000000 → 0.000000; especificidad 1.000000 → 1.000000. Es un diagnóstico de inferencia con pesos congelados, no otro entrenamiento completo de dos épocas.

Características globales en validación antes de recalibrar: rango efectivo 3.604, fracción sin variación 0.0000, distancia entre centroides de clase/radio RMS interno 0.1374. Después: rango efectivo 3.499, fracción sin variación 0.0000, razón de centroides 0.1363. Son descriptores exploratorios, no una prueba formal de separabilidad ni de causalidad.

La recalibración BN redujo ligeramente la BCE pero empeoró el AUC; por tanto, no se aplica al modelo histórico ni se recomienda como mejora demostrada. Los parámetros aprendidos se mantuvieron exactamente iguales. Las características no están colapsadas, pero la distancia entre centroides de clase es pequeña respecto a la variación interna de cada clase. Esto es compatible con una representación todavía poco discriminativa, sin demostrar una causa única.

### Tiempo, alcance y preservación

- AUG_UNWEIGHT: 23.06 min; `models/RUN_20261020_210343_F5_AUG_UNWEIGHT_bc0b6ae50aef/`.
- RAW_WEIGHTED_NOCLIP: 22.27 min; `models/RUN_20261020_212844_F5_RAW_WEIGHTED_NOCLIP_c281fe1f7229/`.

Runs completos: 45.34 min sumados. Memorización técnica de tres CNN: 3.75 min. La campaña tiene un deadline absoluto de 75 minutos desde el primer diagnóstico, más conservador que contabilizar solo entrenamiento con gradientes. Cada batch comprueba el presupuesto; no se amplía retroactivamente.

El benchmark espacial estimó 54.76 min para dos épocas completas y no permitía combinarlo con los demás runs previstos con margen. La espacial sí se probó y entrenó en el diagnóstico de 40 cortes; no se le inventa un AUC de validación completa. Su configuración completa queda preparada para GPU.

Se verificaron 385 archivos históricos por SHA-256: cero cambios o ausencias. Las reevaluaciones independientes de todos los nuevos mejores checkpoints reproducen AUC y probabilidades guardadas; los checkpoints completos pasan el contrato de continuación. Las 25 pruebas distintas verificadas pasan y las cinco configuraciones GPU mantienen comparaciones de una variable principal. Dataset, particiones y .gitignore se conservan. Sin commits, pushes, instalaciones de PyTorch ni entrenamientos pendientes en segundo plano.

### Qué se ha aprendido y qué sigue abierto

La CNN tiene capacidad de memorizar pacientes y todos sus cortes, con gradientes útiles y características variables. El AUC bajo en pacientes nuevos no se explica por una incapacidad total de actualizar pesos ni por una ruptura evidente de la asociación paciente/corte/fase en las rutas. Eso tampoco descarta ruido clínico de etiquetas, señal predictiva débil o factores de adquisición que no se han auditado.

El solapamiento y la cercanía a los predictores constantes muestran que el aprendizaje discriminativo logrado hasta ahora es pequeño. Cambiar ponderación desplaza el sesgo de clasificación; la augmentation no ha demostrado mejora en dos épocas. Las pruebas pequeñas no justifican eliminar BN ni afirmar que el pooling sea la causa única. La evaluación congelada y la auditoría BN anteriores deben interpretarse junto a estos límites.

Dos épocas, una semilla y selección repetida en la misma validación son evidencia exploratoria. No se afirma haber alcanzado un modelo robusto ni se garantiza AUC 0,60–0,70. No se ha invertido la salida ni optimizado un umbral para presentar un AUC mejor.

### Batería GPU propuesta

La batería principal preparada compara BASE_UNWEIGHT, SPATIAL_UNWEIGHT y AUG_UNWEIGHT con un control común: original sin ponderación, solo extracción espacial adicional y solo geometría suave. Todos usan hasta diez épocas, scheduler plateau y early stopping por paciente. La pareja opcional BASE_WEIGHTED_NOCLIP/RAW_WEIGHTED_NOCLIP permite comprobar por separado la representación de fases con el mismo protocolo. Prioridad 1: BASE_UNWEIGHT, control necesario para repetir el mejor protocolo anterior. Prioridad 2: SPATIAL_UNWEIGHT, porque aún falta su comparación completa y GPU permite medirla. Prioridad 3: AUG_UNWEIGHT, para comprobar regularización durante más épocas con el mismo control; no porque haya mejorado en dos épocas. RAW_WEIGHTED_NOCLIP baja de prioridad al no superar su referencia en este piloto, aunque su pareja de control queda preparada. Recalibrar BN no es prioritario: en este control redujo el AUC y no se incorpora a la batería como supuesta mejora.

Comandos Windows, selección de intérprete, comprobación real de CUDA y recuperación segura: [GPU_FASE5.md](GPU_FASE5.md). CUDA no se ha podido ejecutar en este equipo CPU. No sustituir el PyTorch GPU del equipo de clase.

Evidencias en la campaña: comparison.csv, verification.json, unittest.log, label_audit.json, path_alignment_audit.json, patient_slice_audit.json, null_baselines.json y los diagnósticos. Cada nuevo run conserva checkpoints, historial, curvas y predicciones con identificadores. Los diagnósticos se etiquetan como tales y se excluyen de la comparación de entrenamientos completos.
