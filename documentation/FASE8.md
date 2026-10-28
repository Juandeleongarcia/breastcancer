# Fase 8 · generalización con contrastes controlados

Preparación: 28 de octubre de 2026, Windows/CPU. Antes de modificar: árbol limpio,
main, origin correcto; fetch y pull --ff-only sin cambios pendientes sobre
`ef161f3af1745d8424050c7ead366b3aab33acb5`. Backups previos en backups/phase8_...
(excluidos de Git). No se ejecutan entrenamientos completos ni nuevos diagnósticos
de pacientes: preflight sintético y tests son suficientes para esta preparación.

## Evidencia disponible y preguntas abiertas

| Referencia de clase comunicada en la misión F8 | AUC paciente aproximado | Estado local |
|---|---:|---|
| SPATIAL_UNWEIGHT | 0,636 | No reproducido ni pesos identificados |
| SPATIAL_CONTROL | 0,636 | Sin historial/checkpoint de clase disponible |
| SPATIAL_SMALL | 0,6305 | Sin historial/checkpoint de clase disponible |
| SPATIAL_SCHEDULER | 0,636 | Sin historial/checkpoint de clase disponible |
| SPATIAL_GENTLE_AUG | 0,536 | Sin historial/checkpoint de clase disponible |

La comunicación anterior de CONTROL fue 0,6306 y la actual ≈0,636. No se resuelve
esa diferencia inventando precisión: hace falta el history y config del RUN real.
F5/F6 también cambiaron protocolo y alguna configuración apunta a stats de clase
ausentes aquí. No asumir que todos los resultados son estrictamente comparables.
Se conservan como referencias comunicadas, separadas de la tabla automática real.

Sí están disponibles los resultados locales anteriores: por ejemplo CPU3 F6 WD
0,611315 en dos épocas completas, con recall/F1 cero; Fase 7.1 tiene diagnósticos
separados de 8 train / 8 val y no representa generalización clínica. En la sonda
BN local, recalibrar buffers de una copia redujo la pérdida de train y empeoró
validación. No se adopta esa recalibración ni se usa el AUC train=1 como rendimiento
de validación. Ver FASE7_1.md. No se reevalúan ni modifican los modelos históricos.

## Auditoría metodológica

No se detectó un error nuevo en la pérdida/agregación que justifique modificar el
entrenamiento. Se preservan src/model.py, dataset.py, augmentation.py, experiment.py,
metrics.py, configuration.py y checkpointing.py. Sus contratos de reanudación y
semántica de métricas se mantienen. Se fortalecen los preflights y el análisis de
lectura, sin cambiar el comportamiento del optimizador ni las curvas guardadas.

- 877 pacientes / 8.745 cortes de train: 620 no pCR y 257 pCR. Validación interna:
  220 / 2.200, 155 no pCR y 65 pCR. Se verifican disjunción de pacientes, sample_id,
  rutas de imagen, etiquetas coherentes y procedencia del train público. No se usa
  test para evaluar, ajustar ni escoger experimentos.
- Se comprueba que PRE/EARLY/LATE pertenecen al mismo identificador y corte y que
  los archivos existen. Esto no certifica registro anatómico, veracidad de etiquetas
  ni ausencia de imágenes duplicadas de píxeles entre identificadores diferentes.
- Entrada de tres PNG 256×256, gris /255; `[PRE, EARLY-PRE, LATE-PRE]`, clipping y
  z-score originales. La representación conserva las diferencias de contraste antes
  del clipping; no recupera señal cuantitativa ni tiempos de adquisición DICOM.
- Las estadísticas efectivas se verifican contra el CSV train y la representación.
  F8 conserva exactamente las estadísticas de los controles F7 del repositorio:
  config/phase3/stats/baseline_clipped.json. No se sustituyen stats de clase ausentes
  ni se recalculan. La suite firma medias, stds y clipping efectivos; un cambio
  posterior bloquea el contraste afectado. Si aparecen snapshots de controles de
  clase, se contrastan también antes de entrenar. Sin ellos no se acredita réplica
  del número de AUC histórico.
- No augmentation en los tres nuevos modelos. BN/dropout se activan en train online;
  train eval y validación usan pesos finales en eval, sin augmentation, mismo
  criterio BCE y reducción por corte. Train eval no altera pesos/BN/RNG. Las cinco
  curvas originales se conservan, sin suavizar ni reconstruir valores ausentes.
- Salida logit única; BCEWithLogitsLoss sin ponderación, media acumulada con el
  tamaño real de cada batch. Entre 5 y 10 cortes por paciente en train: pacientes
  con más cortes tienen más peso en BCE. AUC paciente da un registro a cada paciente
  después de promediar probabilidades de todos sus cortes. Son estimandos distintos,
  no un error de implementación.
- AUC por corte y paciente a partir de probabilidades continuas; nunca AUC de
  predicciones binarias ni promedio de AUC por batch. Umbral fijo 0,5 solo para
  métricas de clasificación. Accuracy mayoritaria: 155/220=70,45 %, con recall/F1=0.
- Mejor checkpoint por mejora estricta de AUC paciente; empates conservan el primero.
  Early stopping 8. Scheduler original ReduceLROnPlateau, mode=max, factor 0,5,
  patience 2, min_lr 1e-6, threshold 0. AUC oscila con solo 65 positivos y no garantiza
  que su tendencia coincida con la de BCE. No se cambia el selector para favorecer
  retrospectivamente un experimento.

Sobreajuste temprano es una hipótesis compatible con train eval mejorando y BCE
val empeorando; no prueba causal única. Puede coexistir con exceso de confianza,
señal tumoral desigual por corte, confusores de adquisición y agregación que diluye
la señal. Más capas no resuelven automáticamente esos problemas. La selección
repetida sobre la misma validación puede sobreajustarla; reservar test para la
evaluación final independiente, fuera de este flujo.

## Tres experimentos, una intervención cada uno

| ID | Control versionado | Variable | Hipótesis y riesgo |
|---|---|---|---|
| F8_SMALL_LR | F7 SPATIAL_SMALL | LR 0,0005 → 0,0001 | Pasos menores podrían estabilizar aprendizaje; también ralentizarlo o inducir subajuste |
| F8_SMALL_DROPOUT | F7 SPATIAL_SMALL | Dropout 0,40 → 0,50 | Regularizar más el clasificador; puede perder discriminación |
| F8_CONTROL_LR | F7 SPATIAL_CONTROL | LR 0,0005 → 0,0001 | Aislar LR en la red original; mayor coste que SMALL |

Común: 20 épocas máximas, early stopping 8, seed 42, batch 16, AdamW, WD 0,0005,
BCE no ponderada, tres canales, dos convoluciones por bloque, BN/ReLU/MaxPool,
AdaptiveAvgPool 1×1 y dropout final. SMALL [16,32,64] = 72.593 parámetros;
CONTROL [32,64,128] = 288.033, máximo 128 filtros. Misma partición, estadísticas,
scheduler y criterio de selección que el padre. Sin transfer learning.

El dropout solo está en el clasificador, no entre convoluciones: aumentar ese valor
no garantiza impedir memorización de las features. El efecto de LR debe interpretarse
con LR usado/siguiente y el número de épocas efectivas, no solo con curvas visuales.
El antiguo LR bajo comunicado de F5 no demuestra el resultado de F8_CONTROL_LR,
pues no está acreditada igualdad de contratos, estadísticas y protocolo.

Los JSON se clonan de su padre cambiando únicamente la variable experimental y
las dos claves identificativas RUN_NAME/COMPARISON_PARENT. La suite declara también
control_config, expected_changes y firma de estadísticas. Se comprueban las
diferencias en preflight y justo antes de cada entrenamiento del lote. Un error
se registra y no lanza ese modelo; los otros trabajos continúan. No se alteran
los controles para que pasen los tests. Dentro de F8 se documentan comparaciones
contra su propio padre: LR en SMALL no se compara causalmente con dropout en
CONTROL ni se atribuyen todas las diferencias entre SMALL y CONTROL a LR.

Orden recomendado: SMALL_LR, SMALL_DROPOUT y CONTROL_LR. Los dos primeros exploran
dos hipótesis ortogonales en la red más barata; el tercero contrasta si el efecto
de LR se mantiene en la red grande. No se elige ganador sin resultados reales.

## Análisis automático y significado de nuevas columnas

El ejecutor mantiene modelos y RUNs únicos con fecha/hora/UUID, historial, todos
los checkpoints completos, mejor checkpoint y cinco PNG. Matrices, histogramas,
errores, predicciones y técnicos siguen en internals. Nada se añade a la raíz del
RUN ni se escribe en históricos para ampliar el análisis.

La comparación de lotes incorpora los nuevos campos de lectura. `python -m
src.phase8` genera una tabla nueva con los tres F8 y los dos controles F7 cuando
están realmente disponibles. Los diagnósticos parciales se excluyen de esa tabla;
controles ausentes y modelos pendientes no tienen métricas inventadas.
Las filas disponibles llevan `Comparabilidad` y `Diferencias_config`: se cotejan
sus snapshots y particiones con la receta versionada, sin alterar el histórico.
Una coincidencia de configuración no demuestra superioridad estadística ni igualdad
numérica entre backends/hardware; una discrepancia se hace visible y debe revisarse.

- BCE val y train eval, y su gap, corresponden a la misma mejor época por AUC.
  Se conserva también la loss online como otra medida con protocolo distinto.
- Balanced accuracy = (sensibilidad+especificidad)/2; para siempre no pCR vale 0,5.
  Sensibilidad y recall son la misma métrica binaria.
- PR_AUC_AP_paciente es **average precision no interpolada**, calculada del CSV
  paciente del mejor checkpoint con scores continuos. No es el área trapezoidal PR.
  Si faltan predicciones, queda vacía. Prevalencia positiva se informa para darle
  contexto; baseline constante AP ≈0,2955 en la validación completa actual.
- Parámetros entrenables: conteo exacto de la CNN propia, sin construir un modelo
  ni consumir RNG. Se valida contra PyTorch en tests.
- Se mantienen tiempo del segmento y tiempo acumulado cuando el resumen lo ofrece.
  En históricos sin tiempo acumulado se conserva el tiempo guardado, no se adivina.
- Evidencia_sobreajuste describe cambios primera→última época de train eval/val;
  no es una prueba estadística ni detecta todas las formas de sobreajuste. Se añaden
  mínimo val loss y su época para contrastar con el mejor AUC; revisar toda la curva.

AP/balanced accuracy son análisis complementarios de la tabla, no nuevos criterios
de selección ni columnas reconstruidas en history de runs anteriores. Una mejora
pequeña de AUC, con 220 pacientes y una semilla, no prueba superioridad. No optimizar
umbrales repetidamente ni seleccionar solo curvas que parezcan más bonitas.

## Uso y alcance de las verificaciones

Guía exacta para Linux, `conda activate cancer`, preflight AMD ROCm RX 7600 XT,
ejecución secuencial, individual, resume y comparación: [GPU_FASE8.md](../GPU_FASE8.md).
No se instala ni reemplaza PyTorch. El preflight Windows/CPU verifica formas y
gradientes sintéticos; no acredita ejecución nativa en esa GPU. Debe pasar en clase.

No se entrenan pacientes en esta fase de preparación, ni siquiera un smoke real
adicional: los ensayos F7.1 ya existen y no justificarían repetir sus métricas.
La prueba nueva de extremo a extremo usa únicamente PNG sintéticos temporales y
no es un resultado clínico. Los tests previos verifican reanudación histórica F6/F7;
los nuevos cubren contrastes, firmas, bloqueo aislado, asociación de fases, AP,
conteo de parámetros, huecos sin inventar AUC y guardado/reanudación sintética.

Verificación de cierre: 46 tests iniciales y **57 tests finales superados**, sin
fallos. Preflight CPU sintético de los tres F8 superado; comparación local generada
con tres experimentos pendientes y dos controles de clase no disponibles, sin AUC
inventados. Los hashes de **889 archivos históricos/configuraciones/metadatos** y
**38.109 archivos originales de imágenes** permanecen iguales. Los hashes de imágenes
solo verifican integridad: no se usó test para decisiones experimentales.
Logs y manifiesto de publicación en el backup local excluido de Git. ROCm nativo
queda pendiente de verificar en el entorno `cancer` del ordenador de clase.
