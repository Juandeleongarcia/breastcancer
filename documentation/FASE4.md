# Fase 4 — diagnóstico y experimentos reproducibles

20 de octubre de 2026. Campaña terminada: tres entrenamientos completos de dos épocas y reevaluación independiente.

## Protocolo y preservación

Entrada exclusiva DCE-MRI, etiquetas pCR/no pCR. Se mantienen las particiones de 877 pacientes (8.745 cortes) de train y 220 pacientes (2.200 cortes) de validación interna. Test no se utiliza. Semilla 42; batch 16; cuatro hilos CPU; sin aumentos. Dos épocas completas por experimento, resultados preliminares.

La CNN original [32,64,128], con una convolución por bloque, BatchNorm, ReLU, MaxPool, pooling global y dropout 0,4, conserva sus 93.825 parámetros y las claves de sus pesos. Los checkpoints históricos siguen siendo evaluables. La variante espacial opcional añade una segunda convolución por bloque: 288.033 parámetros, campo receptivo antes del pooling global de 36 frente a 22 píxeles. Se ha probado forward/backward finito y carga estricta del baseline histórico. Esta variante no sustituye al baseline.

Copias de seguridad previas: `backups/phase4_20261020_183839_0ff6decf/`. Los nuevos runs se identifican por fecha, hora, configuración y UUID; raíz con historial, resumen, curvas y matrices, pesos en `checkpoints/`, fuentes/configuración/predicciones en `details/`. No se renombran ni sobrescriben runs históricos. No se hacen commits ni pushes.

## Preprocesamiento revisado

Se conserva el baseline [PRE, EARLY−PRE, LATE−PRE]. La resta se realiza en float32 después de convertir los PNG a [0,1], por lo que conserva realces negativos. El clipping es opcional y anterior a la estandarización global por canal. No se añade normalización independiente por fase o imagen. La configuración sin clipping carga su propio perfil de medias/desviaciones, ajustado exclusivamente en los 8.745 cortes de train; no reutiliza las estadísticas de imágenes recortadas. El contrato verifica procedencia, representación y huellas del CSV.

Eliminar clipping adicional no recupera información que ya se hubiera perdido al exportar las imágenes originales. Las estadísticas y la integridad del código no demuestran por sí solas que la señal disponible sea suficiente para predecir respuesta.

## Entrenamiento y métricas

AdamW; weight decay 0,0005; LR baseline 0,0005. Las comparaciones cambian una variable principal respecto al baseline: quitar ponderación, quitar clipping, o LR 0,0001. Se conservan BatchNorm y dropout en estos runs para no confundir efectos. El baseline ponderado usa pos_weight=2,408028 calculado en train. Las loss ponderada y no ponderada tienen distinta escala y no se comparan directamente como mejoras de rendimiento.

El AUC principal utiliza la media de probabilidades de todos los cortes de cada paciente de validación. Se calculan también métricas por corte. El umbral es siempre 0,50 y no se optimiza. Las métricas de train son online, con pesos que cambian durante la época; no equivalen a reevaluar todo train con el checkpoint final. Las curvas accuracy/AUC tienen escala [0,1], épocas enteras y puntos visibles. Se añade registro de LR actual/siguiente y su curva.

## Reanudación y batería GPU

Los checkpoints nuevos incluyen pesos, optimizador, scheduler, historial, mejor modelo, contador de early stopping y estados aleatorios Python/NumPy/PyTorch/DataLoader/CUDA. Se guardan de forma atómica. La continuación es explícita y crea otra carpeta con referencia al origen; los checkpoints parciales no se presentan como épocas completas. Un checkpoint de inicio de época permite repetir de forma coherente una época interrumpida.

Se rechazan cambios incompatibles de arquitectura, preprocesamiento, entrenamiento, particiones o código. La interfaz comprueba versión mayor/menor de PyTorch. Cambiar de CPU a GPU requiere una opción explícita; no promete trayectoria numérica idéntica. Los checkpoints antiguos sin todos los estados conservan su utilidad para evaluación, pero no se presentan como continuaciones exactas.

La batería GPU permite selección por ID, ejecución secuencial, omisión de completados, continuación compatible, errores registrados y comparación automática. El protocolo común GPU tiene máximo 10 épocas, early stopping con paciencia 4 y ReduceLROnPlateau con paciencia 2. Su baseline propio permite comparaciones controladas; no se mezclan sus resultados con los runs cortos sin scheduler. La prueba CUDA real debe hacerse en el equipo de clase; aquí solo hay PyTorch CPU.

Comandos e instrucciones completas: [GPU_MANANA.md](GPU_MANANA.md). El archivo auxiliar de dependencias GPU excluye torch/torchvision; no se sustituye el PyTorch GPU del ordenador de clase.

## Archivos y pruebas

- `src/model.py` y `src/configuration.py`: variante espacial opcional y validación de opciones, con defaults compatibles con la CNN original.
- `src/checkpointing.py` y `src/experiment.py`: contrato de continuación, guardado atómico, recuperación en límites de época y estado completo de entrenamiento.
- `src/train.py` y `src/resume_validation.py`: continuación explícita y comprobación de versión de PyTorch antes de iniciarla.
- `src/metrics.py`: escala coherente de curvas, épocas enteras y curva de learning rate; AUC continuo y agregación por paciente conservados.
- `src/evaluate.py`: reconstrucción compatible de CNN históricas y evaluación independiente sin el deadline antiguo.
- `src/compare_runs.py`: columnas de interpretación preliminar, scheduler y convoluciones por bloque.
- `src/batch_train.py`: batería secuencial portable, selección por ID, omisión de completados, recuperación y registro de errores.
- `src/phase4.py`: benchmark previo, decisiones registradas, selección manual informada y presupuesto absoluto de 90 minutos; `src/phase3.py` acepta deadline en su benchmark.
- `config/phase4/`, `requirements-gpu-tools.txt`, `tests/test_phase4.py` y `documentation/GPU_MANANA.md`: protocolos congelados, dependencias auxiliares y uso sin Codex.

Las 20 pruebas unitarias y de compatibilidad pasaron. Se comprobó continuidad exacta en CPU de pesos, optimizador, scheduler y estados RNG, rechazo de contratos incompatibles, recuperación sin usar pesos de media época, omisión de runs completos, registro de errores y continuidad de la batería tras un fallo. La variante espacial pasó forward/backward y gradientes finitos. Además, se verificaron las cinco configuraciones GPU: una variable principal por variante y estadísticas de train coherentes. Los perfiles GPU baseline se enlazan explícitamente a `baseline_clipped.json`, cuyas constantes difieren de las históricas en menos de 5e-11; los runs CPU de hoy no se modificaron retroactivamente.

## Resultados finales verificados

Validación interna completa. Accuracy, recall, especificidad y F1 son por paciente; loss es BCE por corte. Todos los mejores checkpoints corresponden a la época 2. Resultados preliminares de una semilla.

| Experimento | Variable principal | AUC paciente | AUC corte | Accuracy | Val loss | Recall | Especificidad | F1 | Minutos |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NO_WEIGHT_PC | BCE sin ponderación | 0.552357 | 0.551901 | 0.704545 | 0.605551 | 0.000000 | 1.000000 | 0.000000 | 18.89 |
| NO_CLIP_PC | Sin clipping adicional | 0.541241 | 0.541623 | 0.340909 | 1.001718 | 0.938462 | 0.090323 | 0.456929 | 17.15 |
| LOW_LR_PC | LR=0,0001 | 0.543424 | 0.541625 | 0.400000 | 0.986308 | 0.876923 | 0.200000 | 0.463415 | 17.71 |

La loss de NO_WEIGHT_PC no es directamente comparable con las dos loss ponderadas. Referencia histórica P3_01: AUC paciente 0,530521 y corte 0,529881 en época 2; se conserva sin reentrenar ni modificar. El mayor AUC observado ahora es 0,552357, mejora descriptiva de 0,021836 frente a esa referencia, sin demostrar superioridad estable.

### Evolución por época

| Experimento | Época | Train loss online | Val loss | AUC paciente | AUC corte | Accuracy | Recall | Especificidad | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NO_WEIGHT_PC | 1 | 0.605705 | 0.613341 | 0.518511 | 0.516520 | 0.704545 | 0.000000 | 1.000000 | 0.000000 |
| NO_WEIGHT_PC | 2 | 0.601645 | 0.605551 | 0.552357 | 0.551901 | 0.704545 | 0.000000 | 1.000000 | 0.000000 |
| NO_CLIP_PC | 1 | 0.979180 | 0.995010 | 0.516129 | 0.514415 | 0.409091 | 0.661538 | 0.303226 | 0.398148 |
| NO_CLIP_PC | 2 | 0.972415 | 1.001718 | 0.541241 | 0.541623 | 0.340909 | 0.938462 | 0.090323 | 0.456929 |
| LOW_LR_PC | 1 | 0.984601 | 0.992778 | 0.507891 | 0.507218 | 0.409091 | 0.738462 | 0.270968 | 0.424779 |
| LOW_LR_PC | 2 | 0.975304 | 0.986308 | 0.543424 | 0.541625 | 0.400000 | 0.876923 | 0.200000 | 0.463415 |

### Análisis y problemas pendientes

NO_WEIGHT_PC: matriz [[155,0],[65,0]]. Todas las puntuaciones son menores de 0,50; la accuracy 70,45% corresponde a predecir siempre no pCR. Medias por clase: 0,322135 frente a 0,329696, con desviaciones cercanas a 0,040. Quitar ponderación desplaza las puntuaciones, pero no resuelve la separación ni la sensibilidad.

NO_CLIP_PC: matriz [[14,141],[4,61]]. Medias 0,574690 y 0,580702, desviaciones 0,060936 y 0,068068. El AUC sube entre épocas aunque la loss de validación empeora. Retirar clipping no corrige el exceso de falsos positivos ni aporta evidencia suficiente de mejora estable.

LOW_LR_PC: matriz [[31,124],[8,57]]. Medias 0,531025 y 0,536900, desviaciones 0,058457 y 0,057813. Train y val loss bajan y el AUC aumenta; la especificidad sigue siendo solo 20%. Es un candidato de estabilidad para entrenar más tiempo, sin afirmar que dos puntos definan una curva estable.

Las puntuaciones de ambas clases siguen fuertemente solapadas. El ponderado equilibra la contribución de clases, pero no asegura probabilidades calibradas ni buen ranking. Sin ponderación, la salida queda cerca de la prevalencia; con ponderación, se desplaza hacia 0,5. La sensibilidad y la accuracy cambian mucho sin que aparezca una separación clara. No se ha ajustado el umbral ni utilizado test. No se ha alcanzado 0,60 y no se garantiza 0,60–0,70.

Dos épocas y una sola semilla no permiten atribuir una causa única ni descartar aprendizaje más lento. La selección del mejor checkpoint y de configuraciones usa repetidamente esta misma validación: sus resultados son exploratorios. Si diez épocas no mejoran el ranking, conviene auditar registro espacial y escala relativa de PRE/EARLY/LATE en la exportación, comprobar localización del tejido informativo y capacidad de aprendizaje en controles de train antes de aumentar más la complejidad. Estos son pasos propuestos, no defectos demostrados del dataset.

### Tiempo y decisión sobre la cuarta configuración

Benchmark inicial de ocho batches por fase: 21.66 min por run de dos épocas; tres prioritarios estimados en 64.98 min. Los tres runs reales consumieron 53.75 min sumados. Se respetó el presupuesto máximo de 90 minutos con deadline absoluto entre batches.

La variante SPATIAL_PC pasó pruebas y se midió solo para estimar coste: 39.34 min por dos épocas. Quedaban aproximadamente 33,66 min, insuficientes incluso antes del margen de seguridad del 10% y 45 segundos. No se entrenó ni se le asignaron métricas; queda preparada para GPU. Los batches del benchmark son mediciones técnicas de tiempo, no un cuarto entrenamiento completo ni resultados de rendimiento.

### Verificación y conservación

Se verificaron SHA-256 de 226 archivos históricos: cero cambios o ausencias. La reevaluación independiente de cada mejor checkpoint usa los 220 pacientes/2.200 cortes completos y reproduce AUC y predicciones guardadas; diferencias máximas de probabilidades inferiores a 1e-7. Los tres last_epoch.pt pasan el contrato de continuación desde época 2. La igualdad exacta de una continuación real se verificó además en las pruebas pequeñas de CPU.

Dataset, particiones y .gitignore conservados; sin commits ni pushes. No quedaron entrenamientos en segundo plano. CUDA no está disponible en este equipo y no se afirma que haya sido probado aquí.

### Qué entrenar mañana

Primero BASELINE y NO_WEIGHT_PC durante hasta diez épocas con el protocolo GPU común, para estudiar si la débil mejora de ranking se mantiene y si desaparece la sensibilidad nula. Después LOW_LR_PC y NO_CLIP_PC, manteniendo comparaciones de una variable. SPATIAL_PC queda como exploración posterior si hay tiempo; no se prioriza su complejidad sobre el diagnóstico de señal. Todos están preparados en gpu_suite.json, con early stopping por paciente, selección del mejor checkpoint y scheduler comunes.

Desde la raíz del proyecto y usando el intérprete GPU existente en VS Code:

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -m src.batch_train --check-device cuda
python -m unittest discover -s tests -v
python -m src.batch_train --suite config/phase4/gpu_suite.json --only BASELINE NO_WEIGHT_PC LOW_LR_PC NO_CLIP_PC --device cuda --resume-interrupted
```

Para incluir también la espacial:

```powershell
python -m src.batch_train --suite config/phase4/gpu_suite.json --device cuda --resume-interrupted
```

El orden efectivo es el de la suite, aunque se escriban los IDs en otro orden. Los completados se omiten. Los ejemplos de modelo individual y reanudación manual, requisitos de versiones y traslado seguro de archivos están en GPU_MANANA.md. No instalar el lock CPU sobre el entorno de clase.

### Ubicación exacta de evidencias

Campaña: `models/PHASE4_20261020_184701_0f2a68cca0ea/`. Contiene comparison.csv, probability_history.csv, verification.json, gpu_configuration_check.json, benchmarks, plan y decisiones. Los scripts verify_results.py y write_report.py permiten reproducir la comprobación y este cierre.

- NO_WEIGHT_PC: `models/RUN_20261020_184801_F4_NO_WEIGHT_PC_a59a7ae89dc0/`.
- NO_CLIP_PC: `models/RUN_20261020_190721_F4_NO_CLIP_PC_d1d28a0f68c6/`.
- LOW_LR_PC: `models/RUN_20261020_192456_F4_LOW_LR_PC_02101097934c/`.

En cada raíz: history.csv, epoch_summary.csv, summary.json/txt, analysis.md, RESULTADOS.md y gráficas; los errores y predicciones se consultan en details/. El historial de probabilidades permite revisar también cada época, no solo el checkpoint seleccionado.
