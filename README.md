# breastcancer

CNN 2D propia para predecir **pCR/no pCR**, no presencia/ausencia de cáncer.
Entrada baseline: PRE, EARLY−PRE y LATE−PRE; selección por **ROC AUC de paciente**.

## Entorno y datos

Ejecutar los comandos desde la raíz del proyecto con Python 3.13 y el entorno del equipo.
En PowerShell, sustituir `python` por `& .\.env\Scripts\python.exe` si no está activado.
Instalar `requirements.txt` en el entorno elegido. La instalación PyTorch CPU/CUDA debe
seleccionarse para cada ordenador; no copiar `.env` ni instalar automáticamente el lock CPU en la GPU.
`requirements-cpu-lock.txt` registra las versiones del portátil; cada run registra backend y versiones.

Se conservan `metadata/internal_train.csv` y `metadata/internal_val.csv`.
No modificar particiones ni utilizar test durante el desarrollo. No incluir dataset ni pesos en commits.

## Comprobaciones rápidas

```powershell
python -m unittest discover -s tests -v
python -m src.diagnostics
python -m src.train --preset A --smoke
```

`diagnostics` usa un batch fijo de cuatro muestras de train, ambas clases, y un máximo
de 40 actualizaciones o unos 25 segundos de entrenamiento. Falla si no reduce la pérdida.
`--smoke` usa los mismos dos pacientes por clase en cada partición (semilla 42),
todos sus cortes y dos épocas. **Son diagnósticos técnicos parciales**, no estimaciones de generalización.

## Experimentos de dos épocas para el ordenador de clase

```powershell
python -m src.train --preset A --device auto
python -m src.train --preset B --device auto
python -m src.train --preset D --device auto
```

A conserva tres bloques, clipping y pérdida ponderada. B cambia solo a dos bloques.
D cambia solo a pérdida normal. Todos conservan LR, dropout, weight decay, batch y semilla.
Estos comandos utilizan train y validación interna completos: medir primero el coste del equipo.

Para C, ajustar previamente estadísticas sobre TODO train, sin clipping y antes del z-score:

```powershell
python -m calcular_stats --preset C --output models/stats_C_full.json
python -m src.train --preset C --stats models/stats_C_full.json --device auto
```

Para verificar las constantes del baseline, sin cambiarlas automáticamente:

```powershell
python -m calcular_stats --preset A --output models/stats_A_full.json
```

`--max-samples 32` permite medir el coste del cálculo de estadísticas. Su resultado queda
marcado como parcial y **se rechaza para normalizar entrenamientos**. Las salidas nunca se sobrescriben.
Cambiar representación o clipping exige estadísticas compatibles; no reutilizar las del baseline para `raw`.

## Configuración normal y opciones

`python -m src.train` conserva la configuración normal de **25 épocas**; no es una prueba rápida.
`config/config.py` mantiene los nombres originales; `RANDOM_SEED` y
`EARLY_STOPPING_PATIENCE`, si no son None, prevalecen sobre `SEED` y `PATIENCE`.
El número de bloques es `len(CONV_CHANNELS)`.

Opciones: `--config archivo.json`, `--epochs N`, `--workers N`, `--stats archivo.json`,
`--device auto|cpu|cuda`, `--preset A|B|C|D` y `--smoke`.
Las configuraciones Python externas solo admiten asignaciones literales, sin ejecutar código.
`SMOKE_TEST=False`, augmentation desactivado, límites de batches None y guardado cada época son los valores normales.
Augmentation opcional aplica los mismos volteos a los tres canales y solo durante train.

## Resultados y evaluación

Se conserva la carpeta `models/` y el concepto de un directorio por run.
Los identificadores nuevos `RUN_fecha_hora_nombre_sufijo` evitan reutilizar números del ordenador de clase.
Cada run guarda configuración exacta, fuentes, entorno, selecciones, hashes de CSV, checkpoints,
historial, predicciones por corte/paciente, matrices y curvas. `best_epoch.pt` se elige
exclusivamente por AUC de paciente; los empates conservan la primera época.
`val_auc`/`train_auc` históricos siguen significando AUC por corte; usar columnas `*_patient_auc` para pacientes.
Las métricas de train son online, con pesos y dropout cambiantes; validación usa pesos fijos en modo eval.

```powershell
python -m src.evaluate models/RUN_IDENTIFICADOR/best_epoch.pt --output models/EVALUACION_NUEVA
python -m src.compare_runs
```

El evaluador recupera arquitectura y preprocesamiento desde el checkpoint, comprueba la partición,
no activa augmentation y no utiliza test. Para pesos históricos sin contrato exige `--legacy-config`
con todos los campos necesarios; no adivina su representación. Solo cargar pesos de procedencia conocida.
No hay todavía reanudación automática de entrenamientos interrumpidos: los comandos anteriores inician runs nuevos.

La ausencia local de runs históricos no permite conocer sus métricas. La comparación automática distingue
diagnósticos parciales, entrenamientos completos, pendientes e históricos no disponibles.

Ver `documentation/FASE2.md` para cambios, resultados medidos y limitaciones.


## Fase 3: organización de runs futuros

Los runs anteriores no se mueven ni modifican. Los nuevos entrenamientos guardan:

```text
models/RUN_identificador/
  history.csv                 # todas las métricas, paciente y corte
  epoch_summary.csv           # resumen legible por época
  summary.txt / summary.json
  analysis.md
  loss_curve.png
  accuracy_curve_patient.png / accuracy_curve_slice.png
  auc_curve_patient.png / auc_curve_slice.png
  roc_curve_patient.png / roc_curve_slice.png
  confusion_matrix_patient.png / confusion_matrix_slice.png
  probability_distribution_patient.png / probability_distribution_slice.png
  checkpoints/
    best_epoch.pt
    epoch_001.pt ...
  details/
    config.py / config.json
    environment.json
    train_selection.csv / validation_selection.csv
    best_predictions/
    epoch_001/train/ ... /validation/ ...
    errors_patient.csv / errors_slice.csv
    probability_analysis.json
    sources/
```

Los nombres históricos de loss, accuracy y AUC por corte también se conservan.
Las métricas y gráficas con sufijo `patient` son las principales. Las estadísticas
clínicas no forman parte de la entrada: la CNN utiliza exclusivamente imágenes.

La lectura de checkpoints antiguos continúa disponible. Para nuevos pesos:

```powershell
python -m src.evaluate models/RUN_IDENTIFICADOR/checkpoints/best_epoch.pt --output models/EVALUACION_NUEVA
python -m src.compare_runs
```

`src.phase3` define diez comparaciones de tres épocas, con una variable principal
por relación padre/hijo. P3_10 (fases raw) se compara con P3_02 (sin clipping);
P3_02 y P3_10 requieren sus propias estadísticas completas de train. Los demás se
comparan con P3_01. No se cambia el umbral de 0.50.

```powershell
python -m src.phase3 --prepare
python -m src.phase3 --session models/PHASE3_IDENTIFICADOR --run P3_01
```

La preparación mide ocho batches de entrenamiento y validación y abre un presupuesto
compartido de 60 minutos. Cada run comprueba el tiempo disponible antes de empezar;
una fecha límite se comprueba entre batches y conserva los pesos si interrumpe una
época. Se reservan 30 segundos para cerrar y guardar. Una validación interrumpida no
se usa para seleccionar el mejor checkpoint.

Las diez plantillas portables están en `config/phase3/` y pueden viajar con el código,
sin copiar el entorno, los datos ni los pesos. Cada sesión conserva además su propio snapshot.
Para un experimento pendiente con CUDA verificada:

```powershell
python -m src.train --config config/phase3/P3_07.json --device cuda
```

Si se utiliza `TRAIN_PATIENTS_PER_CLASS`, solo se limita train; la validación interna
permanece completa. Esos runs quedan marcados como preliminares y se comparan únicamente
con otros que usen exactamente el mismo subconjunto, semilla y configuración de referencia.


Las estadísticas verificadas de la Fase 3 se conservan como agregados en
`config/phase3/stats/`, sin imágenes ni pesos. Para comparar el clipping con
normalización consistente, cuando exista `enhancement_no_clip.json`:

```powershell
python -m src.train --config config/phase3/P3_02.json --stats config/phase3/stats/enhancement_no_clip.json --device cuda
```

El perfil `baseline_clipped.json` permite verificar las constantes del baseline
sin modificar los checkpoints ya entrenados. Consultar `documentation/FASE3.md`
para resultados completos y las variantes pendientes.


## Fase 4: ejecución secuencial y continuación segura

Guía completa para Windows/VS Code y la GPU de clase: `documentation/GPU_MANANA.md`.
No reemplazar el PyTorch GPU existente por el entorno o lock CPU del portátil.

```powershell
python -m src.batch_train --check-device cuda
python -m src.batch_train --suite config/phase4/gpu_suite.json --device cuda --resume-interrupted
```

La batería contiene un baseline GPU propio y cuatro variantes de una variable principal,
con máximo de diez épocas, early stopping por AUC de paciente y scheduler plateau común.
Los completados compatibles se omiten. Cada lote registra errores y comparación automática.
Para un único modelo, añadir `--only NO_WEIGHT_PC` (u otro ID del suite).

Continuación explícita en un directorio nuevo, sin sobrescribir el run de origen:

```powershell
python -m src.train --resume models/RUN_IDENTIFICADOR/checkpoints/last_epoch.pt --epochs 10 --device cuda
```

La continuación restaura optimizador, scheduler, RNG, DataLoader, historial y early stopping.
Los checkpoints Fase 4 de media época no son continuaciones válidas; usar el punto
`recovery_epoch_start_XXX.pt` para repetir esa época. Los pesos antiguos siguen siendo
válidos para evaluación, pero se rechaza prometer una continuación exacta sin todo su estado.
Cambiar de CPU a GPU requiere `--allow-device-change`, con advertencia explícita en el registro
sobre la distinta trayectoria numérica. No cambiar loss, LR inicial, batch, arquitectura,
preprocesamiento ni scheduler dentro de una continuación; preparar un experimento nuevo.

Se mantiene una convolución por bloque como baseline. `CONVS_PER_BLOCK=2` activa la variante
espacial propia, con dos convoluciones antes de cada MaxPool; no descarga pesos preentrenados.
Las figuras de accuracy/AUC nuevas usan escala [0,1] y ticks enteros para no exagerar cambios pequeños.
Los ensayos de hoy usan dos épocas completas y se etiquetan como preliminares, aunque usen todos los pacientes.

## Fase 5: diagnóstico y geometría sincronizada

La baseline original se conserva. Los diagnósticos de memorización de pacientes de train se separan de los entrenamientos completos y no se interpretan como validación.
Informe y evidencias de la campaña: [documentation/FASE5.md](documentation/FASE5.md).
Guía actual para el ordenador GPU, sin reinstalar PyTorch: [documentation/GPU_FASE5.md](documentation/GPU_FASE5.md).

```powershell
python -m src.batch_train --check-device cuda
python -m src.batch_train --suite config/phase5/gpu_suite.json --only BASE_UNWEIGHT SPATIAL_UNWEIGHT AUG_UNWEIGHT --device cuda --resume-interrupted
```

## Fase 6: GPU AMD / ROCm en Linux

Guía de la batería de cinco modelos, veinte épocas, preflight y reanudación: [GPU_FASE6.md](GPU_FASE6.md). Las imágenes y checkpoints son locales; no se incluyen en Git.

## Fase 7: curvas comparables y controles espaciales

Cuatro configuraciones independientes con evaluación completa de train al final de
cada época, sin augmentation: [GPU_FASE7.md](GPU_FASE7.md).
Auditoría, diferencias y límites de las referencias históricas:
[documentation/FASE7.md](documentation/FASE7.md).

## Fase 8: LR y regularización con controles explícitos

Tres contrastes de un hiperparámetro, sin cambiar arquitecturas ni estadísticas:
[GPU_FASE8.md](GPU_FASE8.md). Auditoría y límites de comparación:
[documentation/FASE8.md](documentation/FASE8.md).

## Fase 9: cinco folds oficiales

Validación cruzada con `samples.csv:fold`, únicamente sobre train público, y
normalización ajustada en cada fold: [GPU_FASE9.md](GPU_FASE9.md).
Recuentos verificados, test reservado y límites del OOF:
[documentation/FASE9.md](documentation/FASE9.md).
