# Fase 9 · cinco folds oficiales en Linux / ROCm

La ejecución puede repartirse entre varias sesiones. Mantener la misma campaña:
los folds completos se omiten y los interrumpidos se reanudan explícitamente.

Desde la raíz del repositorio en VS Code, sin sustituir el entorno existente:

```bash
git status --short
git pull --ff-only origin main
conda activate cancer
python -c "import torch; print(torch.__version__, 'HIP=', torch.version.hip); assert torch.version.hip and torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m src.batch_train --check-device cuda
python -m src.cross_validation audit
python -m unittest discover -s tests -v
```

Los folds están en `metadata/samples.csv`, no en los nombres de carpetas de imágenes.
Solo `split=train` y fold 0–4 participan. `split=test`, fold −1, queda reservado.
No ejecutar `src.split_data` ni crear un nuevo StratifiedKFold.

## Comenzar con SPATIAL_SMALL

Preparar una campaña única: crea solamente CSV derivados y manifiestos, sin entrenar
ni leer píxeles. Conservar la ruta impresa en la variable:

```bash
CV_SMALL=$(python -m src.cross_validation prepare --model SPATIAL_SMALL --print-path)
echo "$CV_SMALL"
```

Primero puede ejecutarse solo el fold 0 para medir el coste real de estadísticas y
entrenamiento; máximo 20 épocas y early stopping 8 por AUC paciente:

```bash
python -u -m src.cross_validation run --campaign "$CV_SMALL" --device cuda --folds 0
```

Para continuar con todos los pendientes, omitiendo los completos:

```bash
python -u -m src.cross_validation run --campaign "$CV_SMALL" --device cuda
```

Ejemplo progresivo solicitado: primero 0, 1 y 2; después 3 y 4:

```bash
python -u -m src.cross_validation run --campaign "$CV_SMALL" --device cuda --folds 0 1 2
python -u -m src.cross_validation run --campaign "$CV_SMALL" --device cuda --folds 3 4
```

Si se vuelve a solicitar 0–4 con la misma campaña, los completados no se repiten.
No ejecutar prepare otra vez para continuar; guardar la ruta de CV_SMALL.

Si se interrumpió un fold:

```bash
python -u -m src.cross_validation run --campaign "$CV_SMALL" --device cuda --resume-interrupted
```

También se puede elegir `--folds 1 2` en un comando run. La preparación de estadísticas
lee exclusivamente el train de ese fold, una vez; la reanudación reutiliza el perfil
firmado. No mezclar carpetas de campañas ni modificar sus CSV/configs/stats.
Una ejecución incompleta exige `--resume-interrupted`; sin ello se informa del error
y no se reinicia silenciosamente desde cero. La continuación crea un RUN nuevo y
restaura modelo, optimizador, scheduler, RNG, historial y early stopping. Un fold
ya detenido por early stopping está completo y no vuelve a entrenarse.

Al volver a abrir la terminal, recuperar la ruta mostrada en la preparación:

```bash
CV_SMALL="models/CV9_FECHA_HORA_SPATIAL_SMALL_IDENTIFICADOR"
python -m src.cross_validation summarize --campaign "$CV_SMALL"
```

Sustituir ese ejemplo por la ruta real existente; no ejecutar prepare de nuevo si
se pretende continuar la misma campaña. Una campaña nueva inicia cinco experimentos
nuevos desde cero, no continúa los anteriores.

## Evaluar SPATIAL_CONTROL después

No se mezclan arquitecturas en un promedio CV. Preparar otra campaña:

```bash
CV_CONTROL=$(python -m src.cross_validation prepare --model SPATIAL_CONTROL --print-path)
python -u -m src.cross_validation run --campaign "$CV_CONTROL" --device cuda
python -m src.cross_validation summarize --campaign "$CV_CONTROL"
```

Mismos folds y semilla, modelos/optimizadores nuevos por fold. Cada campaña ajusta
sus perfiles en los cuatro folds de train; la estadística no depende de la
arquitectura, pero se conserva por campaña para simplificar la trazabilidad.
Nada se inicializa con pesos históricos ni con pesos de otro fold.

## Archivos y comparación

```text
models/CV9_FECHA_HORA_SPATIAL_SMALL_UUID/
    campaign.json, template.json, audit.json, fold_inventory.csv, sources.zip
    fold_0/ ... fold_4/
        train.csv, validation.csv
        statistics_UUID.json, config_UUID.json  # aparecen al ajustar ese fold
        runs/RUN_FECHA_HORA_F9_SPATIAL_SMALL_FOLD0_UUID/
            config.py, history.csv
            epoch_001.pt ... best_epoch.pt
            loss_curve.png, loss_comparison.png
            accuracy_curve.png, auc_curve.png, roc_curve.png
            internals/
    fold_comparison_UUID.csv
    cv_summary_UUID.json
    oof_predictions_UUID.csv, oof_roc_UUID.png  # solo con cinco folds completos
```

El resumen se actualiza tras cada fold mediante archivos nuevos, sin sobrescribir
resúmenes anteriores. Comparar AUC medio no ponderado de los cinco folds y desviación
estándar muestral (`ddof=1`), junto con las métricas de cada fold y AUC OOF por
paciente. Una media de menos de cinco folds se marca provisional; no se genera
AUC OOF si falta cobertura. Los smoke tests nunca se incluyen en ese agregado.

El OOF debe tener exactamente 1.097 pacientes, cada uno predicho por el modelo que
lo excluyó de train. No incluye los 176 de test. Recordar que cada mejor checkpoint
se selecciona con la validación del propio fold: son métricas de desarrollo CV,
con posible optimismo por selección; no sustituyen al test externo final.

Por época: train online y train eval separados, train eval/validation loss comparables,
AUC y accuracy paciente, balanced accuracy, sensibilidad, especificidad y TN/FP/FN/TP.
La matriz completa está también en cada checkpoint; la del mejor checkpoint y sus
distribuciones de probabilidades permanecen en internals. Umbral de clasificación
0,5; AUC con probabilidades continuas. No optimizar umbrales para perseguir un AUC.

Smoke diagnóstico en Windows/CPU, una vez si se desea comprobar instalación:

```bash
python -m src.cross_validation smoke --model SPATIAL_SMALL
```

Hace dos épocas solo en fold 0, una paciente de cada clase en train y en validación,
todos sus cortes; ajusta stats de ese pequeño train. No acredita rendimiento clínico.
La ejecución CV completa está bloqueada en CPU y requiere GPU ROCm.

Auditoría, recuentos reales y diferencias respecto a la guía original:
[FASE9.md](documentation/FASE9.md).
