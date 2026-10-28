# Fase 8 · Linux / conda cancer / AMD RX 7600 XT

En la terminal de VS Code y desde la raíz de `breastcancer`:

```bash
git status --short
git pull --ff-only origin main
conda activate cancer
python -c "import sys, torch; print(sys.executable); print(torch.__version__, 'HIP=', torch.version.hip); assert torch.version.hip and torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m src.check_suite --suite config/phase8/suite.json --device cuda --require-rocm
python -m unittest discover -s tests -v
```

No instalar PyTorch CPU ni sustituir el entorno ROCm que ya funciona. PyTorch HIP
utiliza `cuda` también para AMD, según la
[documentación oficial](https://docs.pytorch.org/docs/stable/notes/hip.html).
La GPU indicada en esta misión es RX 7600 XT; las referencias antiguas RX 6700 XT
se conservan como historia, no se asume que sea el mismo equipo. El preflight real
debe confirmar la GPU y el forward/backward antes de entrenar.

## Experimentos y orden recomendado

Todos usan 20 épocas máximas, early stopping con paciencia 8 por AUC paciente,
seed 42, batch 16, AdamW, WD 0,0005, BCE sin ponderar, scheduler plateau original,
sin augmentation, train/validación completos y estadísticas originales verificadas.
No se usa test. No se modificó ninguna arquitectura ni el preprocesamiento.

| Orden | ID | Control exacto | Único cambio de hiperparámetro |
|---|---|---|---|
| 1 | F8_SMALL_LR | config/phase7/SPATIAL_SMALL.json | LR 0,0005 → 0,0001 |
| 2 | F8_SMALL_DROPOUT | config/phase7/SPATIAL_SMALL.json | Dropout 0,40 → 0,50 |
| 3 | F8_CONTROL_LR | config/phase7/SPATIAL_CONTROL.json | LR 0,0005 → 0,0001 |

Los dos primeros conservan canales [16,32,64]; el tercero [32,64,128]. Se mantienen
dos convoluciones por bloque, BN/ReLU, MaxPool, AdaptiveAvgPool 1×1 y clasificador
binario propio. SMALL tiene 72.593 parámetros; CONTROL 288.033.

Comandos individuales:

```bash
python -u -m src.batch_train --suite config/phase8/suite.json --device cuda --only F8_SMALL_LR
python -u -m src.batch_train --suite config/phase8/suite.json --device cuda --only F8_SMALL_DROPOUT
python -u -m src.batch_train --suite config/phase8/suite.json --device cuda --only F8_CONTROL_LR
```

Los tres secuencialmente, con sus configuraciones independientes:

```bash
python -u -m src.batch_train --suite config/phase8/suite.json --device cuda
```

Para reanudar interrupciones compatibles (omite los ya completados):

```bash
python -u -m src.batch_train --suite config/phase8/suite.json --device cuda --resume-interrupted
```

También se puede añadir `--only F8_SMALL_LR` a ese comando. No cambiar LR, dropout,
batch, scheduler, estadísticas ni semilla al reanudar. Los históricos se conservan
en su carpeta; una continuación crea otro RUN y recupera optimizador, scheduler,
RNG, historial y contador de early stopping. Un run que alcanzó early stopping
ya terminó; no se reinicia su contador. No usar los diagnósticos CPU parciales
para continuar entrenamientos completos.

## Verificación del control y discrepancias

Cada entrada de suite declara el control, el cambio permitido y una firma de las
estadísticas efectivas. El preflight comprueba diferencias y el origen train de
las estadísticas; el ejecutor vuelve a hacerlo antes de cada modelo. Si encuentra
una discrepancia, registra error para ese trabajo y continúa con los demás.

Si existen localmente RUNs completos con nombre de experimento F7_SPATIAL_SMALL o
F7_SPATIAL_CONTROL, se contrastan sus snapshots. Una discrepancia de configuración
o estadísticas bloquea solo sus contrastes afectados; **no sustituir stats ni
editar el run antiguo para forzar que pase**. Revisar el error y recuperar el
contrato/estadísticas originales. Si no están disponibles los controles de clase,
se informa: el contraste sigue la receta Fase 7 del repositorio, pero todavía no
acredita equivalencia con el run histórico comunicado.

Si hay varios controles con el mismo nombre y contratos distintos, copiar la suite
a un archivo nuevo y añadir `control_run` a la entrada afectada con la ruta del
control correcto. Ese campo selecciona el snapshot a verificar; no autoriza cambiar
hiperparámetros ni estadísticas. Conservar la suite original y documentar la elección.

Preflight únicamente del modelo no afectado, si otro requiere revisión:

```bash
python -m src.check_suite --suite config/phase8/suite.json --device cuda --require-rocm --only F8_CONTROL_LR
```

No pasar `--epochs` para modificar silenciosamente esta comparación controlada.
Las configuraciones están fijadas en un máximo de 20; early stopping decide cuántas
épocas completas se realizan realmente. No se exige llegar a AUC 0,70.

## Comparación y lectura de resultados

Durante el lote se actualiza `models/BATCH_.../comparison.csv`. Para generar una
tabla nueva de Fase 8 más los dos controles Fase 7, sin sobrescribir anteriores:

```bash
python -m src.phase8
```

Si los runs están en otra raíz local: `python -m src.phase8 --root /ruta/a/models`.
La tabla usa solo resultados guardados. Pendientes y controles ausentes quedan
sin métricas, no reciben los AUC comunicados verbalmente. Se excluyen diagnósticos
parciales de esta tabla clínica; no se mezclan con los entrenamientos completos.
`Comparabilidad` contrasta el snapshot con la referencia versionada; una fila
marcada como diferente/no verificable requiere revisión, no una conclusión de mejora.

Comparar mejor AUC paciente/época, BCE validación y train eval en esa misma época,
gap, toda la trayectoria de pérdidas, precision, recall/sensibilidad, especificidad,
F1, balanced accuracy, AP paciente, parámetros y tiempo. AP se calcula con scores
continuos del mejor checkpoint si existe su CSV; no es integración trapezoidal PR.
La prevalencia pCR de referencia en validación completa es 65/220 ≈ 0,2955.
Umbral 0,5 para clasificación; el AUC ROC y AP no dependen de ese umbral.

Una accuracy de 70,45 % puede ser el clasificador siempre no pCR; revisar balanced
accuracy (0,5 para ese caso), recall, F1 y matriz. Una diferencia pequeña de AUC con
una única semilla no demuestra superioridad. No reajustar umbrales repetidamente.

Cada RUN nuevo mantiene config.py, history.csv, epoch_NNN.pt completos, best_epoch.pt,
loss_curve.png, loss_comparison.png, accuracy_curve.png, auc_curve.png y roc_curve.png.
Predicciones, matriz de confusión, histogramas, configuración JSON y estado técnico
permanecen en internals/. No se cambia la selección del mejor checkpoint.

Auditoría y límites históricos: [FASE8.md](documentation/FASE8.md).
