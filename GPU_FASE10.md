# Fase 10: completar tres validaciones cruzadas en Linux / ROCm

Ejecutar desde la raíz de `breastcancer`, en la terminal Bash de VS Code del
ordenador de clase. Cada comando de entrenamiento ocupa la terminal hasta terminar.
Ejecutar **un comando cada vez**, esperando a que finalice antes del siguiente.
No se instala ni modifica PyTorch, ROCm, los drivers o Ubuntu.

Estos cambios se han preparado en Windows sin entrenar modelos clínicos. La campaña
de clase no está en este ordenador: el estado comunicado (0/1 completos, 2 fallido,
3/4 pendientes) se verificará con los archivos reales al ejecutar `status`.

## 1. Actualizar el código y activar el entorno

La Fase 10 debe haberse publicado previamente con autorización. **Mientras los
archivos sigan solamente en Windows, `git pull` no los transportará a clase.**
Este trabajo no hace commit ni push automáticamente.

```bash
git status --short
git pull --ff-only origin main
conda activate cancer
python -m src.phase10 --help
python -m src.phase10 configs
```

Conservar el entorno `cancer` existente; no copiar el entorno CPU de Windows.
Si Git muestra cambios locales, revisarlos antes de actualizar; no usar `reset
--hard` ni borrar `models/`. Las campañas y el dataset continúan siendo locales.

## 2. Comprobar AMD / ROCm

```bash
python -c 'import torch; print("PyTorch:", torch.__version__, "HIP:", torch.version.hip, "GPU disponible:", torch.cuda.is_available()); assert torch.version.hip and torch.cuda.is_available(), "Se requiere PyTorch ROCm y GPU accesible"; print(torch.cuda.get_device_name(0))'
python -m src.batch_train --check-device cuda
```

La segunda prueba realiza forward/backward sintético y sincroniza la GPU. En ROCm
PyTorch utiliza la interfaz `torch.cuda` y el argumento `cuda` también con AMD.
Cada comando Fase 10 repite este control en un proceso nuevo antes de entrenar.
Si falla, detenerse y revisar el error; no intentar continuar en CPU.

## 3. Localizar e inspeccionar SPATIAL_SMALL

```bash
CV_SMALL="models/CV9_20261028_183759_SPATIAL_SMALL_531c826715d2"
test -f "$CV_SMALL/campaign.json"
python -m src.phase10 status --campaign "$CV_SMALL"
```

Usar esta campaña existente; **no ejecutar `prepare` para SPATIAL_SMALL**.
Si falta la ruta, localizar su copia real en el PC de clase y asignarla a
`CV_SMALL`. No crear una campaña de sustitución.

`status` comprueba hashes del código, metadatos, particiones y estadísticas ya
ajustadas. Para considerar un fold completo exige resumen, historial, checkpoints,
gráficas y predicciones que cubran exactamente sus pacientes de validación.
Muestra `COMPLETO`, `RECUPERABLE`, `SIN_CHECKPOINT` o `PENDIENTE`, junto con rutas y
causas de rechazo de checkpoints. No entrena ni cambia esos resultados.
Si los archivos declarados completos son inconsistentes, se detiene para revisión;
no repite el fold automáticamente.

## 4. Paso A: recuperar fold 2 y terminar SMALL

Primero recuperar únicamente el fold fallido:

```bash
python -u -m src.phase10 small --campaign "$CV_SMALL" --folds 2
```

Se selecciona el checkpoint compatible con mayor época completada, incluyendo el
checkpoint de inicio de época si quedó una época interrumpida. Se verifican el
entorno, la configuración, el código y las particiones. La reanudación restaura
pesos, optimizador, scheduler, RNG, historial y contador de early stopping en un
**RUN nuevo**, conservando los anteriores. Repite una época incompleta desde su
inicio; no continúa desde sus pesos parciales.

Si no hay checkpoint compatible, el comando se detiene y registra los motivos.
Si tras revisar el error procede empezar fold 2 desde cero, el comando explícito
es:

```bash
python -u -m src.phase10 small --campaign "$CV_SMALL" --folds 2 --restart-fold 2
```

Este último comando inicia un RUN independiente aunque exista un checkpoint
compatible; documenta el reinicio en `campaign.json` y conserva todos los intentos.
Nunca reinicia un fold que ya se ha verificado completo.

Después de completar fold 2:

```bash
python -u -m src.phase10 small --campaign "$CV_SMALL" --folds 3 4
python -m src.phase10 status --campaign "$CV_SMALL"
python -m src.cross_validation summarize --campaign "$CV_SMALL"
```

También se puede completar todo lo pendiente con **un único comando A**:

```bash
python -u -m src.phase10 small --campaign "$CV_SMALL"
```

Los folds completos se omiten, incluidos 0 y 1. Si una interrupción dejó resultados
completos pero el manifiesto desactualizado, se reconcilia su estado sin entrenarlos
otra vez. No se borran ni modifican sus RUNs. El resumen de Fase 9 indica
`provisional: true` mientras no haya cinco folds completos; no interpretarlo como
resultado final. El controlador también registra el error antiguo del fold 2 en
el historial de intentos, aunque después se haya recuperado.

## 5. Pasos B y C: candidatos F8

Una vez terminado SMALL, ejecutar **B**, esperar a que termine y después ejecutar
**C**:

```bash
python -u -m src.phase10 dropout --campaign "$CV_SMALL"
python -u -m src.phase10 lr --campaign "$CV_SMALL"
```

Son dos comandos independientes; si B falla, resolverlo antes de iniciar C. No
usar `src.cross_validation prepare --model F8_SMALL_DROPOUT`: esa interfaz de Fase
9 solo acepta `SPATIAL_SMALL` y `SPATIAL_CONTROL`. El adaptador Fase 10 utiliza el
preparador existente con las configuraciones originales F8 y sin modificarlo.

| Parámetro | SMALL (control F9) | F8_SMALL_DROPOUT | F8_SMALL_LR |
|---|---:|---:|---:|
| Filtros, dos convoluciones por bloque | 16, 32, 64 | 16, 32, 64 | 16, 32, 64 |
| Dropout | 0,40 | **0,50** | 0,40 |
| Learning rate inicial | 0,0005 | 0,0005 | **0,0001** |
| Optimizador | AdamW | AdamW | AdamW |
| Weight decay | 0,0005 | 0,0005 | 0,0005 |
| Batch / semilla | 16 / 42 | 16 / 42 | 16 / 42 |
| Máximo épocas / paciencia | 20 / 8 | 20 / 8 | 20 / 8 |

Comunes: BatchNorm, pooling espacial 1×1 final, loss no ponderada, sin augmentation,
scheduler plateau por AUC paciente (factor 0,5; paciencia 2; LR mínimo 0,000001),
evaluación completa de train al final de cada época y umbral fijo 0,5.
`small` imprime además los valores **reales** del template de la campaña existente;
no los sustituye por valores de esta tabla.

Se conserva la representación enhancement de PRE/EARLY/LATE, clipping y
estandarización. La única adaptación necesaria respecto a F8 es sustituir las
estadísticas de la antigua partición interna por las ajustadas en los cuatro folds
de entrenamiento de cada fold CV. Nunca entran su validación ni `fold=-1`.
Cada fold empieza con un modelo nuevo y semilla 42; no hereda pesos de otros folds.

Las campañas B/C quedan bajo `models/phase10/campaigns/<candidato>/CV9_.../`.
Sus rutas exactas se conservan en `models/phase10/campaigns.json`. Repetir el mismo
comando recupera esa campaña, omite lo completo y reanuda lo interrumpido.
No cambiar `--state-dir` ni eliminar el registro si se quiere continuar esas mismas
campañas. Si se interrumpió su creación antes de guardar el registro, se adopta la
única campaña del candidato; ante varias candidatas se detiene para revisión.

Para ver el estado de B:

```bash
CV_DROP=$(python -c 'import json; print(json.load(open("models/phase10/campaigns.json"))["F8_SMALL_DROPOUT"])')
python -m src.phase10 status --campaign "$CV_DROP"
```

Para C, sustituir la clave por `F8_SMALL_LR`. Se pueden ejecutar grupos progresivos
con `--folds 0 1 2` y después `--folds 3 4`; el comando sin `--folds` procesa todo lo
pendiente. `--restart-fold N` también está disponible para B/C, con las mismas
condiciones explícitas de conservación de resultados.

## 6. Interrupciones, logs y duplicados

Se lanza un proceso Python nuevo por fold. Ante un error GPU se registra el estado
incompleto y se detiene la orden; no se lanzan otros folds ni se reintenta en bucle.
Volver a lanzar el comando tras comprobar la GPU inicia un proceso nuevo.
El error `CUDA error` en esta instalación no implica una GPU NVIDIA.

Con `Ctrl+C`, pulsar una vez y esperar a que el hijo termine y guarde la recuperación
cuando sea posible. Un fallo duro de GPU/energía puede impedir el guardado final;
se buscará el último límite de época compatible que sí exista. No borrar sus
archivos ni sustituir manualmente el checkpoint mejor por el último.

Logs acumulativos (salida del controlador y del hijo):

```text
models/phase10/logs/SPATIAL_SMALL.log
models/phase10/logs/F8_SMALL_DROPOUT.log
models/phase10/logs/F8_SMALL_LR.log
models/phase10/logs/comparison.log
```

En otra terminal se puede consultar, por ejemplo:

```bash
tail -f models/phase10/logs/SPATIAL_SMALL.log
```

Un bloqueo exclusivo impide dos controladores Fase 10 simultáneos. También se
comprueban los procesos de los ejecutores anteriores antes de preparar/lanzar un
fold. Si hay uno activo, se informa del PID y no se inicia otro. No se mata ningún
proceso. No lanzar manualmente ejecutores antiguos mientras corre Fase 10: estos
no participan de su bloqueo. Si `/proc` no permite comprobarlos, se detiene.

## 7. Consultar resultados y comparar

Cada RUN conserva el formato vigente de Fase 9:

```text
<campaña>/fold_N/runs/RUN_FECHA_HORA_CONFIG_UUID/
├── config.py
├── history.csv
├── epoch_001.pt ... epoch_NNN.pt
├── best_epoch.pt
├── loss_curve.png
├── loss_comparison.png
├── accuracy_curve.png
├── auc_curve.png
├── roc_curve.png
└── internals/
```

Las épocas de un intento anterior continúan en su RUN original; una reanudación
conserva el historial completo y guarda los nuevos checkpoints en su nuevo RUN.
`internals/` contiene configuración JSON, entorno, predicciones, matrices de
confusión, diagnóstico y estados técnicos. El mejor checkpoint se selecciona
por AUC paciente; `roc_curve.png` corresponde a sus predicciones de validación.

En `history.csv` comparar `train_eval_loss` con `val_loss`,
`train_eval_patient_accuracy` con `val_patient_accuracy` y
`train_eval_patient_auc` con `val_patient_auc`. Son evaluaciones con los mismos
pesos finales y sin augmentation. `train_loss` y `train_patient_*` son métricas
online con pesos cambiantes; no tratarlas como esa evaluación comparable.
`loss_comparison.png` muestra las pérdidas de evaluación comparables.

Al completar una campaña se generan snapshots únicos `fold_comparison_*.csv`,
`cv_summary_*.json`, `oof_predictions_*.csv` y `oof_roc_*.png` dentro de ella.
Incluyen AUC medio, desviación estándar muestral y AUC OOF paciente. La tabla de
folds incluye ambas pérdidas, AUC paciente/corte, accuracy, balanced accuracy,
sensibilidad, especificidad y F1. El historial de cada RUN contiene también
accuracy y AUC de train evaluado, por paciente y corte.

La comparación de los tres modelos se actualiza al terminar cada orden correcta
y se puede regenerar sin entrenamiento:

```bash
python -m src.phase10 compare --campaign "$CV_SMALL"
```

Consultar los nuevos `comparison_FECHA_HORA_UUID.csv`, `folds_*.csv` y
`report_*.json` en `models/phase10/reports/`. La tabla por fold da la ruta del
checkpoint seleccionado, época, pérdidas y métricas. Las campañas incompletas se
marcan `INCOMPLETA`, sin inventar medias finales ni AUC OOF. No se mezclan métricas
históricas de una sola partición con estas cinco validaciones cruzadas.

Esta es CV de desarrollo: el checkpoint se selecciona en el mismo fold validado.
El OOF no equivale a una evaluación independiente del conjunto reservado.
`fold=-1` permanece fuera del entrenamiento, las estadísticas y la selección.

## 8. Comprobaciones técnicas opcionales

```bash
python -B -m unittest discover -s tests -p 'test_phase10.py' -v
```

Estas pruebas usan temporales y mocks; no entrenan modelos clínicos ni necesitan
la campaña de clase. La suite completa es `python -B -m unittest discover -s tests
-v`; incluye pruebas sintéticas pequeñas, sin entrenamientos completos.
