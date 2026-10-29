# Fase 11: CNN experimental de cuatro bloques

Ejecutar los comandos desde la raíz de `breastcancer` en Bash / VS Code, Linux y
el entorno conda `cancer`. No reinstalar PyTorch ni modificar ROCm o los drivers.
Esperar a que termine la campaña que actualmente está entrenando antes de hacer
pruebas GPU o iniciar F11. El ejecutor no detiene otros procesos.

## Arquitectura exacta y parámetros

Entrada `[batch, 3, 256, 256]`: tres canales temporales DCE-MRI, no RGB.
Cada convolución usa kernel 3×3, padding 1, **sin bias**, seguida de BatchNorm y
ReLU. Las convoluciones de cada bloque son:

| Bloque | Convoluciones | Pooling al final | Salida espacial |
|---|---|---|---|
| 1 | 3 → 8 | MaxPool 2×2 | 8 × 128 × 128 |
| 2 | 8 → 8 → 16 | MaxPool 2×2 | 16 × 64 × 64 |
| 3 | 16 → 16 → 32 | MaxPool 2×2 | 32 × 32 × 32 |
| 4 | 32 → 32 → 64 → 64 | Ninguno | 64 × 32 × 32 |

Clasificador: `AdaptiveAvgPool2d(1) → Flatten → Dropout(0.30) → Linear(64,1)`.
La salida es un único logit; no hay Sigmoid dentro del modelo. Se utiliza
`BCEWithLogitsLoss` y solo se aplica Sigmoid para calcular probabilidades/métricas.

**73.913 parámetros entrenables**: 73.368 pesos de convoluciones, 480 parámetros
de BatchNorm y 65 del clasificador. Son 87 menos que 74.000; la cifra aproximada
es coherente. Las estadísticas acumuladas de BatchNorm son buffers, no parámetros
entrenables. No se ha alterado la arquitectura para ajustar el recuento.

## Configuración inicial

La configuración independiente es `config/phase11/F11_CNN4.json`:

| Parámetro | Valor |
|---|---|
| Optimizador / LR inicial | AdamW / 0,001 |
| Weight decay | 0,0001 |
| Batch / semilla | 16 / 42 |
| Máximo de épocas | 10 |
| Dropout | 0,30 |
| Scheduler | ReduceLROnPlateau, mode=max, AUC paciente |
| Factor / paciencia del scheduler | 0,5 / 3 |
| LR mínimo | 0,000001, conservado del pipeline |
| Early stopping | AUC paciente; paciencia 8, conservada de F9/F10 |
| Loss | BCEWithLogitsLoss no ponderada, como F9/F10 |
| Augmentation | Desactivada |
| Umbral para métricas binarias | 0,5, fijo |

Una AUC no finita detiene el fold antes de avanzar el scheduler o seleccionar un
checkpoint. No se ajustan umbrales ni se buscan hiperparámetros en esta fase.

Se conserva enhancement: `[PRE, EARLY−PRE, LATE−PRE]`, el ajuste de clipping 1–99 %
y la estandarización del pipeline F9. Cada fold calcula sus estadísticas solo con
sus cuatro folds de entrenamiento, sin validación ni test (`fold=-1`). No hay
normalización ImageNet, augmentation RGB, transfer learning ni pesos descargados.

Los bloques heterogéneos están declarados mediante `ARCHITECTURE`, `BLOCK_CHANNELS`
y `POOL_AFTER_BLOCK`. El campo histórico `CONVS_PER_BLOCK=2` permanece únicamente
para compatibilidad con el validador antiguo; **no construye esta arquitectura**.
Su distribución real es `[1,2,2,3]`, registrada también en el resumen de cada RUN.

F11 cambia además LR, weight decay, dropout y horizonte respecto a SMALL, siguiendo
la configuración solicitada. Una diferencia de rendimiento no puede atribuirse
exclusivamente a la arquitectura. No mezclar una campaña parcial con CV completa
ni presentar estos ensayos como garantía de mejorar AUC.

## 1. Actualizar y comprobar GPU

Los archivos F11 deben haberse publicado después de su revisión y autorización.
Mientras sigan solamente en Windows, `git pull` no los descargará. Este trabajo
no hace commit ni push automáticamente.

```bash
git status --short
git pull --ff-only origin main
conda activate cancer
python -m src.phase11 --help
python -m src.phase11 check --device cuda
```

`check` exige PyTorch ROCm con GPU accesible, prueba la CNN real con un tensor
`[2,3,256,256]`, calcula BCE y ejecuta backward, sin `optimizer.step` ni datos
clínicos. Debe mostrar `trainable_parameters: 73913`, `output_shape: [2,1]`, HIP y
`forward_backward_passed: true`. En AMD se sigue usando el dispositivo `cuda`.
En Windows, la misma prueba sintética se ejecuta con `--device cpu`.

## 2. Opción A: solamente fold 0

```bash
python -u -m src.phase11 run --folds 0
python -m src.phase11 status
python -m src.phase11 summarize
```

Se prepara una campaña F11 nueva una sola vez y se entrena únicamente el fold 0.
Esta prueba usa **todo el entrenamiento y la validación oficial de ese fold**,
con máximo de diez épocas; no es un smoke test ni usa subconjuntos. Los demás
folds quedan pendientes y el resumen es `provisional: true`, sin AUC OOF final.
Podemos detenernos aquí para revisar resultados antes de continuar.

Los cinco folds proceden de `metadata/samples.csv`. No se crean nuevos splits.
Cada fold inicia un modelo propio desde cero con semilla 42; no se transfieren
pesos entre folds ni desde SMALL/F8. Se omiten folds completos verificados.

## 3. Opción B: continuar con los otros cuatro folds

```bash
python -u -m src.phase11 run --folds 1 2 3 4
python -m src.phase11 status
python -m src.phase11 summarize
```

También se puede procesar todo lo pendiente con:

```bash
python -u -m src.phase11 run
```

La ruta exacta persiste en `models/phase11/route.json`; los comandos repetidos la
reutilizan, sin repetir fold 0 si ya está completo. No eliminar ese registro ni
cambiar `--state-dir` para continuar la misma campaña.

Consultar la ruta:

```bash
CV11=$(python -c 'import json; print(json.load(open("models/phase11/route.json"))["campaign"])')
printf '%s\n' "$CV11"
```

La campaña se encuentra bajo `models/phase11/campaigns/CV9_.../`. El preparador
histórico utiliza `SPATIAL_SMALL` en el nombre físico de esa carpeta; el manifiesto,
template, RUNs y checkpoints la identifican como **F11_CNN4**. Es una campaña
independiente y nunca reutiliza ni renombra las carpetas SMALL de clase.

## 4. Interrupciones y errores GPU

Repetir el mismo comando para continuar:

```bash
python -u -m src.phase11 run --folds 0
```

O repetir `run --folds 1 2 3 4` si esos eran los folds pendientes. Se recupera el
último límite de época compatible, restaurando pesos, optimizador, scheduler,
RNG, historial y contador de early stopping. Una época interrumpida se repite
desde su inicio; no se utiliza un checkpoint de pesos parciales para reanudar.

Cada continuación crea un RUN nuevo y conserva intactos los anteriores. Si no
existe checkpoint compatible, se detiene y explica el motivo. Solo después de
revisar el error, un reinicio explícito de fold 0 desde cero sería:

```bash
python -u -m src.phase11 run --folds 0 --restart-fold 0
```

Este comando conserva los intentos previos y registra el reinicio. Un fold ya
completo se omite incluso con esa opción. Sustituir 0 por el fold afectado.

Se usa un proceso Python nuevo por fold. Ante un fallo GPU se detiene la orden,
sin iniciar otro fold ni reintentar en bucle. Antes de repetirla, comprobar de
nuevo la GPU en un proceso limpio. Con `Ctrl+C`, pulsar una vez y esperar al cierre
y guardado de recuperación cuando sea posible. Un fallo duro puede impedir el
último guardado; se conserva lo que ya se haya escrito correctamente.

El bloqueo de F10/F11 y la comprobación de ejecutores antiguos evitan lanzamientos
simultáneos desde estos controladores. No lanzar manualmente `src.train` u otros
ejecutores en paralelo; los antiguos no participan de ese bloqueo. El worker
`src.phase11 train` es interno; gestionar las campañas mediante `run`.

Log acumulativo:

```bash
tail -f models/phase11/logs/F11_CNN4.log
```

## 5. Resultados y mejor checkpoint

Cada RUN utiliza el formato existente, con fecha, hora e identificador único:

```text
<campaña>/fold_N/runs/RUN_FECHA_HORA_F9_F11_CNN4_FOLDN_UUID/
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

`status` muestra para cada fold completo su `best_epoch`, `best_checkpoint`, RUN,
AUC paciente y pérdidas. `best_epoch.pt` corresponde a la mayor AUC de validación
por paciente; no necesariamente a la última época. La ROC usa sus predicciones.
Las épocas anteriores de una reanudación siguen en el RUN original; el nuevo
guarda el historial acumulado y los nuevos checkpoints, sin duplicar todos los pesos.

Comparar `train_eval_loss` con `val_loss`, `train_eval_patient_accuracy` con
`val_patient_accuracy` y `train_eval_patient_auc` con `val_patient_auc`. Se evalúan
con los mismos pesos finales y sin augmentation. `train_loss` y `train_patient_*`
son métricas online, con pesos cambiantes; no son esa evaluación comparable.
`loss_comparison.png` muestra las pérdidas de evaluación comparables.

`internals/` conserva las predicciones por paciente/corte, matriz de confusión
2×2, configuración JSON, entorno y fuentes. Los checkpoints contienen el estado
completo y un contrato específico de arquitectura y hashes F11. No usar los
ejecutores genéricos `src.train` o `src.evaluate` con estos checkpoints. Para
reevaluar, copiar la ruta `best_checkpoint` que muestra `status` en la variable
`BEST` y usar la interfaz correspondiente:

```bash
# Definir BEST con la ruta real mostrada por status, entre comillas.
python -m src.phase11 evaluate "$BEST" --output "$CV11/reevaluacion_fold0_$(date +%Y%m%d_%H%M%S)"
```

La carpeta de salida debe ser nueva. Se utiliza exclusivamente la validación
guardada, y el dispositivo guardado en la configuración del checkpoint.

Al completar los cinco folds, `summarize` genera snapshots únicos de
`fold_comparison_*.csv`, `cv_summary_*.json`, `oof_predictions_*.csv` y
`oof_roc_*.png` en la campaña. Incluye AUC medio, desviación estándar muestral,
AUC OOF paciente y métricas por fold, incluida balanced accuracy. Las predicciones
OOF permiten producir posteriormente una matriz 2×2 y figuras agregadas sin
volver a entrenar. Hasta entonces el resumen permanece provisional.

Es CV de desarrollo con selección del checkpoint en el mismo fold validado; no
equivale a medir el rendimiento en el test reservado, que permanece sin utilizar.

## Compatibilidad y pruebas

No se ha editado ningún archivo existente: `src/model.py`, el entrenamiento,
F9, F10, las configuraciones previas y `.gitignore` permanecen intactos. El adaptador
enlaza temporalmente la nueva clase y amplía su contrato solo dentro del proceso
F11; al salir restaura las referencias originales. Las otras configuraciones
conservan su factory y contrato. F11 comprueba además sus propias fuentes y
configuración para impedir una recuperación incompatible tras modificarlas.

```bash
python -B -m unittest discover -s tests -p 'test_phase11.py' -v
python -B -m unittest discover -s tests -v
```

Estas pruebas usan CPU y datos sintéticos en temporales; no requieren la campaña
de clase ni entrenan modelos con el dataset clínico. La comprobación real de
ROCm y los entrenamientos quedan pendientes para el ordenador GPU.
