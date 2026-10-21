# Fase 5: entrenar mañana en Windows / VS Code sin Codex

Abrir la carpeta del proyecto y seleccionar **Python: Select Interpreter** con el entorno GPU existente del ordenador de clase. Abrir una terminal PowerShell nueva en la raíz del proyecto. No copiar el entorno `.env` del portátil ni instalar el lock CPU sobre el PyTorch GPU de clase.

Actualizar primero desde la raíz del repositorio, después de que el commit y el push de cierre hayan sido autorizados y realizados. `--ff-only` evita crear un merge automático. El dataset debe estar disponible localmente en `breastdcedl/`; Git no lo transporta.

```powershell
git pull --ff-only origin main
```

Las pruebas utilizan imágenes sintéticas temporales; no entrenan sobre pacientes ni crean runs clínicos.

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -m src.batch_train --check-device cuda
python -m unittest discover -s tests -v
```

CUDA debe estar disponible (`True`). La segunda orden hace forward/backward real y comprueba valores finitos; falla si se pide CUDA y no existe. Aquí se ha probado CPU, no CUDA. Si faltan herramientas auxiliares, `requirements-gpu-tools.txt` excluye torch/torchvision; conservar el PyTorch GPU existente.

## Batería principal de tres modelos

```powershell
python -m src.batch_train --suite config/phase5/gpu_suite.json --only BASE_UNWEIGHT SPATIAL_UNWEIGHT AUG_UNWEIGHT --device cuda --resume-interrupted
```

- `BASE_UNWEIGHT`: CNN original [32,64,128], una convolución por bloque, sin ponderación, sin augmentation.
- `SPATIAL_UNWEIGHT`: solo añade una segunda convolución por bloque.
- `AUG_UNWEIGHT`: solo activa la geometría suave y sincronizada en la CNN original.

Los tres conservan LR 0,0005, dropout 0,4, BatchNorm, semilla 42, batch 16 y clipping/normalización de train verificados. El protocolo GPU común utiliza hasta diez épocas, early stopping con paciencia 4 y ReduceLROnPlateau por AUC de paciente con paciencia 2. No se comparan directamente sus diez épocas con la batería CPU corta sin scheduler.

Para un único modelo, con omisión automática de completados:

```powershell
python -m src.batch_train --suite config/phase5/gpu_suite.json --only AUG_UNWEIGHT --device cuda --resume-interrupted
```

## Comparación opcional de representación de fases

```powershell
python -m src.batch_train --suite config/phase5/gpu_suite.json --only BASE_WEIGHTED_NOCLIP RAW_WEIGHTED_NOCLIP --device cuda --resume-interrupted
```

Esta pareja mantiene pérdida ponderada y ausencia de clipping en ambos modelos. Solo cambia [PRE, EARLY−PRE, LATE−PRE] por [PRE, EARLY, LATE], con estadísticas correspondientes de train. Es una familia distinta de la batería principal; no atribuir diferencias frente al baseline sin ponderación únicamente a la representación.

## Recuperación, errores y conservación

Repetir la misma orden con `--resume-interrupted` omite completados y continúa desde límites de época compatibles. Si no hay estado compatible, indica que empieza desde cero. No mezclar `--force-new` con `--resume-interrupted`: el ejecutor lo rechaza para evitar una intención ambigua. `--force-new` crea deliberadamente nuevos runs desde cero.

Para continuar manualmente un checkpoint nuevo:

```powershell
python -m src.train --resume 'models\RUN_IDENTIFICADOR\epoch_002.pt' --epochs 10 --device cuda
```

Sustituir `RUN_IDENTIFICADOR` por el nombre real. Diez significa épocas totales. Se conservan optimizador, scheduler, historial y estados RNG. La continuación crea otra carpeta y conserva el origen. Para los runs nuevos, elegir el último `epoch_NNN.pt` completo o `internals/recovery_epoch_start.pt` si hubo una interrupción; para los históricos siguen siendo válidas sus rutas originales `checkpoints/last_epoch.pt` y `checkpoints/recovery_epoch_start_XXX.pt`; los pesos parciales de una época interrumpida son diagnósticos. Un cambio explícito CPU→GPU requiere `--allow-device-change` y la misma versión mayor/menor de PyTorch; puede cambiar la trayectoria numérica.

La incorporación de augmentation añade código y parámetros al contrato de compatibilidad. Los checkpoints de Fase 4 siguen siendo válidos para evaluación; su continuación exacta requiere sus fuentes congeladas originales. No se ignora una incompatibilidad ni se cambia su contrato para hacerla pasar. Sus fuentes están guardadas en cada run (`details/sources/`) y existe backup previo de Fase 5. Para la batería de mañana se recomienda empezar experimentos nuevos con las configuraciones Fase 5 y el PyTorch GPU existente. No reinstalar PyTorch para forzar compatibilidad con el portátil.

Cada run nuevo tiene fecha, hora, configuración y UUID. En su raíz están `config.py`, `history.csv`, todos los `epoch_NNN.pt`, `best_epoch.pt` y solo cuatro gráficas: `loss_curve.png`, `accuracy_curve.png`, `auc_curve.png` y `roc_curve.png`. Configuración JSON, resumen, predicciones, matrices, distribuciones y fuentes comprimidas están en una única carpeta `internals/`. Los runs antiguos conservan íntegramente sus rutas. Las curvas accuracy/AUC principales son por paciente: train online se dibuja discontinuo y validación de final de época en trazo continuo; no son protocolos equivalentes. Loss es BCE por corte, no loss por paciente. El AUC por corte sigue disponible en `history.csv` y el resumen técnico. Cada lote genera su propia carpeta `models/BATCH_.../` con `results.json`, `comparison.csv`, `execution.json` y `errors.jsonl` si hay fallos. Se selecciona el mejor checkpoint por AUC de paciente sobre toda la validación interna; umbral fijo 0,5. Los errores se registran y la batería continúa con el siguiente modelo.

Transportar código, `config/phase5/`, perfiles de `config/phase3/stats/`, metadatos y dataset por medios locales. Los checkpoints que se necesiten deben copiarse por separado; no subir imágenes ni pesos a GitHub. No modificar código mientras corre una batería. Ctrl+C conserva puntos de recuperación; no apagar el ordenador durante el guardado.

El cambio de organización admite continuar los checkpoints de la versión inmediatamente anterior de Fase 5 mediante una pareja exacta y auditada de hashes de fuentes; no acepta cambios arbitrarios de código o configuración. Los checkpoints históricos anteriores mantienen sus restricciones originales. Al reanudar se crea otro RUN: las épocas anteriores permanecen en el RUN de origen, identificado en `internals/resume.json`; no se duplican sus archivos. El snapshot `config.py` registra el protocolo online y la configuración JSON equivalente permite lectura automática.
