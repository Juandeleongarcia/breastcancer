# Fase 6 — Linux, GPU AMD y PyTorch ROCm

Abrir el repositorio en VS Code Linux, seleccionar **Python: Select Interpreter** con el entorno ROCm existente y abrir una terminal Bash nueva en la raíz. No reinstalar PyTorch ni instalar el lock CPU o `requirements.txt` sobre ese entorno. Debe estar disponible localmente `breastdcedl/dataset/train/`; Git no transporta imágenes, pesos ni entornos.

## 1. Actualizar y comprobar GPU, datos y modelos

```bash
git pull --ff-only origin main
python -m src.check_suite --device cuda --require-rocm
python -m unittest discover -s tests -v
```

El preflight exige `torch.version.hip` y GPU disponible; muestra backend `rocm`, versión HIP, GPU y memoria. Verifica los cinco modelos con forward/backward **sintético**, todas las rutas de train/validación, sus particiones y estadísticas. Debe acabar sin errores antes de entrenar. En ROCm se utiliza `--device cuda`, no `--device rocm`: es la API de PyTorch para AMD también ([documentación oficial](https://docs.pytorch.org/docs/stable/notes/hip.html)). Este repositorio se ha probado aquí en CPU Windows; la ejecución real en ROCm/Linux se verifica con esa primera orden, no se da por realizada.

Los CSV conservan exactamente sus filas y pacientes. Los hashes de identidad de partición normalizan únicamente CRLF/LF para Windows/Linux; cualquier otro cambio de contenido sigue invalidando estadísticas o contratos. Los hashes binarios originales también se conservan en los registros históricos. No editar CSV ni recalcular estadísticas con validación.

## 2. Entrenar los cinco modelos secuencialmente

```bash
python -u -m src.batch_train --suite config/phase6/rocm20_suite.json --device cuda
```

| ID | Único cambio frente a BASE_SPATIAL |
|---|---|
| BASE_SPATIAL | CNN espacial [32,64,128], dos convoluciones por bloque, pooling 1×1 |
| WEIGHT_DECAY | Weight decay 0,0005 → 0,002 |
| REGIONAL_2X2 | Pooling 1×1 → 2×2 y entrada del Linear correspondiente |
| GENTLE_AUG | Activar una política geométrica conjunta suave, sin cambios fotométricos |
| DROPOUT_050 | Dropout 0,40 → 0,50 |

La batería exige backend ROCm también al entrenar y rechaza CPU o CUDA NVIDIA.

Todos: máximo **20 épocas**, early stopping por AUC de validación **por paciente**, paciencia **5**, semilla 42, LR 0,0005, AdamW, batch 16, pérdida no ponderada y el mismo scheduler plateau. Preprocesamiento PRE/EARLY−PRE/LATE−PRE, clipping y normalización train intactos. Todo train (877 pacientes / 8745 cortes) y validación interna (220 / 2200); nunca test. La paciencia común 5 es parte del protocolo actual; no se presupone que toda la configuración del histórico 0,6360 esté verificada. No se garantiza repetir ese AUC ni alcanzar 0,70. WEIGHT_DECAY obtuvo 0,6113 en dos épocas CPU completas, no veinte.

## 3. Reanudar después de una interrupción

```bash
python -u -m src.batch_train --suite config/phase6/rocm20_suite.json --device cuda --resume-interrupted
```

La orden omite completados y continúa desde el último límite de época compatible, en un RUN nuevo. Si no encuentra estado compatible, informa que inicia desde cero. **No confundir esto con continuar siempre**: revisar el mensaje de cada modelo. Para exigir una continuación concreta, usar la ruta real:

```bash
python -m src.train --resume 'models/RUN_IDENTIFICADOR/epoch_007.pt' --epochs 20 --device cuda
```

Veinte son épocas **totales**, no veinte adicionales. Conservar el RUN completo, incluido `internals/environment.json`. Se restauran modelo, optimizador, scheduler, RNG, historial y contador de early stopping. Los pesos parciales de una época no son reanudables: usar `epoch_NNN.pt` completo o `internals/recovery_epoch_start.pt`. Si ya alcanzó early stopping no se reinicia su contador. La versión mayor/menor de PyTorch debe coincidir y se rechaza cambiar backend CUDA NVIDIA↔ROCm o versión HIP guardada. Para exactitud numérica, conservar también GPU y entorno; no se promete identidad entre plataformas. Los runs CPU cortos y los GPU de veinte épocas son experimentos distintos; empezar estos últimos desde cero. No forzar ni reinstalar PyTorch para cargar un checkpoint incompatible.

## 4. Revisar resultados

Cada RUN único incluye fecha/hora Madrid, configuración y UUID. En la raíz están **solo** `config.py`, `history.csv`, todos los `epoch_NNN.pt`, `best_epoch.pt` y `loss_curve.png`, `accuracy_curve.png`, `auc_curve.png`, `roc_curve.png`. Técnicos, predicciones y diagnósticos están en `internals/`. Los runs anteriores no se migran ni sobrescriben.

Cada lote crea `models/BATCH_.../comparison.csv`, actualizado después de cada modelo, con **las cinco filas desde el principio**: métricas de completados/omitidos y estados de pendientes o errores. Incluye train loss online de la mejor época, val loss, AUC paciente/corte, accuracy, precision, recall, F1 y especificidad; configuración y estado de ejecución. `results.json` registra decisiones; `errors.jsonl` guarda tracebacks si hay errores. Revisar estados: un error no se convierte en resultado válido y el ejecutor continúa con los demás.

Las curvas accuracy/AUC/ROC principales son por paciente. Loss es por corte. Train es **online**, con pesos cambiantes y dropout/augmentation si corresponde; validación usa pesos fijos al final de época. No interpretar su diferencia como medida pura de sobreajuste. Seleccionar por AUC continuo, no por accuracy mayoritaria. Accuracy 70,45% puede acompañar recall 0: mirar matriz, F1 y distribución de puntuaciones al umbral fijo 0,50. Los resultados históricos 0,6360 y CPU 0,6113 son referencias separadas, no métricas inventadas para esta batería.

El commit de clase conservado contiene configuraciones Fase 5 que apuntan a `models/stats_baseline_clase.json`, archivo local no presente en Git ni en este ordenador. Esta batería usa explícitamente `config/phase3/stats/baseline_clipped.json`, perfil train verificado y versionado. Si el perfil de clase tenía otros valores, no se puede afirmar una reproducción exacta del histórico 0,6360: contrastar ese archivo en clase antes de atribuir diferencias al modelo. Las cinco configuraciones nuevas comparten exactamente el mismo perfil, y nunca se lo sustituye silenciosamente.
