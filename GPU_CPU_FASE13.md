# Fase 13 — ejecución CPU / GPU ROCm

Versión experimental multimodal independiente. No se ha elegido como modelo final
de Streamlit. No se han entrenado folds reales desde Codex ni evaluado el test
externo reservado. Las pruebas de código utilizan únicamente datos ficticios.

## 1. Entorno y datos en Windows

Abrir la raíz del proyecto en VS Code. Usar el entorno existente, sin copiar el
entorno del portátil al PC GPU:

```powershell
.\.env\Scripts\python.exe -c "import torch, numpy, pandas, sklearn, PIL, matplotlib; print(torch.__version__); print('GPU:', torch.cuda.is_available(), 'HIP:', torch.version.hip)"
.\.env\Scripts\python.exe -m pip check
.\.env\Scripts\python.exe -B -m unittest discover -s tests -p test_phase13.py -v
.\.env\Scripts\python.exe -m src.phase13 check --device cpu --check-images
```

El último comando valida metadatos, unión clínica, folds, disponibilidad de todas
las imágenes de desarrollo y un forward con tensores sintéticos. No entrena ni
abre imágenes externas. `audit` ofrece el informe sin crear la red:

```powershell
.\.env\Scripts\python.exe -m src.phase13 audit --check-images
```

Datos esperados: `metadata/samples.csv`, `metadata/patients.csv` y las imágenes
originales bajo `breastdcedl/dataset/`. El hash oficial del índice se conserva en
`config/phase9/protocol.json`. No se descargan datos o pesos automáticamente.

## 2. Prueba técnica con el fold 0 completo

```powershell
.\.env\Scripts\python.exe -u -m src.phase13 smoke --device cpu --epochs 1 --workers 0
```

**No es una prueba de pocos batches:** utiliza todos los pacientes y cortes de
train y validación del fold 0, resolución 256 × 256. Incluye ajuste de estadísticas,
train online, train eval y validación. En CPU puede tardar bastante. Se limita
solo el número de épocas; no se reduce silenciosamente el dataset.

La consola imprime la carpeta `CV13_...SMOKE_TEST...`. Para recuperar la más reciente:

```powershell
$smoke = (Get-ChildItem models/phase13 -Directory -Filter '*SMOKE_TEST*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
.\.env\Scripts\python.exe -m src.phase13 status --campaign "$smoke"
.\.env\Scripts\python.exe -u -m src.phase13 run --campaign "$smoke" --folds 0 --device cpu --resume-interrupted
```

Un fold completo se omite al repetir el comando. `SMOKE_TEST` permanece fijado
para esa campaña y sus métricas nunca entran en la CV definitiva, OOF o recomendación
de épocas. Sus curvas e historial sí permiten inspeccionar el pipeline y medir
`epoch_time_seconds`. La estimación del coste de la campaña debe considerar además
el ajuste inicial de estadísticas y el coste de escribir gráficas/checkpoints.

## 3. Crear la campaña definitiva

Configuración única: `config/phase13/SPATIAL_SMALL_MULTIMODAL.json`.
Máximo 20 épocas, paciencia 8, selección por ROC AUC de paciente.

```powershell
$cv = & .\.env\Scripts\python.exe -m src.phase13 prepare --epochs 20 --workers 0 --print-path
Write-Output $cv
```

Guardar esa ruta. No volver a ejecutar `prepare` para continuar: crearía una
campaña nueva, con inicializaciones nuevas.

Entrenar los cinco folds en CPU, secuencialmente:

```powershell
.\.env\Scripts\python.exe -u -m src.phase13 run --campaign "$cv" --device cpu
```

También se pueden ejecutar progresivamente:

```powershell
.\.env\Scripts\python.exe -u -m src.phase13 run --campaign "$cv" --device cpu --folds 0 1 2
.\.env\Scripts\python.exe -u -m src.phase13 run --campaign "$cv" --device cpu --folds 3 4
```

Cada fold empieza desde cero con semilla 42; no hereda pesos de otro fold.
Un bloqueo del sistema operativo impide ejecutar dos procesos de entrenamiento
sobre la misma campaña. Se libera también si el proceso termina abruptamente.

## 4. Consultar y reanudar

```powershell
.\.env\Scripts\python.exe -m src.phase13 status --campaign "$cv"
.\.env\Scripts\python.exe -u -m src.phase13 run --campaign "$cv" --device cpu --resume-interrupted
```

Usar `Ctrl+C` para interrumpir. La recuperación conserva pesos, BatchNorm,
optimizador, scheduler, historial, mejor época, contador de paciencia y RNG de
Python/NumPy/Torch/DataLoader. Se crea un RUN nuevo. La época incompleta se repite
desde su inicio; las épocas completas y el mejor checkpoint anterior se conservan.

No borrar `epoch_*.pt`, `best_epoch.pt` ni `internals/recovery_epoch_start.pt` de
una ejecución que deba reanudarse. No mezclar reanudación con otro experimento.
Cambiar fuentes, clínica, particiones o parámetros científicos bloquea la
continuación. Para aceptar explícitamente CPU → GPU o viceversa, añadir
`--allow-device-change`; eso puede cambiar la trayectoria numérica. No garantiza
igualdad entre versiones/hardware aunque se conserve el estado.

## 5. Linux con AMD ROCm

No sustituir el PyTorch ROCm de clase ni instalar el lock CPU de Windows.
Esta fase no añade dependencias. Con el entorno de clase ya operativo:

```bash
git pull --ff-only origin main
conda activate cancer
python -c "import torch; print(torch.__version__, torch.version.hip); assert torch.cuda.is_available() and torch.version.hip; print(torch.cuda.get_device_name(0))"
python -m pip check
python -B -m unittest discover -s tests -p test_phase13.py -v
python -m src.phase13 check --device cuda --check-images
CV=$(python -m src.phase13 prepare --epochs 20 --workers 0 --print-path)
printf '%s\n' "$CV"
python -u -m src.phase13 run --campaign "$CV" --device cuda --resume-interrupted
```

En ROCm, la API de PyTorch sigue llamándose `cuda`. `--device auto` detecta el
backend disponible; `cpu` y `cuda` son selecciones explícitas. El código se ha
probado en CPU Windows; la ejecución física ROCm debe verificarse en clase.

Para continuar otro día, recuperar la **misma** ruta, sin usar `prepare`:

```bash
CV='models/phase13/RUTA_REAL_DE_LA_CAMPANA'
python -m src.phase13 status --campaign "$CV"
python -u -m src.phase13 run --campaign "$CV" --device cuda --resume-interrupted
```

## 6. Curvas, comparación y mejores épocas

Se generan automáticamente al terminar cada fold. Estos comandos pueden repetirse
y crean informes nuevos, sin reescribir informes anteriores:

```powershell
.\.env\Scripts\python.exe -m src.phase13 summarize --campaign "$cv"
.\.env\Scripts\python.exe -m src.phase13 plots --campaign "$cv" --loss-ymin 0 --loss-ymax 1
.\.env\Scripts\python.exe -m src.phase13 recommend --campaign "$cv"
```

`loss_curve.png` conserva escala automática para apreciar sobreajuste. La versión
con eje limitado queda en `internals/loss_curve_limited.png`; no modifica el CSV
y señala que debe consultarse también la versión sin recorte. El comando `plots`
guarda otra copia en `reports/PLOTS_.../`, con los límites solicitados.

Las medias no prolongan curvas después del early stopping; incluyen DE muestral
y número de folds por época. OOF, matriz global y recomendación se habilitan solo
con cinco folds completos y verificados, nunca con `SMOKE_TEST`.

Para comparar con la CNN de imagen, aportar la campaña original real, con sus
cinco folds, historiales, resúmenes, particiones y predicciones. No hacen falta pesos:

```powershell
.\.env\Scripts\python.exe -m src.phase13 compare --campaign "$cv" --baseline-campaign 'models/RUTA_REAL_CV_ORIGINAL'
```

No se convierte la referencia aproximada comunicada por el usuario en resultados
de una campaña local ni se presupone que sus archivos estén disponibles.

## 7. Entrenamiento final — solo después de elegir la versión

El programa genera `recommended_final_training.json` tras verificar cinco folds.
Contiene sus mejores épocas, mediana, media, DE, configuración y recomendación.
El número recomendado se basa en la mediana, sin usar test ni garantizar un óptimo.

**Ejecutar lo siguiente únicamente si se decide seleccionar esta versión multimodal:**

```powershell
.\.env\Scripts\python.exe -u -m src.phase13 final-train --campaign "$cv" --device cpu --confirm-selection
```

En GPU, usar `python -u -m src.phase13 final-train --campaign "$CV" --device cuda --confirm-selection`.
Se utilizan todos los pacientes de desarrollo, con estadísticas nuevas de desarrollo
y pesos inicializados desde cero. No hay validación ni early stopping en este ajuste.

El scheduler adaptativo de CV no puede consultar validación cuando todos los
pacientes son train. Por ello se fija antes de empezar un calendario con la mediana
del LR observado en CV en cada época recomendada. No se utiliza la pérdida de test
ni se adapta el calendario a resultados externos. Esta diferencia está registrada
en la recomendación y en el contrato del entrenamiento final.

La consola imprime la carpeta final. Para continuar una interrupción:

```powershell
$final = 'models/phase13/RUTA_CV/final_models/RUTA_FINAL'
.\.env\Scripts\python.exe -u -m src.phase13 final-train --campaign "$cv" --final-run "$final" --device cpu --confirm-selection --resume-interrupted
```

Se guarda `final_epoch.pt`, seleccionado por el número de épocas fijado, sin
afirmar que sea el «mejor» mediante una validación inexistente.

## 8. Test externo — preparado, NO ejecutarlo todavía

Tras terminar y analizar CV, comparar versiones, seleccionar explícitamente el
modelo final y entrenarlo, se podrá utilizar este comando independiente:

```powershell
.\.env\Scripts\python.exe -m src.phase13 external-test --final-run "$final" --device cpu --confirm-external-test
```

Linux: `python -m src.phase13 external-test --final-run "$FINAL" --device cuda --confirm-external-test`.
Definir `FINAL` con la ruta real del entrenamiento final.

El umbral queda fijado en 0,50 antes de abrir test. Se aplican los transformadores
guardados sin ajustarlos de nuevo. La evaluación externa escribe en una carpeta
independiente, registra el hash del checkpoint y bloquea una segunda evaluación
silenciosa de ese modelo. Un fallo tras iniciar test deja la reserva bloqueada
para revisión, sin volver a ejecutar automáticamente. No cambiar umbral, épocas,
pesos o configuración basándose en esos resultados.

Streamlit permanece intacto y sin un modelo definitivo integrado.
