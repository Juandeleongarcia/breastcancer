# Ejecutar mañana en Windows / VS Code, sin Codex

**Actualización Fase 5:** usar [GPU_FASE5.md](GPU_FASE5.md) para la batería actual. La nueva augmentation amplía el contrato de código: los checkpoints Fase 4 se conservan y siguen siendo evaluables, pero su continuación exacta requiere sus fuentes congeladas originales. No se fuerza una continuación incompatible.

Preparado el 20 de octubre de 2026 para la sesión GPU del 21 de octubre.
El modelo predice pCR/no pCR usando exclusivamente imágenes DCE-MRI.

## 1. Abrir el proyecto y elegir el intérprete GPU existente

En VS Code: **Archivo → Abrir carpeta → breastcancer**. Después, **Python: Select Interpreter**
y seleccionar el entorno que ya dispone de PyTorch con CUDA. Abrir una terminal PowerShell nueva.
Todos los comandos se ejecutan desde la raíz del proyecto.

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Debe aparecer CUDA disponible (`True`) y una versión CUDA distinta de `None`.
Si aparece `+cpu`, seleccionar el intérprete GPU correcto. **No instalar el lock CPU,
ni reinstalar torch/torchvision, ni crear un entorno nuevo sobre el de clase.**
El código no instala PyTorch ni modifica drivers.

Si `python` no es el intérprete seleccionado, utilizar su ruta explícita en todos los comandos:

```powershell
& 'C:\RUTA\ENTORNO_GPU\Scripts\python.exe' -c "import torch; print(torch.cuda.is_available())"
```

La ruta anterior es un ejemplo: sustituirla por la del entorno GPU existente.

## 2. Dependencias auxiliares y comprobación real de CUDA

Mantener las versiones del equipo si ya funcionan. Solo si faltan librerías auxiliares:

```powershell
python -m pip install -r requirements-gpu-tools.txt
```

Ese archivo no contiene torch ni torchvision. No usar `requirements-cpu-lock.txt` ni
el `requirements.txt` general para modificar el entorno GPU de clase.

```powershell
python -m src.batch_train --check-device cuda
python -m unittest discover -s tests -v
```

La primera orden comprueba GPU real, forward, backward y valores finitos, e informa
del intérprete, versión de torch, CUDA, nombre y memoria de GPU. Si falla, no empieza ningún run.
CUDA no se ha podido probar en el portátil CPU: esta comprobación se realiza en clase.
Las pruebas unitarias son pequeñas y no utilizan test; no requieren checkpoints históricos descargados.

## 3. Lanzar automáticamente los cinco modelos de diez épocas

```powershell
python -m src.batch_train --suite config/phase4/gpu_suite.json --device cuda --resume-interrupted
```

Orden de la batería:

| ID | Cambio respecto al baseline GPU |
|---|---|
| BASELINE | CNN [32,64,128] original, clipping y pérdida ponderada |
| NO_WEIGHT_PC | Solo elimina la ponderación de la pérdida |
| NO_CLIP_PC | Solo elimina clipping adicional, con estadísticas de train consistentes |
| LOW_LR_PC | Solo cambia LR de 0.0005 a 0.0001 |
| SPATIAL_PC | Solo cambia de una a dos convoluciones por bloque |

`PC` forma parte del identificador de configuración; el dispositivo lo determina `--device cuda`.
Todos usan 877 pacientes de train y los mismos 220 de validación completa; no se usa test.
El máximo es diez épocas por modelo. Early stopping: cuatro épocas sin mejora del AUC de paciente.
Umbral fijo: 0.50. Semilla 42. No se aplican aumentos por defecto.

En esta batería GPU se aplica el mismo scheduler `ReduceLROnPlateau(mode=max)` a los cinco
modelos: factor 0.5, paciencia 2 y LR mínimo 1e-6. Es un protocolo común de diez épocas,
con baseline GPU propio; no se mezcla con los runs cortos de dos épocas sin scheduler.
Se selecciona `best_epoch.pt` por AUC de paciente. LR actual y siguiente se registran por época.
El funcionamiento del scheduler y su estado se apoya en la
[documentación oficial de PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).

Los runs completados con el mismo contrato, particiones, código y objetivo de épocas se omiten.
Un fallo se registra y el ejecutor continúa con el siguiente modelo. Ctrl+C detiene la batería
después de conservar los estados de recuperación del modelo activo.

## 4. Ejecutar uno o varios modelos concretos

Solo el modelo sin ponderación:

```powershell
python -m src.batch_train --suite config/phase4/gpu_suite.json --only NO_WEIGHT_PC --device cuda --resume-interrupted
```

Sin clipping y LR bajo:

```powershell
python -m src.batch_train --suite config/phase4/gpu_suite.json --only NO_CLIP_PC LOW_LR_PC --device cuda --resume-interrupted
```

Un modelo directamente, desde cero y en un run nuevo:

```powershell
python -m src.train --config config/phase4/gpu/NO_WEIGHT_PC.json --device cuda
```

La orden directa no omite automáticamente runs ya completados. Para repetir deliberadamente
la batería desde cero, usar `--force-new`; no se combina conceptualmente con una reanudación.

## 5. Reanudar sin confundirlo con iniciar otro experimento

La continuación conserva pesos, estado del optimizador, scheduler, mejor época/AUC,
contador de early stopping, historial, estados aleatorios de Python/NumPy/PyTorch/CUDA
y generador del DataLoader. La restauración CPU se ha comprobado contra entrenamiento
continuo, con igualdad exacta de pesos y estados. La
[API oficial de estado aleatorio](https://docs.pytorch.org/docs/stable/generated/torch.set_rng_state.html)
documenta la restauración del generador de PyTorch; CUDA dispone de su propio estado.

Para la batería, volver a ejecutar la orden del apartado 3 con `--resume-interrupted`.
Si hay un checkpoint compatible, continúa desde él; si no existe, indica explícitamente
que inicia desde cero. Los completados se omiten.

Para elegir manualmente un checkpoint:

```powershell
python -m src.train --resume 'models\RUN_IDENTIFICADOR\checkpoints\last_epoch.pt' --epochs 10 --device cuda
```

`RUN_IDENTIFICADOR` debe sustituirse por el nombre real. `--epochs 10` significa diez épocas
totales, no diez adicionales. No cambiar loss, arquitectura, batch size, preprocessing,
learning rate inicial, scheduler ni semilla en una continuación. Se verifica el contrato,
las huellas de código y las particiones; una incompatibilidad se rechaza.

La reanudación crea otra carpeta con fecha/hora, sufijo RESUME e identificador único.
La carpeta de origen y sus checkpoints no se sobrescriben. `details/resume.json` registra
la relación; el historial incluye las épocas heredadas y las nuevas.

### Interrupción a mitad de época

Los pesos `interrupted_epoch_XXX.pt` o `failed_epoch_XXX.pt` se conservan para diagnóstico,
pero no se consideran una época completada ni una continuación exacta.
Para repetir esa época desde su inicio, por ejemplo:

```powershell
python -m src.train --resume 'models\RUN_IDENTIFICADOR\checkpoints\recovery_epoch_start_003.pt' --epochs 10 --device cuda
```

Así no se pierde el trabajo de épocas completas ni se mezcla media época con un historial completo.
Si el mejor modelo se hereda de otra carpeta, su copia puede ser solo de inferencia; para
continuar usar `last_epoch.pt`, `epoch_XXX.pt` o `recovery_epoch_start_XXX.pt`, no asumir que
`best_epoch.pt` contiene el optimizador de una época posterior.

### Continuar un checkpoint CPU de hoy en GPU

```powershell
python -m src.train --resume 'models\RUN_CPU_DE_HOY\checkpoints\last_epoch.pt' --epochs 10 --device cuda --allow-device-change
```

Conserva el protocolo original del checkpoint, incluido scheduler `none` si así se entrenó hoy.
No lo convierte en el protocolo GPU con plateau. Cambiar de CPU a CUDA no garantiza la misma
trayectoria numérica; el cambio requiere la opción explícita y queda registrado.
Las interfaces de continuación exigen la misma versión mayor/menor de PyTorch y el
archivo `details/environment.json` del run. Si el entorno GPU tiene otra versión,
iniciar un experimento nuevo con las configuraciones preparadas; no sustituir su PyTorch
para forzar una continuación. Copiar el run completo, incluidos sus detalles de entorno.
Los checkpoints de fases anteriores siguen disponibles para evaluación, pero carecen de
todo el estado necesario para prometer una reanudación exacta y se rechazan como continuación.
Un entrenamiento que ya alcanzó early stopping no reinicia silenciosamente su contador.

## 6. Dónde mirar los resultados

Cada run usa `RUN_FECHA_HORA_CONFIG_UUID`. En su raíz: `history.csv`, `epoch_summary.csv`,
`summary.json`, `summary.txt`, `analysis.md`, curvas de loss/accuracy/AUC/ROC por paciente y corte,
matrices y distribuciones. `checkpoints/` contiene pesos; `details/`, configuración, versiones,
fuentes, predicciones identificadas y errores.

Cada ejecución del lote crea `models/BATCH_FECHA_HORA_UUID/` con:

- `results.json`: completados, omitidos, continuaciones y errores.
- `errors.jsonl`: errores y traceback, si los hay.
- `comparison.csv`: comparación de los runs reales de esa ejecución.
- `execution.json`: duración y estado final.

Los errores no se rellenan con métricas inventadas. No se comparan loss ponderada y normal
como si tuvieran la misma escala. Una accuracy alta con recall cero no implica buen AUC.

No editar el código mientras corre una batería. Para detenerla usar Ctrl+C; no apagar el equipo
ni borrar checkpoints. Si hay OOM, registrar el error y preparar otra batería con batch más
pequeño en todos sus comparadores; cambiar batch no es una reanudación compatible.

## 7. Qué llevar al ordenador de clase

Código actualizado, `config/phase4/`, perfiles de `config/phase3/stats/`, metadatos y estas instrucciones.
El dataset permanece en su ubicación relativa `breastdcedl/`. Los checkpoints necesarios para
continuar deben copiarse por separado, nunca subirse a GitHub. No copiar el entorno `.env`
del portátil sobre el de GPU. Los archivos nuevos aún deben trasladarse con el proyecto;
no se ha hecho ningún commit ni push automático.
