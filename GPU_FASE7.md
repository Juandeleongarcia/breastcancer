# Fase 7 · Linux / AMD ROCm

Desde la terminal de VS Code, en la raíz del repositorio y con **el entorno ROCm
de clase ya activado**. No instalar `requirements-lock.txt` de CPU ni sustituir
PyTorch. En ROCm, PyTorch usa la API y el dispositivo `cuda`.
Así lo documenta [PyTorch HIP](https://docs.pytorch.org/docs/stable/notes/hip.html).

Fase 7 está publicada en el commit `dd91c802`. Para la planificación posterior al
control ya entrenado, consultar [GPU_FASE7_1.md](GPU_FASE7_1.md). Sincronizar antes:

```bash
git status --short
git pull --ff-only origin main
python -c "import torch; print(torch.__version__, 'HIP=', torch.version.hip); assert torch.version.hip and torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m src.check_suite --suite config/phase7/suite.json --device cuda --require-rocm
python -m unittest discover -s tests -v
```

El preflight comprueba las particiones completas (877/8.745 y 220/2.200), rutas
Linux, estadísticas firmadas de train y forward/backward **sintético** de las cuatro
CNN. No lee test ni entrena con pacientes. Si faltan dataset/estadísticas, detenerse
y recuperar los archivos originales; no generar particiones nuevas ni refit de stats.
ROCm nativo y compatibilidad real de RX 6700 XT deben superar este preflight:
haber pasado las pruebas CPU no acredita esa GPU ni sus drivers.

Comandos individuales sin alterar configuraciones. Si el control ya está completado,
conservarlo y priorizar GENTLE_AUG y SMALL según la guía Fase 7.1:

```bash
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda --only SPATIAL_CONTROL
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda --only SPATIAL_GENTLE_AUG
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda --only SPATIAL_SMALL
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda --only SPATIAL_SCHEDULER
```

Todos secuencialmente, o reanudar interrupciones compatibles:

```bash
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda --resume-interrupted
```

El ejecutor omite runs ya completados del mismo contrato, registra errores y
actualiza `models/BATCH_.../comparison.csv`. Las métricas pendientes quedan vacías.
`--resume-interrupted` busca el último límite de época íntegro y crea un RUN nuevo;
no modifica el anterior. Para una fuente explícita (sustituir la ruta de ejemplo):

```bash
python -u -m src.train --resume models/RUN_ORIGEN/epoch_005.pt --epochs 20 --device cuda
```

`20` es el objetivo **total**, no 20 épocas adicionales. Restaurar modelo,
optimizador, scheduler, historial, RNG y contador de early stopping exige contrato,
backend y versión compatibles. No convertir un histórico sin train eval en un
experimento Fase 7 mediante resume. Los históricos conservan su protocolo
`EVALUATE_TRAIN=False`; las cuatro configuraciones nuevas usan `True`.
Una copia de pesos sin estado completo solo sirve para evaluación, no reanudación.

Cada RUN nuevo conserva en su raíz:

```text
RUN_YYYYMMDD_HHMMSS_F7_SPATIAL_CONTROL_<uuid12>/
├── config.py
├── history.csv
├── epoch_001.pt ... epoch_020.pt  # solo épocas completas realmente realizadas
├── best_epoch.pt
├── loss_curve.png
├── loss_comparison.png
├── accuracy_curve.png
├── auc_curve.png
├── roc_curve.png
└── internals/                   # configuración JSON, estados y diagnósticos técnicos
```

No sobrescribir checkpoints ni renombrar runs. Ver análisis y limitaciones en
[FASE7.md](documentation/FASE7.md). No hay resultados nuevos de pacientes en esta fase.
