# Fase 7.1 · entrenamientos pendientes en Linux / AMD ROCm

En la raíz de `breastcancer`, con el entorno ROCm **existente** de clase activado:

```bash
git status --short
git pull --ff-only origin main
python -c "import torch; print(torch.__version__, 'HIP=', torch.version.hip); assert torch.version.hip and torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m src.check_suite --suite config/phase7/suite.json --device cuda --require-rocm
python -m unittest discover -s tests -v
```

No reinstalar PyTorch ni instalar el lock de CPU. `cuda` es también el nombre
del dispositivo en PyTorch ROCm. El preflight debe pasar en la RX 6700 XT real;
las verificaciones Windows/CPU no certifican el backend de clase.

SPATIAL_CONTROL ya fue entrenado en clase: 10 épocas, mejor AUC paciente 0,6306 en
época 2, **según resultados comunicados por el usuario**. Sus pesos/historial no se
encuentran aquí. Conservar ese run y usar su configuración guardada como referencia;
no repetirlo sin una diferencia metodológica demostrada. Fase 7.1 no cambia las
cuatro configuraciones Fase 7, las estadísticas ni la CNN.

Prioridad 1, augmentación suave:

```bash
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda --only SPATIAL_GENTLE_AUG
```

Prioridad 2, menor capacidad:

```bash
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda --only SPATIAL_SMALL
```

Si se dispone de tiempo para ambas secuencialmente, sin lanzar el control:

```bash
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda --only SPATIAL_GENTLE_AUG SPATIAL_SMALL
```

Prioridad 3, solo después de revisar las curvas/métricas anteriores:

```bash
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda --only SPATIAL_SCHEDULER
```

Reanudar interrupciones compatibles de los dos candidatos prioritarios:

```bash
python -u -m src.batch_train --suite config/phase7/suite.json --device cuda --only SPATIAL_GENTLE_AUG SPATIAL_SMALL --resume-interrupted
```

El ejecutor omite runs completos del mismo contrato, registra errores y actualiza
`models/BATCH_.../comparison.csv`. Un resume crea un RUN distinto y restaura
modelo, optimizador, scheduler, RNG, historial y contador de early stopping.
No utilizar checkpoints diagnósticos de subconjuntos para iniciar o reanudar estos
entrenamientos completos. Una ejecución que alcanzó early stopping ya terminó;
reanudar no reinicia su contador. No cambiar hiperparámetros durante una continuación.

Para generar otra comparación global de los resultados locales de clase:

```bash
python -m src.compare_runs --root models
```

Se genera un CSV nuevo, sin sobrescribir comparaciones anteriores. Revisar las
columnas de grupo comparable; un diagnóstico parcial no equivale a train completo.

## Protocolo fijo

Todos: 877 pacientes/8.745 cortes train y 220/2.200 validación interna; semilla 42,
batch 16, AdamW, LR 0,0005, weight decay 0,0005, dropout 0,40, BCE sin ponderación,
20 épocas máximas, early stopping 8 por mejora estricta del AUC paciente. Pooling
global 1×1, ReLU/BatchNorm, dos convoluciones por bloque, sin preentrenamiento.
Preprocesamiento/estadísticas idénticos a Fase 7. Sin test.

| Configuración exacta | Diferencia frente al control | Hipótesis |
|---|---|---|
| `config/phase7/SPATIAL_GENTLE_AUG.json` | Activar gentle_affine: p=0,8, ±5°, traslación/escala ±2 %, sin flips | Menor dependencia de posición |
| `config/phase7/SPATIAL_SMALL.json` | Canales [16,32,64] en lugar de [32,64,128] | Menor capacidad de memorización |
| `config/phase7/SPATIAL_SCHEDULER.json` | Plateau patience 3 en lugar de 2 | Menos reducciones prematuras por AUC ruidoso |

Comparar **toda** la trayectoria de train eval loss y validation loss, su gap,
AUC paciente/corte, precision, recall/sensibilidad, F1, especificidad y fracción
predicha pCR. La accuracy mayoritaria de validación completa es 155/220 = 70,45 %.
No seleccionar por accuracy aislada ni retocar umbrales para aparentar mayor AUC.
El checkpoint principal sigue siendo el de mejor AUC paciente; informar también
si su BCE empeora o si la separación de pérdidas crece.

Cada RUN conserva config.py, history.csv, todos los epoch_NNN.pt completos,
best_epoch.pt, loss_curve.png, loss_comparison.png, accuracy_curve.png,
auc_curve.png y roc_curve.png; técnicos en internals/. No alterar los runs previos.

Informe diagnóstico y límites: [FASE7_1.md](documentation/FASE7_1.md).
Esta guía utiliza las configuraciones originales Fase 7 del commit `dd91c802`;
su publicación se autoriza al cerrar Fase 7.1, sin incluir resultados pesados.
