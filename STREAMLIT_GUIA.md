# BreastDCEDL AI — base visual de Streamlit

La aplicación permite explorar estudios DCE-MRI y revisar una campaña de
validación cruzada compatible. No selecciona, compara ni ejecuta arquitecturas.
La presentación está preparada para incorporar posteriormente **un único modelo**,
cuando el usuario entregue los artefactos definitivos.

> Prototipo académico de investigación. No destinado a diagnóstico ni decisiones clínicas.

## Arranque en Windows

Desde la raíz del proyecto, con su entorno existente:

```powershell
.\.env\Scripts\python.exe -m pip install -r requirements-app.txt
.\.env\Scripts\python.exe -m pip check
.\.env\Scripts\python.exe -m streamlit run app.py
```

Abrir <http://127.0.0.1:8501>. Detener con `Ctrl+C` en la terminal. Después de
modificar código, reiniciar el servidor: el observador de archivos está desactivado
para no vigilar las carpetas de entrenamiento.

Las dependencias nuevas son Streamlit 1.66.1 y Plotly 7.1.0. El archivo de requisitos
de la app no instala ni sustituye PyTorch. El entorno local comprobado conserva
Python 3.13.7 y PyTorch 2.14.0+cpu.

## Arranque en Linux / ordenador de clase

Conservar el entorno ROCm que ya funciona:

```bash
conda activate cancer
python -m pip install -r requirements-app.txt
python -m pip check
python -m streamlit run app.py
```

No instalar el lock CPU de Windows sobre el entorno de clase. Esta base visual
no utiliza la GPU, carga pesos ni interactúa con entrenamientos en ejecución.
La verificación realizada aquí corresponde a Windows; el arranque real en Linux
debe comprobarse en ese equipo. El servidor escucha únicamente en `127.0.0.1`.

## Navegación

| Página | Funcionalidad actual |
|---|---|
| Inicio | Problema pCR/no pCR, fases DCE-MRI, recorrido conceptual y acceso al explorador y resultados |
| Explorador DCE-MRI | PRE/EARLY/LATE sincronizados, selección de corte, brillo/contraste, zoom y pila 3D esquemática opcional |
| Predicción de pCR | Carga y revisión de estudios; botón de análisis deshabilitado y resultados pendientes |
| Resultados y CV | Una campaña compatible: cinco folds, ROC, matrices, historiales y medias de pérdidas |
| Cómo funciona una CNN | Esquema conceptual interactivo y explicaciones generales, sin filtros, dimensiones o parámetros de un modelo particular |

No hay nombres de candidatos, selectores de arquitecturas, tablas de comparación,
configuraciones de entrenamiento expuestas ni recuentos de parámetros en la UI.
Los nombres internos de campañas no se presentan como títulos. La ruta local se
introduce en un campo oculto para facilitar las demostraciones en pantalla.

## Pendientes explícitos

- Arquitectura definitiva.
- Número de parámetros.
- Checkpoint final.
- Configuración de inferencia.
- Métricas definitivas.
- Resultados del test.

No se descargan o cargan checkpoints. Las antiguas claves de configuración
`models` e `inference`, incluso con `enabled: true`, no pueden activar la predicción.
El bloque `integration` es una declaración de pendientes; tampoco activa código.

## Cargar estudios en el explorador

Se admite un paciente por carga: PNG de 8 bits en gris y 256 × 256 píxeles,
con tripletas completas para cada índice de corte:

```text
PACIENTE_z012_PRE.png
PACIENTE_z012_EARLY.png
PACIENTE_z012_LATE.png
```

Se aceptan varios PNG o un ZIP. La validación rechaza fases ausentes, duplicados,
mezcla de pacientes, dimensiones o modos de color incompatibles y rutas peligrosas.
Los archivos se procesan en memoria, sin extracción del ZIP ni envío a servicios
externos. Límites: 128 MiB por estudio, 1 MiB por PNG y 256 cortes.

Los controles de brillo y contraste solo modifican la vista. La vista de diferencias
temporales muestra PRE, EARLY − PRE y LATE − PRE de forma ilustrativa; no aplica el
preprocesamiento del modelo ni calcula estadísticas con imágenes cargadas.
El preprocesamiento de inferencia definitivo sigue pendiente.

La pila 3D muestra como máximo 12 planos de referencia más el corte seleccionado.
Usa orden de corte y coordenadas de píxel, sin atribuir orientación ni distancias
físicas. **No es una reconstrucción anatómica.** La app tampoco puede comprobar
que las tripletas aportadas constituyen el estudio anatómico original completo.

La demostración del visor utiliza exclusivamente formas geométricas sintéticas,
identificadas como tales. No crea un paciente real, probabilidades ni resultados.

## Conectar resultados reales

Abrir **Resultados y CV → Cargar resultados reales** e introducir la carpeta de
una campaña compatible. No requiere seleccionar una arquitectura. Solo se lee
una campaña cada vez; no existe una comparación entre modelos.

Para recordar la ruta, copiar `config/streamlit.example.json` a
`models/app_config.json`, conservando cualquier archivo local anterior. Configurar
únicamente `results.campaign` con la ruta real:

```json
{
  "results": {"campaign": "/ruta/real/campana_cv"},
  "integration": {
    "enabled": false,
    "architecture": null,
    "parameters": null,
    "checkpoint": "",
    "inference_config": "",
    "final_metrics": null,
    "test_results": null
  }
}
```

También se admite un archivo externo:

```powershell
$env:BREASTDCEDL_APP_CONFIG = 'C:\ruta\app_config.json'
```

```bash
export BREASTDCEDL_APP_CONFIG=/ruta/app_config.json
```

Las rutas relativas se resuelven desde el repositorio. Dejar la campaña vacía
mantiene el dashboard preparado, sin inventar cifras ni curvas.

El lector utiliza `campaign.json`, `template.json`, los CSV oficiales y derivados,
el estado de cada fold y su último RUN (`history.csv`, configuración, resumen y
predicciones por paciente). Reconoce el formato simplificado con `internals/`
y formatos históricos compatibles mediante las utilidades existentes. No abre
pesos ni escribe resúmenes en las campañas.

Para copias procedentes de Linux, las rutas de RUN se remapean a la copia local
conservando el nombre. Si falta la ruta original del CSV oficial, se admite el
CSV del repositorio únicamente si coincide su hash guardado. El resto del contrato
y las predicciones deben estar completos y ser verificables; no se crea información
ausente para conseguir que una campaña aparezca como completa.

## Cómo interpretar el dashboard

- **ROC AUC medio y DE:** media de los folds completados y verificados. La DE es
  muestral (`ddof=1`); con un solo fold permanece ausente. Menos de cinco folds se
  identifica como validación provisional.
- **OOF:** se construye solo con los cinco folds completos, pacientes únicos y
  cobertura oficial íntegra. No se mezclan pacientes reservados para test.
- **ROC y matriz:** se calculan a partir de probabilidades reales guardadas por
  paciente. El umbral procede de la configuración guardada; no se optimiza en la UI.
- **Loss:** `train_eval_loss` y validación usan pesos de final de época. Si un
  historial antiguo solo tiene train loss online, se etiqueta explícitamente y
  se representa con una línea discontinua.
- **Medias por época:** se muestran todos los folds disponibles o solo las épocas
  comunes a los cinco. Se informa cuántos folds contribuyen a cada punto. No se
  rellenan épocas con ceros, valores repetidos ni interpolación; una sola
  contribución no tiene banda de DE.
- **Granularidad:** accuracy y AUC se identifican por paciente. La pérdida conserva
  la granularidad por corte del entrenamiento y se etiqueta en el eje.

Los resultados cargados constituyen evidencia aportada, no una elección automática
del modelo final ni resultados de test. Los datos numéricos para comprobar el
dashboard se crean solo en directorios temporales de los tests; no se muestran
como métricas clínicas de demostración.

## Estructura modular

```text
app.py                         # navegación de cinco páginas
app_components/                # vistas y gráficos
app_services/workspace.py      # configuración visual sin dependencias del modelo
app_services/studies.py        # validación y lectura de estudios
app_services/results.py        # lectura y agregación de una campaña
app_services/                  # adaptadores técnicos reservados, sin conexión a la UI
assets/                        # CSS e ilustración temporal propia
.streamlit/config.toml         # tema, servidor local y límites
config/streamlit.example.json   # resultados y pendientes, sin pesos
tests/test_phase12_ui.py        # navegación y ausencia de detalles de candidatos
tests/test_phase12_services.py  # contratos de entrada y cálculos
```

Los adaptadores experimentales anteriores permanecen internos y desconectados.
La importación de la presentación no importa esos adaptadores ni construye CNN.
Al elegir el modelo se conectará un único proveedor de inferencia, con su
arquitectura, configuración exacta, estadísticas ajustadas en train y checkpoint
fiable. Esa integración requerirá su propia verificación y no está habilitada
mediante una casilla o selector de esta base visual.

## Pruebas

```powershell
.\.env\Scripts\python.exe -B -m unittest discover -s tests -p "test_phase12_*.py" -v
```

```bash
python -B -m unittest discover -s tests -p 'test_phase12_*.py' -v
```

Las pruebas usan estudios geométricos y campañas pequeñas en carpetas temporales.
Verifican navegación, cargas válidas e inválidas, selección sincronizada, pila 3D,
predicción siempre deshabilitada, ausencia de identidades de modelos en pantallas,
lectura sin escrituras, cobertura de folds, OOF y medias sin relleno.

Cuatro pruebas de arquitecturas/checkpoints permanecen aplazadas y se indican
como `skipped`. No activar `BREASTDCEDL_TEST_CHECKPOINTS` en esta fase visual.
No se ejecutan entrenamientos ni se abren checkpoints históricos.

Los archivos originales del proyecto, particiones, preprocesamiento y resultados
se conservan. Las copias de seguridad y registros de verificación son locales,
dentro de `backups/`, excluido por Git. No se ha hecho commit ni push de esta fase.
