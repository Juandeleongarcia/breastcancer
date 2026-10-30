# BreastDCEDL AI Lab — aplicación final

Aplicación Streamlit real, con dos modalidades de inferencia CPU y cinco pantallas.
Predice **pCR / no pCR tras tratamiento**, nunca cáncer / ausencia de cáncer.
Es un prototipo universitario de investigación, sin validación clínica prospectiva.

## Inicio en Windows

Desde la raíz del repositorio, con el entorno existente:

```powershell
.\.env\Scripts\python.exe -m pip install -r requirements-app.txt
.\.env\Scripts\python.exe -m pip check
.\.env\Scripts\python.exe -m app_services.lab
.\.env\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Para la entrega final se ha preparado además un **repositorio de app separado**:
ver [README del paquete limpio](deploy/README_APP.md) y [matriz de cumplimiento y
pruebas](documentation/ENTREGA_FINAL_APP.md). Su arranque limpio usa Python 3.13,
un venv nuevo y deploy/requirements.txt. No depende del entorno de entrenamiento.

Abrir la URL que imprime Streamlit (normalmente `http://127.0.0.1:8501`). Detener
con `Ctrl+C`. El observador de archivos está desactivado: reiniciar el servidor
después de actualizar código. La revisión visual de esta entrega usa el puerto
8502 para no interferir con servidores previos.

`requirements-app.txt` solo añade interfaz al entorno PyTorch existente. **No
instalar `deploy/requirements.txt` sobre el entorno ROCm de clase**: es un lock
CPU exclusivamente para hosting. Se ha comprobado la existencia del wheel oficial
PyTorch 2.14.0+cpu para Python 3.13 y Linux x86_64.

## Qué hace cada página

| Página | Función |
|---|---|
| Inicio | Problema pCR, comparación interna real, AUC externo guardado y limitaciones visibles |
| Probar el modelo | Una tripleta o múltiples cortes, edad/subtipo opcionalmente declarados, inferencia real y defensa 1–5 |
| Explorador DCE-MRI | Fases sincronizadas, diferencias, zoom, contraste visual, pila axial relativa, superficie de intensidad y ROI temporal |
| Arquitectura y método | Tensores y parámetros leídos de redes/configuraciones reales, rama clínica, folds, preprocesamiento y trazabilidad |
| Resultados y validación | Comparación de cinco folds, OOF, curvas por época, medias/DE sin relleno y test externo independiente |

La modalidad de imágenes utiliza los cinco checkpoints de SPATIAL_SMALL; evita
elegir un fold arbitrario y utiliza cada perfil de normalización correspondiente.
La modalidad multimodal reproduce el ensemble final F13 de cinco checkpoints.
No se atribuye al modelo de imagen el resultado externo del multimodal.

## Demostración con las cinco muestras del profesor

1. Abrir **Probar el modelo**. El modo defensa está activo y muestra 1/5.
   Para la primera prueba, pulsar **Probar con un ejemplo público**: carga la
   tripleta autorizada y selecciona imágenes, sin inventar valores clínicos.
2. Elegir modalidad. Para multimodal, introducir edad y subtipo. Si faltan datos,
   puede usarse imagen únicamente. La imputación clínica exige declarar la ausencia
   expresamente: nunca se muestra como un dato conocido.
   Elegir también el umbral antes de analizar: original 0,50, recomendado OOF
   multimodal (exploratorio) o personalizado. Queda bloqueado tras registrar la
   muestra. No ajustarlo utilizando las etiquetas reveladas por el profesor.
3. Cargar PRE, EARLY y LATE en los tres controles. PNG real, gris de 8 bits,
   256 × 256; hasta 1 MiB por fase. No DICOM, RGB ni imágenes reescaladas por la app.
4. Comprobar la validación y las imágenes. Pulsar **Analizar muestra**.
5. Registrar la etiqueta real solo si el profesor ya la ha revelado.
6. Pulsar **Siguiente muestra**: se limpian cargas, clínica y predicción anterior.
7. Tras la quinta, descargar el CSV. Conservarlo antes de cerrar o recargar la sesión.

El registro contiene número de muestra, timestamp de Madrid, modalidad, checksum,
umbral, probabilidad, clase, número de modelos/cortes, dispositivo, tiempo y etiqueta
revelada opcional. No contiene imágenes, nombres de archivos, identificadores,
edad o subtipo. Todo permanece en memoria de la sesión, sin guardar entradas en disco.
Los identificadores de sesión/ejecución añadidos son UUID técnicos aleatorios,
sin relación con pacientes. El reinicio exige confirmación y borra solo esa sesión;
descargar el CSV antes de utilizarlo.
No hay caché de imágenes o predicciones de usuarios. Solo los modelos públicos
inmutables se comparten mediante `st.cache_resource`, protegidos por un lock.

Una única tripleta representa **un corte**, no un paciente completo como en la
evaluación de varios cortes. La app lo advierte. En modo avanzado se admite un
paciente por carga con nombres `PACIENTE_z012_PRE.png`, `_EARLY.png` y `_LATE.png`,
para todos sus índices z, o un ZIP. Se validan duplicados, fases, pacientes,
dimensiones y rutas; el ZIP nunca se extrae al sistema. Límite: 256 cortes y 128 MiB.

El ejemplo integrado procede solo del train público autorizado. No tiene etiqueta
ni datos clínicos precargados y no demuestra generalización. Ver `ATTRIBUTION.md`.

## Modelos y resultados auténticos

La exportación administrativa descubre campañas completas y no diagnósticas en
`models/CV9_*SPATIAL_SMALL*` y `models/phase13/CV13_*`. Reconoce una carpeta CV9
duplicada al copiar desde clase, sin moverla. Si hay más de una campaña candidata,
exige selección explícita: no elige por fecha ni por el mejor AUC.

El informe externo debe referirse a la campaña multimodal seleccionada y los hashes
de sus cinco checkpoints. Se verifican las métricas guardadas contra sus propias
predicciones registradas, sin volver a cargar imágenes ni ejecutar el modelo sobre test.

El paquete de esta entrega confirma 72.593 parámetros por CNN de imagen y 72.600
por CNN multimodal. Mejores épocas multimodales: 6, 2, 7, 5, 6.
Las cifras que aparecen en la interfaz se leen del paquete verificado; no son
constantes de demostración introducidas en las pantallas.

Cada miembro usa sus propios percentiles, medias/DE y transformador clínico de
train. Se reutilizan `BreastDCEDataset`, `ImageTransform`, `ClinicalTransform`,
`predict_patient` y `average_patients` originales. Se aplica `eval()` e
`inference_mode()`. La media se hace primero por corte/paciente/modelo y después
entre los cinco modelos, con pesos 0,20; nunca se promedian logits ni se votan clases.

El dashboard distingue CV interna, OOF, test guardado y nuevas muestras. Destaca
sensibilidad cero al umbral 0,50 en el test externo disponible. ROC AUC y matriz al
umbral fijo no son medidas intercambiables. No hay un control para optimizar umbral
con test ni un botón para repetir la evaluación externa.

## Análisis exploratorio del umbral pCR

`config/app_oof_threshold.json` es un informe agregado independiente; no modifica
el ZIP de pesos, los checkpoints ni los informes históricos. El Dockerfile y el
generador del paquete de hosting incluyen este JSON; Community Cloud lo lee desde
el repositorio. La app comprueba que su checksum de modelo corresponda al ensemble
multimodal cargado antes de ofrecer la recomendación.

Se recuperan las predicciones de validación por paciente de los cinco mejores
checkpoints F13 y se contrastan con los dos informes OOF completos guardados, los
hashes de campaña y los folds oficiales. Son 1.097 pacientes (322 pCR, 775 no pCR),
cada uno con una sola predicción del modelo que lo excluyó de train.

Rejilla fijada: 0,05–0,95, paso 0,01, 91 puntos. Decisión: pCR si probabilidad ≥
umbral. Criterio: máximo Youden = sensibilidad + especificidad − 1. Empates dentro
de 1e-12: más próximo a 0,50 y después mayor umbral. Precisión/F1 sin positivos
predichos se definen como cero. Resultado real: **0,31**, Youden 0,273212,
sensibilidad 0,608696, especificidad 0,664516, balanced accuracy 0,636606.

En **Resultados y validación → Umbrales OOF** se comparan original/recomendado y un
umbral interactivo: métricas, matrices y curvas. No requiere publicar filas por
paciente. El control del dashboard no cambia el de la demostración. El CSV de la
defensa guarda el umbral y su origen junto a la decisión; la probabilidad no cambia.

**Limitación:** los mismos folds seleccionaron las épocas. Este análisis no es
CV anidada ni validación independiente tras seleccionar el umbral. Las puntuaciones
OOF proceden de un modelo por paciente y no del ensemble completo, por lo que la
transferencia del umbral al ensemble para nuevas muestras es provisional. No hay
garantía de sensibilidad externa. El informe externo original queda a 0,50 y no
se recalcula. ROC AUC continuo no cambia al elegir otro umbral.

Reproducción administrativa local (usar una salida nueva; nunca sobrescribe):

```powershell
.\.env\Scripts\python.exe -B -m app_services.thresholds --campaign models/phase13/CV13_SPATIAL_SMALL_MULTIMODAL_20261029_160300_a18cb0802b7b --output models/phase13/oof_threshold_reproduccion.json
.\.env\Scripts\python.exe -B -m unittest discover -s tests -p test_app_thresholds.py -v
```

## Paquete de inferencia y privacidad

`config/app_release.json` fija la URL HTTPS de release y el SHA256 del ZIP. La app
usa primero `models/app_bundle/` si existe; en hosting descarga el paquete, verifica
su hash y cada archivo, y lo conserva en caché local del servidor. El navegador
no puede cambiar esa URL ni introducir rutas del sistema.

El ZIP contiene 10 checkpoints **reducidos**, 10 historiales agregados, catálogo de
curvas/métricas, atribución y tres PNG públicos de un corte de train. No contiene
optimizador, scheduler, RNG, predicciones por paciente, etiquetas, CSV completos,
identificadores ni validación privada del profesor. Pesa aproximadamente 2,9 MB.
Los originales nunca se modifican. Se cargan mediante `weights_only=True`.

Si hay que regenerarlo, usar una salida nueva:

```powershell
.\.env\Scripts\python.exe -m app_services.package --output models/app_bundle_v2
```

Con varias campañas, indicar sus nombres completos mediante `--image`,
`--multimodal` y `--external`; `--models` cambia la raíz administrativa de búsqueda.
Estos argumentos solo existen en la CLI del propietario, nunca en la interfaz web.
Actualizar URL/hash únicamente al publicar una nueva versión revisada.

## Despliegue elegido: Hugging Face Spaces con Docker

La documentación actual de Hugging Face indica que crear Spaces con cómputo
(incluido Docker) exige una cuenta con acceso a ese servicio; CPU Basic figura
sin coste horario, pero no implica que cualquier cuenta gratuita pueda crear el
Space. Comprobar la elegibilidad de la cuenta antes de contratar nada:
[requisitos y recursos oficiales](https://huggingface.co/docs/hub/spaces-overview).
No se contrata ni amplía ningún plan automáticamente.

Comprobación del 30/10/2026 en la cuenta `Juaandeleoon`: la pantalla de creación
mostraba Docker deshabilitado y el aviso de suscripción PRO. El despliegue queda
pendiente de que el titular habilite ese servicio o elija la alternativa inferior.
No se ha contratado ningún plan ni creado un Space vacío como sustituto de la app.

1. Iniciar sesión en https://huggingface.co y crear un Space **Docker / Blank**,
   con un nombre como `breastdcedl-ai-lab`, acceso público y CPU Basic si está disponible.
   No seleccionar Gradio o Static para esta aplicación Streamlit.
2. Preparar una carpeta de código autorizada, independiente del repositorio completo:

```powershell
.\.env\Scripts\python.exe deploy/prepare_space.py --output models/space_upload_final
```

3. Subir el **contenido** de `models/space_upload_final/` a la raíz del Space. El script
   coloca `Dockerfile` y `README.md` con `sdk: docker` y `app_port: 7860` en la raíz.
   No subir `models/`, `metadata/`, `breastdcedl/`, backups, el entorno ni este repo entero.
4. Puede utilizarse el panel **Files → Add file → Upload files**, o la CLI oficial:

```powershell
hf auth login
hf upload Juaandeleoon/breastdcedl-ai-lab models/space_upload_final . --repo-type space
```

   `hf` forma parte de `huggingface_hub`; instalarlo en una herramienta/entorno
   separado si hace falta. No enviar tokens por chat ni escribirlos en archivos
   del repositorio. Ver [CLI oficial](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
5. Esperar al build. El contenedor instala el lock CPU y ejecuta primero
   `python -m app_services.lab`: si el paquete o los pesos no están listos, falla
   antes de abrir una app aparentemente válida. Después escucha en 0.0.0.0:7860.
6. Comprobar que el Space esté **Running** y abra la app con modelos verificados.
   Probar el ejemplo público, no el test externo. Comprobar descarga del CSV.
7. Compartir la URL **real que asigne Hugging Face**, tanto su página `/spaces/...`
   como el dominio `.hf.space` que muestre la plataforma. No asumir que una URL
   construida por nombre ya está desplegada. Verificarla desde otra sesión antes
   de la defensa. No depende del ordenador del estudiante.

El Dockerfile usa una lista explícita de carpetas de código y excluye metadata,
datasets, modelos originales y ficheros privados. La descarga de los artefactos
necesita conectividad HTTPS al release de GitHub. Los datos de entrada del profesor
solo se envían a la sesión del servidor de la app, nunca a servicios de inferencia
terceros. La telemetría de Streamlit está desactivada.

Referencia: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).
No se ha ejecutado Docker localmente si el motor no está instalado. La autenticación
en HF y el build real son pasos distintos de la verificación local Windows CPU.

### Despliegue elegido: Streamlit Community Cloud

Tras comprobar el bloqueo de Docker en Hugging Face, el titular ha elegido
Streamlit Community Cloud. Tras la auditoría de privacidad se ha elegido preparar
un repositorio separado para la app: repositorio público **[Juandeleongarcia/breastdcedl-ai-lab](https://github.com/Juandeleongarcia/breastdcedl-ai-lab)**,
creado y publicado tras la autorización final del autor. No desplegar copiando indiscriminadamente el repo de entrenamiento:
ya contiene CSV históricos de metadata versionados, que se conservan sin cambios.
Usar el paquete generado con `--target streamlit`, rama `main`, entrypoint **deploy/app.py**
y Python 3.13. Ese directorio contiene el lock CPU que tiene prioridad sobre los
requisitos históricos del repositorio. No usar el requirements de entrenamiento.
La app descarga el mismo paquete verificado. Requiere iniciar sesión y conectar
la cuenta de GitHub; no se afirma que esté publicada hasta verificar su URL.

[Despliegue oficial](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
y [prioridad de requisitos](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
La opción Docker usa un árbol más reducido porque Community Cloud clona el repo.

Pasos exactos desde el navegador:

1. Abrir https://share.streamlit.io/ e iniciar sesión personalmente. Revisar y
   aceptar sus condiciones y los permisos de GitHub en las páginas del proveedor;
   nunca enviar credenciales por chat. Conectar únicamente el repositorio necesario
   si el proveedor permite limitar el alcance.
2. Pulsar **Create app / Deploy a public app from GitHub**.
3. Repositorio: el nuevo repositorio limpio autorizado; branch: `main`;
   main file path: `deploy/app.py`.
4. En **Advanced settings**, elegir **Python 3.13**. No hacen falta secrets para
   este paquete público verificado. Puede solicitarse un subdominio disponible,
   pero su URL solo se considera publicada cuando la plataforma la confirme.
5. Pulsar **Deploy**. Revisar el log de instalación y esperar al estado activo.
6. Comprobar las cinco páginas y predecir con el ejemplo público en modalidad de
   imágenes. Confirmar que no aparece ningún error de descarga/checksum/modelos.
7. Abrir la URL real desde otra sesión y conservarla para la defensa. No cargar
   muestras privadas para comprobar el despliegue. Las entradas de una sesión se
   envían al servidor de Community Cloud para inferencia en memoria.

Si hay un error de versión de Python o dependencia, revisar el log y corregir la
configuración de hosting; no modificar pesos ni instalar requisitos ROCm allí.

## QA y límites de la comprobación

Verificación final: **55 pruebas superadas, cero fallos y cero omitidas**,
incluidas las pruebas opcionales con pesos reales (`BREASTDCEDL_TEST_CHECKPOINTS=1`).
Se han comprobado sin cambios por SHA256 los **566 archivos históricos** inventariados.
Docker no se ha ejecutado localmente porque este ordenador no dispone del motor.
El paquete limpio con Python 3.13.7 y dependencias recién instaladas ha pasado
además sus **11 pruebas portables**, sin necesitar el dataset ni las metadata.

```powershell
.\.env\Scripts\python.exe -B -m unittest discover -s tests -p test_app_final.py -v
.\.env\Scripts\python.exe -B -m unittest discover -s tests -p test_phase12_ui.py -v
```

Se han verificado entradas ausentes/duplicadas/corruptas/RGB/dimensiones/rutas,
clínica ausente y desconocida, los diez checkpoints reales en CPU, repetibilidad,
coincidencia numérica con inferencia original sobre un caso público de train,
agregación/ensemble, cinco muestras consecutivas, limpieza, CSV y 3D de uno/varios cortes.
Las pruebas no reentrenan modelos ni ejecutan el test externo. Los artefactos
históricos y las fuentes protegidas se contrastan mediante hashes antes/después.

Las nuevas predicciones no están cacheadas. La caché de recursos solo conserva
modelos públicos con checksum. Una recarga completa de la sesión puede perder
el registro: descargar el CSV antes de cerrar. Cinco muestras no bastan para
estimar rendimiento clínico. El prototipo no decide tratamientos ni evita cirugía.
