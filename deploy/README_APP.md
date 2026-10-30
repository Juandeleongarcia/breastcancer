# BreastDCEDL AI Lab

**Juan de León García · Universidad Alfonso X el Sabio (UAX)**

Grado en Ingeniería Matemática. Trabajo académico universitario de investigación.

Aplicación Streamlit con inferencia real CPU de CNN propias, imágenes DCE-MRI y
predicción de respuesta patológica completa (pCR/no pCR). Prototipo educativo:
no destinado a diagnóstico ni decisiones clínicas. Una puntuación no demuestra
calibración, supervivencia, curación ni permite decidir sobre cirugía.

## Qué incluye este repositorio de aplicación

Solo código, configuración de inferencia, gráficos agregados, recursos vectoriales,
documentación y pruebas. No incluye dataset, metadata por paciente, etiquetas
individuales, validación privada ni estados de entrenamiento. El código de src/
se conserva para reutilizar los operadores originales; no se entrena desde la UI.

Los diez checkpoints reducidos y el ejemplo público se descargan del release
fijado en config/app_release.json. Se verifica SHA256 del ZIP y de sus miembros.
El paquete tiene 2,9 MB; no contiene optimizadores, imágenes privadas ni CSV por
paciente. Atribución y licencia de los datos: ver ATTRIBUTION.md, **CC BY-NC 4.0**.
El LICENSE del código no relaja las condiciones de los datos.

## Arranque Windows desde un entorno limpio

Desde la raíz del repositorio, con Python 3.13 instalado:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r deploy/requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m app_services.lab
.\.venv\Scripts\python.exe -m streamlit run deploy/app.py --server.address 127.0.0.1
```

Abrir la URL local que imprima Streamlit. El entorno CPU es exclusivo de la app;
no instalarlo sobre el PyTorch ROCm del ordenador de clase. La primera descarga
necesita acceso HTTPS a GitHub. Sin artefactos válidos, la inferencia no se habilita.

## Streamlit Community Cloud

1. El titular crea un repositorio nuevo para esta carpeta de código limpio; nombre
   propuesto: `breastdcedl-ai-lab`. No copiar el repositorio de entrenamiento completo.
2. Revisar git status y publicar solo el contenido de esta carpeta. **No subir**
   models/, metadata/, datasets, entornos, secretos o backups. Este paso requiere
   autorización final del titular y acceso seguro a su cuenta.
3. Iniciar sesión personalmente en https://share.streamlit.io/ y revisar sus términos
   y permisos de GitHub. No compartir contraseñas, códigos o tokens por chat.
4. Create app → Yup, I have an app. Repositorio: el nuevo repositorio limpio;
   branch: `main`; main file: **deploy/app.py**; Advanced settings → **Python 3.13**.
   No hacen falta secrets. deploy/requirements.txt tiene prioridad sobre la raíz.
5. Pulsar Deploy solo tras la revisión/autorización del titular. Esperar al build
   y revisar errores de descarga/dependencias. No contratar ningún servicio.
6. Verificar la URL real desde otra sesión con el ejemplo público. No se considera
   publicado solo por haber preparado o creado el repositorio.

Tras la autorización final y **solo dentro de la carpeta limpia**, con el repositorio
nuevo creado vacío en GitHub, estos son los comandos de publicación (no ejecutarlos
en el directorio del proyecto de entrenamiento):

```powershell
git init -b main
git add -- .gitignore README.md ATTRIBUTION.md LICENSE Dockerfile .streamlit/config.toml app.py app_components app_services assets src config deploy tests
git diff --cached --stat
git status --short
git commit -m "Preparar BreastDCEDL AI Lab para la defensa"
git remote add origin https://github.com/Juandeleongarcia/breastdcedl-ai-lab.git
git push -u origin main
```

El nombre de repo es una propuesta hasta crearlo. Si existe contenido remoto o
aparece un conflicto, detenerse; no utilizar force-push. No compartir credenciales.

Documentación oficial:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Docker alternativo: el Dockerfile de raíz escucha en 7860 y usa las mismas
dependencias CPU. HF Spaces exige actualmente acceso a cómputo habilitado en la
cuenta; no contratar planes automáticamente. No se ha validado un build Docker
local si no está instalado el motor.

## Demostración de cinco muestras

Probar el modelo → **Probar con un ejemplo público** → Analizar muestra. El botón
selecciona imágenes para permitir una prueba inicial sin inventar clínica. Para
multimodal, introducir edad y subtipo conocidos, o declarar expresamente ausencia.
El ejemplo no incluye etiqueta ni clínica precargadas. También se aceptan tres PNG8
en gris de 256×256 o varios cortes sincronizados de un único paciente (ZIP estándar
Deflate/sin compresión). Cada fase: 1 MiB; conjunto: 128 MiB y 256 cortes como máximo.

Revisar modalidad, umbral, número de imágenes y cinco modelos antes de analizar.
La media es por cortes y después por cinco modelos. Un único corte no equivale al
protocolo de paciente completo. Etiquetas reales solo después de predecir. El umbral
queda bloqueado por muestra. Tras cinco, descargar CSV antes de cerrar la sesión.
El reinicio exige confirmación y solo descarta datos en memoria de esa sesión.

Las entradas del usuario se envían al servidor de hosting para inferencia en memoria;
no se escriben ni se envían a servicios externos de inferencia. El registro contiene
puntuación, decisión, umbral, origen del umbral, timestamp, modalidad, checksum,
tiempo, dispositivo y UUID técnicos; no imágenes, nombres, IDs de paciente o clínica.
El proveedor de hosting procesa la sesión conforme a sus propias condiciones.

## Resultados y limitaciones

La UI distingue CV media/DE muestral, OOF y test externo ya guardado. El test
congelado tiene AUC aproximado 0,6722 y sensibilidad cero al umbral 0,50. No se
recalcula. El umbral multimodal 0,31 maximiza Youden en una rejilla OOF de desarrollo
0,05–0,95, paso 0,01. Es exploratorio: los folds también seleccionaron las épocas,
no es CV anidada y OOF no reproduce la distribución del ensemble completo. No está
validado externamente ni se ofrece como recomendación para el modelo de imágenes.

Tres bloques y parámetros se verifican en redes y pesos reales. El visor 3D muestra
índices relativos o alturas de intensidad, sin inventar orientación, milímetros ni
segmentación. Las cinco muestras de la defensa comprueban funcionamiento, no
establecen rendimiento clínico. No hay validación prospectiva ni de calibración.

## Pruebas portables de la aplicación

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Estas pruebas usan el paquete público verificado para inferencia real CPU y formas
sintéticas para contratos técnicos. No ejecutan entrenamientos ni acceden a test
externo. La suite científica completa y las referencias OOF originales se conservan
en el repositorio de entrenamiento; no se copian sus datos a este repositorio.
