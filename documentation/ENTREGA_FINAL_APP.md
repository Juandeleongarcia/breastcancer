# Auditoría final de BreastDCEDL AI Lab

Fecha: 30/10/2026 (Europe/Madrid).

**Autor:** Juan de León García. **Universidad:** Universidad Alfonso X el Sabio
(UAX). **Titulación:** Grado en Ingeniería Matemática. Trabajo académico
universitario de investigación; no destinado a diagnóstico ni decisiones clínicas.

## Alcance y estado de entrega

Se perfecciona la aplicación existente, sin reentrenar ni modificar src/, los
checkpoints, las configuraciones históricas, las particiones, las etiquetas o los
informes de evaluación. El test externo no se vuelve a evaluar. Las cifras mostradas
proceden del paquete verificado y del análisis OOF ya guardado.

La versión local y el paquete limpio se han probado. **La entrega web no está
cerrada hasta publicar y verificar una URL real**. La autorización final se recibió y el repositorio de aplicación ya se publicó.
Queda pendiente la aceptación personal de términos/permisos de Streamlit y la
verificación del despliegue con URL real. No se ha comprado ningún servicio.

## Correcciones implementadas

- Botón directo al ejemplo público: carga PRE/EARLY/LATE y selecciona imágenes,
  para probar sin rellenar clínica ficticia. Las etiquetas y los datos clínicos
  del ejemplo no se precargan. Se mantiene la opción multimodal con datos explícitos.
- Resumen previo: modalidad, umbral, número de imágenes/cortes y cinco modelos.
- Gráfico por fold: cinco categorías X, puntuación Y en 0–1, media y umbral como
  líneas identificadas; la media se contrasta matemáticamente con los cinco miembros.
- Defensa: UUID no personales de sesión/ejecución, controles bloqueados al registrar,
  CSV mínimo y reinicio con confirmación. Se conserva el flujo de cinco muestras.
- Métricas OOF 0–1 y Youden −1–1 en gráficos separados. Matrices de dos en dos con
  recuentos enteros y colores categóricos para acierto/error, no colores ambiguos
  que se interpreten como frecuencia. Ejes X=predicción, Y=etiqueta real.
- Loss completa 0–1 si cabe; rango ampliado si excede 1, zoom automático/manual
  identificado. Mejor época marcada en loss, accuracy y AUC. Media por época de
  loss/AUC/accuracy con DE muestral y número de folds; no se rellenan épocas.
- Leyendas bajo las gráficas, títulos separados, marcadores discretos y tablas
  compactas. La tabla completa sigue disponible en un desplegable y como CSV.
- Arquitectura: operaciones, dimensiones, dropout y parámetros obtenidos de redes
  reales cargadas. La rama clínica y la concatenación se muestran visualmente.
- Explorador: procedencia pública solo cuando corresponde, región manual visible
  sobre PRE, curva temporal en PNG/255 y vista 3D opcional sin anatomía inventada.
- Descargas con conexión cerrada y limpieza del temporal incluso ante errores;
  comprobación de hashes antes de hacer visible el paquete completo. ZIP de entrada
  limitado a formatos estándar, sin enlaces/cifrado ni extracción a disco.
- Autoría real en portada y pie; identidad visual azul marino/turquesa conservada.

## Matriz de cumplimiento del enunciado oficial

Fuente leída completa: [caso práctico, ocho páginas](caso_breastdcedl.pdf).
Se distingue lo implementado, lo documentado históricamente y lo pendiente.

| Requisito oficial | Implementación o evidencia | Estado |
|---|---|---|
| pCR=1, no pCR=0; respuesta tras tratamiento, no cáncer/no cáncer | Inicio, metodología y predicción; aviso educativo en todas las páginas | Comprobado |
| No interpretar puntuación como curación/supervivencia o decisión de cirugía | Avisos de inferencia y limitaciones | Comprobado |
| Tres fases PNG8 gris 256×256, no RGB | study_from_files/triplet; tests de formato, tamaño, duplicados y fases | Comprobado |
| Canales temporales y realce; geometría sincronizada | Operadores originales de entrenamiento; vista separada PRE/EARLY/LATE/diferencias | Comprobado |
| Unidad paciente y particiones sin fuga | Folds oficiales conservados; contrato de cinco miembros, OOF y agregación por paciente | Heredado y verificado sin modificar |
| Estadísticas ajustadas exclusivamente con train | Estadísticas verificadas propias de cada checkpoint; igualdad con pipeline original | Comprobado |
| CNN 2D propia desde cero, sin transfer learning | src/model.py y src/phase13_model.py intactos; seis Conv2d, tres bloques, 72.593/72.600 parámetros | Comprobado |
| Un logit y BCEWithLogitsLoss; comparación ponderada/no ponderada | Entrenamiento histórico; comparación documentada en FASE3/FASE4/FASE5; no se ejecuta entrenamiento en esta entrega | Documentado históricamente |
| Configuración, semilla, parada y parámetros guardados | Artefactos originales y desplegables de configuración; seed/hparams no alterados | Comprobado |
| Dispositivo y tiempos de ejecución | Inferencia CPU real, tiempo medido; logs de entorno limpio | Comprobado para la app; entrenamiento GPU histórico no repetido |
| Test solo final, sin tuning | Resultados externos guardados al umbral 0,50; no hay botón de reevaluación | Comprobado |
| Cargar tres fases o elegir ejemplo | Tres controles PNG, varios cortes/ZIP y botón de ejemplo público | Comprobado |
| Mostrar imágenes y realce antes de inferir | Vista simultánea y advertencia de corte único | Comprobado |
| Probabilidad, clase, umbral y checksum | Resultado, cinco puntuaciones, arquitectura y trazabilidad | Comprobado |
| Dispositivo y tiempo de respuesta visibles | CPU/tiempo real en tarjetas y CSV | Comprobado |
| Parámetros de entrenamiento y métricas internas | Arquitectura y cuatro pestañas de resultados | Comprobado |
| Errores claros, entradas limitadas, no rutas arbitrarias ni ejecución de uploads | Validadores PNG/ZIP, límites 1 MiB/fase, 128 MiB/conjunto y 256 cortes | Comprobado |
| model.eval() e inferencia sin gradientes, reproducible | Contraste numérico con originales y pesos reales; estado BN/pesos sin cambios | Comprobado |
| Privacidad: validación/etiquetas del profesor no publicadas | No se han utilizado esas muestras; cargas solo en memoria; paquete sin filas/IDs de paciente | Comprobado en el paquete de app |
| URL accesible durante defensa, no solo localhost | Punto de entrada deploy/app.py y dependencias CPU; repositorio separado preparado | **Pendiente autorización y despliegue real** |
| Repositorio, pesos, informe y enlace reproducibles | ZIP de código limpio; pesos reducidos en release con SHA256 fijo; guía de publicación | Preparado; publicación del nuevo repo pendiente |
| Ejemplos con licencia y atribución | Ejemplo de train sin etiqueta/ID; CC BY-NC 4.0, fuente Zenodo y ATTRIBUTION.md | Comprobado |
| Trabajo individual y autoría del estudiante | Nombre, UAX y titulación confirmados por el autor e incorporados a UI | Comprobado |
| Exactamente cinco diapositivas | No se ha encontrado una presentación .ppt/.pptx en el repositorio | **Pendiente aportar/verificar las cinco diapositivas** |
| Cinco muestras privadas de cinco pacientes, una a una | Flujo probado con cinco entradas técnicas y CNN reales, sin reentrenar ni reiniciar servidor | Funcionamiento comprobado; muestra privada real pendiente de defensa |
| Guardar puntuación, umbral y clase; etiquetas solo después | CSV de sesión, bloqueo de umbral, UUID y registro posterior de etiqueta | Comprobado |
| Cinco muestras no estiman rendimiento clínico | Aviso explícito en pantalla y documentación | Comprobado |

## Matriz de aceptación de la aplicación

| Comprobación solicitada | Evidencia ejecutada |
|---|---|
| 1. Arranque limpio | venv sin paquetes del entorno de entrenamiento; instalación deploy/requirements.txt y pip check correctos; deploy/app.py arranca |
| 2. Cinco secciones | AppTest en ambos entornos y navegación de navegador |
| 3. Cuatro pestañas de resultados | Comparación interna, curvas/folds, OOF y test congelado; AppTest y revisión de UI |
| 4. Inferencia de imagen pública | Botón público → cinco checkpoints → media real; tests y navegador |
| 5. Multimodal con clínica válida | Edad/subtipo explícitos en prueba funcional; equivalencia adicional con pipeline y clínica del caso público en suite de referencia |
| 6. Diez checkpoints | Carga estricta y hashes por archivo; 5 por modalidad |
| 7. Agregación | Media de cortes por modelo y media de cinco miembros; igualdad numérica y comprobación en gráfico |
| 8. Referencia CPU | Tests de igualdad con rutinas originales, sin cambiar operadores de preprocesamiento |
| 9–10. Umbrales 0,50/0,31 | Original en ambas modalidades; recomendado solo multimodal y advertido como exploratorio |
| 11. Cinco muestras | Cinco entradas consecutivas técnicas; no se presentan como validación clínica |
| 12. CSV | Columnas mínimas, UUID, timestamp, umbral/origen y etiqueta revelada opcional; sin nombres/IDs/imágenes/clínica |
| 13. Etiquetas | No se cargan labels en inferencia; se registran después; bloqueo de umbral y muestras duplicadas |
| 14. Entradas inválidas | Ausentes, duplicadas, corruptas, RGB, tamaños, rutas, ZIP malicioso y clínica desconocida/ausente |
| 15–16. Exploración/3D | Fases sincronizadas, diferencias, ROI, z relativos y superficie de intensidad para un corte |
| 17. Escalas/leyendas | Tests de rango y datos sin alteración; revisión visual; Youden separado |
| 18. Matrices | Posiciones TN/FP/FN/TP, enteros y ejes comprobados; colores categóricos explicados |
| 19. Integridad | SHA256 de fuentes protegidas, particiones y artefactos históricos sin cambios |
| 20. Despliegue | ZIP limpio comprobado, descarga anónima de modelos, entorno limpio Windows y resolución binaria Linux 3.13 |

## Pruebas ejecutadas y límites

- **55 pruebas** de servicios, UI, umbrales, aceptación y empaquetado: sin fallos ni
  omitidas. Se impidió explícitamente backward/Optimizer.step en la suite.
- **11 pruebas portables** ejecutadas desde el paquete limpio en un entorno nuevo:
  sin fallos ni omitidas. No necesitan el dataset ni los CSV históricos.
- pip check sin dependencias rotas. Python limpio 3.13.7, torch 2.14.0+cpu,
  Streamlit 1.66.1 y Plotly 7.1.0. Resolución binaria de dependencias para Linux
  x86_64/Python 3.13 correcta; no equivale a ejecutar Linux.
- Prueba funcional de un corte público, dos llamadas por modalidad: imágenes
  aproximadamente 0,195–0,232 s y multimodal 0,266–0,271 s. Estos tiempos excluyen
  carga inicial, dependen de este PC y **no son métricas clínicas**.
- Descarga/verificación en el arranque aislado: aproximadamente 2,0 s. Pico de
  working set medido en el proceso de inferencia, antes de abrir la UI AppTest:
  aproximadamente 387,7 MiB. No estima el máximo con múltiples usuarios o 256 cortes.
- Docker no se ha ejecutado: no hay motor instalado. El build/URL de Community
  Cloud y el comportamiento real bajo concurrencia de hosting quedan por verificar.
- Navegador real: cinco páginas y cuatro pestañas de resultados, inferencia de
  imagen y multimodal sobre el ejemplo público (clínica conocida introducida
  explícitamente), 3D y revisión de gráficas. Vistas de 1366×768, 1920×1080 y
  800×900. Se corrigió el truncado de botones a 800 píxeles apilando columnas;
  no se detectó desbordamiento horizontal del documento en las medidas comprobadas.
- Las cinco muestras privadas del profesor no se han consultado ni inventado.
- Integridad final: **315 archivos protegidos** de esta auditoría y **566 archivos
  históricos** del inventario anterior contrastados por SHA256, sin cambios. Ambos
  inventarios se solapan; las cifras no deben sumarse como archivos únicos.

Registros y capturas se conservan solo en backups/ (ignorado). Los datos originales
y las copias de seguridad no se incluyen en el paquete público.

## Hallazgo de privacidad del repositorio histórico

El repositorio de entrenamiento ya versiona metadata/patients.csv, samples.csv,
internal_train.csv, internal_val.csv y excluded_patients.csv. Contienen metadatos,
identificadores o etiquetas individuales. Esto es un **hallazgo previo**, no una
adición de esta entrega. No se puede afirmar que ese repositorio completo carezca
de información por paciente. También hay figuras históricas versionadas.

Se han conservado todos esos archivos y el historial, sin git rm, reset o force-push.
El autor ha elegido preparar un **repositorio separado para la app**, a partir de
una allowlist de código y configuración; no es una copia recursiva del repositorio.
Ese paquete excluye metadata/, dataset/, breastdcedl/, models/, entornos, backups
y artefactos clínicos. La exposición histórica debe revisarse por separado antes
de una difusión adicional; crear el repo limpio no borra el historial anterior.

El release de inferencia existente se revisó: solo diez pesos reducidos,
historiales agregados, curvas/métricas agregadas, atribución y una tripleta pública
de train sin etiqueta/ID/clínica. El SHA256 está fijado en config/app_release.json.
No se ha publicado ni sustituido un release en esta auditoría.

Licencia de datos confirmada en la fuente oficial:
https://zenodo.org/records/18114231 (License Information / Rights): CC BY-NC 4.0,
con atribución y uso no comercial. La licencia del código no sustituye esas condiciones.

## Ejecución Windows y publicación

Ver [guía del repositorio limpio](../deploy/README_APP.md) para comandos exactos de
venv, instalación, pip check, diagnóstico, arranque y pruebas portables.

Empaquetado desde el repositorio de entrenamiento (usar una salida nueva):

```powershell
.\.env\Scripts\python.exe -B deploy/prepare_space.py --target streamlit --output models/breastdcedl_ai_lab_final
```

La salida final existente ya está preparada. El generador nunca sobrescribe una
salida anterior. La carpeta contiene README específico, .gitignore, deploy/app.py,
deploy/requirements.txt, app.py, app_components/, app_services/, src/ intacto,
config/config.py y tres JSON de app, assets/ vectoriales/CSS, prueba portable y
atribución/licencia. No incluye las campañas ni las referencias individuales OOF.

ZIP del código publicado: **77 archivos, 207.762 bytes**. SHA256:
`4d3bec1c259a7bd93e5f4b2c0a10f8f39d2e9cbbcb4aa32528262acc88d87de0`.
Archivo: models/breastdcedl_ai_lab_public.zip, creado desde los 77 blobs del commit público.
Las pruebas y el servidor se ejecutan desde una copia separada; la carpeta/ZIP
listos para publicación permanecen limpios, sin cachés descargadas durante QA.

Repositorio público limpio creado y verificado: **[Juandeleongarcia/breastdcedl-ai-lab](https://github.com/Juandeleongarcia/breastdcedl-ai-lab)**.
Commit de aplicación: `c298a33fa5ba01770410b3f466cffc28d80b2747` (77 archivos). Punto de entrada **deploy/app.py**, rama **main**, Python
**3.13**, sin secrets. Su URL no se inventa: se anotará tras un despliegue verificado.
El código de origen de estas mejoras puede sincronizarse en breastcancer sin añadir
metadata; la publicación de código ha sido autorizada por el autor; el despliegue web sigue pendiente de su acceso personal.

## Limitaciones para explicar en la defensa

1. Cohorte retrospectiva y desbalanceada; accuracy puede reflejar la clase mayoritaria.
2. AUC mide ordenación; sensibilidad depende del umbral. Test congelado: AUC
   0,6722, sensibilidad cero, matriz [[122,1],[53,0]] al umbral original 0,50.
3. El umbral 0,31 se seleccionó de forma exploratoria en OOF, que también sirvió
   para seleccionar épocas. No es CV anidada ni validación independiente tras tuning.
4. OOF da una predicción del modelo que excluyó a cada paciente; su distribución
   no es la del ensemble de cinco miembros. La transferencia del umbral es provisional.
5. No se ha demostrado calibración, utilidad clínica, equidad por subgrupos o
   validación prospectiva. La dispersión entre cinco miembros no es un IC clínico.
6. Una tripleta es un corte, no un estudio completo. La UI no verifica por sí sola
   adquisición, registro anatómico, representatividad o completitud del estudio.
7. No hay metadatos geométricos suficientes para reconstrucción 3D anatómica.
   ROI es selección manual; el modelo no proporciona segmentación ni explicación causal.
8. Las cinco muestras prueban funcionamiento; no estiman fiablemente generalización.
9. El archivo subido se transmite al servidor de hosting para procesarlo en RAM;
   no se almacena por la app, pero el proveedor tiene sus propias condiciones.

## Cinco comprobaciones personales antes de entregar

1. Abrir la **URL pública real** desde otra sesión/red y comprobar que no depende
   del PC local, que carga los pesos y que el nombre/autoría son correctos.
2. Pulsar el ejemplo público y verificar cinco puntos, media, umbral, clase,
   dispositivo y tiempo. Cambiar a multimodal solo con clínica conocida o ausencia declarada.
3. Ensayar cinco entradas técnicas distintas, revelar etiquetas solo después,
   descargar el CSV y verificar cinco filas/UUID/umbrales; probar el reinicio confirmado.
4. Revisar a tamaño de pantalla de la defensa las cuatro pestañas, loss/leyendas,
   matrices, OOF 0,31 y test congelado 0,50; no recalcular test ni ajustar con sus resultados.
5. Revisar el contenido exacto del repo limpio y las **cinco diapositivas**; confirmar
   atribución/licencia, ausencia de datos por paciente y capacidad de explicar las limitaciones.

## Git y autorización final

El punto de partida era 3572fdc6a1cfa03890eb8b04ea7e22982b3bb89b. La autorización final se recibió tras la revisión y las pruebas. Se ha creado y
publicado el repositorio limpio de aplicación; la revisión de origen se publica
mediante un commit específico. Los hashes finales y la sincronización se verifican
contra los remotos. **No se ha desplegado Community Cloud sin el acceso personal
del titular**. No se reescribe el historial ni se retiran las particiones históricas.
