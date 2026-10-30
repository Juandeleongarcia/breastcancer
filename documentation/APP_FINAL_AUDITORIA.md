# Auditoría previa — aplicación final (30/10/2026)

Se ha leído completo `documentation/caso_breastdcedl.pdf` (ocho páginas), antes
de modificar código. Secciones 6, 7, 9–13 y 15 revisadas expresamente.

## Inventario y decisiones

| Componente existente | Estado | Acción |
|---|---|---|
| app.py, navegación y CSS F12 | Cinco pantallas funcionales; integración deshabilitada | Mantener identidad visual, conectar artefactos reales y actualizar navegación |
| app_services/studies.py | Validación PNG8 y ZIP en memoria, fases y pacientes sincronizados | Reutilizar; añadir carga guiada por roles para una tripleta |
| app_services/preprocessing.py | Operadores originales del Dataset | Conservar; en multimodal utilizar ImageTransform y ClinicalTransform de F13 |
| app_services/inference.py | Adaptador CPU de checkpoints de imagen | Reutilizar para comprobar equivalencia; ampliar integración sin editar src/ |
| app_services/results.py y charts.py | Lectura CV9 y curvas sin rellenar épocas | Reutilizar agregación/figuras; adaptar lectura a F13 y resultados externos congelados |
| Explorador | 2D, contraste, zoom y pila esquemática | Añadir coordenadas relativas z reales, superficie de intensidad de un corte y curva temporal de ROI |
| Predicción | Botón deshabilitado, sin registros de defensa | Implementar flujo privado por sesión, 1–5, sin guardar imágenes ni identificadores |
| Modelo de imagen | Campaña completa dentro de carpeta duplicada | Detectar raíz real, no mover ni renombrar |
| Modelo multimodal | Cinco folds completos, checkpoints reales legibles en CPU | Integrar cinco mejores checkpoints, con perfiles independientes |
| Test externo | Informe completo de ensemble disponible | Solo leer métricas/predicciones guardadas para visualización e integridad; nunca inferir test desde Streamlit |
| Despliegue | Solo localhost; pesos ignorados por Git | Paquete de inferencia reducido y verificado por SHA256; dependencias CPU separadas y guía de nube |

## Hallazgos confirmados

- Diez checkpoints reales disponibles: cinco de imagen y cinco multimodales.
- Épocas de imagen: 4, 5, 2, 20, 6. Épocas multimodales: 6, 2, 7, 5, 6.
- Parámetros comprobados en estados reales: 72.593 (imagen) y 72.600 (multimodal).
- Informe externo: AUC 0,6721889860408038; accuracy 0,6931818181818182;
  sensibilidad 0; especificidad 0,991869918699187; matriz [[122,1],[53,0]].
  Estas cifras se leerán de artefactos, no se introducirán como constantes de UI.
- El resultado es pCR/no pCR, nunca presencia/ausencia de cáncer ni supervivencia.

## Requisitos obligatorios y diferencias

- Demostración con cinco muestras de cinco pacientes, una a una; etiquetas reales
  introducidas únicamente después de predecir, sin adaptación del modelo/umbral.
- Una tripleta representa un corte. Se advertirá que no equivale al protocolo de
  agregación de múltiples cortes de un paciente completo.
- No se admiten rutas del sistema, pesos o código subidos desde el navegador.
- No se modifican src/, configuración de entrenamiento, folds, pesos o campañas.
- El enunciado exige comparar BCE ponderada/no ponderada; los experimentos previos
  ya contemplan esa comparación. Esta fase web no entrena ni cambia esa decisión.
- El enunciado permite publicar pesos y ejemplos docentes con atribución y licencia
  **CC BY-NC 4.0 para los datos**. El LICENSE local dice CC BY 4.0: no se toma como
  permiso para relajar la licencia de datos. El paquete público contendrá únicamente
  pesos de inferencia, estadísticas agregadas y un ejemplo de train autorizado,
  sin pacientes privados, etiquetas por paciente ni CSV completos de verdad terreno.
- Los artefactos .pt originales incluyen optimizador, RNG y predicciones: no se
  publicarán en bruto. Se exportará únicamente la información necesaria para inferir.
- La URL pública exige una cuenta de hosting conectada. Se preparará un despliegue
  reproducible y se distinguirá la verificación local de la publicación real.

## Orden de implementación

1. Catálogo inequívoco y paquete de inferencia seguro; lectores de resultados.
2. Inferencia CPU real y equivalencia con rutinas originales, sin reentrenamiento.
3. Flujo de cinco muestras y cinco páginas, reutilizando diseño y visor.
4. Tests sintéticos y con un ejemplo público de train; revisión visual.
5. Preparación de nube, revisión de datos incluidos y publicación del código.

## Verificación final de esta entrega

- 35 pruebas de servicios, inferencia real y UI: superadas, cero omitidas.
- Pesos y preprocesamiento: coincidencia numérica con las rutinas originales sobre
  un ejemplo público de train; diez checkpoints cargados en CPU.
- Flujo AppTest: cinco muestras consecutivas, etiquetas reveladas después de
  predecir, limpieza de entradas y CSV sin identificadores ni clínica individual.
- Los 566 archivos históricos inventariados coinciden con sus SHA256 iniciales.
  No se han modificado fuentes protegidas de entrenamiento ni particiones.
- ZIP de inferencia: 25 archivos, 2.909.149 bytes, sin etiquetas por paciente,
  optimizador, imágenes privadas ni dataset. SHA256 fijado en app_release.json.
- Paquete Docker: 70 archivos de código/configuración/recursos vectoriales;
  diagnóstico CPU independiente de metadata y dataset superado.
- Docker local no ejecutado: motor no instalado. El build Linux de la plataforma
  y la URL pública requieren una comprobación posterior al despliegue real.
- HF mostraba Docker deshabilitado por requerir PRO; no se contrató ningún plan.
  El titular eligió preparar Streamlit Community Cloud, con deploy/app.py y Python 3.13.
