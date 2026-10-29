"""Explicación general de una CNN; no representa una arquitectura concreta."""
import streamlit as st

from app_components.layout import section,plot,pending_integration
from app_components import charts

TERMS = [
 ('CNN','Una red convolucional aprende patrones espaciales a partir de ejemplos. El entrenamiento se realiza fuera de esta aplicación.'),
 ('Información temporal','Las fases de un estudio dinámico aportan información de momentos diferentes del mismo corte. No son canales de color RGB.'),
 ('Convolución','Un filtro aprende a reconocer patrones locales y produce mapas de características. No identifica por sí solo una explicación clínica validada.'),
 ('Normalización','Ayuda a mantener escalas de intensidad coherentes. Sus estadísticas deben proceder únicamente del entrenamiento y se fijarán al integrar el modelo.'),
 ('BatchNorm','Es un posible componente de una CNN. Sus estadísticas se aprenden durante entrenamiento y se mantienen fijas en evaluación.'),
 ('ReLU','Es un ejemplo de activación que introduce no linealidad al conservar los valores positivos y anular los negativos.'),
 ('Pooling','Reduce la resolución de mapas de características. Permite reunir contexto, con pérdida de parte del detalle espacial.'),
 ('Representación global','Algunas CNN resumen sus mapas mediante una agregación espacial antes de clasificar. La arquitectura final determinará cómo hacerlo.'),
 ('Regularización','Técnicas como Dropout pueden reducir el sobreajuste durante entrenamiento. Su configuración queda pendiente del modelo seleccionado.'),
 ('Pérdida binaria','Compara las salidas del modelo con las etiquetas. La pérdida y su configuración se documentarán con el modelo definitivo.'),
 ('Early stopping','Detiene el entrenamiento si deja de mejorar la validación. No utiliza el test final para decidir cuándo parar.'),
 ('ROC AUC','Evalúa la ordenación de puntuaciones entre las clases a distintos umbrales. No equivale a accuracy ni demuestra calibración clínica.'),
 ('Validación por paciente','Todos los cortes del mismo paciente permanecen juntos. Cada fold se valida con pacientes excluidos de su entrenamiento.'),
 ('Sobreajuste','Puede aparecer cuando mejora la pérdida de entrenamiento y empeora la de validación. Las curvas deben mostrarse completas, sin ocultarlo.'),
 ('Test reservado','Debe permanecer separado de la selección del modelo. Sus resultados están pendientes y no se mezclan con validación cruzada.'),
 ('Limitaciones','La calidad de las imágenes, el dominio de los datos y la calibración condicionan la generalización. El prototipo no es una herramienta clínica.')]


def render():
    section('CONCEPTOS DEL PROYECTO','Cómo funciona una CNN','Un recorrido visual por sus funciones habituales, sin describir un modelo particular.')
    st.info('Esquema conceptual. La arquitectura definitiva y su número de parámetros están pendientes.')
    selected = st.selectbox('Explorar una etapa',range(len(charts.CNN_CONCEPTS)),
        format_func=lambda i:charts.CNN_CONCEPTS[i][0],key='cnn_concept')
    plot(charts.conceptual_cnn_figure(selected),'cnn_conceptual_flow')
    title,description = charts.CNN_CONCEPTS[selected]
    st.write('**'+title+'** · '+description)
    st.caption('Los bloques representan funciones generales. No fijan número de capas, filtros, tamaños intermedios ni configuración del futuro modelo.')
    st.subheader('Claves para interpretar el proyecto')
    columns = st.columns(2,gap='large')
    for i,(title,text) in enumerate(TERMS):
        with columns[i%2]:
            with st.expander(title,expanded=i<2):st.write(text)
    with st.expander('Integración posterior del único modelo seleccionado'):pending_integration()
