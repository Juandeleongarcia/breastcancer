"""Pantalla visual de inferencia futura; no carga ni ejecuta ningún modelo."""
import streamlit as st

from app_components.layout import section,empty,cards,pending_integration
from app_components.explorer import upload_study,viewer


def render():
    section('FLUJO DE ANÁLISIS','Predicción de pCR','Carga y revisión del estudio. La ejecución del modelo se incorporará en el siguiente paso.')
    st.html('<div class="b-pipeline"><div class="b-step"><b>01</b><strong>Cargar el estudio</strong><span>Tripletas temporales compatibles</span></div>'
            '<div class="b-step"><b>02</b><strong>Revisar las imágenes</strong><span>Cortes sincronizados</span></div>'
            '<div class="b-step"><b>03</b><strong>Analizar resonancia</strong><span>Pendiente de integración</span></div></div>')
    upload_study()
    st.info('Modelo pendiente de integrar')
    st.caption('La arquitectura, el número de parámetros, los pesos y la configuración de inferencia se incorporarán cuando el usuario entregue el modelo seleccionado.')
    study = st.session_state.study
    if study is not None:
        with st.expander('Revisar las imágenes cargadas',expanded=True):viewer(study)
    st.button('Analizar resonancia',type='primary',icon=':material/biotech:',disabled=True,key='analyze',width='stretch')
    cards([('Probabilidad estimada de pCR','Pendiente','Se mostrará después de una inferencia real'),
           ('Probabilidad complementaria','Pendiente','No se generan valores de demostración')])
    empty('El espacio de resultados está preparado','La aplicación presentará la estimación del único modelo seleccionado cuando su integración esté completada.')
    with st.expander('Qué falta para activar esta sección'):pending_integration()
