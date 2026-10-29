import streamlit as st

from app_components.layout import cards, goto, pending_integration
from app_services.workspace import ROOT


def render():
    left,right = st.columns([1.25,1],gap='large')
    with left:
        st.html('<div class="b-hero"><div class="b-kicker">DCE-MRI · DEEP LEARNING · RESPUESTA AL TRATAMIENTO</div>'
                '<h1>Una mirada temporal a la <span>respuesta al tratamiento.</span></h1>'
                '<p>Exploramos si una CNN propia puede estimar la respuesta patológica completa (pCR) '
                'a partir de imágenes dinámicas de resonancia de mama. Una herramienta para investigar, visualizar y comprender.</p></div>')
        a,b = st.columns([1,1])
        a.button('Explorar estudio',icon=':material/view_in_ar:',type='primary',width='stretch',on_click=goto,args=('explorer',),key='home_explore')
        b.button('Ver validación',icon=':material/monitoring:',width='stretch',on_click=goto,args=('results',),key='home_results')
        st.caption('CNN 2D propias · sin transfer learning · evaluación por paciente')
    with right:
        with st.container(border=True):
            st.image(str(ROOT/'assets/temporal_channels.svg'),width='stretch',alt='Esquema de los tres canales temporales')
            st.caption('ESQUEMA ILUSTRATIVO · NO ES UNA RESONANCIA')
    st.write('')
    cards([('Entrada','Tres fases DCE','PRE · EARLY · LATE, no RGB'),('Integración','Pendiente','Un único modelo seleccionado posteriormente'),
           ('Protocolo','5 folds oficiales','Sin mezclar cortes de un paciente'),('Agregación','Por paciente','Media de probabilidades de sus cortes')])
    st.write('')
    st.subheader('De la imagen a una estimación interpretable')
    st.caption('Pipeline conceptual previsto. La arquitectura y la configuración de inferencia se incorporarán cuando se seleccione el modelo.')
    steps = [('01','DCE-MRI','PRE · EARLY · LATE'),('02','Preprocesamiento','Contrato pendiente de integración'),
             ('03','CNN propia','Características espaciales'),('04','Por corte','Probabilidades estimadas'),
             ('05','Por paciente','Media de los cortes aportados')]
    st.html('<div class="b-pipeline">'+''.join(f'<div class="b-step"><b>{n}</b><strong>{title}</strong><span>{text}</span></div>' for n,title,text in steps)+'</div>')
    a,b = st.columns(2,gap='large')
    with a:
        st.html('<div class="b-panel"><h3>El contraste cambia con el tiempo</h3>'
                '<div class="b-phase"><strong>PRE · referencia inicial</strong><p>Imagen previa al contraste utilizada como referencia.</p></div>'
                '<div class="b-phase"><strong>EARLY − PRE · cambio temprano</strong><p>Diferencia de intensidad tras la adquisición temprana.</p></div>'
                '<div class="b-phase"><strong>LATE − PRE · cambio tardío</strong><p>Diferencia respecto a PRE en la fase tardía.</p></div></div>')
    with b:
        st.html('<div class="b-panel"><h3>¿Qué significa pCR?</h3><p>Respuesta patológica completa: ausencia de signos de cáncer '
                'en las muestras de tejido evaluadas después del tratamiento. La etiqueta sigue el criterio definido por el dataset.</p>'
                '<div class="b-feature"><strong>La CNN estima; la anatomía patológica determina.</strong><br>'
                'No medimos curación ni sustituimos la evaluación clínica.</div>'
                '<div class="b-feature"><strong>La unidad de evaluación es el paciente.</strong><br>'
                'AUC, sensibilidad y balanced accuracy ayudan a interpretar resultados que la accuracy por sí sola no explica.</div></div>')
        st.caption('Definición general: [NCI — Pathologic complete response](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/pathologic-complete-response).')
    st.subheader('Preparado para el siguiente paso')
    pending_integration()
