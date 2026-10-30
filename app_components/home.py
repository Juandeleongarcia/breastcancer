import streamlit as st

from app_components.layout import cards, goto
from app_services.workspace import ROOT


def render():
    left,right = st.columns([1.25,1],gap='large')
    with left:
        st.html('<div class="b-hero"><div class="b-kicker">DCE-MRI · DEEP LEARNING · RESPUESTA AL TRATAMIENTO</div>'
                '<h1>Una mirada temporal a la <span>respuesta al tratamiento.</span></h1>'
                '<p>Exploramos si una CNN propia puede estimar la respuesta patológica completa (pCR) '
                'a partir de imágenes dinámicas de resonancia de mama. Una herramienta para investigar, visualizar y comprender.</p></div>')
        a,b = st.columns([1,1])
        a.button('Probar el modelo',icon=':material/biotech:',type='primary',width='stretch',on_click=goto,args=('predict',),key='home_predict')
        b.button('Explorar las resonancias',icon=':material/view_in_ar:',width='stretch',on_click=goto,args=('explorer',),key='home_explore')
        st.caption('CNN 2D propias · sin transfer learning · evaluación por paciente')
    with right:
        with st.container(border=True):
            st.image(str(ROOT/'assets/temporal_channels.svg'),width='stretch',alt='Esquema de los tres canales temporales')
            st.caption('ESQUEMA ILUSTRATIVO · NO ES UNA RESONANCIA')
    st.write('')
    catalog=st.session_state.catalog
    cards([('Desarrollo',str(catalog['modalities']['image']['development_patients']) if catalog else '—','Pacientes · separación por identidad'),
           ('Protocolo','5 folds','Validación cruzada interna'),('Comparación','2 modalidades','Imagen / imagen + clínica'),
           ('AUC externo',f"{catalog['external']['metrics']['auc']:.4f}" if catalog else '—','Ensemble multimodal · test guardado')])
    if catalog:
        st.caption('Trabajo universitario de Ingeniería Matemática · CNN 2D entrenadas desde cero.')
        st.subheader('Dos fuentes de información, una evaluación trazable')
        a,b=st.columns(2)
        for col,kind,title in [(a,'image','SPATIAL_SMALL · imágenes'),(b,'multimodal','SPATIAL_SMALL multimodal')]:
            entry=catalog['modalities'][kind];metric=entry['metrics']
            with col:
                st.html(f'<div class="b-panel"><h3>{title}</h3><p>AUC CV: <strong>{metric["auc_mean"]:.4f} ± {metric["auc_std"]:.4f}</strong> · DE muestral de cinco folds.<br>{entry["members"][0]["parameters"]:,} parámetros por red. Cinco checkpoints, cada uno con su preprocesamiento.</p></div>')
        metrics=catalog['external']['metrics']
        if metrics['recall']==0:st.warning('Test externo · sensibilidad 0 % al umbral 0,50. La discriminación ROC AUC y la clasificación binaria son aspectos distintos; esta limitación se muestra sin ocultarla.')
    st.write('')
    st.subheader('De la imagen a una estimación interpretable')
    st.caption('La predicción parte de la resonancia inicial; la respuesta real se conoce después del tratamiento y la evaluación patológica.')
    steps = [('01','DCE-MRI inicial','PRE · EARLY · LATE'),('02','Preprocesamiento','Clipping + estadísticas de train por fold'),
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
        st.html('<div class="b-panel"><h3>¿Qué significa pCR?</h3><p>Respuesta patológica completa: ausencia de enfermedad invasiva residual '
                'en mama y ganglios regionales tras el tratamiento, según la evaluación patológica y el criterio del dataset.</p>'
                '<div class="b-feature"><strong>La CNN estima; la anatomía patológica determina.</strong><br>'
                'No medimos curación ni sustituimos la evaluación clínica.</div>'
                '<div class="b-feature"><strong>La unidad de evaluación es el paciente.</strong><br>'
                'AUC, sensibilidad y balanced accuracy ayudan a interpretar resultados que la accuracy por sí sola no explica.</div></div>')
        st.caption('Definición general: [NCI — Pathologic complete response](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/pathologic-complete-response).')
    st.caption('1 = respuesta patológica completa · 0 = no pCR, enfermedad invasiva residual. No estamos detectando cáncer frente a ausencia de cáncer.')
