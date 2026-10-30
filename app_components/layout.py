"""Identidad visual, navegación, estado vacío y conexiones locales."""
from html import escape

import streamlit as st

from app_services.workspace import ROOT, NOTICE, PENDING_ITEMS, Workspace, load_workspace
from app_services.defence import Defence

PAGES = [('Inicio','home','home'),('Probar el modelo','predict','biotech'),
         ('Explorador DCE-MRI','explorer','view_in_ar'),('Arquitectura y método','method','account_tree'),
         ('Resultados y validación','results','monitoring')]


def goto(page):
    st.session_state.page = page


def initialize():
    st.session_state.setdefault('page','home')
    st.session_state.setdefault('study',None)
    st.session_state.setdefault('prediction',None)
    st.session_state.setdefault('upload_nonce',0)
    if st.session_state.get('visual_base_version') != 2:
        for key in ['selected_model','architecture_preview','architecture_stage','inference_plan','paths','settings_error']:
            st.session_state.pop(key,None)
        st.session_state.visual_base_version = 2
        st.session_state.prediction=None
        st.session_state.defence=Defence()
        st.session_state.input_study=None
        st.session_state.input_nonce=0
    if 'workspace' not in st.session_state:
        try:st.session_state.workspace = load_workspace()
        except (OSError,ValueError,TypeError):
            st.session_state.workspace = Workspace()
            st.session_state.settings_error = 'No se pudo leer la configuración visual. La aplicación sigue disponible en modo vacío.'
    if 'catalog' not in st.session_state:
        try:
            from app_services.lab import read_catalog
            root,catalog=read_catalog()
            st.session_state.catalog=catalog;st.session_state.bundle_root=str(root)
        except (OSError,ValueError,KeyError,TypeError) as error:
            st.session_state.catalog=None;st.session_state.bundle_root=None
            st.session_state.artifact_error=str(error)


def clear_study():
    st.session_state.study = None
    st.session_state.prediction = None
    st.session_state.upload_nonce += 1
    st.session_state.pop('last_upload_signature',None)


def sidebar():
    with st.sidebar:
        st.html('<div class="b-brand">BreastDCEDL <span>AI Lab</span></div><div class="b-brand-sub">Predicción de respuesta pCR</div>')
        st.html('<div class="b-overline">Espacio de investigación</div>')
        for label,name,icon in PAGES:
            st.button(label,key='nav_'+name,icon=':material/'+icon+':',width='stretch',
                      type='primary' if st.session_state.page==name else 'tertiary',on_click=goto,args=(name,))
        ready=st.session_state.catalog is not None
        st.html('<div class="b-live"><b>'+('Integridad de pesos verificada' if ready else 'Artefactos no disponibles')+
                '</b><br>CNN propias · inferencia sin entrenamiento<br>Evaluación por paciente</div>')
        if ready:
            defense=st.session_state.defence
            st.caption(f'Demostración · {len(defense.records)} / 5 muestras registradas')
        else:
            with st.expander('Diagnóstico de preparación'):
                st.caption(st.session_state.get('artifact_error','Consultar STREAMLIT_GUIA.md'))
        if st.session_state.study is not None:
            st.button('Descartar estudio',icon=':material/delete_outline:',on_click=clear_study,width='stretch')
        st.html('<div class="b-small-note">Prototipo académico<br>Sin decisiones clínicas</div>')


def topbar():
    status='Inferencia CPU · artefactos verificados' if st.session_state.catalog else 'Preparación de artefactos pendiente'
    st.html('<div class="b-topline"><div class="crumb">Laboratorio de investigación / DCE-MRI</div>'
            '<div class="b-pill"><i></i>'+status+'</div></div>')
    st.html('<div class="b-research-strip">'+escape(NOTICE)+'</div>')


def section(kicker,title,subtitle):
    st.html('<div class="b-kicker">'+escape(kicker)+'</div>')
    st.title(title)
    st.caption(subtitle)


def cards(items):
    columns = st.columns(len(items),gap='small')
    for column,(label,value,detail) in zip(columns,items):
        with column:
            st.html(f'<div class="b-card"><div class="label">{escape(str(label))}</div>'
                    f'<div class="value">{escape(str(value))}</div><div class="detail">{escape(str(detail))}</div></div>')


def pending_integration():
    for column,labels in zip(st.columns(2),[PENDING_ITEMS[:3],PENDING_ITEMS[3:]]):
        with column:
            for label in labels:st.caption(label+' · Pendiente')


def empty(title,text,icon='◇'):
    st.html(f'<div class="b-empty"><div class="icon">{escape(icon)}</div><h3>{escape(title)}</h3><p>{escape(text)}</p></div>')


def notice():
    st.html('<div class="b-notice">'+escape(NOTICE)+'<br>Las puntuaciones estiman una respuesta; no constituyen un diagnóstico ni una probabilidad clínica calibrada.</div>')


def plot(figure,key):
    st.plotly_chart(figure,theme=None,key=key,width='stretch',config={'displaylogo':False,'scrollZoom':False})


def load_style():
    st.html('<style>'+(ROOT/'assets/dashboard.css').read_text(encoding='utf-8')+'</style>')


@st.cache_resource(show_spinner=False)
def resource(root,kind,checksum):
    from app_services.lab import load_resource
    return load_resource(root,kind,'cpu')


def clear_input():
    st.session_state.prediction=None
    st.session_state.input_study=None


def next_sample():
    st.session_state.defence.next()
    st.session_state.input_nonce+=1
    clear_input();st.session_state.study=None
    for key in list(st.session_state):
        if key.startswith(('input_','upload_','reveal_')) and key not in ['input_nonce','input_study']:
            st.session_state.pop(key,None)
