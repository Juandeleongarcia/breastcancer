"""Identidad visual, navegación, estado vacío y conexiones locales."""
from html import escape

import streamlit as st

from app_services.workspace import ROOT, NOTICE, PENDING_ITEMS, Workspace, load_workspace

PAGES = [('Inicio','home','home'),('Explorador DCE-MRI','explorer','view_in_ar'),
         ('Predicción de pCR','predict','biotech'),('Resultados y CV','results','monitoring'),
         ('Cómo funciona una CNN','method','account_tree')]


def goto(page):
    st.session_state.page = page


def initialize():
    st.session_state.setdefault('page','home')
    st.session_state.setdefault('study',None)
    st.session_state.setdefault('prediction',None)
    st.session_state.setdefault('upload_nonce',0)
    if st.session_state.get('visual_base_version') != 1:
        for key in ['selected_model','architecture_preview','architecture_stage','inference_plan','paths','settings_error']:
            st.session_state.pop(key,None)
        st.session_state.visual_base_version = 1
    if 'workspace' not in st.session_state:
        try:st.session_state.workspace = load_workspace()
        except (OSError,ValueError,TypeError):
            st.session_state.workspace = Workspace()
            st.session_state.settings_error = 'No se pudo leer la configuración visual. La aplicación sigue disponible en modo vacío.'
    st.session_state.prediction = None
    for key in list(st.session_state.keys()):
        if key.startswith('bundle_'):st.session_state.pop(key,None)


def clear_study():
    st.session_state.study = None
    st.session_state.prediction = None
    st.session_state.upload_nonce += 1
    st.session_state.pop('last_upload_signature',None)


def sidebar():
    with st.sidebar:
        st.html('<div class="b-brand">BreastDCEDL <span>AI</span></div><div class="b-brand-sub">Treatment response research</div>')
        st.html('<div class="b-overline">Espacio de investigación</div>')
        for label,name,icon in PAGES:
            st.button(label,key='nav_'+name,icon=':material/'+icon+':',width='stretch',
                      type='primary' if st.session_state.page==name else 'tertiary',on_click=goto,args=(name,))
        st.html('<div class="b-live"><b>Base visual preparada</b><br>Integración del modelo pendiente<br>Exploración y resultados locales</div>')
        with st.expander('Pendiente de integrar',icon=':material/pending:'):
            for label in PENDING_ITEMS:st.caption(label+' · Pendiente')
        if st.session_state.study is not None:
            st.button('Descartar estudio',icon=':material/delete_outline:',on_click=clear_study,width='stretch')
        st.html('<div class="b-small-note">Prototipo académico<br>Sin decisiones clínicas</div>')


def topbar():
    st.html('<div class="b-topline"><div class="crumb">Research workspace / DCE-MRI</div>'
            '<div class="b-pill"><i></i>Integración del modelo pendiente</div></div>')
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
