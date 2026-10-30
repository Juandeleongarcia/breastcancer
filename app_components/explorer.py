import hashlib

import numpy as np
import streamlit as st

from app_services.studies import study_from_uploads, synthetic_study, display_image, MAX_BYTES
from app_components.layout import section, plot, cards, empty
from app_components import charts


def _upload_changed(widget_key):
    if not st.session_state.get(widget_key):
        st.session_state.study = None
        st.session_state.prediction = None
        st.session_state.pop('last_upload_signature',None)


def upload_study():
    widget_key = 'study_upload_'+str(st.session_state.upload_nonce)
    uploaded = st.file_uploader('Cargar estudio compatible',type=['png','zip'],accept_multiple_files=True,
        key=widget_key,on_change=_upload_changed,args=(widget_key,),
        help='Un paciente: PNG gris de 8 bits, 256×256; PRE/EARLY/LATE para cada índice z. También se admite un ZIP.')
    if uploaded:
        if sum(item.size for item in uploaded)>MAX_BYTES:
            st.session_state.study = None
            st.session_state.prediction = None
            st.session_state.pop('last_upload_signature',None)
            st.error('El estudio supera 128 MiB');return
        files = [(item.name,item.getvalue()) for item in uploaded]
        digest = hashlib.sha256()
        for name,content in files:digest.update(name.encode());digest.update(content)
        token = digest.hexdigest()
        if token != st.session_state.get('last_upload_signature'):
            try:
                study = study_from_uploads(files)
                st.session_state.study = study
                st.session_state.prediction = None
                st.session_state.last_upload_signature = token
                st.session_state.slice_pos = 0
                st.session_state.demo_mode = False
                st.session_state.study_public = False
                st.success(f'Estudio validado: {len(study.slice_indices)} cortes, tres fases sincronizadas.')
            except ValueError as error:
                st.session_state.study = None
                st.session_state.prediction = None
                st.session_state.pop('last_upload_signature',None)
                st.error(str(error))
    with st.expander('Formato de entrada y alcance',icon=':material/info:'):
        st.markdown('**Un paciente por carga.** Nombres como `PACIENTE_z012_PRE.png`, '
                    '`PACIENTE_z012_EARLY.png` y `PACIENTE_z012_LATE.png`. Repetir la tripleta para todos los cortes.')
        st.caption('PNG de 8 bits en gris, 256 × 256. No DICOM, NIfTI, RGB ni un PNG aislado. '
                   'Los cortes deben proceder de un procesamiento compatible con el dataset. '
                   'La app comprueba las tripletas aportadas; no puede verificar la completitud anatómica del estudio original. '
                   'Se procesan en memoria del servidor de la aplicación, sin extraer el ZIP ni enviar datos a servicios de inferencia externos.')


@st.dialog('Detalle del corte',width='large')
def zoom_dialog(image,title,signed):
    st.caption(title+' · ampliar con zoom, selección y desplazamiento del gráfico.')
    plot(charts.image_zoom(image,title,signed),'zoom_image')


def _signed_rgb(array):
    limit = max(float(np.abs(array).max()),1e-6)
    value = np.clip((array/limit+1)/2,0,1)
    stops = np.array([[.22,.35,.54],[.065,.12,.19],[.34,.83,.75]])
    return np.stack([np.interp(value,[0,.5,1],stops[:,i]) for i in range(3)],axis=-1)


def viewer(study,enable_3d=False):
    n = len(study.slice_indices)
    if st.session_state.get('slice_pos',0)>=n:st.session_state.slice_pos=0
    controls = st.columns([3,1,1])
    with controls[0]:
        position = st.slider('Corte sincronizado',0,n-1,key='slice_pos') if n>1 else 0
    with controls[1]:brightness=st.slider('Brillo',-.4,.4,0.,.02,key='brightness')
    with controls[2]:contrast=st.slider('Contraste',.5,2.,1.,.05,key='contrast')
    index = study.slice_indices[position]
    st.caption(f'Corte z{index:03d} · {position+1} de {n} · Brillo/contraste solo para visualización')
    cols = st.columns(3)
    displays = []
    for channel,(column,label) in enumerate(zip(cols,['PRE','EARLY','LATE'])):
        view = display_image(study.phases[position,channel],brightness,contrast)
        displays.append(view)
        with column:
            st.image(view,caption=label,width='stretch',clamp=True,alt=f'Fase {label}, corte {index}')
    if st.toggle('Ver cambios temporales de intensidad',value=True,key='show_channels'):
        pre,early,late = study.phases[position].astype(np.float32)/255.
        values = np.stack([pre,early-pre,late-pre])
        st.caption('Diferencias temporales calculadas con las intensidades originales. Esta vista no modifica el clipping ni la normalización utilizados por los modelos.')
        titles = ['PRE','EARLY − PRE','LATE − PRE']
        for column,value,title in zip(st.columns(3),values,titles):
            with column:
                st.image(_signed_rgb(value),caption=title,width='stretch',clamp=True,alt='Canal '+title)
                st.caption(f'Rango real: {value.min():.3f} a {value.max():.3f}')
    else:values=None;titles=[]
    a,b = st.columns([3,1])
    choices = ['PRE','EARLY','LATE'] + (['Canal: '+title for title in titles] if values is not None else [])
    selected = a.selectbox('Imagen a ampliar',choices,key='zoom_choice')
    if b.button('Ampliar',icon=':material/zoom_in:',width='stretch',key='zoom_button'):
        if selected in ['PRE','EARLY','LATE']:
            zoom_dialog(displays[['PRE','EARLY','LATE'].index(selected)],selected,False)
        else:
            zoom_dialog(values[choices.index(selected)-3],selected,True)
    if enable_3d:
        st.subheader('Representación 3D · opcional')
        st.caption('Varios cortes: planos ordenados por su índice axial relativo. Un corte: altura matemática de intensidades. No se inventan orientación, espaciado milimétrico, volúmenes o máscaras.')
        if st.toggle('Activar visualización 3D',key='enable_3d'):
            phase = st.selectbox('Fase del visor 3D',['PRE','EARLY','LATE'],key='volume_phase')
            try:
                if n>1:
                    figure,count = charts.slice_stack(study,['PRE','EARLY','LATE'].index(phase),position)
                    plot(figure,'stack_3d')
                    st.caption(f'{count} de {n} planos representados; borde turquesa: corte seleccionado. Coordenadas relativas, sin mm ni interpolación de cortes.')
                else:
                    plot(charts.intensity_surface(study,['PRE','EARLY','LATE'].index(phase),position),'surface_3d')
                    st.caption('Un único corte no permite reconstruir anatomía 3D. La altura representa intensidad, no profundidad física.')
            except (ValueError,RuntimeError) as error:
                st.info('La visualización 3D no está disponible en este entorno. La vista 2D sigue funcionando. '+str(error))
    st.subheader('Explorar el realce temporal de una región')
    a,b,c=st.columns(3)
    x=a.number_input('Centro X (píxel)',0,255,128,key='temporal_x')
    y=b.number_input('Centro Y (píxel)',0,255,128,key='temporal_y')
    radius=c.slider('Semiancho de región cuadrada (píxeles)',0,32,8,key='temporal_radius')
    with st.expander('Ver la región manual sobre PRE'):
        figure=charts.image_zoom(study.phases[position,0]/255.,'Región manual · PRE')
        figure.add_shape(type='rect',x0=max(0,x-radius),x1=min(255,x+radius),
                         y0=max(0,y-radius),y1=min(255,y+radius),line=dict(color=charts.TEAL,width=2))
        plot(figure,'temporal_roi')
        st.caption('Cuadrado seleccionado manualmente para medir intensidades; no representa una máscara o segmentación tumoral.')
    plot(charts.temporal_curve(study,position,int(x),int(y),int(radius)),'temporal_curve')
    st.caption('Se representan tres puntos temporales, sin tiempos físicos ni interpolación fisiológica. La región es una selección manual, no una segmentación tumoral. Brillo y contraste no afectan a esta curva.')


def render():
    section('EXPLORACIÓN TEMPORAL','Explorador DCE-MRI','Tres fases sincronizadas. Una lectura espacial y temporal del estudio.')
    upload_study()
    if st.session_state.catalog is not None and st.button('Explorar ejemplo público de train',key='explorer_example'):
        from app_services.lab import public_example
        st.session_state.study=public_example(st.session_state.bundle_root)
        st.session_state.slice_pos=0;st.session_state.demo_mode=False
        st.session_state.study_public=True
    if st.session_state.study is None and st.session_state.catalog is not None and 'demo_mode' not in st.session_state:
        from app_services.lab import public_example
        st.session_state.study=public_example(st.session_state.bundle_root)
        st.session_state.demo_mode=False
        st.session_state.study_public=True
    if st.session_state.study is not None and st.session_state.get('study_public') and not st.session_state.get('demo_mode'):
        st.caption('Ejemplo público autorizado: BreastDCEDL MinCrop · CC BY-NC 4.0 · Fridman et al. · doi:10.5281/zenodo.18114231. No se suministra etiqueta.')
    study = st.session_state.study
    demo = st.toggle('Demostración sintética del visor',value=study is None,key='demo_mode')
    if demo:
        study = synthetic_study()
        st.warning('DATOS SINTÉTICOS · Formas geométricas para probar el visor. No representan anatomía real y no se usan para predecir pCR.')
    if study is None:
        empty('El explorador está preparado','Carga tripletas de un único paciente o activa la demostración sintética para conocer el visor.')
        return
    cards([('Cortes',str(len(study.slice_indices)),'Ordenados por índice z'),('Fases','3','PRE · EARLY · LATE'),
           ('Resolución','256 × 256','Píxeles · 8 bits por fase'),('Geometría','Esquemática','Sin orientación ni espaciado físico')])
    st.write('')
    viewer(study,enable_3d=True)
