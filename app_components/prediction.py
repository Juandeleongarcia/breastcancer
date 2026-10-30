"""Demostración privada por sesión: cinco muestras, sin almacenamiento de imágenes."""
import streamlit as st

from app_components.layout import section,cards,plot,resource,clear_input,next_sample,empty
from app_components import charts
from app_components.explorer import _signed_rgb
from app_services import lab
from app_services.thresholds import decision
from app_components.thresholds import analysis_for
from app_services.studies import PHASES,study_from_uploads


def render():
    section('DEMOSTRACIÓN CON NUEVAS MUESTRAS','Probar el modelo',
        'Tres fases del mismo corte. Una estimación de respuesta al tratamiento, nunca un diagnóstico de cáncer.')
    catalog=st.session_state.catalog
    if catalog is None:
        empty('Modelos no disponibles','No se puede inferir sin pesos verificados. Consultar el diagnóstico de preparación en la barra lateral.')
        return
    defense=st.session_state.defence;nonce=st.session_state.input_nonce
    defense_mode=st.toggle('Modo defensa · cinco muestras',value=True,key='defence_mode',disabled=defense.current is not None)
    locked=defense_mode and defense.current is not None
    if defense_mode:
        st.progress(len(defense.records)/5,text=f'Muestra {defense.slot} / 5 · {len(defense.records)} resultados registrados')
        st.caption('Imágenes y datos clínicos permanecen en memoria de esta sesión. El registro no incluye archivos, nombres ni identificadores.')
    st.html('<div class="b-pipeline"><div class="b-step"><b>01</b><strong>Seleccionar y cargar</strong><span>Modalidad y tres fases</span></div>'
        '<div class="b-step"><b>02</b><strong>Validar y visualizar</strong><span>PNG8 · 256 × 256 · sin reescalado</span></div>'
        '<div class="b-step"><b>03</b><strong>Analizar y registrar</strong><span>Inferencia real · umbral explícito</span></div></div>')
    st.subheader('1 · Seleccionar modelo y datos')
    kind=st.radio('Modalidad',['multimodal','image'],format_func=lambda v:lab.LABELS[v],
        key=f'input_kind_{nonce}',horizontal=True,on_change=clear_input,disabled=locked)
    report=analysis_for(catalog) if kind=='multimodal' else None
    choices=['Original 0,50']
    if report is not None:choices.append(f"Recomendado OOF {report['recommended_threshold']:.2f} · exploratorio")
    choices.append('Personalizado · exploratorio')
    policy=st.selectbox('Umbral de clasificación',choices,key=f'input_threshold_policy_{kind}_{nonce}',
                        disabled=locked,on_change=clear_input)
    threshold=.5
    if policy.startswith('Recomendado'):threshold=report['recommended_threshold']
    elif policy.startswith('Personalizado'):
        threshold=st.slider('Umbral pCR personalizado',.05,.95,.5,.01,key=f'input_threshold_{kind}_{nonce}',
                            disabled=locked,on_change=clear_input)
    st.caption(f'Umbral activo: {threshold:.2f} · pCR si probabilidad ≥ umbral. La probabilidad y el AUC no cambian.')
    if threshold!=.5:
        st.warning('Cambio exploratorio de decisión. No se ha validado en test este umbral. '
                   'La recomendación OOF utiliza un modelo excluido por paciente y no el ensemble completo.')
    st.caption('En modo defensa, el umbral se bloquea al registrar la muestra y no se cambia tras revelar su etiqueta. '
               'No debe ajustarse con las etiquetas del profesor.')
    clinical=None;clinical_ok=True
    if kind=='multimodal':
        a,b=st.columns(2)
        age=a.number_input('Edad (años)',min_value=0.,max_value=120.,value=None,step=1.,
            key=f'input_age_{nonce}',disabled=locked,on_change=clear_input)
        subtype=b.selectbox('Subtipo molecular',[None,'HRposHER2neg','TripleNeg','HER2pos','Otro / desconocido'],
            format_func=lambda v:'Seleccionar subtipo' if v is None else v,key=f'input_subtype_{nonce}',
            disabled=locked,on_change=clear_input)
        allow_missing=st.checkbox('Declarar explícitamente datos clínicos ausentes',key=f'input_missing_{nonce}',
            disabled=locked,on_change=clear_input)
        clinical_ok=(age is not None and subtype is not None) or allow_missing
        clinical=dict(age=age,HR_HER2_STATUS='UNKNOWN_REPORTED' if subtype=='Otro / desconocido' else subtype)
        if allow_missing:
            st.info('La edad ausente usa la mediana guardada de cada fold y su indicador de ausencia. El subtipo ausente/desconocido usa su categoría específica; no se presenta como información conocida.')
        elif not clinical_ok:
            st.caption('Completa edad y subtipo o utiliza la modalidad de imágenes. No se imputan silenciosamente datos desconocidos.')
    source=st.radio('Entrada',['Una tripleta PNG','Varios cortes · avanzado','Ejemplo público autorizado'],
        horizontal=True,key=f'input_source_{nonce}',disabled=locked,on_change=clear_input)
    study=None;error=None
    try:
        if source=='Una tripleta PNG':
            uploaded={}
            for column,phase in zip(st.columns(3),PHASES):
                with column:
                    item=st.file_uploader(phase+' · PNG en gris',type=['png'],key=f'upload_{nonce}_{phase}',max_upload_size=1,
                        disabled=locked,on_change=clear_input,help='PNG real de 8 bits, 256 × 256, máximo 1 MiB. Las tres fases deben pertenecer al mismo corte.')
                    uploaded[phase]=(item.name,item.getvalue()) if item else None
            if all(uploaded.values()):study=lab.triplet(uploaded)
            elif any(uploaded.values()):st.info('Faltan fases: '+', '.join(p for p,v in uploaded.items() if v is None))
        elif source=='Varios cortes · avanzado':
            items=st.file_uploader('Tripletas de una sola paciente (PNG o un ZIP)',type=['png','zip'],
                accept_multiple_files=True,key=f'upload_{nonce}_advanced',disabled=locked,on_change=clear_input)
            st.caption('Nombres PACIENTE_z012_PRE.png / EARLY / LATE. Todos los índices z deben tener sus tres fases. Máximo 256 cortes y 128 MiB; no se extrae el ZIP.')
            if items:study=study_from_uploads([(item.name,item.getvalue()) for item in items])
        else:
            study=lab.public_example(st.session_state.bundle_root)
            example=catalog['example']
            st.caption(example['label']+' · '+example['license']+' · '+example['attribution'])
            st.caption('Este ejemplo procede de train y solo demuestra funcionamiento. No incluye etiqueta ni datos clínicos preseleccionados.')
    except (OSError,ValueError,TypeError) as problem:error=str(problem)
    st.session_state.input_study=study
    if error:st.error(error)
    st.subheader('2 · Validar y visualizar')
    if study is not None:
        st.success(f'Muestra válida · {len(study.slice_indices)} corte(s) · PRE, EARLY y LATE sincronizadas')
        position=st.select_slider('Corte a revisar',options=list(range(len(study.slice_indices))),
            format_func=lambda i:f'z{study.slice_indices[i]:03d}',key=f'input_preview_{nonce}') if len(study.slice_indices)>1 else 0
        phases=study.phases[position]
        for col,image,label in zip(st.columns(4),[phases[0],phases[1],phases[2],_signed_rgb((phases[1].astype(float)-phases[0])/255.)],
                                   ['PRE','EARLY','LATE','EARLY − PRE · realce']):
            with col:st.image(image,caption=label,width='stretch',clamp=True)
        if len(study.slice_indices)==1:
            st.warning('Se utiliza un solo corte. Su resultado no equivale directamente al procedimiento por paciente con múltiples cortes del estudio completo.')
    else:st.caption('Carga las tres fases para validar la muestra. No se generan predicciones de demostración.')
    key=defense.input_key(study,kind,clinical)+f':threshold:{threshold:.2f}' if study else None
    result=st.session_state.prediction
    if result is not None and result.get('input_key')!=key:result=None;st.session_state.prediction=None
    st.subheader('3 · Ejecutar inferencia')
    if st.button('Analizar muestra',key='analyze',type='primary',icon=':material/biotech:',width='stretch',
                 disabled=study is None or not clinical_ok or locked or (defense_mode and len(defense.records)>=5)) and not locked:
        try:
            with st.spinner('Cargando modelos verificados y ejecutando inferencia…'):
                bundle=resource(st.session_state.bundle_root,kind,catalog['modalities'][kind]['checksum'])
                result=lab.predict(study,bundle,clinical)
                result=decision(result,threshold)
                result['threshold_policy']=policy
                result['input_key']=key
            if defense_mode:defense.record(result,study,key)
            st.session_state.prediction=result
            if defense_mode:st.rerun()  # Mostrar inmediatamente controles bloqueados y contador actualizado.
        except (OSError,ValueError,RuntimeError,KeyError) as problem:
            st.session_state.prediction=None;result=None
            st.error('No se pudo ejecutar una inferencia válida: '+str(problem))
    if result is not None:
        a,b=st.columns([1,1.5])
        with a:plot(charts.probability_donut(result['patient_probability']),'prediction_probability')
        with b:
            st.subheader('pCR' if result['prediction'] else 'no pCR · enfermedad residual')
            st.write(f"Clasificación al umbral **{result['threshold']:.2f}**. La etiqueta describe respuesta al tratamiento; no presencia o ausencia de cáncer.")
            cards([('Modelos',result['n_models'],result['model']),('Tiempo',f"{result['seconds']:.3f} s",'Inferencia y transformación; sin carga inicial')])
            st.caption(f"Dispositivo: {result['device']} · Cortes: {result['n_slices']} · SHA256: {result['checksum']}")
        st.info('Estimación experimental, sin intervalo de confianza clínico validado. No equivale a supervivencia o curación ni debe cambiar decisiones terapéuticas.')
        if kind=='multimodal' and catalog['external']['metrics']['recall']==0:
            st.warning('Limitación observada: en el test externo guardado, la sensibilidad para pCR fue 0 % con el umbral 0,50. Un AUC superior al azar no garantiza detectar pCR con este umbral.')
        with st.expander('Trazabilidad de la predicción'):
            values=result['member_probabilities']
            st.caption(f'Rango entre los cinco miembros: {min(values):.1%}–{max(values):.1%}. Es dispersión descriptiva, no un intervalo de confianza clínico.')
            st.dataframe([dict(fold=m['fold'],epoca=m['epoch'],parametros=m['parameters'],
                sha256_origen=m['original_sha256'],probabilidad=result['member_probabilities'][m['fold']])
                for m in catalog['modalities'][kind]['members']],hide_index=True,width='stretch')
            plot(charts.slice_probabilities(dict(slices=result['slices'],patient_probability=result['patient_probability'],threshold=result['threshold'])),
                 'slice_probabilities')
        if defense_mode:
            label=st.selectbox('Etiqueta real · solo si el profesor ya la ha revelado',[None,0,1],
                format_func=lambda v:'Aún no revelada' if v is None else ('0 · no pCR' if v==0 else '1 · pCR'),key=f'reveal_{nonce}')
            if st.button('Registrar etiqueta revelada',disabled=label is None,key='record_label'):
                try:defense.reveal(label);st.success('Etiqueta registrada sin modificar el modelo ni el umbral.')
                except ValueError as problem:st.error(str(problem))
            st.button('Siguiente muestra',key='next_sample',type='primary',on_click=next_sample,
                disabled=defense.slot>=5)
            if len(defense.records)==5:st.success('Las cinco muestras están registradas. Descarga el CSV antes de cerrar la sesión.')
    if defense.records:
        st.subheader('Registro de la defensa · esta sesión')
        st.dataframe(defense.records,hide_index=True,width='stretch')
        st.download_button('Descargar resultados de la defensa (CSV)',defense.csv(),file_name='defensa_cinco_muestras.csv',
            mime='text/csv',key='download_defence')
        st.caption('Cinco muestras prueban el funcionamiento; no estiman de forma fiable el rendimiento clínico. Las etiquetas reveladas nunca se utilizan para ajustar parámetros.')
