import numpy as np
import pandas as pd
import streamlit as st

from app_components.layout import section, cards, empty, plot
from app_components import charts
from app_services.results import read_campaign, aggregate_histories


def _format(value, digits=4):
    return '—' if value is None or not np.isfinite(value) else f'{value:.{digits}f}'


def _read(campaign):
    if not campaign:return None,None
    try:return read_campaign(campaign),None
    except (OSError,ValueError,KeyError,TypeError) as error:return None,str(error)


def _public_issue(message):
    text = str(message).lower()
    if 'pacient' in text or 'predic' in text:return 'Las predicciones no cumplen la cobertura requerida por paciente.'
    if any(word in text for word in ['oficial','partición','csv','protocolo']):return 'No se ha podido verificar el protocolo y las particiones oficiales.'
    if 'época' in text or 'historial' in text:return 'El historial y el resumen requieren revisión.'
    return 'Faltan archivos requeridos o los resultados no superan la verificación de integridad.'


def render(campaign_path=''):
    section('EVIDENCIA EXPERIMENTAL','Resultados y validación cruzada','Resultados reales, métricas por paciente y cobertura explícita de los cinco folds oficiales.')
    with st.expander('Cargar resultados reales',expanded=not bool(campaign_path)):
        direct = st.text_input('Carpeta de resultados CV',value=campaign_path,key='direct_campaign',type='password',
                           help='Ruta local de una campaña compatible. Se oculta en pantalla durante la demostración.')
    st.button('Actualizar resultados',icon=':material/refresh:',key='refresh_results')
    report,error = _read(direct.strip())
    if report:
        st.caption('Datos de validación aportados · lectura local y de solo lectura. No se presentan como resultados del test ni como selección del modelo definitivo.')
    metrics = report.metrics if report else {}
    complete = report is not None and report.complete
    qualifier = '5 folds verificados' if complete else 'Provisional · solo folds verificados' if report else 'Pendiente de integrar'
    cards([('ROC AUC medio',_format(metrics.get('auc_mean')),qualifier),
           ('DE del AUC',_format(metrics.get('auc_std')),'Desviación estándar muestral'),
           ('ROC AUC OOF',_format(metrics.get('oof_auc')),'Solo con cobertura completa'),
           ('Folds completos',f"{metrics.get('complete_folds',0)} / 5",'Integrados y verificados en esta app')])
    st.write('')
    cards([('Accuracy',_format(metrics.get('accuracy')),'Media por paciente entre folds'),
           ('Balanced accuracy',_format(metrics.get('balanced_accuracy')),'Equilibrio entre ambas clases'),
           ('Pacientes evaluados',str(metrics['patients']) if metrics.get('patients') else '—','Pacientes de folds verificados')])
    if report is None:
        if error:st.warning(_public_issue(error))
        empty('Los resultados están por llegar',
              'La estructura del dashboard está preparada para los archivos reales de los cinco folds. Las métricas definitivas y los resultados del test siguen pendientes.')
    else:
        if not complete:
            st.info('VALIDACIÓN PROVISIONAL · No se presenta como CV final hasta verificar los cinco folds. No se calcula una matriz global ni AUC OOF parcial.')
        if report.issue:st.warning(_public_issue(report.issue))
        for fold in report.folds:
            if fold.issue:st.warning(f'Fold {fold.fold}: '+_public_issue(fold.issue))
        summary,individual,means = st.tabs(['Resumen de CV','Detalle por fold','Curvas medias'])
        with summary:
            table = pd.DataFrame([{'Fold':f.fold,'Estado':f.status,'AUC paciente':f.metrics.get('auc') if f.complete else None,
                'Mejor época':f.summary.get('best_epoch'),'Pacientes':len(f.predictions) if f.complete else None,
                'Accuracy':f.metrics.get('accuracy') if f.complete else None,
                'Balanced accuracy':f.metrics.get('balanced_accuracy') if f.complete else None} for f in report.folds])
            st.dataframe(table,hide_index=True,width='stretch')
            a,b = st.columns(2)
            with a:
                st.subheader('Discriminación entre folds')
                plot(charts.fold_auc_figure(report),'auc_by_fold')
                st.caption('La variación entre folds refleja sensibilidad a la partición. La DE usa ddof=1; con un único fold no se define.')
            with b:
                st.subheader('Curvas ROC por paciente')
                if report.completed:plot(charts.roc_figure(report.completed,report.oof),'roc_campaign')
                else:empty('ROC pendiente','Se necesitan predicciones por paciente válidas de un fold completo.')
                st.caption('Cada línea procede de probabilidades guardadas. La OOF agrupa cinco folds completos; no es una media interpolada de curvas ROC.')
            if complete:
                st.subheader('Matriz global OOF · paciente')
                a,b = st.columns([1.3,1])
                with a:plot(charts.confusion_figure(report.oof_metrics),'oof_confusion')
                with b:
                    st.write('Umbral guardado:',report.completed[0].config['CLASSIFICATION_THRESHOLD'])
                    st.write('Sensibilidad:',_format(report.oof_metrics['recall']))
                    st.write('Especificidad:',_format(report.oof_metrics['specificity']))
                    st.write('F1:',_format(report.oof_metrics['f1']))
                    st.caption('Una predicción por paciente, procedente únicamente del fold donde fue excluido del entrenamiento.')
            st.download_button('Descargar tabla de folds',table.to_csv(index=False).encode(),
                               file_name='validacion_folds.csv',mime='text/csv',key='download_folds')
        with individual:
            fold_index = st.selectbox('Fold a revisar',list(range(5)),format_func=lambda n:f'Fold {n} · {report.folds[n].status}',key='validation_fold')
            fold = report.folds[fold_index]
            if fold.history.empty:
                empty('Este fold aún no tiene historial','Los datos aparecerán cuando su RUN esté disponible. No se completan épocas inexistentes.')
            else:
                st.caption(f"Historial real · {len(fold.history)} épocas · mejor época declarada: {fold.summary.get('best_epoch','pendiente')}")
                figure,comparable = charts.history_curve(fold.history,'loss')
                if fold.summary.get('best_epoch'):figure.add_vline(x=fold.summary['best_epoch'],line_dash='dot',line_color=charts.MUTED)
                st.subheader('Evolución de las pérdidas')
                plot(figure,'individual_losses')
                if not comparable:st.warning('Train loss online: pesos cambiantes. No equivale a train_eval_loss; se identifica con línea discontinua.')
                else:st.caption('Train eval y validación: pesos de final de época, sin augmentation. Se muestra toda la evolución, sin suavizar.')
                a,b = st.columns(2)
                with a:
                    figure,_ = charts.history_curve(fold.history,'accuracy');plot(figure,'individual_accuracy')
                with b:
                    figure,_ = charts.history_curve(fold.history,'auc');plot(figure,'individual_auc')
                if 'learning_rate' in fold.history:
                    st.subheader('Learning rate utilizado')
                    plot(charts.lr_figure(fold.history),'individual_lr')
                if fold.metrics:
                    a,b = st.columns(2)
                    with a:plot(charts.roc_figure([fold]),'individual_roc')
                    with b:plot(charts.confusion_figure(fold.metrics),'individual_confusion')
                    st.caption('Matriz por paciente, umbral '+str(fold.config['CLASSIFICATION_THRESHOLD'])+'.')
                with st.expander('Datos numéricos del historial'):
                    columns = [name for name in fold.history if name=='epoch' or name in ['learning_rate','next_learning_rate']
                               or name.endswith(('_loss','_auc','_accuracy','_recall','_specificity','_f1'))]
                    st.dataframe(fold.history[columns],hide_index=True,width='stretch')
        with means:
            histories = {f.fold:f.history for f in report.completed if not f.history.empty}
            st.caption('Se agregan únicamente folds completados y verificados. Los historiales en ejecución pueden revisarse en Detalle por fold.')
            label = st.radio('Cobertura por época',['Todas las épocas disponibles','Solo épocas comunes a los cinco folds'],key='mean_mode',horizontal=True)
            mode = 'common' if label.startswith('Solo') else 'all'
            use_eval = any('train_eval_loss' in frame for frame in histories.values())
            column = 'train_eval_loss' if use_eval else 'train_loss'
            if not use_eval:st.warning('No hay train_eval_loss: la curva de train es online, con pesos cambiantes, y no equivale a evaluación de final de época.')
            elif any(column not in frame for frame in histories.values()):
                st.info('Los folds sin train_eval_loss no contribuyen a esta curva comparable; su número se muestra por época.')
            aggregated = aggregate_histories(histories,mode,column)
            if aggregated.empty:
                empty('Aún no hay cobertura para esta vista','La vista común requiere datos de los cinco folds en la misma época. Puedes elegir todas las épocas disponibles.')
            else:
                plot(charts.mean_loss_curve(aggregated,'Train eval' if use_eval else 'Train online'),'mean_losses')
                st.caption('Media y ±1 DE muestral. Con un único fold no se dibuja banda. No hay relleno con ceros, repetición de valores ni interpolación de épocas.')
                st.dataframe(aggregated,hide_index=True,width='stretch')
                st.download_button('Descargar medias por época',aggregated.to_csv(index=False).encode(),file_name='perdidas_medias.csv',mime='text/csv')
