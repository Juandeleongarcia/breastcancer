"""Aceptación final: gráficos, defensa, procedencia y limpieza; sin entrenamiento."""
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from zipfile import ZipFile
import hashlib

import numpy as np
import pandas as pd
import torch
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.errors import AppTestError

from app_components import charts
from app_components.methodology import describe
from app_services import lab
from app_services.results import aggregate_histories
from app_services.thresholds import analyze
from app_services.studies import MAX_BYTES

ROOT=Path(__file__).resolve().parents[1]


class DeliveryChartTests(unittest.TestCase):
    def test_five_fold_points_explicit_categories_mean_and_threshold(self):
        values=[.14,.21,.28,.32,.37]
        result=dict(member_probabilities=values,patient_probability=float(np.mean(values)),threshold=.31)
        fig=charts.fold_probabilities(result)
        self.assertEqual(list(fig.data[0].x),[f'Fold {i}' for i in range(5)])
        np.testing.assert_array_equal(fig.data[0].y,values)
        np.testing.assert_allclose(fig.data[1].y,result['patient_probability'])
        np.testing.assert_allclose(fig.data[2].y,.31)
        self.assertEqual(tuple(fig.layout.yaxis.range),(0,1))
        with self.assertRaises(ValueError):charts.fold_probabilities({**result,'patient_probability':.9})

    def test_threshold_metrics_and_youden_have_separate_valid_scales(self):
        report=analyze([0,0,1,1],[.8,.9,.1,.2])
        proportions=charts.threshold_curves(report,.31)
        youden=charts.threshold_curves(report,.31,youden=True)
        self.assertEqual(tuple(proportions.layout.yaxis.range),(0,1))
        self.assertEqual(tuple(youden.layout.yaxis.range),(-1,1))
        self.assertTrue(min(youden.data[0].y)<0)
        self.assertEqual(len(proportions.data[0].y),91)
        self.assertEqual(list(proportions.data[0].y),[r['sensitivity'] for r in report['thresholds']])
        self.assertEqual(list(youden.data[0].y),[r['youden'] for r in report['thresholds']])
        self.assertLess(proportions.layout.legend.y,0)

    def test_confusion_cells_counts_orientation_and_categorical_colors(self):
        matrix=[[122,1],[53,0]]  # Verifica la visualización; no ejecuta evaluación externa.
        fig=charts.confusion_figure(dict(confusion_matrix=matrix))
        np.testing.assert_array_equal(fig.data[0].customdata,matrix)
        self.assertEqual(list(fig.data[0].text[0]),['TN · 122','FP · 1'])
        self.assertEqual(list(fig.data[0].text[1]),['FN · 53','TP · 0'])
        self.assertEqual(fig.layout.yaxis.autorange,'reversed')
        self.assertEqual(fig.data[0].z[0][0],fig.data[0].z[1][1])
        self.assertNotEqual(fig.data[0].z[0][0],fig.data[0].z[0][1])

    def test_loss_full_scale_never_crops_values_and_patient_metrics_zero_one(self):
        data=pd.DataFrame(dict(epoch=[1,2],train_eval_loss=[.6,.4],val_loss=[.65,.7],
            train_eval_patient_auc=[.6,.8],val_patient_auc=[.6,.7]))
        self.assertEqual(tuple(charts.history_curve(data,'loss')[0].layout.yaxis.range),(0,1))
        data.loc[1,'val_loss']=1.8
        self.assertGreater(charts.history_curve(data,'loss')[0].layout.yaxis.range[1],1.8)
        self.assertEqual(tuple(charts.history_curve(data,'auc')[0].layout.yaxis.range),(0,1))

    def test_mean_auc_uses_existing_epochs_only_with_explicit_counts(self):
        a=pd.DataFrame(dict(epoch=[1,2],train_eval_patient_auc=[.5,.8],val_patient_auc=[.6,.7]))
        b=a.iloc[:1].copy()
        table=aggregate_histories({0:a,1:b},train_column='train_eval_patient_auc',val_column='val_patient_auc')
        self.assertEqual(table.n_folds.tolist(),[2,1]);self.assertTrue(np.isnan(table.val_std.iloc[1]))
        self.assertEqual(len(charts.mean_loss_curve(table,metric='auc').data[-1].x),2)

    def test_failed_download_removes_only_its_new_temporary_directory(self):
        content=BytesIO()
        with ZipFile(content,'w') as archive:archive.writestr('../escape.txt','invalid')
        raw=content.getvalue();response=MagicMock()
        response.__enter__.return_value=response;response.iter_content.return_value=[raw]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir()
            (root/'config/app_release.json').write_text(json.dumps(dict(local_directory='models/absent',
                url='https://github.com/owner/repo/releases/download/v1/package.zip',sha256=hashlib.sha256(raw).hexdigest())))
            with patch.object(lab,'ROOT',root),patch('app_services.lab.requests.get',return_value=response),patch.dict(os.environ):
                os.environ.pop('BREASTDCEDL_BUNDLE_DIR',None)
                with self.assertRaises(ValueError):lab.package_root()
            self.assertEqual(list((root/'models/.app_cache').iterdir()),[])
            self.assertFalse((root/'escape.txt').exists())


@unittest.skipUnless((ROOT/'config/app_release.json').exists(),'Requiere el contrato del paquete de inferencia')
class DeliveryRealTests(unittest.TestCase):
    def app(self):return AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()

    def test_public_button_real_prediction_reset_confirmation_and_unique_ids(self):
        app=self.app();app.button(key='nav_predict').click().run()
        app.button(key='public_example').click().run()
        self.assertEqual(app.radio(key='input_kind_0').value,'image')
        self.assertFalse(app.session_state.input_study.synthetic)
        self.assertIsNone(app.session_state.prediction)
        options=app.selectbox(key='input_threshold_policy_image_0').options
        self.assertFalse(any('Recomendado' in option for option in options))
        app.button(key='analyze').click().run()
        self.assertFalse(app.exception,[e.value for e in app.exception])
        result=app.session_state.prediction
        self.assertEqual(result['patient_probability'],float(np.mean(result['member_probabilities'])))
        old=app.session_state.defence.session_id;row=app.session_state.defence.records[0]
        self.assertEqual(row['sesion_id'],old);self.assertEqual(len(row['ejecucion_id']),32)
        self.assertTrue(app.button(key='reset_defence').disabled)
        app.checkbox(key='confirm_reset_defence').check().run()
        app.button(key='reset_defence').click().run()
        self.assertFalse(app.exception,[e.value for e in app.exception])
        self.assertNotEqual(app.session_state.defence.session_id,old)
        self.assertFalse(app.session_state.defence.records)
        self.assertIsNone(app.session_state.prediction);self.assertIsNone(app.session_state.input_study)
        self.assertTrue(app.button(key='analyze').disabled)

    def test_real_architecture_shapes_operations_counts_and_bn_unchanged(self):
        for kind,count,clinical in [('image',72593,0),('multimodal',72600,7)]:
            resource=lab.load_resource(lab.read_catalog()[0],kind)
            model=resource.members[0]['model']
            before={key:value.clone() for key,value in model.state_dict().items()}
            rng=torch.get_rng_state().clone();description=describe(resource)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(value,model.state_dict()[key]) for key,value in before.items()))
            self.assertFalse(model.training);self.assertEqual(description['parameters'],count)
            self.assertEqual(description['clinical_dim'],clinical);self.assertEqual(description['dropout'],.4)
            self.assertEqual([row[1] for row in description['stages'][1:4]],['16 × 128 × 128','32 × 64 × 64','64 × 32 × 32'])
            self.assertEqual(sum(isinstance(layer,torch.nn.Conv2d) for layer in model.features),6)

    def test_two_sessions_isolated_and_both_modalities_render_with_author_identity(self):
        a=self.app();b=self.app()
        self.assertNotEqual(a.session_state.defence.session_id,b.session_state.defence.session_id)
        self.assertTrue(any('Universidad Alfonso X' in item.value for item in a.caption))
        a.button(key='nav_method').click().run()
        for kind in ['image','multimodal']:
            a.radio(key='architecture_kind').set_value(kind).run()
            self.assertFalse(a.exception,[e.value for e in a.exception])
        self.assertFalse(b.session_state.defence.records)

    def test_multimodal_inference_with_explicit_clinical_and_exploratory_threshold(self):
        app=self.app();app.button(key='nav_predict').click().run()
        app.button(key='public_example').click().run()
        app.radio(key='input_kind_0').set_value('multimodal').run()
        with patch('app_components.prediction.lab.predict',side_effect=AssertionError('No imputar sin declaración')) as forward:
            with self.assertRaises(AppTestError):app.button(key='analyze').click()
            app.run()
            forward.assert_not_called()
        self.assertIsNone(app.session_state.prediction)
        # Entradas explícitas de prueba funcional, sin atribuirlas a la paciente del ejemplo.
        app.number_input(key='input_age_0').set_value(45.).run()
        app.selectbox(key='input_subtype_0').select('HRposHER2neg').run()
        widget=app.selectbox(key='input_threshold_policy_multimodal_0')
        widget.select(widget.options[1]).run();app.button(key='analyze').click().run()
        self.assertFalse(app.exception,[e.value for e in app.exception])
        result=app.session_state.prediction
        self.assertEqual(result['n_models'],5);self.assertEqual(result['threshold'],.31)
        self.assertEqual(result['patient_probability'],float(np.mean(result['member_probabilities'])))
        self.assertTrue(app.selectbox(key='input_threshold_policy_multimodal_0').disabled)

    def test_oversize_multi_upload_rejected_before_copying_content(self):
        app=self.app();app.button(key='nav_predict').click().run()
        app.radio(key='input_kind_0').set_value('image').run()
        app.radio(key='input_source_0').set_value('Varios cortes · avanzado').run()
        item=MagicMock();item.name='too_big.zip';item.size=MAX_BYTES+1
        item.getvalue.side_effect=AssertionError('No copiar un upload que supera el límite')
        with patch('app_components.prediction.st.file_uploader',return_value=[item]):app.run()
        self.assertFalse(app.exception,[e.value for e in app.exception])
        self.assertTrue(any('128 MiB' in e.value for e in app.error))
        item.getvalue.assert_not_called();self.assertTrue(app.button(key='analyze').disabled)


if __name__=='__main__':unittest.main()
