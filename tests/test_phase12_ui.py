"""AppTest de la base visual, sin selección, comparación ni integración de modelos."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from streamlit.proto.TextInput_pb2 import TextInput

from phase12_fixtures import study_files,campaign_fixture
from app_services.workspace import PENDING_ITEMS

FORBIDDEN = ('SPATIAL_SMALL','F11_CNN4','72593','73913','72.593','73.913','72,593','73,913',
             'Comparación de arquitecturas','Comparar candidatos')


class Phase12UITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.settings=self.root/'app.json'
        self.settings.write_text(json.dumps({'results':{'campaign':''}}))
        self.environment=patch.dict(os.environ,{'BREASTDCEDL_APP_CONFIG':str(self.settings)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop();self.temp.cleanup()

    def app(self):
        return AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=90).run()

    def assert_generic(self,app):
        self.assertFalse(app.exception,[e.value for e in app.exception])
        visible=[]
        for kind in ['title','header','subheader','caption','markdown','info','warning','error',
                     'success','text','html','plotly_chart']:
            visible.extend(str(element.proto) for element in app.get(kind))
        for frame in app.dataframe:visible.append(frame.value.to_string())
        text='\n'.join(visible)
        for forbidden in FORBIDDEN:self.assertNotIn(forbidden,text)
        keys=[element.key for element in app.selectbox]
        self.assertFalse(set(keys)&{'selected_model','architecture_preview','architecture_stage'})
        self.assertFalse(app.get('json'))

    def upload(self,app,n=3):
        app.file_uploader[0].set_value([(name,content,'image/png') for name,content in study_files(n)]).run()
        return app

    def test_five_generic_pages_and_six_explicit_pending_items(self):
        app=self.app();self.assert_generic(app)
        for label in PENDING_ITEMS:
            self.assertTrue(any(label in item.value for item in app.caption),label)
        for page in ['explorer','predict','results','method','home']:
            app.button(key='nav_'+page).click().run();self.assert_generic(app)
            if page=='predict':
                self.assertTrue(app.button(key='analyze').disabled)
                self.assertTrue(any('Modelo pendiente de integrar' in info.value for info in app.info))
        self.assertIsNone(app.session_state.prediction)
        self.assertIsNone(app.session_state.study)

    def test_synthetic_explorer_3d_and_synchronized_selection_never_predict(self):
        app=self.app();app.button(key='nav_explorer').click().run()
        app.slider(key='slice_pos').set_value(7).run()
        app.toggle(key='enable_3d').set_value(True).run();self.assert_generic(app)
        self.assertEqual(app.session_state.slice_pos,7)
        self.assertTrue(any('SINTÉTICOS' in item.value for item in app.warning))
        self.assertGreaterEqual(len(app.get('plotly_chart')),1)
        self.assertIsNone(app.session_state.study)
        self.assertIsNone(app.session_state.prediction)
        app.button(key='nav_predict').click().run()
        self.assertTrue(app.button(key='analyze').disabled)

    def test_upload_validation_brightness_and_clear_leave_analysis_disabled(self):
        app=self.app();app.button(key='nav_predict').click().run()
        one=study_files(n=1)[0]
        app.file_uploader[0].set_value([(one[0],one[1],'image/png')]).run()
        self.assertTrue(app.error);self.assertIsNone(app.session_state.study)
        self.upload(app,3);self.assert_generic(app)
        fingerprint=app.session_state.study.fingerprint
        app.slider(key='brightness').set_value(.2).run()
        app.slider(key='contrast').set_value(1.5).run()
        self.assertEqual(app.session_state.study.fingerprint,fingerprint)
        self.assertTrue(app.button(key='analyze').disabled)
        app.file_uploader[0].set_value([(one[0],one[1],'image/png')]).run()
        self.assertIsNone(app.session_state.study)
        self.upload(app,3);self.assertEqual(app.session_state.study.fingerprint,fingerprint)
        app.file_uploader[0].clear().run()
        self.assertIsNone(app.session_state.study)
        self.assertTrue(app.button(key='analyze').disabled)

    def test_legacy_model_settings_cannot_activate_or_load_weights(self):
        self.settings.write_text(json.dumps({
            'inference':{'enabled':True,'architecture':'F11_CNN4','checkpoint':'DO_NOT_READ.pt'},
            'models':{name:{'checkpoint':'LEGACY_DO_NOT_READ.pt'} for name in ['SPATIAL_SMALL','F11_CNN4']}}))
        with patch('torch.load',side_effect=AssertionError('No cargar checkpoints')) as loader:
            app=self.app()
            for page in ['predict','explorer','results','method']:
                app.button(key='nav_'+page).click().run();self.assert_generic(app)
            app.button(key='nav_predict').click().run();self.upload(app)
            self.assertTrue(app.button(key='analyze').disabled)
            self.assertIsNone(app.session_state.prediction)
            loader.assert_not_called()

    def test_one_real_campaign_at_a_time_with_no_model_identity_in_interface(self):
        for index,kind in enumerate(['SPATIAL_SMALL','F11_CNN4']):
            with self.subTest(source=index):
                campaign=campaign_fixture(self.root/str(index),kind=kind)
                app=self.app();app.button(key='nav_results').click().run()
                app.text_input(key='direct_campaign').set_value(str(campaign)).run()
                self.assert_generic(app)
                self.assertFalse(app.warning,[w.value for w in app.warning])
                self.assertGreaterEqual(len(app.get('plotly_chart')),7)
                self.assertEqual(app.text_input(key='direct_campaign').proto.type,TextInput.PASSWORD)
                app.radio(key='mean_mode').set_value('Solo épocas comunes a los cinco folds').run()
                app.selectbox(key='validation_fold').select_index(3).run();self.assert_generic(app)

    def test_partial_results_remain_provisional_without_fabricated_epochs(self):
        campaign=campaign_fixture(self.root,completed=2)
        self.settings.write_text(json.dumps({'results':{'campaign':str(campaign)}}))
        app=self.app();app.button(key='nav_results').click().run();self.assert_generic(app)
        self.assertTrue(any('PROVISIONAL' in info.value for info in app.info))
        app.radio(key='mean_mode').set_value('Solo épocas comunes a los cinco folds').run();self.assert_generic(app)

    def test_conceptual_cnn_explainer_never_exposes_specific_shapes_or_counts(self):
        app=self.app();app.button(key='nav_method').click().run();self.assert_generic(app)
        app.selectbox(key='cnn_concept').select_index(3).run();self.assert_generic(app)
        self.assertEqual(len(app.get('plotly_chart')),1)
        chart=str(app.get('plotly_chart')[0].proto)
        self.assertNotIn('32 × 32',chart)
        self.assertNotIn('parámetros',chart)

    def test_malformed_settings_keep_empty_app_and_navigation_available(self):
        self.settings.write_text('[]')
        app=self.app();self.assert_generic(app);self.assertTrue(app.warning)
        app.button(key='nav_predict').click().run();self.assert_generic(app)
        self.assertTrue(app.button(key='analyze').disabled)


if __name__=='__main__':unittest.main()
