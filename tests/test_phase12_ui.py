"""Regresión de las cinco páginas F12 y del flujo final de defensa, sin entrenamiento."""
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from phase12_fixtures import study_files,png_bytes

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT/'models/app_bundle/catalog.json').is_file(),'Requiere el paquete de modelos reales')
class Phase12UITests(unittest.TestCase):
    def app(self):return AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
    def clean(self,app):self.assertFalse(app.exception,[x.value for x in app.exception])

    def upload(self,app,slot):
        for phase,(name,data) in zip(['PRE','EARLY','LATE'],study_files(1,patient=f'PRIVATE_CASE_{slot}')):
            if phase=='PRE':
                array=np.array(Image.open(BytesIO(data)));array[0,0]=slot;data=png_bytes(array)
            app.file_uploader(key=f'upload_{slot}_{phase}').set_value((name,data,'image/png')).run()
        self.clean(app)

    def test_five_pages_render_without_errors_and_no_path_entry_widget(self):
        app=self.app()
        for name in ['home','predict','explorer','method','results']:
            app.button(key='nav_'+name).click().run();self.clean(app)
            self.assertFalse(any(w.key=='direct_campaign' for w in app.text_input))
        self.assertTrue(any('cero' in item.value.lower() for item in app.error))

    def test_complete_five_sample_defence_clears_inputs_and_exports_only_safe_fields(self):
        app=self.app();app.button(key='nav_predict').click().run()
        self.assertFalse(app.selectbox(key='input_subtype_0').value)
        self.assertTrue(app.button(key='analyze').disabled)
        for slot in range(5):
            app.radio(key=f'input_kind_{slot}').set_value('image').run()
            self.upload(app,slot)
            self.assertFalse(app.button(key='analyze').disabled)
            app.button(key='analyze').click().run();self.clean(app)
            self.assertEqual(len(app.session_state.defence.records),slot+1)
            self.assertEqual(app.session_state.defence.records[-1]['etiqueta_real'],None)
            app.selectbox(key=f'reveal_{slot}').select(1 if slot%2 else 0).run()
            app.button(key='record_label').click().run();self.clean(app)
            self.assertEqual(app.session_state.defence.records[-1]['etiqueta_real'],slot%2)
            self.assertTrue(app.button(key='analyze').disabled)
            if slot<4:
                app.button(key='next_sample').click().run();self.clean(app)
                self.assertIsNone(app.session_state.prediction)
                self.assertIsNone(app.session_state.input_study)
                self.assertTrue(app.button(key='analyze').disabled)
        self.assertTrue(app.button(key='next_sample').disabled)
        exported=app.session_state.defence.csv()
        frame=pd.read_csv(BytesIO(exported))
        self.assertEqual(len(frame),5);self.assertNotIn(b'PRIVATE_CASE',exported)
        self.assertNotIn('patient_id',frame);self.assertNotIn('age',frame)
        self.assertTrue(app.get('download_button'))

    def test_corrupt_upload_cannot_leave_old_study_or_probability(self):
        app=self.app();app.button(key='nav_predict').click().run()
        app.radio(key='input_kind_0').set_value('image').run();self.upload(app,0)
        app.file_uploader(key='upload_0_PRE').set_value(('bad.png',b'broken','image/png')).run()
        self.clean(app);self.assertTrue(app.error)
        self.assertIsNone(app.session_state.input_study);self.assertTrue(app.button(key='analyze').disabled)

    def test_clinical_missing_requires_explicit_choice_and_unknown_is_supported(self):
        app=self.app();app.button(key='nav_predict').click().run()
        app.radio(key='input_source_0').set_value('Ejemplo público autorizado').run()
        self.assertTrue(app.button(key='analyze').disabled)
        app.checkbox(key='input_missing_0').set_value(True).run()
        self.assertFalse(app.button(key='analyze').disabled)
        app.button(key='analyze').click().run();self.clean(app)
        self.assertEqual(app.session_state.prediction['n_models'],5)

    def test_single_cut_surface_and_multi_cut_synthetic_are_explicitly_distinguished(self):
        app=self.app();app.button(key='nav_explorer').click().run()
        app.toggle(key='enable_3d').set_value(True).run();self.clean(app)
        self.assertTrue(any('surface' in str(p.proto).lower() for p in app.get('plotly_chart')))
        app.toggle(key='demo_mode').set_value(True).run();self.clean(app)
        self.assertTrue(any('SINTÉTICOS' in x.value for x in app.warning))
        app.slider(key='slice_pos').set_value(7).run();self.clean(app)
        self.assertIsNone(app.session_state.prediction)

    def test_missing_artifacts_give_no_fake_prediction(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'BREASTDCEDL_BUNDLE_DIR':tmp}):
            app=self.app();app.button(key='nav_predict').click().run();self.clean(app)
            self.assertIsNone(app.session_state.catalog);self.assertFalse(app.get('plotly_chart'))
            self.assertIsNone(app.session_state.prediction)


if __name__=='__main__':unittest.main()
