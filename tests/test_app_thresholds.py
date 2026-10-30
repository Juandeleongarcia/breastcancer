"""Selección OOF, métricas, procedencia y demostración sin tocar test ni entrenar."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score

from app_services.thresholds import GRID, analyze, build, decision, load_analysis, digest
from app_services.workspace import ROOT


class ThresholdTests(unittest.TestCase):
    def test_metrics_match_independent_sklearn_and_auc_does_not_change(self):
        y=np.array([0,0,1,1,1,0]);p=np.array([.1,.4,.3,.7,.8,.2])
        report=analyze(y,p)
        self.assertEqual(len(report['thresholds']),91)
        for row in report['thresholds']:
            pred=p>=row['threshold'];tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
            self.assertEqual(row['confusion_matrix'],[[tn,fp],[fn,tp]])
            self.assertEqual(row['sensitivity'],recall_score(y,pred,zero_division=0))
            self.assertEqual(row['specificity'],tn/(tn+fp))
            self.assertEqual(row['precision'],precision_score(y,pred,zero_division=0))
            self.assertEqual(row['f1'],f1_score(y,pred,zero_division=0))
            self.assertEqual(row['balanced_accuracy'],balanced_accuracy_score(y,pred))
            self.assertEqual(row['auc'],report['original']['auc'])

    def test_youden_grid_tie_break_and_permutation_reproducibility(self):
        y=[0,0,1,1];p=[.1,.2,.7,.8]
        a=analyze(y,p);b=analyze(y[::-1],p[::-1])
        self.assertEqual(a,b)
        self.assertEqual(a['recommended_threshold'],.5)  # Perfect scores plateau includes .50.
        self.assertEqual(a['recommended']['youden'],1)
        self.assertEqual(tuple(r['threshold'] for r in a['thresholds']),GRID)

    def test_invalid_values_and_degenerate_class_rejected(self):
        for y,p in [([0,0],[.1,.2]),([0,1],[np.nan,.2]),([0,1],[-.1,.2]),
                    ([0,1],[1.1,.2]),([0,1],[.1]),([[0,1]],[[.1,.2]])]:
            with self.assertRaises(ValueError):analyze(y,p)

    def test_threshold_only_changes_decision_and_never_mutates_probabilities(self):
        old=dict(patient_probability=.4,prediction=0,threshold=.5,member_probabilities=[.4]*5)
        new=decision(old,.31)
        self.assertEqual(old['threshold'],.5);self.assertEqual(old['prediction'],0)
        self.assertEqual(new['prediction'],1);self.assertEqual(new['threshold'],.31)
        self.assertEqual(new['patient_probability'],old['patient_probability'])
        self.assertEqual(decision(dict(patient_probability=.31),.31)['prediction'],1)
        for value in [0,1,float('nan')]:
            with self.assertRaises(ValueError):decision(old,value)

    def test_report_schema_checksum_guard_and_missing_file(self):
        report=analyze([0,1],[.1,.8]);report['model_checksum']='expected'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'analysis.json'
            self.assertIsNone(load_analysis(path))
            path.write_text(json.dumps(report),encoding='utf-8')
            self.assertEqual(load_analysis(path,'expected'),report)
            with self.assertRaises(ValueError):load_analysis(path,'different')
            report['methodology']['external_test_used']=True
            path.write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaises(ValueError):load_analysis(path)


@unittest.skipUnless((ROOT/'config/app_oof_threshold.json').exists() and (ROOT/'models/phase13').exists(),
                    'Requiere campaña real local para verificar procedencia')
class RealOOFTests(unittest.TestCase):
    def test_real_report_reproduces_exactly_without_reading_external_or_loading_models(self):
        saved=json.loads((ROOT/'config/app_oof_threshold.json').read_text(encoding='utf-8'))
        campaign=ROOT/'models/phase13'/saved['source']['campaign_id']
        original=pd.read_csv
        def guarded(path,*args,**kwargs):
            self.assertNotIn('external_test',str(path));self.assertNotIn('fold_-1',str(path))
            return original(path,*args,**kwargs)
        before=digest(campaign/'campaign.json')
        with patch('pandas.read_csv',side_effect=guarded),patch('torch.load',side_effect=AssertionError('No models needed')):
            rebuilt=build(campaign)
        self.assertEqual(rebuilt,saved);self.assertEqual(rebuilt['n'],1097)
        self.assertEqual(before,digest(campaign/'campaign.json'))
        self.assertFalse(rebuilt['methodology']['external_test_used'])
        # Public report contains aggregates/provenance only, never patient rows or IDs.
        self.assertNotIn('patient_id',json.dumps(saved))

    def test_ui_selects_oof_threshold_records_it_and_locks_after_prediction(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.button(key='nav_predict').click().run()
        key='input_threshold_policy_multimodal_0'
        options=app.selectbox(key=key).options
        app.selectbox(key=key).select(options[1]).run()
        app.checkbox(key='input_missing_0').check().run()
        app.radio(key='input_source_0').set_value('Ejemplo público autorizado').run()
        app.button(key='analyze').click().run()
        self.assertFalse(app.exception,[e.value for e in app.exception])
        row=app.session_state['defence'].records[0]
        expected=load_analysis()['recommended_threshold']
        self.assertEqual(row['umbral'],expected)
        self.assertEqual(row['clase_predicha'],int(row['probabilidad_pcr']>=expected))
        self.assertTrue(app.selectbox(key=key).disabled)
        self.assertIn(str(expected),app.session_state['defence'].csv().decode('utf-8-sig'))
        app.button(key='next_sample').click().run()
        self.assertEqual(app.selectbox(key='input_threshold_policy_multimodal_1').value,'Original 0,50')

    def test_interactive_oof_section_renders_without_external_reevaluation(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.button(key='nav_results').click().run()
        app.slider(key='oof_threshold').set_value(.25).run()
        self.assertFalse(app.exception,[e.value for e in app.exception])
        self.assertTrue(any('exploratorio' in w.value.lower() for w in app.warning))
        self.assertTrue(any('0,50' in w.value for w in app.info))


if __name__=='__main__':unittest.main()
