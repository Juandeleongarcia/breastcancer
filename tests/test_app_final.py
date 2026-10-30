"""Aplicación final: validación, defensa, privacidad y equivalencia con pesos reales."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from app_services import lab
from app_services.defence import Defence
from app_services.studies import Study,study_from_files,study_from_uploads,synthetic_study
from app_components.charts import slice_stack,intensity_surface,temporal_curve
from phase12_fixtures import study_files,png_bytes


class FinalInputTests(unittest.TestCase):
    def files(self):return {phase:(name,content) for phase,(name,content) in zip(['PRE','EARLY','LATE'],study_files(1))}

    def test_valid_triplet_and_missing_phase(self):
        files=self.files();study=lab.triplet(files)
        self.assertEqual(study.phases.shape,(1,3,256,256));self.assertFalse(study.phases.flags.writeable)
        files['LATE']=None
        with self.assertRaises(ValueError):lab.triplet(files)

    def test_duplicate_mismatched_phase_and_patient(self):
        files=self.files();files['EARLY']=files['PRE']
        with self.assertRaises(ValueError):lab.triplet(files)
        files=self.files();files['PRE']=('OTHER_z000_PRE.png',files['PRE'][1])
        with self.assertRaises(ValueError):lab.triplet(files)
        files=self.files();files['PRE']=('TEST_z000_LATE.png',files['PRE'][1])
        with self.assertRaises(ValueError):lab.triplet(files)

    def test_corrupt_dimensions_rgb_wrong_type_size_and_paths(self):
        bad=[b'not png',png_bytes(np.zeros((32,32),dtype=np.uint8)),png_bytes(np.zeros((256,256,3),dtype=np.uint8)),
             b'a'*(1024*1024+1),'C:/private/image.png']
        for content in bad:
            files=self.files();files['PRE']=('PRE.png',content)
            with self.subTest(kind=type(content).__name__):
                with self.assertRaises(ValueError):lab.triplet(files)
        for path in ['../PRE.png','C:\\secret\\PRE.png','/etc/PRE.png']:
            files=self.files();files['PRE']=(path,files['PRE'][1])
            with self.assertRaises(ValueError):lab.triplet(files)

    def test_defence_five_slots_labels_after_prediction_csv_has_no_identifiers(self):
        session=Defence()
        with self.assertRaises(ValueError):session.reveal(1)
        with self.assertRaises(ValueError):session.next()
        for n in range(5):
            study=study_from_files(study_files(1,patient=f'PRIVATE_TEST_{n}'))
            # Variar el contenido además del ID para representar muestras distintas.
            phases=study.phases.copy();phases[0,0,0,0]=n;study=Study(study.patient_id,study.slice_indices,phases)
            result=dict(model='MODEL_TEST_ONLY',checksum='f'*64,threshold=.5,patient_probability=.3,
                prediction=0,n_models=5,n_slices=1,device='cpu',seconds=.1)
            key=session.input_key(study,'image',None);session.record(result,study,key)
            with self.assertRaises(ValueError):session.record(result,study,key)
            session.reveal(n%2)
            if n<4:session.next()
        data=session.csv().decode('utf-8-sig');frame=pd.read_csv(BytesIO(session.csv()))
        self.assertEqual(frame.muestra.tolist(),[1,2,3,4,5]);self.assertEqual(len(frame),5)
        self.assertNotIn('PRIVATE_TEST',data);self.assertNotIn('age',frame);self.assertNotIn('HR_HER2_STATUS',frame)
        self.assertNotIn('study_fingerprint',frame)
        with self.assertRaises(ValueError):session.next()

    def test_3d_real_indices_single_surface_and_temporal_three_points(self):
        demo=synthetic_study();study=Study('test',(2,5,11),demo.phases[:3])
        figure,count=slice_stack(study,0,1)
        self.assertEqual(set(np.unique(figure.data[1].z)),{5});self.assertEqual(set(figure.data[-1].z),{5})
        one=Study('test',(0,),demo.phases[:1]);surface=intensity_surface(one,0)
        np.testing.assert_allclose(surface.data[0].z,one.phases[0,0,::4,::4]/255.)
        self.assertIn('Intensidad',surface.layout.scene.zaxis.title.text)
        curve=temporal_curve(study,0,128,128,8)
        self.assertEqual(list(curve.data[0].x),['PRE','EARLY','LATE'])
        self.assertEqual(len(curve.data[0].y),3)

    def test_package_cannot_extract_arbitrary_files(self):
        from zipfile import ZipFile
        import hashlib
        stream=BytesIO()
        with ZipFile(stream,'w') as archive:archive.writestr('../escape.py','print(1)')
        content=stream.getvalue()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):lab.extract_package(content,hashlib.sha256(content).hexdigest(),Path(tmp)/'package')


@unittest.skipUnless((Path(__file__).resolve().parents[1]/'models/app_bundle/catalog.json').is_file(),'Se necesitan los artefactos reales locales; no se simulan modelos')
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root,cls.catalog=lab.read_catalog()
        cls.study=lab.public_example(cls.root)
        cls.image=lab.load_resource(cls.root,'image');cls.multi=lab.load_resource(cls.root,'multimodal')

    def test_ten_real_checkpoints_counts_and_repeatable_cpu_predictions(self):
        for kind,resource,count in [('image',self.image,72593),('multimodal',self.multi,72600)]:
            self.assertEqual(len(resource.members),5)
            self.assertTrue(all(sum(p.numel() for p in b['model'].parameters())==count for b in resource.members))
            clinical=dict(age=45.,HR_HER2_STATUS='HRposHER2neg') if kind=='multimodal' else None
            with patch('torch.Tensor.backward',side_effect=AssertionError('No training')):
                a=lab.predict(self.study,resource,clinical);b=lab.predict(self.study,resource,clinical)
            self.assertEqual(a['patient_probability'],b['patient_probability'])
            self.assertEqual(a['patient_probability'],np.mean(a['member_probabilities']))
            self.assertEqual(a['device'],'cpu')

    def test_multimodal_equals_original_pipeline_on_authorized_public_train_example(self):
        from app_services.package import discover,select
        from src.phase13_inference import predict_patient,ImageTransform
        from src.phase13_clinical import ClinicalTransform
        from src.phase13_model import SpatialSmallMultimodal
        sources=select(discover());campaign=sources['multimodal']
        state=json.loads((campaign/'campaign.json').read_text())
        index=pd.read_csv(Path(__file__).resolve().parents[1]/'metadata/samples.csv')
        row=index[index.split.eq('train')].iloc[0]
        patients=pd.read_csv(Path(__file__).resolve().parents[1]/'metadata/patients.csv')
        clinical=patients[patients.pid.eq(row.patient_id)].iloc[0]
        values=dict(age=None if pd.isna(clinical.age) else float(clinical.age),
                    HR_HER2_STATUS=None if pd.isna(clinical.HR_HER2_STATUS) else clinical.HR_HER2_STATUS)
        reference=[]
        for fold in range(5):
            raw=campaign/state['folds'][str(fold)]['last_run']/'best_epoch.pt'
            payload=torch.load(raw,map_location='cpu',weights_only=True);cfg=payload['config']
            transform=ClinicalTransform(payload['clinical_transform'])
            model=SpatialSmallMultimodal(cfg,transform.dimension);model.load_state_dict(payload['model_state_dict']);model.eval()
            bundle=dict(model=model,config=cfg,clinical=transform,images=ImageTransform(cfg),device=torch.device('cpu'),contract=payload['inference_contract'])
            before={k:v.clone() for k,v in model.state_dict().items()}
            result=predict_patient(bundle,self.study.phases,values,'input')
            reference.append(result['patient_probability'])
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        actual=lab.predict(self.study,self.multi,values)
        np.testing.assert_array_equal(actual['member_probabilities'],reference)
        self.assertEqual(actual['patient_probability'],float(np.mean(reference)))

    def test_missing_unknown_clinical_and_synthetic_never_fabricate_outputs(self):
        with self.assertRaises(ValueError):lab.predict(self.study,self.multi)
        with self.assertRaises(ValueError):lab.predict(synthetic_study(),self.multi,dict(age=45.,HR_HER2_STATUS='TripleNeg'))
        result=lab.predict(self.study,self.multi,dict(age=None,HR_HER2_STATUS='UNSEEN_CATEGORY'))
        self.assertTrue(np.isfinite(result['patient_probability']))
        with self.assertRaises(ValueError):lab.predict(self.study,self.multi,dict(age=45.,HR_HER2_STATUS='TripleNeg',pCR=1))

    def test_image_pipeline_equals_original_real_checkpoint_inference(self):
        from app_services.package import discover,select
        from app_services.inference import load_checkpoint,predict_study
        source=select(discover())['image'];state=json.loads((source/'campaign.json').read_text())
        reference=[]
        for fold in range(5):
            name=str(state['folds'][str(fold)]['last_run']).replace('\\','/').split('/')[-1]
            raw=source/f'fold_{fold}'/'runs'/name/'best_epoch.pt'
            reference.append(predict_study(self.study,load_checkpoint(raw,'SPATIAL_SMALL'))['patient_probability'])
        actual=lab.predict(self.study,self.image)
        np.testing.assert_array_equal(actual['member_probabilities'],reference)


if __name__=='__main__':unittest.main()
