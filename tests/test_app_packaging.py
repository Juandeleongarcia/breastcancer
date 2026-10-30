"""Regresión del paquete público separado: allowlist, destino y ausencia de datos."""
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from deploy.prepare_space import prepare


class AppPackagingTests(unittest.TestCase):
    def test_streamlit_package_contains_portable_tests_readme_identity_and_no_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination=prepare(Path(tmp)/'application',target='streamlit')
            self.assertIn('Juan de León García',(destination/'README.md').read_text(encoding='utf-8'))
            with ZipFile(destination.with_suffix('.zip')) as archive:
                names=archive.namelist()
                for name in ['deploy/app.py','deploy/requirements.txt','config/app_identity.json','tests/test_app_delivery.py','.gitignore']:
                    self.assertIn(name,names)
                self.assertFalse(any(name.startswith(('metadata/','models/','dataset/','breastdcedl/','backups/','.env/','.venv/')) for name in names))
                self.assertFalse(any(Path(name).suffix.lower() in ['.pt','.pth','.csv','.png','.dcm','.zip'] for name in names))
            with self.assertRaises(ValueError):prepare(destination,target='streamlit')


if __name__=='__main__':unittest.main()
