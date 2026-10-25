"""Reference checksum and cache behavior, using a local file-backed URL."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from navieco.reference_download import get_reference
from navieco.reference_registry import REFERENCE_ASSETS


class TestReferenceRelease(unittest.TestCase):
    def test_download_cache_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            zipfile = root / "NAViECO_v1.zip"
            with ZipFile(zipfile, "w") as out:
                out.writestr("seed.h5ad", b"placeholder-h5ad")
            sha = hashlib.sha256(zipfile.read_bytes()).hexdigest()
            previous_spec = REFERENCE_ASSETS.get("NAViECO_v1")
            old_env = dict(os.environ)
            try:
                os.environ.pop("NAVIECO_REFERENCE_DIR", None)
                os.environ["XDG_CACHE_HOME"] = str(root / "cache")
                REFERENCE_ASSETS["NAViECO_v1"] = {
                    "url": zipfile.as_uri(), "sha256": sha
                }
                result = get_reference("NAViECO_v1")
                self.assertTrue((result / "seed.h5ad").exists())
                self.assertEqual(get_reference("NAViECO_v1"), result)
            finally:
                os.environ.clear()
                os.environ.update(old_env)
                if previous_spec is None:
                    REFERENCE_ASSETS.pop("NAViECO_v1", None)
                else:
                    REFERENCE_ASSETS["NAViECO_v1"] = previous_spec

    def test_refuses_bad_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            zipfile = root / "NAViECO_v1.zip"
            with ZipFile(zipfile, "w") as out:
                out.writestr("seed.h5ad", b"placeholder-h5ad")
            previous_spec = REFERENCE_ASSETS.get("NAViECO_v1")
            old_env = dict(os.environ)
            try:
                os.environ.pop("NAVIECO_REFERENCE_DIR", None)
                os.environ["XDG_CACHE_HOME"] = str(root / "cache")
                REFERENCE_ASSETS["NAViECO_v1"] = {
                    "url": zipfile.as_uri(), "sha256": "0" * 64
                }
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    get_reference("NAViECO_v1")
            finally:
                os.environ.clear()
                os.environ.update(old_env)
                if previous_spec is None:
                    REFERENCE_ASSETS.pop("NAViECO_v1", None)
                else:
                    REFERENCE_ASSETS["NAViECO_v1"] = previous_spec


if __name__ == "__main__":
    unittest.main()
