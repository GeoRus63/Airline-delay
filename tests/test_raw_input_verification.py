from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_downloads


class RawInputVerification(unittest.TestCase):
    def fixture(self, root):
        (root / 'raw').mkdir()
        (root / 'scripts').mkdir()
        (root / 'tables').mkdir()
        files = {'raw/example.csv.gz': gzip.compress(b'x,y\n1,2\n', mtime=0),
                 'scripts/evolving.py': b'print(1)\n'}
        manifest = []
        for name, value in files.items():
            (root / name).write_bytes(value)
            manifest.append({'file': name, 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()})
        (root / 'tables/file_manifest.json').write_text(json.dumps({'files': manifest}), encoding='utf-8')

    def test_raw_scope_does_not_misclassify_committed_code_evolution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (root / 'scripts/evolving.py').write_bytes(b'print(2)\n')
            with patch.object(verify_downloads, 'ROOT', root), redirect_stdout(io.StringIO()):
                result = verify_downloads.verify(raw_only=True)
                self.assertTrue(result['passed'])
                self.assertEqual(result['files_checked'], 1)
                self.assertEqual(result['nonraw_entries_excluded'], 1)
                with self.assertRaises(SystemExit):
                    verify_downloads.verify()

    def test_raw_corruption_is_never_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (root / 'raw/example.csv.gz').write_bytes(b'broken archive')
            with patch.object(verify_downloads, 'ROOT', root), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    verify_downloads.verify(raw_only=True)


if __name__ == '__main__':
    unittest.main()
