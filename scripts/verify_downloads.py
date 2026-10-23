"""离线检查搬迁文件的完整性；不处理航班或训练模型。"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def verify(raw_only=False):
    manifest=json.loads((ROOT/'tables/file_manifest.json').read_text(encoding='utf-8'))
    items=[item for item in manifest['files'] if not raw_only or item['file'].replace('\\','/').startswith('raw/')]
    if not items: raise ValueError('No files match the requested verification scope')
    failures=[];total=0
    for item in items:
        path=ROOT/item['file']
        if not path.is_file(): failures.append(f"Missing: {item['file']}");continue
        total+=path.stat().st_size
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        if path.stat().st_size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
            failures.append(f"Checksum mismatch: {item['file']}");continue
        try:
            if path.suffix=='.zip':
                with zipfile.ZipFile(path) as archive:
                    bad=archive.testzip()
                    if bad: failures.append(f"ZIP CRC failed: {item['file']} / {bad}")
            elif path.suffix=='.gz':
                with gzip.open(path,'rb') as stream:
                    for _ in iter(lambda:stream.read(1024*1024),b''):pass
        except (OSError,EOFError,zipfile.BadZipFile) as exc:
            failures.append(f"Unreadable archive: {item['file']}: {exc}")
    result={'files_checked':len(items),'total_bytes':total,'failures':failures,'passed':not failures,
            'scope':'raw inputs only' if raw_only else 'original delivery snapshot',
            'nonraw_entries_excluded':len(manifest['files'])-len(items)}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-only',action='store_true',help='Verify unchanged raw inputs without treating later source-code edits as corrupted data')
    verify(raw_only=parser.parse_args().raw_only)
