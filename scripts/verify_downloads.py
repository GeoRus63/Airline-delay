"""离线检查搬迁文件的完整性；不处理航班或训练模型。"""
from pathlib import Path
import gzip
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def verify():
    manifest=json.loads((ROOT/'tables/file_manifest.json').read_text(encoding='utf-8'))
    failures=[];total=0
    for item in manifest['files']:
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
    result={'files_checked':len(manifest['files']),'total_bytes':total,'failures':failures,'passed':not failures}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)
    return result

if __name__=='__main__':verify()
