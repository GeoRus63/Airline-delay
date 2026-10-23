"""P7 receipt audit and inventory Notebook execution; no model fitting."""
from pathlib import Path
import json

import nbformat

try:
    from . import verify_downloads, verify_research_archive
    from .render_research_notebooks import _render
    from .run_pipeline import receipt, write_json
except ImportError:
    import verify_downloads
    import verify_research_archive
    from render_research_notebooks import _render
    from run_pipeline import receipt, write_json

ROOT = Path(__file__).resolve().parents[1]


def run(config, run_dir):
    """Seal actual published-file hashes after the formal run and reproduction exist."""
    output = Path(run_dir).resolve()
    if config.get('mode') != 'audit_delivery':
        raise ValueError('Expected audit_delivery mode')
    ledger_path = ROOT / 'experiments/final_access/2026-08.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    if ledger.get('status') != 'completed':
        raise ValueError('P7 delivery requires the completed formal final experiment')
    paths = config.get('delivery_paths', [])
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Provide a nonempty unique list of actual delivery files')
    delivery_files = []
    mutable_paths = {ROOT / 'experiments/index.csv', ROOT / 'research/研究记录.md'}
    for value in paths:
        path = (ROOT / value).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Delivery file is missing, empty or outside the project: {value}')
        if path in mutable_paths or path.is_relative_to(output):
            raise ValueError(f'Pipeline bookkeeping/current-run files cannot be sealed before completion: {value}')
        if path in delivery_files:
            raise ValueError(f'Duplicate resolved delivery file: {value}')
        delivery_files.append(path)
    raw = verify_downloads.verify(raw_only=True)
    write_json(output / 'raw_input_verification.json', raw)
    audit = verify_research_archive.verify_archive(all_runs=True)
    write_json(output / 'archive_integrity.json', audit)
    pending = set(audit['unsealed_runs_skipped']) - {output.name}
    if not audit['passed'] or not audit['final_verified'] or pending:
        raise ValueError(f'Archive audit incomplete: {audit["errors"]}; pending={sorted(pending)}')
    reproductions = [item['run_id'] for item in audit['runs']
                    if item.get('reproduction_verification', {}).get('verified') is True]
    if not reproductions:
        raise ValueError('P7 requires an integrity-verified complete frozen final reproduction')
    notebook = nbformat.read(ROOT / 'notebooks/00_数据清单.ipynb', as_version=4)
    rendered = _render(notebook, '00_数据清单.ipynb', output, ROOT)
    # Rendering updates the public 00 Notebook, so seal its delivered bytes only
    # after that update. The pipeline's later index/journal writes are excluded.
    delivered = [receipt(path) for path in delivery_files]
    write_json(output / 'delivery_receipts.json', delivered)
    return {'raw_inputs_passed': True, 'archive_integrity_passed': True,
            'formal_run_id': ledger['run_id'], 'verified_reproduction_runs': reproductions,
            'delivery_files': len(delivered), 'inventory_notebook': rendered,
            'refitted': False, 'raw_test_targets_analyzed': False,
            'raw_read_scope': 'Byte hashes and archive CRC/decompression only; no target parsing'}
