"""Small JSON-driven entry point with immutable experiment receipts."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def receipt(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def update_index(run_id, config, status, decision):
    path = ROOT / 'experiments/index.csv'
    fields = ['run_id', 'parent_run_id', 'stage', 'status', 'change', 'decision', 'artifact_dir']
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    row = dict(zip(fields, [run_id, config.get('parent_run_id', ''), config['stage'], status,
                           config['change'], decision, f'experiments/runs/{run_id}']))
    rows = [r for r in rows if r['run_id'] != run_id] + [row]
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['run_id']))


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def monitor_resources(path, stop):
    import psutil
    process = psutil.Process()
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['utc', 'process_rss_bytes', 'process_cpu_percent', 'system_available_bytes',
                         'process_tree_rss_bytes', 'process_tree_cpu_seconds', 'child_processes'])
        while not stop.is_set():
            family = [process, *process.children(recursive=True)]
            tree_rss, tree_cpu = 0, 0.0
            for member in family:
                try:
                    tree_rss += member.memory_info().rss
                    usage = member.cpu_times()
                    tree_cpu += usage.user + usage.system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            writer.writerow([now(), process.memory_info().rss, process.cpu_percent(), psutil.virtual_memory().available,
                             tree_rss, tree_cpu, len(family) - 1])
            stream.flush()
            stop.wait(5)


def execute(config, run_dir):
    if config['mode'] == 'select_features':
        from select_models import select_features
        return select_features([ROOT / 'experiments/runs' / value for value in config['candidate_run_ids']],
                               run_dir / 'selection')
    if config['mode'] == 'select_model':
        from select_models import select_model
        from explain_models import explain_run
        from render_research_notebooks import render_analysis
        feature_selection_run = ROOT / 'experiments/runs' / config['feature_selection_run_id']
        selection_path = feature_selection_run / 'selection/feature_selection.json'
        if json.loads((feature_selection_run / 'manifest.json').read_text(encoding='utf-8')).get('status') != 'completed':
            raise ValueError('Model selection requires completed feature-selection evidence')
        chosen_features = json.loads(selection_path.read_text(encoding='utf-8'))
        if chosen_features['selected_run_id'] != config['feature_source_run_id']:
            raise ValueError('Algorithm comparison source differs from the selected mainline feature run')
        write_json(run_dir / 'feature_selection_reference.json', receipt(selection_path))
        result = select_model(ROOT / 'experiments/runs' / config['feature_source_run_id'],
                              [ROOT / 'experiments/runs' / value for value in config['algorithm_run_ids']],
                              run_dir / 'selection')
        for source_id in dict.fromkeys([config['feature_source_run_id'], result['selected_run_id']]):
            explain_run(ROOT / 'experiments/runs' / source_id, run_dir / 'explanations' / source_id)
        render_analysis(ROOT / 'experiments/runs' / config['rq1_development_run_id'],
                        model_runs=[ROOT / 'experiments/runs' / name for name in config['model_run_ids']],
                        output_dir=run_dir)
        return result
    if config['mode'] == 'prepare':
        import prepare_flights
        prepare_flights.self_check()
        print('Synthetic boundary checks passed.', flush=True)
        summary = prepare_flights.run()
        assert summary['raw_flight_rows'] == config['expected_raw_rows']
        for name in ['cleaning_summary.json', 'connection_exclusions.csv']:
            shutil.copy2(ROOT / 'tables' / name, run_dir / name)
        outputs = [receipt(ROOT / 'processed' / (name + '.csv.gz')) for name in
                   ['flights_clean', 'connection_candidates', 'flight_connections']]
        data_manifest = {'data_id': config['data_id'], 'config': config, 'outputs': outputs, 'summary': summary}
        write_json(ROOT / 'processed' / (config['data_id'] + '_manifest.json'), data_manifest)
        write_json(run_dir / 'data_manifest.json', data_manifest)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    if config['mode'] == 'development' and config.get('analysis_only', False):
        from complete_rq1 import run as complete_rq1
        from analyze_single_flights import run as single_flight_eda
        from explain_models import explain_run
        from render_research_notebooks import render_analysis
        result = {'rq1': complete_rq1(config, run_dir), 'eda': single_flight_eda(config, run_dir)}
        for source_id in config.get('explanation_source_run_ids', []):
            explain_run(ROOT / 'experiments/runs' / source_id, run_dir / 'explanations' / source_id)
        render_analysis(run_dir, model_runs=[ROOT / 'experiments/runs' / name for name in config.get('model_run_ids', [])])
        return result
    if config['mode'] == 'development':
        from build_features import load_configured_features
        from train_models import run_development
        from train_models import resolve_features
        if config.get('feature_archive_path'):
            from feature_cache import load_development_archive
            frame = load_development_archive(config)
        else:
            frame = load_configured_features(config, development_only=True)
        write_json(run_dir / 'feature_audit.json', dict(frame.attrs))
        shutil.copy2(ROOT / 'tables/feature_dictionary.csv', run_dir / 'feature_dictionary.csv')
        (run_dir / 'data').mkdir(exist_ok=True)
        frame.to_pickle(run_dir / 'data/features.pkl')
        (run_dir / 'tables').mkdir(exist_ok=True)
        if str(config.get('feature_group', '')).upper() in ['G4A', 'G4B']:
            from weather_pipeline_features import write_weather_audit
            write_weather_audit(frame, run_dir, config)
        missing = frame[resolve_features(config)['features']].isna().sum().rename('missing_n').to_frame()
        missing['rows'] = len(frame)
        missing['missing_share'] = missing.missing_n / len(frame)
        missing.rename_axis('feature').to_csv(run_dir / 'tables/feature_missing.csv')
        result = run_development(frame, config, run_dir)
        if config.get('compare_parent', False):
            from compare_experiments import compare_runs
            compare_runs(ROOT / 'experiments/runs' / config['parent_run_id'], run_dir,
                         run_dir / 'comparison', allow_running_candidate=True)
        if config.get('common_weather_no_weather_run'):
            from weather_pipeline_features import compare_common_weather
            compare_common_weather(frame, run_dir, config)
        from analyze_eda import run as eda
        from analyze_rq1 import run as rq1
        from render_notebooks import render_notebooks
        eda(frame, run_dir)
        rq1(run_dir, development=True)
        for name in ['cleaning_summary.json', 'connection_exclusions.csv']:
            shutil.copy2(ROOT / config.get('preparation_evidence_dir', 'tables') / name, run_dir / name)
        render_notebooks(run_dir)
        return result
    if config['mode'] == 'prepare_aircraft':
        import prepare_aircraft
        prepare_aircraft.main()
        for name in ['aircraft_model_mapping.csv', 'aircraft_monthly_match_rates.csv', 'aircraft_unmatched_tails.csv']:
            shutil.copy2(ROOT / 'references' / name, run_dir / name)
        return {'inventory_main': 2024, 'coverage_comparison': 2025,
                'model_granularity': 'family main; variant replacement branch; no seats or age'}
    if config['mode'] == 'prepare_weather':
        from prepare_weather_dataset import run
        return run(config, run_dir)
    if config['mode'] == 'final':
        from final_evaluation import run, verify_saved
        from render_research_notebooks import render_final
        result = run(config, run_dir)
        write_json(run_dir / 'final_verification.json', verify_saved(run_dir))
        render_final(run_dir)
        return result
    if config['mode'] == 'reproduce_final':
        from reproduce_final import run as reproduce_final, render_notebook
        result = reproduce_final(config, run_dir)
        # The original formal experiment and its public Notebook stay untouched.
        rendered = render_notebook(run_dir)
        write_json(run_dir / 'reproduction_notebook_receipt.json', rendered)
        return result
    if config['mode'] == 'rq1_final':
        from final_evaluation import verify_saved
        from complete_rq1 import run as complete_rq1
        from analyze_single_flights import run as single_flight_eda
        from render_research_notebooks import render_final
        prediction_run = ROOT / 'experiments/runs' / config['prediction_final_run_id']
        ledger = json.loads((ROOT / 'experiments/final_access/2026-08.json').read_text(encoding='utf-8'))
        if ledger.get('status') != 'completed' or ledger.get('run_id') != prediction_run.name:
            raise ValueError('Final RQ1 requires the completed, uniquely claimed August prediction run')
        if json.loads((prediction_run / 'manifest.json').read_text(encoding='utf-8')).get('status') != 'completed':
            raise ValueError('Final prediction archive is not completed')
        write_json(run_dir / 'prediction_final_verification.json', verify_saved(prediction_run))
        result = {'rq1': complete_rq1(config, run_dir), 'eda': single_flight_eda(config, run_dir)}
        render_final(prediction_run, rq1_run_dir=run_dir, output_dir=run_dir)
        return result
    if config['mode'] == 'audit_preparation':
        from audit_preparation import run
        return run(run_dir)
    if config['mode'] == 'rebuild_cohorts':
        from rebuild_cohorts import run
        return run(run_dir, ROOT / 'processed' / config['data_id'])
    raise ValueError(f"Unsupported mode: {config['mode']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if not (args.run_id.startswith('R') and args.run_id[1:].isdigit()):
        raise ValueError('run_id must be R followed by digits')
    run_dir = ROOT / 'experiments/runs' / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / 'config.json', config)
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = str(config.get('threads', 16))
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source_names = ['scripts/run_pipeline.py', 'scripts/prepare_flights.py']
    if config['mode'] in ['select_features', 'select_model']:
        source_names += ['scripts/select_models.py', 'scripts/final_evaluation.py', 'scripts/train_models.py',
                         'scripts/build_features.py', 'scripts/evaluate_models.py', 'scripts/compare_experiments.py',
                         'configs/model_selection_protocol.json']
        if config['mode'] == 'select_model':
            source_names += ['scripts/explain_models.py', 'scripts/render_research_notebooks.py']
    elif config['mode'] == 'audit_preparation':
        source_names.append('scripts/audit_preparation.py')
    elif config['mode'] == 'prepare_aircraft':
        source_names += ['scripts/prepare_aircraft.py', 'references/aircraft_reference_2024.csv',
                         'references/aircraft_reference_2025.csv', 'references/aircraft_type_aliases.csv']
    elif config['mode'] == 'prepare_weather':
        source_names += ['scripts/prepare_weather.py', 'scripts/prepare_weather_dataset.py',
                         'scripts/download_taf_text.py', 'references/airports.csv']
    elif config['mode'] in ['final', 'reproduce_final']:
        frozen = json.loads((ROOT / config['freeze_manifest_path']).read_text(encoding='utf-8'))
        source_names += [item['path'] for item in frozen['source_files']]
        source_names += ['scripts/reproduce_final.py', 'scripts/render_research_notebooks.py', config['freeze_manifest_path']]
    elif config['mode'] == 'rq1_final':
        source_names += ['scripts/final_evaluation.py', 'scripts/complete_rq1.py', 'scripts/rq1_standardization.py', 'scripts/analyze_rq1.py',
                         'scripts/analyze_eda.py', 'scripts/analyze_single_flights.py', 'scripts/render_research_notebooks.py',
                         'references/airports.csv', 'references/aircraft_reference_2024.csv',
                         'references/aircraft_reference_2025.csv', config['rq1_frozen_reference_path']]
    elif config['mode'] == 'rebuild_cohorts':
        source_names.append('scripts/rebuild_cohorts.py')
    elif config['mode'] == 'development':
        source_names += ['scripts/build_features.py', 'scripts/train_models.py', 'scripts/evaluate_models.py',
                         'scripts/analyze_eda.py', 'scripts/analyze_rq1.py', 'scripts/render_notebooks.py',
                         'scripts/operational_features.py']
        if str(config.get('feature_group', 'G0')).upper() not in ['G0', 'G1']:
            source_names += ['references/aircraft_reference_2024.csv', 'references/aircraft_type_aliases.csv']
        if str(config.get('feature_group', '')).upper() in ['G4A', 'G4B']:
            source_names += ['scripts/weather_pipeline_features.py', 'scripts/weather_features.py',
                             'scripts/prepare_weather.py']
        source_names.append('references/airports.csv')
        if config.get('compare_parent', False):
            source_names.append('scripts/compare_experiments.py')
        if config.get('feature_archive_path'):
            source_names.append('scripts/feature_cache.py')
        if config.get('analysis_only', False):
            source_names += ['scripts/complete_rq1.py', 'scripts/rq1_standardization.py',
                             'scripts/analyze_single_flights.py', 'scripts/explain_models.py',
                             'scripts/render_research_notebooks.py', 'configs/rq1_protocol.json',
                             'references/aircraft_reference_2024.csv', 'references/aircraft_reference_2025.csv']
        source_names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'notebooks').glob('*.ipynb')) if p.name[:2] in ['01', '02', '03', '04', '05']]
        source_names.append('tables/feature_dictionary.csv')
    source_names.append(config_path.relative_to(ROOT).as_posix())
    source_names = list(dict.fromkeys(source_names))
    for name in source_names:
        committed = subprocess.check_output(['git', 'show', f'{git_commit}:{name}'], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != sha256(ROOT / name):
            raise RuntimeError(f'Commit the executed source first: {name}')
    manifest = {'run_id': args.run_id, 'status': 'running', 'started_at_utc': now(), 'git_commit': git_commit,
                'command': subprocess.list2cmdline([sys.executable, *sys.argv]), 'python': sys.version,
                'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
                'environment': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                'source_files': [receipt(ROOT / name) for name in source_names],
                'input_manifest': receipt(ROOT / 'tables/file_manifest.json'), 'config': config}
    write_json(run_dir / 'manifest.json', manifest)
    if config.get('data_manifest_path'):
        data_manifest_file = ROOT / config['data_manifest_path']
        data_manifest = json.loads(data_manifest_file.read_text(encoding='utf-8'))
        for item in data_manifest['outputs'] + data_manifest.get('inputs', []):
            if sha256(ROOT / item['path']) != item['sha256']:
                raise RuntimeError(f"Derived input changed: {item['path']}")
        shutil.copy2(data_manifest_file, run_dir / 'data_manifest.json')
        manifest['derived_input_manifest'] = receipt(data_manifest_file)
    update_index(args.run_id, config, 'running', 'Execution in progress')
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_resources, args=(run_dir / 'resources.csv', stop), daemon=True)
    monitor.start()
    started = time.perf_counter()
    status, decision = 'completed', 'Retain outputs; review before next stage'
    with (run_dir / 'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f'{now()} {args.run_id} started at commit {git_commit}', flush=True)
            try:
                manifest['result'] = execute(config, run_dir)
            except BaseException:
                status, decision = 'failed', 'Retain failure; fix under a new run_id'
                traceback.print_exc()
            finally:
                stop.set()
                monitor.join(timeout=10)
                manifest.update(status=status, ended_at_utc=now(), elapsed_seconds=time.perf_counter() - started)
                print(f"{now()} status={status}; seconds={manifest['elapsed_seconds']:.1f}", flush=True)
    manifest['outputs'] = [receipt(p) for p in sorted(run_dir.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    write_json(run_dir / 'manifest.json', manifest)
    update_index(args.run_id, config, status, decision)
    with (ROOT / 'research/研究记录.md').open('a', encoding='utf-8') as log:
        log.write(f"\n### {now()} — {args.run_id} — {config['stage']}\n\n"
                  f"- **依据**：完整执行流程；直接前轮 {config.get('parent_run_id') or '无'}。\n"
                  f"- **唯一改动**：{config['change']}。\n"
                  f"- **预期**：遵守固定样本/时间协议并保留可核查证据。\n"
                  f"- **观察**：状态 {status}，耗时 {manifest['elapsed_seconds']:.1f} 秒；证据 `experiments/runs/{args.run_id}/manifest.json` 与 `run.log`。\n"
                  f"- **决定**：{decision}。\n"
                  "- **下一步**：检查运行产物和方法审查意见后推进下一阶段。\n")
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
