"""Versioned weather loading and query-level coverage for the shared pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .weather_features import augment_metar, augment_taf, augment_noaa, noaa_overlap_diagnostics
    from .analyze_eda import plot_style, save_figure
except ImportError:
    from weather_features import augment_metar, augment_taf, augment_noaa, noaa_overlap_diagnostics
    from analyze_eda import plot_style, save_figure

ROOT = Path(__file__).resolve().parents[1]
METAR_MODEL_NUMERIC = [f'metar_{name}' for name in [
    'wind_knots', 'gust_knots', 'visibility_miles', 'ceiling_ft', 'precip_amount_in',
    'precip_flag', 'rain_flag', 'snow_flag', 'thunder_flag', 'age_minutes', 'missing',
    'sky_reported', 'ceiling_reported', 'weather_codes_reported']]
TAF_MODEL_NUMERIC = ['taf_bulletin_age_minutes', 'taf_lead_minutes', 'taf_missing'] + [
    f'taf_{kind}_{name}' for kind in ['prevailing', 'tempo', 'probability']
    for name in ['wind_knots', 'gust_knots', 'visibility_miles', 'ceiling_ft',
                 'precip_flag', 'rain_flag', 'snow_flag', 'thunder_flag', 'segment_count', 'missing']]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(config):
    manifest_path = ROOT / config['weather_manifest_path']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for record in manifest['outputs'] + manifest.get('inputs', []):
        if file_sha(ROOT / record['path']) != record['sha256']:
            raise ValueError(f"Weather input changed: {record['path']}")
    return manifest_path, manifest


def load_noaa(directory):
    path = Path(directory) / 'noaa_observations.csv.gz'
    columns = pd.read_csv(path, nrows=0).columns
    string_columns = [name for name in columns if name in ['STATION', 'Station_name', 'station_name', 'source_file']
                      or name.endswith(('_Source_Code', '_Quality_Code', '_Measurement_Code'))]
    return pd.read_csv(path, dtype={name: 'string' for name in string_columns}, low_memory=False)


def augment_weather(frame, config):
    """Preserve cohort and order; only use the predeclared archive availability rules."""
    group = str(config['feature_group']).upper()
    if group not in ['G4A', 'G4B']:
        raise ValueError('Weather composition requires G4A or G4B')
    manifest_path, manifest = load_manifest(config)
    directory = manifest_path.parent
    original_attrs = dict(frame.attrs)
    original_ids = frame.connection_id.to_numpy(copy=True)
    metar = pd.read_pickle(directory / 'metar_observations.pkl')
    result = augment_metar(frame, metar)
    coverage_before = int(result.metar_missing.eq(0).sum())
    if config.get('noaa_backfill', False):
        if group != 'G4B':
            raise ValueError('The registered NOAA branch follows G4B')
        result = augment_noaa(result, load_noaa(directory), pd.read_csv(ROOT / 'references/airports.csv'))
    noaa_attrs = dict(result.attrs)
    if group == 'G4B':
        result = augment_taf(result, pd.read_pickle(directory / 'taf_segments.pkl'),
                             pd.read_pickle(directory / 'taf_bulletins.pkl'),
                             scheduled_dep_col='downstream_scheduled_dep_utc',
                             allow_unverified_revisions=False)
    if len(result) != len(frame) or not np.array_equal(result.connection_id.to_numpy(), original_ids):
        raise ValueError('Weather augmentation changed the evaluation cohort or row order')
    numeric = METAR_MODEL_NUMERIC + (TAF_MODEL_NUMERIC if group == 'G4B' else [])
    for column in numeric:
        if column not in result:
            raise ValueError(f'Required weather feature absent: {column}')
        result[column] = pd.to_numeric(result[column], errors='raise').astype(float)
    result.attrs = {**original_attrs, **noaa_attrs, 'weather_audit': {
        'data_id': manifest['data_id'], 'manifest': manifest_path.relative_to(ROOT).as_posix(),
        'manifest_sha256': file_sha(manifest_path), 'rows': len(result), 'cohort_unchanged': True,
        'metar_available_before_noaa': coverage_before,
        'metar_available_after_noaa': int(result.metar_missing.eq(0).sum()),
        'taf_available': int(result.taf_missing.eq(0).sum()) if group == 'G4B' else None,
        'noaa_backfill': bool(config.get('noaa_backfill', False)),
        'metar_model_columns': METAR_MODEL_NUMERIC,
        'taf_model_columns': TAF_MODEL_NUMERIC if group == 'G4B' else [],
        'assumptions': manifest['assumptions'],
        'metar_correction_policy': 'Exclude untimed COR values; retain prior non-COR observations within 120 minutes',
        'taf_policy': 'Latest bulletin state first; untimed/revised/cancelled/expired state cannot fall back to an older bulletin',
        'noaa_policy': 'Retrospective sensitivity branch; source-specific QC, units aligned, IEM observed values retained',
    }}
    return result


def write_weather_audit(frame, run_dir, config):
    """Archive query-level coverage; no flight outcome is read by this function."""
    output = Path(run_dir)
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    weather_flags = [name for name in ['metar_missing', 'taf_missing'] if name in frame]
    if not weather_flags:
        return None
    queries = frame[['airport_id', 'prediction_time', *weather_flags]].copy()
    queries['date_utc'] = pd.to_datetime(queries.prediction_time, utc=True).dt.strftime('%Y-%m-%d')
    queries['month_utc'] = queries.date_utc.str[:7]
    for column in weather_flags:
        queries[column.replace('_missing', '_available')] = 1 - queries[column]
    available = [name.replace('_missing', '_available') for name in weather_flags]
    pieces = {}
    for name, keys in [('airport_date', ['airport_id', 'date_utc']), ('month', ['month_utc']), ('airport', ['airport_id'])]:
        table = queries.groupby(keys, dropna=False)[available].agg(['count', 'sum', 'mean'])
        table.columns = [f'{kind}_{stat}' for kind, stat in table.columns]
        table.reset_index().to_csv(tables / f'weather_coverage_{name}.csv', index=False)
        pieces[name] = len(table)
    if 'taf_status' in frame:
        frame.taf_status.value_counts(dropna=False).rename_axis('status').rename('n').to_csv(tables / 'taf_status_counts.csv')
    daily = queries.groupby(['airport_id', 'date_utc'])[available].mean()
    plot_style()
    fig, axes = plt.subplots(len(available), 1, figsize=(13, 4.5 * len(available)), squeeze=False, layout='constrained')
    for ax, column in zip(axes.flat, available):
        pivot = daily[column].unstack('date_utc').sort_index()
        im = ax.imshow(pivot.to_numpy(float), aspect='auto', vmin=0, vmax=1, interpolation='none', cmap='viridis')
        xs = np.arange(0, len(pivot.columns), 14)
        ys = np.arange(0, len(pivot), 25)
        ax.set(xticks=xs, xticklabels=pivot.columns[xs], yticks=ys, yticklabels=pivot.index[ys],
               xlabel='Prediction date (UTC)', ylabel='AirportID (ordered)',
               title=f'{column}: fraction of eligible RQ2 queries with matched source; blank = no queries')
        ax.tick_params(axis='x', rotation=30)
        fig.colorbar(im, ax=ax, label='Available fraction')
    save_figure(fig, figures / 'weather_query_coverage.png')
    if config.get('noaa_backfill', False):
        directory = (ROOT / config['weather_manifest_path']).parent
        diagnosis = noaa_overlap_diagnostics(pd.read_pickle(directory / 'metar_observations.pkl'),
                                             load_noaa(directory), pd.read_csv(ROOT / 'references/airports.csv'))
        diagnosis['summary'].to_csv(tables / 'noaa_overlap_summary.csv', index=False)
        (output / 'data').mkdir(exist_ok=True)
        diagnosis['pairs'].to_csv(output / 'data/noaa_overlap_pairs.csv.gz', index=False, compression='gzip')
    summary = {'rows': len(queries), 'tables': pieces,
               'availability': {name: int(queries[name].sum()) for name in available},
               'interpretation': 'Source match availability is not complete per-variable coverage; no-query cells are missing, not zero'}
    (output / 'weather_coverage_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
