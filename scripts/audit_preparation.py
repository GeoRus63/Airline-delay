"""Audit prepared rows and test whether future-period QA changes earlier membership."""
from pathlib import Path
import json

import pandas as pd

from prepare_flights import build_connections

ROOT = Path(__file__).resolve().parents[1]


def run(output):
    output = Path(output)
    print('Reading prepared flights for structural and prefix audits', flush=True)
    flights = pd.read_csv(ROOT / 'processed/flights_clean.csv.gz', low_memory=False)
    for col in ['FlightDate', 'scheduled_dep_utc', 'scheduled_arr_utc', 'actual_dep_utc', 'actual_arr_utc']:
        flights[col] = pd.to_datetime(flights[col], utc=col.endswith('_utc'))
    assert len(flights) == 3057590 and flights.flight_id.is_unique
    assert flights.source_row.ge(2).all()
    assert flights.FlightDate.dt.to_period('M').astype(str).isin([f'2026-{m:02d}' for m in range(4, 9)]).all()
    monthly = flights.assign(month=flights.FlightDate.dt.strftime('%Y-%m')).groupby('month').agg(
        raw_rows=('flight_id', 'size'), missing_tail=('Tail_Number', lambda x: x.isna().sum()),
        duplicate_rows=('duplicate_key', 'sum'), cancelled=('Cancelled', 'sum'), diverted=('Diverted', 'sum'),
        time_valid=('time_valid', 'sum'), schedule_missing=('scheduled_dep_utc', lambda x: x.isna().sum()))
    monthly.to_csv(output / 'monthly_structural_audit.csv')
    codes = pd.read_csv(ROOT / 'references/airport_codes.csv', parse_dates=['first_date', 'last_date'])
    date_map = codes.set_index('airport_code')
    code_audit = {}
    for side in ['Origin', 'Dest']:
        first = flights[side].map(date_map.first_date)
        last = flights[side].map(date_map.last_date)
        code_audit[side] = int((~flights.FlightDate.between(first, last)).sum())
    pd.DataFrame({'field': flights.columns, 'dtype': [str(t) for t in flights.dtypes],
                  'role': ['source_or_preparation_audit'] * len(flights.columns)}).to_csv(output / 'flight_fields.csv', index=False)
    # Retrospective validity checks are not model features. Compare prefix processing
    # without printing any August targets or aggregate target distributions.
    boundary_rows = []
    for cutoff_text in ['2026-05-01', '2026-06-01', '2026-07-01', '2026-08-01']:
        cutoff = pd.Timestamp(cutoff_text, tz='UTC')
        prefix = flights.loc[flights.scheduled_dep_utc.lt(cutoff)].copy()
        _, flags = build_connections(prefix)
        old = prefix.set_index('flight_id')
        new = flags.set_index('flight_id')
        for flag in ['tail_schedule_incomplete', 'schedule_tie', 'actual_order_conflict']:
            difference = old.loc[new.index, flag].fillna(False).astype(bool).ne(new[flag].astype(bool))
            boundary_rows.append({'cutoff_utc': cutoff.isoformat(), 'flag': flag,
                                  'prefix_flights': len(prefix), 'changed_flight_flags': int(difference.sum())})
            if difference.any():
                ids = new.index[difference]
                old.loc[ids, ['source_file', 'source_row', 'Tail_Number', 'scheduled_dep_utc', flag]].to_csv(
                    output / f'prefix_difference_{cutoff_text}_{flag}.csv', index=True)
        print(f'QA prefix checked through {cutoff_text}', flush=True)
    pd.DataFrame(boundary_rows).to_csv(output / 'prefix_qa_audit.csv', index=False)
    candidates = pd.read_csv(ROOT / 'processed/connection_candidates.csv.gz', low_memory=False)
    candidates['prediction_time'] = pd.to_datetime(candidates.upstream_actual_arr_utc, utc=True)
    development = candidates.loc[candidates.prediction_time.ge('2026-04-01') & candidates.prediction_time.lt('2026-08-01')]
    development.groupby('first_exclusion_reason').head(3).to_csv(output / 'development_trace_examples.csv', index=False)
    connections = candidates.loc[candidates.main_analysis_eligible]
    dev = development.loc[development.main_analysis_eligible]
    summary = {'raw_rows': len(flights), 'raw_row_ids_unique': True,
               'code_date_outside_observed_range': code_audit,
               'candidate_rows': len(candidates), 'accepted_rows': len(connections),
               'development_accepted_rows': len(dev),
               'development_same_instant_pairs': int(dev.actual_ground_minutes.eq(0).sum()),
               'future_qa_changed_flags_total': int(sum(r['changed_flight_flags'] for r in boundary_rows)),
               'scope': 'Structural full-period audit; outcome analysis confined to April-July UTC',
               'limitation': 'BTS and accepted adjacent connections remain a retrospective outcome-qualified cohort.'}
    (output / 'preparation_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
