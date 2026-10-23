"""Aircraft granularity and event-time airport operations for the shared builder."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AIRCRAFT_COLUMNS = ['aircraft_family', 'aircraft_variant']
AIRPORT_NUMERIC = ['airport_departures_60m_n', 'airport_departures_60m_delay_mean',
                   'airport_planned_departures_next60m_n', 'airport_planned_arrivals_next60m_n',
                   'airport_history_window_incomplete']
AIRPORT_SOURCE_COLUMNS = ['flight_id', 'OriginAirportID', 'DestAirportID', 'scheduled_dep_utc',
                          'scheduled_arr_utc', 'actual_dep_utc', 'actual_dep_clock_error',
                          'DepDelay', 'DepDelayMinutes', 'duplicate_key']


def augment_aircraft(frame, reference, aliases, inventory_year=2024):
    """Join by airline+tail, preserving unknowns and rejecting ambiguous mappings."""
    result = frame.copy()
    ref = reference[['airline', 'tail_number', 'aircraft_model']].copy()
    for column in ref:
        ref[column] = ref[column].astype('string').str.strip().replace('', pd.NA)
    ref = ref.dropna(subset=['airline', 'tail_number']).drop_duplicates()
    conflicts = ref.groupby(['airline', 'tail_number']).aircraft_model.nunique().gt(1)
    if conflicts.any():
        raise ValueError('Aircraft reference has conflicting models for an airline-tail key')
    ref = ref.drop_duplicates(['airline', 'tail_number'])
    mapping = aliases.rename(columns={'aircraft_model_raw': 'aircraft_model'})
    if mapping.aircraft_model.duplicated().any():
        raise ValueError('Aircraft alias labels must be unique')
    ref = ref.merge(mapping[['aircraft_model', *AIRCRAFT_COLUMNS]], on='aircraft_model',
                    how='left', validate='many_to_one')
    keys = pd.MultiIndex.from_arrays([frame.downstream_airline.astype('string').str.strip(),
                                      frame.tail_number.astype('string').str.strip()])
    lookup = ref.set_index(['airline', 'tail_number'])
    for column in AIRCRAFT_COLUMNS:
        result[column] = lookup[column].reindex(keys).to_numpy(dtype=object)
        result[column] = result[column].where(result[column].notna(), np.nan)
    result['aircraft_inventory_year'] = inventory_year
    result.attrs['aircraft_audit'] = {
        'inventory_year': inventory_year, 'join_keys': ['downstream_airline', 'tail_number'],
        'rows': len(frame), 'family_matched': int(result.aircraft_family.notna().sum()),
        'variant_matched': int(result.aircraft_variant.notna().sum()),
        'unknown_preserved': True,
        'limitation': 'Static B43 inventory snapshot, not the historical fleet composition at each prediction time',
        'linear_rule': 'Use family OR variant, never both nested categories in one design matrix',
    }
    return result


def _event_windows(queries, events, airport_column, time_column, before_minutes, after_minutes, value_column=None):
    """Count [t-before,t+after); include neither future actual events nor end edge."""
    columns = [airport_column, time_column] + ([value_column] if value_column else [])
    source = events[columns].dropna(subset=[airport_column, time_column]).copy()
    source['airport'] = source[airport_column].astype('Int64').astype('string')
    source['timestamp'] = pd.to_datetime(source[time_column], utc=True).astype('datetime64[ns, UTC]')
    source = source.sort_values(['airport', 'timestamp'], kind='stable').reset_index(drop=True)
    event_groups = source.groupby('airport', sort=False).indices
    query_airport = queries.airport_id.astype('string')
    query_time = pd.to_datetime(queries.prediction_time, utc=True).astype('datetime64[ns, UTC]').array.asi8
    query_groups = pd.DataFrame({'airport': query_airport.to_numpy()}).groupby('airport', sort=False).indices
    counts, means = np.full(len(queries), np.nan), np.full(len(queries), np.nan)
    times = source.timestamp.array.asi8
    before = pd.Timedelta(minutes=before_minutes).value
    after = pd.Timedelta(minutes=after_minutes).value
    for airport, positions in query_groups.items():
        counts[positions] = 0
        ix = event_groups.get(airport)
        if ix is None:
            continue
        event_times = times[ix]
        left = np.searchsorted(event_times, query_time[positions] - before, side='left')
        right = np.searchsorted(event_times, query_time[positions] + after, side='left')
        n = right - left
        counts[positions] = n
        if value_column:
            prefix = np.r_[0., source[value_column].to_numpy(dtype=float)[ix].cumsum()]
            means[positions] = np.divide(prefix[right] - prefix[left], n,
                                          out=np.full(len(n), np.nan), where=n > 0)
    return counts, means


def augment_airport_operations(frame, flights, source_start_utc='2026-04-01T00:00:00Z'):
    """Departure-event quality uses only fields known at departure, not future arrival."""
    result = frame.copy()
    if set(AIRPORT_SOURCE_COLUMNS) - set(flights):
        raise ValueError(f'Missing airport event fields: {set(AIRPORT_SOURCE_COLUMNS) - set(flights)}')
    if flights.flight_id.isna().any() or flights.flight_id.duplicated().any():
        raise ValueError('Airport source flight_id must be present and unique')
    source = flights[AIRPORT_SOURCE_COLUMNS].copy()
    for column in ['scheduled_dep_utc', 'scheduled_arr_utc', 'actual_dep_utc']:
        source[column] = pd.to_datetime(source[column], utc=True, errors='coerce')
    # A later diverted/abnormal arrival must not retroactively remove a valid departure.
    dep_delay = pd.to_numeric(source.DepDelayMinutes, errors='coerce')
    dep_signed = pd.to_numeric(source.DepDelay, errors='coerce')
    duplicate = source.duplicate_key.astype('string').str.lower().isin(['true', '1'])
    max_query = pd.to_datetime(frame.prediction_time, utc=True).max()
    history_ok = (source.actual_dep_utc.lt(max_query) & np.isfinite(dep_delay) & dep_delay.ge(0)
                  & (dep_signed.clip(lower=0) - dep_delay).abs().le(1)
                  & source.actual_dep_clock_error.le(1) & ~duplicate)
    source['departure_delay'] = dep_delay
    departures = source.loc[history_ok]
    n, mean = _event_windows(frame, departures, 'OriginAirportID', 'actual_dep_utc', 60, 0, 'departure_delay')
    result['airport_departures_60m_n'] = n
    result['airport_departures_60m_delay_mean'] = mean
    # Full archived plan is a declared availability assumption; cancellations are
    # not removed by their eventual outcome. This prevents future operational leakage.
    plans = source.loc[~duplicate]
    result['airport_planned_departures_next60m_n'] = _event_windows(
        frame, plans, 'OriginAirportID', 'scheduled_dep_utc', 0, 60)[0]
    result['airport_planned_arrivals_next60m_n'] = _event_windows(
        frame, plans, 'DestAirportID', 'scheduled_arr_utc', 0, 60)[0]
    result['airport_history_window_incomplete'] = (
        pd.to_datetime(frame.prediction_time, utc=True) - pd.Timedelta(minutes=60)
    ).lt(pd.Timestamp(source_start_utc)).to_numpy(dtype=np.int8)
    result.attrs['airport_operations_audit'] = {
        'historical_window': '[prediction_time-60min, prediction_time)',
        'scheduled_window': '[prediction_time, prediction_time+60min)',
        'historical_departures': len(departures), 'planned_rows': len(plans),
        'historical_quality': 'Departure clock/delay consistency and unique source key only; no arrival/outcome/full-tail QA',
        'plan_assumption': 'Archived BTS scheduled times are treated as known; historical schedule revisions are unavailable',
        'no_event_mean': 'NA; count is zero, training-only imputation',
        'left_window_incomplete_queries': int(result.airport_history_window_incomplete.sum()),
    }
    return result
