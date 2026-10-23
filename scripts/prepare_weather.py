"""整理机场实况和预报分段；不运行航班模型。/ Prepare observation and forecast tables."""
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def station_lookup():
    airports=pd.read_csv(ROOT/'references/airports.csv')
    lookup={}
    for row in airports.itertuples():
        for station in [row.metar_station,row.icao,*row.airport_aliases.split('|')]:
            lookup[station]=int(row.airport_id)
            if len(station)==3: lookup['K'+station]=int(row.airport_id)
    return lookup


def read_batches(kind):
    files=sorted((ROOT/'raw/weather'/kind).glob('batch_*.csv.gz'))
    if not files: raise FileNotFoundError(f'No {kind} files in raw/weather/{kind}')
    d=pd.concat([pd.read_csv(f,low_memory=False) for f in files],ignore_index=True)
    d['airport_id']=d.station.map(station_lookup())
    assert d.airport_id.notna().all(), 'A weather station has no airport ID'
    d['airport_id']=d.airport_id.astype('int64')
    return d


def ceiling(codes,heights):
    values=[float(h) for c,h in zip(codes,heights) if c in {'BKN','OVC','VV'} and h is not None and pd.notna(h)]
    return min(values) if values else np.nan


def parsed_list(value):
    if pd.isna(value) or value=='': return []
    return ast.literal_eval(value)


def prepare_metar():
    d=read_batches('metar')
    d['weather_time']=pd.to_datetime(d.valid,utc=True)
    for col in ['sknt','gust','vsby','tmpf','dwpf','relh','drct','p01i','skyl1','skyl2','skyl3','skyl4']:
        d[col+'_num']=pd.to_numeric(d[col],errors='coerce')
    d['ceiling_ft']=[ceiling(c,h) for c,h in zip(d[['skyc1','skyc2','skyc3','skyc4']].values,d[['skyl1_num','skyl2_num','skyl3_num','skyl4_num']].values)]
    d['sky_reported']=d[['skyc1','skyc2','skyc3','skyc4']].apply(lambda s:s.notna()&s.ne('M')).any(axis=1).astype('int8')
    d['ceiling_reported']=d.ceiling_ft.notna().astype('int8')
    codes=d.wxcodes.fillna('').astype(str)
    d['thunder_flag']=codes.str.contains('TS',regex=False).astype('int8')
    d['precip_flag']=(d.p01i_num.gt(0)|d.p01i.astype(str).eq('T')|codes.str.contains('RA|DZ|SN|SG|PL|GR|GS|UP',regex=True)).astype('int8')
    rename={'sknt_num':'wind_knots','gust_num':'gust_knots','vsby_num':'visibility_miles','tmpf_num':'temperature_f','dwpf_num':'dewpoint_f','relh_num':'relative_humidity','drct_num':'wind_direction_deg'}
    d=d.rename(columns=rename)
    # The archive supplies observation time; five minutes is an explicit availability assumption.
    d['available_at']=d.weather_time+pd.Timedelta(minutes=5)
    d['_complete']=d[['wind_knots','visibility_miles','temperature_f','ceiling_ft']].notna().sum(axis=1)
    d=d.sort_values(['airport_id','weather_time','_complete']).drop_duplicates(['airport_id','weather_time'],keep='last')
    keep=['airport_id','station','weather_time','available_at','wind_knots','wind_direction_deg','gust_knots','visibility_miles','ceiling_ft','ceiling_reported','sky_reported','precip_flag','thunder_flag','temperature_f','dewpoint_f','relative_humidity','metar']
    return d[keep].sort_values(['weather_time','airport_id'])


def prepare_taf():
    d=read_batches('taf').drop_duplicates(['airport_id','station','product_id','valid','fx_valid','ftype','raw'])
    validity_path=ROOT/'references/taf_validity.csv'
    if not validity_path.exists(): raise FileNotFoundError('Run scripts/download_taf_text.py before preparing TAF')
    bulletins=pd.read_csv(validity_path)
    for col in ['issue_time','valid_from','valid_to']:
        bulletins[col]=pd.to_datetime(bulletins[col],utc=True,errors='coerce')
    bulletins['airport_id']=bulletins.station.map(station_lookup())
    bulletins=bulletins[bulletins.airport_id.notna()].copy()
    bulletins['airport_id']=bulletins.airport_id.astype('int64')
    bulletins=bulletins.drop_duplicates(['product_id','station'],keep='last')
    d=d.merge(bulletins[['product_id','station','issue_time','valid_from','valid_to','status']],on=['product_id','station'],how='left',validate='many_to_one')
    d['archive_issue_time']=pd.to_datetime(d.valid,utc=True)
    d['segment_start']=pd.to_datetime(d.fx_valid,utc=True)
    d['segment_end']=pd.to_datetime(d.fx_valid_end,utc=True,errors='coerce')
    initial=d.ftype.eq('Observation')
    # IEM calls the opening prevailing forecast "Observation"; it remains a forecast.
    d.loc[initial,'segment_start']=d.loc[initial,'valid_from']
    prevailing=d.ftype.isin(['Observation','Forecast'])
    base=d.loc[prevailing].sort_values(['product_id','station','segment_start'])
    next_start=base.groupby(['product_id','station']).segment_start.shift(-1)
    d.loc[base.index,'segment_end']=next_start.fillna(base.valid_to)
    d['segment_start']=d[['segment_start','valid_from']].max(axis=1)
    d['segment_end']=d[['segment_end','valid_to']].min(axis=1)
    d['validity_matched']=d.valid_to.notna()&d.valid_from.notna()&d.issue_time.notna()
    d['interval_usable']=d.validity_matched & d.segment_end.gt(d.segment_start)&d.status.eq('normal')
    d['ceiling_ft']=[ceiling(parsed_list(c),parsed_list(h)) for c,h in zip(d.skyc,d.skyl)]
    wx=d.presentwx.fillna('').astype(str)
    d['thunder_flag']=wx.str.contains('TS',regex=False).astype('int8')
    d['precip_flag']=wx.str.contains('RA|DZ|SN|SG|PL|GR|GS|UP',regex=True).astype('int8')
    d=d.rename(columns={'sknt':'wind_knots','drct':'wind_direction_deg','gust':'gust_knots','visibility':'visibility_miles'})
    keep=['airport_id','station','product_id','issue_time','archive_issue_time','valid_from','valid_to','segment_start','segment_end','ftype','is_amendment','status','validity_matched','interval_usable','wind_knots','wind_direction_deg','gust_knots','visibility_miles','ceiling_ft','precip_flag','thunder_flag','raw']
    return d[keep],bulletins


def run():
    out=ROOT/'processed';out.mkdir(exist_ok=True)
    metar=prepare_metar();taf,bulletins=prepare_taf()
    compression={'method':'gzip','compresslevel':3,'mtime':0}
    for name,d in [('metar_observations',metar),('taf_segments',taf),('taf_bulletins',bulletins)]:
        d.to_csv(out/f'{name}.csv.gz',index=False,compression=compression)
    summary={'metar_rows':len(metar),'metar_airports':int(metar.airport_id.nunique()),'taf_segment_rows':len(taf),'taf_airports':int(taf.airport_id.nunique()),'taf_validity_matched_rows':int(taf.validity_matched.sum()),'taf_usable_interval_rows':int(taf.interval_usable.sum()),'taf_bulletins':len(bulletins),'metar_assumed_lag_minutes':5,'metar_max_observation_age_minutes':120}
    (ROOT/'tables/weather_preparation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
