"""下载项目天气档案；已保存的文件直接复用。/ Download and reuse weather snapshots."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START = '2026-03-30T00:00:00Z'
END = '2026-09-02T00:00:00Z'
FIELDS = ['tmpf','dwpf','relh','drct','sknt','gust','vsby','p01i','skyc1','skyc2','skyc3','skyc4','skyl1','skyl2','skyl3','skyl4','wxcodes','metar']


def jobs():
    airports = pd.read_csv(ROOT/'references/airports.csv')
    stations = sorted(set(airports.metar_station.dropna()) | {'PBI'})
    tasks=[]
    for offset in range(0,len(stations),20):
        group=stations[offset:offset+20]
        for kind in ['metar','taf']:
            if kind=='metar':
                params=[('station',s) for s in group]+[('sts',START),('ets',END),('tz','UTC'),('format','onlycomma'),('missing','M'),('trace','T'),('report_type','3'),('report_type','4')]+[('data',s) for s in FIELDS]
                url='https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'+urlencode(params)
            else:
                url='https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py?'+urlencode(dict(station=','.join(group),sts=START,ets=END,fmt='csv',tz='UTC'))
            tasks.append(dict(kind=kind,file=f'raw/weather/{kind}/batch_{offset//20+1:02}.csv.gz',stations=group,url=url))
    return tasks


def download(item):
    path=ROOT/item['file'];path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        part=path.with_suffix('.part')
        with tempfile.TemporaryDirectory() as folder:
            raw=Path(folder)/'response.csv'
            subprocess.run(['curl','-L','--fail','--retry','3','--retry-all-errors','--retry-delay','3','--connect-timeout','30','--max-time','480','-sS',item['url'],'-o',str(raw)],check=True)
            with raw.open(encoding='utf-8-sig',newline='') as stream:
                header=next(csv.reader(stream))
                if not {'station','valid'}.issubset(header): raise ValueError(f"{item['file']}: unexpected CSV header {header[:4]}")
            with raw.open('rb') as src,part.open('wb') as target:
                with gzip.GzipFile(fileobj=target,mode='wb',mtime=0) as compressed: shutil.copyfileobj(src,compressed)
            part.replace(path)
    d=pd.read_csv(path,usecols=['station','valid'],dtype=str)
    return {**item,'rows':len(d),'returned_stations':sorted(d.station.unique()),'first_utc':d.valid.min(),'last_utc':d.valid.max(),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize(records):
    airports=pd.read_csv(ROOT/'references/airports.csv')
    lookups={}
    for r in airports.itertuples():
        for station in [r.metar_station,r.icao,*str(r.airport_aliases).split('|')]:
            lookups[station]=r.airport_id
            if len(station)==3: lookups['K'+station]=r.airport_id
    pieces=[]
    for item in records:
        d=pd.read_csv(ROOT/item['file'],usecols=['station','valid'],dtype=str)
        d['airport_id']=d.station.map(lookups)
        if d.airport_id.isna().any(): raise ValueError(f"Unmapped returned stations: {d.loc[d.airport_id.isna(),'station'].unique()}")
        d['month']=d.valid.str[:7];d['day']=d.valid.str[:10]
        d=d[d.month.between('2026-04','2026-08')]
        d['kind']=item['kind'];pieces.append(d[['airport_id','month','kind','valid','day']])
    coverage=pd.concat(pieces).groupby(['airport_id','month','kind']).agg(rows=('valid','size'),days=('day','nunique'),first_utc=('valid','min'),last_utc=('valid','max')).reset_index()
    index=pd.MultiIndex.from_product([airports.airport_id,[f'2026-{m:02}' for m in range(4,9)],['metar','taf']],names=['airport_id','month','kind'])
    coverage=coverage.set_index(['airport_id','month','kind']).reindex(index).reset_index()
    coverage[['rows','days']]=coverage[['rows','days']].fillna(0).astype(int)
    coverage=coverage.merge(airports[['airport_id','airport_code','metar_station','taf_station']],on='airport_id',validate='many_to_one')
    coverage.to_csv(ROOT/'tables/weather_coverage.csv',index=False)
    totals={kind:{'raw_rows_with_buffer':sum(r['rows'] for r in records if r['kind']==kind),'airports_with_records':int(coverage.loc[(coverage.kind==kind)&coverage.rows.gt(0),'airport_id'].nunique()),'airports_all_five_months':int(coverage.loc[(coverage.kind==kind)&coverage.rows.gt(0)].groupby('airport_id').size().eq(5).sum())} for kind in ['metar','taf']}
    summary={'retrieved_date_utc':datetime.now(timezone.utc).isoformat(),'flight_rows':3057590,'airport_ids':355,'flight_months':['2026-04','2026-05','2026-06','2026-07','2026-08'],'weather_start_utc':START,'weather_end_exclusive_utc':END,'weather':totals,'weather_files':len(records),'training_executed':False,'full_flight_cleaning_executed':False}
    (ROOT/'tables/download_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def run():
    tasks=jobs();records=[];failures=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(download,item):item for item in tasks}
        for future in as_completed(futures):
            item=futures[future]
            try:
                result=future.result();records.append(result)
                print(item['file'],result['rows'],'rows',len(result['returned_stations']),'stations',flush=True)
            except Exception as exc:
                failures.append({'file':item['file'],'error':str(exc)});print('FAILED',item['file'],str(exc),flush=True)
    records.sort(key=lambda r:r['file'])
    (ROOT/'tables/weather_download_manifest.json').write_text(json.dumps({'files':records,'failures':failures},ensure_ascii=False,indent=2),encoding='utf-8')
    if failures: raise RuntimeError(f'{len(failures)} downloads failed. Rerun to retry only missing files.')
    print(json.dumps(summarize(records),ensure_ascii=False,indent=2))

if __name__=='__main__': run()
