import pathlib,sys,urllib.request,subprocess,json,hashlib
root=pathlib.Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
commit='2aa49062f411928acd02d90ac8dddffe1b77a00e';base='https://raw.githubusercontent.com/wangxian001/SQL_LLM_benchmark/'+commit+'/'
files={}
for name in ['sql_benchmark.html']+['tables/'+n+'.csv' for n in ['Customer','Date','Product','Reseller','Sales','Sales_Order','Sales_Territory']]:
 data=(root/name).read_bytes() if (root/name).exists() else urllib.request.urlopen(base+name,timeout=120).read();p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);files[name]=hashlib.sha256(data).hexdigest();print('已下载 '+name,flush=True)
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-cache-dir','--target',str(root/'deps'),'duckdb==1.4.4'],check=True)
sys.path.insert(0,str(root/'deps'));import duckdb
con=duckdb.connect(str(root/'benchmark.duckdb'))
for name in ['Customer','Date','Product','Reseller','Sales','Sales_Order','Sales_Territory']:
 con.execute('CREATE OR REPLACE TABLE "'+name+'" AS SELECT * FROM read_csv_auto(?)',[str(root/'tables'/(name+'.csv'))])
schema={}
for table,col,typ in con.execute("SELECT table_name,column_name,data_type FROM information_schema.columns WHERE table_schema='main' ORDER BY table_name,ordinal_position").fetchall():schema.setdefault(table,[]).append({'col':col,'type':typ})
con.close();(root/'schema.json').write_text(json.dumps(schema))
(root/'manifest.json').write_text(json.dumps({'project':'wangxian001/SQL_LLM_benchmark','commit':commit,'files':files,'executor':'DuckDB Python 1.4.4','scoring':'upstream row/column/first-row'}))

for name in ['run_server.py','wasm/duckdb-browser-mvp.worker.js','wasm/duckdb-browser-eh.worker.js','wasm/duckdb-mvp.wasm','wasm/duckdb-eh.wasm']:
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.write_bytes(urllib.request.urlopen(base+name,timeout=240).read())
manifest=json.loads((root/'manifest.json').read_text());manifest['original_ui']=1;(root/'manifest.json').write_text(json.dumps(manifest))
