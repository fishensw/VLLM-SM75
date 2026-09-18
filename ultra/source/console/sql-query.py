import json,pathlib,sys,datetime,decimal,threading
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root/'deps'));import duckdb
con=duckdb.connect(str(root/'benchmark.duckdb'),read_only=True,config={'enable_external_access':False,'memory_limit':'256MB','threads':2})
sql=sys.stdin.read();statements=con.extract_statements(sql)
if len(statements)!=1 or statements[0].type!=duckdb.StatementType.SELECT:raise ValueError('测试数据库仅允许单条 SELECT 查询')
timer=threading.Timer(10,con.interrupt);timer.daemon=True;timer.start()
try:
 result=con.execute(sql);cols=[x[0] for x in result.description];rows=result.fetchmany(100001)
 if len(rows)>100000:raise ValueError('结果超过 100000 行')
 data=[dict(zip(cols,[None if x is None else str(x).lower() if isinstance(x,bool) else str(x) for x in row])) for row in rows]
 print(json.dumps({'columns':cols,'rows':data,'numRows':len(rows)}))
finally:timer.cancel();con.close()
