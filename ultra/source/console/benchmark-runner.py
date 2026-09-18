"""Run pinned llm_speedtest core locally, without starting its server or cloud routes."""
import asyncio, importlib.util, json, os, pathlib, sys, time
root=pathlib.Path(sys.argv[1]); cfg=json.loads(pathlib.Path(sys.argv[2]).read_text())
sys.path.insert(0,str(root/'deps'))
spec=importlib.util.spec_from_file_location('speedtest',root/'llm_test_backend.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
# Apply the user's thinking setting to the request, without changing engine configuration.
original_post=mod.httpx.AsyncClient.stream
def stream(self,method,url,**kwargs):
    if isinstance(kwargs.get('json'),dict):kwargs['json']['chat_template_kwargs']={'enable_thinking':cfg['thinking']}
    return original_post(self,method,url,**kwargs)
mod.httpx.AsyncClient.stream=stream
async def main():
    rows=[];warmup=None
    if cfg.get('warmup',True):
        warmup=await mod.execute_single_request(cfg['url'],os.environ['SM75_BENCH_KEY'],'openai',cfg['model'],32,16,cfg['timeout']*1000,0,1,0,0,seed=time.time_ns()%1000000000)
        if not warmup.get('success'):raise RuntimeError('预热失败: '+str(warmup.get('error')))

    for length in cfg['lengths']:
        for repeat in range(cfg['repeats']):
            started=time.perf_counter()
            batch=await asyncio.gather(*[mod.execute_single_request(cfg['url'],os.environ['SM75_BENCH_KEY'],'openai',cfg['model'],length,cfg['output'],cfg['timeout']*1000,0,1,0,0,seed=time.time_ns()%1000000000+i) for i in range(cfg['concurrency'])])
            elapsed=time.perf_counter()-started
            for row in batch:row.update(repeat=repeat+1,group_seconds=elapsed,concurrency=cfg['concurrency'])
            rows.extend(batch)
            tmp=pathlib.Path(sys.argv[2]+'.results.tmp');tmp.write_text(json.dumps({'rows':rows,'warmup':warmup,'updated':time.time()}));tmp.replace(sys.argv[2]+'.results.json')
            print('SM75_PROGRESS '+json.dumps({'completed':len(rows),'total':len(cfg['lengths'])*cfg['repeats']*cfg['concurrency']}),flush=True)
    if any(not r.get('success') for r in rows):sys.exit(2)
asyncio.run(main())
