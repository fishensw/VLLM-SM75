import unittest,tempfile,pathlib,subprocess,json,os,sys
RUNNER=pathlib.Path(__file__).parents[1]/'benchmark-runner.py'
class RunnerContract(unittest.TestCase):
 def test_units_warmup_and_failure_retention(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d)
   (root/'llm_test_backend.py').write_text('''class Client:
 def stream(self,*args,**kwargs): return kwargs
class H: AsyncClient=Client
httpx=H()
async def execute_single_request(url,key,api,model,length,output,timeout,*args,**kwargs):
 assert timeout==600000, timeout
 assert httpx.AsyncClient().stream('POST',url,json={})['json']['chat_template_kwargs']['enable_thinking'] is False
 return {'success':length!=4096,'prompt_tokens':length,'output_tokens':output,'error':'fixture failure' if length==4096 else None}
''')
   f=root/'config.json';f.write_text(json.dumps(dict(url='http://local',model='test',lengths=[1024,4096],output=64,timeout=600,thinking=False,concurrency=2,repeats=1,warmup=True)))
   r=subprocess.run([sys.executable,str(RUNNER),str(root),str(f)],env={**os.environ,'SM75_BENCH_KEY':'fixture'},capture_output=True,text=True)
   self.assertEqual(r.returncode,2,r.stderr)
   result=json.loads(pathlib.Path(str(f)+'.results.json').read_text());self.assertTrue(result['warmup']['success']);self.assertEqual(len(result['rows']),4);self.assertEqual(sum(not x['success'] for x in result['rows']),2)
if __name__=='__main__':unittest.main()
