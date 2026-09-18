"""Launch upstream applications unchanged; adapt only local transport/auth and bind."""
import os


def _engine_key():
    p = os.environ.get('SM75_ENGINE_KEY_FILE')
    if p:
        try:
            k = open(p, encoding='utf-8').read().strip()
            if k:
                return k
        except OSError:
            pass
    return os.environ.get('SM75_ENGINE_KEY', '')
import sys,pathlib,importlib.util,urllib.parse,http.server
root=pathlib.Path(sys.argv[2]);sys.path.insert(0,str(root/'deps'));sys.path.insert(0,str(root));os.chdir(root)
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,str(root/file));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
if sys.argv[1]=='sql':
 m=load('sql_original','run_server.py')
 http.server.ThreadingHTTPServer(('127.0.0.1',18002),m.CORSHTTPRequestHandler).serve_forever()
else:
 import httpx,uvicorn
 original=httpx.AsyncClient.send
 async def send(self,request,*a,**kw):
  if request.headers.get('authorization')=='Bearer sm75-managed' and request.url.port==8000 and request.url.host in ('127.0.0.1','localhost',os.environ.get('SM75_CONTAINER_IP','')):
   request.url=request.url.copy_with(host='127.0.0.1');request.headers['authorization']='Bearer '+_engine_key()
  return await original(self,request,*a,**kw)
 httpx.AsyncClient.send=send
 m=load('speed_original','llm_test_backend.py');m.current_port=18001
 uvicorn.run(m.app,host='127.0.0.1',port=18001,log_level='warning')
