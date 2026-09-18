import hashlib,json,pathlib,subprocess,sys,urllib.request
root=pathlib.Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
commit='eb19940368f0d811ef7b14f77f61d1c7e8676c7a'
data=urllib.request.urlopen('https://raw.githubusercontent.com/gengchaogit/llm_speedtest/'+commit+'/python/llm_test_backend.py',timeout=60).read()
(root/'llm_test_backend.py').write_bytes(data)
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-cache-dir','--target',str(root/'deps'),'slowapi==0.1.9'],check=True)
(root/'manifest.json').write_text(json.dumps({'project':'gengchaogit/llm_speedtest','commit':commit,'sha256':hashlib.sha256(data).hexdigest()}))
print('测试组件部署完成',flush=True)

for name in ['python/LLM_Speed_Test_v3_Python_Backend.html','LLM_Speed_Test_v3_Leaderboard.html']:
 data=urllib.request.urlopen('https://raw.githubusercontent.com/gengchaogit/llm_speedtest/'+commit+'/'+name,timeout=120).read()
 (root/pathlib.Path(name).name).write_bytes(data)
manifest=json.loads((root/'manifest.json').read_text());manifest['original_ui']=1;(root/'manifest.json').write_text(json.dumps(manifest))
