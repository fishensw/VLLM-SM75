import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';import os from 'node:os';import path from 'node:path';import http from 'node:http';import {spawn} from 'node:child_process';import {fileURLToPath} from 'node:url';
const delay=ms=>new Promise(r=>setTimeout(r,ms));
test('console login, strict profile isolation, and streaming Chat proxy',async()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-console-test-'));let received;
 const mock=http.createServer(async(req,res)=>{let body='';for await(const c of req)body+=c;received=JSON.parse(body);res.writeHead(200,{'Content-Type':'text/event-stream'});res.write('data: {"choices":[{"delta":{"content":"391"}}]}\n\n');res.end('data: [DONE]\n\n');});
 await new Promise(r=>mock.listen(0,'127.0.0.1',r));const mockPort=mock.address().port;
 const child=spawn(process.execPath,[fileURLToPath(new URL('../server.mjs',import.meta.url))],{env:{...process.env,SM75_SINGLE_CONTAINER:'0',SM75_CONSOLE_ROOT:dir,SM75_CONSOLE_HOST:'127.0.0.1',SM75_CONSOLE_PORT:'19165',SM75_HARNESS_PROXY_PORT:'19167'},stdio:'ignore'});
 try{let up=false;for(let i=0;i<80;i++){try{await fetch('http://127.0.0.1:19165/');up=true;break;}catch{await delay(50);}}assert(up,'test server starts');const base='http://127.0.0.1:19165';assert.equal((await fetch(base+'/console-api/profiles')).status,401);
 const token=fs.readFileSync(path.join(dir,'key'),'utf8');const h={'Content-Type':'application/json',Authorization:'Bearer '+token};const login=await fetch(base+'/console-api/login',{method:'POST',headers:h,body:JSON.stringify({token})});assert.equal(login.status,200);assert.match(login.headers.get('set-cookie'),/HttpOnly/);
 const profile={id:'test',port:mockPort,args:['test-model'],format:'fp8',cacheRoot:'/tmp/sm75-test-cache',backend:'docker'};
 assert.equal((await fetch(base+'/console-api/profiles',{method:'POST',headers:{...h,Origin:'http://untrusted.invalid'},body:JSON.stringify(profile)})).status,403);
 assert.equal((await fetch(base+'/console-api/profiles',{method:'POST',headers:h,body:JSON.stringify(profile)})).status,200);
 const chat=await fetch(base+'/console-api/profiles/test/chat',{method:'POST',headers:h,body:JSON.stringify({model:'test-model',messages:[{role:'user',content:'17*23'}]})});assert.equal(chat.status,200);assert.match(await chat.text(),/391/);assert.equal(received.stream,true);assert.equal(received.stream_options.include_usage,true);

 const cold={...profile,id:'cold',port:1};await fetch(base+'/console-api/profiles',{method:'POST',headers:h,body:JSON.stringify(cold)});const unavailable=await fetch(base+'/console-api/profiles/cold/models',{headers:h});assert.equal(unavailable.status,400);assert.equal((await fetch(base+'/console-api/profiles',{headers:h})).status,200,'unavailable model must not terminate manager');
 const extension=await(await fetch(base+'/console-api/benchmarks/extension',{headers:h})).json();assert.equal(extension.enabled,false);assert.equal(extension.installed,false);
 const blocked=await fetch(base+'/console-api/benchmarks',{method:'POST',headers:h,body:JSON.stringify({profile:'test'})});assert.equal(blocked.status,400);assert.match((await blocked.json()).error,/开启/);
 assert.deepEqual(await(await fetch(base+'/console-api/benchmarks',{headers:h})).json(),[]);
 assert.equal(fs.existsSync(path.join(dir,'extensions')),false,'disabled extension does not install dependencies');
 const fixture=path.join(dir,'delete-fixture');fs.mkdirSync(fixture);fs.writeFileSync(path.join(fixture,'config.json'),'{}');
 const registered=await(await fetch(base+'/console-api/models/register',{method:'POST',headers:h,body:JSON.stringify({path:fixture})})).json();
 assert.equal((await fetch(base+'/console-api/models/delete',{method:'POST',headers:h,body:JSON.stringify({id:registered.id,path:dir})})).status,400);
 assert.equal(fs.existsSync(fixture),true);
 assert.equal((await fetch(base+'/console-api/models/delete',{method:'POST',headers:h,body:JSON.stringify({id:registered.id,path:fixture})})).status,200);
 assert.equal(fs.existsSync(fixture),false);
 const monitor=await fetch(base+'/console-api/profiles/test/monitor',{headers:h});assert.equal(monitor.status,200);assert.match(await monitor.text(),/SM75_HISTORY_URL='\/console-api\/profiles\/test\/history'/);
 const messages=[{role:'user',content:'保存会话'},{role:'assistant',content:'已保存'}];
 assert.equal((await fetch(base+'/console-api/chats/test',{method:'POST',headers:h,body:JSON.stringify({messages})})).status,200);
 assert.deepEqual((await(await fetch(base+'/console-api/chats/test',{headers:h})).json()).messages,messages);
 assert.deepEqual(JSON.parse(fs.readFileSync(path.join(dir,'chats/test.json'))).messages,messages);
 assert.equal((await fetch(base+'/console-api/chats/unknown',{headers:h})).status,400);
 assert.equal((await fetch(base+'/console-api/chats/test',{method:'POST',headers:h,body:JSON.stringify({messages:[{role:'tool',content:'invalid'}]})})).status,400);
 assert.deepEqual((await(await fetch(base+'/console-api/profiles/test/history',{headers:h})).json()).points,[]);
 }finally{child.kill();await new Promise(r=>child.once('exit',r));await new Promise(r=>mock.close(r));}
});
