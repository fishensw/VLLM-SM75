import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';import os from 'node:os';import path from 'node:path';import {spawn} from 'node:child_process';import {fileURLToPath} from 'node:url';
const delay=ms=>new Promise(r=>setTimeout(r,ms));
test('native executable launch failure is returned rather than reported as started',{skip:process.platform!=='linux'},async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-native-failure-'));
 const child=spawn(process.execPath,[fileURLToPath(new URL('../server.mjs',import.meta.url))],{env:{...process.env,SM75_SINGLE_CONTAINER:'0',SM75_CONSOLE_ROOT:root,SM75_CONSOLE_HOST:'127.0.0.1',SM75_CONSOLE_PORT:'19275',SM75_HARNESS_PROXY_PORT:'19277',SM75_VLLM_BIN:path.join(root,'missing-vllm'),PATH:root},stdio:'ignore'});
 try{const base='http://127.0.0.1:19275';let ready=false;for(let i=0;i<80;i++){try{await fetch(base);ready=true;break;}catch{await delay(50);}}assert(ready);const headers={Authorization:'Bearer '+fs.readFileSync(path.join(root,'key'),'utf8').trim(),'Content-Type':'application/json'};const p={id:'native-failure',backend:'native',port:8029,args:['/models/test'],format:'fp8',cacheRoot:path.join(root,'cache')};assert.equal((await fetch(base+'/console-api/profiles',{method:'POST',headers,body:JSON.stringify(p)})).status,200);
 const preview=await(await fetch(base+'/console-api/profiles/native-failure/preview',{headers})).json();assert.equal(preview.bin,path.join(root,'missing-vllm'));
 const response=await fetch(base+'/console-api/profiles/native-failure/start',{method:'POST',headers,body:'{}'});const body=await response.json();assert(!response.ok);assert.match(body.error,/ENOENT/);assert(!body.started);assert.match(fs.readFileSync(path.join(root,'native-failure.log'),'utf8'),/ENOENT/);
 }finally{child.kill();await new Promise(r=>child.once('exit',r));}
});
