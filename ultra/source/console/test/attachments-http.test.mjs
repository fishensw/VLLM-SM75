import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {fileURLToPath} from 'node:url';

test('HTTP attachment access requires login and local validated references before model forwarding',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-attachment-http-'));
 const modelDir=path.join(root,'model');fs.mkdirSync(modelDir);fs.writeFileSync(path.join(modelDir,'config.json'),JSON.stringify({vision_config:{}}));
 const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=','base64');
 let forwarded=null;
 const engine=http.createServer(async(req,res)=>{let payload='';for await(const part of req)payload+=part;forwarded=JSON.parse(payload);res.writeHead(200,{'Content-Type':'text/event-stream'});res.end('data: [DONE]\n\n');});
 await new Promise(resolve=>engine.listen(0,'127.0.0.1',resolve));
 const base='http://127.0.0.1:19199';
 const child=spawn(process.execPath,[fileURLToPath(new URL('../server.mjs',import.meta.url))],{env:{...process.env,SM75_CONSOLE_ROOT:root,SM75_CONSOLE_HOST:'127.0.0.1',SM75_CONSOLE_PORT:'19199',SM75_HARNESS_PROXY_PORT:'19197',SM75_SINGLE_CONTAINER:'0',DSH_HOME:path.join(root,'dsh')},stdio:'ignore'});
 try {
  let ready=false;for(let i=0;i<100;i++){try{ready=(await fetch(base+'/')).ok;if(ready)break;}catch{}await new Promise(resolve=>setTimeout(resolve,30));}assert(ready);
  const token=fs.readFileSync(path.join(root,'key'),'utf8');
  const login=await fetch(base+'/console-api/login',{method:'POST',headers:{Origin:base,'Content-Type':'application/json'},body:JSON.stringify({token})});
  const headers={Cookie:login.headers.get('set-cookie').split(';')[0],Origin:base};
  const post=(url,data)=>fetch(base+url,{method:'POST',headers:{...headers,'Content-Type':'application/json'},body:JSON.stringify(data)});
  const profile={id:'image',port:engine.address().port,args:[modelDir],format:'fp8',cacheRoot:'/tmp/sm75-image-test',backend:'docker'};
  assert.equal((await post('/console-api/profiles',profile)).status,200);
  const endpoint='/console-api/attachments?profile=image';
  assert.equal((await fetch(base+endpoint,{method:'POST',body:png})).status,401);
  assert.equal((await fetch(base+endpoint,{method:'POST',headers:{...headers,Origin:'https://untrusted.invalid'},body:png})).status,403);
  const upload=await fetch(base+endpoint,{method:'POST',headers,body:png});assert.equal(upload.status,200);
  const attachment=await upload.json();const get='/console-api/attachments/'+attachment.id;
  assert.equal((await fetch(base+get)).status,401);
  const download=await fetch(base+get,{headers});assert.equal(download.status,200);assert.equal(download.headers.get('x-content-type-options'),'nosniff');assert.deepEqual(Buffer.from(await download.arrayBuffer()),png);
  const remote=await post('/console-api/profiles/image/chat',{model:'image',messages:[{role:'user',content:[{type:'image_url',image_url:{url:'https://example.invalid/private'}}]}]});
  assert.equal(remote.status,400);assert.equal(forwarded,null);
  const messages=[{role:'user',content:[{type:'image',attachmentId:attachment.id}]}];
  assert.equal((await post('/console-api/chats/image',{messages})).status,200);
  const response=await post('/console-api/profiles/image/chat',{model:'image',messages});assert.equal(response.status,200);await response.text();
  assert.equal(forwarded.messages[0].content[0].image_url.url,'data:image/png;base64,'+png.toString('base64'));
  assert.equal((await post('/console-api/profiles',{...profile,args:[modelDir,'--language-model-only']})).status,200);
  assert.equal((await fetch(base+endpoint,{method:'POST',headers,body:png})).status,422);
  assert.equal((await post('/console-api/profiles/image/chat',{model:'image',messages})).status,400);
  assert.equal((await post('/console-api/logout',{})).status,200);
  assert.equal((await fetch(base+get,{headers})).status,401);
 }finally{
  if(child.exitCode===null){const exited=once(child,'exit');child.kill();await exited;}
  await new Promise(resolve=>engine.close(resolve));fs.rmSync(root,{recursive:true,force:true});
 }
});