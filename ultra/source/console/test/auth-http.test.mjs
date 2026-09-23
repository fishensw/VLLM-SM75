import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {fileURLToPath} from 'node:url';

test('HTTP login survives process restart; downstream failure is not logout; logout revokes cookie',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-auth-http-'));
 const base='http://127.0.0.1:19188';let child;
 const start=async()=>{
  child=spawn(process.execPath,[fileURLToPath(new URL('../server.mjs',import.meta.url))],{env:{...process.env,SM75_CONSOLE_ROOT:root,SM75_CONSOLE_HOST:'127.0.0.1',SM75_CONSOLE_PORT:'19188',SM75_SINGLE_CONTAINER:'1',DSH_HOME:path.join(root,'dsh')},stdio:'ignore'});
  for(let i=0;i<100;i++){try{if((await fetch(base+'/')).ok)return;}catch{}await new Promise(r=>setTimeout(r,30));}
  throw Error('manager did not start');
 };
 const stop=async()=>{const exited=once(child,'exit');let timer;child.kill();try{await Promise.race([exited,new Promise((_,reject)=>{timer=setTimeout(()=>{child.kill('SIGKILL');reject(Error('graceful shutdown timed out'));},3000);})]);}finally{clearTimeout(timer);child=null;}};
 try{
  await start();const token=fs.readFileSync(path.join(root,'key'),'utf8');
  const login=async(origin)=>fetch(base+'/console-api/login',{method:'POST',headers:{'Content-Type':'application/json',Origin:origin},body:JSON.stringify({token})});
  assert.equal((await login('http://untrusted.invalid')).status,403);
  const response=await login(base);assert.equal(response.status,200);
  const cookie=response.headers.get('set-cookie');assert.match(cookie,/HttpOnly; SameSite=Strict/);assert.doesNotMatch(cookie,/Secure/);
  const headers={Cookie:cookie.split(';')[0]};
  const engineKey=fs.readFileSync(path.join(root,'api-access.json'),'utf8');assert.notEqual(JSON.parse(engineKey).key,token);
  const stream=await fetch(base+'/console-api/session/stream',{headers});assert.equal(stream.status,200);
  await stop();await stream.body.cancel().catch(()=>{});await start();
  assert.equal((await(await fetch(base+'/console-api/session',{headers})).json()).authenticated,true);
  assert.equal((await fetch(base+'/',{headers,redirect:'manual'})).status,200);
  const consoleHtml=await(await fetch(base+'/',{headers})).text();
  assert.doesNotMatch(consoleHtml, /id="menuToggle"|id="menuPanel"/);
  assert.match(consoleHtml, /id="workspaceSidebar"/);
  assert.match(consoleHtml, /id="root"/);
  assert.doesNotMatch(consoleHtml, /id="harnessFrame"|id="harnessSettings"/);
  assert.match(consoleHtml.split("</nav>")[0], /data-page="settings"/);
  const legacy=await fetch(base+'/dsh/',{headers,redirect:'manual'});assert.equal(legacy.status,308);assert.equal(legacy.headers.get('location'),'/');
  // Usage is a native Watcher page inside DSH; legacy standalone routes are gone.
  for (const suffix of ['/token-usage','/token-usage.json?range=all'])
    assert.equal((await fetch(base+suffix,{headers})).status,404);

  assert.equal((await(await fetch(base+'/console-api/session',{headers})).json()).authenticated,true);
  assert.equal((await fetch(base+'/brand/favicon.svg')).status,200);
  assert.equal((await fetch(base+'/live-summary.js')).status,200);
  assert.equal((await fetch(base+'/console-api/live-summary')).status,401);
  assert.equal((await fetch(base+'/console-api/live-summary?profile=missing',{headers})).status,404);
  const live = await fetch(base+'/console-api/live-summary',{headers});
  assert.equal(live.status,200);assert.equal(live.headers.get('cache-control'),'no-store');
  const summary = await live.json();assert.equal(summary.fresh,false);assert.equal(summary.decode,null);
  assert.equal(summary.running,false);assert.equal(summary.powerState,null);
  const security=async(payload,origin=base)=>fetch(base+'/console-api/security',{method:'POST',headers:{...headers,Origin:origin,'Content-Type':'application/json'},body:JSON.stringify(payload)});
  assert.equal((await fetch(base+'/console-api/security')).status,401);
  assert.equal((await security({action:'account',username:'admin',password:'test-admin-password-123'},'http://evil.invalid')).status,403);
  assert.equal((await security({action:'account',username:'admin',password:'test-admin-password-123'})).status,200);
  assert.equal((await login(base)).status,401);
  assert.equal((await(await fetch(base+'/console-api/session')).json()).accountConfigured,true);
  assert.equal((await security({action:'networks',networks:['127.0.0.1/32'],currentPassword:'test-admin-password-123'})).status,200);
  await stop();await start();
  assert.equal((await(await fetch(base+'/console-api/session',{headers})).json()).authenticated,true);
  const accountLogin=await fetch(base+'/console-api/login',{method:'POST',headers:{Origin:base,'Content-Type':'application/json'},body:JSON.stringify({username:'admin',password:'test-admin-password-123'})});
  assert.equal(accountLogin.status,200);
  assert.equal((await fetch(base+'/console-api/logout',{method:'POST',headers:{...headers,Origin:base}})).status,200);
  assert.equal((await fetch(base+'/console-api/profiles',{headers})).status,401);
  assert.equal(fs.readFileSync(path.join(root,'api-access.json'),'utf8'),engineKey);
 }finally{if(child)await stop();fs.rmSync(root,{recursive:true,force:true});}
});
