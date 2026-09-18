import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {Store,applyPreset} from '../store.mjs';

test('fresh cache stays inside the persistent data root; existing settings retain their path',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-cache-default-'));const s=new Store(root);
 try{
  assert.equal(s.settings().cacheRoot,path.join(root,'cache'));
  s.put('settings','main',{cacheRoot:'/existing/cache',modelRoot:'/existing/models'});
  assert.equal(s.settings().cacheRoot,'/existing/cache');
 }finally{s.close();fs.rmSync(root,{recursive:true,force:true});}
});
test('settings and versioned personal templates survive reopening; snapshots stay independent',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-store-'));let s=new Store(root);
 try{const p={id:'fp8',args:['model','--custom','preserve'],env:{ORIGINAL:'yes'}};
 s.saveSettings({defaultProfile:'fp8',autoStart:true},[p]);const t=s.saveTemplate('个人',p);p.args[2]='changed';assert.equal(s.get('template',t.id).profile.args[2],'preserve');
 s.saveTemplate('第二版',p,t.id);s.close();s=new Store(root);assert.equal(s.settings().autoStart,true);assert.equal(s.get('template',t.id).revision,2);assert.equal(s.get('template-version',t.id+':1').profile.args[2],'preserve');
 assert.throws(()=>s.saveSettings({defaultProfile:'missing'},[p]));
 }finally{s.close();fs.rmSync(root,{recursive:true,force:true});}
});
test('preset keeps draft, CPU KV and unknown arguments',()=>{const args=['model','--speculative-config','{"method":"dflash"}','--kv-transfer-config','{"x":1}','--max-num-seqs','4'];const result=applyPreset(args,'single-long');assert.deepEqual(result.slice(0,5),args.slice(0,5));assert.equal(result[result.indexOf('--max-num-seqs')+1],'1');assert.equal(args[6],'4');});
