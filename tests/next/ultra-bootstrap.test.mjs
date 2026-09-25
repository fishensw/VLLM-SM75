import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {EventEmitter} from 'node:events';
import {initializeNext} from '../../next-ultra.mjs';
import {Store} from '../../ultra/source/console/store.mjs';
import {Standalone} from '../../ultra/source/console/standalone.mjs';
const consoleDir=fileURLToPath(new URL('../../ultra/source/console/',import.meta.url));
const selected=JSON.parse(fs.readFileSync(new URL('../../config/selected-config.json',import.meta.url)));
function temp(t) {const root=fs.mkdtempSync(path.join(os.tmpdir(),'next-panel-'));t.after(()=>fs.rmSync(root,{recursive:true,force:true}));return root;}
test('first start seeds tested args; panel owns launch and stop; restart preserves edits',async t=>{
 const root=temp(t);await initializeNext(root,consoleDir,selected);
 const file=path.join(root,'profiles.json');const [p]=JSON.parse(fs.readFileSync(file));
 assert.deepEqual(p.args,selected.args.slice(1));
 const store=new Store(root);assert.equal(store.settings().autoStart,true);assert.equal(store.settings().defaultProfile,p.id);
 store.put('settings','main',{...store.settings(),autoStart:false});store.close();
 const manager=new Standalone(root,'test-only-key');manager.prepareCache=()=>({});
 let invocation;const child=new EventEmitter();child.exitCode=null;child.signalCode=null;
 manager.launch=async (bin,args,env)=>{invocation={bin,args,env};return child;};
 await manager.start(p);
 assert.equal(invocation.bin,'vllm');assert.deepEqual(invocation.args,['serve',...p.args]);
 assert.equal(invocation.env.VLLM_API_KEY,'test-only-key');assert.equal(manager.status(p).running,true);
 await assert.rejects(manager.start({...p,id:'other'}),/停止当前模型/);
 manager.terminate=async c=>{if(c===child)child.emit('exit',0);};await manager.stop(p);assert.equal(manager.status(p).running,false);
 p.name='user edit';fs.writeFileSync(file,JSON.stringify([p]));
 await initializeNext(root,consoleDir,selected);
 assert.equal(JSON.parse(fs.readFileSync(file))[0].name,'user edit');
 const reopened=new Store(root);assert.equal(reopened.settings().autoStart,false);reopened.close();
});
test('unrelated existing panel data is refused without modification',async t=>{
 const root=temp(t);fs.writeFileSync(path.join(root,'profiles.json'),'original');
 await assert.rejects(initializeNext(root,consoleDir,selected),/dedicated empty/);
 assert.equal(fs.readFileSync(path.join(root,'profiles.json'),'utf8'),'original');
 assert.equal(fs.existsSync(path.join(root,'workbench.sqlite')),false);
});
