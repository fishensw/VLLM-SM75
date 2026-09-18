import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {Store} from '../store.mjs';
import {JobLedger} from '../job-ledger.mjs';
test('restart preserves destination and finished jobs, marks running jobs interrupted, excludes command credentials',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-jobs-'));let store=new Store(root);
 try{const ledger=new JobLedger(store),running=ledger.create('download',{download:{repo:'owner/model',provider:'modelscope',modelRoot:'/models',destination:'/models/modelscope/owner--model'}});
 running.command={env:{TOKEN:'secret'}};ledger.persist(running);
 const done=ledger.create('download');done.state='complete';ledger.persist(done);store.close();store=new Store(root);
 const recovered=new JobLedger(store);assert.equal(recovered.jobs.get(running.id).state,'interrupted');assert.equal(recovered.jobs.get(done.id).state,'complete');assert.equal(recovered.jobs.get(running.id).download.destination,'/models/modelscope/owner--model');assert.equal(JSON.stringify(recovered.list()).includes('secret'),false);
 }finally{store.close();fs.rmSync(root,{recursive:true,force:true});}
});
