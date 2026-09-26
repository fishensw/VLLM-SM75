import path from 'node:path';import test from 'node:test';import assert from 'node:assert/strict';
import {makeCommand,validateProfile} from '../config.mjs';
const p={id:'fp8',port:8015,args:['/models/qwen','--tensor-parallel-size','4'],cacheRoot:'/test/cache',format:'fp8',binds:['/weights:/models:ro']};
test('stable cache across container generations and sleep time',()=>{const a=makeCommand(p,'secret');const b=makeCommand({...p,args:[...p.args,'--auto-sleep-idle-timeout','30']},'secret');const mounts=c=>c.args.filter((v,i)=>c.args[i-1]==='--volume');assert.deepEqual(mounts(a),mounts(b));assert.equal(mounts(a).length,7);assert(!a.args.some(x=>x.includes('secret')));assert.equal(a.env.VLLM_API_KEY,'secret');});
test('only candidate namespace may launch',()=>{assert.throws(()=>makeCommand({...p,image:'vllm-sm75:v0.1.4'},'x'));assert(makeCommand(p,'x').name.startsWith('sm75-v015-test-'));});
test('native backend uses the selected virtual environment and stable cache paths',()=>{const cmd=makeCommand({...p,backend:'native'},'secret',{vllmBin:'/opt/sm75/venv/bin/vllm'});assert.equal(cmd.bin,'/opt/sm75/venv/bin/vllm');assert.equal(cmd.args[0],'serve');assert.equal(cmd.env.VLLM_CACHE_ROOT,path.join('/test/cache','fp8/vllm'));assert.equal(cmd.env.TRITON_CACHE_DIR,path.join('/test/cache','fp8/triton'));assert(!cmd.args.includes('secret'));});
test('invalid profile refused before invoking docker',()=>{for(const bad of [{id:'../prod'},{args:'shell'},{port:80},{cacheRoot:'relative'},{args:['model','--api-key','secret']},{binds:['/var/run/docker.sock:/var/run/docker.sock']}])assert.throws(()=>makeCommand({...p,...bad},'x'));});


test('workbench namespace gives each engine its own API port without host networking',()=>{const p={id:'fp8',args:['model'],port:8015,format:'fp8',cacheRoot:'/cache',backend:'docker'};const c=makeCommand(p,'secret',{engineNetwork:'container:sm75-v015-test-workbench'});assert.equal(c.args[c.args.indexOf('--network')+1],'container:sm75-v015-test-workbench');assert.equal(c.args[c.args.indexOf('--port')+1],'8015');assert.ok(!c.args.includes('--publish'));assert.throws(()=>makeCommand(p,'secret',{engineNetwork:'host'}));});

test('API key cannot be overridden with equals syntax',()=>{assert.throws(()=>validateProfile({...p,args:['model','--api-key=override']}),/API key/);});

test('engine commands default NCCL to SYS and respect container/profile overrides', () => {
  const previous = process.env.NCCL_P2P_LEVEL;
  try {
    for (const backend of ['native', 'docker']) {
      delete process.env.NCCL_P2P_LEVEL;
      const defaults = makeCommand({...p, backend, env:{}}, 'test-key');
      assert.equal(defaults.env.NCCL_P2P_LEVEL, 'SYS');
      if (backend === 'docker') assert.ok(defaults.args.includes('NCCL_P2P_LEVEL=SYS'));
      process.env.NCCL_P2P_LEVEL = 'PIX';
      assert.equal(makeCommand({...p, backend, env:{}}, 'test-key').env.NCCL_P2P_LEVEL, 'PIX');
      const custom = makeCommand({...p, backend, env:{NCCL_P2P_LEVEL:'PHB'}}, 'test-key');
      assert.equal(custom.env.NCCL_P2P_LEVEL, 'PHB');
      if (backend === 'docker') assert.ok(custom.args.includes('NCCL_P2P_LEVEL=PHB'));
    }
  } finally {
    if (previous === undefined) delete process.env.NCCL_P2P_LEVEL;
    else process.env.NCCL_P2P_LEVEL = previous;
  }
});
