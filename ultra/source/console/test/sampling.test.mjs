import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';import os from 'node:os';import path from 'node:path';import {validateSampling,dshSampling} from '../sampling.mjs';
test('recommended presence penalty reaches OpenAI-compatible workbench requests',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'presence-'));
 try {
  assert.throws(()=>validateSampling({presence_penalty:2.1}));
  const params=validateSampling({temperature:.7,top_p:.8,top_k:20,min_p:0,presence_penalty:1.5,repetition_penalty:1});
  const file=path.join(dir,'prefs.json');fs.writeFileSync(file,JSON.stringify({models:{'local/qwen':params}}));
  const payload={};dshSampling('local',{id:'qwen',api:'openai-completions'},file).onPayload(payload);
  assert.deepEqual(payload,params);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
test('sampling validation and per-model request payload',()=>{assert.deepEqual(validateSampling({top_k:0}),{top_k:0});assert.throws(()=>validateSampling({top_k:-2}));assert.throws(()=>validateSampling({top_p:2}));assert.throws(()=>validateSampling({temperature:null}));const dir=fs.mkdtempSync(path.join(os.tmpdir(),'sampling-'));try{const file=path.join(dir,'prefs.json');fs.writeFileSync(file,JSON.stringify({models:{'local/qwen':{temperature:0.7,top_k:20,max_tokens:1024}}}));const opt=dshSampling('local',{id:'qwen',api:'openai-completions'},file);const payload={messages:[]};opt.onPayload(payload);assert.equal(payload.top_k,20);assert.equal(opt.temperature,0.7);assert.equal(opt.maxTokens,1024);const other={};dshSampling('other',{id:'qwen',api:'openai-completions'},file).onPayload(other);assert.deepEqual(other,{});}finally{fs.rmSync(dir,{recursive:true,force:true});}});
