import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../public/app.js',import.meta.url),'utf8');
const fn=vm.runInNewContext(source.slice(source.indexOf('export async function toggleMonitorSpec')).replace('export async function','(async function').trim() + ')');
test('monitor toggle reads back real state and preserves pending acknowledgement',async()=>{
 for(const pending of [false,true]){
 const calls=[];const state=await fn(async payload=>{calls.push(payload);return calls.length===1?{spec_configured:true,enabled:true}:calls.length===2?{ok:true}:{enabled:false,desired_enabled:false,pending};});
 assert.equal(calls.length,3);assert.equal(calls[1].enabled,false);assert.equal(state.pending,pending);
 }
});
test('monitor does not claim success on unchanged state, HTTP error or unsupported scheduler',async()=>{
 await assert.rejects(fn(async()=>({spec_configured:false})),/未启用/);
 await assert.rejects(fn(async()=>({spec_configured:true,enabled:true,pending:true})),/尚未生效/);
 await assert.rejects(fn(async p=>{if(p)throw Error('HTTP 401');return {spec_configured:true,enabled:true};}),/HTTP 401/);
 await assert.rejects(fn(async()=>({spec_configured:true,enabled:true})),/状态未改变/);
});
