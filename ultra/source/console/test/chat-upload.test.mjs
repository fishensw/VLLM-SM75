import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../public/app.js',import.meta.url),'utf8');
const uploadSource=source.slice(source.indexOf('export async function uploadChatImages'),source.indexOf('// Read the engine state before toggling'));
const upload=vm.runInNewContext(uploadSource.replace('export async function','(async function').trim()+')');

test('image batch is profile-bound, cancellable, and rejects oversized submissions before upload',async()=>{
 const controller=new AbortController(),calls=[];
 const request=async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>({id:'saved'})};};
 const files=[{size:1}];await upload(files,'first/profile',controller.signal,request);
 assert.equal(calls[0].url,'/console-api/attachments?profile=first%2Fprofile');
 assert.equal(calls[0].options.signal,controller.signal);
 await assert.rejects(upload(Array(5).fill({size:1}),'p',controller.signal,request),/最多/);
 await assert.rejects(upload([{size:21*1024*1024}],'p',controller.signal,request),/20 MiB/);
 assert.equal(calls.length,1);
 controller.abort();await assert.rejects(upload(files,'p',controller.signal,request),{name:'AbortError'});
 assert.equal(calls.length,1);
});

for(const action of ['cancel','switch']) test(`upload holds the chat lock and rejects ${action} before mutating history`,async()=>{
 const elements=new Map();const $=id=>{if(!elements.has(id))elements.set(id,{value:id==='prompt'?'hello':id==='model'?'model':'',hidden:false});return elements.get(id);};
 let release;const pending=new Promise(resolve=>release=resolve);
 const context=vm.createContext({$,run:fn=>fn,active:'profile-one',chat:[],selectedImages:[{size:1}],abort:null,AbortController,route:()=>'/captured-chat',chatRequestOptions:()=>({}),uploadChatImages:()=>pending});
 const snippet=source.slice(source.indexOf('  $("chatForm").onsubmit = run(async (e) => {'),source.indexOf('  $("installHarness").onclick'));
 vm.runInContext(snippet,context);
 const event={preventDefault(){}};const sending=$('chatForm').onsubmit(event);
 assert(context.abort);assert.equal($('sendChat').hidden,true);
 await assert.rejects($('chatForm').onsubmit(event),/尚未完成/);
 if(action==='cancel')context.abort.abort();else context.active='profile-two';
 release([]);
 await assert.rejects(sending,action==='cancel'?{name:'AbortError'}:/配置已变化/);
 assert.equal(context.chat.length,0);assert.equal(context.abort,null);
 assert.equal($('sendChat').hidden,false);assert.equal($('cancel').hidden,true);
});
