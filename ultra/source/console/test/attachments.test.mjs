import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';import os from 'node:os';import path from 'node:path';import {Attachments} from '../attachments.mjs';
test('attachment upload validates, isolates and expands image messages',async()=>{const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-att-'));try{const a=new Attachments(root), png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=','base64');const req=(async function*(){yield png})();const m=await a.upload(req,'fp8');assert.equal(m.mime,'image/png');const x=a.expand([{role:'user',content:[{type:'text',text:'x'},{type:'image',attachmentId:m.id}]}],'fp8',true);assert.match(x[0].content[1].image_url.url,/^data:image\/png;base64/);assert.throws(()=>a.expand([{role:'user',content:[{type:'image',attachmentId:m.id}]}],'other',true),/不属于/);assert.throws(()=>a.expand([{role:'user',content:[{type:'image',attachmentId:m.id}]}],'fp8',false),/未启用/)}finally{fs.rmSync(root,{recursive:true,force:true})}});


test('attachment references reject traversal, remote URLs and malformed message parts',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-attachment-boundary-'));
 try {
  const a=new Attachments(root);
  for(const id of ['../key','/etc/passwd','a'.repeat(31),'A'.repeat(32)])assert.throws(()=>a.read(id),/ID/);
  for(const messages of [[null],[{role:'user',content:[null]}],[{role:'user',content:[{type:'image_url',image_url:{url:'https://example.invalid/private'}}]}]])
   assert.throws(()=>a.expand(messages,'p',true),/无效/);
  assert.deepEqual(a.expand([{role:'user',content:'legacy plain text'}],'p',false),[{role:'user',content:'legacy plain text'}]);
 }finally{fs.rmSync(root,{recursive:true,force:true});}
});
