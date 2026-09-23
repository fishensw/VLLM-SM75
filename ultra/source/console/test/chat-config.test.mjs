import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';import os from 'node:os';import path from 'node:path';import {chatConfig,harnessThinking} from '../chat-config.mjs';
test('chat recommendations read model generation config and remain available when server uses vllm defaults',()=>{const dir=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-chat-'));fs.writeFileSync(path.join(dir,'generation_config.json'),JSON.stringify({temperature:1,top_p:.95,top_k:20,bos_token_id:12}));const p={args:[dir,'--reasoning-parser','qwen3']};assert.deepEqual(chatConfig(p).sampling,{temperature:1,top_p:.95,top_k:20});assert.equal(chatConfig(p).thinking,true);assert.deepEqual(chatConfig({args:[dir,'--generation-config','vllm']}).sampling,{temperature:1,top_p:.95,top_k:20});assert.deepEqual(harnessThinking({args:[dir]}),{});assert.equal(harnessThinking(p).compat.thinkingTokenBudgetField,'thinking_token_budget');assert.deepEqual(Object.keys(harnessThinking(p).reasoningEfforts),['off','low','medium','high']);});


test('image capability follows the checkpoint and explicit language-only limits',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-vision-'));
 try {
  fs.writeFileSync(path.join(dir,'config.json'),JSON.stringify({vision_config:{}}));
  assert.deepEqual(chatConfig({args:[dir]}).input,['text','image']);
  assert.deepEqual(chatConfig({args:[dir,'--language-model-only']}).input,['text']);
  for(const limit of [{image:0},{image:{count:0}}])
   assert.deepEqual(chatConfig({args:[dir,'--limit-mm-per-prompt',JSON.stringify(limit)]}).input,['text']);
  fs.writeFileSync(path.join(dir,'config.json'),'{}');
  assert.deepEqual(chatConfig({args:[dir]}).input,['text']);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
