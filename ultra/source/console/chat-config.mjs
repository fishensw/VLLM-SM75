import fs from 'node:fs';
import path from 'node:path';
export const thinkingBudgets={low:1024,medium:4096,high:16384};
export function chatConfig(p){
 const val=k=>{const i=p.args.indexOf(k);return i<0?undefined:p.args[i+1];};
 let generation={};const source=val('--generation-config');
 try{generation=JSON.parse(fs.readFileSync(path.join(source&&source!=='auto'&&source!=='vllm'?source:p.args[0],'generation_config.json'),'utf8'));}catch{}
 try{Object.assign(generation,JSON.parse(val('--override-generation-config')||'{}'));}catch{}
 const sampling={};for(const k of ['temperature','top_p','top_k','min_p','repetition_penalty','presence_penalty','frequency_penalty','max_new_tokens'])if(typeof generation[k]==='number')sampling[k==='max_new_tokens'?'max_tokens':k]=generation[k];
 return {input:modelInput(p),sampling,thinking:!!val('--reasoning-parser'),budgets:thinkingBudgets,source:Object.keys(sampling).length?'模型生成配置':'推理服务默认值'};
}
export function harnessThinking(p){return chatConfig(p).thinking?{reasoningEfforts:{off:null,low:'low',medium:'medium',high:'high'},compat:{thinkingFormat:'qwen-chat-template',thinkingTokenBudgetField:'thinking_token_budget'}}:{};}

export function modelInput(p) {
 if (p.args.includes('--language-model-only')) return ['text'];
 try {
  const i=p.args.indexOf('--limit-mm-per-prompt');
  const limit=i>=0?JSON.parse(p.args[i+1]):{};
  if (limit.image===0 || limit.image?.count===0) return ['text'];
  const cfg=JSON.parse(fs.readFileSync(path.join(p.args[0],'config.json'),'utf8'));
  return cfg.vision_config ? ['text','image'] : ['text'];
 } catch { return ['text']; }
}
