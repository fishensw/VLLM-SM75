import fs from 'node:fs';
export const samplingFields={temperature:[0,2],top_p:[0,1],top_k:[-1,1000000],min_p:[0,1],presence_penalty:[-2,2],repetition_penalty:[0.01,100],max_tokens:[1,10000000]};
export function validateSampling(input){const out={};for(const [k,v]of Object.entries(input||{})){if(!(k in samplingFields))throw Error('未知生成参数 '+k);const [min,max]=samplingFields[k];if(typeof v!=='number'||!Number.isFinite(v)||v<min||v>max||(['top_k','max_tokens'].includes(k)&&!Number.isInteger(v)))throw Error('无效参数 '+k);out[k]=v;}return out;}
export function dshSampling(provider,model,file='/dsh/home/sm75-sampling.json'){
 let params={};try{params=JSON.parse(fs.readFileSync(file,'utf8')).models?.[provider+'/'+model.id]||{};}catch{}
 return {...(params.temperature!==undefined?{temperature:params.temperature}:{}),...(params.max_tokens!==undefined?{maxTokens:params.max_tokens}:{}),onPayload:payload=>{if(model.api==='openai-completions')Object.assign(payload,params);}};
}
