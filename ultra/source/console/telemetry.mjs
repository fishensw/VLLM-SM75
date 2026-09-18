import fs from 'node:fs';import path from 'node:path';import vm from 'node:vm';
const html=fs.readFileSync(new URL('../vllm/entrypoints/serve/instrumentator/dashboard.html',import.meta.url),'utf8');
const scope={};vm.createContext(scope);vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],scope);export const metrics=scope.MonitorCore;
export class Telemetry {
 constructor(root){this.root=root;this.rows=new Map();this.lastSave=new Map();this.failed=new Set();fs.mkdirSync(root,{recursive:true});}
 file(id){if(!/^[a-z0-9-]+$/.test(id))throw Error('invalid telemetry id');return path.join(this.root,id+'.json');}
 read(id){if(!this.rows.has(id)){try{this.rows.set(id,JSON.parse(fs.readFileSync(this.file(id))));}catch{this.rows.set(id,[]);}}const rows=this.rows.get(id);while(rows.length&&rows[0].t<Date.now()-900000)rows.shift();return rows;}
 miss(id){this.failed.add(id);}
 observe(id,text,t=Date.now()){
  const m=metrics.parsePrometheus(text);if(metrics.sum(m,'num_requests_running')===null)throw Error('engine metrics unavailable');
  const rows=this.read(id);let prev=rows.at(-1);if(prev&&metrics.counterReset(m,prev.m)){rows.length=0;prev=null;}
  const gap=this.failed.delete(id),rates=gap?{prefill:null,decode:null}:metrics.sampleRates(m,prev?.m,prev?(t-prev.t)/1000:0);
  const kv=m['vllm:kv_cache_usage_perc'];rows.push({t,m,...rates,gap,kv:kv?.length===1?kv[0].value*100:null,waiting:metrics.sum(m,'num_requests_waiting')});
  while(rows.length>181||rows.length&&rows[0].t<t-900000)rows.shift();
  if(t-(this.lastSave.get(id)||0)>30000){this.flush(id);this.lastSave.set(id,t);}return rows.at(-1);
 }
 flush(id){const file=this.file(id);let rows=this.read(id),data=JSON.stringify(rows);while(data.length>8_000_000&&rows.length>2){rows.shift();data=JSON.stringify(rows);}fs.writeFileSync(file+'.tmp',data,{mode:0o600});fs.renameSync(file+'.tmp',file);}
}
