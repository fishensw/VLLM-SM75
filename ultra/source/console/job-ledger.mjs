import crypto from 'node:crypto';
// Persist task metadata only; executable commands, environment and child handles stay in memory.
export class JobLedger {
 constructor(store){this.store=store;this.jobs=new Map();for(const j of store.list('job')){if(j.state==='running'){j.state='interrupted';j.finished=Date.now();j.log=(j.log||'')+'\n工作台重新启动，任务已中断。点击继续下载复用 SDK 下载缓存。';this.persist(j);}this.jobs.set(j.id,j);}}
 public(j){const {child,command,...value}=j;return value;}
 persist(j){this.store.put('job',j.id,this.public(j));}
 create(kind,metadata={}){const j={id:crypto.randomUUID(),kind,state:'running',log:'',started:Date.now(),...metadata};this.jobs.set(j.id,j);this.persist(j);return j;}
 list(){return [...this.jobs.values()].sort((a,b)=>b.started-a.started).map(j=>this.public(j));}
}
