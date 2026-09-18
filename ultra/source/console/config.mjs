import path from 'node:path';
import fs from 'node:fs';

export function flashinferWorkspace(root, mkdir = false) {
  const base = path.join(root, 'shared/flashinfer-home');
  const target = path.join(root, 'shared/flashinfer');
  if (mkdir) {
    fs.mkdirSync(path.join(base, '.cache'), { recursive: true });
    fs.mkdirSync(target, { recursive: true });
    const link = path.join(base, '.cache/flashinfer');
    let existing;
    try { existing = fs.lstatSync(link); } catch (e) { if (e.code !== 'ENOENT') throw e; }
    if (!existing) fs.symlinkSync(target, link, 'dir');
    else if (fs.realpathSync(link) !== fs.realpathSync(target)) throw Error('FlashInfer 缓存映射冲突');
  }
  return base;
}

export function cacheLayout(root, format='fp8') {
  if (!path.isAbsolute(root)) throw Error('缓存根目录必须是绝对路径');
  if (!['fp8','awq','kat'].includes(format)) throw Error('未知模型格式');
  return Object.fromEntries(Object.entries({
    VLLM_CACHE_ROOT:`${format}/vllm`, TRITON_CACHE_DIR:`${format}/triton`,
    TORCH_EXTENSIONS_DIR:'shared/torch_extensions',
    TORCHINDUCTOR_CACHE_DIR:`${format}/inductor`, CUDA_CACHE_PATH:'shared/cuda',
  }).map(([k,v])=>[k,path.join(root,v)]));
}
export const containerCache = {
  VLLM_CACHE_ROOT:'/root/.cache/vllm',TRITON_CACHE_DIR:'/root/.triton/cache',
  TORCH_EXTENSIONS_DIR:'/root/.cache/torch_extensions',
  TORCHINDUCTOR_CACHE_DIR:'/root/.cache/torchinductor',CUDA_CACHE_PATH:'/root/.nv/ComputeCache',
};
export function validateProfile(p) {
  if (!/^[a-z0-9][a-z0-9-]{0,47}$/.test(p.id||'')) throw Error('配置 ID 仅允许小写字母、数字和连字符');
  if (!Array.isArray(p.args)||!p.args.length||p.args.some(a=>typeof a!=='string'||a.includes('\0'))) throw Error('模型与参数须为字符串数组');
  if (!Number.isInteger(p.port)||p.port<1024||p.port>65535) throw Error('端口范围 1024–65535');
  if (!['docker','native'].includes(p.backend||'docker')) throw Error('未知运行方式');
  if (p.args.some(a=>a==='--api-key'||a.startsWith('--api-key='))) throw Error('API key 由服务端管理');
  if(p.backend!=='native' && p.image && !p.image.startsWith('local/vllm-sm75:v015-')) throw Error('本地候选阶段仅运行 v015 独立镜像');
  if(!p.power)p.power={mode:'pstate',idleSeconds:1,util:5,confirm:60,low:8,high:16,poll:5,gpus:'0,1,2,3'};
  if(p.power){
    const q=p.power,m=q.mode||'pstate';
    if(!['sleep','pstate'].includes(m))throw Error('电源模式无效');
    const num=(v,d,lo,hi,label)=>{const x=Number(v ?? d);if(!Number.isFinite(x)||x<lo||x>hi)throw Error(label+' 超出范围');return x;};
    p.power={mode:m,idleSeconds:num(q.idleSeconds??(q.idleMinutes!=null?q.idleMinutes*60:1),1,1,86400,'空闲时长（秒）'),util:num(q.util,5,0,100,'负载阈值'),confirm:num(q.confirm,60,0,600,'确认时长'),low:num(q.low,8,0,16,'P8 值'),high:num(q.high,16,0,16,'高态值'),poll:num(q.poll,5,1,60,'轮询间隔'),gpus:typeof q.gpus==='string'&&/^[0-9,]+$/.test(q.gpus)?q.gpus:'0,1,2,3'};
    if(p.power.low===0&&p.power.high===0)throw Error('P8/高态不能同时为 0（会钉在 645MHz）');
  }
  const ki=p.args.indexOf('--kv-transfer-config');
  if(ki>=0){
    let kv;try{kv=JSON.parse(p.args[ki+1]);}catch{throw Error('CPU KV 配置不是合法 JSON');}
    const KVDEF={kv_connector:'OffloadingConnector',kv_role:'kv_both',kv_connector_extra_config:{spec_name:'CPUOffloadingSpec',cpu_bytes_to_use:8589934592}};
    kv={...KVDEF,...kv,kv_connector_extra_config:{...KVDEF.kv_connector_extra_config,...(kv.kv_connector_extra_config||{})}};
    p.args[ki+1]=JSON.stringify(kv);
  }
  cacheLayout(p.cacheRoot,p.format);
  return p;
}
export function makeCommand(profile, apiKey, options={}) {
  const p=validateProfile(profile), name=`sm75-v015-test-${p.id}`;
  const layout=cacheLayout(p.cacheRoot,p.format);
  if (options.mkdir) for(const v of Object.values(layout)) fs.mkdirSync(v,{recursive:true});
  const env={...p.env,VLLM_API_KEY:apiKey,VLLM_MONITOR:'1'};
  if (p.backend==='native') {
    const base=flashinferWorkspace(p.cacheRoot, options.mkdir);
    const args=[...p.args];const i=args.indexOf('--port');if(i>=0)args[i+1]=String(p.port);else args.push('--port',String(p.port));
    return {name,bin:options.vllmBin||'vllm',args:['serve',...args],env:{...env,...layout,FLASHINFER_WORKSPACE_BASE:base}};
  }
  const args=['run','--detach','--name',name,'--label','sm75.managed=v015-candidate',
    '--gpus',p.gpus||'all','--shm-size','16g'];
  const sharedNetwork=options.engineNetwork||process.env.SM75_ENGINE_NETWORK;
  if(sharedNetwork){if(!/^container:sm75-v015-test-[a-z0-9-]+$/.test(sharedNetwork))throw Error('无效候选网络');args.push('--network',sharedNetwork);}else args.push('--publish',`${p.port}:8000`);
  const binds=p.binds||[];
  for(const bind of binds) {
    if(typeof bind!=='string'||!bind.startsWith('/')||bind.includes('docker.sock')) throw Error('无效挂载');
    const dest=bind.split(':')[1];
    if(!Object.values(containerCache).includes(dest)) args.push('--volume',bind);
  }
  for(const [k,host] of Object.entries(layout)) args.push('--volume',`${host}:${containerCache[k]}`);
  args.push('--volume',`${path.join(p.cacheRoot,'shared/flashinfer')}:/root/.cache/flashinfer`);
  for(const [k,v] of Object.entries({...env,...containerCache,FLASHINFER_WORKSPACE_BASE:'/root'})) {
    if(!/^[A-Z][A-Z0-9_]*$/.test(k)) throw Error('无效环境变量');
    // Docker inherits credentials through the child environment, not argv.
    args.push('--env',k==='VLLM_API_KEY'?k:`${k}=${v}`);
  }
  const modelArgs=[...p.args];if(sharedNetwork){const i=modelArgs.indexOf('--port');if(i>=0)modelArgs[i+1]=String(p.port);else modelArgs.push('--port',String(p.port));}
  args.push(p.image||'local/vllm-sm75:v015-candidate-20260912',...modelArgs);
  return {name,bin:'docker',args,env};
}

export function recommend({gpus=[],ramGiB=0,weightGiB=0,draftGiB=0,cpuKvGiB=8}) {
  const usable=gpus.reduce((n,g)=>n+Number(g.memoryGiB||0)*.85,0);
  return {gpuCount:gpus.length,usableGpuGiB:usable,cpuKvGiB,
    canFitEstimate:usable>weightGiB+draftGiB,
    warnings:[...(ramGiB<cpuKvGiB+8?['CPU KV 配额可能挤占宿主内存']:[]),
    ...(usable<=weightGiB+draftGiB?['权重与草稿超出估算显存预算']:[])],
    note:'按容量初筛；TP、模型结构及 GPU 拓扑仍需校验。保留已有参数。'};
}
