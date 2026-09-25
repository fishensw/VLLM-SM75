import json,pathlib,torch,types,gc,os
from safetensors import safe_open
from vllm.models.qwen4_exp.nvidia.sm75_mtp_norm import sm75_mtp_gemma_norm as fused
r=pathlib.Path(os.environ.get('TEST_OUTPUT_DIR','/tmp/next-tests'))
(r/'results').mkdir(parents=True,exist_ok=True)
weights={}
with safe_open(os.environ.get('MTP_DENSE_PATH','/models/fp8ple/runtime/mtp-int4-g32/mtp-dense.safetensors'),framework='pt',device='cpu') as f:
 for k in f.keys():
  if 'pre_fc_norm_' in k:weights[k]=f.get_tensor(k).to(torch.float16)
assert sorted(v.numel() for v in weights.values())==[2560,10240],[(k,v.shape) for k,v in weights.items()]
def ref(x,m):
 y=x.float();y=y*torch.rsqrt((y*y).mean(-1,keepdim=True)+m.variance_epsilon)
 return (y*(m.weight.float()+1)).to(x.dtype)
rows=[]
for gpu in range(8):
 torch.cuda.set_device(gpu);torch.manual_seed(42)
 for name,w in weights.items():
  m=types.SimpleNamespace(weight=w.cuda(gpu),variance_epsilon=1e-6)
  for n in ([1,2,4,8,64,4096,6144] if gpu==0 else [1,4,4096]):
   for scale in ([0,1e-4,1,100] if gpu==0 else [1]):
    x=(torch.randn(n,w.numel(),device=gpu,dtype=torch.float16)*scale).contiguous()
    y=fused(x,m);z=ref(x,m)
    torch.testing.assert_close(y,z,atol=1e-2,rtol=2e-3)
    rows.append(dict(gpu=gpu,n=n,h=w.numel(),scale=scale,max_abs=(y-z).abs().max().item(),rel_l2=((y.float()-z.float()).norm()/z.float().norm().clamp_min(1e-10)).item()))
   if n in [1,4,4096]:
    st=torch.cuda.Stream(device=gpu);st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
     for _ in range(3):gy=fused(x,m)
    torch.cuda.current_stream().wait_stream(st);g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=st):gy=fused(x,m)
    x.add_(0.125);g.replay();torch.cuda.synchronize()
    torch.testing.assert_close(gy,ref(x,m),atol=1e-2,rtol=2e-3)
    del g,gy
   del x,y,z
  del m
 torch.cuda.empty_cache()
print('NUMERICS_AND_GRAPH_PASS',len(rows),flush=True)
torch.cuda.set_device(0)
w=next(v for v in weights.values() if v.numel()==10240)
m=types.SimpleNamespace(weight=w.cuda(),variance_epsilon=1e-6)
x=torch.randn(4096,10240,device='cuda',dtype=torch.float16)
bench={}
for name,fn in [('native',ref),('fused',fused)]:
 for _ in range(5):y=fn(x,m)
 del y;torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
 before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();y=fn(x,m);torch.cuda.synchronize()
 peak=torch.cuda.max_memory_allocated()-before;del y
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
 for _ in range(50):y=fn(x,m)
 b.record();torch.cuda.synchronize();bench[name]=dict(peak_bytes=peak,ms=a.elapsed_time(b)/50);del y
out=dict(rows=rows,benchmark=bench);(r/'results/mtp-norm-test.json').write_text(json.dumps(out,indent=2))
print(json.dumps(bench),flush=True)
