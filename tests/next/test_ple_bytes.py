import torch,json
from types import SimpleNamespace
from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpPLEPinnedHostEmbedding
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
results=[]
for gpu in range(8):
 torch.cuda.set_device(gpu)
 for dtype in (torch.float8_e4m3fn,torch.float8_e5m2):
  raw=torch.arange(256,dtype=torch.int16).to(torch.uint8)[:,None].expand(256,160).contiguous().pin_memory()
  weight=raw.view(dtype)
  obj=SimpleNamespace(weight=weight,_uva_weight=get_accelerator_view_from_cpu_tensor(weight),embedding_dim=160,shard_indices=SimpleNamespace(org_vocab_start_index=10,org_vocab_end_index=266),_block_d=256)
  ids=torch.tensor([9]+list(range(10,266))+[266,10,265],device='cuda')
  golden=torch.cat([torch.zeros((1,160),dtype=torch.uint8),raw,torch.zeros((1,160),dtype=torch.uint8),raw[:1],raw[-1:]]).cuda()
  out=Qwen4ExpPLEPinnedHostEmbedding._lookup(obj,ids)
  assert torch.equal(out.view(torch.uint8),golden)
  for _ in range(3):Qwen4ExpPLEPinnedHostEmbedding._lookup(obj,ids,output=out)
  torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=gpu)):Qwen4ExpPLEPinnedHostEmbedding._lookup(obj,ids,output=out)
  graph.replay();torch.cuda.synchronize()
  assert torch.equal(out.view(torch.uint8),golden)
  results.append(dict(gpu=gpu,dtype=str(dtype),all_256_bytes=True,boundary_repeat=True,cuda_graph=True))
print(json.dumps(results))
