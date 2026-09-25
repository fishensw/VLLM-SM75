from pathlib import Path
import ast
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py')
s=p.read_text()
assert s.count('params_dtype=torch.bfloat16,')==1
s=s.replace('params_dtype=torch.bfloat16,','params_dtype=torch.float16,')
ast.parse(s);p.write_text(s)
print('MTP HC dtype patched FP16; remaining v0.30 MTP implementation retained')
