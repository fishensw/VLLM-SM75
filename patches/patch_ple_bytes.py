from pathlib import Path
import ast
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ngram_embedding.py')
s=p.read_text()
old='''                self._uva_weight,
                flat_ids,
                output,
'''
new='''                self._uva_weight.view(torch.uint8) if self.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else self._uva_weight,
                flat_ids,
                output.view(torch.uint8) if self.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else output,
'''
assert s.count(old)==1
s=s.replace(old,new);ast.parse(s);p.write_text(s)
print('PLE pinned lookup copies FP8 raw bytes, no conversion')
