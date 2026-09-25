from pathlib import Path
import ast,shutil
root=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia')
p=root/'mtp.py';s=p.read_text()
s='from vllm.models.qwen4_exp.nvidia.sm75_mtp_norm import sm75_mtp_gemma_norm\n'+s
old='inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)'
new='inputs_embeds = sm75_mtp_gemma_norm(inputs_embeds, self.pre_fc_norm_embedding)'
assert s.count(old)==1;s=s.replace(old,new)
old='self.pre_fc_norm_hidden(hidden_states.flatten(-2))'
new='sm75_mtp_gemma_norm(hidden_states.flatten(-2), self.pre_fc_norm_hidden)'
assert s.count(old)==1;s=s.replace(old,new)
ast.parse(s);p.write_text(s)
shutil.copyfile(Path(__file__).with_name('sm75_mtp_norm.py'),root/'sm75_mtp_norm.py')
print('Fused MTP-only Gemma norms; FP32 reduction/multiply, FP16 output; original parameters retained')
