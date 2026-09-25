"""Opt-in patch for the isolated test image; exact source assertions before writes."""
from pathlib import Path
import shutil
import vllm

root = Path(vllm.__file__).parent / 'models/qwen4_exp/nvidia'
source = root / 'hyperconnection.py'
text = source.read_text(encoding='utf-8')
marker = '\n    def mix(\n'
assert text.count(marker) == 1, 'HC insertion point changed'
assert 'install_hc_candidate' not in text, 'Already patched'
assert 'disable_tp=True' in text and 'self.input_mix_weight_up = ReplicatedLinear(' in text
updated = text.replace(marker, '\n        from .hc_gemv_next import install_hc_candidate\n'
                             '        install_hc_candidate(self)\n' + marker)
compile(updated, str(source), 'exec')
candidate = Path('/test/hc_gemv_candidate.py')
compile(candidate.read_text(encoding='utf-8'), str(candidate), 'exec')
shutil.copyfile(candidate, root / 'hc_gemv_next.py')
source.write_text(updated, encoding='utf-8')
print('HC candidate dispatch installed; requires explicit VLLM_SM75_QWEN38_HC_GEMV=1')
