"""Apply the five handover baseline edits to this test container only."""
from pathlib import Path

root = Path('/usr/local/lib/python3.12/dist-packages')
edits = [
    ('vllm/models/qwen4_exp/nvidia/model.py',
     'params_dtype=torch.bfloat16,', 'params_dtype=torch.float16,'),
    ('vllm/model_executor/models/utils.py',
     'return any(iup) or any(ius)',
     'return "self_attn.indexer" in qualname or any(iup) or any(ius)'),
    ('torch/_inductor/compile_fx.py',
     'if device_interface.is_bf16_supported(including_emulation=False):', 'if True:'),
    ('vllm/models/qwen4_exp/common/hyperconnection.py',
     'params_dtype: torch.dtype = torch.bfloat16', 'params_dtype: torch.dtype = torch.float16'),
]
pending = {}
for relative, old, new in edits:
    path = root / relative
    text = pending.get(path, path.read_text())
    count = text.count(old)
    if count == 0:
        raise RuntimeError(f'Baseline patch source mismatch: {relative}: {old}')
    pending[path] = text.replace(old, new)
    print(f'baseline patch: {relative}: {count} replacements')
path = root / 'vllm/models/qwen4_exp/nvidia/model.py'
lines = pending[path].splitlines(keepends=True)
matches = [i for i, line in enumerate(lines) if 'use_qsa = getattr' in line]
if len(matches) != 1:
    raise RuntimeError(f'Expected exactly one use_qsa selector, got {len(matches)}')
i = matches[0]
if lines[i].strip() != 'use_qsa = getattr(config, "indexer_n_heads", None) is not None':
    raise RuntimeError('Unrecognized QSA selector requires manual review')
indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
lines[i] = indent + 'use_qsa = False\n'
pending[path] = ''.join(lines)
for path, text in pending.items():
    path.write_text(text)
print('All baseline patch assertions passed')
