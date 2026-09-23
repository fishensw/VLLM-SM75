#!/usr/bin/env python3
"""CPU-only metadata validation; never loads weights or starts an engine."""
import argparse
import copy
import hashlib
import json
import os
import tempfile
from types import SimpleNamespace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--overrides-json', type=Path)
    parser.add_argument('--expected-max-length', type=int, default=1048576)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    assert os.environ.get('SM75_DISPOSABLE_AUDIT') == '1', 'Requires disposable audit guard'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide all GPUs for this config test'
    assert not Path('/dev/nvidia0').exists(), 'This audit must not have GPU access'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '0'
    import torch
    import transformers
    import vllm
    from vllm.config import ModelConfig
    import vllm.config.model as model_module
    assert vllm.__version__.split("+")[0] == "0.30.0", vllm.__version__
    model_sha256 = hashlib.sha256(Path(model_module.__file__).read_bytes()).hexdigest()
    assert not torch.cuda.is_available()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    original = config['text_config']['max_position_embeddings']
    target = args.expected_max_length
    rope = copy.deepcopy(config['text_config']['rope_parameters'])
    rope.update(rope_type='yarn', factor=target / original,
                original_max_position_embeddings=original)
    positive = {'text_config': {'max_position_embeddings': target, 'rope_parameters': rope}}
    cases = [
        ('baseline_native_length', {}, original, True),
        ('unscaled_too_long', {}, target, False),
        ('factor_only_too_long', {'text_config': {'rope_parameters': rope}}, target, False),
        ('wrong_root_too_long', {'max_position_embeddings': target, 'rope_parameters': rope}, target, False),
        ('nested_yarn_scaled_length', positive, target, True),
    ]
    if args.overrides_json:
        cases.append(('generated_overrides', json.loads(args.overrides_json.read_text(encoding='utf-8-sig')), target, True))
    results = []
    with tempfile.TemporaryDirectory(prefix='sm75-public-config-') as tmp:
        Path(tmp, 'config.json').write_text(json.dumps(config), encoding='utf-8')
        for name, overrides, maximum, expected in cases:
            try:
                loaded = ModelConfig(model=tmp, tokenizer=tmp, max_model_len=maximum,
                    hf_overrides=overrides, skip_tokenizer_init=True, dtype='float16',
                    enforce_eager=True, language_model_only=True)
            except Exception as error:
                message = str(error).replace(tmp, '<public-config>')
                assert not expected, f'{name} unexpectedly failed: {type(error).__name__}: {message}'
                assert 'max_model_len' in message and 'derived_max_model_len' in message, message
                results.append({'name': name, 'passed': True, 'loaded': False,
                    'reason': 'max_model_len exceeds derived model limit'})
                continue
            assert expected, f'{name} unexpectedly accepted an unscaled text config'
            assert loaded.max_model_len == maximum
            text = loaded.hf_text_config
            for field in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'layer_types'):
                assert getattr(text, field) == config['text_config'][field], f'{name}: lost {field}'
            for field in ('mrope_interleaved', 'mrope_section', 'partial_rotary_factor', 'rope_theta'):
                assert text.rope_parameters[field] == config['text_config']['rope_parameters'][field], f'{name}: lost {field}'
            if maximum == target:
                assert text.max_position_embeddings == target
                assert text.rope_parameters['rope_type'] == 'yarn'
                assert text.rope_parameters['factor'] == target / original
                assert text.rope_parameters['original_max_position_embeddings'] == original
            results.append({'name': name, 'passed': True, 'loaded': True,
                'max_model_len': loaded.max_model_len, 'max_position_embeddings': text.max_position_embeddings,
                'rope_parameters': text.rope_parameters})
    # Exercise the actual upstream admission check with the model's full-attention
    # subset at TP4. No tensors/cache allocations or model forwards are involved.
    # This is a lower bound: GDN, draft, and hybrid padding are deliberately absent.
    from vllm.config import CacheConfig
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    full_layers = sum(kind == 'full_attention' for kind in config['text_config']['layer_types'])
    tp = 4
    assert config['text_config']['num_key_value_heads'] % tp == 0
    spec = FullAttentionSpec(block_size=32,
        num_kv_heads=config['text_config']['num_key_value_heads'] // tp,
        head_size=config['text_config']['head_dim'], dtype=torch.float8_e4m3fn)
    specs = {f'layer.{i}': spec for i in range(full_layers)}
    capacity_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=target),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=CacheConfig(kv_offloading_size=32, kv_offloading_backend='native'))
    needed = kv_cache_utils.max_memory_usage_bytes(capacity_config, specs.values())
    null_block = full_layers * spec.page_size_bytes
    capacity_cases = []
    for label, available, expected in [
        ('cpu32_gib_cannot_replace_gpu_capacity', needed - (1 << 30), False),
        ('raw_gpu_bytes_still_need_null_block', needed, False),
        ('raw_full_attention_and_null_block_fit', needed + null_block, True),
    ]:
        try:
            kv_cache_utils.check_enough_kv_cache_memory(capacity_config, specs, available)
        except ValueError as error:
            assert not expected, str(error)
            assert 'larger than the available KV cache memory' in str(error), str(error)
            accepted = False
        else:
            assert expected, f'{label}: unexpectedly accepted insufficient GPU capacity'
            accepted = True
        assert capacity_config.model_config.max_model_len == target
        capacity_cases.append({'name': label, 'passed': True, 'accepted': accepted,
            'cpu_offloading_gib_total': 32, 'gpu_available_bytes_per_rank': available,
            'raw_required_bytes_per_rank': needed, 'null_block_bytes_per_rank': null_block})
    report = {'vllm': vllm.__version__, 'transformers': transformers.__version__,
        'model_py_sha256': model_sha256, 'gpu_available': torch.cuda.is_available(), 'weights_loaded': False,
        'allow_long_max_model_len': False, 'cases': results, 'kv_capacity_cases': capacity_cases,
        'kv_capacity_scope': 'Synthetic raw full-attention subset, TP4, FP8, one request; excludes GDN, draft, and hybrid padding. The positive case is not a deployment memory recommendation.',
        'kv_cache_utils_sha256': hashlib.sha256(Path(kv_cache_utils.__file__).read_bytes()).hexdigest()}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
