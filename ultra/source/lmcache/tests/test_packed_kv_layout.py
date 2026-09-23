"""CPU regression for the fixed LMCache 0.5.5 packed-page backport.

Run with the real vLLM 0.30 / Torch runtime, without a GPU or model weights:
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python test_packed_kv_layout.py -v

This exercises real FullAttentionSpec, MambaSpec, KVCacheConfig and CPU tensors.
A successful CPU test does not replace full-model GPU cache-restore validation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import torch
import vllm
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detection import detect_format
from lmcache.v1.gpu_connector.utils import get_block_size, get_num_blocks
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KVCacheLayout,
    create_kv_cache_views,
    MambaSpec,
)

PATCH_DIR = Path(__file__).resolve().parents[1] / "patches"
module_spec = importlib.util.spec_from_file_location(
    "sm75_lmcache_packed_layout", PATCH_DIR / "kv_cache_group_edits.py"
)
assert module_spec is not None and module_spec.loader is not None
edits = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(edits)


def fa_spec(block=1568, heads=1, size=256, dtype=torch.uint8, padded=None):
    return FullAttentionSpec(
        block_size=block,
        num_kv_heads=heads,
        head_size=size,
        dtype=dtype,
        page_size_padded=padded,
    )


def mamba_spec(block=1568):
    return MambaSpec(
        block_size=block,
        shapes=((block, 16),),
        dtypes=(torch.uint8,),
        mamba_cache_mode="align",
    )


def config(spec, blocks=3, hybrid=True):
    groups = [KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)]
    if hybrid and not isinstance(spec, MambaSpec):
        groups.append(KVCacheGroupSpec(layer_names=[], kv_cache_spec=mamba_spec(spec.block_size)))
    return KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=groups)


def edit(raw, spec=None, layout="NHD", hybrid=True):
    return edits.apply_kv_cache_group_edits(
        config(spec or fa_spec(), hybrid=hybrid),
        {"layer.0": raw},
        {"kv_layout": layout},
    )["layer.0"]


def packed(spec, kernel=32, blocks=3, layout="NHD"):
    # Use the real vLLM allocator view constructor. Raw axes remain BHNC,
    # including NHD layouts, whose physical order is encoded only in strides.
    size = blocks * spec.page_size_bytes
    backing = torch.empty(size, dtype=torch.int8)
    placement = KVCacheTensor(
        size=size, layers=["layer.0"], layer_stride=size,
        block_stride=spec.page_size_bytes,
    )
    order = KVCacheLayout.LBNHC if layout == "NHD" else KVCacheLayout.LBHNC
    raw = create_kv_cache_views(
        backing, spec, blocks, order, placement, kernel_block_size=kernel,
    )[0]
    pages = blocks * (spec.block_size // kernel)
    raw.copy_(((torch.arange(pages, dtype=torch.int32) % 251) + 1).to(spec.dtype).view(-1, 1, 1, 1))
    return raw


def physical(raw, layout="NHD"):
    return raw.permute(0, 2, 1, 3) if layout == "NHD" else raw


def detected(raw, layout="NHD"):
    return detect_format([raw], EngineType.VLLM, {"kv_layout": layout})


class PackedKVLayoutTests(unittest.TestCase):
    def test_exact_vendor_identity(self):
        meta = json.loads((PATCH_DIR / "packed4d-provenance.json").read_text())
        actual = hashlib.sha256((PATCH_DIR / "kv_cache_group_edits.py").read_bytes()).hexdigest()
        self.assertEqual(actual, meta["patched_file_sha256"])
        self.assertNotEqual(actual, meta["base_file_sha256"])
        self.assertEqual(meta["upstream_pr_head"], "f180b9ffce7df45ce3037011d95a22db947fefcb")

    def test_observed_tp4_fp8_geometry_keeps_all_49_kernel_pages(self):
        spec = fa_spec()
        self.assertEqual(spec.page_size_bytes, 802816)
        raw = packed(spec, blocks=83)
        self.assertEqual(tuple(raw.shape), (4067, 1, 32, 512))
        self.assertEqual(tuple(raw.stride()), (16384, 512, 512, 1))
        self.assertEqual(raw.numel(), 66633728)
        # The stock registry has no matching edit for this exact real shape.
        stock_rules = [rule for rule in edits._EDITS if rule.name != "subpaged-packed-attention-view"]
        self.assertFalse(any(rule.matches(spec, raw) for rule in stock_rules))
        got = edit(raw, spec)
        self.assertEqual(tuple(got.shape), (83, 1568, 1, 512))
        self.assertEqual(got.data_ptr(), raw.data_ptr())
        self.assertEqual(got.storage_offset(), raw.storage_offset())
        self.assertEqual(got.numel(), raw.numel())
        for block in (0, 1, 41, 82):
            expected = raw[block * 49 : (block + 1) * 49].flatten()
            self.assertTrue(torch.equal(got[block].flatten(), expected))
            self.assertEqual(got[block].numel() * got.element_size(), spec.page_size_bytes)
            self.assertEqual(torch.unique(got[block]).numel(), 49)

    def test_fp8_spec_with_uint8_registered_storage_preserves_bytes(self):
        # vLLM may publish an FP8 dtype while its unified registered page is
        # byte-addressed. Geometry depends on actual bytes, not float values.
        spec = fa_spec(dtype=torch.float8_e4m3fn)
        raw = packed(fa_spec(), blocks=2)
        got = edit(raw, spec)
        self.assertEqual(spec.page_size_bytes, 802816)
        self.assertEqual(tuple(got.shape), (2, 1568, 1, 512))
        self.assertEqual(got.dtype, torch.uint8)
        self.assertEqual(got.data_ptr(), raw.data_ptr())
        self.assertTrue(torch.equal(got.flatten(), raw.flatten()))

    def test_nhd_hnd_dtype_and_kernel_geometry_matrix(self):
        for layout in ("NHD", "HND"):
            for dtype in (torch.uint8, torch.float16, torch.bfloat16):
                for block, kernel in ((544, 32), (1600, 64)):
                    with self.subTest(layout=layout, dtype=dtype, block=block):
                        spec = fa_spec(block=block, heads=2, size=16, dtype=dtype)
                        raw = packed(spec, kernel=kernel, layout=layout)
                        self.assertEqual(raw.shape[1], 2)
                        self.assertEqual(raw.is_contiguous(), layout == "HND")
                        raw[:, 1].add_(7)
                        source = physical(raw, layout)
                        got = edit(raw, spec, layout)
                        shape = (3, block, 1, 64) if layout == "NHD" else (3, 1, block, 64)
                        self.assertEqual(tuple(got.shape), shape)
                        self.assertEqual(got.data_ptr(), raw.data_ptr())
                        self.assertTrue(torch.equal(got.view(torch.uint8).flatten(), source.view(torch.uint8).flatten()))
                        fmt, normalized = detected(got, layout)
                        self.assertEqual(get_block_size(normalized, fmt), block)
                        self.assertEqual(get_num_blocks(normalized, fmt), 3)
                        ratio = block // kernel
                        for i in range(3):
                            self.assertTrue(torch.equal(got[i].flatten(), source[i * ratio : (i + 1) * ratio].flatten()))

    def test_nonzero_offset_and_larger_storage_do_not_include_sibling_bytes(self):
        spec = fa_spec(block=64, size=4)
        payload = 4 * 32 * 8
        backing = torch.full((payload + 256,), 237, dtype=torch.uint8)
        raw = backing[128 : 128 + payload].view(4, 1, 32, 8)
        raw.fill_(17)
        got = edit(raw, spec)
        self.assertEqual(got.storage_offset(), 128)
        self.assertEqual(got.numel(), payload)
        self.assertEqual(got.data_ptr(), raw.data_ptr())
        got[1].fill_(42)
        self.assertTrue(torch.all(backing[:128] == 237))
        self.assertTrue(torch.all(backing[128 + payload :] == 237))
        self.assertTrue(torch.all(raw[:2] == 17))
        self.assertTrue(torch.all(raw[2:] == 42))

    def test_logical_block_store_and_load_cover_the_complete_raw_page(self):
        spec = fa_spec(block=544, size=8)
        raw = packed(spec)
        logical = edit(raw, spec)
        # Simulate opaque block transfer to different destination block IDs.
        stored = [logical[i].clone() for i in (2, 0)]
        restored_raw = torch.zeros_like(raw)
        restored = edit(restored_raw, spec)
        restored[0].copy_(stored[0])
        restored[2].copy_(stored[1])
        ratio = 17
        self.assertTrue(torch.equal(restored_raw[:ratio], raw[2 * ratio : 3 * ratio]))
        self.assertTrue(torch.equal(restored_raw[2 * ratio :], raw[:ratio]))
        self.assertEqual(torch.count_nonzero(restored_raw[ratio : 2 * ratio]).item(), 0)

    def test_missing_or_unknown_layout_is_rejected(self):
        for layout in ("none", "", "HND-unknown"):
            with self.subTest(layout=layout), self.assertRaisesRegex(ValueError, "Unsupported kv_layout"):
                edit(packed(fa_spec()), layout=layout)

    def test_nondivisible_logical_block_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a multiple of kernel block size"):
            edit(torch.zeros((4, 1, 32, 512), dtype=torch.uint8), fa_spec(block=65))

    def test_incomplete_logical_page_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kernel page count"):
            edit(torch.zeros((50, 1, 32, 512), dtype=torch.uint8))

    def test_page_byte_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "do not tile the logical page"):
            edit(packed(fa_spec()), fa_spec(padded=802817))

    def test_noncontiguous_kernel_pages_are_rejected(self):
        spec = fa_spec(block=64, heads=2, size=8)
        raw = torch.zeros((4, 2, 32, 32), dtype=torch.uint8)[..., ::2]
        self.assertFalse(raw.is_contiguous())
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            edit(raw, spec)

    def test_ratio_one_and_uniform_paths_use_stock_downstream_normalization(self):
        for hybrid in (True, False):
            for layout in ("NHD", "HND"):
                for heads in (1, 2):
                    with self.subTest(hybrid=hybrid, layout=layout, heads=heads):
                        spec = fa_spec(block=64, heads=heads, size=8)
                        raw = packed(spec, kernel=64, layout=layout)
                        raw[:, -1].add_(11)
                        # The original early return / unmatched path is retained.
                        got = edit(raw, spec, layout, hybrid=hybrid)
                        self.assertIs(got, raw)
                        fmt, normalized = detected(got, layout)
                        self.assertEqual(get_block_size(normalized, fmt), 64)
                        self.assertEqual(get_num_blocks(normalized, fmt), 3)
                        self.assertEqual(normalized[0].data_ptr(), raw.data_ptr())
                        self.assertTrue(torch.equal(normalized[0].flatten(), physical(raw, layout).flatten()))

    def test_wrong_spec_axis_is_rejected_before_geometry_guessing(self):
        spec = fa_spec(block=64, heads=2, size=8)
        raw = packed(spec)
        with self.assertRaisesRegex(ValueError, r"logical \[B, H, N, C\]"):
            edit(physical(raw), spec)

    def test_wrong_layout_hint_does_not_copy_or_guess(self):
        spec = fa_spec(block=64, heads=2, size=8)
        raw = packed(spec, layout="NHD")
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            edit(raw, spec, "HND")

    def test_legacy_5d_rule_preserves_all_bytes(self):
        spec = fa_spec(block=544, heads=2, size=8, dtype=torch.bfloat16)
        raw = torch.arange(3 * 17 * 2 * 32 * 2 * 8, dtype=torch.float32).to(torch.bfloat16).view(51, 2, 32, 2, 8)
        got = edit(raw, spec)
        self.assertEqual(tuple(got.shape), (3, 2, 544, 1, 16))
        self.assertEqual(got.data_ptr(), raw.data_ptr())
        self.assertTrue(torch.equal(got.view(torch.uint8).flatten(), raw.view(torch.uint8).flatten()))

    def test_gdn_keeps_its_existing_rule(self):
        spec = mamba_spec(block=64)
        raw = torch.arange(3 * 64 * 16, dtype=torch.int32).to(torch.uint8).view(3, 1, 1, 64 * 16)
        self.assertFalse(edits._SubpagedPackedAttentionViewEdit().matches(spec, raw))
        got = edit(raw, spec)
        self.assertEqual(tuple(got.shape), (3, 64, 1, 16))
        self.assertEqual(got.data_ptr(), raw.data_ptr())
        self.assertTrue(torch.equal(got.flatten(), raw.flatten()))


if __name__ == "__main__":
    print(json.dumps({"torch": torch.__version__, "vllm": vllm.__version__, "device": "cpu", "vendor_sha256": hashlib.sha256((PATCH_DIR / "kv_cache_group_edits.py").read_bytes()).hexdigest()}), flush=True)
    unittest.main()
