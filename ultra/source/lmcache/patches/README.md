# LMCache 0.5.5 packed KV page compatibility patch

`kv_cache_group_edits.py` is vendored under Apache-2.0 from LMCache, with a
minimal backport of the packed 4D attention rule in upstream
[PR #4731](https://github.com/LMCache/LMCache/pull/4731), head
`f180b9ffce7df45ce3037011d95a22db947fefcb`. The PR was **open and unmerged** when
checked on 2026-09-23. This is a project compatibility patch, not an upstream
released fix. The base is tag `v0.5.5-cu129`, commit
`05a013b29da78cf2321b9b46ec5039dde2fb0bb0`.

The patch copies the PR geometry helpers and packed-4D matching rule, then
locally adapts the new rule's `apply` method to vLLM 0.30. It is **not a verbatim
PR backport**. The new rule is registered after the existing rules. It preserves the original 5D attention,
MLA and Mamba implementations; it does not backport the PR's refactoring of
those rules. `packed4d.patch` is the complete difference against the base file.
`packed4d-provenance.json` pins the original, patched and diff SHA256 values.
`LICENSE` is the upstream Apache-2.0 license. The installer must reject unknown
installed module contents, and the runtime probe must require the patched hash.

The stock rule recognizes only 5D full-attention tensors. On the observed
vLLM 0.30 hybrid layout, a full-attention view of `(4067, 32, 1, 512)` uint8
is the downstream normalized view, and contains 83 logical blocks of 1568 tokens, with 49 kernel pages per block.
The actual registered view is `(4067, 1, 32, 512)` with hint `NHD`: vLLM
0.30 always registers logical `[B,H,N,C]`, while the hint describes physical
strides. The unmodified PR selected the wrong raw axis and failed closed. The
local adaptation validates the head count and content bytes against the real
spec, then uses a metadata-only NHD permutation (or HND identity) before the
original strict page checks. This follows `create_kv_cache_views` in fixed
vLLM commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

Without this edit LMCache treats the smaller kernel page as slot compression,
and transfers only a fraction of the logical block. The patch reinterprets the
same bytes as `(83, 1568, 1, 512)` without copying, so block IDs address complete
logical pages. See [issue #4701](https://github.com/LMCache/LMCache/issues/4701).

The rule is limited to hybrid models and supported non-MLA packed attention.
Declared slot compression keeps its existing path. It requires an NHD or HND
layout hint, contiguous storage, integral page ratios, and exact byte coverage;
invalid geometry fails instead of guessing. The resulting dimensions describe
opaque transfer pages, so content-aware transformations are not justified.

Run `../tests/test_packed_kv_layout.py` with the real vLLM 0.30 / Torch CPU
runtime and `CUDA_VISIBLE_DEVICES=''`. These tests cover the observed shape,
all 49 kernel-page payloads, byte offsets and sibling storage, logical block
remapping, NHD/HND, uint8/fp16/bf16, invalid layouts, and the unchanged 5D/GDN
paths. Registered tensors are constructed by the real vLLM
`create_kv_cache_views`, including multi-head NHD views that are logically
non-contiguous. Actual LMCache format detection verifies the unchanged
ratio-one and non-hybrid paths; the original dispatcher scope is preserved. Full-model GPU cold/warm, disk restore, server restart, salt isolation
and alternating-prefix tests remain separate acceptance requirements. A CPU
geometry pass alone does not establish end-to-end cache correctness.
