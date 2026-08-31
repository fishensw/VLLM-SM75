# FlashQLA SM75 source provenance

This directory vendors the SM70/SM75 CUDA GDN implementation from:

- Repository: `https://github.com/1CatAI/1Cat-vLLM`
- Commit: `187b932dbd11940f0bcf52fb3675dd47fd69f313`
- Original path: `flash_qla/ops/gated_delta_rule/chunk/sm70/`
- License: MIT; see `LICENSE` in this directory.

The production image compiles this source only for `sm_75`. Internal `sm70`
function names are retained to keep comparison with the fixed upstream source
straightforward. The vLLM-facing selector is named `flashqla_sm75` and does not
select the speculative/DDTree route.

Local integration changes are limited to the loader, packaging, and naming;
the CUDA math and launch heuristics are unchanged:

- prefer an image-owned prebuilt extension;
- disable runtime JIT unless `FLASH_QLA_SM75_ALLOW_JIT=1` is explicit;
- compile only `compute_75,sm_75`;
- use SM75-specific environment and extension names;
- rename SM70 environment-variable strings and user-facing CUDA diagnostics to
  SM75 while retaining the original internal function names.
