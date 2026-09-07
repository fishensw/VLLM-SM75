"""B1 uniform speculative GDN metadata staging, preserving native fallback.

Only replaces metadata preparation, not the GDN arithmetic. Copies exact GPU
sequence/accepted counts; never substitutes scheduler upper bounds.
"""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _stage(TABLE, SEQ, QUERY, ACCEPTED, STATE, MASK, TOKENS, QLOC, OUT_ACCEPTED,
           Q: tl.constexpr, BLOCK_SIZE: tl.constexpr, ALIGN: tl.constexpr):
    i = tl.arange(0, 32)
    start = tl.full((), 0, tl.int32)
    if ALIGN:
        length = tl.load(SEQ)
        start = tl.maximum((length - 1) // BLOCK_SIZE, 0)
    state = tl.load(TABLE + start + i, i < Q, 0)
    tl.store(STATE + i, state, i < Q)
    tl.store(TOKENS + i, i, i < Q)
    query = tl.load(QUERY + i, i < 2, 0)
    tl.store(QLOC + i, query, i < 2)
    accepted = tl.load(ACCEPTED)
    tl.store(OUT_ACCEPTED + i, accepted, i == 0)
    tl.store(MASK + i, True, i == 0)


def eligible(builder, m, accepted, drafts, common_prefix_len=0):
    q = builder.num_spec + 1
    if not (common_prefix_len == 0 and builder.use_full_cuda_graph
            and 1 <= builder.num_spec <= 7 and m.num_reqs == 1
            and m.num_actual_tokens == q and m.max_query_len == q
            and builder.decode_cudagraph_max_bs >= q
            and accepted is not None and accepted.is_cuda and accepted.numel() == 1
            and drafts is not None and drafts.device.type == 'cpu' and drafts.numel() == 1
            and m.query_start_loc_cpu.device.type == 'cpu'
            and m.query_start_loc_cpu.numel() == 2
            and m.block_table_tensor.is_cuda and m.block_table_tensor.ndim == 2
            and m.block_table_tensor.shape[0] == 1 and m.block_table_tensor.stride(1) == 1
            and m.block_table_tensor.shape[1] >= q
            and m.seq_lens.is_cuda and m.seq_lens.numel() == 1
            and m.query_start_loc.is_cuda and m.query_start_loc.numel() == 2
            and m.query_start_loc.stride(0) == 1
            and accepted.device == m.seq_lens.device == m.query_start_loc.device
                == m.block_table_tensor.device == builder.spec_state_indices_tensor.device):
        return False
    mode = builder.vllm_config.cache_config.mamba_cache_mode
    if mode not in ('all', 'none', 'align'):
        return False
    if mode == 'align' and builder.kv_cache_spec.num_speculative_blocks < builder.num_spec:
        return False
    if mode == 'align' and (max((m.max_seq_len-1)//builder.kv_cache_spec.block_size,0)+q
                            > m.block_table_tensor.shape[1]):
        return False
    return (drafts.item() == builder.num_spec
            and m.query_start_loc_cpu[0].item() == 0
            and m.query_start_loc_cpu[1].item() == q)


def build_fast(builder, m, accepted):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    q = builder.num_spec + 1
    _stage[(1,)](m.block_table_tensor, m.seq_lens, m.query_start_loc, accepted,
                 builder.spec_state_indices_tensor, builder.spec_sequence_masks,
                 builder.spec_token_indx, builder.spec_query_start_loc,
                 builder.num_accepted_tokens, q, builder.kv_cache_spec.block_size,
                 builder.vllm_config.cache_config.mamba_cache_mode == 'align',
                 num_warps=1)
    return GDNAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=1, num_spec_decode_tokens=q, num_actual_tokens=q,
        spec_query_start_loc=builder.spec_query_start_loc[:2],
        spec_state_indices_tensor=builder.spec_state_indices_tensor[:1],
        spec_sequence_masks=builder.spec_sequence_masks[:1],
        spec_token_indx=builder.spec_token_indx[:q],
        non_spec_token_indx=builder.non_spec_token_indx[:0],
        num_accepted_tokens=builder.num_accepted_tokens[:1])


def install():
    if os.environ.get('SM75_GDN_B1_METADATA', '0') != '1':
        return
    from vllm.platforms import current_platform
    if not current_platform.is_device_capability(75):
        return
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.logger import init_logger
    log=init_logger('vllm.sm75_gdn_metadata')
    original = GDNAttentionMetadataBuilder.build
    def build(self, common_prefix_len, common_attn_metadata,
              num_accepted_tokens=None, num_decode_draft_tokens_cpu=None, fast_build=False):
        if (self.num_spec in (5,6,7)
                and self.vllm_config.parallel_config.decode_context_parallel_size == 1
                and eligible(self, common_attn_metadata, num_accepted_tokens,
                    num_decode_draft_tokens_cpu, common_prefix_len)):
            if not getattr(self,'_sm75_metadata_fast_logged',False):
                log.info('SM75 GDN B1 metadata fast path: q=%d cache=%s',
                         self.num_spec+1,self.vllm_config.cache_config.mamba_cache_mode)
                self._sm75_metadata_fast_logged=True
            return build_fast(self, common_attn_metadata, num_accepted_tokens)
        return original(self, common_prefix_len, common_attn_metadata,
                        num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build)
    GDNAttentionMetadataBuilder.build = build
    log.info('Local SM75 B1 GDN metadata fusion enabled; mixed/prefill/batch fallback retained')
