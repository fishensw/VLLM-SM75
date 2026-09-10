"""Local SM75 native-FA2 small-query FULL-Graph capability and stable buffers.

Keeps FlashInfer's graph-aware automatic split-KV planner. Does not pretend
TRTLLM/XQA is available, change attention arithmetic, or route long prefills to
the small-query path. Each (batch, query-size, causal) capture owns its metadata.
"""
import math
import os
import torch


def storage_supported(dtype,cache_dtype):
    # vLLM can store FP8 bytes as uint8 while FlashInfer operates on E4M3.
    return dtype in (torch.float16,torch.float8_e4m3fn) or (
        dtype==torch.uint8 and cache_dtype in ('fp8','fp8_e4m3'))


def uniform_smallq(batch,max_query_len,num_tokens):
    # Native FI splits single-token decodes from prefills. Mixed batches must
    # use its original wrapper; a buffer sized for all requests is not the
    # same size as the prefill-only sub-batch passed to wrapper.plan().
    return (1<=batch<=8 and 2<=max_query_len<=8
            and num_tokens==batch*max_query_len)


def captured_query_rows(num_spec_tokens,capture_sizes,max_capture_size):
    query=num_spec_tokens+1
    if not 1<=num_spec_tokens<=7 or not max_capture_size:
        return query,frozenset()
    rows=frozenset(((s+query-1)//query)*query for s in (capture_sizes or [])
        if s>0 and ((s+query-1)//query)*query<=max_capture_size
        and ((s+query-1)//query)<=8)
    return query,rows


def capture_config(config):
    sd=config.speculative_config
    cc=config.compilation_config
    return captured_query_rows(sd.num_speculative_tokens if sd else 0,
        cc.cudagraph_capture_sizes,cc.max_cudagraph_capture_size)


def sync_attention_cache_layout(config):
    """Propagate the engine's resolved layout into draft dtype config copies."""
    from vllm.v1.attention.backends.utils import record_kv_cache_layout

    layout = config.cache_config.get_resolved_kv_cache_layout().name
    for layer in config.compilation_config.static_forward_context.values():
        cache_config = getattr(getattr(layer, 'impl', None), 'cache_config', None)
        if cache_config is not None and cache_config is not config.cache_config:
            record_kv_cache_layout(cache_config, layout)


def install():
    if os.environ.get('SM75_FA2_SMALLQ_GRAPH','0')!='1':return
    import vllm.v1.attention.backends.flashinfer as fi
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.kv_cache_interface import AttentionSpec,UniformTypeKVCacheSpecs
    from vllm.platforms import current_platform
    from vllm import envs
    from vllm.logger import init_logger
    log=init_logger('vllm.sm75_fa2_graph')
    cls=fi.FlashInferMetadataBuilder
    old_support=cls.get_cudagraph_support
    old_build=cls.build
    old_wrapper=cls._get_prefill_wrapper

    def supports(config,spec):
        sd=config.speculative_config
        query,capture_rows=capture_config(config)
        if not (current_platform.is_device_capability(75)
                and config.model_config.dtype==torch.float16
                and config.parallel_config.decode_context_parallel_size==1
                and not envs.VLLM_BATCH_INVARIANT
                and sd is not None and 1<=sd.num_speculative_tokens<=7
                and capture_rows==frozenset((query,))):
            return False
        specs=spec.kv_cache_specs.values() if isinstance(spec,UniformTypeKVCacheSpecs) else [spec]
        attention=[s for s in specs if isinstance(s,AttentionSpec)]
        return bool(attention) and all(s.head_size in (128,256)
            and not s.kv_quant_mode.is_nvfp4
            and storage_supported(s.dtype,config.cache_config.cache_dtype) for s in attention)

    @classmethod
    def graph_support(builder_cls,config,spec):
        if supports(config,spec):
            log.info_once('SM75 native FA2 small-query capability: UNIFORM_BATCH / FULL Graph')
            return AttentionCGSupport.UNIFORM_BATCH
        return old_support(config,spec)

    def build(self,common_prefix_len,common_attn_metadata,fast_build=False):
        m=common_attn_metadata
        if not hasattr(self,'_sm75_capture_rows'):
            self._sm75_capture_query,self._sm75_capture_rows=capture_config(self.vllm_config)
        self._sm75_smallq_key=None
        if (current_platform.is_device_capability(75) and not self.use_dcp
                and not self.has_sinks and not self.is_kvcache_nvfp4
                and self.q_data_type_prefill==torch.float16
                and self.kv_cache_dtype in (torch.float16,torch.float8_e4m3fn)
                and self.head_dim in (128,256)
                and uniform_smallq(m.num_reqs,m.max_query_len,m.num_actual_tokens)
                # This first demo accelerates B1 only. Other batches keep the
                # original dynamic wrapper, rather than allocating per-shape
                # Graph workspaces that will never be replayed.
                and self.enable_cuda_graph and m.num_reqs==1
                and m.max_query_len==self._sm75_capture_query
                and self._sm75_capture_rows==frozenset((self._sm75_capture_query,))
                and common_prefix_len==0
                and self.prefill_fixed_split_size in (-1,None)
                and not self.disable_split_kv):
            self._sm75_smallq_key=(m.num_reqs,m.max_query_len)
        try:
            return old_build(self,common_prefix_len,common_attn_metadata,fast_build)
        finally:
            self._sm75_smallq_key=None

    def wrapper(self,causal=True):
        key=getattr(self,'_sm75_smallq_key',None)
        if key is None:return old_wrapper(self,causal)
        key=(*key,bool(causal))
        if not hasattr(self,'_sm75_graph_prefills'):self._sm75_graph_prefills={}
        if key not in self._sm75_graph_prefills:
            batch,query,_=key
            pages=batch*math.ceil(self.model_config.max_model_len/self.page_size)
            def buf(n):return torch.empty(n,dtype=torch.int32,device=self.device)
            self._sm75_graph_prefills[key]=fi.BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),fi.get_flashinfer_layout_string(self.kv_cache_layout),backend='fa2',use_cuda_graph=True,
                qo_indptr_buf=buf(batch+1),paged_kv_indptr_buf=buf(batch+1),
                paged_kv_indices_buf=buf(pages),paged_kv_last_page_len_buf=buf(batch))
            log.info('SM75 native FA2 Graph buffers: B=%d q=%d heads=%d/%d D=%d page=%d causal=%s',
                batch,query,self.num_qo_heads,self.num_kv_heads,self.head_dim,self.page_size,causal)
        return self._sm75_graph_prefills[key]

    cls.get_cudagraph_support=graph_support
    cls.build=build
    cls._get_prefill_wrapper=wrapper
    log.info('Local SM75 FA2 small-query Graph adapter enabled; automatic split-KV retained')
