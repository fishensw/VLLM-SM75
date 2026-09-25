ARG BASE_IMAGE=vllm-sm75:v0.1.6-ultra
FROM ${BASE_IMAGE}
COPY patches/ /test/
RUN python3 /test/patch_baseline.py && \
    python3 /test/patch_hc_candidate.py && \
    python3 /test/patch_mtp_fp16.py && \
    python3 /test/patch_ple_bytes.py && \
    python3 /test/patch_mtp_norm.py
COPY config/selected-config.json serve.py /opt/flashnext/
COPY tests/next/ /opt/flashnext/tests/
ENV OMP_NUM_THREADS=1 VLLM_SM75_QWEN38_HC_GEMV=1 VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728
LABEL next.scope="experimental-flashnext-sm75-tp8" next.release="0924"
RUN python3 -m py_compile /opt/flashnext/serve.py
