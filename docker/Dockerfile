# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=vllm/vllm-openai:v0.30.0-cu129@sha256:58fdb6bb123a81aa53f46fa4652ad8cc87e817bd1077c9832c6258ef12c1c688

FROM ${BASE_IMAGE} AS sm75-base
RUN uv pip uninstall --system flashinfer-jit-cache && \
    uv pip install --system --no-cache --no-deps flashinfer-python==0.6.18 flashinfer-cubin==0.6.18 \
        --extra-index-url https://flashinfer.ai/whl/

# The pinned official cu129 image contains Torch 2.14/cu130 after its optional
# connector install, while its vLLM/torchvision wheels require Torch 2.13.
# Restore the CUDA 12.9 ABI before compiling any extension; keep the official
# /etc/uv-overrides.txt NCCL pin. Resolve exact CUDA wheel and Triton versions.
RUN uv pip install --system --no-cache \
        --index https://download.pytorch.org/whl/cu129 \
        torch==2.13.0+cu129 triton==3.7.1 && \
    python3 -c 'import torch, importlib.metadata as m; assert torch.__version__ == "2.13.0+cu129"; assert torch.version.cuda == "12.9"; assert m.version("triton") == "3.7.1"'

# CUDA 12/13 wheels share .so paths. A version-only downgrade leaves overwritten
# files behind, so remove conflicting distributions and reinstall all four.
RUN uv pip uninstall --system nvidia-cudnn-cu13 nvidia-cusparselt-cu13 \
        nvidia-nccl-cu13 nvidia-nvshmem-cu13 && \
    uv pip install --system --no-cache --no-deps \
        --reinstall-package nvidia-cudnn-cu12 --reinstall-package nvidia-cusparselt-cu12 \
        --reinstall-package nvidia-nccl-cu12 --reinstall-package nvidia-nvshmem-cu12 \
        nvidia-cudnn-cu12==9.20.0.48 nvidia-cusparselt-cu12==0.8.1 \
        nvidia-nccl-cu12==2.30.7 nvidia-nvshmem-cu12==3.4.5
# Keep CUDA Python API consistent with Torch and the optional LMCache wheel.
RUN uv pip install --system --no-cache --no-deps cuda-python==12.9.7 cuda-bindings==12.9.7
COPY docker/helpers/verify_cuda_libraries.py /tmp/verify_cuda_libraries.py
RUN python3 /tmp/verify_cuda_libraries.py && rm /tmp/verify_cuda_libraries.py

FROM sm75-base AS flashqla-builder

ENV TORCH_CUDA_ARCH_LIST=7.5
ARG MAX_JOBS=1
ENV MAX_JOBS=${MAX_JOBS}
WORKDIR /opt/flashqla-build
COPY vllm/third_party/flash_qla_sm75 /opt/flashqla-build/source
RUN python3 /opt/flashqla-build/source/build_extension.py \
    --build-directory /opt/flashqla-build/work \
    --output /opt/flashqla-build/flash_qla_sm75_gdn.so \
    --verbose && \
    cuobjdump --list-elf /opt/flashqla-build/flash_qla_sm75_gdn.so \
        > /opt/flashqla-build/cuobjdump.txt && \
    test "$(grep -Eo 'sm_[0-9]+' /opt/flashqla-build/cuobjdump.txt | \
        sort -u)" = "sm_75"

FROM sm75-base AS runtime
ENV NCCL_P2P_LEVEL=SYS

# Match the reviewed startup fixes rather than resolving a moving latest release.
RUN uv pip install --system transformers==5.15.1

# Keep the production image's proven SM75 FlashInfer runtime.  In particular,
# flashinfer-jit-cache 0.6.18 contains AOT cubins without an SM75 image and
# makes BatchPrefillWithPagedKVCache fail during CUDA graph warmup.

ARG BUILD_DATE
ARG SOURCE_REVISION
ARG BASE_IMAGE_ID
ARG VLLM_SM75_RELEASE=vllm-sm75
ARG UPSTREAM_VLLM_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607
ARG ONECAT_FLASHQLA_COMMIT=187b932dbd11940f0bcf52fb3675dd47fd69f313

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE_ID}" \
      org.opencontainers.image.version="${VLLM_SM75_RELEASE}" \
      org.opencontainers.image.title="VLLM-SM75" \
      org.opencontainers.image.description="VLLM-SM75: FlashQLA-SM75 GDN prefill with Triton decode" \
      ai.vllm.upstream.commit="${UPSTREAM_VLLM_COMMIT}" \
      ai.vllm.upstream.version="v0.30.0" \
      ai.vllm.flashqla.commit="${ONECAT_FLASHQLA_COMMIT}" \
      ai.vllm.sm75.release="${VLLM_SM75_RELEASE}"

COPY vllm/device_allocator/disk_snapshot.py /tmp/sm75-overlay/device_allocator/disk_snapshot.py
COPY vllm/device_allocator/disk_sleep.py /tmp/sm75-overlay/device_allocator/disk_sleep.py
COPY vllm/config/model.py /tmp/sm75-overlay/config/model.py
COPY vllm/envs_sm75.py /tmp/sm75-overlay/envs_sm75.py
COPY vllm/v1/core/sched/scheduler_sm75.py /tmp/sm75-overlay/v1/core/sched/scheduler_sm75.py
COPY vllm/config/compilation.py /tmp/sm75-overlay/config/compilation.py
COPY vllm/engine/arg_utils.py /tmp/sm75-overlay/engine/arg_utils.py
COPY vllm/entrypoints/serve/utils/api_utils.py \
    /tmp/sm75-overlay/entrypoints/serve/utils/api_utils.py
COPY vllm/entrypoints/serve/instrumentator/metrics.py \
    /tmp/sm75-overlay/entrypoints/serve/instrumentator/metrics.py
COPY vllm/entrypoints/serve/instrumentator/monitor.py \
    /tmp/sm75-overlay/entrypoints/serve/instrumentator/monitor.py
COPY vllm/entrypoints/serve/instrumentator/dashboard.html \
    /tmp/sm75-overlay/entrypoints/serve/instrumentator/dashboard.html
COPY vllm/distributed/kv_transfer/kv_connector/v1/base.py \
    /tmp/sm75-overlay/distributed/kv_transfer/kv_connector/v1/base.py
COPY vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py \
    /tmp/sm75-overlay/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
COPY vllm/model_executor/layers/quantization/kv_cache.py \
    /tmp/sm75-overlay/model_executor/layers/quantization/kv_cache.py
COPY vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py \
    /tmp/sm75-overlay/model_executor/layers/quantization/utils/marlin_utils_fp8.py
COPY vllm/v1/attention/backends/flashinfer.py \
    /tmp/sm75-overlay/v1/attention/backends/flashinfer.py
COPY vllm/v1/attention/backends/gdn_attn.py \
    /tmp/sm75-overlay/v1/attention/backends/gdn_attn.py
COPY vllm/model_executor/kernels/linear/mixed_precision/marlin.py \
    /tmp/sm75-overlay/model_executor/kernels/linear/mixed_precision/marlin.py
COPY vllm/model_executor/kernels/linear/scaled_mm/marlin.py \
    /tmp/sm75-overlay/model_executor/kernels/linear/scaled_mm/marlin.py
COPY vllm/model_executor/kernels/linear/scaled_mm/cutlass.py \
    /tmp/sm75-overlay/model_executor/kernels/linear/scaled_mm/cutlass.py
COPY vllm/model_executor/kernels/linear/scaled_mm/humming.py \
    /tmp/sm75-overlay/model_executor/kernels/linear/scaled_mm/humming.py
COPY vllm/model_executor/layers/quantization/utils/firefly.py \
    /tmp/sm75-overlay/model_executor/layers/quantization/utils/firefly.py
COPY vllm/model_executor/layers/quantization/utils/firefly.cu \
    /tmp/sm75-overlay/model_executor/layers/quantization/utils/firefly.cu
COPY vllm/distributed/device_communicators/firefly_allreduce.py \
    /tmp/sm75-overlay/distributed/device_communicators/firefly_allreduce.py
COPY vllm/distributed/device_communicators/firefly_allreduce.cu \
    /tmp/sm75-overlay/distributed/device_communicators/firefly_allreduce.cu
COPY vllm/distributed/device_communicators/cuda_communicator.py \
    /tmp/sm75-overlay/distributed/device_communicators/cuda_communicator.py
COPY vllm/v1/engine/auto_sleep.py \
    /tmp/sm75-overlay/v1/engine/auto_sleep.py
COPY vllm/v1/engine/core.py \
    /tmp/sm75-overlay/v1/engine/core.py
COPY vllm/v1/engine/async_llm.py \
    /tmp/sm75-overlay/v1/engine/async_llm.py
COPY vllm/v1/engine/core_client.py \
    /tmp/sm75-overlay/v1/engine/core_client.py
COPY vllm/third_party/flash_qla_sm75 \
    /tmp/sm75-overlay/third_party/flash_qla_sm75
COPY docker/helpers/install_sm75_overlay.py /tmp/install_sm75_overlay.py
COPY docker/helpers/patch_transformers_startup.py /tmp/patch_transformers_startup.py
COPY docker/helpers/verify_sm75_image.py /tmp/verify_sm75_image.py
RUN mkdir -p /opt/vllm-sm75/source /opt/vllm-sm75/extensions && \
    python3 /tmp/patch_transformers_startup.py && \
    python3 /tmp/install_sm75_overlay.py /tmp/sm75-overlay \
        --source-copy /opt/vllm-sm75/source/vllm && \
    cp -a /tmp/sm75-overlay/third_party/flash_qla_sm75 \
        /opt/vllm-sm75/source/ && \
    rm -rf /tmp/sm75-overlay /tmp/install_sm75_overlay.py \
        /tmp/patch_transformers_startup.py

COPY requirements/cuda.txt /opt/vllm-sm75/source/
COPY tests/model_executor/test_sm75_gdn_selection.py \
    /opt/vllm-sm75/source/tests/model_executor/test_sm75_gdn_selection.py
COPY tests/kernels/attention/test_attention_selector.py \
    /opt/vllm-sm75/source/tests/kernels/attention/test_attention_selector.py
COPY tests/entrypoints/serve/utils/test_api_utils.py \
    /opt/vllm-sm75/source/tests/entrypoints/serve/utils/test_api_utils.py
COPY tests/config/test_model_dtype_override.py \
    /opt/vllm-sm75/source/tests/config/test_model_dtype_override.py
COPY tests/v1/engine/test_auto_sleep.py \
    /opt/vllm-sm75/source/tests/v1/engine/test_auto_sleep.py

COPY --from=flashqla-builder /opt/flashqla-build/flash_qla_sm75_gdn.so \
    /opt/vllm-sm75/extensions/flash_qla_sm75_gdn.so
COPY --from=flashqla-builder /opt/flashqla-build/cuobjdump.txt \
    /opt/vllm-sm75/evidence/cuobjdump.txt

# VLLM_FIREFLY=auto(=1, 等价): firefly 默认开, int4(AWQ/GPTQ) 走 int8 prefill
# 加速, fp8 走上游 marlin(sm75 实测 firefly-fp8 不赚, fp8 加速走 VLLM_FIREFLY_AR)。
# 纯 marlin 基线 -e VLLM_FIREFLY=0 关。MIN_M 用 envs.py 默认(1024)。
ENV FLASH_QLA_SM75_PREBUILT_EXTENSION_PATH=/opt/vllm-sm75/extensions/flash_qla_sm75_gdn.so \
    FLASH_QLA_SM75_ALLOW_JIT=0 \
    PYTHONWARNINGS=ignore::FutureWarning \
    TRANSFORMERS_VERBOSITY=error \
    VLLM_FIREFLY=auto \
    VLLM_USE_MODELSCOPE=true \
    MODELSCOPE_CACHE=/root/.cache/modelscope/hub

RUN python3 /tmp/verify_sm75_image.py && \
    test -s /opt/vllm-sm75/extensions/flash_qla_sm75_gdn.so && \
    test -s /opt/vllm-sm75/source/flash_qla_sm75/LICENSE && \
    test -s /opt/vllm-sm75/source/flash_qla_sm75/SOURCE.md && \
    rm -f /tmp/verify_sm75_image.py

COPY docker/speculative /opt/vllm-sm75/speculative
COPY docker/helpers/install_speculative.py /tmp/install_speculative.py
RUN python3 /tmp/install_speculative.py && rm /tmp/install_speculative.py
COPY docker/speculative /opt/vllm-sm75/source/docker/speculative
COPY tests /opt/vllm-sm75/source/tests

FROM runtime AS selector-tests

RUN uv pip install --system pytest==9.1.0 && \
    python3 -m pytest \
        /opt/vllm-sm75/source/tests/kernels/attention/test_attention_selector.py::test_flashinfer_compute_capability_includes_sm75 \
        /opt/vllm-sm75/source/tests/model_executor/test_sm75_gdn_selection.py \
        /opt/vllm-sm75/source/tests/entrypoints/serve/utils/test_api_utils.py::test_sensitive_args_are_redacted_without_mutating_input \
        /opt/vllm-sm75/source/tests/config/test_model_dtype_override.py \
        /opt/vllm-sm75/source/tests/v1/engine/test_auto_sleep.py \
        /opt/vllm-sm75/source/tests/model_executor/test_dflash_quantization.py \
        /opt/vllm-sm75/source/tests/model_executor/test_dflash_warmup.py \
        /opt/vllm-sm75/source/tests/model_executor/test_v030_firefly_api.py \
        /opt/vllm-sm75/source/tests/model_executor/test_v030_cutlass_fp8_selection.py \
        /opt/vllm-sm75/source/tests/model_executor/test_v030_humming_fp8_selection.py \
        /opt/vllm-sm75/source/tests/entrypoints/serve/instrumentator/test_monitor_control_api.py \
        -v > /tmp/selector-pytest.log 2>&1 || { cat /tmp/selector-pytest.log; exit 1; }

FROM runtime AS verified-runtime

COPY --from=selector-tests /tmp/selector-pytest.log \
    /opt/vllm-sm75/evidence/selector-pytest.log


# All modes use the same image; runtime configuration selects speculation.
FROM verified-runtime AS final
# Inherit the official ENTRYPOINT ["vllm", "serve"] and unset CMD.
# Keep upstream CLI startup presentation and default UI assets.

# 容器内 pstate 管理: 固定版本二进制 + 启动包装脚本(GPL-3.0, 见 NOTICE)。
# 容器运行时不会挂载 libnvidia-api.so.1, run.sh 在 POWER_MODE=pstate 时只读挂载宿主库。
ARG NVIDIA_PSTATE_VERSION=1.0.9
ARG NVIDIA_PSTATE_SHA256=4bf5b5eba202c65f5955e2981d184b4b4baa833d1d5df36b062445d736d1eda6
COPY docker/helpers/pstate-entrypoint.sh /opt/vllm-sm75/pstate-entrypoint.sh
COPY docker/helpers/pstate-supervisor.sh /opt/vllm-sm75/pstate-supervisor.sh
COPY docker/helpers/nvidia-pstated.NOTICE /opt/vllm-sm75/nvidia-pstated.NOTICE
RUN curl -fsSL -o /usr/local/bin/nvidia-pstated \
        "https://github.com/sasha0552/nvidia-pstated/releases/download/v${NVIDIA_PSTATE_VERSION}/nvidia-pstated" && \
    echo "${NVIDIA_PSTATE_SHA256}  /usr/local/bin/nvidia-pstated" | sha256sum -c - && \
    chmod +x /usr/local/bin/nvidia-pstated /opt/vllm-sm75/pstate-entrypoint.sh /opt/vllm-sm75/pstate-supervisor.sh

ENV SM75_FA2_SMALLQ_GRAPH=1 SM75_GDN_B1_METADATA=1

LABEL ai.vllm.sm75.modes="base,mtp,dflash2"
