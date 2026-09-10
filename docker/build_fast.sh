#!/usr/bin/env bash
# 构建 fast 基础镜像 vllm-sm75-fast:v0.1.4(底座 0.29 + flashinfer + transformers
# + patch + 预编译 flashqla + 软编译工具)。overlay .py 不在此,
# 由容器内 /opt/vllm-sm75/fast_compile.sh 从挂载项目目录软编译。
# 迭代: 改代码 -> 同步项目目录到挂载点 -> 容器内 fast_compile.sh -> docker restart。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v docker >/dev/null
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.29.0-cu129@sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2}"
docker build --file "$ROOT/docker/Dockerfile.fast.vllm-sm75-v0.1.4" \
    --target final \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg MAX_JOBS="${MAX_JOBS:-1}" \
    --tag vllm-sm75-fast:v0.1.4 "$ROOT"
