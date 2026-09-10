#!/usr/bin/env bash
set -euo pipefail
# Run with bash on a Linux Docker host. Does not use GPUs or stop containers.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'source-archive')}"
command -v docker >/dev/null
command -v git >/dev/null
BASE_IMAGE='vllm/vllm-openai:v0.29.0-cu129@sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2'
# Reuse the official runtime; compile only the SM75 extension.
docker build --file "$ROOT/docker/Dockerfile.vllm-sm75-v0.1.4" \
    --target final --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg MAX_JOBS="${MAX_JOBS:-1}" \
    --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --build-arg SOURCE_REVISION="$SOURCE_REVISION" \
    --build-arg BASE_IMAGE_ID="$BASE_IMAGE" \
    --tag vllm-sm75:v0.1.4 "$ROOT"
