# vllm-sm75-next-0924

A dedicated Flash-Next branch based on vLLM-SM75 v0.1.6 Ultra for eight Tesla T10 GPUs (SM75).

See [README.md](README.md) for model prerequisites, build instructions, and the recommended TP8 + EP / MTP5 / 256K / concurrency-4 configuration.

This branch adds runtime patches, not just a launch preset. The Dockerfile starts from an existing, unpatched `vllm-sm75:v0.1.6-ultra` image. Independent compatible MTP draft weights are required and are not distributed here.

The final `vllm-sm75-next-ultra-0924:latest` image was built and validated in a fresh container before the publication files were prepared. Validation covered eight-GPU TP8 + EP startup with MTP5, one 256K-total-token request, and four concurrent mixed-length requests, with no OOM or request preemption in those capacity checks.

The repository Dockerfile subsequently consolidated the original layered build procedure and has not itself been rerun. This distinction does not mean the final image or its capacity checks were left untested. Prefill parity and full production-quality validation are not claimed. No benchmark datasets or private deployment records are added by this branch.

The recommended `scripts/run-next.sh` starts the Ultra panel on host port 1615 and the inference API on host port 18001. A dedicated persistent `ULTRA_DATA_ROOT` is required. On first launch, the panel imports the tested TP8 + MTP5 / 256K / concurrency-4 configuration and automatically starts the engine. Later launches preserve panel edits and credentials. The panel manages its own API key, separate from the Web login token. API-only mode remains available as `scripts/run-next-api.sh`.

The new panel bootstrap has local initialization, argument-preservation, process-management and existing-data-protection tests. It has not yet undergone an eight-GPU container validation; previous GPU results apply to the original API launch mode.

For the general-purpose project, see [VLLM-SM75 main](https://github.com/fishensw/VLLM-SM75/tree/main).
