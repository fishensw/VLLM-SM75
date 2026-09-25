# vllm-sm75-next-0924

A dedicated Flash-Next branch based on vLLM-SM75 v0.1.6 Ultra for eight Tesla T10 GPUs (SM75).

See [README.md](README.md) for model prerequisites, build instructions, and the recommended TP8 + EP / MTP5 / 256K / concurrency-4 configuration.

This branch adds runtime patches, not just a launch preset. The Dockerfile starts from an existing, unpatched `vllm-sm75:v0.1.6-ultra` image. Independent compatible MTP draft weights are required and are not distributed here.

The consolidated Dockerfile still requires a fresh build validation. Prefill parity and full production-quality validation are not claimed. No benchmark datasets or private deployment records are added by this branch.

For the general-purpose project, see [VLLM-SM75 main](https://github.com/fishensw/VLLM-SM75/tree/main).
