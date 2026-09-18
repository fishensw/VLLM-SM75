# 完整构建与长上下文验收

> 脱敏说明：部署地址、容器名及宿主路径已替换为示例占位；验证结论与哈希保持原值。

## 当前边界

本轮从官方固定基座重新构建 standard，再由同一构建图中的 `final` 阶段构建 ultra。源码包固定为 `42626f015626c25eb5fe1310c78e5d2ed59b8d99`；后续增加的验收工具不改变该镜像的运行时代码。

两个完整镜像已经构建、归档和导入成功；Ultra 16 项、环境对齐后的 standard 6 项 GPU 验收通过。首次 standard 哈希差异及复测条件见下文，不能省略。本页不将功能请求耗时用作性能提升结论。

## 完整镜像产物

| 项目 | standard | ultra |
| --- | --- | --- |
| 本地标签 | `local/vllm-sm75:v0.1.5-full-rc1` | `local/vllm-sm75:v0.1.5-ultra-full-rc1` |
| 镜像 ID | `sha256:3dafd53886552108ddb3e446b299ad748d841f2d37c3b4cb82c7167f8ea5af58` | `sha256:3bdad12bfe5b855ddb637cbb2fcf754ecaa7ad54eb2f9f4638b297bc2aabbcb9` |
| 完整 OCI SHA256 | `11414e8de07463c7b2a40aa4db6d32c59c1c28dcb2a6ebb4cfe176d2b5921f2f` | `5991e73a04667904db691b5c1af75b2021dc6a710bbcb69af5ab603d89aadc59` |

完整 OCI 已从 worker volume 的 `sm75-output/` 复制到 Docker 存储之外的 `/path/to/sm75-artifacts/full-build-20260918/portable-oci/`，文件名分别为 `v0.1.5-full-rc1.oci.tar`、`v0.1.5-ultra-full-rc1.oci.tar`。目录同时包含 `SHA256SUMS` 和 `SOURCE.json`。复制容器限制 256 MiB / 0.5 CPU，校验容器限制 128 MiB / 1 CPU；两份复制后的 SHA256 均与原完整 OCI 一致，校验后才将临时目录改为正式目录。后述 `.local.tar` 仅为宿主导入包。

- FlashQLA 从源码重新编译，约 160 秒；架构检查只检出 `sm_75`。
- standard 引擎测试 48/48 通过，ultra 控制台 Linux 测试 52/52 通过。
- 版本契约：vLLM `0.29.0+cu129`、Torch `2.13.0+cu129`、CUDA 12.9、Transformers 5.15.1、FlashInfer python/cubin 0.6.18；`flashinfer-jit-cache` 不安装。
- standard / ultra 各 43 个受审补丁文件完全一致，所需安装钩子存在；ultra 79 个控制台源码/资源文件完全一致。DSH 品牌、实时顶栏和聊天状态修复检查通过。
- 扩大对照显示：审计脚本覆盖的 8,298 个 vLLM/FlashInfer 安装文件与旧引擎取证样本零差异。重新编译的 FlashQLA `.so` 也与 RC5 原二进制逐字节相同，SHA256 为 `35b217137a7c4fc341e22574a45fbc71b991e0a2b5698d87a2b38deb3dc9baec`。范围和基座 ID 见 [文件及二进制对照](2026-09-18-full-source-comparison.json)，不代表所有依赖或整个容器都已做二进制重现证明。
- 独立无 GPU 容器完成 DSH UID/GID 1000、零有效 capabilities、管理文件隔离、真实 HTTP 启停和配置符号链接拒绝检查。
- 07:17:33–07:32:49 的 183 次资源采样中，宿主最低可用内存约 24.04 GiB，Docker daemon 最高观测 RSS 约 277.02 MiB。worker 的 OOMKilled 为 false；daemon PID 始终为 26579。
- 恢复脚本第一次误用了 `/api/harness` 路径，导致构建总包装脚本在恢复检查阶段返回 1；模型已正常就绪，DSH 未自动启动。改用 `/console-api/harness` 并携带匹配的 Origin 后，已实测恢复控制台、DSH 和引擎。这是验收恢复脚本修正，没有修改镜像运行时代码。

## 构建隔离

- BuildKit v0.29.0 worker 固定 amd64 digest：`moby/buildkit@sha256:e5d9d1763945ca186d48cbf5b0ff3e062eb917fb8a80da6da87144edb7320a90`。
- 独立 docker-container worker，编译/导出阶段 8 GiB 内存硬限制、2 CPU、单任务并行；基础层下载阶段降低为 2 GiB。宿主无 swap。
- 观察到 RUN 进程位于 worker 的 buildkit 子 cgroup，父 cgroup 的 `memory.max=8589934592`、CPU 配额为 2 核。子 cgroup 的 `max` 不代表绕过父限制。
- `buildctl` 客户端和 OCI 导出都在 worker 内运行；不自动 load 到宿主 Docker，不挂载生产数据或 GPU。
- 编译阶段先停止相关模型容器；资源守卫只停止 worker，退出恢复模型容器。宿主 Docker daemon 不停止、不重启；隔离保存的旧 BuildKit 数据库不恢复。
- 归档由 `git archive` 生成，不包含未跟踪文件；根 `.dockerignore` 另行排除私有 `evidence/`。

## 宿主导入验证

官方基座 amd64 的 37 个 diffID 与宿主现有 standard 镜像的前 37 层逐项一致。
工具 `tools/oci-local-import.py` 只允许省略完全匹配的父链，保留完整 config 和新增层；输出的是依赖宿主已有层的本地导入包，不能当作可分发镜像。完整 OCI 另行保留。

依据宿主 Docker 29.4.3 的提交 `56be731`，loader 先查询完整 ChainID，已存在的链复用，缺失的层才打开归档内容并注册。参见 [对应 loader 源码](https://github.com/moby/moby/blob/56be731/daemon/internal/image/tarexport/load.go)。工具不修改 Docker layerdb。

小镜像试验已完成：完整 OCI 约 104 MiB，本地导入包 10 KiB；复用 7 层，包含 1 个 32 字节压缩空层。导入后 ID 与源 config digest 完全相同：`sha256:64d240fff4417cf203732d3c7281d974b5ab4c8144d14978e619c0832c5b9abf`，无网络、64 MiB 限制下启动 `/bin/true` 成功。

随后两个完整镜像也已实际导入并核验 ID 等于 OCI config digest。standard 本地包为 99,880,960 字节，复用 37 个父层、包含 52 个新层；ultra 本地包为 112,916,480 字节，复用已导入 standard 的 89 个父层、包含 17 个新层。完整 OCI 分别为 10,629,060,608 / 10,741,907,968 字节，不能用本地包大小代表完整镜像大小。

## 长上下文契约

`tools/validate-long-context.py` 使用精确 token-ID 输入，固定 greedy/seed、同一 thinking 模板、前中后三位置合成记录检索、自然停止、真实输出 token-ID SHA256，以及真实 CPU KV 写出/读回计数。请求具有截止时间和宿主剩余内存保护；每个结果即时写入 JSONL，不覆盖已有结果。

自然停止检索契约长度为 8,192 / 32,768 / 65,536 / 131,072 / 262,016，每档两次；预留 128 输出 token，输入加输出预算不超过 262,144。这是合成资料的功能与边界验收，不替代多样化长文质量或吞吐基准；512-token 持续 decode 使用下述独立契约。

现有 RC5 上全部五档已各两次通过。同档输出 token 哈希一致；64K 第二次观测 CPU→GPU KV 读回 375,521,280 字节，输出一致。262,016 首次/重复请求分别读回 860,094,464 / 430,047,232 字节。首次边界请求发生 2 次调度抢占，重复请求为 0；因此只能判定本契约功能通过，不能判定边界性能已经达标。测试期间宿主最低可用内存约 4.77 GiB。

生产环境 `/reset_prefix_cache` 返回 404，未启用 dev API；单独记录的显式 reset 试验失败表示接口不可用，不是 offload 故障。自然缓存路径已经测到真实读回。

详见 [RC5 十次请求的合成验收数据](2026-09-18-rc5-long-context.json)。输出仅 51–55 token，因此另行补测 131,072 / 261,632 输入 + 512 输出的持续 decode，刻意要求用满输出预算并以 length 停止，不能混同自然停止测试。新完整镜像支持通过 `--baseline` 校验相同输入和输出 token 哈希。

持续 decode 的四次请求已全部通过：131,072 / 261,632 输入均生成完整 512 token，同档重复哈希一致；JSON 前缀的三位置记录准确。128K 两次都没有抢占；261,632 首次出现 2 次、重复为 0，并分别测到 860,094,464 / 430,047,232 字节 CPU KV 读回。详见 [持续 decode 数据](2026-09-18-rc5-long-decode.json)。这复现了边界抢占现象，未修改生产配置来隐藏该现象。

传输字节是累计计数增量，不代表 CPU KV 缓存同时驻留容量。现场 CPU KV 容量配置仍为 8 GiB。

启动日志还显示：虽然 CLI 为 block-size 32，该混合模型将 attention block 实际调整为 1,664 token，并对 mamba page 补齐 1.71%。后续 block-size、batch 和 KV 余量扫参必须记录这个有效页粒度。现阶段只能确认抢占来自 KV 块分配不足路径，尚未区分容量边界和临时占用因素。

## 新完整镜像验收

两版使用独立、无网络的 GPU canary，顺序运行；模型权重只读挂载，数据、凭据、工作目录和编译缓存与生产分开。测试期间停止原模型容器，Docker daemon 继续运行。canary 采用 26 GiB 内存硬上限，保留 TP4、FP8 KV、8 GiB CPU KV、DFlash2 7 token、262144 总上下文的原配置。

Ultra 已完成 16 次请求：五档自然停止检索各两次、262K 后再访问 64K 两次、128K / 261632 输入加 512 输出各两次。每次均核对实际输入长度、完整输出 token-ID 数量、输出 token 哈希、三位置记录，以及 RC5 的输入/输出哈希。所有请求通过。返回 64K 后首次重新计算、重复请求测到 375,521,280 字节 CPU KV 读回；这证明该顺序下的回读正确，不能推广为任意并发驱逐都已覆盖。

Ultra 的登录 Cookie、会话检查、DSH 启动及认证页面、实时顶栏 API、greedy 重复、自然停止、JSON Schema、工具调用往返均通过。这里的 Cookie 检查发生在同一新容器中，不替代 RC5 已单独完成的跨容器重建会话测试；短输入 prefix 冒烟命中为 0，正命中证据来自长上下文请求。

标准版首轮 8K 两次通过，但 128K + 512 输出从第一个 token 开始与 RC5 基线不一致（紧凑 JSON 与换行 JSON 的差异，后续正文也不同）。三位置检索、实际长度和完整 512 输出检查通过，但严格哈希门禁失败，自动停止 canary 并恢复 RC5 的控制台、DSH 和模型。原始失败记录保留，未改写为通过。

核对 `cache_key_factors.json` 发现，两版缓存因素的唯一差异是首轮 standard 漏设 `NVCC_THREADS=1`，导致计算图缓存标识分别为 `e48ed0940c` / `028319af02`。在新的 standard canary 中只补齐该变量后，复用已有编译缓存，128K 的两次 512-token 输出均恢复为与 RC5 / ultra 相同的 token 哈希。这个对照证明了修正测试环境后的结果；不能单凭缓存键差异就断定编译线程数是数值差异的直接原因，也不能宣称不同冷编译缓存之间已经证明逐 token 稳定。

标准版环境对齐后 8K 两次、128K / 261632 输入加 512 输出各两次共 6 项通过；与 Ultra 共 22 项通过。每次均匹配 RC5 的输入/输出 token 哈希，独立汇总脚本还重新核验了原始 token-ID 数组的 SHA256 和数量。五组场景分开记录，避免把缓存循环的 64K 请求与首次 64K 混为同一条基线。详见 [完整镜像验收数据（含首轮失败）](2026-09-18-full-long-context.json)。首轮另外两项 8K 成功不计入这 22 项。

两版首次上限持续生成均出现 2 次抢占，重复请求为 0；实际 CPU KV 回读分别为 860,094,464 / 430,047,232 字节，与 RC5 相同。最终通过的 22 项请求中，宿主最低可用内存约 4.35 GiB。这里只能确认现有生产栈的功能契约通过，边界抢占、不同冷编译缓存的输出稳定性、target-only / MTP、14B / 35B FP8 / AWQ / 后续 INT8 矩阵、长文质量及正式配对性能仍为后续门禁。

08:40 完成最终恢复：原 RC5 镜像 ID 未改变，模型 `/health` 返回 200，控制台、DSH 健康检查通过，DSH 进程 UID/GID 为 1000。三个 canary 和 BuildKit worker 均已停止，OOMKilled 均为 false。最终脚本返回 `acceptance exit=0 production-restored=1`；Docker daemon 仍为原 PID 26579。未修改生产标签、Unraid 模板或原凭据，未推送 GitHub / 公开镜像。
