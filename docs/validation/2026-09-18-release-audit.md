# 标准版 / ultra 发布对账与 DSH 权限修复

> 脱敏说明：部署地址、容器名及宿主路径已替换为示例占位；验证结论与哈希保持原值。

## 可追溯范围

对现有标准镜像 `f5683926f8db…`、ultra RC4 `7f5dd0f41785…` 分别启动无 GPU、无网络、512 MiB 内存限制的一次性容器，只读取安装后的源码字节，不导入 torch 或启动模型。

每版 43 个文件全部匹配仓库：28 个主覆盖文件、7 个 FlashQLA 源码/许可证文件、8 个投机覆盖文件。ultra 的 `monitor.py` 按其第三层覆盖文件核对；env 注入和 disk backend 注册标记均存在。清单见 `docs/releases/patch-inventory-{standard,ultra}.csv`。工具 `tools/audit-runtime.py` 按安装顺序计算预期值，重复目标以最后一次覆盖为准。

这证明列明补丁来源一致，不代表所有上游包、CUDA 二进制或完整镜像已可复现。公开底座、依赖版本和全量构建仍需各自验收，不能据此把 `existingImageSourceFullyVerified` 改成 true。

目标及草稿的 config/tokenizer/index 哈希见 `docs/releases/model-config-audit.json`。目标 config 为 `qwen3_5` / `Qwen3_5ForConditionalGeneration`，文本 64 层中 48 层 linear_attention、16 层 full_attention，FP8 e4m3、128×128 权重块；草稿为 `DFlash2DraftModel`、5 层、hidden_size 5120。目录名、served-model-name 与 config 类型分别记录，不互相代替。尚无确定权重 revision，也未读取全部权重计算哈希。

## 完整构建覆盖顺序

原 `ultra/Dockerfile` 先运行 `install-native-plugins.mjs`，之后 COPY 界面覆盖层，导致 layout/chat 上的部分品牌改动被覆盖。已改为覆盖层、字号适配完成后，再运行插件与品牌安装器。

在 RC4 的一次性无 GPU 容器内重放全部界面覆盖层，验证工作台标题、statusbar slot、首步用量提示；重复运行安装器后所有 DSH JS 文件哈希不变。`tools/verify-ultra-overlays.sh` 可重放此检查。这是覆盖顺序和幂等验证，不是完整源码构建结果。

## DSH 权限边界

现场 RC4 的 DSH 实际 UID/GID 为 0，CapEff 为 0。旧逻辑在 spawn UID 1000 失败时退回 root；即使没有 Linux capabilities，同 UID 工具仍可读取管理进程拥有的 0600 文件。

修复：

- 固定 UID/GID 1000，失败明确报错，禁止 root fallback。
- 不继承容器全部环境；保留工具路径、语言/时区、CA、DSH 参数和专用引擎 API key。
- 配置采用临时文件、文件描述符设置所有者、原子 rename；读取 settings 拒绝符号链接，避免低权限用户通过链接让管理进程覆盖私有文件。
- 管理进程仅添加 CHOWN/FOWNER/DAC_OVERRIDE/SETUID/SETGID/KILL，保留 cap-drop ALL 和 no-new-privileges；降权后的 DSH 有效能力为零。

隔离 Linux 验证实际启动 DSH，完成 token/cookie 握手后 HTTP 200；工具 UID 无法读取管理专用文件；环境中没有测试管理秘密；启动、停止、再次启动以及配置符号链接拒绝均通过。脚本为 `tools/verify-harness-identity.mjs`，不使用生产数据或 GPU。

RC5 源码 `8b74252`，增量镜像 `sha256:f1f0a00b24d44d5c5505c4e78ac4a0025de86bd014957955d19a0273d64ff420`。镜像内 52 项测试全部通过；Windows 50 通过、2 项 Linux 测试跳过。最终镜像另外通过上述真实 DSH 权限检查。

旧 DSH 数据全部归 root 所有。部署使用停止容器后复制的 `dsh-home-rc5`、`workspace-rc5` 两个目录，只调整副本为 1000:1000；旧目录原样保留。管理数据、模型和缓存保持原挂载。回退必须使用旧容器及旧 DSH 目录，不能将新镜像与旧 cap-drop ALL 配置随意组合。

## RC5 现场部署结果

约 02:16 完成 RC5 验证与名称恢复。运行容器 ID 为 `97afee552cd587bf11885a3e0097fed2ce448ee143a9f7c9415e948880c94902`，仍使用原 ultra 名称和 `192.0.2.53:1615` 地址。

- 原 Cookie 跨容器切换仍有效，DSH 认证页面和实时顶栏接口通过。
- 现场 DSH UID/GID 为 1000、CapEff 为 0。UID 1000 无法读取 `/data/key`、`/data/api-access.json`，可以读写两个 DSH 目录。
- 环境变量、GPU 参数与 RC4 完全相同；仍为 7 条挂载，仅两个 DSH 宿主路径切换为副本。Web 与模型凭据逐字节保留，且两者彼此不同。
- `/health` 200；greedy 重复、自然停止、JSON Schema、合成工具结果往返通过。模型尚未监听时的首次 smoke 连接拒绝属于预热阶段；就绪后完整 smoke 通过。
- 短前缀重复输出一致，cached tokens 仍为 0，不算缓存命中验收。
- 顶栏 `fresh=true`，4 卡采样完整；验收时最高温度 45°C，总功耗约 43.8 W，空闲 prefill/decode/KV 为 0，无最近缓存查询时命中率为 null。
- 与 RC4 重建验收记录相比，两个 FlashInfer `.so` 的时间戳及大小完全一致；日志显示加载已有 AOT 编译产物。结论仅覆盖此次配置和形状。
- Unraid 模板 Repository、6 个管理 capabilities 和两个 DSH 挂载已同步。落盘 XML SHA256 为 `dac287dfc811b7c186894ac017ac3c8beb599c8223ff00e988b143b3ec9228f1`。
- 构建使用 RC4 固定基座的小增量路径；Docker daemon PID 一直为 26579，构建后 RSS 约 91 MiB，部署后约 113 MiB。没有停止 daemon 或无关容器。

原 RC4 保留为停止的 `vllm-sm75-ultra-rollback-rc4-20260918`。恢复目录 `recovery-20260918-012137/deploy-rc5` 为 0700；切换前 `console-data.before.tar` 为 0600，SHA256 为 `a532af9a525757ce2c4d3a4382a975bf6f650438cf887ed98007d0e82fd311de`。原 DSH 目录未改所有者，新副本保留所有旧会话文件。

### 前缀缓存补充实测

保持 `fp8-dflash2` 的 `--enable-prefix-caching`、`--mamba-cache-mode align`、batch 8192、block 32 不变，发送两次相同的较长输入。实际 API usage 的 prompt tokens 均为 9,630：首次 cached tokens 为 0，第二次为 8,320，两次输出均为 `OK` 并自然停止。结果见 [原始结果](2026-09-18-prefix-cache.json)。`/tokenize` 请求未传相同 thinking 模板参数，计数为 9,670；验收采用真实生成请求的 9,630，不混用两个数。

顶栏随后采到 `fresh=true`、近期 Cache 命中率约 86.40%、窗口 55 秒，完成真实缓存命中到控制台展示的检查。混合模型 align 模式下，之前较短输入未命中与本次较长输入命中并不矛盾；本次未单独证明所有缓存粒度边界。

两次墙钟耗时约 7.87 / 1.19 秒，仅为这次功能 smoke 的观测。没有进行规范预热、五次配对或 token hash 契约，不能据此宣布稳定性能提升。缓存驱逐/交替前缀、128K/256K 和 CPU KV 实际写出读回仍需隔离模型测试。

完整构建、其他模型矩阵、长上下文 CPU KV 往返与真实性能对照仍待完成；本次没有推送 GitHub 或发布公共镜像。
