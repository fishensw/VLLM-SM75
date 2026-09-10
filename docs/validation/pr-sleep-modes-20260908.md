# PR #6 原版模式实测（2026-09-08）

基线镜像 `vllm-sm75:v0.1.3`，摘要 `e5e03506b4ee7335a42c1897d0a75a5134118145e1854067c03e82ddbf48d011`。测试容器从正式 Docker 配置克隆，保留原始 Cmd、Env、挂载、设备和网络；校验原始参数一致，互斥运行。用户在三个配置通过后要求停止继续测试，将五个正式配置统一添加 30 分钟 exit，并启动 FP8 DFlash2 日常使用；已执行配置更新。

`--auto-sleep-idle-timeout 1` 单位是分钟，即 60 秒。FP8 MTP5、FP8 DFlash2、AWQ MTP5 完成验收；用户明确授权 AWQ DFlash2、KAT 不再完成测试，直接配置 30 分钟。

| 模式 | 实际行为 | 当前验证状态 |
|---|---|---|
| exit | 退出引擎与 Worker，保留 API；请求到来重新加载模型 | FP8 MTP5、FP8 DFlash2、AWQ MTP5 已通过 P8、低功耗及并发唤醒；AWQ DFlash2 测试按用户要求中止，KAT 未实测 |
| reload | level 2 丢弃权重，唤醒重读 checkpoint | 按用户要求不继续；代码发现 reload_weights 只操作主模型，草稿权重恢复需核实 |
| cpu | level 1 将权重保存在主机 RAM | 用户明确排除，不启用、不实测 |
| disk（新增，非原 PR） | 保存实际运行时权重到文件，原地址恢复 | FP8 MTP5 两轮保存/恢复通过，但仍 P0、38–43 W/卡，未通过主要 P8 节能验收；停止该方向部署 |

验收包括启动后推理、60 秒空闲自动休眠、GPU 资源释放、请求触发唤醒、确定性结果以及并发请求。仅 HTTP 健康检查通过不代表模式可用。

宿主机内存 31 GiB、无 swap。已清理无进程引用的 8 GiB 旧 CPU KV offload 共享文件，原配置 FP8 MTP5 和 FP8 DFlash2 已分别恢复启动并通过推理。磁盘测试时并行镜像构建导致 dockerd 占用约 8 GiB；现已在所有容器停止时重启 Docker，恢复可用 RAM 约 28 GiB。后续不在模型测试期间构建镜像。

## 主要验收条件纠正

用户确认主要目的为 GPU 进入 P8 并降低功耗。此前新增 disk 模式的 FP8 MTP5 两轮写盘/恢复虽通过，但只观察到 P0、每卡约 38–43 W，不能记为整体通过，也不得据此部署 30 分钟配置。暂停实验性 disk 模式，回到 PR #6 原生 exit 模式进行原参数测试。CPU 复制模式仍明确排除。

exit 验收需要四张卡连续三次采样 P8、每卡显存低于 50 MiB、功耗低于 20 W，并继续验证 API 健康和并发请求透明重建。exit 重读原始 checkpoint，不将其描述为运行时模型状态快照恢复。

### 原版 exit 首次 P8 复现

2026-09-08 11:20:59 空闲退出；11:21:02 清理 8 GiB KV mmap。随后四卡均为 P8、3 MiB，功耗分别 10.71/10.73/11.73/10.70 W。11:21:15 首个请求触发透明重建，推理成功；后续第二轮并发唤醒及连续请求也通过。测试镜像 `e5e03506...` 的 auto_sleep、async_llm、core、core_client 与仓库 HEAD `79f1442a72` SHA-256 全部一致。

### 已完成的原参数 exit 验收

| 配置 | 结果 | 并发唤醒请求总耗时 | 睡眠功耗与显存 |
|---|---|---|---|
| FP8 MTP5，原 powersave | 11:34:08 通过 | 335.076 / 337.443 秒 | 四卡 P8，每卡 3 MiB，稳定样本约 10–12 W |
| FP8 DFlash2，原 powersave | 11:58:19 通过 | 342.442 / 342.444 秒 | 两轮四卡 P8，每卡 3 MiB，约 10–16 W |
| FP8 MTP5，临时 schedutil 对照 | 12:06:43 通过，随后恢复 powersave | 169.493 / 169.421 秒 | 四卡 P8，每卡 3 MiB，稳定样本 9.87–12.32 W |
| AWQ MTP5，唤醒时 ondemand | 12:20:40 通过 | 132.599 / 132.598 秒 | 两轮四卡 P8，每卡 3 MiB，稳定样本 9.98–13.30 W |

每轮功耗验收为连续三次、约 10 秒间隔采样；不等同于旧报告 12 次采样覆盖 56 秒。CPU 对照明细见 wake-latency-20260908.md。

### 正式配置交付

五个正式容器及对应 Unraid XML 模板追加 `--auto-sleep-idle-timeout 30 --auto-sleep-offload-target exit`。12:26:16 最终配置审计通过：五份原 Cmd 前缀、Env、HostConfig 和 XML 除追加部分均保持一致；Docker 的 Dns null/空数组、OomKillDisable null/false 按等价默认值核对。原容器保留为 sm75-before-exit-20260908-<variant>，XML 备份位于宿主机 disk-sleep-20260908/production-backup。

按用户指定启动正式 FP8 DFlash2，其他四个正式配置未启动。CPU 保留用户自行设置的 ondemand。AWQ DFlash2/KAT 不标记为实测通过。没有部署实验性 disk 快照代码，exit 通过释放 CUDA 进程实现 P8，唤醒重读磁盘 checkpoint。

12:32:04 正式 FP8 DFlash2 可用性检查通过：health 200、chat 200，17+26 返回 43，请求耗时 1.833 秒。运行日志确认 idle_timeout=1800.0s、target=exit。此次正式启动至健康检查就绪 392.42 秒，包含新增编译约 167.40 秒，不作为休眠唤醒耗时。容器保持运行供用户日常使用，停止后续测试。
