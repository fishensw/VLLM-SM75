> v0.1.4 使用说明。本文历史实测按原日期保留；不等于 Firefly 整合版本已完成验收。新增对比见[验证记录](validation/v0.1.4.md)。

# 自动休眠与持久化缓存（v0.1.4）

目标是让长时间闲置的推理服务释放 GPU 资源并降低功耗，新请求到达后自动恢复。本稿目标镜像为 `vllm-sm75:v0.1.4`。

已完成本地 GPU 验证的配置、可复制的完整命令与实测效果见[FP8 DFlash2 推荐配置](recommended-fp8-dflash2.md)。

## 先选模式

日常希望空闲后进入低功耗、主机内存较小，或使用 DFlash2：选择 **exit**。它保留 API 服务，退出引擎及 GPU worker；下一个推理请求负责触发重建并等待结果。

| 模式 | 休眠与恢复方式 | 主机内存要求 | 磁盘要求 | 功耗与适用范围 |
| --- | --- | --- | --- | --- |
| `exit` | 退出引擎和 worker，释放其 CUDA 上下文；唤醒重新加载主模型、草稿模型和匹配编译缓存 | 不保留整模型 pinned RAM 备份；API、系统以及启动时的权重加载仍需内存 | 主模型、草稿模型、配置及 tokenizer 持续可读；编译缓存目录持久化且可写 | 本地 T10 ×4 已验证 P8；优先用于省电和 DFlash2，代价是首个请求等待完整重建 |
| `cpu` | level 1：权重复制到 pinned CPU RAM，唤醒复制回 GPU | 额外预留实际 GPU 权重备份的总量，包含草稿；另加服务、缓冲区及原 CPU KV offload 预算，不能按压缩模型文件大小直接计算 | 仍需原模型文件供启动；不创建磁盘休眠快照 | 保留引擎/CUDA 上下文，不保证 P8；适合 RAM 充足、希望减少恢复加载工作量的部署 |
| `reload` | level 2：丢弃 GPU 权重，唤醒分配显存并从 checkpoint 重载主模型 | 无整模型权重 RAM 备份，但仍有进程、非权重缓冲、CPU KV offload 和可回收 OS 文件缓存，绝非零内存 | checkpoint 必须持续可读；量化格式需支持重载路径 | 保留上下文，不保证 P8；DFlash2 草稿不随主模型的 reload RPC 一起加载，当前不要用于 DFlash2 |

`cpu` 和 `reload` 必须启用 `--enable-sleep-mode`；`exit` 不需要。当前 `exit` 验证拓扑为 `vllm serve`、单 API server、DP=1、TP4；多 API server、DP>1 和离线进程内引擎不在本次支持验证范围。

**上述 cpu/reload/exit 三种模式不把完整运行时状态写入磁盘快照。** `exit`/`reload` 读取的是已有模型文件；编译缓存保存的是可复用的编译产物。休眠后不要依赖请求前缀/KV 缓存仍然存在，对话历史应由客户端继续提交。

## 30 分钟日常使用，60 秒验收

保留原推理参数，在 `vllm serve` 命令末尾添加：

```bash
--auto-sleep-idle-timeout 30 --auto-sleep-offload-target exit
```

单位是**分钟**：60 秒测试用 `1`，30 分钟用 `30`，关闭用 `0`。计时从请求完成、引擎进入空闲后开始；有新请求时取消本轮空闲计时。

仓库启动脚本默认 `30` 分钟、`exit`；直接调用 `vllm serve` 默认关闭自动休眠，且其 target 默认 `cpu`。不要混淆两套默认值。

```bash
# 已按 README 设置 API key、模型路径、缓存目录及 VARIANT/FORMAT。
AUTO_SLEEP_IDLE_TIMEOUT=1 AUTO_SLEEP_OFFLOAD_TARGET=exit bash docker/run.sh
# 通过后停止/重建同名容器，将 1 改为 30；脚本不会自动替换现有容器。
AUTO_SLEEP_IDLE_TIMEOUT=30 AUTO_SLEEP_OFFLOAD_TARGET=exit bash docker/run.sh
# 关闭自动休眠
AUTO_SLEEP_IDLE_TIMEOUT=0 bash docker/run.sh
```

若确实需要其他模式，在原命令上使用以下对应参数；两种模式本轮仅有状态机/参数回归测试，没有真实 GPU 性能承诺：

```bash
# cpu：先确认有足够的额外 pinned RAM
--enable-sleep-mode --auto-sleep-idle-timeout 30 --auto-sleep-offload-target cpu
# reload：仅用于已验证可重载的非 DFlash2 配置；路径必须是容器内 checkpoint 路径
--enable-sleep-mode --auto-sleep-idle-timeout 30 --auto-sleep-offload-target reload \
  --auto-sleep-reload-path /models/your-model
```

新版 `docker/run.sh` 选择 `cpu`/`reload` 时自动补齐 `--enable-sleep-mode`；可用 `AUTO_SLEEP_RELOAD_PATH` 传容器路径。它只生成参数，不代表模型重载兼容性已经验证。

## 参数与资源预算

| CLI 参数 | 直接 CLI 默认 | 启动脚本变量 | 作用 |
| --- | --- | --- | --- |
| `--auto-sleep-idle-timeout` | `0` | `AUTO_SLEEP_IDLE_TIMEOUT`，默认 `30` | 空闲分钟数，可为小数；`0` 关闭 |
| `--auto-sleep-offload-target` | `cpu` | `AUTO_SLEEP_OFFLOAD_TARGET`，默认 `exit` | 选择上述三种恢复方式 |
| `--enable-sleep-mode` | 关闭 | 脚本为 cpu/reload 添加 | 启用 level 1/2 分配器支持，exit 不需要 |
| `--auto-sleep-reload-path` | 启动模型路径 | `AUTO_SLEEP_RELOAD_PATH` | reload 使用的 checkpoint；也影响文件页预热。exit 重建仍按原模型参数启动，不用它替换模型 |
| `--auto-sleep-page-cache-keep-interval` | `600` 秒 | `AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL` | reload 睡眠期间预热文件页的间隔；`0` 关闭睡眠时预热及后台定时预热；唤醒前仍有一次预热提示 |

exit 在退出前对主模型 safetensors 做一次文件页预热提示，不保留后台预热进程。它是 OS 提示，不保证文件全部驻留或唤醒必然命中；内存紧张时页会被回收，也不保证覆盖草稿模型。OS page cache 使用的是可回收主机内存，与整模型 pinned 备份不同。

本仓库 Qwen 启动脚本原有 **8 GiB CPU KV offload**，与 `cpu` 权重休眠是不同功能。切换休眠模式不会移除这项原配置；不要因为选择 exit/reload 就把它从内存预算中漏掉。31 GiB 主机不适合再为 27B 模型冒险分配整模型 pinned 备份；其他硬件应按实际峰值评估。

磁盘至少容纳完整主模型、DFlash2 的完整草稿模型及其配置/tokenizer，并为编译产物留余量。FP8 27B 本地检查点约 28.75 GiB，仅是该主模型示例，不是通用最低容量。多种参数和版本的编译缓存可累积到数十 GiB；没有按模式固定的休眠快照空间需求。首次下载/解压和镜像构建需要另外的临时空间。

恢复时重新读取权重、量化处理和初始化仍会发生，缓存只消除能复用的编译工作。客户端和反向代理的请求超时需覆盖实际完整唤醒时间；不要用通常几秒的在线请求延迟作为唤醒超时。

## 缓存持久化

模型和编译缓存目录的设置统一见 [README](../README.md#编译缓存持久化)。保留宿主机挂载，主模型和 DFlash2 草稿模型都能复用匹配缓存。

## 怎么确认有效

1. 空闲阈值先设为 `1`，启动后完成一次推理；等待该请求完成至少 60 秒，再留出退出清理和显卡降频时间。
2. 不发新推理请求，检查 `/health` 仍可访问，并采样：

   ```bash
   nvidia-smi --query-gpu=index,pstate,memory.used,power.draw --format=csv
   ```

3. 发新请求，确认自动恢复、输出成功。查看日志中主模型、草稿模型、候选选择器各 rank 的 `Directly load AOT compilation`；不能只看主模型的一条命中记录。
4. 改回 `30`，确认启动仍使用相同缓存键。仅改变休眠阈值不应重新编译；模型、TP、dtype、计算图参数、相关代码或依赖变化仍可合理触发缓存失效。

P8 取决于显卡/驱动和是否有其他 GPU 进程；exit 释放的是本引擎资源，不保证所有机器必进 P8，也不要求 `nvidia-smi` 显存显示绝对 0。保持 API 在线与停止整个容器是不同测试条件。不要混用自动模式与开发用的手动 sleep/wake API。

## 本次实测边界

2026-09-08，修复镜像，FP8 DFlash2、T10 ×4、TP4、31 GiB 主机、CPU ondemand：60 秒自动 exit 后四卡连续三次 P8、显存 3 MiB、9.97–15.31 W/卡，API health 200。完整唤醒两个并发请求均成功，耗时约 135.7 秒。

已有缓存启动、exit 唤醒、改为 30 分钟后的启动，三个阶段均 12 次 AOT 命中（3 组件 × 4 rank）、零重新编译，三组键一致。唤醒时主模型/草稿/候选选择器编译缓存加载阶段分别约 3.81/0.76/0.06 秒，不能当作完整唤醒耗时。30 分钟阶段未额外等待完整休眠周期。

这不是所有五个配置、CPU/reload 模式或所有 GPU 的验收。修复原理与同版本更新范围见[更新说明](releases/v0.1.3.zh-CN.md)。

## v0.1.4 编译目录补充

沿用 `VLLM_SM75_CACHE_ROOT`，脚本按格式保留 `fp8/vllm`、`awq/vllm`，共享 `shared/flashinfer`；新增 `<format>/triton` 和 `shared/torch_extensions`，对应 `/root/.triton/cache` 和 `/root/.cache/torch_extensions`。模型下载目录继续由 `VLLM_SM75_MODEL_CACHE_ROOT` 单独指定。上述目录必须可写且在容器外持久化，不能只写入 docker.img 的容器层。

迁移已有容器时先保留旧宿主目录及挂载；若 Triton/扩展缓存此前只在容器层，先复制到新的宿主目录，再重建挂载。脚本不会自动迁移旧缓存。KAT 自定义容器需自行保留原参数及相应挂载，`docker/run.sh` 不生成 KAT 配置。

> 本地已有的 `disk` 快照休眠开发代码已保留：保存模型分配并原位恢复，KV按休眠协议失效，CUDA上下文仍保留，不能保证P8。需要可写真实磁盘目录和足够快照空间，`--enable-sleep-mode --auto-sleep-offload-target disk --auto-sleep-disk-path /sleep-state`；脚本会挂载 `cache/sleep/<variant>-<format>`。它不属于本轮v0.1.4 AWQ GPU验收或默认推荐配置。
