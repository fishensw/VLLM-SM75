# 磁盘休眠实验：未通过 P8 节能验收

> 状态：仅为实验实现，未部署。FP8 MTP5 的磁盘保存和恢复通过，但观察到 GPU 仍为 P0、38–43 W/卡，未满足用户主要目的。已暂停此方案，优先验证 PR 原生 exit 的 P8 和透明唤醒。

## 核对基线

核对 `main` 提交 `79f1442a72`，已合并 PR #6（`558ff3a05b`），以及已关闭未合并的 PR #5（`20c7cf067f`）。原 PR 只有 cpu/reload/exit；reload 重读原始 checkpoint，exit 重建进程，都不是将运行时模型状态写入硬盘。

2026-09-08 启动故障中，内核在新建 8 GiB CPU KV offload 时 global OOM 杀死 Worker。宿主机有一个无人打开或映射的旧 `/dev/shm/vllm_offload_*.mmap` 实占 8 GiB；清理后，可用 RAM 从约 19 GiB 恢复到 27 GiB。保持参数不变的 FP8 MTP5、FP8 DFlash2 均已恢复启动并通过 HTTP 推理。原来的编译等待信息不能证明编译死锁。

## 磁盘实现

- `--auto-sleep-offload-target disk`，要求 `--enable-sleep-mode` 与 `--auto-sleep-disk-path`。
- 保存 CuMem 池中实际运行时模型字节，包括量化/重排后的主模型和草稿权重。每个 Worker 复用 16 MiB 传输缓冲；每写入约 64 MiB 同步磁盘并释放已写页缓存，避免整模型的 CPU 副本。
- 每个 Worker 完成 payload、SHA-256 元数据和 fsync 后才释放 GPU 分配。恢复原虚拟地址并校验每个分配，保留原进程的 CUDA Graph 和 communicator。
- 正常 sleep 协议会清除 prefix/KV 缓存内容，KV 容量、上下文长度、并发、CPU offload 预算不变。CPU KV offload 和 CUDA context 等原进程资源仍在；这不是跨进程、重启容器后恢复的进程检查点。
- 写盘失败回滚所有 TP Worker 并恢复调度；恢复校验失败保持暂停，不使用未验证的权重进行推理。拒绝 tmpfs/ramfs 作为快照目录。
- 修复原控制器休眠失败后调度仍暂停、唤醒失败却报告 ACTIVE、手动唤醒后不重新计时、定时器早触发后漏掉休眠的问题。

## 配置

直接启动追加：

```sh
--enable-sleep-mode \
--auto-sleep-idle-timeout 1 \
--auto-sleep-offload-target disk \
--auto-sleep-disk-path /sleep-state
```

`--auto-sleep-idle-timeout` 的单位是分钟，`1` 即 60 秒。必须将 `/sleep-state` 绑定到有足够空间的实际硬盘目录。验收通过后仅将 `1` 改成 `30`。

`docker/run.sh` 也支持 `AUTO_SLEEP_OFFLOAD_TARGET=disk AUTO_SLEEP_IDLE_TIMEOUT=1`，自动挂载 `$VLLM_SM75_CACHE_ROOT/sleep/<variant>-<format>`。

## 验收状态

- GPU 小规模实测：两次落盘/恢复、主/草稿字节一致、原地址不变、CUDA Graph 重放通过。
- 故意损坏快照：校验拒绝恢复，保留快照，修复文件后重试通过；内存盘拒绝测试通过。
- 完整模型的 60 秒自动休眠、恢复后确定性与并发请求正在验证；不能把上述测试等同于五个模型均已验收。

### FP8 MTP5 复测中间结果

镜像 `3417495177d256239264b2288ba3d9226bc704dcb94047e9985131eeb30ae50c`，与工作区四个运行时修改文件 SHA-256 一致。停止构建并恢复宿主内存后，四个 Worker 首次写盘全部提交，总量约 32 GiB；写入过程 MemAvailable 保持约 5.4 GiB，无 OOM。每卡释放 11.74 GiB，剩约 1.01 GiB CUDA/进程相关分配。首次自动恢复后的推理 HTTP 200，`17 + 26` 返回 `43`，恢复和响应共 105.134 秒。第二次休眠及并发唤醒仍在运行，暂不计为完整验收通过。

用户已明确排除 `cpu` 模式。`exit`/`reload` 不满足运行时状态落盘目标，不再继续其他模式的完整测试。
