# 2026-09-18 完整构建事故记录

> 脱敏说明：部署地址、容器名及宿主路径已替换为示例占位；验证结论与哈希保持原值。

## 已确认事实

1. 在生产宿主停止 FP8/DFlash2 模型、确认 GPU 内存释放后，增量控制台 RC1 构建成功。镜像层导出约 10 秒，Linux 41 项测试通过，原模型随后恢复，容器内 `/health` 返回 200。
2. 随后错误地在同一生产宿主发起标准版完整源码重建，计划再构建 ultra RC2。该操作没有遵循既有 fast 迭代边界。
3. 完整构建中的 FlashQLA 编译、`cuobjdump` 架构检查、运行时验证及最后依赖下载均已完成。最后持久日志为 `#64 exporting to image` / `#64 exporting layers`，没有成功导出镜像 ID。
4. 随后 SSH 横幅超时，Unraid Web 管理页无法加载；当时控制台 HTTP 仍可返回 200。用户手动重启宿主。
5. 重启后短暂读取到：主机 31 GiB RAM、无 swap；Docker 存储为 loop2 上的 Btrfs；Docker 文件系统 512 GiB、约 47 GiB 已用、约 460 GiB 可用；缓存盘约 472 GiB 可用。Btrfs 设备读写/flush/corruption/generation 错误计数均为 0。
6. 没有取得事故前持久内核日志。重启后的 syslog 不能用来证明事故前没有 OOM。启动日志中的 PCI BAR `no space` 不等于磁盘空间耗尽。
7. 重启后 SSH 再次出现横幅超时。重启之后仅执行只读排查，没有发起新构建；生产容器恢复状态尚未确认。
8. 后续 SSH 恢复后取得明确日志：`01:08:37` 发生 `global_oom`，内核杀死 `dockerd`（PID 7916），`anon-rss:29458568kB`，约 **28.1 GiB**。被回收后主机回到约 3.1 GiB 已用 / 28 GiB 可用，Docker socket 无可用 daemon。`nvidia-smi invoked oom-killer` 是触发内存分配失败的进程，不能据此将主要内存占用归给 nvidia-smi。
9. 当前 Docker 客户端版本 29.4.3，Buildx v0.33.0。现存 BuildKit 数据库分别约 4/8/16/16/1 MiB；尚未修改、删除或重建这些数据库。

## 原因判断与边界

**已定位触发环节：生产宿主上的完整镜像层导出。** 原 fast Dockerfile 的注释明确说明用于避开反复 Docker 构建引起的 snapshotter 抖动；原迭代流程为同步源码、容器内 `fast_compile.sh`、重启容器。

本次执行错误是将这一约束缩减为“停模型即可全量构建”，在增量 RC1 已成功后仍启动完整重建。停止推理不能约束 Docker daemon / BuildKit 的镜像导出、快照和宿主文件系统开销。

**重启后再次失联的直接原因已确认：dockerd 匿名内存膨胀并触发宿主全局 OOM。** Docker 被杀后服务停止，内存压力缓解，SSH 恢复。首次故障发生于完整层导出，与相同机制高度吻合，但缺少首次的内核日志，不能将推断写为已证实。

具体是哪一段快照、导出或持久构建状态处理导致 dockerd 分配约 28 GiB，尚未取得调用栈/heap profile，不能把它直接归为已确认的 Btrfs 缺陷，也不能声称只是 32 GiB 内存“不够用”。没有磁盘满证据。Docker 官方 Btrfs 文档只作为风险背景：<https://docs.docker.com/engine/storage/drivers/btrfs-driver/>。

## 已落实的修正

- `docker/build.sh` 检测到 Unraid 直接拒绝完整构建。
- `docker/BUILD.md` 与 `ultra/README.md` 明确完整构建只能在独立构建机执行；不能绕过脚本在生产宿主直接调用完整 Dockerfile。
- 生产迭代沿用已经存在的 fast 基座与挂载源码/容器内覆盖；当前不再构建新的 fast 基座，因为那仍是完整构建。
- 没有覆盖旧生产镜像标签，没有更改宿主驱动、电源脚本、swap 或 Docker 存储配置。完整源码构建和 RC2 验收记为未完成，不能发布为成功。

## 恢复核对

首先确认 Unraid 存储与 Docker 服务稳定、没有残留构建和高 I/O；然后恢复原生产容器及原 `fp8-dflash2` 配置，核验容器内模型 `/health` 和 DSH。不能在构建状态不明时再启动模型叠加负载。

最新状态：Docker 已被 OOM 杀死，生产尚未恢复。不得直接循环重启 daemon 或删除 Docker 数据；应先备份失败构建的状态与日志，再在有内存监控/中止条件的维护窗口排查启动内存增长。

旧完整构建脚本的 EXIT 恢复钩子只对正常脚本退出有效，宿主强制重启不能保证执行。因此不能将“已配置恢复钩子”写成“生产已恢复”。后续完成恢复后须在本记录补充实测状态。

## 01:21 恢复进展与用户约束澄清

用户明确允许完整构建，也允许临时停止相关运行容器；要求不能拖垮 Docker daemon 或破坏原有功能，不能影响无关容器。之前“生产机永久禁止完整构建”的表述过度扩大了限制。现有脚本拦截暂用于阻止已出事故的未隔离构建路径，不代表禁止经过资源隔离和恢复验证的完整构建。

恢复动作：确认 dockerd 不存在、无活跃构建后，将 BuildKit 状态打包到 `/path/to/sm75-artifacts/recovery-20260918-012137/buildkit.before.tar`，SHA256 为 `affe9481ff1c20907cb6c554c5acd427174d6d4749dd26d5f45e8419fbb85dfe`；原目录重命名为 `/var/lib/docker/buildkit.quarantine-20260918-012137`，未删除内容。镜像、容器、卷、存储配置未变更。

使用 Unraid 原生 `rc.docker start` 启动，临时环境 `GOMEMLIMIT=2GiB` 作为 Go GC 软目标（不是硬内存限额，也未写入永久配置）。启动监护设置超阈值保全条件，未触发暂停或终止；daemon RSS 约 92–99 MiB，Docker 29.4.3 API 恢复。10 个原容器均可枚举。

这一结果支持失败构建状态与启动异常有关，但由于同时设置了 Go 内存软目标，不能把具体根因认定为已证明。保留隔离目录用于后续离线分析；不得直接回填到运行中的 BuildKit 或执行全局 prune。

已启动原生产 ultra 容器，核验 image ID 仍为 `sha256:64898eeada4d3e2ebd0becf3d6a0ff489388fe49b958ef041b3e2d4fb415d535`，原挂载及 `autoStart=true/defaultProfile=fp8-dflash2` 保留。模型就绪结果另行记录，不能用容器 Up 代替模型健康。

## 01:26 恢复验收

- 原 ultra 容器保持原 image ID 和挂载运行，`/health` HTTP 200。
- 实际调用原模型 `/v1/chat/completions`：HTTP 200，回复 `OK`，finish_reason=`stop`，15 输入 / 2 输出 tokens。
- 经原控制台登录代理访问 `/dsh/` HTTP 200。
- Docker daemon 连续运行 5 分钟，RSS 约 54 MiB，宿主可用内存约 5.2 GiB（模型已加载）；未发现新的 OOM。短期恢复通过，不能据此保证下一次完整构建安全。
- 未恢复历史上已停止的其他模型容器，避免争用同一组 GPU。原标准版及 AWQ/MTP 容器均保留。
- 独立 UI 候选经 1 GiB 内存 / 1 CPU 限制验收后停止，释放资源；生产容器代码没有更新到 UI 候选版本。
