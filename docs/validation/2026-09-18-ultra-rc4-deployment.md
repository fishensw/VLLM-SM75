# ultra RC4 现场部署

> 脱敏说明：部署地址、容器名及宿主路径已替换为示例占位；验证结论与哈希保持原值。

日期：2026-09-18。此记录对应增量候选，不等于标准版 / ultra 完整源码发布验收。

## 镜像与来源

- 源码：`f76287a`，分支 `codex/release-v0.1.5-consolidation`。
- 镜像：`local/vllm-sm75:v0.1.5-ultra-ui-rc4`。
- Image ID：`sha256:7f5dd0f41785f4199f82aa594009c72b35e69aa0949f51fc18320f45ed190ade`。
- 打包清单 SHA256：`14f512499d74c27c0dc10c13d9500a75d6b2d8598906411fa0d9cd250751b4d8`。
- 原推理底座：`sha256:64898eeada4d3e2ebd0becf3d6a0ff489388fe49b958ef041b3e2d4fb415d535`。
- RC3 中间候选：`sha256:de5b4d5b01c8404a3f31dea4b0cbbfa89f8a8b810df4558ebbc7d2bd31898d54`。

本次使用小型增量 Dockerfile，保留原 CUDA / vLLM / FlashQLA 运行时；镜像内更新 console、原生插件、品牌资产和 DSH 布局。使用 legacy builder 避开事故中的 BuildKit 状态，RUN 限制 1 GiB / 1 CPU、禁网络。该限额不覆盖宿主 daemon，不能据此宣称完整构建已解决资源风险。构建期间另行观察 daemon RSS 和宿主可用内存。

## 备份与回退边界

服务器保留原容器、原 image ID 和旧标签。原始容器 inspect、环境文件及一致性数据快照位于权限 0700 的目录：

`/path/to/sm75-artifacts/recovery-20260918-012137/deploy-rc3/`

数据快照 `runtime-data.before.tar` 包括 console-data、dsh-home、workspace；独立 cache 挂载未整体打包。快照可能含凭据与私有会话，不上传 GitHub。

新容器的环境变量、7 条 bind mount、GPU 0–3、cap-drop 和 no-new-privileges 与原容器核对一致。IP 仍为 `192.0.2.53`。需要回退时先停止新容器，再启动旧容器，避免相同 IP / GPU 并发；不得同时启动两个版本。若回退数据库快照，必须先保留升级后新产生的数据。

## 现场修复

- 工作台状态条补齐原生 slot 的 children 声明，避免根布局渲染失败。
- ultra 原先遗漏 FlashInfer 持久缓存，重建后发生 JIT 编译。RC4 与 native 后端复用 workspace 生成逻辑，映射到 `/data/cache/shared/flashinfer-home/.cache/flashinfer`，实际目标 `/data/cache/shared/flashinfer`。
- RC3 已编译的约 12 MiB FlashInfer 文件先独立复制备份，再以“不覆盖已有文件”的方式合入持久目录。
- SIGTERM 先关闭认证长连接并停止接收连接，完成子进程和遥测收尾后退出；不撤销持久登录会话。
- 已被信号终止的子进程不再被重复发送信号或等待不存在的 exit 事件。

## 验证

- RC4 镜像内部 Linux：50 项全部通过，包括打开会话 SSE 时关闭进程、重启后保留 cookie、FlashInfer 文件跨实例保留和已退出子进程处理。
- Windows：48 通过，2 项 Linux 环境测试跳过；标准版构建契约 9 项通过。
- RC3 真实模型：greedy 重复输出一致、自然停止、JSON Schema、工具调用往返通过。工具结果使用合成状态数据，没有执行外部工具。
- RC3 长前缀重复输出一致，但返回 cached tokens 为 0，Prometheus prefix hits 为 0；不能记作缓存命中通过，需另做达到混合模型缓存粒度的长前缀验证。
- RC4：登录 200、会话有效、DSH 200、未登录指标接口 401；真实四卡温度与功耗可读。模型加载时引擎指标显示空值，旧遥测不冒充在线数据。

## 最终验收与运行状态

01:55 已完成部署，运行容器恢复原名 `vllm-sm75-ultra`，地址仍为 `http://192.0.2.53:1615/`，镜像为本记录 RC4。Unraid XML Repository 和图标同步更新，原模板单独备份。

- RC4 首次持久目录迁移触发 Ninja 更新旧产物，未将这一轮宣称为完全免编译。
- 随后从同一 RC4 镜像创建另一全新容器做实际重建验收，FlashInfer 两个动态库的时间戳和大小前后完全一致；vLLM 日志显示使用已有 AOT 产物。此结论仅覆盖本次形状与软件组合。
- 原 rc4 容器约 4 秒完成退出。上一容器颁发的 Cookie 在新容器中继续有效，登录会话重建验收通过。
- 重建后 `/health` 200；greedy 重复、自然停止、JSON Schema、工具调用往返再次通过。短前缀命中仍为 0，保留待测状态。
- 新版顶栏接口 `fresh=true`，读到四卡真实温度/功耗，吞吐和 KV 为当前引擎采样值；空闲状态为 0，无缓存查询窗口显示 null。
- DSH 200、会话有效；未登录指标请求 401。
- Docker daemon 约 90 MiB RSS，宿主仍有约 5.5 GiB 可用内存。未停止或杀死 Docker 服务，也未修改无关模型容器。

原生产容器保留为 `vllm-sm75-ultra-rollback-20260918`（停止态），对应原 image ID `64898eeada4d…`。回退时先停当前容器，保留其名称/数据副本，再将该回退容器恢复原名并启动。模板回退必须指向原镜像对应的 `local/vllm-sm75:release-base-64898eeada4d`，不能误用原先已与运行 ID 不一致的 `v0.1.5-ultra` 标签。

本次未执行 GitHub push 或正式版本发布。完整模型矩阵、长上下文 offload 往返、性能配对和公开镜像发布仍单独验收。
