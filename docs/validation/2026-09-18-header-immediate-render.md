# 页头立即显示与遥测独立刷新

> 脱敏说明：部署地址、容器名及宿主路径已替换为示例占位；验证结论与哈希保持原值。

2026-09-18，源码 b610a6b。之前 mountLiveSummary 在 5 秒采样循环中同时设置 host.hidden；首次挂载时认证尚未完成，会将页头隐藏到下一轮采样。现在组件同步生成标签及 — 占位值，由已认证页面控制整个 header 的可见性；采样仅更新数值。认证状态改变时立即重新采样，不再等待定时周期。请求取消使用独立 AbortController 与代次校验，过期请求不能覆盖新值或另建重复定时器。

## 部署与验证

- 对运行中的 RC7 容器热更新 public/app.js 和 public/live-summary.js；更新前原文件保存在服务器 /path/to/sm75-artifacts/ui-rc8-20260918/backup/，可复制回对应 public 路径回滚。
- 容器 StartedAt、PID 更新前后完全一致，模型 /health 返回 200；没有停止容器、模型或 Docker daemon。HTTP 实际返回的两个文件与本地发布文件逐字节一致。
- Windows 50 通过、2 Linux 专用项跳过；独立 RC8 镜像 Linux 52/52 通过。
- 用真实组件配合延迟 6 秒的遥测接口测试：未开始采样、请求等待及超时期间标签/占位值均保持显示。
- 在已登录的正式 DSH 浏览器刷新：首个可见页头已经包含 GPU、功耗、核心、显存、P、D、KV、命中，占位值随后被真实数据替换；切到运行配置时不再空等 5 秒。认证要求保持不变。

## 镜像状态

独立候选 local/vllm-sm75:v0.1.5-ultra-ui-rc8 构建成功，ID sha256:f6ad9a5a67ebcc4d74867c19ed00595c0f4faeac27d99b68d429ca73041703ef，源清单 SHA256 e373ae79e2ff5d837244470d98b8ae06d1d9489bfe529c6687bba10c53d315fc。

生产仍是 RC7 容器加上述静态资源热修复，并未宣称完成 RC8 容器切换。Unraid 模板仍为 RC7；常规 stop/start 保留容器文件，但从 RC7 镜像重新创建会丢失此次热修复，后续重建应使用已验证的 RC8 候选。未推送 GitHub 或公开镜像。

## 性能监控页例外

按用户后续要求，性能监控隐藏共享遥测指标，并停用该页的额外遥测轮询；保留标题、配置选择和监控操作栏。切页时立即设置可见性，其他页面仍立即显示共享指标。正式 DSH 浏览器验证：性能监控 summaryHidden=true、headerVisible=true、logsVisible=true；快速会话 summaryHidden=false。仅 app.js 热更新，无服务重启；原文件备份在 /path/to/sm75-artifacts/ui-monitor-header-20260918/app.before.js。RC8 候选未包含此次后续调整，下一次镜像构建需使用当前源码。


## 加速标识增强

将闪电从右上角小角标改为贯穿 vLLM 蓝/金 V 形的亮黄色主体，带深色隔离轮廓；同步再生成 SVG、PNG、ICO、maskable 资产。Pillow 12.1.1 生成，检查 16/32/48/64 像素预览及正式 DSH 24/34 像素显示。品牌静态资源热更新，未重启服务；原资源备份 /path/to/sm75-artifacts/ui-logo-20260918/branding-before/。RC8 候选不含此后续图标修改，下一版镜像使用当前源码。


用户随后明确 vLLM 是主体：缩小闪电并移至蓝色右翼，保留 V 中心和基础轮廓。重新生成全部图标，正式 SVG 内容及 192 PNG 的 SHA256 与本地一致。静态资源热更新无重启，回滚资产在 /path/to/sm75-artifacts/ui-logo-vllm-primary-20260918/branding-before/；镜像候选仍需后续合入当前资源。


随后按用户要求降低闪电至 78% 不透明度，进一步缩小并右移至主体边缘，描边减至 8/512。SVG 与 PNG 均采用一致的透明合成；完成小尺寸预览与线上 SVG/PNG 校验。无重启热更新，前版备份 /path/to/sm75-artifacts/ui-logo-translucent-20260918/branding-before/。


最终按用户指定，将闪电进一步缩小到画布高度约 29%，放在左下角作为独立点缀，与 V 形轮廓分离，保留 78% 不透明度。重新生成并目视检查小尺寸预览，线上 SVG 与 192 PNG 校验通过；无重启热更新，前版备份 /path/to/sm75-artifacts/ui-logo-corner-20260918/branding-before/。
