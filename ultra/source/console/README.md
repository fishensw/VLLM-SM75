# SM75 本地候选控制台

独立管理服务，Node.js 22。启动：

```bash
SM75_CONSOLE_ROOT=/absolute/path/console-data node console/server.mjs
```

默认地址 `http://127.0.0.1:1616`，登录凭据首次生成在数据目录的 `key` 文件。局域网使用 `SM75_CONSOLE_HOST=0.0.0.0` 并设置 `SM75_CONSOLE_PUBLIC_HOST`；所有管理 API 必须登录。Harness 代理默认端口 1617，需要同时可访问。

已有原生环境可通过 `SM75_VLLM_BIN=/absolute/path/venv/bin/vllm` 指定引擎入口。启动程序失败会直接返回错误，详细原因写入对应配置日志。

测试环境仅启动 `local/vllm-sm75:v015-*` 镜像，仅管理带 `sm75.managed=v015-candidate` 标签的候选容器。正式配置可作为参数输入，不会接管已有容器。

功能：配置导入与参数向导、模型下载任务、候选引擎启停、流式 Chat、GPU 遥测、双轴看板和 Harness 原生页面。原生进程运行后端已提供配置生成，原生完整安装器尚未验收。

模型与编译缓存分离。所有候选编译产物写入配置中的 `cacheRoot`。vLLM、Triton、FlashInfer、TorchInductor、PyTorch 扩展与 CUDA 驱动缓存均持久化。FlashInfer 原生路径通过基础目录下的 `.cache/flashinfer` 映射，避免变量含义导致重复嵌套。

Harness 固定 `0.1.5-rc.1`，运行于独立无 GPU 容器，仅挂载自己的 home 和 workspace，内部只监听回环地址。控制台代理完成原生认证，浏览器不获取原生 token。当前本地离线镜像复用 v0.1.4 基础层和已下载 Node/DSH 运行环境；标准 Dockerfile 使用 Node 基础镜像。

运行测试：`node --test console/test/*.test.mjs`。Chat 代理单测使用模拟上游，不代表实际模型推理或工具调用验收。真实模型、休眠/P8 和零重复编译以 GPU 测试记录为准。


看板提供 MTP / DFlash2 接受率（接受草稿 tokens ÷ 提出草稿 tokens）、平均接受长度（1 + 接受 tokens ÷ 验证轮次，含主模型补充 token）和各位置接受比例。趋势只绘制发生投机验证的区间，缺测与空闲留空。普通模式显示未配置投机解码。

控制台按运行配置保存最近 15 分钟指标和 Chat 会话；刷新页面会恢复，检测到引擎计数重置时开始新统计。模型停止后仍可查看保留的历史。下载任务支持取消与重试，文件保存在独立模型目录；模型库可以把本地路径填入运行配置。
