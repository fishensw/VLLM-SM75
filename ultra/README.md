# vLLM-SM75 v0.1.6 Ultra

本目录将 Ultra 控制台、DSH 插件和运行时覆盖层纳入源码管理。标准版的构建入口为 `docker/Dockerfile`。v0.1.6 回收已部署 v0.1.5 Ultra 的管理员账号与网段限制、图片会话、投机状态确认和用量统计修复。监控 API 统一继承匹配版本的标准镜像，避免 Ultra 的旧覆盖层覆盖引擎修复。

完整镜像验收见 [完整镜像报告](../docs/validation/v0.1.6-full-images.md)。Harness 更新验收及官方 wheel 加候选覆盖的四张 T10 测试见 [v0.1.6 验证记录](../docs/validation/v0.1.6.md)。历史版本内容见 [v0.1.5 Ultra 发布说明](../docs/releases/v0.1.5-ultra.zh-CN.md) 与 [v0.1.5 验证记录](../docs/validation/v0.1.5.md)。

## 构建与运行

统一版本来自 `../docker/VERSION`，从仓库根目录使用同一入口：

```bash
bash docker/build.sh
EDITION=ultra bash docker/build.sh
ULTRA_DATA_ROOT=/path/to/ultra MODEL_ROOT=/path/to/models \
  PSTATE_NVAPI_LIB=/usr/lib64/libnvidia-api.so.1 \
  EDITION=ultra bash docker/run.sh
```

输出为本地 `vllm-sm75:v<版本>` 与 `vllm-sm75:v<版本>-ultra`，尚无公开镜像发布声明。完整构建在独立构建机进行；生产 Unraid 未隔离导出仍受保护。fast/UI 迭代和参数说明统一见 [构建运行指南](../docker/BUILD.md)，不再维护按 RC 编号复制的命令。

首次安装使用新数据目录；已有部署不能套用新目录丢弃原配置。启动脚本只创建管理服务，登录后登记 `/models` 下已有模型并选择配置。默认常驻 P-State（空闲 P8 / 负载回 16），自动休眠 exit 保留为可选配置。P-State 需要设置 `PSTATE_NVAPI_LIB`；未挂载时选择 sleep 电源模式。

Node 22.23.2 固定 amd64 镜像 digest，DSH 0.1.7-alpha.2（预发布版）、pnpm 11.7.0 和传递依赖由 `harness/package-lock.json` 锁定，使用 `npm ci`。完整构建不再复制宿主 Node 或私有 Harness 镜像。标准版基础环境、FlashQLA 编译和 sm_75 检查沿用 `docker/Dockerfile`。

上面的 `PSTATE_NVAPI_LIB` 要替换为宿主机实际路径。Linux 需要匹配宿主驱动的 `libnvidia-api.so.1` 和 `libnvidia-ml.so.1`；Ultra 显式挂载前者，后者由 NVIDIA 容器运行时提供。发行版软件包、路径查找和容器内检查命令见 [Linux 驱动库准备](../docker/BUILD.md#linux-驱动库准备)。

`Dockerfile.candidate` 仅供快速验证：保留已部署推理镜像，重新安装锁定的 Harness 运行时并复制控制台，必须通过 `BASE_IMAGE` 指定已核对完整 image ID 的本地独立标签。它不构成完整源码构建证明。仓库根目录的 [`tools/package-ultra-candidate.py`](../tools/package-ultra-candidate.py) 生成源码包和 SHA256 清单。

## v0.1.6 兼容与修复

- 升级继续使用原 `/data`，保留已有 Web token、管理员账号、网段白名单、会话和独立引擎 API key。账户凭据使用 scrypt 派生摘要保存；账户设置变更使其他会话失效。
- 创建管理员账号后使用账号密码登录；本地 `auth-cli.mjs recover` 可重置账号及网段限制并轮换 Web token，原配置会改名备份。该命令属于遗失凭据时的恢复操作，普通升级不要执行。
- 快速会话支持 PNG/JPEG/WebP/GIF，单张不超过 20 MiB、4000 万像素。每条消息最多 4 张；附件按模型配置隔离，转发前检查当前模型是否允许图片；纯语言模式或图片数量限制为零时不启用图片输入。上传与生成共用请求锁和取消信号，上传期间重复发送或切换配置不会混写会话。
- 投机按钮先读取引擎真实状态，写入后再次确认，并区分已生效与等待排空；关闭投机保留草稿权重显存。
- 主侧栏平级提供快速会话、工作台、模型库、运行配置、性能监控、模型测试、使用统计、插件和设置。模型列表、配置列表及编辑表单使用主内容区全宽。
- “使用统计”展示 Watcher 原生 Insights，按 DSH 会话投影汇总；需要 DSH 运行，加载后切换页面保留统计视图与状态。旧控制台独立统计接口已移除。
- “设置”使用横向六页签：外观显示、助手工具、模型连接、参数模板、部署硬件、账号访问。尚未创建运行配置时，仍可先编辑部署与账户设置。
- 公共顶栏高 64 px，工作台会话工具与实时指标合并为单行。标签和单位默认 14 px，数值为 16 px 粗体等宽数字，状态文字为 15 px；随全局字号调整，指标保持完整显示，不横向滚动、不换行、不缩小字号。
- “设置 → 外观显示 → 全局字号”统一调整侧栏、表单、工作台、Watcher、监控与模型测试页面，保留标题、正文和辅助文字的原有层级。字号范围为 12–17，默认 14；侧栏默认基准恢复为 16 px。连续调整会等待正确的保存顺序，刷新后保留选择。

使用 `node --test --test-concurrency=1 --test-timeout=30000 test/*.test.mjs` 验证控制台；本轮控制台 203 项检查中 196 项通过、7 项环境相关跳过；实际宽度、原生页面与浏览器验收记录见 [工作台整合验收](../docs/validation/v0.1.6-workbench-integration.md)。图片回归需要通过 `SM75_PYTHON` 指定已安装 Pillow 的 Python。测试覆盖配置迁移、原生设置写入、原生工作台 HTTP/WS 鉴权代理、插件重复安装、字体事件以及 V3/V4 日志统计。控制台测试不启动 GPU 模型；真实浏览器和推理验证分别记录。

## Harness 0.1.7 适配

固定到官方 `dsh-v0.1.7-alpha.2`。管理端口根地址始终由现有 SM75 主界面承载，DSH 对话、工作区、工具和 Watcher 直接挂载在同一页面，共用一条主侧栏；工作台不包含 iframe、第二个页面外壳或重复设置入口。原生静态资源、RPC 与 WebSocket 经同源鉴权代理提供，原生组件的状态与插槽保持上游契约，品牌、页面联动和 Watcher 统计使用插件接口；字体通过主题事件同步全局 12–17 设置，默认 14。

已有 `settings.yaml` 由上游一次性迁移到 profile；控制台只维护本地 vLLM provider 与默认模型，保留其他 provider 和原生设置编辑。升级不清空会话，Watcher 同时识别 V3/V4 工具结果并重建旧投影缓存。未启用的 spill-policy 不新增默认值；若自行配置旧 `maxInlineBytes`，需按 token 预算重新选择 `maxInlineTokens`，不要直接复制数值。

本次额外验证范围及证据见 [Harness 升级记录](../docs/validation/v0.1.6-harness.md)。Harness 上游版本仍为 alpha；本轮界面验收范围另见 [工作台整合验收](../docs/validation/v0.1.6-workbench-integration.md)。

### 初始化与配置写入恢复

工作台、使用统计、插件和原生设置共用 Harness 服务。初始化按插件加载、本地模型配置、默认模型三个阶段完成；就绪接口记录失败阶段，写锁超时标识为 `CONFIG_LOCK_TIMEOUT`。锁正常释放后，就绪轮询会限频重试并合并并发初始化，不需要重启模型。

异常退出可能留下旧写锁。程序不会自动删除其他写入者的锁；维护时应先确认锁所属进程已退出、没有其他实例共用该 profile，再备份并清理孤立锁。不要为恢复初始化删除整个配置或会话目录。配置监听只覆盖目标文件和必需的父目录，避免无关受保护备份干扰设置加载。

## 首次使用与升级

### 首次登录：自动密钥在哪里、怎么看

首次启动会自动生成 **Web 登录 token**，无需自行填写或预设。容器内保存在 `/data/key`；使用上面的统一运行脚本时，宿主机对应 `<ULTRA_DATA_ROOT>/console/key`，例如 `/path/to/ultra/console/key`。重启或升级保留该数据目录，就会继续使用原 token。

在 **Docker 宿主机终端**执行以下命令，输出的一整行就是登录 token：

```bash
docker exec vllm-sm75-ultra node /opt/sm75-workbench/console/auth-cli.mjs show
```

若设置过 `CONTAINER_NAME`，将 `vllm-sm75-ultra` 换成实际容器名；可用 `docker ps --format '{{.Names}}'` 查看。Unraid 可在 Docker 页面点击 Ultra 容器图标 → **Console / 控制台**，进入容器后执行：

```bash
node /opt/sm75-workbench/console/auth-cli.mjs show
```

浏览器打开 `http://<宿主机局域网IP>:1615`（独立容器 IP 部署则使用容器 IP；自定义端口使用实际端口），将输出的 token 粘贴到登录框即可。此 token 用于 Web 管理登录，**不是模型推理 API key**。查看命令不会轮换已有 token；不要为查看密钥执行 `reset`。


### DSH 工具身份

DSH 固定以 UID/GID 1000 运行，降权失败会报错，不再回退 root。

智能体默认使用简体中文回应，包括进度和工具失败说明；用户明确指定其他语言时遵从用户选择。原生 `list_directory` 使用 Harness 文件系统接口列出当前会话工作区的直属文件、目录及空目录，支持分页；拒绝越界路径和外指符号链接，不启动命令行。查找文件、搜索内容和读取文件继续使用 `glob`、`grep`、`read`。

命令执行依然需要可用的系统沙盒。当系统缺少可用的 Linux 沙盒运行器时，智能体会用中文说明命令无法运行，并继续使用可用的原生文件工具；这不代表命令执行已恢复，也不通过扩大工作区权限或关闭安全限制绕过。
管理进程需要以下容器能力来维护配置文件和停止工具进程：

```text
--cap-drop ALL --cap-add CHOWN --cap-add FOWNER --cap-add DAC_OVERRIDE
--cap-add SETUID --cap-add SETGID --cap-add KILL --security-opt no-new-privileges
```

这些能力属于管理进程；DSH 降权后有效能力为零。不要授予 privileged、SYS_ADMIN 或挂载 docker.sock。
从旧 root DSH 升级时，先停止相关容器，将 `/dsh/home` 和 `/dsh/workspace`
对应宿主目录复制到独立升级目录，仅将副本所有者改为 1000:1000，不跟随符号链接。
新容器将副本挂到相同的容器内路径，原目录和原容器保留用于回退。
`/data` 及其管理凭据保持原挂载、所有者和权限，并在切换前另做一致性备份。
若升级后已产生新工作台数据，回退前应先保留新目录，不能丢弃新增会话。
部署必须同时更新镜像、上述能力和 DSH 数据权限，不能只热替换 `standalone.mjs`。

管理环境变量不再全量传给 DSH；仅提供工具路径、语言/时区、额外 CA、
DSH 工作目录及专用引擎 API key。管理 token 不应配置为工具凭据。

### 登录与数据

- 控制台数据独立挂载到 `/data`，设置 `SM75_CONSOLE_ROOT=/data`；目录、模型、编译缓存、`/dsh/home`、`/dsh/workspace` 分别持久化。
- LAN 控制台默认监听 1615。沿用实际部署网络与 GPU 参数，先用不同容器名和端口进行候选验收；不要同时启动两个争用同一批 GPU 的模型。
- 首次启动生成 32 随机字节的 Web token，保存在 `/data/key`。从管理终端读取：

```bash
docker exec <ultra容器> node /opt/sm75-workbench/console/auth-cli.mjs show
```

- 浏览器打开 `http://<局域网地址>:1615`，输入 token。服务端保存会话摘要，浏览器仅使用 HttpOnly、SameSite=Strict Cookie；默认绝对有效期 12 小时，管理进程重启保留会话。
- 旧安装保留已有 `key`。如旧安装没有单独的模型 API key，迁移时保持原模型 key，避免破坏客户端；新安装分别生成两个 key。
- 退出只撤销当前 Web 会话，不停止推理。恢复工具仅在管理终端使用：

```bash
docker exec <ultra容器> node /opt/sm75-workbench/console/auth-cli.mjs reset
```

轮换 Web token 会撤销旧 Web 会话和连接，不修改模型 API key。丢失凭据时不要删除整个数据目录。

`SM75_SESSION_TTL_SECONDS` 可设置 60–2592000 秒。LAN HTTP 不设置 Secure；HTTPS 反代仅在 `SM75_TRUSTED_PROXIES` 显式列出的直连代理 IP 上信任 `X-Forwarded-Proto: https`，反代须保留原 Host。该变量是代理信任列表，**不是客户端网段白名单**。管理员账号和 IPv4/IPv6 CIDR 白名单可在“设置 → 账号访问”中配置。白名单按直连地址判断，不信任客户端自填的转发地址；反代部署应使用实际直连代理地址规划策略。保存白名单时必须包含当前连接地址，避免锁定管理入口。

## 页面行为

- 登录首先确认管理会话；`http://<局域网地址>:1615/` 始终显示现有 SM75 主界面。点击“工作台”后，DSH 在内容区加载；尚未运行时显示启动状态。旧 `/dsh/` 地址仅兼容返回根地址，不作为另一套操作入口。
- DSH/模型 503、网络超时与 Web 登录失败分开处理；不使用构建号强制刷新页面。
- 主侧栏导航支持深链、刷新、前进/后退。各页使用一致的顶栏空间预算，切换工作台不临时移动运行配置选择框或推移导航。设置使用内容页中的横向六页签；使用统计和插件直接从主侧栏进入。工作台会话工具位于公共顶栏，原生会话及工具内容由 Harness 管理。
- 配置离开时可保存、放弃或取消；配置/设置草稿、快速会话、测试页面与已加载的 Watcher 页面在切换后保留，隐藏页面暂停重复遥测请求。
- 测试页面保持挂载不能保证浏览器后台计时精度，性能结果仍需固定前台/后台契约。

## 图标

`source/console/branding` 为统一资产目录，`/brand/` 统一提供 favicon、PWA 和页面图标。`tools/build-brand.py` 从矢量几何生成 PNG/ICO，依赖 Pillow。保留透明、深色底板和单色版本；图标不表达官方背书。

## 回退

候选验证使用独立数据目录。切换前停止旧容器并备份配置、SQLite、会话、DSH 数据；启动候选失败时，停止候选，再用原 image ID、原挂载和原启动参数恢复旧容器。不要只依赖可被移动的 `-prev` 标签。新增会话文件不要求删除，旧程序可忽略；数据库迁移后如需恢复快照，应同时恢复对应版本数据，不能用旧快照覆盖新产生的会话而不告知用户。

v0.1.5 的发布结论与历史门禁见 [历史发布状态](../docs/releases/v0.1.5.md)；逐轮开发历史另见 [RC 记录](../docs/releases/v0.1.5-ultra-rc.md)。v0.1.6 的验证边界以对应验收记录为准。
## 运行配置中的工具调用与自定义参数

在“运行配置”展开“高级启动参数”，进入“工具调用与对话解析”设置自动工具选择、工具调用/推理解析器、对话模板及工具解析器插件。解析器须与模型和引擎支持范围匹配，模板/插件路径是引擎容器内路径；未填写的项目沿用引擎默认。

“自定义启动参数”每次添加一个参数，例如 `--seed` 与 `42`；无值开关留空。空格和 JSON 会作为完整参数值保留，不需要 shell 引号。已有参数请直接编辑，API key、模型、监听地址和端口使用对应专用配置。添加或修改后保存运行配置，模型下次启动时生效，不会立即改变正在运行的模型。

## YaRN 长上下文与 KV 配额

运行配置的“上下文与 GPU KV”区域提供 YaRN 1M（1,048,576 tokens）快捷设置、原生长度/倍率和每卡 GPU KV GiB。GPU KV 留空为自动分配；在“CPU KV 与分层缓存”选择“默认”，可开启原生 CPU KV 并填写 TP 合计容量。模型配置自动识别 text_config 并保留 mRoPE 等字段，未知原生长度需手动填写。保存后在模型下次启动时生效。

CPU KV 卸载用于前缀缓存复用，不能替代单个活动请求所需的 GPU KV；降低 GPU KV 容量不保证能运行更长上下文。YaRN 1M 是可选配置目标，实际可用长度仍取决于模型和 GPU KV 容量。超过草稿上限的长请求请先关闭投机，或使用已验证的长上下文草稿。详见 [配置与验证说明](../docs/validation/v0.1.6-context.md)。


## LMCache 分层缓存（实验）

v0.1.6 新增独立 CPU＋磁盘缓存配置、受管服务生命周期与布局隔离目录。使用固定的 LMCache 0.5.5 cu129 和候选 packed4D 回移修复；默认关闭，构建时以 `INSTALL_LMCACHE=1` 安装扩展。未修补版本会被启动检查拒绝。

适合重复前缀和会话复用；CPU/磁盘容量不能替代活动请求的 GPU KV。当前不与 DFlash/MTP、显存休眠或原生 CPU KV 同时启用。磁盘可用空间门槛不是硬配额。配置、安装、实际块粒度及验证范围见 [LMCache 使用说明](../docs/lmcache.md)。

## 参数与面板推荐

[统一配置模板](../docs/configuration-v0.1.6.md) 提供各参数的单位、建议起点和效果。面板按模型、上下文与 GPU KV、CPU KV 与分层缓存、电源分区。“缓存方案”选择“默认”或“LMCache”后只显示对应字段，统一点击顶部“保存配置”。快速组合、P-State 细项和高级参数按需展开；保存不等于重启或已生效。最大上下文以实际请求验收为准，不能把 CPU/磁盘缓存直接加到活动 GPU KV 容量。
