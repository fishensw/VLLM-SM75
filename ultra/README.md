# v0.1.5-ultra

本目录将现有 ultra 控制台、DSH 插件和运行时覆盖层纳入源码管理。标准版的构建入口仍为 `docker/Dockerfile`。本次改动修复控制台认证、跳转、编辑状态与生命周期。

更新内容见 [ultra 发布说明](../docs/releases/v0.1.5-ultra.zh-CN.md)，实测与边界见 [统一验证记录](../docs/validation/v0.1.5.md)。

## 构建与运行

统一版本来自 `../docker/VERSION`，从仓库根目录使用同一入口：

```bash
bash docker/build.sh
EDITION=ultra bash docker/build.sh
ULTRA_DATA_ROOT=/path/to/ultra MODEL_ROOT=/path/to/models \
  EDITION=ultra bash docker/run.sh
```

输出为本地 `vllm-sm75:v<版本>` 与 `vllm-sm75:v<版本>-ultra`，尚无公开镜像发布声明。完整构建在独立构建机进行；生产 Unraid 未隔离导出仍受保护。fast/UI 迭代和参数说明统一见 [构建运行指南](../docker/BUILD.md)，不再维护按 RC 编号复制的命令。

首次安装使用新数据目录；已有部署不能套用新目录丢弃原配置。启动脚本只创建管理服务，登录后登记 `/models` 下已有模型并选择配置。默认常驻 P-State（空闲 P8 / 负载回 16），自动休眠 exit 保留为可选配置。P-State 需要设置 `PSTATE_NVAPI_LIB`；未挂载时选择 sleep 电源模式。

Node 22.18.0 固定 amd64 镜像 digest，DSH 0.1.5-rc.1 和传递依赖由 `harness/package-lock.json` 锁定，使用 `npm ci`。完整构建不再复制宿主 Node 或私有 Harness 镜像。标准版基础环境、FlashQLA 编译和 sm_75 检查沿用 `docker/Dockerfile`。

`Dockerfile.candidate` 仅供快速验证：从已部署镜像复制控制台覆盖层，必须通过 `BASE_IMAGE` 指定已核对完整 image ID 的本地独立标签。它不构成完整源码构建证明。`tools/package-ultra-candidate.py` 生成源码包和 SHA256 清单。

## 首次使用与升级

### 首次登录：自动密钥在哪里、怎么看

首次启动会自动生成 **Web 登录 token**，无需自行填写或预设。容器内保存在 `/data/key`；使用上面的统一运行脚本时，宿主机对应 `<ULTRA_DATA_ROOT>/console/key`，例如 `/path/to/ultra/console/key`。重启或升级保留该数据目录，就会继续使用原 token。

在 **Docker 宿主机终端**执行以下命令，输出的一整行就是登录 token：

```bash
docker exec vllm-sm75-ultra node /opt/sm75-workbench/console/auth-cli.mjs show
```

若设置过 `CONTAINER_NAME`，将 `vllm-sm75-ultra` 换成实际容器名；可用 `docker ps --format '{{.Names}}'` 查看。Unraid 可在 Docker 页面点击 ultra 容器图标 → **Console / 控制台**，进入容器后执行：

```bash
node /opt/sm75-workbench/console/auth-cli.mjs show
```

浏览器打开 `http://<宿主机局域网IP>:1615`（独立容器 IP 部署则使用容器 IP；自定义端口使用实际端口），将输出的 token 粘贴到登录框即可。此 token 用于 Web 管理登录，**不是模型推理 API key**。查看命令不会轮换已有 token；不要为查看密钥执行 `reset`。


### DSH 工具身份

DSH 固定以 UID/GID 1000 运行，降权失败会报错，不再回退 root。
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

`SM75_SESSION_TTL_SECONDS` 可设置 60–2592000 秒。LAN HTTP 不设置 Secure；HTTPS 反代仅在 `SM75_TRUSTED_PROXIES` 显式列出的直连代理 IP 上信任 `X-Forwarded-Proto: https`，反代须保留原 Host。该变量是代理信任列表，**不是客户端网段白名单**。账户登录与 CIDR 策略尚未实现；`Auth.principal()` 是后续策略接入点。

## 页面行为

- 登录首先确认管理会话，成功后进入管理页；仅显式从 DSH 返回登录时恢复 `/dsh/`。
- DSH/模型 503、网络超时与 Web 登录失败分开处理；不使用构建号强制刷新页面。
- 管理导航支持深链、刷新、前进/后退。DSH 自己管理内部面板，不重复写顶层路由。
- 配置离开时可保存、放弃或取消；DSH 中保留配置/设置草稿、快速会话和测试页面，隐藏页面暂停重复遥测请求。
- 测试页面保持挂载不能保证浏览器后台计时精度，性能结果仍需固定前台/后台契约。

## 图标

`source/console/branding` 为统一资产目录，`/brand/` 统一提供 favicon、PWA 和页面图标。`tools/build-brand.py` 从矢量几何生成 PNG/ICO，依赖 Pillow。保留透明、深色底板和单色版本；图标不表达官方背书。

## 回退

候选验证使用独立数据目录。切换前停止旧容器并备份配置、SQLite、会话、DSH 数据；启动候选失败时，停止候选，再用原 image ID、原挂载和原启动参数恢复旧容器。不要只依赖可被移动的 `-prev` 标签。新增会话文件不要求删除，旧程序可忽略；数据库迁移后如需恢复快照，应同时恢复对应版本数据，不能用旧快照覆盖新产生的会话而不告知用户。

当前结论与未完成门禁见 [发布状态](../docs/releases/v0.1.5.md)；逐轮开发历史另见 [RC 记录](../docs/releases/v0.1.5-ultra-rc.md)。
