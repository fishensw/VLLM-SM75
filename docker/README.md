# 构建目录约定

公共入口只有 `build.sh`、`run.sh`，两者读取唯一版本源 `VERSION`。以 `EDITION=standard|ultra` 选择产品，以 `VARIANT=base|mtp|dflash2` 选择标准版推理模式。

| 文件 | 作用 |
|---|---|
| `VERSION` | 标准版与 ultra 共用的版本号 |
| `build.sh` | `BUILD_MODE=full`；另有 standard `fast` 和 ultra `ui` 开发模式 |
| `run.sh` | 标准版/ultra 统一启动，不停止或删除现有容器 |
| `Dockerfile` | 固定官方底座的标准版完整构建 |
| `Dockerfile.fast` | 开发基座，复用 `helpers/` 安装器；不替代完整发布验收 |
| `helpers/` | 唯一安装器、校验器、内部启动及 fast 编译实现 |
| `speculative/` | SM75 投机 overlay 源码 |
| `BUILD.md` | 当前构建、运行和回退说明 |
| `buildkit-isolated.toml` | 独立构建 worker 的资源策略，不是宿主 daemon 配置 |

ultra 产品层由 `../ultra/Dockerfile` 维护，UI 增量覆盖由 `../ultra/Dockerfile.candidate` 维护；这两者分工不同，不是按版本复制的配方。旧版 Dockerfile、安装器和 BUILD 副本已从当前树移除，通过 Git 历史恢复。历史性能及发布说明保留在 `docs/`，不与当前操作入口混排。
