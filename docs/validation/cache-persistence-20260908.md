# v0.1.3 编译缓存排查（2026-09-08）

## 已确认的结论

五个正式配置均将 `/root/.cache/vllm` 和 `/root/.cache/flashinfer` 以可写目录挂载到宿主机，宿主目录全部存在。FP8 DFlash2 正在使用的缓存目录与 60 秒测试容器一致。缓存持久化有效，没有清理或替换缓存，也没有重启正在使用的服务。

FP8 DFlash2 的 vLLM 与 FlashInfer 编译缓存均存放在宿主机持久化目录，并分别挂载到容器内对应缓存路径。

## 30 分钟正式启动为何重新编译

镜像 `e5e03506b4ee7335a42c1897d0a75a5134118145e1854067c03e82ddbf48d011` 中，`vllm.envs.compile_factors()` 将四个 `VLLM_AUTO_SLEEP_*` 运行设置纳入编译缓存环境指纹。普通编译缓存和 AOT 缓存均使用该指纹。

逐项比较真实的 `cache_key_factors.json`：

| 项目 | 60 秒测试 | 30 分钟正式启动 |
|---|---|---|
| 编译缓存目录键 | `012a178047` | `b8d72a3915` |
| `VLLM_AUTO_SLEEP_IDLE_TIMEOUT` | `1.0` | `30.0` |
| 其余环境因子 | 完全一致 | 完全一致 |
| 模型配置、代码、编译器哈希 | 完全一致 | 完全一致 |

因此这次全套重新编译由休眠超时进入缓存键触发，不是缓存挂载丢失。正式启动日志记录总编译约 167.40 秒，主模型 torch.compile 120.91 秒、DFlash head 39.85 秒、候选选择器 3.60 秒；各计时存在统计边界差异。

## 命中证据与覆盖边界

同镜像此前 11:54:07 明确记录四个 rank `Directly load AOT compilation`，FP8 DFlash2 主模型 torch.compile 为 9.53 秒，证明本地 v0.1.3 能读取持久化缓存。

那一轮 DFlash head 仍编译 71.91 秒、候选选择器 7.75 秒。其目录从 `012a178047` 变为 `9081871bb9`；环境因子一致，但 config_hash 与 code_hash 发生变化。这是独立的草稿模型缓存失配，不能归因于休眠超时，也没有在本轮证明其具体成因或完成修复。不能据主模型缓存命中承诺整个 DFlash 启动或每次唤醒都只编译数秒。

README 的 3.70 秒是历史 FP8 MTP5 匹配缓存记录，202 秒是当次完整启动时间，不能直接套用为当前 DFlash2 的完整启动指标。

## 本地修复

`vllm/envs.py` 的 compile_factors 忽略四个与计算图无关的休眠运行设置：空闲超时、卸载目标、重新加载路径、页缓存维护间隔。启用睡眠分配器等真正影响模型配置的因素仍由模型配置哈希控制，没有整体关闭缓存校验。

`tests/test_sleep_compile_cache.py` 用真实环境变量 getter 与 compile_factors 检查缓存键：逐一改变五组休眠设置时键保持不变，改变图相关 VLLM_USE_LAYERNAME 时键改变。旧实现复现五个子用例失败，修复后两个测试全部通过。新测试文件 lint 通过；envs.py 全文件 lint 仍报告原有七项问题，均不在本次变更行。

修复仅位于本地源码，未构建或替换正式镜像、未修改在线容器、未推送 GitHub。应用这项修复会形成新的缓存键，不能保证直接沿用旧键下的缓存；可能需要一次重新编译。本轮没有进行新的 GPU 重启或睡眠实验。

原始对比与回归输出保存在本地 `local-tests/startup-audit-20260908/cache-comparison.txt`、`cache-baseline-regression.txt`，检查脚本为 cache-audit.sh、cache-all.sh、cache-compare.sh。
