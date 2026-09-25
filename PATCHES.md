# 定制边界与合入顺序

所有`patches/`文件来自实测构建输入，SHA见`PATCH-SHA256SUMS`；仅整理文件布局及文件末尾换行，不改变运行逻辑；没有把实验实现冒充成新的通用实现。

| 文件 | 作用 | 合入边界 |
|---|---|---|
| patch_baseline.py | 模型/HC FP16、强制关闭QSA、跳过indexer、Torch Inductor判断修改 | 实验基线；涉及全局工具函数和Torch代码，不能直接设为所有模型的默认行为 |
| patch_hc_candidate.py、hc_gemv_candidate.py | 三个HC投影形状的GEMV分派 | H1环境变量开启，其他形状回退；需要保持开关及适用形状限制 |
| patch_mtp_fp16.py | MTP中显式BF16 HC参数改为FP16 | 应限定模型、dtype和设备能力 |
| patch_ple_bytes.py | FP8 PLE pinned lookup按uint8视图搬运 | 不改变反量化；覆盖FP8编码、边界索引和Graph |
| patch_mtp_norm.py、sm75_mtp_norm.py | MTP两处Gemma归一化融合 | 保留参数名及FP32归约/乘法、FP16输出；限定兼容输入和后端 |

B9是新版`engram-config.cpu_offload=true`对应的放置策略：PLE embedding表CPU卸载，投影驻GPU。配置中的MTP步数、上下文、并发、KV大小、Graph和工作区均属于推荐模板。

旧`patch_fp8ple.py`选择器补丁不属于本版本：v0.1.6已原生识别PLE dtype，本包明确不包含它。

## 建议后续next分支提交拆分

1. SM75 FP8 PLE搬运修复及验证。
2. 模型/MTP dtype适配，使用模型和设备条件限定。
3. 可选H1及MTP融合归一化。
4. Flash-Next推荐模板、构建及测试资料。
5. 单独处理QSA/indexer路线和Torch编译修改；不得将实验中的`if True`当作通用BF16能力判断。

这些是后续集成建议，不代表本次已完成重构。当前补丁用于隔离实验镜像复现。
