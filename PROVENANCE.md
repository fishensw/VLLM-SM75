# 版本与复现来源

- 基础版本：vllm-sm75 v0.1.6 Ultra。
- 测试基础镜像ID：`sha256:a807a7e286d4fc80842fece6a68f852a67754a163d50a6f2325185d42de37e16`。
- vLLM 0.30.0，上游提交`ced6857afa0ea7b2e3f0846a62e1394e90f15607`；Torch 2.13.0+cu129；驱动580.173.02。
- SM75发行源码的完整对应提交未独立补齐，不能用vLLM上游提交代替。
- 测试最终镜像ID：`sha256:0ca7679e0c6d0aa82d69ce1e1ce162974216c6fc7890d7aef556ee751cfd1d5b`。这是来源身份记录，不是可直接拉取的注册表地址。
- 测试容器与r3基础派生镜像2696个vLLM Python文件SHA一致；最终新镜像添加配置/启动文件并通过新容器复验。
- 本包Dockerfile合并了原来的分层构建；未在本次脱敏整理中重新构建验证，不保证镜像ID相同。

主模型：公开仓库`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`，固定修订`79899a0ce76f569c755ae3691dd55f6ef24ff40a`。

主checkpoint未含本轮使用的MTP权重。独立草稿为已有本地RTN INT4 group32转换，约3.86GiB，来源声明为`RadixArk/Qwen3.8-Flash-Next-NVFP4`修订`7b719225242aacd3dbd3f9407468c2ee9a9d2594`。本包未包含权重或完整转换流程，因此不能称为一条命令从公开主模型重建全部依赖；使用者需要准备并另行验证兼容草稿。

GPU验证脚本的核心算法与断言保持来源实现；本次仅将MTP权重/输出位置参数化，并从已安装补丁模块导入融合算子。整理后只做语法和静态校验，尚未重跑GPU测试。

可在空闲八卡、已构建隔离镜像中分别运行：

```bash
python3 /opt/flashnext/tests/test_ple_bytes.py
MTP_DENSE_PATH=/models/fp8ple/runtime/mtp-int4-g32/mtp-dense.safetensors \
  TEST_OUTPUT_DIR=/tmp/next-tests \
  python3 /opt/flashnext/tests/test_mtp_norm.py
```

不要与已占用八卡的模型服务同时运行这些GPU验证脚本。
