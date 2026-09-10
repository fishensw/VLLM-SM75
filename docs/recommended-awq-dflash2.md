# AWQ DFlash2 推荐配置（v0.1.4）

27B AWQ在PR兼容修补镜像已完成1–128K吞吐验证；整合镜像已完成构建、AWQ启动和单请求检查，其他验收范围见详细记录。采用4×T10 16GiB、TP4、DFlash draft7、seq8、batch16384、利用率0.87、4GiB/卡GPU KV、8GiB CPU KV、FP8 e4m3 KV及capture[8]。参数不按FP8配置覆盖。

```bash
export VLLM_API_KEY='replace-with-your-api-key'
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-sm75/cache
export VLLM_SM75_MODEL_CACHE_ROOT=/path/to/existing-model-cache
export MODEL_ROOT=/path/to/downloaded-models
export MODEL=/models/Qwen3.8-27B-W4A16-AWQ
export DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2
VARIANT=dflash2 FORMAT=awq AUTO_SLEEP_IDLE_TIMEOUT=30 \
  AUTO_SLEEP_OFFLOAD_TARGET=exit bash docker/run.sh
```

主模型使用 `philbert440/Qwen3.8-27B-W4A16-AWQ`，草稿使用 `incoai/Qwen3.8-27B-DFlash2`；路径换成已有目录，避免重复下载。模型和草稿需持续可读，编译缓存可写并保留挂载；主机预留8GiB CPU KV及系统/加载内存。

脚本保留AWQ独立vLLM/Triton缓存，共享FlashInfer/扩展缓存。默认启用Firefly及其自动通信路径，官方FlashInfer all-reduce设为0，与实测一致。关闭Firefly用 `VLLM_FIREFLY=0 VLLM_FIREFLY_AR=0`。

32K测得1433.67 tok/s prefill、215.57 tok/s decode；128K为1124.88/198.40 tok/s。单轮合成测试，不能视为日常聊天或旧版提升比例。见[完整数据](validation/v0.1.4-awq.md)。

exit是日常省电推荐策略，超时单位分钟，60秒检查用1；AWQ本轮未测试休眠/P8与唤醒，不将FP8的P8数据当成AWQ验收。详见[休眠要求](sleep-and-cache.md)。
