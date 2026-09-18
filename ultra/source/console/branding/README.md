# 品牌资产

母版保留项目原 favicon 中 vLLM 的蓝/金 V 形，以左下角独立的半透明黄色小闪电表达对 vLLM 推理的加速，V 形为主、闪电为辅。闪电采用深色隔离轮廓，避免与 V 的金色左翼混在一起；保持 V 的中心负空间和蓝金主体轮廓，不内嵌旧 PNG。透明方案 A 为 `mark-transparent.svg`，圆角底板方案 B 为 `favicon.svg`，当前页面默认 B。闪电覆盖约 29% 画布高度，采用 78% 不透明度，与 V 形主体分离，不遮挡蓝金轮廓，兼顾侧栏、浏览器 favicon 和应用图标的小尺寸辨识。

`mark-mono.svg` 使用 currentColor；PNG、ICO、maskable 和 manifest 由仓库根目录 `tools/build-brand.py` 生成，验证环境 Pillow 12.1.1。生成器同时输出 `evidence/brand-preview.png`，含 A/B 和小尺寸预览。运行：

```bash
python -m pip install Pillow==12.1.1
python tools/build-brand.py
```

登录页、主导航和 DSH 使用 `/brand/favicon.svg`；PWA 使用 `/brand/icon-192.png`、`icon-512.png`、`maskable-512.png`。控制台返回 no-cache，移除三处分别维护的 `?v=14`。静态文档引用本目录文件。

Unraid 模板尚未入库；在其 Icon 字段采用最终发布的 512 PNG 地址。正式源码或镜像尚未发布前，不填写虚构的 raw.githubusercontent.com 地址。现有 vLLM 与第三方版权/许可证保留；本图标表示 SM75 分支，不表示上游官方背书。
