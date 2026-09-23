# SM75 主题与 Harness 0.1.7

外观由 Harness 原生主题服务保存；侧栏快捷按钮与原生设置使用同一个 `ctx.theme` 接口。浅色、深色和跟随系统沿用新版完整原生色板，插件面板从 `--dsw-*` 变量取值，不再用旧版整包覆盖。

## 字号

保留 ultra 的 12–26 px 范围，默认 14。构建时仅把新版主题客户端和服务端的字号上限从 17 扩展至 26，并将 UI 中固定的字体和行高替换为 `--dsh-font-scale` 表达式。工作台插件通过公开 `theme/change` 事件设置 `--dsh-font-scale = fontSize / 14` 和 `--ui-fs`，不使用 zoom 或 transform 缩放布局。

## 插件变量与嵌入页面

面板 `--bg`、`--card`、`--text`、`--muted`、`--line`、`--blue` 等映射到对应的原生背景、文本、边框和状态变量。监控与测试 iframe 继续通过 `postMessage {type:'sm75-theme'}` 接收外观变更。

## 构建入口

- `overlay/font-scale.py`：针对固定版本的字号范围和可重复执行的字体规则补丁。
- `source/console/plugins`：品牌、工作台入口、字体桥接和统计插件。
- `source/console/install-native-plugins.mjs`：安装插件及最小运行时 hook。

禁止复制旧版 theme/layout/chat/sidebar/preview/deliverables bundle 覆盖新版；不要硬编码整页色板或缩放布局。
