# SM75 主题规范（唯一参考）

主题来源：DSH 外观设置（浅色 / 深色 / 跟随系统）；两个快捷入口（侧栏 ☀/☾ 悬浮按钮、页面标题栏 ☾）调用同一接口。

## 1. 色板（我们的定制值）

| 语义 | 变量 / token | 浅色 | 深色 |
|---|---|---|---|
| 页面底 | `--dsw-alias-bg-base` | `#f4f6f9` | `#151517`（DSH 原值） |
| 侧栏 | `--dsw-specific-sidebar-fill` | `#eef1f5` | `#1b1b1c`（DSH 原值） |
| 卡片/弹窗 | `--dsw-alias-bg-layer-2` | `#ffffff` | `#202020` |
| 输入/按钮底 | `--dsw-specific-selector` | `#f5f6f7` | `#353638` |
| 悬停 | `--dsw-alias-interactive-bg-hover` | 深 6% | 白 8% |
| 选中 | `--dsw-alias-interactive-bg-active` | 深 10% | 白 14% |
| 边框 l1/l2/l3 | `--dsw-alias-border-l1/2/3` | 黑 4/10/12% | 白 6/12/16% |
| 文字 主/次/辅 | `--dsw-alias-label-primary/secondary/tertiary` | `#0f1115`/`#61666b`/`#8b9096` | bluish-50/300/400 |
| 链接蓝 | `--dsw-alias-link` | deepseek-500 | deepseek-400 |
| 用户气泡 | `--dsw-specific-bubble` | `#e4edfd` | bluish-850 |
| 成功/警告/错误 | `--dsw-alias-state-success/warn/error-primary` | green-500 / amber-500 / red-600 | 同（error 用 red-400） |
| 分隔线/阴影 | `color-mix(in srgb, var(--text) 20~32%, transparent)` | 自动 | 自动 |

## 2. 面板变量映射（插件 :host）

`--bg`→bg-base、`--card`→bg-layer-2、`--text`→label-primary、`--muted`→label-secondary、`--line`→border-l3、`--blue`→link、`--green/--warn/--error`→state-*、`--field`→specific-selector、`--active`→interactive-bg-active、`--gpu-low/medium/elevated/high`→state-*、字体族→`--dsw-font-family`。

## 3. 字号

单一来源：DSH 字号设置（12–26，默认 14）。`--dsh-font-scale = 字号/14` → 面板 `--ui-fs`。只缩放 `font-size` / `line-height`，**不缩放布局**（已否决 zoom）。
类型刻度：正文 16、次要 14、辅助 13、标题 20–24（均随缩放）。

## 4. iframe

监控面板 `dashboard.html` 与测试 iframe 通过 `postMessage {type:'sm75-theme'}` 接收模式；调色板与本规范一致（浅 `#f4f6f9`/白卡、深 `#151517`/`#202020`）。

## 5. 构建固化

- `overlay/dsh-theme/client.js|index.js` → 主题插件（浅色表面值、字号上限 26）
- `overlay/dashboard.html` → 监控面板
- `overlay/font-scale.py` → 字号缩放变量注入（构建时执行）
- `source/console` → 控制台（style.css / app.js / plugins / server.mjs）

## 6. 禁止事项

- 不得向页面注入整页调色板覆盖（会造成半浅半深）。
- 不得写死颜色；一律走变量。
- 不得用 zoom/transform 缩放布局。
