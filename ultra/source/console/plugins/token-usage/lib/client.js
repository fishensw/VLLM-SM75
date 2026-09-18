// Deployment plugin: Settings > 使用统计 (token usage) section for the DeepSeek
// Harness web UI. Faithful port of the rc.6-era patched UsageSection (summary
// cards, daily bar chart, model share pie, detail table) from
// patched-client-settings-general.js, re-registered as a first-class
// `dsh.client` plugin (see cordis.patch.yml). The data comes from the host
// /token-usage.json route (same origin as the GUI).
//
// The section markup/CSS/helpers below are copied verbatim from the original
// patched bundle so the page renders exactly as before; only the module
// registration (window.__ModuleLoader__ + apply/slots) is plugin-shaped.

window.__ModuleLoader__.load({
  id: "@deepseek-ai/dsh-client-ui-token-usage",
  factory: (require) => {
    var module = { exports: {} };
    var exports = module.exports;

    let react = require("react");
    let jsxRuntime = require("react/jsx-runtime");

    // ---- CSS (verbatim from original UsageSection.module.css) ----
    const usageCss = "._Usage_root{flex-direction:column;gap:12px;width:100%;display:flex}._Usage_top{display:flex;align-items:center;justify-content:space-between;gap:12px}._Usage_meta{color:var(--dsw-alias-label-secondary);font-size:11px;line-height:16px}._Usage_btn{background:var(--dsw-alias-interactive-bg-hover);color:var(--dsw-alias-label-primary);border:1px solid var(--dsw-alias-border-l3);border-radius:6px;padding:5px 10px;font-size:11px;line-height:16px;cursor:pointer}._Usage_btn:hover{background:var(--dsw-specific-sidebar-nav-item-hover)}._Usage_filters{display:flex;gap:6px;flex-wrap:wrap}._Usage_filter{background:var(--dsw-alias-bg-layer-2);color:var(--dsw-alias-label-secondary);border:1px solid var(--dsw-alias-border-l3);border-radius:6px;padding:4px 8px;font-size:11px;line-height:16px;cursor:pointer}._Usage_filter:hover{background:var(--dsw-alias-interactive-bg-hover);color:var(--dsw-alias-label-primary)}._Usage_filterActive{background:var(--dsw-specific-sidebar-nav-item-active);color:var(--dsw-alias-label-primary);border-color:var(--dsw-alias-border-l3)}._Usage_cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:8px}._Usage_card{background:var(--dsw-alias-bg-layer-2);border:1px solid var(--dsw-alias-border-l3);border-radius:8px;padding:9px 11px;min-width:0}._Usage_cardInline{display:flex;align-items:center;justify-content:space-between;gap:10px;grid-column:span 2;padding:9px 11px}._Usage_labelInline{color:var(--dsw-alias-label-secondary);font-size:11px;line-height:16px;white-space:nowrap}._Usage_valueInline{color:var(--dsw-alias-label-primary);font-size:12px;line-height:16px;font-weight:600;text-align:right;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}._Usage_value{font-size:15px;line-height:20px;font-weight:600;color:var(--dsw-alias-label-primary);font-variant-numeric:tabular-nums;white-space:nowrap}._Usage_label{color:var(--dsw-alias-label-secondary);font-size:11px;line-height:16px;margin-top:1px}._Usage_charts{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;align-items:stretch}._Usage_panel{background:var(--dsw-alias-bg-layer-2);border:1px solid var(--dsw-alias-border-l3);border-radius:8px;padding:10px 12px;min-width:0;height:230px;display:flex;flex-direction:column}._Usage_chartBody{flex:1;min-height:0;display:flex;align-items:center;justify-content:center}._Usage_barBody{flex:1;min-height:0;display:flex}._Usage_bars{display:flex;gap:6px;width:100%;height:100%;min-height:150px;padding-top:2px}._Usage_barCol{flex:1;min-width:0;display:flex;flex-direction:column}._Usage_barStack{flex:1;min-height:0;position:relative;border-radius:4px;overflow:hidden;width:50%;margin:0 auto}._Usage_barSeg{position:absolute;left:0;right:0}._Usage_barLabel{font-size:14px;line-height:22px;font-weight:500;color:var(--dsw-alias-label-primary);text-align:center;margin-top:6px;white-space:nowrap}._Usage_barLabelHidden{visibility:hidden}._Usage_chartTitle{font-size:12px;font-weight:600;color:var(--dsw-alias-label-primary);margin:0 0 8px}._Usage_legend{display:flex;flex-direction:column;gap:8px;min-width:0;flex:1}._Usage_legendRow{display:flex;flex-direction:column;gap:2px;font-size:11px;color:var(--dsw-alias-label-secondary);min-width:0}._Usage_legendRow span:first-child{min-width:0;overflow-wrap:anywhere}._Usage_table{width:100%;table-layout:fixed;border-collapse:collapse;background:var(--dsw-alias-bg-layer-2);border:1px solid var(--dsw-alias-border-l3);border-radius:8px;overflow:hidden;font-size:11px;line-height:16px}._Usage_th:nth-child(1),._Usage_td:nth-child(1){width:24%}._Usage_th:nth-child(2),._Usage_td:nth-child(2),._Usage_th:nth-child(3),._Usage_td:nth-child(3){width:7%}._Usage_th:nth-child(4),._Usage_td:nth-child(4),._Usage_th:nth-child(5),._Usage_td:nth-child(5){width:12%}._Usage_th:nth-child(6),._Usage_td:nth-child(6),._Usage_th:nth-child(7),._Usage_td:nth-child(7){width:9%}._Usage_th:nth-child(8),._Usage_td:nth-child(8){width:20%;text-align:center}._Usage_th{padding:6px 8px;text-align:right;color:var(--dsw-alias-label-secondary);font-weight:500;white-space:nowrap;border-bottom:1px solid var(--dsw-alias-border-l3);overflow:hidden;text-overflow:ellipsis}._Usage_th:first-child{text-align:left}._Usage_td{padding:6px 8px;text-align:right;color:var(--dsw-alias-label-primary);font-variant-numeric:tabular-nums;border-bottom:1px solid var(--dsw-alias-border-l3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}._Usage_td:first-child{text-align:left;white-space:normal;overflow:visible;text-overflow:clip;word-break:break-word}._Usage_row:hover{background:var(--dsw-alias-interactive-bg-hover)}._Usage_row:last-child ._Usage_td{border-bottom:none}._Usage_error{color:var(--dsw-alias-state-error-primary);font-size:12px;line-height:18px}";
    const usageCssId = "@deepseek-ai/dsh-client-ui-token-usage/UsageSection.module.css";
    if (typeof document !== "undefined" && document.querySelector("style[data-plugin-css=" + JSON.stringify(usageCssId) + "]") === null) {
      const tag = document.createElement("style");
      tag.dataset.plugin = "@deepseek-ai/dsh-client-ui-token-usage";
      tag.dataset.pluginCss = usageCssId;
      tag.textContent = usageCss;
      document.head.appendChild(tag);
    }
    const C = {
      "root": "_Usage_root", "top": "_Usage_top", "meta": "_Usage_meta", "btn": "_Usage_btn",
      "filters": "_Usage_filters", "filter": "_Usage_filter", "filterActive": "_Usage_filterActive",
      "cards": "_Usage_cards", "card": "_Usage_card", "cardInline": "_Usage_cardInline", "labelInline": "_Usage_labelInline", "valueInline": "_Usage_valueInline", "value": "_Usage_value", "label": "_Usage_label",
      "charts": "_Usage_charts", "panel": "_Usage_panel", "chartTitle": "_Usage_chartTitle", "chartBody": "_Usage_chartBody", "barBody": "_Usage_barBody", "bars": "_Usage_bars", "barCol": "_Usage_barCol", "barStack": "_Usage_barStack", "barSeg": "_Usage_barSeg", "barLabel": "_Usage_barLabel", "barLabelHidden": "_Usage_barLabelHidden",
      "legend": "_Usage_legend", "legendRow": "_Usage_legendRow",
      "table": "_Usage_table", "th": "_Usage_th", "td": "_Usage_td", "row": "_Usage_row", "error": "_Usage_error"
    };

    // ---- helpers (verbatim) ----
    function fmtUsage(n) {
      if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
      if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
      if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
      return String(n);
    }
    function usageBig(n) {
      if (n >= 1e8) return (n / 1e8).toFixed(1) + "亿";
      if (n >= 1e4) return (n / 1e4).toFixed(1) + "万";
      return fmtUsage(n);
    }
    function fmtMoney(n) {
      if (n < 0.01) return "$" + n.toFixed(4);
      return "$" + n.toFixed(2);
    }
    const usageColors = ["var(--dsw-static-blue-450)", "#a78bfa", "var(--dsw-static-neutral-bluish-400)", "var(--dsw-alias-label-tertiary)"];
    function usageColor(model, index) {
      if (model === "deepseek-v4-flash") return "var(--dsw-static-blue-450)";
      if (model === "deepseek-v4-pro") return "#a78bfa";
      return usageColors[index % usageColors.length];
    }
    function usageBarHtml(report) {
      const days = (report.daily ?? []).slice(-31);
      if (days.length === 0) return "";
      const max = Math.max(...days.map((d) => d.total), 1);
      const labelEvery = Math.max(1, Math.ceil(days.length / 8));
      return `<div class="_Usage_bars">${days.map((day, i) => {
        const models = Object.entries(day.models ?? {}).sort((a, b) => a[0].localeCompare(b[0])).reverse();
        let bottom = 0;
        const segs = models.map(([model, tokens]) => {
          const h = tokens / max * 100;
          const seg = `<div class="_Usage_barSeg" style="bottom:${bottom.toFixed(2)}%;height:${h.toFixed(2)}%;background:${usageColor(model, Object.keys(day.models).indexOf(model))}"></div>`;
          bottom += h;
          return seg;
        }).join("");
        const show = i % labelEvery === 0 || i === days.length - 1;
        return `<div class="_Usage_barCol"><div class="_Usage_barStack">${segs}</div><div class="_Usage_barLabel${show ? "" : " _Usage_barLabelHidden"}">${day.date.slice(5).replace("-", "/")}</div></div>`;
      }).join("")}</div>`;
    }
    function usagePieSvg(report) {
      const rows = (report.rows ?? []).filter((r) => r.totalTokens > 0);
      if (rows.length === 0) return "";
      const total = rows.reduce((sum, r) => sum + r.totalTokens, 0) || 1;
      const cx = 80, cy = 80, r = 66, ir = 46;
      let start = -Math.PI / 2;
      const parts = [];
      rows.forEach((row, i) => {
        const ang = row.totalTokens / total * Math.PI * 2;
        const end = start + ang;
        const x1 = cx + r * Math.cos(start), y1 = cy + r * Math.sin(start);
        const x2 = cx + r * Math.cos(end), y2 = cy + r * Math.sin(end);
        const large = ang > Math.PI ? 1 : 0;
        parts.push(`<path d="M${cx},${cy} L${x1},${y1} A${r},${r} 0 ${large} 1 ${x2},${y2} Z" fill="${usageColor(row.model, i)}"/>`);
        start = end;
      });
      parts.push(`<circle cx="${cx}" cy="${cy}" r="${ir}" fill="var(--dsw-alias-bg-layer-2)"/>`);
      parts.push(`<text x="${cx}" y="${cy - 4}" text-anchor="middle" font-size="13" font-weight="600" fill="var(--dsw-alias-label-primary)">${usageBig(total)}</text>`);
      parts.push(`<text x="${cx}" y="${cy + 13}" text-anchor="middle" font-size="10" fill="var(--dsw-alias-label-secondary)">tokens</text>`);
      return `<svg viewBox="0 0 160 160" style="width:160px;height:160px;flex:none;display:block">${parts.join("")}</svg>`;
    }

    // ---- section component (verbatim markup from original UsageSection) ----
    function UsageSection() {
      const [report, setReport] = react.useState(void 0);
      const [error, setError] = react.useState(void 0);
      const [range, setRange] = react.useState("all");
      const ranges = [["all", "全部"], ["24h", "24小时"], ["yesterday", "昨天"], ["7d", "近7天"], ["30d", "近30天"]];
      const load = react.useCallback(() => {
        const tryJson = (url) => fetch(url, { cache: "no-store" }).then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        });
        tryJson(`/token-usage.json?range=${range}`).then((value) => {
          setReport(value);
          setError(void 0);
        }).catch((cause) => setError(String(cause && cause.message || cause)));
      }, [range]);
      react.useEffect(() => {
        load();
        const timer = setInterval(load, 10000);
        return () => clearInterval(timer);
      }, [load]);
      if (report === void 0 && error === void 0) return jsxRuntime.jsx("div", { className: C.meta, children: "正在加载本地 token 使用统计…" });
      if (report === void 0) return jsxRuntime.jsx("div", { className: C.error, children: `无法加载 token 使用统计：${error}` });
      const rangeLabel = ranges.find(([value]) => value === range)?.[1] ?? "全部";
      const summary = report.summary ?? {};
      const total = report.rows.reduce((sum, row) => sum + row.totalTokens, 0) || 1;
      const cards = [
        { label: "tokens 用量", value: usageBig(summary.totalTokens ?? report.totals.totalTokens), inline: false },
        { label: "总计费用", value: (report?.rows ?? []).some(row => row.cost !== undefined) ? fmtMoney(summary.cost ?? 0) : "—", inline: false },
        { label: "会话数量", value: String(summary.sessionCount ?? 0), inline: false },
        { label: "消息数量", value: String(summary.messageCount ?? 0), inline: false },
        { label: "活跃天数", value: String(summary.activeDays ?? 0), inline: false },
        { label: "当前连续天数", value: String(summary.currentStreakDays ?? 0), inline: false },
        { label: "最常用模型", value: summary.topModel === void 0 || summary.topModel === null ? "—" : `${summary.topModel} · ${((summary.topModelShare ?? 0) * 100).toFixed(1)}%`, inline: true }
      ];
      return jsxRuntime.jsx("div", {
        className: C.root,
        children: [
          jsxRuntime.jsx("div", {
            className: C.filters,
            children: ranges.map(([value, label]) => jsxRuntime.jsx("button", {
              type: "button",
              className: range === value ? `${C.filter} ${C.filterActive}` : C.filter,
              onClick: () => setRange(value),
              children: label
            }, value))
          }),
          jsxRuntime.jsxs("div", {
            className: C.top,
            children: [
              jsxRuntime.jsx("div", { className: C.meta, children: `更新于 ${new Date(report.generatedAt).toLocaleTimeString()} · ${rangeLabel} · 扫描 ${report.files} 个会话，跳过 ${report.skipped} 个` }),
              jsxRuntime.jsx("button", { type: "button", className: C.btn, onClick: load, children: "刷新" })
            ]
          }),
          jsxRuntime.jsx("div", {
            className: C.cards,
            children: cards.map((item) => item.inline ? jsxRuntime.jsxs("div", {
              className: `${C.card} ${C.cardInline}`,
              children: [jsxRuntime.jsx("div", { className: C.labelInline, children: item.label }), jsxRuntime.jsx("div", { className: C.valueInline, children: item.value })]
            }, item.label) : jsxRuntime.jsxs("div", {
              className: C.card,
              children: [jsxRuntime.jsx("div", { className: C.value, children: item.value }), jsxRuntime.jsx("div", { className: C.label, children: item.label })]
            }, item.label))
          }),
          jsxRuntime.jsx("div", {
            className: C.charts,
            children: [
              jsxRuntime.jsxs("div", {
                className: C.panel,
                children: [
                  jsxRuntime.jsx("h3", { className: C.chartTitle, children: "按天 Token 趋势" }),
                  jsxRuntime.jsx("div", { className: C.barBody, dangerouslySetInnerHTML: { __html: usageBarHtml(report) } })
                ]
              }),
              jsxRuntime.jsxs("div", {
                className: C.panel,
                children: [
                  jsxRuntime.jsx("h3", { className: C.chartTitle, children: "模型用量占比" }),
                  jsxRuntime.jsxs("div", {
                    className: C.chartBody,
                    style: { display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" },
                    children: [
                      jsxRuntime.jsx("div", { dangerouslySetInnerHTML: { __html: usagePieSvg(report) } }),
                      jsxRuntime.jsx("div", {
                        className: C.legend,
                        children: report.rows.filter((row) => row.totalTokens > 0).map((row, index) => jsxRuntime.jsxs("div", {
                          className: C.legendRow,
                          children: [
                            jsxRuntime.jsxs("span", { children: [jsxRuntime.jsx("span", { style: { display: "inline-block", width: "8px", height: "8px", borderRadius: "50%", marginRight: "6px", background: usageColor(row.model, index) } }), row.model] }),
                            jsxRuntime.jsx("span", { children: `${usageBig(row.totalTokens)} · ${(row.totalTokens / total * 100).toFixed(1)}%` })
                          ]
                        }, row.model))
                      })
                    ]
                  })
                ]
              })
            ]
          }),
          jsxRuntime.jsxs("table", {
            className: C.table,
            children: [
              jsxRuntime.jsx("thead", {
                children: jsxRuntime.jsx("tr", {
                  children: ["模型", "会话", "请求", "输入", "读缓存", "输出", "推理", "总量 / 费用"].map((label) => jsxRuntime.jsx("th", { className: C.th, children: label }, label))
                })
              }),
              jsxRuntime.jsx("tbody", {
                children: report.rows.map((row) => jsxRuntime.jsxs("tr", {
                  className: C.row,
                  children: [row.model, row.sessions, row.requests, fmtUsage(row.uncachedInputTokens), fmtUsage(row.cacheReadTokens), fmtUsage(row.outputTokens), fmtUsage(row.reasoningTokens), row.cost === void 0 ? usageBig(row.totalTokens) : jsxRuntime.jsxs("div", { style: { lineHeight: "16px" }, children: [usageBig(row.totalTokens), jsxRuntime.jsx("br", {}), fmtMoney(row.cost)] })].map((value) => jsxRuntime.jsx("td", { className: C.td, children: value }, `${row.model}:${value}`))
                }, `${row.provider}:${row.model}`))
              })
            ]
          })
        ]
      });
    }

    // ---- plugin registration ----
    const inject = ["slots", "locale", "connection"];

    function apply(ctx) {
      const NS = "token-usage";
      const zh = { "section.nav": "使用统计" };
      const en = { "section.nav": "Token Usage" };
      ctx.effect(() => ctx.locale.register(NS, { zh, en }), "ui-token-usage: dictionaries");
      const t = ctx.locale.bind(NS);
      ctx.slots.inject("settings.section", () =>
        ctx.slots.register(
          {
            name: "settings.section",
            id: "token-usage",
            order: 10,
            label: () => t("section.nav"),
            locale: NS,
          },
          () => jsxRuntime.jsx(UsageSection, {})
        )
      );
    }

    exports.apply = apply;
    exports.inject = inject;
    return module.exports;
  },
});
