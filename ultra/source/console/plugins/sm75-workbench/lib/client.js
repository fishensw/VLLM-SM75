window.__ModuleLoader__.load({
  id: "sm75-workbench",
  factory: (require) => {
    const React = require("react");
    // Quick chat owns its stream for the browser session, not for a navigation mount.
    const retained = new Map();
    const park = (host) => {
      host.style.cssText =
        "position:fixed;left:-20000px;top:0;width:1024px;height:768px;pointer-events:none";
      host.inert = true;
      host.setAttribute("aria-hidden", "true");
      globalThis.document.body.append(host);
    };
    function Panel({ page, renderSlot, sampling = false, headerOnly = false }) {
      const [usage, setUsage] = React.useState(false);
      const ref = React.useRef(null);
      React.useEffect(() => {
        let disposed = false,
          cleanup;
        const container = ref.current;
        const cacheKey = page + ":" + sampling;
        const keep = ["chat", "tests", "profiles", "settings"].includes(page);
        if (retained.has(cacheKey)) {
          const host = retained.get(cacheKey);
          host.style.cssText = "height:100%";
          host.inert = false;
          host.removeAttribute("aria-hidden");
          container.append(host);
          return () => park(host);
        }
        const host = globalThis.document.createElement("div");
        host.style.height = "100%";
        container.append(host);
        if (keep) retained.set(cacheKey, host);
        const root = host.attachShadow({ mode: "open" });
        (async () => {
          const [html, css, app] = await Promise.all([
            fetch("/console-ui").then((r) => r.text()),
            fetch("/style.css").then((r) => r.text()),
            import("/app.js"),
          ]);
          if (disposed) return;
          const doc = new DOMParser().parseFromString(html, "text/html"),
            style = document.createElement("style");
          style.textContent =
            css +
            "\n:host{--bg:var(--dsw-alias-bg-base,#151515);--card:var(--dsw-alias-bg-layer-2,#202020);--text:var(--dsw-alias-label-primary,#eee);--muted:var(--dsw-alias-label-secondary,#aaa);--line:var(--dsw-alias-border-l3,#393939);--blue:var(--dsw-alias-link,#70bbff);--green:var(--dsw-alias-state-success-primary,#62e0b6);--warn:var(--dsw-alias-state-warn-primary,#f5d35c);--error:var(--dsw-alias-state-error-primary,#ff7373);--gpu-low:var(--dsw-alias-state-success-primary,#58e0bc);--gpu-medium:var(--dsw-alias-state-warn-primary,#f5d35c);--gpu-elevated:color-mix(in srgb,var(--dsw-alias-state-warn-primary,#f5d35c) 60%,var(--dsw-alias-state-error-primary,#ff7373));--gpu-high:var(--dsw-alias-state-error-primary,#ff7373);--field:var(--dsw-specific-selector,#1b293c);--active:var(--dsw-alias-interactive-bg-active,#20354d);--composer-line:color-mix(in srgb,var(--text) 16%,transparent);display:block;width:100%;height:100%;min-height:0;color:var(--text);font-family:var(--dsw-font-family,system-ui);font-size:calc(14px * var(--ui-fs,1));line-height:1.6}:host-context(body[data-ds-dark-theme]){--field:#1b293c;--active:#20354d;--blue:#70bbff;--green:#62e0b6;--gpu-low:#58e0bc;--gpu-medium:#f5d35c;--gpu-elevated:#ffab55;--gpu-high:#ff7373;--composer-line:#3a536d;}aside{display:none!important}main{margin:0!important;width:100%!important;height:100%!important;max-width:none!important;padding:18px!important}";
          if (page === "settings")
            style.textContent +=
              "header{display:none!important}main{height:auto!important;padding:0!important}#settings{overflow:visible!important}";
          if (page === "settings")
            style.textContent += sampling
              ? "#deploymentSettings{display:none!important}"
              : "#modelSamplingSettings{display:none!important}";
          if (headerOnly)
            style.textContent += "main{height:auto!important;padding-bottom:0!important}:host{height:auto!important}";
          root.append(style);
          root.append(
            ...Array.from(doc.body.children).filter(
              (n) => n.tagName !== "SCRIPT",
            ),
          );
          const scoped = {
            getElementById: (id) => root.getElementById(id),
            querySelector: (s) => root.querySelector(s),
            querySelectorAll: (s) => root.querySelectorAll(s),
            createElement: document.createElement.bind(document),
            createElementNS: document.createElementNS.bind(document),
          };
          cleanup = app.mountConsole(scoped, {
            embedded: true,
            headerOnly,
            initialPage: page,
            isActive: () => host.isConnected && !host.inert,
            onUsage: () => setUsage(true),
          });
        })().catch((e) => {
          if (!disposed) root.textContent = "页面加载失败：" + e.message;
        });
        return () => {
          if (keep) {
            park(host);
            return;
          }
          disposed = true;
          cleanup?.();
        };
      }, [page, sampling, headerOnly]);
      return React.createElement(
        "div",
        { style: { height: headerOnly ? "auto" : "100%", width: "100%", minWidth: 0, minHeight: 0 } },
        React.createElement("div", {
          ref,
          hidden: usage,
          style: { height: headerOnly ? "auto" : "100%", display: usage ? "none" : "block" },
        }),
        usage &&
          React.createElement(
            "div",
            { style: { height: "100%", overflow: "auto", padding: 20 } },
            React.createElement(
              "button",
              { onClick: () => setUsage(false) },
              "← 性能监控",
            ),
            renderSlot("sm75.usage", {}),
          ),
      );
    }
    function apply(ctx) {
      ctx.slots.inject("sm75.statusbar", () =>
        ctx.slots.register(
          { name: "sm75.statusbar", id: "sm75-live-status" },
          (props) => React.createElement(Panel, { ...props, page: "harness", headerOnly: true }),
        ),
      );
      globalThis.__sm75ThemeApi = {
        toggle() {
          const s = ctx.theme.getTheme();
          ctx.theme.setTheme(
            s.active.colorScheme === "dark" ? "light" : "dark",
          );
        },
      };
      const pages = [
        ["chat", "快速会话", "◌"],
        ["conversation", "工作台", "▣"],
        ["models", "模型库", "▦"],
        ["profiles", "运行配置", "⚙"],
        ["monitor", "性能监控", "⌁"],
        ["tests", "模型测试", "◫"],
      ];
      for (const [page, label, icon] of pages) {
        const id = page === "conversation" ? "conversation" : "sm75-" + page;
        ctx.slots.inject("sidebar.panellist", () =>
          ctx.slots.register(
            {
              name: "sidebar.panellist",
              id,
              order: pages.findIndex((p) => p[0] === page),
              label,
            },
            () => React.createElement("span", null, icon),
          ),
        );
        if (page !== "conversation")
          ctx.slots.inject("main", () =>
            ctx.slots.register({ name: "main", key: id }, (props) =>
              React.createElement(Panel, { ...props, page }),
            ),
          );
      }
      ctx.slots.inject("settings.section", () =>
        ctx.slots.register(
          {
            name: "settings.section",
            id: "sm75-sampling",
            order: -19,
            label: "模型参数与模板",
          },
          (props) =>
            React.createElement(Panel, {
              ...props,
              page: "settings",
              sampling: true,
            }),
        ),
      );
      ctx.slots.inject("settings.section", () =>
        ctx.slots.register(
          {
            name: "settings.section",
            id: "sm75-system",
            order: -20,
            label: "部署与硬件",
          },
          (props) => React.createElement(Panel, { ...props, page: "settings" }),
        ),
      );
    }

    return { apply, inject: ["slots", "layout", "theme"] };
  },
});
