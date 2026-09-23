// DSH's native authentication must not compete with the management session.
let checkingSession = false;
async function checkWebSession() {
  if (checkingSession) return;
  checkingSession = true;
  try {
    const r = await fetch("/console-api/session", {
      cache: "no-store",
      signal: AbortSignal.timeout(10000),
    });
    if (r.ok && !(await r.json()).authenticated) {
      if (window.parent !== window)
        window.parent.postMessage({type: "sm75-session-expired"}, location.origin);
      else location.replace("/");
    }
  } catch {
    /* A temporary transport failure is not a logout. */
  } finally {
    checkingSession = false;
  }
}
const sessionEvents = new EventSource("/console-api/session/stream");
sessionEvents.onerror = checkWebSession;
sessionEvents.onmessage = (e) => {
  try {
    if (!JSON.parse(e.data).authenticated) void checkWebSession();
  } catch {}
};
window.addEventListener("pageshow", checkWebSession);
window.addEventListener("focus", checkWebSession);
import("/app.js");
window.__ModuleLoader__.load({
  id: "@deepseek-ai/dsh-client-ui-brand-official",
  factory: (require) => {
    const React = require("react");
    function Mark({ size = 28, className }) {
      return React.createElement("img", {
        src: "/brand/favicon.svg",
        width: size,
        height: size,
        className,
        alt: "工作台",
        style: { display: "block", flexShrink: 0 },
      });
    }
    function Name() {
      return React.createElement(
        "strong",
        { style: { fontSize: 15, whiteSpace: "nowrap" } },
        "工作台",
      );
    }
    function SettingsLauncher({openSettings}) {
      React.useEffect(() => {
        if (window.parent === window) return;
        const receive = event => {
          if (event.origin !== location.origin || event.source !== window.parent) return;
          if (event.data?.type === "sm75-harness-action" && event.data.action === "settings") openSettings();
        };
        window.addEventListener("message", receive);
        window.parent.postMessage({type: "sm75-dsh-ready"}, location.origin);
        return () => window.removeEventListener("message", receive);
      }, [openSettings]);
      return window.parent === window
        ? React.createElement("button", {onClick: openSettings}, "设置")
        : null;
    }
    function apply(ctx) {
      ctx.slots.inject("settings.launcher", () =>
        ctx.slots.register({name: "settings.launcher"}, SettingsLauncher));
      ctx.effect(() => {
        if (window.parent === window) return;
        const receive = event => {
          if (event.origin !== location.origin || event.source !== window.parent) return;
          if (event.data?.type === "sm75-harness-action" && event.data.action === "usage") {
            document.dispatchEvent(new KeyboardEvent("keydown", {key: "Escape"}));
            ctx.layout.selectPanel("sm75-usage");
          }
          if (event.data?.type === "sm75-theme" && ["dark", "light"].includes(event.data.mode)
              && ctx.theme.getTheme().active.colorScheme !== event.data.mode)
            ctx.theme.setTheme(event.data.mode);
        };
        const off = ctx.on("theme/change", snapshot => window.parent.postMessage({
          type: "sm75-dsh-preferences", mode: snapshot.active.colorScheme, fontSize: snapshot.fontSize,
        }, location.origin));
        window.addEventListener("message", receive);
        return () => {off(); window.removeEventListener("message", receive);};
      }, "sm75: embedded settings, Watcher and theme bridge");
      for (const [name, Component] of [
        ["sidebar.brand.mark", Mark],
        ["sidebar.brand.name", Name],
        ["conversation.hero.brand.mark", Mark],
      ])
        ctx.slots.inject(name, () => ctx.slots.register({ name }, Component));
    }
    return { apply, inject: ["slots", "layout", "theme"] };
  },
});
