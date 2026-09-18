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
    if (r.ok && !(await r.json()).authenticated)
      location.replace("/?returnTo=%2Fdsh%2F");
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
    function apply(ctx) {
      for (const [name, Component] of [
        ["sidebar.brand.mark", Mark],
        ["sidebar.brand.name", Name],
        ["conversation.hero.brand.mark", Mark],
      ])
        ctx.slots.inject(name, () => ctx.slots.register({ name }, Component));
    }
    return { apply, inject: ["slots"] };
  },
});
