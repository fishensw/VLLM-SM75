// Shared workbench palette, including embedded console and monitoring frames.
const SM75_UI_BUILD = "20260914-5";
// Asset cache validation is handled by HTTP; never reload a live conversation for a build marker.
function installWorkbenchTheme() {
  if (globalThis.__sm75ThemeInstalled) return;
  globalThis.__sm75ThemeInstalled = true;
  const doc = globalThis.document,
    key = "sm75-color-mode",
    embedded = !!globalThis.__ModuleLoader__;
  let mode = embedded
    ? doc.body?.hasAttribute("data-ds-dark-theme")
      ? "dark"
      : "light"
    : localStorage.getItem(key) || "dark";
  const palettes = {
    dark: {
      bg: "#151517",
      card: "#202020",
      line: "rgba(255,255,255,.16)",
      ink: "#f9fafb",
      muted: "#a6adb8",
      blue: "#679efe",
      onBlue: "#101820",
      mint: "#4ed17e",
      field: "#353638",
      active: "#20354d",
      switchOff: "#697483",
      switchOn: "#167d5e",
      track: "#253243",
      line2: "#33475e",
      line3: "#58718c",
      field2: "#182434",
      field3: "#202e40",
      chipErrorBg: "#442829",
      chipWaitBg: "#3c3422",
      chipOkBg: "#183a32",
      errorText: "#ffa4a4",
      focus: "#3b6fd4",
      shadow2: "rgba(0,0,0,.42)",
      amber: "#ffcb76",
    },
    light: {
      bg: "#f4f6f9",
      card: "#ffffff",
      line: "rgba(0,0,0,.16)",
      ink: "#0b0d0f",
      muted: "#3f4145",
      blue: "#2b4a92",
      onBlue: "#ffffff",
      mint: "#0e5229",
      field: "#f5f6f7",
      active: "#e7efff",
      switchOff: "#7b8796",
      switchOn: "#167d5e",
      track: "#d9dfe7",
      line2: "rgba(0,0,0,.14)",
      line3: "rgba(0,0,0,.20)",
      field2: "#f2f5f8",
      field3: "#eaeff5",
      chipErrorBg: "#fdecec",
      chipWaitBg: "#fbf1dc",
      chipOkBg: "#e7f6ee",
      errorText: "#b01515",
      focus: "#2b4a92",
      shadow2: "rgba(0,0,0,.12)",
      amber: "#6f3004",
    },
  };
  function forceTheme(dark) {
    if (dark) doc.body.setAttribute("data-ds-dark-theme", "");
    else doc.body.removeAttribute("data-ds-dark-theme");
  }
  function sm75ToggleTheme() {
    const wasDark = doc.body.hasAttribute("data-ds-dark-theme");
    if (embedded && globalThis.__sm75ThemeApi) {
      globalThis.__sm75ThemeApi.toggle();
      setTimeout(() => {
        if (doc.body.hasAttribute("data-ds-dark-theme") === wasDark)
          forceTheme(!wasDark);
      }, 400);
      return;
    }
    forceTheme(!wasDark);
  }
  globalThis.__sm75ToggleTheme = sm75ToggleTheme;
  const button = doc.createElement("button");
  button.id = "sm75ThemeToggle";
  button.type = "button";
  button.hidden = true;
  doc.body.append(button);
  let placeMiss = 0;
  function placeThemeButton() {
    if (doc.getElementById("consoleSidebar")) { button.hidden = true; return; }
    const all = Array.from(doc.querySelectorAll("button[aria-label]"));
    let toggle = all.find((b) =>
      /侧边栏|sidebar|collapse|expand|收起|展开/i.test(
        b.getAttribute("aria-label") || "",
      ),
    );
    if (!toggle) {
      const img = doc.querySelector('img[src="/favicon.svg"]');
      const row = img && img.closest("div");
      const btns = row ? Array.from(row.querySelectorAll("button")) : [];
      toggle = btns.length ? btns[btns.length - 1] : null;
    }
    if (!toggle) {
      placeMiss++;
      if (placeMiss > 6 && !button.isConnected) {
        button.style.left = "170px";
        button.style.top = "12px";
        doc.body.append(button);
        button.hidden = false;
      }
      return;
    }
    placeMiss = 0;
    const r = toggle.getBoundingClientRect();
    button.style.position = "fixed";
    button.style.left = Math.max(8, Math.round(r.left - 34)) + "px";
    button.style.top = Math.round(r.top + (r.height - 28) / 2) + "px";
    button.style.zIndex = "60";
    button.style.background = "transparent";
    if (button.parentElement !== doc.body) doc.body.append(button);
    button.className = Array.from(toggle.classList)
      .filter((c) => c.includes("iconButton"))
      .join(" ");
    button.hidden = false;
  }
  function css(isolatedDocument = false) {
    const p = palettes[mode] || palettes.dark;
    const vars = {
      bg: p.bg,
      card: p.card,
      line: p.line,
      ink: p.ink,
      text: p.ink,
      muted: p.muted,
      blue: p.blue,
      mint: p.mint,
      green: p.mint,
      field: p.field,
      active: p.active,
      "switch-off": p.switchOff,
      "switch-on": p.switchOn,
      primary: p.blue,
      "on-primary": p.onBlue,
      info: p.blue,
      "text-color": p.ink,
      "text-muted": p.muted,
      "bg-color": p.bg,
      "border-color": p.line,
      "card-bg": p.card,
      "gpu-low": mode === "light" ? "#087f69" : "#58e0bc",
      "gpu-medium": mode === "light" ? "#8a6500" : "#f5d35c",
      "gpu-elevated": mode === "light" ? "#b45309" : "#ffab55",
      "gpu-high": mode === "light" ? "#c62828" : "#ff7373",
    };
    Object.assign(vars, {
      track: p.track,
      line2: p.line2,
      line3: p.line3,
      field2: p.field2,
      field3: p.field3,
      "chip-error-bg": p.chipErrorBg,
      "chip-wait-bg": p.chipWaitBg,
      "chip-ok-bg": p.chipOkBg,
      "error-text": p.errorText,
      focus: p.focus,
      shadow2: p.shadow2,
      amber: p.amber,
    });
    return (
      (isolatedDocument ? ":root,:host,.dark,.light,*{color-scheme:" : ":root,:host{color-scheme:") +
      mode +
      ";" +
      Object.entries(vars)
        .map(([k, v]) => "--" + k + ":" + v + "!important")
        .join(";") +
      "}" +
      (isolatedDocument
        ? "select,input,textarea{color:var(--ink)!important}"
        : "@scope (body) to (.dsh-surface, body > :not(aside):not(main):not(script):not(style)) {") +
      "body{background:var(--bg)!important;color:var(--ink)!important}:host{color:var(--ink)!important}:host button,main button,input,textarea,select,#sm75ThemeToggle{background:var(--field)!important;border-color:var(--line)!important}.card,.profile-card,.hardware-card,.chat-parameter-panel,.message{background:var(--card)!important;color:var(--ink)!important;border-color:var(--line)!important}.message.user{background:var(--field)!important}.chart-tip{background:var(--card)!important;color:var(--ink)!important}body[data-sm75-benchmark]{background:var(--bg)!important;color:var(--ink)!important}body[data-sm75-benchmark] :is(.container,.card,.panel,.config-section,.chart-container,.tab-content,.section,.config-card,#userCodeWidget,table){background:var(--card)!important;color:var(--ink)!important;border-color:var(--line)!important}body[data-sm75-benchmark] :is(h1,h2,h3,h4,label,p,summary){color:var(--ink)!important}body[data-sm75-benchmark] #userCodeWidget *{color:var(--ink)!important}body[data-sm75-benchmark] input,body[data-sm75-benchmark] select{background:var(--field)!important}#sm75ThemeToggle{background:transparent!important;border:0;display:inline-flex;align-items:center;justify-content:center;padding:0;flex-shrink:0}button.primary{background:var(--primary)!important;color:var(--on-primary)!important;border-color:var(--primary)!important}input[role=switch]{background:var(--switch-off)!important}input[role=switch]:checked{background:var(--switch-on)!important}input[role=switch]:before{background:#fff!important}#sm75ThemeToggle[hidden]{display:none!important}#sm75ThemeToggle:hover{background:var(--field)!important}[class*=collapsed] [class*=logoRow]:has(#sm75ThemeToggle){height:auto;flex-wrap:wrap;overflow:visible}" +
      (isolatedDocument ? `
        /* These rules belong only to the existing, same-origin test documents.
           Native DSH buttons in the shared page keep their own component theme. */
        body[data-sm75-benchmark] :is(.api-description,.history-content,.history-modal-content,.modal-content,.batch-upload-content,.result-section,.comparison-section,.notes-panel,.detail-dialog,.detail-section,.history-record){
          background:var(--field2)!important;color:var(--ink)!important;border-color:var(--line)!important
        }
        body[data-sm75-benchmark] :is(.detail-field-value,.detail-content-block,.detail-concurrent-item){
          background:var(--card)!important;color:var(--ink)!important;border-color:var(--line)!important
        }
        body[data-sm75-benchmark] :is(.history-modal-header,.detail-header){
          background:var(--field3)!important;color:var(--ink)!important
        }
        body[data-sm75-benchmark] :is(.history-modal-close,.history-record-time,.history-record-info,.detail-section-title,.detail-field-label,.detail-block-label,.detail-subsection-title,.detail-concurrent-item-title){
          color:var(--ink)!important
        }
        body[data-sm75-benchmark] :is(.api-option,.btn-secondary,.history-btn-secondary,.history-btn-back){
          background:var(--field)!important;color:var(--ink)!important;border-color:var(--line)!important
        }
        body[data-sm75-benchmark] .api-option.active{
          background:var(--primary)!important;color:var(--on-primary)!important;border-color:var(--primary)!important
        }
        body[data-sm75-benchmark] :is(.api-option:hover:not(.active),.btn-secondary:hover:not(:disabled)){
          background:var(--field3)!important
        }
        body[data-sm75-benchmark] :is(.api-description h3,a){
          color:var(--blue)!important
        }
        body[data-sm75-benchmark] :is(.notes,.model-suggest-status,#selectedCount){
          color:var(--muted)!important
        }
        body[data-sm75-benchmark] :is(.container > div[style*="background: #fffbf0"],[data-sm75-benchmark-notice]){
          background:var(--chip-wait-bg)!important;color:var(--amber)!important;border-color:var(--amber)!important
        }
        body[data-sm75-benchmark] textarea{
          background:var(--field)!important;color:var(--ink)!important
        }
      ` : "}")
    );
  }

  // Child documents do not inherit the shared shell's font multiplier. Keep
  // their existing hierarchy, and scale only their own absolute font sizes.
  const frameFonts = new WeakMap();
  const chartFonts = new WeakMap();
  const chartFontPaths = new WeakMap();
  const chartThemes = new WeakMap();
  let fontScale = 1;
  function readFontScale() {
    const value = Number.parseFloat(getComputedStyle(doc.documentElement).getPropertyValue("--ui-fs"));
    return Number.isFinite(value) && value > 0 ? value : 1;
  }
  function scalableFont(value) {
    if (!value || value.includes("--ui-fs")) return value;
    return value.replace(/(-?(?:\d+\.?\d*|\.\d+))(px|pt|pc|in|cm|mm|q|vw|vh|vmin|vmax)\b/gi,
      (_, number, unit) => "calc(" + number + unit + " * var(--ui-fs, 1))");
  }
  function scaleDeclaration(style) {
    const original = style.getPropertyValue("font-size");
    const value = scalableFont(original);
    if (value !== original) style.setProperty("font-size", value, style.getPropertyPriority("font-size"));
  }
  function scaleRules(rules) {
    for (const rule of rules || []) {
      if (rule.style) scaleDeclaration(rule.style);
      if (rule.cssRules) scaleRules(rule.cssRules);
    }
  }
  function scaleChartFontOptions(value, visited = new WeakSet(), paths = null, parentPath = "") {
    if (!value || typeof value !== "object" || visited.has(value)) return;
    visited.add(value);
    for (const [key, entry] of Object.entries(value)) {
      if (!entry || typeof entry !== "object") continue;
      const path = parentPath + "/" + key;
      if (/font$/i.test(key) && Number.isFinite(entry.size)) {
        // Chart.js rebuilds scale option objects during update. Their option
        // paths remain stable, so the original size must survive those copies.
        const saved = paths ? paths.get(path) : chartFonts.get(entry);
        const base = saved && entry.size === saved.applied ? saved.base : entry.size;
        const applied = base * fontScale;
        entry.size = applied;
        if (paths) paths.set(path, { base, applied });
        else chartFonts.set(entry, { base, applied });
      }
      scaleChartFontOptions(entry, visited, paths, path);
    }
  }
  function syncFrameCharts(win) {
    const C = win.Chart;
    if (!C?.defaults) return;
    C.defaults.color = palettes[mode].muted;
    C.defaults.borderColor = palettes[mode].line;
    scaleChartFontOptions(C.defaults);
    for (const c of Object.values(C.instances || {})) {
      const previous = chartThemes.get(c);
      if (previous?.mode === mode && previous?.scale === fontScale) continue;
      let paths = chartFontPaths.get(c);
      if (!paths) { paths = new Map(); chartFontPaths.set(c, paths); }
      scaleChartFontOptions(c.config?.options || c.options, new WeakSet(), paths);
      for (const axis of Object.values(c.options?.scales || {})) {
        if (axis.ticks) axis.ticks.color = palettes[mode].muted;
        if (axis.grid) axis.grid.color = palettes[mode].line;
      }
      if (c.options?.plugins?.legend?.labels) c.options.plugins.legend.labels.color = palettes[mode].ink;
      chartThemes.set(c, { mode, scale: fontScale });
      c.update("none");
    }
  }
  function syncFrameFonts(frame) {
    const child = frame.contentDocument, win = frame.contentWindow;
    if (!child?.documentElement || !win) return;
    let state = frameFonts.get(child);
    if (!state) {
      state = { rootSize: win.getComputedStyle(child.documentElement).fontSize, scale: null, scheduled: false };
      frameFonts.set(child, state);
      state.style = child.createElement("style");
      state.style.dataset.sm75FontScale = "1";
      (child.head || child.documentElement).append(state.style);
      // The monitor draws its own canvas text; test tools use Chart.js options.
      if (frame.id === "monitorFrame") {
        const proto = win.CanvasRenderingContext2D?.prototype;
        const descriptor = proto && Object.getOwnPropertyDescriptor(proto, "font");
        if (descriptor?.set && descriptor.configurable) {
          Object.defineProperty(proto, "font", { ...descriptor, set(value) {
            const scaled = String(value).replace(/(\d+(?:\.\d+)?)px\b/g, (_, px) => String(Number(px) * fontScale) + "px");
            descriptor.set.call(this, scaled);
          } });
        }
      }
      const schedule = () => {
        if (state.scheduled) return;
        state.scheduled = true;
        win.setTimeout(() => {
          state.scheduled = false;
          if (frame.contentDocument === child) { syncFrameFonts(frame); syncFrameCharts(win); }
        }, 30);
      };
      new win.MutationObserver(records => {
        if (records.some(record =>
          (record.type === "attributes" && !record.target.hasAttribute("data-sm75-font-scale")
            && !record.target.style?.getPropertyValue("font-size").includes("--ui-fs"))
          || (record.type === "childList" && !record.target.matches?.("style[data-sm75-font-scale],style[data-sm75-theme]")
            && [...record.addedNodes].some(node => node.nodeType === 1 || record.target.tagName === "STYLE"))))
          schedule();
      }).observe(child.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ["style"] });
      child.addEventListener("load", event => { if (event.target.tagName === "LINK") schedule(); }, true);
    }
    for (const sheet of child.styleSheets) {
      if (sheet.ownerNode?.matches?.("style[data-sm75-font-scale],style[data-sm75-theme]")) continue;
      try { scaleRules(sheet.cssRules); } catch {}
    }
    // CSSOM serializes inline colors when font-size is edited. Keep the
    // existing benchmark notice theme independent of that serialization.
    for (const element of child.querySelectorAll(".container > div[style]")) {
      if (element.style.backgroundColor === "rgb(255, 251, 240)") element.dataset.sm75BenchmarkNotice = "1";
    }
    for (const element of child.querySelectorAll("[style]")) scaleDeclaration(element.style);
    const cssText = ":root{--ui-fs:" + fontScale + "!important;font-size:calc(" + state.rootSize + " * var(--ui-fs, 1))!important}";
    if (state.style.textContent !== cssText) state.style.textContent = cssText;
    if (state.scale !== fontScale) {
      state.scale = fontScale;
      win.dispatchEvent(new win.Event("resize"));
    }
  }

  const frames = new WeakSet();
  let lastCss = "", lastFrameCss = "";
  function scan(root) {
    const isFrameDoc = !!(root.defaultView && root.defaultView !== globalThis);
    const inject = !embedded || isFrameDoc;
    let style = inject ? root.querySelector("style[data-sm75-theme]") : null;
    if (inject) {
      if (!style) {
        style = doc.createElement("style");
        style.dataset.sm75Theme = "1";
        (root.head || root).append(style);
      }
      // Embedded test/monitor documents own their entire UI. The DSH boundary
      // only applies to the shared workbench document, not these child pages.
      const themeCss = isFrameDoc || (root.ownerDocument && root.ownerDocument !== doc)
        ? lastFrameCss : lastCss;
      if (style.textContent !== themeCss) style.textContent = themeCss;
    }
    for (const el of root.querySelectorAll("*")) {
      if (el.shadowRoot) scan(el.shadowRoot);
      if (
        el.tagName === "IFRAME" &&
        (el.id === "monitorFrame" || el.id?.startsWith("benchFrame-"))
      ) {
        if (!frames.has(el)) {
          frames.add(el);
          el.addEventListener("load", apply);
        }
        try {
          if (el.contentDocument) {
            if (el.id?.startsWith("benchFrame-")) {
              el.contentDocument.body?.setAttribute("data-sm75-benchmark", "");
              if (el.id === "benchFrame-speedtest") {
                const st = el.contentWindow.localStorage;
                if (!st.getItem("sm75-benchmark-defaults-20260913")) {
                  for (const [id, key, v] of [
                    ["minLength", "MinLength", 512],
                    ["maxLength", "MaxLength", 131072],
                    ["step", "Step", 128],
                    ["stepMultiplier", "StepMultiplier", 2],
                    ["outputLength", "OutputLength", 512],
                  ]) {
                    st.setItem("llmPerfTest" + key, String(v));
                    const input = el.contentDocument.getElementById(id);
                    if (input) input.value = v;
                  }
                  st.setItem("sm75-benchmark-defaults-20260913", "1");
                }
              }

            }
            syncFrameFonts(el);
            syncFrameCharts(el.contentWindow);
            scan(el.contentDocument);
            el.contentWindow.postMessage(
              { type: "sm75-theme", mode },
              location.origin,
            );
          }
        } catch {}
      }
    }
  }
  let announcedMode;
  function apply() {
    fontScale = readFontScale();
    if (!lastCss || announcedMode !== mode) {
      lastCss = css();
      lastFrameCss = css(true);
    }
    if (announcedMode !== mode) {
      announcedMode = mode;
      doc.dispatchEvent(new CustomEvent("sm75-theme-mode", { detail: mode }));
    }
    const icon = mode === "dark" ? "sun" : "moon";
    if (button.dataset.icon !== icon) {
      button.dataset.icon = icon;
      button.innerHTML =
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
        (icon === "sun"
          ? '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>'
          : '<path d="M20 14a8 8 0 0 1-10-10 8 8 0 1 0 10 10Z"/>') +
        "</svg>";
    }
    placeThemeButton();
    button.title = mode === "dark" ? "切换日间模式" : "切换夜间模式";
    button.setAttribute("aria-label", button.title);
    scan(doc);
  }
  button.onclick = () => {
    if (embedded) {
      sm75ToggleTheme();
      return;
    }
    mode = mode === "dark" ? "light" : "dark";
    localStorage.setItem(key, mode);
    apply();
  };
  doc.addEventListener("sm75-set-theme", (e) => {
    if (embedded || !["light", "dark"].includes(e.detail) || mode === e.detail) return;
    mode = e.detail;
    localStorage.setItem(key, mode);
    apply();
  });
  window.addEventListener("storage", (e) => {
    if (e.key === key) {
      mode = e.newValue || "dark";
      apply();
    }
  });
  doc.addEventListener("sm75-toggle-local", () => {
    if (embedded) return;
    mode = mode === "dark" ? "light" : "dark";
    localStorage.setItem(key, mode);
    apply();
  });
  if (embedded)
    new MutationObserver(() => {
      const m = doc.body.hasAttribute("data-ds-dark-theme") ? "dark" : "light";
      if (m !== mode) {
        mode = m;
        apply();
      }
    }).observe(doc.body, {
      attributes: true,
      attributeFilter: ["data-ds-dark-theme"],
    });
  new MutationObserver(() => {
    if (readFontScale() !== fontScale) apply();
  }).observe(doc.documentElement, { attributes: true, attributeFilter: ["style"] });
  let scheduled = false;
  new MutationObserver(() => {
    if (scheduled) return;
    scheduled = true;
    setTimeout(() => {
      scheduled = false;
      apply();
    }, 100);
  }).observe(doc.body, { childList: true, subtree: true });
  apply();
  // The shared shell owns its theme control. Only the legacy standalone view
  // needs to follow a movable native header button.
  if (!doc.getElementById("consoleSidebar")) setInterval(placeThemeButton, 600);
}



if (globalThis.document?.body) installWorkbenchTheme();

import { mountLiveSummary, metricLevel as gpuLevel } from "/live-summary.js";
import {readContextSettings, applyContextSettings, argValue as readArgument, setArg, oneMillionContext, retargetContextModel} from "./context-config.js";
import {validateLMCacheProfile, isLMCacheEnabled} from "./lmcache-config.js";

// The dedicated form edits the argv stored by profiles/templates. This binding
// can be exercised without mounting the rest of the console or starting an engine.
export function bindContextEditor(document, {getArgs, setArgs, getMetadata = () => ({}), onApply = () => {}, onDirty = () => {}, run = fn => fn}) {
  const $ = id => document.getElementById(id);
  const fields = {maxModelLen: "ctxMaxModelLen", yarnOriginal: "ctxYarnOriginal", yarnFactor: "ctxYarnFactor",
    gpuKvGiB: "ctxGpuKvGiB", cpuKvMode: "ctxCpuKvMode", cpuKvGiB: "ctxCpuKvGiB"};
  let dirty = false, editable = true, readError = null;
  function values() {
    const value = {...Object.fromEntries(Object.entries(fields).map(([name, id]) => [name, $(id).value])),
      yarnEnabled: $("ctxYarnEnabled").checked};
    const mode = $("cacheBackend")?.value;
    const current = readContextSettings(getArgs()).cpuKvMode;
    const backend = readArgument(getArgs(), "--kv-offloading-backend");
    // Hidden CPU fields are a retained draft, not an instruction to rewrite the
    // connector. LMCache's save validates the full candidate before replacing it.
    if ((mode && mode !== "default") || current === "custom" || (backend !== null && backend !== "native"))
      value.cpuKvMode = "custom";
    return value;
  }
  function visibility() {
    $("ctxYarnFields").hidden = !$("ctxYarnEnabled").checked;
    $("ctxCpuKvFields").hidden = $("ctxCpuKvMode").value !== "native";
  }
  function fill(state) {
    for (const [name, id] of Object.entries(fields)) $(id).value = state[name] ?? "";
    $("ctxYarnEnabled").checked = state.yarnEnabled;
    editable = state.yarnEditable !== false;
    for (const id of ["ctxYarnEnabled", "ctxYarnOriginal", "ctxYarnFactor", "ctxOneMillion"])
      $(id).disabled = !editable;
    const custom = $("ctxCpuKvMode").querySelector('option[value="custom"]');
    if (custom) custom.disabled = state.cpuKvMode !== "custom";
    visibility();
  }
  function refresh(force = false) {
    if (dirty && !force && !readError) return;
    dirty = false;
    readError = null;
    const metadata = getMetadata();
    try {
      const state = readContextSettings(getArgs(), metadata);
      fill(state);
      $("contextWarnings").textContent = state.warnings.join("\n");
      $("contextMetadata").textContent =
        "模型原生位置长度：" + (metadata.originalMaxPositionEmbeddings || "未知，可手动填写") +
        "；模型配置上下文：" + (metadata.maxModelLen || "未知") +
        (editable ? "。" : "。该模型使用分层 RoPE，请通过高级参数按模型结构配置。");
    } catch (error) {
      readError = error;
      $("contextWarnings").textContent = "无法读取上下文配置：" + error.message + "。请先检查高级参数。";
    }
  }
  function flush() {
    if (!dirty) return false;
    if (readError) throw readError;
    setArgs(applyContextSettings(getArgs(), values(), getMetadata()));
    refresh(true);
    return true;
  }
  function changed() {dirty = true; onDirty(); visibility();}
  $("contextSettings").addEventListener("input", changed);
  $("contextSettings").addEventListener("change", changed);
  $("nativeCacheFields")?.addEventListener("input", changed);
  $("nativeCacheFields")?.addEventListener("change", changed);
  $("cacheBackend")?.addEventListener("change", () => { if ($("cacheBackend").value === "default") changed(); });
  if ($("ctxApply")) $("ctxApply").onclick = run(() => {dirty = true; flush(); onApply();});
  $("ctxOneMillion").onclick = run(() => {
    if (!editable) throw Error("该模型使用分层 RoPE，请通过高级参数配置");
    fill({...oneMillionContext(values(), getMetadata()), yarnEditable: editable});
    changed();
    flush();
    onApply();
  });
  return {refresh, flush};
}

export function bindLMCacheEditor(document, {getProfile, setProfile, beforeApply = () => {}, onApply = () => {}, onDirty = () => {}, run = fn => fn}) {
  const $ = id => document.getElementById(id);
  const fields = {cpuGiB: "lmcacheCpuGiB", chunkSize: "lmcacheChunkSize", diskPath: "lmcacheDiskPath",
    minFreeDiskGiB: "lmcacheMinFreeDiskGiB", port: "lmcachePort", httpPort: "lmcacheHttpPort"};
  const defaults = {cpuGiB: 8, chunkSize: "", diskPath: "", minFreeDiskGiB: 4, port: 5555, httpPort: 5556};
  let dirty = false, selection = "default", nativeDraft = null;
  const nativeValues = () => ({mode: $("ctxCpuKvMode").value, gib: $("ctxCpuKvGiB").value});
  function restoreNative() {
    if (!nativeDraft) return;
    $("ctxCpuKvMode").value = nativeDraft.mode;
    $("ctxCpuKvGiB").value = nativeDraft.gib;
    $("ctxCpuKvFields").hidden = nativeDraft.mode !== "native";
  }
  function connectorMode() {
    const profile = getProfile(), backend = readArgument(profile.args, "--kv-offloading-backend");
    if (backend !== null && backend !== "native") return "custom";
    return readContextSettings(profile.args).cpuKvMode;
  }
  function values() {
    const value = {...getProfile().lmcache, enabled: $("cacheBackend").value === "lmcache"};
    for (const [name, id] of Object.entries(fields)) value[name] = name === "diskPath" ? $(id).value.trim()
      : $(id).value.trim() === "" ? null : Number($(id).value);
    return value;
  }
  function status() {
    const backend = $("cacheBackend").value, enabled = backend === "lmcache";
    $("nativeCacheFields").hidden = backend !== "default";
    $("lmcacheFields").hidden = !enabled;
    if ($("lmcacheEnabled")) $("lmcacheEnabled").checked = enabled;
    if ($("lmcacheReplaceNative")) $("lmcacheReplaceNative").hidden = true;
    let mode = "custom";
    try {mode = connectorMode();} catch {}
    const custom = $("cacheBackend").querySelector('option[value="custom"]');
    if (custom) custom.disabled = mode !== "custom";
    const notes = [];
    if (mode === "custom") notes.push("已有自定义缓存连接器，请在高级参数中修改；切换方案不会删除它。");
    if (enabled) {
      if (mode === "native") notes.push("保存时将用 LMCache 替换原生 CPU KV。");
      if (readArgument(getProfile().args, "--speculative-config") !== null) notes.push("请先关闭 DFlash/MTP 投机解码，再启用 LMCache。");
      if (getProfile().power?.mode === "sleep") notes.push("请先将电源模式改为 P-State；LMCache 暂不支持释放显存的休眠。");
      if (notes.length === 0) notes.push("保存后下次启动生效；依赖与缓存服务在启动时检查。");
    }
    $("lmcacheStatus").textContent = notes.join("\n");
  }
  function refresh(force = false) {
    if (dirty && !force) {
      if (selection !== "default") restoreNative();
      status();
      return;
    }
    const profile = getProfile(), config = profile.lmcache || {};
    let mode = "custom";
    try {mode = connectorMode();} catch {}
    selection = isLMCacheEnabled(profile) ? "lmcache" : mode === "custom" ? "custom" : "default";
    $("cacheBackend").value = selection;
    if (force || !nativeDraft || selection === "default") nativeDraft = nativeValues();
    for (const [name, id] of Object.entries(fields)) $(id).value = String(config[name] ?? defaults[name]);
    dirty = false;
    if (selection !== "default") restoreNative();
    status();
  }
  function flush() {
    if (!dirty) return false;
    beforeApply();
    const profile = getProfile(), selected = $("cacheBackend").value, mode = connectorMode();
    if (!["default", "lmcache", "custom"].includes(selected)) throw Error("缓存方案无效");
    if (mode === "custom" && selected !== "custom")
      throw Error("已有自定义缓存连接器，请先在高级参数中明确关闭；切换方案不会删除它");
    let args = [...profile.args];
    if (selected === "lmcache" && mode === "native")
      for (const key of ["--kv-transfer-config", "--kv-offloading-size", "--kv-offloading-backend"]) args = setArg(args, key, null);
    const next = {...profile, args, lmcache: values()};
    validateLMCacheProfile(next); // Validate conflicts and sizes before replacing native KV.
    setProfile(next);
    dirty = false;
    if (selection !== "default") restoreNative();
    status();
    return true;
  }
  function changed() {
    const next = $("cacheBackend").value;
    if (next !== selection) {
      if (selection === "default") nativeDraft = nativeValues();
      selection = next;
      if (selection === "default") restoreNative();
    }
    dirty = true;
    onDirty();
    status();
  }
  $("cacheBackend").addEventListener("change", changed);
  $("lmcacheSettings").addEventListener("input", changed);
  $("lmcacheSettings").addEventListener("change", changed);
  $("nativeCacheFields").addEventListener("input", () => {nativeDraft = nativeValues();});
  $("nativeCacheFields").addEventListener("change", () => {nativeDraft = nativeValues();});
  if ($("lmcacheApply")) $("lmcacheApply").onclick = run(() => {dirty = true; flush(); onApply();});
  if ($("lmcacheReplaceNative")) $("lmcacheReplaceNative").onclick = run(() => {
    $("cacheBackend").value = "lmcache";
    changed();
    flush();
    onApply();
  });
  return {refresh, flush};
}
// Profiles and personal templates must restore the same power form. Keep the
// target profile identity separate so loading a template does not rename it.
export function fillProfilePowerFields(document, profile) {
  const $ = id => document.getElementById(id);
  const power = {mode: "pstate", util: 5, confirm: 60, low: 8, high: 16, poll: 5, ...(profile.power || {})};
  const idleSeconds = Math.max(1, Number(power.idleSeconds ??
    (power.idleMinutes != null ? power.idleMinutes * 60 : 1)) || 1);
  $("powerMode").value = power.mode;
  for (const [id, value] of [
    ["pstateIdle", idleSeconds], ["pstateUtil", power.util], ["pstateConfirm", power.confirm],
    ["pstateLow", power.low], ["pstateHigh", power.high], ["pstatePoll", power.poll],
  ]) $(id).value = value;
  const sleepIdle = Number(readArgument(profile.args, "--auto-sleep-idle-timeout"));
  $("sleepIdle").value = sleepIdle > 0 ? sleepIdle : 30;
  $("sleepTarget").value = readArgument(profile.args, "--auto-sleep-offload-target") || "exit";
}

export function usesDedicatedCacheField(args, key) {
  if (["--max-model-len", "--kv-cache-memory-bytes"].includes(key)) return true;
  if (!["--kv-offloading-size", "--kv-offloading-backend"].includes(key)) return false;
  const backend = readArgument(args, "--kv-offloading-backend");
  // Unknown backends have no form of their own. Their advanced rows must stay
  // visible so the user can edit or remove the connector before switching.
  return backend === null || backend === "native";
}

export function mountConsole(document = globalThis.document, options = {}) {
  const $ = (id) => document.getElementById(id);
  const p8Btn = $("sm75P8");
  if (p8Btn) {
    p8Btn.onclick = async () => {
      p8Btn.disabled = true;
      try {
        await api("/console-api/power/p8", {});
        $("notice").textContent = "已请求进入 P8（数秒内生效）";
      } catch (e) {
        report(e);
      } finally {
        p8Btn.disabled = false;
      }
    };
  }
  let editorBase = null;
  let globalSettings = {},
    catalog = [],
    profileState = {},
    modelSignature = "",
    profilesSignature = "",
    featureBackup = {};
  let profiles = [],
    active = "",
    current =
      options.initialPage ||
      new URLSearchParams(location.search).get("page") ||
      "profiles",
    abort,
    chat = [];
  const routeTitles = {
    chat: "快速会话",
    harness: "工作台",
    models: "模型库",
    profiles: "运行配置",
    monitor: "性能监控",
    tests: "模型测试",
    settings: "设置",
    plugins: "插件",
    usage: "使用统计",
  };
  let editorDirty = false;
  let authState = "checking",
    authGeneration = 0,
    authEpoch = 0,
    authTask = null,
    disposed = false,
    refreshing = false;
  const stopLiveSummary = mountLiveSummary($("liveSummary"), {
    enabled: () =>
      !disposed &&
      authState === "authenticated" &&
      options.isActive?.() !== false,
  });
  const shell = document.querySelector("main");
  shell.classList.add("engine-toolbar");
  $("liveSummary").hidden = false;
  // Reserve metric text at the user's font size. Move auxiliary controls into
  // the existing sidebar only when their minimum widths no longer fit.
  // Reserve the same conversation-toolbar space on every page so navigation
  // cannot relocate the profile selector or add a row above the main links.
  const header = $("appHeader");
  const headerAux = [$("active"), $("signout"), $("taskToggle")].map(node => {
    const anchor = document.createComment("header-" + node.id);
    node.before(anchor);
    return { node, anchor };
  });
  let headerLayoutFrame = 0;
  const fitHeader = () => {
    headerLayoutFrame = 0;
    if (disposed || header.hidden || !header.clientWidth) return;
    const toolbar = $("workspaceToolbar");
    const scale = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--ui-fs")) || 1;
    const gap = Number.parseFloat(getComputedStyle(header).gap) || 0;
    const leading = Math.max(224, Number.parseFloat(getComputedStyle(toolbar).minWidth) || 0);
    const required = leading + 96 + $("liveSummary").getBoundingClientRect().width
      + 30 + 30 + (20 + 20.5 * scale) + gap * 6;
    const compact = required > header.clientWidth;
    if (header.dataset.auxInSidebar === String(compact)) return;
    header.dataset.auxInSidebar = String(compact);
    for (const {node, anchor} of headerAux) {
      if (compact) $(node.id === "active" ? "sidebarActiveProfile" : "sidebarHeaderActions").append(node);
      else anchor.after(node);
    }
  };
  const queueHeaderLayout = () => {
    if (!headerLayoutFrame) headerLayoutFrame = requestAnimationFrame(fitHeader);
  };
  const headerSizeObserver = new ResizeObserver(queueHeaderLayout);
  headerSizeObserver.observe(header);
  headerSizeObserver.observe($("liveSummary"));
  const headerContentObserver = new MutationObserver(queueHeaderLayout);
  headerContentObserver.observe(header, {childList:true, subtree:true, attributes:true, attributeFilter:["hidden"]});
  const fontScaleObserver = new MutationObserver(queueHeaderLayout);
  fontScaleObserver.observe(document.documentElement, {attributes:true, attributeFilter:["style"]});
  queueHeaderLayout();
  shell.classList.toggle("header-only", !!options.headerOnly);
  for (const type of ["input", "change"])
    $("profileEditor").addEventListener(type, () => {
      editorDirty = true;
    });
  let draftDecision = null;
  async function resolveDraft() {
    if (!editorDirty) return true;
    if (draftDecision) return draftDecision;
    draftDecision = new Promise((resolve) => {
      const dialog = document.createElement("dialog"),
        text = document.createElement("p"),
        actions = document.createElement("div");
      text.textContent = "运行配置有未保存修改";
      actions.className = "actions";
      dialog.append(text, actions);
      shell.append(dialog);
      const finish = (value) => {
        dialog.close();
        dialog.remove();
        resolve(value);
      };
      for (const [label, action] of [
        [
          "保存并继续",
          async () => {
            await $("save").onclick();
            if (!editorDirty) finish(true);
          },
        ],
        [
          "放弃修改",
          () => {
            editorDirty = false;
            fillProfile();
            finish(true);
          },
        ],
        ["取消", () => finish(false)],
      ]) {
        const button = document.createElement("button");
        button.textContent = label;
        button.onclick = async () => {
          button.disabled = true;
          try {
            await action();
          } finally {
            button.disabled = false;
          }
        };
        actions.append(button);
      }
      dialog.oncancel = (e) => {
        e.preventDefault();
        finish(false);
      };
      dialog.showModal();
    }).finally(() => {
      draftDecision = null;
    });
    return draftDecision;
  }
  const beforeUnload = (e) => {
    if (editorDirty) {
      e.preventDefault();
      e.returnValue = "";
    }
  };
  globalThis.addEventListener("beforeunload", beforeUnload);

  let accountConfigured = false;
  function syncLoginMode(configured) {
    accountConfigured = !!configured;
    $("loginUsername").hidden = !accountConfigured;
    $("loginUsernameLabel").hidden = !accountConfigured;
    $("loginSecretLabel").textContent = accountConfigured ? "密码" : "登录 token";
    $("token").placeholder = accountConfigured ? "输入管理员密码" : "输入首次启动生成的 token";
  }
  function setAuth(state, message = "") {
    const changed = authState !== state;
    if (changed) ++authEpoch;
    authState = state;
    shell.dataset.auth = state;
    if (state !== "authenticated") $("notice").textContent = "";
    const allowed = state === "authenticated";
    document.querySelector("aside").hidden = !allowed;
    $("appHeader").hidden = !allowed;
    $("login").hidden = allowed || state === "checking";
    $("authPending").hidden = state !== "checking";
    if (!allowed) {
      for (const id of ["adminCurrentPassword", "adminPassword", "adminPasswordConfirm"]) $(id).value = "";
      for (const section of shell.querySelectorAll(":scope>section"))
        if (section.id !== "login") section.hidden = true;
      $("taskDrawer").hidden = true;
      $("root").hidden = true;
      $("workspaceSidebar").hidden = true;
      if (globalThis.__SM75_WORKBENCH__) location.replace("/");
      abort?.abort();
    }
    $("signin").disabled = state === "checking" || state === "submitting";
    $("signin").textContent = state === "submitting" ? "正在登录…" : "登录";
    $("authStatus").textContent =
      message ||
      {
        checking: "正在确认登录状态…",
        anonymous: accountConfigured ? "使用管理员账号密码登录" : "首次使用 token 登录，然后在设置中创建管理员账号",
        authenticated: "",
        unavailable: "管理服务暂不可用，请重试",
      }[state] ||
      "";
    $("retryAuth").hidden = state !== "unavailable";
    $("token").disabled = state === "checking" || state === "submitting";
    $("loginUsername").disabled = $("token").disabled;
    if (changed) stopLiveSummary.refresh();
  }
  async function api(url, data) {
    const generation = authGeneration;
    const r = await fetch(url, {
      method: data ? "POST" : "GET",
      headers: data ? { "Content-Type": "application/json" } : {},
      body: data ? JSON.stringify(data) : undefined,
      signal: AbortSignal.timeout(30000),
    });
    const d = await r.json();
    if (!r.ok) {
      if (
        r.status === 401 &&
        generation === authGeneration &&
        authState === "authenticated" &&
        url !== "/console-api/login"
      )
        void checkSession();
      throw Object.assign(Error(d.error || r.status), { status: r.status });
    }
    return d;
  }
  const report = (e) => {
    if (!disposed && authState === "authenticated")
      $("notice").textContent = e.message || String(e);
  };
  const run =
    (fn) =>
    async (...a) => {
      try {
        $("notice").textContent = "";
        await fn(...a);
      } catch (e) {
        report(e);
      }
    };
  const contextEditor = bindContextEditor(document, {
    getArgs: () => JSON.parse($("args").value),
    setArgs: args => {$("args").value = JSON.stringify(args, null, 2); editorDirty = true;},
    getMetadata: () => catalog.find(model => model.path === JSON.parse($("args").value)[0])?.context || {},
    onApply: () => renderParameters(),
    onDirty: () => {editorDirty = true;},
    run,
  });

  const lmcacheEditor = bindLMCacheEditor(document, {
    getProfile: () => ({...(editorBase || {}), port: 8000, args: JSON.parse($("args").value),
      power: {...editorBase?.power, mode: $("powerMode").value}}),
    setProfile: profile => {
      editorBase = {...editorBase, lmcache: structuredClone(profile.lmcache)};
      $("args").value = JSON.stringify(profile.args, null, 2);
      editorDirty = true;
      contextEditor.refresh(true);
    },
    beforeApply: () => contextEditor.flush(),
    onApply: () => renderParameters(),
    onDirty: () => {editorDirty = true;},
    run,
  });
  async function enterWorkspace() {
    if (disposed) return;
    setAuth("authenticated");
    await load();
  }
  function checkSession() {
    if (authTask) return authTask;
    const generation = ++authGeneration;
    authTask = (async () => {
      try {
        const r = await fetch("/console-api/session", {
          signal: AbortSignal.timeout(10000),
          cache: "no-store",
        });
        if (!r.ok) throw Error("登录状态暂不可用");
        const d = await r.json();
        if (disposed || generation !== authGeneration) return;
        syncLoginMode(d.accountConfigured);
        if (!d.authenticated) setAuth("anonymous");
        else if (authState !== "authenticated") await enterWorkspace();
      } catch (e) {
        if (!disposed && generation === authGeneration) {
          if (authState !== "authenticated") setAuth("unavailable", e.message);
          else report(e);
        }
      } finally {
        authTask = null;
      }
    })();
    return authTask;
  }
  setAuth("checking");
  $("retryAuth").onclick = () => {
    setAuth("checking");
    void checkSession();
  };
  $("signout").onclick = run(async () => {
    await api("/console-api/logout", {});
    ++authGeneration;
    setAuth("anonymous", "已退出登录");
    if (options.embedded) location.replace("/");
  });
  const onRestore = () => void checkSession();
  globalThis.addEventListener("pageshow", onRestore);
  globalThis.addEventListener("focus", onRestore);
  let navigationGeneration = 0;
  const nativeRoutes = new Set(["harness", "usage", "plugins"]);
  const nativePanelIds = {usage:"sm75-usage", plugins:"plugins"};
  const onNativeNavigate = () => {void page("harness").catch(report);};
  const onNativeSettings = event => {
    selectSettingsTab(event.detail?.id || "connections");
    void page("settings").catch(report);
  };
  document.addEventListener("sm75-native-navigate", onNativeNavigate);
  document.addEventListener("sm75-native-settings", onNativeSettings);
  function route(action) {
    if (!active) throw Error("请先选择运行配置");
    return "/console-api/profiles/" + active + "/" + action;
  }
  async function page(id, { record = true } = {}) {
    if (id === "overview") id = "monitor";
    if (id === "benchmarks") id = "tests";
    if (!Object.hasOwn(routeTitles, id)) id = "profiles";
    const navigation = ++navigationGeneration;
    if (id !== current && current === "profiles" && !(await resolveDraft())) {
      if (!options.embedded) history.replaceState(null, "", "/?page=" + current);
      return;
    }
    if (authState !== "authenticated" || navigation !== navigationGeneration) return;
    if (nativeRoutes.has(id) && !options.headerOnly) {
      const button = document.querySelector('#nav [data-page="' + id + '"]');
      button?.setAttribute("aria-busy", "true");
      $("notice").textContent = "正在打开" + routeTitles[id] + "…";
      try {
        await openHarness();
        if (navigation !== navigationGeneration) return;
        const workbench = globalThis.__SM75_WORKBENCH__;
        if (id !== "harness") {
          const findPanel = () => workbench.panels().find(item => item.id === nativePanelIds[id]);
          const panel = findPanel() || await new Promise((resolve, reject) => {
            const check = () => {const value = findPanel(); if (value) {cleanup(); resolve(value);}};
            const timer = setTimeout(() => {cleanup(); reject(Error(routeTitles[id] + "组件尚未就绪"));}, 10000);
            const cleanup = () => {clearTimeout(timer); document.removeEventListener("sm75-native-panels", check);};
            document.addEventListener("sm75-native-panels", check);
            check();
          });
          if (navigation !== navigationGeneration) return;
          workbench.selectPanel(panel.id);
        } else if (current !== "harness") workbench.selectPanel(null);
      } catch (error) {
        if (navigation !== navigationGeneration) return;
        report(error);
        return;
      } finally {
        button?.removeAttribute("aria-busy");
      }
      $("notice").textContent = "";
    }
    if (navigation !== navigationGeneration) return;
    current = id;
    shell.dataset.page = id;
    document.dispatchEvent(new CustomEvent("sm75-route-change", {detail:{page:id}}));
    shell.classList.add("engine-toolbar");
    $("liveSummary").hidden = false;
    shell.classList.toggle("chat-page", id === "chat");
    shell.classList.toggle("monitor-page", id === "monitor");
    $("monitorLogs").hidden = id !== "monitor";
    $("monitorControls").hidden = id !== "monitor";
    if (id === "tests") refreshBenchmarks().catch(report);
    if (id === "chat") loadChatControls().catch(report);
    if (id === "profiles") refreshProfileCards().catch(report);
    if (!options.embedded && record && location.search !== "?page=" + id)
      history.pushState(null, "", "/?page=" + id);
    const content = nativeRoutes.has(id) ? "harness" : id;
    for (const section of shell.querySelectorAll(":scope>section"))
      section.hidden = section.id !== content;
    $("title").textContent = routeTitles[id];
    $("title").hidden = id === "harness" && !!globalThis.__SM75_WORKBENCH__?.ready;
    $("workspaceToolbar").hidden = id !== "harness";
    $("workspaceSidebar").hidden = id !== "harness";
    shell.classList.toggle("workspace-page", nativeRoutes.has(id));
    for (const button of document.querySelectorAll("#nav button"))
      button.classList.toggle("active", button.dataset.page === id);
    if (id === "settings") activateSettingsTab().catch(report);
    else globalThis.__SM75_WORKBENCH__?.selectSettings(null);
    if (id === "monitor" && active) {
      const src = route("monitor");
      if ($("monitorFrame").getAttribute("src") !== src) $("monitorFrame").src = src;
    }
  }
  for (const button of document.querySelectorAll("[data-page]"))
    button.onclick = run(() => page(button.dataset.page));

  let settingsTab = "deployment";
  let settingsGeneration = 0;
  const nativeSettingsTabs = new Set(["appearance", "assistant", "connections"]);
  function selectSettingsTab(id) {
    const allowed = new Set(["appearance", "assistant", "connections", "sampling", "deployment", "security"]);
    settingsTab = allowed.has(id) ? id : "deployment";
    ++settingsGeneration;
    for (const button of document.querySelectorAll("[data-settings-tab]")) {
      const selected = button.dataset.settingsTab === settingsTab;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    }
    for (const panel of document.querySelectorAll("[data-settings-panel]"))
      panel.hidden = panel.dataset.settingsPanel !== settingsTab;
    $("nativeSettingsSurface").hidden = !nativeSettingsTabs.has(settingsTab);
    $("saveUnifiedSettings").hidden = !["deployment", "sampling"].includes(settingsTab);
    $("settingsSaveHint").textContent = {
      appearance: "主题和字号调整后立即保存。",
      assistant: "默认助手选择立即保存；其他配置使用对应的保存操作。",
      connections: "连接配置在对应表单中校验并保存。",
      sampling: "保存后用于后续请求；当前会话正在生成的内容不受影响。",
      deployment: "目录与默认启动配置保存后生效；正在运行的模型保持原配置。",
      security: "账号与访问网段分别保存；修改已有配置需验证当前密码。",
    }[settingsTab];
    try { localStorage.setItem("sm75-settings-tab", settingsTab); } catch {}
    if (current === "settings" && authState === "authenticated") activateSettingsTab().catch(report);
  }
  async function activateSettingsTab() {
    const generation = settingsGeneration;
    const native = nativeSettingsTabs.has(settingsTab);
    if (!native) {globalThis.__SM75_WORKBENCH__?.selectSettings(null); return;}
    const hint=$("settingsSaveHint").textContent;
    if(!globalThis.__SM75_WORKBENCH__?.ready)
      $("settingsSaveHint").textContent="正在加载工作台设置…";
    try {
      await openHarness();
      if (current !== "settings" || generation !== settingsGeneration) return;
      globalThis.__SM75_WORKBENCH__.selectSettings(settingsTab);
      $("settingsSaveHint").textContent=hint;
    } catch (error) {
      if (current === "settings" && generation === settingsGeneration)
        $("settingsSaveHint").textContent = error.message + "。部署硬件和账号访问仍可直接配置。";
    }
  }
  $("saveUnifiedSettings").onclick = run(async () => {
    const target = settingsTab === "sampling" ? "saveModelSampling" : "saveSettings";
    $("saveUnifiedSettings").disabled = true;
    try {await $(target).onclick();}
    finally {$("saveUnifiedSettings").disabled = false;}
  });
  const settingsButtons = [...document.querySelectorAll("[data-settings-tab]")];
  for (const button of settingsButtons) {
    button.onclick = () => selectSettingsTab(button.dataset.settingsTab);
    button.onkeydown = event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const index = settingsButtons.indexOf(button);
      const next = event.key === "Home" ? 0 : event.key === "End" ? settingsButtons.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + settingsButtons.length) % settingsButtons.length;
      settingsButtons[next].focus();
      selectSettingsTab(settingsButtons[next].dataset.settingsTab);
    };
  }
  try {selectSettingsTab(localStorage.getItem("sm75-settings-tab") || "deployment");}
  catch {selectSettingsTab("deployment");}
  function fillProfile(imported = null) {
    const p = imported || profiles.find((p) => p.id === active);
    if (!p) return;
    editorBase = structuredClone(p);
    $("pid").value = p.id;
    $("profileDefault").checked =
      globalSettings.defaultProfile === p.id && !!globalSettings.autoStart;
    $("profileName").value = profileLabel(p);
    $("format").value = p.format;
    $("apiPort").value = p.port;
    $("cacheRoot").value = globalSettings.cacheRoot || p.cacheRoot;
    $("args").value = JSON.stringify(p.args, null, 2);
    contextEditor.refresh(true);
    lmcacheEditor.refresh(true);
    $("binds").value = JSON.stringify(p.binds || [], null, 2);
    const val = (name, fallback) => {
      const i = p.args.indexOf(name);
      return i >= 0 ? p.args[i + 1] : fallback;
    };
    $("modelPath").value = p.args[0];
    for (const [id, flag, fallback] of [
      ["tp", "--tensor-parallel-size", "4"],
      ["seq", "--max-num-seqs", "4"],
      ["batch", "--max-num-batched-tokens", "8192"],
      ["util", "--gpu-memory-utilization", ".87"],
      ["context", "--max-model-len", "auto"],
      ["idle", "--auto-sleep-idle-timeout", "30"],
    ])
      $(id).value = val(flag, fallback);
    const spec = JSON.parse(val("--speculative-config", "{}"));
    $("variant").value = spec.method || "base";
    syncModelChoices(p.args[0], spec.model);
    $("draftPath").value = spec.model || "";
    $("cpuKv").value = readCpuCompatibility(p.args).cpuKvGiB || "8";
    fillProfilePowerFields(document, p);
    syncPower();
    renderParameters();
    fillCombo();
  }
  async function load() {
    const generation = authGeneration;
    profiles = await api("/console-api/profiles");
    const settings = await api("/console-api/settings");
    globalSettings = settings;
    catalog = await api("/console-api/models");
    if (!active) active = settings.defaultProfile;
    await loadSettings(settings);
    if (
      disposed ||
      generation !== authGeneration ||
      authState !== "authenticated"
    )
      return;
    $("active").replaceChildren(
      ...profiles.map((p) => new Option(profileLabel(p), p.id)),
    );
    active = profiles.some((p) => p.id === active)
      ? active
      : profiles[0]?.id || "";
    $("active").value = active;
    $("active").title = profiles.find(p => p.id === active)?.name || "当前运行配置";
    fillProfile();
    $("login").hidden = true;
    await page(current, { record: false });
    if (active && !options.headerOnly) void openHarness().catch(() => {});
    await loadChat();
    await refresh();
  }
  let benchmarkHistory = [],
    benchmarkSignature = "";
  const benchmarkTypes = ["speedtest", "sql"],
    benchmarkPending = new Set();
  async function refreshBenchmarks() {
    await Promise.all(
      benchmarkTypes.map(async (type) => {
        if (benchmarkPending.has(type)) return;
        const ext = await api("/console-api/benchmarks/extension?type=" + type);
        if (benchmarkPending.has(type)) return;
        const toggle = $("benchToggle-" + type),
          status = $("benchStatus-" + type),
          frame = $("benchFrame-" + type);
        toggle.checked = !!ext.enabled;
        status.textContent = ext.installing
          ? "正在部署…"
          : ext.enabled && !ext.installed
            ? "部署未完成，请关闭后重新开启。"
            : "";
        status.hidden = !status.textContent;
        const ready = ext.enabled && ext.installed;
        frame.hidden = !ready;
        const src =
          "/bench-app/" +
          type +
          "/" +
          (type === "sql" ? "sql_benchmark.html" : "") +
          "?profile=" +
          encodeURIComponent(active);
        if (ready && frame.getAttribute("src") !== src) frame.src = src;
        if (!ready) frame.removeAttribute("src");
      }),
    );
  }
  for (const type of benchmarkTypes)
    $("benchToggle-" + type).onchange = run(async () => {
      const toggle = $("benchToggle-" + type),
        status = $("benchStatus-" + type);
      benchmarkPending.add(type);
      toggle.disabled = true;
      status.textContent = toggle.checked ? "正在开启…" : "正在关闭…";
      status.hidden = false;
      try {
        await api("/console-api/benchmarks/extension?type=" + type, {
          enabled: toggle.checked,
        });
      } finally {
        benchmarkPending.delete(type);
        toggle.disabled = false;
        await refreshBenchmarks();
      }
    });
  $("signin").onclick = run(async () => {
    if (authState === "submitting") return;
    const generation = ++authGeneration;
    setAuth("submitting");
    try {
      await api("/console-api/login", accountConfigured
        ? { username: $("loginUsername").value.trim(), password: $("token").value }
        : { token: $("token").value });
      if (disposed || generation !== authGeneration) return;
      $("token").value = "";
      await enterWorkspace();
    } catch (e) {
      if (generation === authGeneration)
        setAuth(
          e.status === 401 || e.status === 429 ? "anonymous" : "unavailable",
          e.message,
        );
    }
  });
  $("token").onkeydown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      $("signin").click();
    }
  };
  $("loginUsername").onkeydown = $("token").onkeydown;
  const onPop = () =>
    page(new URLSearchParams(location.search).get("page") || "profiles", {
      record: false,
    });
  if (!options.embedded) globalThis.addEventListener("popstate", onPop);
  $("active").onchange = run(async () => {
    const selected = $("active").value;
    if (!(await resolveDraft())) {
      $("active").value = active;
    $("active").title = profiles.find(p => p.id === active)?.name || "当前运行配置";
      return;
    }
    if (abort) {
      $("active").value = active;
    $("active").title = profiles.find(p => p.id === active)?.name || "当前运行配置";
      throw Error("先停止当前回复再切换配置");
    }
    active = selected;
    $("model").value = "";
    fillProfile();
    await loadChat();
    page(current);
    await refresh();
  });
  async function refresh() {
    if (!active) $("state").textContent = "未选择运行配置";
    if (current === "monitor" && !$("monitorLiveView").hidden)
      await refreshHardware();
    if (current === "profiles") await refreshProfileCards();
    if (current === "models") await refreshModels();
    if (current === "tests") await refreshBenchmarks();
    const sys = await api("/console-api/system");
    $("gpu").replaceChildren(
      ...sys.gpus.map((g) => {
        const el = document.createElement("div");
        el.className = "card";
        for (const text of [
          `GPU ${g.index} · ${g.name}`,
          `${g.power} W`,
          `${g.usedMiB} / ${g.totalMiB} MiB`,
          `${g.util}% · ${g.temp}°C · ${g.pstate}`,
        ]) {
          const n = document.createElement(
            el.children.length === 1 ? "b" : "p",
          );
          n.textContent = text;
          el.append(n);
        }
        return el;
      }),
    );
    if (active) {
      const st = await api(route("status"));
      $("state").textContent = st.running ? "模型进程运行" : "未运行";
      if ($("sm75P8"))
        $("sm75P8").hidden = !(
          st.running &&
          profiles.find((p) => p.id === active)?.power?.mode === "pstate"
        );
      if (st.running) {
        try {
          const pw = await api("/console-api/power");
          const pm = profiles.find((p) => p.id === active)?.power?.mode;
          if (pw.state) {
            if (pw.state.mode === "idle")
              $("state").textContent += " · P-State 空闲 P8（已进入）";
            else {
              const r = Math.max(0, pw.state.idle_remaining || 0);
              $("state").textContent +=
                " · P-State 驱动自动（" +
                (r < 120
                  ? r + " 秒后进入 P8"
                  : Math.round(r / 60) + " 分后进入 P8") +
                "）";
            }
          } else if (pm === "pstate")
            $("state").textContent += " · P-State 等待生效";
        } catch {}
      }
      $("preview").textContent = JSON.stringify(
        await api(route("preview")),
        null,
        2,
      );
    }
    const jobs = await api("/console-api/jobs");
    $("taskCount").textContent = String(
      jobs.filter((j) => j.state === "running").length,
    );
    $("jobs").replaceChildren(
      ...jobs.map((j) => {
        const el = document.createElement("div"),
          label = document.createElement("div");
        label.textContent = `${j.download?.repo || j.kind} · ${{ running: "运行中", complete: "已完成", failed: "失败", cancelled: "已取消", interrupted: "已中断，可继续下载" }[j.state] || j.state}`;
        const details = document.createElement("details"),
          summary = document.createElement("summary"),
          log = document.createElement("pre");
        summary.textContent = "任务日志";
        log.className = "job-log";
        log.textContent = j.log.replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
        details.append(summary, log);
        el.append(label, details);
        if (
          ["running", "failed", "cancelled", "interrupted"].includes(j.state)
        ) {
          const b = document.createElement("button");
          b.textContent =
            j.state === "running" ? "取消" : j.download ? "继续下载" : "重试";
          b.onclick = run(async () => {
            await api("/console-api/jobs", {
              [j.state === "running" ? "cancel" : "retry"]: j.id,
            });
            await refresh();
          });
          el.append(b);
        }
        return el;
      }),
    );
  }
  let logsTimer = null,
    logsPaused = false,
    logsAll = [],
    logsLimit = 3000,
    logsOffset = 0,
    logsSize = 0,
    logsStamp = 0;
  function logsRender() {
    const shown = logsAll.slice(0, logsLimit);
    $("logs").textContent = shown.join("\n");
    $("moreLogs").hidden = logsAll.length <= logsLimit;
    $("logsHint").textContent =
      "实时刷新 · 最新在最上 · 显示 " +
      shown.length +
      "/" +
      logsAll.length +
      " 行" +
      (logsSize ? " · 文件 " + (logsSize / 1048576).toFixed(1) + "MB" : "") +
      (logsStamp
        ? " · 更新于 " + new Date(logsStamp).toLocaleTimeString()
        : "");
  }
  function logsPrepend(text) {
    const lines = text.split("\n");
    if (lines.length && lines[lines.length - 1] === "") lines.pop();
    lines.reverse();
    if (lines.length) logsAll = lines.concat(logsAll);
  }
  async function logsInit() {
    const r = await api(route("logs"));
    logsAll = [];
    logsPrepend(r.text || "");
    logsLimit = 3000;
    logsOffset = r.offset || 0;
    logsSize = r.size || 0;
    logsStamp = Date.now();
    logsRender();
  }
  async function logsStep() {
    const r = await api(route("logs") + "?offset=" + logsOffset);
    logsSize = r.size || 0;
    logsStamp = Date.now();
    if (r.reset) {
      logsAll = [];
      logsPrepend(r.text || "");
      logsOffset = r.offset || 0;
    } else {
      logsPrepend(r.text || "");
      logsOffset = r.offset ?? logsOffset;
    }
    logsRender();
  }
  function stopLogs() {
    if (logsTimer) {
      clearInterval(logsTimer);
      logsTimer = null;
    }
  }
  async function openLogs() {
    $("logs").textContent = "加载中…";
    $("logDialog").showModal();
    logsPaused = false;
    $("pauseLogs").textContent = "暂停";
    await logsInit();
    stopLogs();
    logsTimer = setInterval(() => {
      if (!logsPaused) logsStep().catch(() => {});
    }, 1500);
  }
  for (const b of document.querySelectorAll("[data-action]"))
    b.onclick = run(async () => {
      const a = b.dataset.action;
      if (a === "logs") {
        await openLogs();
      } else {
        await api(route(a), {});
        await refresh();
      }
    });
  $("pauseLogs").onclick = () => {
    logsPaused = !logsPaused;
    $("pauseLogs").textContent = logsPaused ? "继续" : "暂停";
  };
  $("moreLogs").onclick = () => {
    logsLimit += 5000;
    logsRender();
  };
  $("logDialog").addEventListener("close", stopLogs);
  $("save").onclick = run(async () => {
    const snapshot = editedProfile();
    profiles = await api("/console-api/profiles", snapshot);
    active = snapshot.id;
    if ($("profileDefault").checked) {
      globalSettings = await api("/console-api/settings", {
        defaultProfile: active,
        autoStart: true,
      });
    } else if (globalSettings.defaultProfile === active) {
      globalSettings = await api("/console-api/settings", {
        defaultProfile: "",
        autoStart: false,
      });
    }
    editorDirty = false;
    // Complete refresh before exposing the cards, so an immediate reopen cannot
    // restore an old profile while load() is still replacing the editor values.
    await load();
    closeEditor();
    await refreshProfileCards();
  });
  $("applyParams").onclick = run(() => {
    const args = contextArgsForModel($("modelPath").value);
    function put(k, v) {
      const i = args.indexOf(k);
      if (i >= 0) args.splice(i, 2);
      if (v !== null) args.push(k, String(v));
    }
    for (const [id, flag] of [
      ["tp", "--tensor-parallel-size"],
      ["seq", "--max-num-seqs"],
      ["batch", "--max-num-batched-tokens"],
      ["util", "--gpu-memory-utilization"],
      ["idle", "--auto-sleep-idle-timeout"],
    ])
      put(flag, $(id).value);
    put("--auto-sleep-offload-target", "exit");
    const mode = $("variant").value;
    if (mode === "base") {
      put("--speculative-config", null);
      put("--scheduler-cls", null);
    } else {
      if (mode === "dflash" && !$("draftPath").value)
        throw Error("填写匹配的草稿模型路径");
      const si = args.indexOf("--speculative-config"),
        previous = si >= 0 ? JSON.parse(args[si + 1]) : {},
        same = previous.method === mode;
      const spec = {
        ...(same ? previous : {}),
        method: mode,
        num_speculative_tokens: same
          ? (previous.num_speculative_tokens ?? (mode === "mtp" ? 5 : 7))
          : mode === "mtp"
            ? 5
            : 7,
      };
      if (mode === "dflash") {
        spec.model = $("draftPath").value;
        spec.draft_tensor_parallel_size = Number($("tp").value);
      }
      put("--speculative-config", JSON.stringify(spec));
      put("--scheduler-cls", "vllm.v1.core.sched.scheduler_sm75.SM75Scheduler");
    }
    $("args").value = JSON.stringify(args, null, 2);
    contextEditor.refresh(true);
    renderParameters();
  });
  $("download").onclick = run(async () => {
    if (!downloadSelection) throw Error("请先从搜索结果选择模型");
    await api("/console-api/download", downloadSelection);
    $("downloadConfirm").hidden = true;
    downloadSelection = null;
    await refresh();
  });

  let samplingData = { templates: [], models: {}, configured: [] };
  async function refreshSampling() {
    samplingData = await api("/console-api/sampling");
    for (const id of ["samplingTemplate", "modelSamplingTemplate"]) {
      const selected = $(id).value;
      $(id).replaceChildren(
        new Option("选择参数模板", ""),
        ...samplingData.templates.map((t) => new Option(t.name, t.id)),
      );
      $(id).value = selected;
    }
    const chosen = $("samplingModel").value;
    $("samplingModel").replaceChildren(
      ...samplingData.configured.map(
        (m) => new Option(m.name + " · " + m.provider, m.id),
      ),
    );
    if (samplingData.configured.some((m) => m.id === chosen))
      $("samplingModel").value = chosen;
    paintModelSampling();
  }
  function paintModelSampling() {
    const params = samplingData.models[$("samplingModel").value] || {};
    const defaults = samplingData.configured.find(m => m.id === $("samplingModel").value)?.defaults;
    for (const key of Object.keys(chatFields)) {
      const input = $("modelSampling-" + key);
      input.value = params[key] ?? "";
      const value = defaults?.values?.[key];
      input.placeholder = Number.isFinite(value) ? `服务默认 ${value}` : "默认值未提供";
      input.title = Number.isFinite(value)
        ? `${key === "max_tokens" ? defaults.outputSource : defaults.source}：${value}`
        : "服务未提供此参数的默认值";
    }
    $("modelSamplingDefaultInfo").textContent = defaults
      ? [defaults.source, defaults.outputNote].filter(Boolean).join("。") : "服务未提供默认值";
    $("modelSamplingRecommendations").hidden = !defaults?.recommendations;
    paintModelRecommendation();
  }
  function paintModelRecommendation() {
    const defaults = samplingData.configured.find(m => m.id === $("samplingModel").value)?.defaults;
    const params = defaults?.recommendations?.[$("modelSamplingRecommendation").value];
    $("modelSamplingRecommendationInfo").textContent = params
      ? `${defaults.recommendationModel} 推荐：温度 ${params.temperature} · Top P ${params.top_p} · Top K ${params.top_k} · Min P ${params.min_p} · Presence ${params.presence_penalty} · 重复惩罚 ${params.repetition_penalty}` : "";
  }
  function applyModelRecommendation() {
    const defaults = samplingData.configured.find(m => m.id === $("samplingModel").value)?.defaults;
    const params = defaults?.recommendations?.[$("modelSamplingRecommendation").value];
    if (!params) throw Error("此模型尚未提供推荐参数");
    for (const [key, value] of Object.entries(params))
      $("modelSampling-" + key).value = value;
  }
  function modelSamplingValues() {
    const params = {};
    for (const key of Object.keys(chatFields)) {
      const input = $("modelSampling-" + key);
      if (input.value !== "") params[key] = Number(input.value);
    }
    return params;
  }
  $("samplingModel").onchange = paintModelSampling;
  $("modelSamplingRecommendation").onchange = paintModelRecommendation;
  $("applyModelRecommendation").onclick = run(() => {
    applyModelRecommendation();
    $("notice").textContent = "推荐参数已填入，保存后生效；思考开关仍由会话控制。";
  });
  $("loadSampling").onclick = run(async () => {
    const t = samplingData.templates.find(
      (t) => t.id === $("samplingTemplate").value,
    );
    if (!t) throw Error("请选择参数模板");
    chatPreference.mode = "custom";
    chatPreference.custom = { ...t.params };
    paintChatControls();
    persistChatControls();
  });
  $("saveSampling").onclick = run(async () => {
    const params = chatRequestOptions();
    delete params.chat_template_kwargs;
    delete params.thinking_token_budget;
    await api("/console-api/sampling", {
      action: "template",
      name: $("samplingName").value,
      params,
    });
    await refreshSampling();
    $("notice").textContent = "参数模板已保存。";
  });
  $("loadModelSampling").onclick = run(() => {
    const t = samplingData.templates.find(
      (t) => t.id === $("modelSamplingTemplate").value,
    );
    if (!t) throw Error("请选择参数模板");
    for (const key of Object.keys(chatFields))
      $("modelSampling-" + key).value = t.params[key] ?? "";
  });
  $("resetModelSampling").onclick = () => {
    for (const key of Object.keys(chatFields))
      $("modelSampling-" + key).value = "";
  };
  $("saveModelSampling").onclick = run(async () => {
    await api("/console-api/sampling", {
      action: "model",
      model: $("samplingModel").value,
      params: modelSamplingValues(),
    });
    await refreshSampling();
    $("notice").textContent = "模型参数已保存，下次请求生效。";
  });
  $("saveModelSamplingTemplate").onclick = run(async () => {
    await api("/console-api/sampling", {
      action: "template",
      name: $("modelSamplingName").value,
      params: modelSamplingValues(),
    });
    await refreshSampling();
    $("notice").textContent = "参数模板已保存。";
  });
  const chatFields = {
    temperature: "chatTemperature",
    top_p: "chatTopP",
    top_k: "chatTopK",
    min_p: "chatMinP",
    presence_penalty: "chatPresence",
    repetition_penalty: "chatRepetition",
    max_tokens: "maxTokens",
  };
  let chatControlsLoading = false,
    chatModelReady = false;
  let chatDefaults = { sampling: {}, thinking: false, budgets: {} },
    chatPreference = { mode: "recommended", thinking: "default", custom: {} },
    chatControlKey = "";
  function persistChatControls() {
    if (chatControlKey)
      localStorage.setItem(chatControlKey, JSON.stringify(chatPreference));
  }
  function paintChatControls() {
    $("chatParameterMode").value = chatPreference.mode;
    $("chatThinking").value = chatPreference.thinking;
    for (const [key, id] of Object.entries(chatFields)) {
      $(id).value =
        (chatPreference.mode === "custom"
          ? chatPreference.custom[key]
          : chatDefaults.sampling[key]) ?? "";
      $(id).disabled = chatPreference.mode !== "custom";
      $(id).placeholder = key === "top_k" ? "-1 不限制" : "服务默认";
    }
    $("chatThinking").disabled = !chatDefaults.thinking;
    $("chatParameterHint").textContent =
      chatDefaults.source +
      "。空值沿用服务默认。" +
      (chatDefaults.thinking
        ? "低 / 中 / 高分别限制思考预算为 1024 / 4096 / 16384 tokens，实际思考可提前结束。"
        : "当前配置未启用推理解析器。");
  }
  async function loadChatControls() {
    if (!active || chatControlsLoading) return;
    chatControlsLoading = true;
    const requestProfile = active;
    try {
      chatDefaults = await api(route("chat-config"));
      const models = await api(route("models"));
      if (requestProfile !== active) return;
      const previous = $("model").value;
      $("model").replaceChildren(
        ...(models.data || []).map((m) => new Option(m.id, m.id)),
      );
      if ((models.data || []).some((m) => m.id === previous))
        $("model").value = previous;
      chatModelReady = !!$("model").value;
      $("sendChat").disabled = !chatModelReady;
      readChatPreference();
      if (
        $("notice").textContent ===
        "模型服务暂不可用；请检查运行配置与日志，恢复后可重试。"
      )
        $("notice").textContent = "";
    } catch {
      chatModelReady = false;
      $("sendChat").disabled = true;
      if (authState === "authenticated")
        $("notice").textContent =
          "模型服务暂不可用；请检查运行配置与日志，恢复后可重试。";
    } finally {
      chatControlsLoading = false;
    }
  }

  function readChatPreference() {
    chatControlKey = "sm75-chat-controls:" + active + ":" + $("model").value;
    try {
      chatPreference = JSON.parse(localStorage.getItem(chatControlKey)) || {
        mode: "recommended",
        thinking: "default",
        custom: {},
      };
    } catch {
      chatPreference = { mode: "recommended", thinking: "default", custom: {} };
    }
    paintChatControls();
  }
  $("model").onchange = readChatPreference;
  $("chatThinking").onchange = () => {
    chatPreference.thinking = $("chatThinking").value;
    persistChatControls();
  };
  $("chatParameterMode").onchange = () => {
    chatPreference.mode = $("chatParameterMode").value;
    if (
      chatPreference.mode === "custom" &&
      !Object.keys(chatPreference.custom).length
    )
      chatPreference.custom = { ...chatDefaults.sampling };
    paintChatControls();
    persistChatControls();
  };
  for (const [key, id] of Object.entries(chatFields))
    $(id).oninput = $(id).onchange = () => {
      if ($(id).value === "") delete chatPreference.custom[key];
      else chatPreference.custom[key] = Number($(id).value);
      persistChatControls();
    };
  function chatRequestOptions() {
    const result =
      chatPreference.mode === "recommended" ? { ...chatDefaults.sampling } : {};
    if (chatPreference.mode === "custom") {
      for (const [key, id] of Object.entries(chatFields)) {
        if (!$(id).checkValidity()) throw Error("请检查生成参数");
        if ($(id).value !== "") result[key] = Number($(id).value);
      }
    }
    if (chatPreference.mode === "custom") {
      chatPreference.custom = { ...result };
      persistChatControls();
    }
    if (result.top_k === 0) throw Error("Top K 使用 -1 或正整数");
    const effort = chatPreference.thinking;
    if (chatDefaults.thinking && effort !== "default") {
      result.chat_template_kwargs = { enable_thinking: effort !== "off" };
      if (effort !== "off") {
        result.thinking_token_budget = chatDefaults.budgets[effort];
        if (
          result.max_tokens &&
          result.max_tokens <= result.thinking_token_budget
        )
          throw Error("最大输出 tokens 需大于所选思考预算，为正文预留空间");
      }
    }
    return result;
  }

  async function loadChat() {
    const id = active;
    if (!id) return;
    const saved = await api("/console-api/chats/" + id);
    if (id !== active) return;
    chat = saved.messages || [];
    $("messages").replaceChildren();
    for (const m of chat) message(m.role, m.content);
  }
  async function saveChat(profileId = active, messages = chat) {
    if (profileId)
      await api("/console-api/chats/" + profileId, { messages: messages.slice(-200) });
  }
  $("clearChat").onclick = run(async () => {
    if (abort) throw Error("先停止当前回复");
    chat = [];
    $("messages").replaceChildren();
    $("chatStats").textContent = "就绪";
    await saveChat();
  });
  $("cancel").onclick = () => abort?.abort();
  $("prompt").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("chatForm").requestSubmit();
    }
  };
  function messageText(content) {
    return Array.isArray(content) ? content.filter(x => x.type === "text").map(x => x.text).join("\n") + content.filter(x => x.type === "image").map(() => "[图片]").join(" ") : String(content || "");
  }
  function message(role, text) {
    const el = document.createElement("div");
    el.className = "message " + role;
    el.textContent = messageText(text).trimStart();
    $("messages").append(el);
    el.scrollIntoView({ block: "end" });
    return el;
  }
  const selectedImages = [];
  $("pickChatImages").onclick = () => $("chatImages").click();
  $("chatImages").onchange = () => { selectedImages.splice(0,selectedImages.length,...$("chatImages").files); $("chatImageList").textContent = selectedImages.map(x=>x.name).join("、"); };
  $("chatForm").onsubmit = run(async (e) => {
    e.preventDefault();
    if (abort) throw Error("当前回复尚未完成");
    const text = $("prompt").value.trim();
    if (!text && !selectedImages.length) return;
    const requestProfile = active, requestUrl = route("chat"), requestChat = chat;
    const images = [...selectedImages];
    const controller = new AbortController();
    abort = controller;
    $("sendChat").hidden = true;
    $("cancel").hidden = false;
    try {
    let model = $("model").value;
    if (!model) {
      await loadChatControls();
      model = $("model").value;
    }
    if (!model) throw Error("模型尚未就绪");
    const requestOptions = chatRequestOptions();
    const imageIds = await uploadChatImages(images, requestProfile, controller.signal);
    controller.signal.throwIfAborted();
    if (active !== requestProfile) throw Error("运行配置已变化，请重新发送");
    const content = imageIds.length ? [{type:"text",text}, ...imageIds.map(attachmentId=>({type:"image",attachmentId}))] : text;
    requestChat.push({ role: "user", content });
    message("user", content);
    $("prompt").value = ""; selectedImages.splice(0); $("chatImages").value = ""; $("chatImageList").textContent = "";
    const out = message("assistant", "等待响应 / 休眠时正在唤醒…");
    const think = document.createElement("details");
    think.className = "thinking";
    think.hidden = true;
    const thinkSum = document.createElement("summary");
    thinkSum.innerHTML =
      '<svg class="thinking-icon" viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M12 2.6l1.8 5.6 5.6 1.8-5.6 1.8L12 17.4l-1.8-5.6L4.6 10l5.6-1.8z"/></svg><span class="thinking-label">思考中…</span><svg class="thinking-caret" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>';
    const thinkLabel = thinkSum.querySelector(".thinking-label");
    const thinkBody = document.createElement("div");
    thinkBody.className = "thinking-body";
    think.append(thinkSum, thinkBody);
    const contentEl = document.createElement("div");
    contentEl.className = "content";
    out.replaceChildren(think, contentEl);
    thinkSum.addEventListener("click", () => {
      think.dataset.touched = "1";
    });
    let thinkStart = null,
      thinkEnd = null;
    function thinkLabelText() {
      if (thinkStart === null) return "思考中…";
      const sec = Math.max(
        1,
        Math.round(((thinkEnd ?? performance.now()) - thinkStart) / 1000),
      );
      return "已思考（用时 " + sec + " 秒）";
    }
    const start = performance.now();
    let first = null,
      result = "",
      reason = "",
      usage = null,
      buffer = "",
      phase = "等待首响应",
      ended = null;
    function paintStatus() {
      const now = ended ?? performance.now(),
        elapsed = (now - start) / 1000;
      const items = [
        phase,
        model,
        `耗时 ${elapsed.toFixed(1)} 秒`,
        first === null
          ? "首响应等待中"
          : `首响应 ${((first - start) / 1000).toFixed(2)} 秒`,
      ];
      if (usage) {
        const n = usage.completion_tokens,
          cached = usage.prompt_tokens_details?.cached_tokens;
        items.push(
          `输入 ${usage.prompt_tokens ?? "—"} / 输出 ${n ?? "—"} Token`,
        );
        if (Number.isFinite(cached) && usage.prompt_tokens > 0)
          items.push(
            `缓存命中 ${((100 * cached) / usage.prompt_tokens).toFixed(0)}%`,
          );
        if (first !== null && now > first && Number.isFinite(n))
          items.push(
            `平均生成 ${(n / ((now - first) / 1000)).toFixed(1)} tok/s`,
          );
      } else {
        items.push(
          `已接收 ${(reason + result).length} 字符`,
          ended === null ? "Token 统计中" : "Token 用量未返回",
        );
      }
      const bar = $("chatStats");
      bar.dataset.phase = phase;
      bar.replaceChildren(
        ...items.map((text, i) => {
          const pill = document.createElement("span");
          pill.className = "chat-status-pill" + (i === 0 ? " phase" : "");
          pill.textContent = text;
          if (text.startsWith("平均生成"))
            pill.title =
              "生成 Token / 首个内容到结束的时间，包含思考 Token；不是引擎瞬时吞吐";
          return pill;
        }),
      );
    }
    paintStatus();
    const statusTimer = setInterval(paintStatus, 250);
    try {
      const r = await fetch(requestUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, messages: requestChat, ...requestOptions }),
        signal: controller.signal,
      });
      if (!r.ok) throw Error(await r.text());
      const decoder = new TextDecoder();
      for await (const chunk of r.body) {
        buffer += decoder.decode(chunk, { stream: true });
        let i;
        while ((i = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, i).trim();
          buffer = buffer.slice(i + 1);
          if (!line.startsWith("data:") || line === "data: [DONE]") continue;
          const d = JSON.parse(line.slice(5));
          if (d.error) throw Error(d.error.message || JSON.stringify(d.error));
          const delta = d.choices?.[0]?.delta || {};
          if (delta.content || delta.reasoning_content || delta.reasoning) {
            first ??= performance.now();
            phase = delta.content ? "输出中" : "思考中";
            result += delta.content || "";
            reason += delta.reasoning_content || delta.reasoning || "";
            if (reason) {
              think.hidden = false;
              thinkBody.textContent = reason.trim();
              if (thinkStart === null) thinkStart = performance.now();
              if (phase === "思考中") {
                think.open = true;
              } else if (thinkEnd === null) thinkEnd = performance.now();
              if (phase !== "思考中" && !think.dataset.touched)
                think.open = false;
              thinkLabel.textContent = thinkLabelText();
            }
            contentEl.textContent = result.replace(/^\n+/, "");
            $("messages").scrollTop = $("messages").scrollHeight;
          }
          if (d.usage) usage = d.usage;
        }
      }
      result = result.replace(/^\s+/, "");
      requestChat.push({ role: "assistant", content: result });
      phase = "已完成";
      if (reason) {
        if (thinkEnd === null) thinkEnd = performance.now();
        if (!think.dataset.touched) think.open = false;
        thinkLabel.textContent = thinkLabelText();
      }
    } catch (e) {
      phase = e.name === "AbortError" ? "已停止" : "请求失败";
      contentEl.textContent =
        result || (e.name === "AbortError" ? "已停止" : e.message);
      if (reason) {
        if (thinkEnd === null) thinkEnd = performance.now();
        if (!think.dataset.touched) think.open = false;
        thinkLabel.textContent = thinkLabelText();
      }
      if (result) requestChat.push({ role: "assistant", content: result });
    } finally {
      ended = performance.now();
      clearInterval(statusTimer);
      paintStatus();
      await saveChat(requestProfile, requestChat);
    }
    } finally {
      controller.abort();
      if (abort === controller) abort = null;
      $("sendChat").hidden = false;
      $("cancel").hidden = true;
    }
  });
  $("installHarness").onclick = run(async () => {
    await api("/console-api/harness/install", {});
    $("harnessState").textContent = "正在安装，可在模型库查看后台任务";
  });
  let harnessBoot;
  let nativeScriptsStarted = false;
  async function resolveHarnessProfile() {
    if (active && profiles.some(profile => profile.id === active)) return active;
    const available = await api("/console-api/profiles");
    const settings = globalSettings.defaultProfile
      ? globalSettings
      : await api("/console-api/settings");
    const selected = available.find(profile => profile.id === active)
      || available.find(profile => profile.id === settings.defaultProfile)
      || available[0];
    if (selected) {
      profiles = available;
      active = selected.id;
    }
    return selected?.id || "";
  }
  async function loadNativeWorkbench(url) {
    if (nativeScriptsStarted) throw Error("请刷新页面后重新连接工作台");
    const entry = new URL(url, location.origin);
    if (entry.origin !== location.origin || entry.pathname !== "/harness-ui/" ||
        entry.username || entry.password || entry.search || entry.hash)
      throw Error("工作台入口必须使用本机已登录的管理通道");
    if (globalThis.__DSH_TRANSPORT__ !== undefined)
      throw Error("存在未知工作台连接，请刷新页面后重试");
    const epoch = authEpoch;
    const checkIdentity = () => {
      if (authState !== "authenticated" || epoch !== authEpoch || disposed)
        throw Error("登录状态已改变，请重新连接工作台");
    };
    checkIdentity();
    const response = await fetch(entry.href, {
      cache:"no-store", credentials:"same-origin", redirect:"error",
      signal:AbortSignal.timeout(15000),
    });
    if (!response.ok || !response.headers.get("content-type")?.includes("text/html"))
      throw Error("工作台资源加载失败：" + response.status);
    const html = new DOMParser().parseFromString(await response.text(), "text/html");
    checkIdentity();
    // Official host ownership applies only to this authenticated same-origin loader.
    // Every subsequent HTTP/RPC/WebSocket request still passes the server's session gate.
    globalThis.__DSH_TRANSPORT__ = Object.freeze({ownsHost:true});
    const sameOriginUrl = value => {
      const resolved = new URL(value, new URL(url, location.origin));
      if (resolved.origin !== location.origin) throw Error("工作台资源来源不匹配");
      return resolved.href;
    };
    // Use the signed-in, same-origin native boot document as the asset manifest.
    // No page, navigation, or second document is mounted.
    for (const node of html.querySelectorAll('link[rel="stylesheet"], style')) {
      const asset = document.createElement(node.tagName.toLowerCase());
      if (node.tagName === "LINK") {asset.rel="stylesheet";asset.href=sameOriginUrl(node.getAttribute("href"));}
      else asset.textContent=node.textContent;
      asset.dataset.sm75Native="true";
      document.head.append(asset);
    }
    const scripts = [...html.querySelectorAll("script")];
    nativeScriptsStarted = true;
    const execute = node => new Promise((resolve,reject) => {
      checkIdentity();
      const script=document.createElement("script");
      if(node.type) script.type=node.type;
      if(node.src) {
        script.src=sameOriginUrl(node.getAttribute("src"));
        script.async=false;
        const timer=setTimeout(()=>reject(Error("工作台脚本加载超时，请刷新页面重试")),15000);
        script.onload=()=>{clearTimeout(timer);resolve();};
        script.onerror=()=>{clearTimeout(timer);reject(Error("工作台脚本加载失败，请刷新页面重试"));};
      } else script.textContent=node.textContent;
      document.head.append(script);
      if(!node.src) resolve();
    });
    for(const node of scripts.filter(node=>node.type!=="module")) await execute(node);
    for(const node of scripts.filter(node=>node.type==="module")) await execute(node);
    await new Promise((resolve,reject)=>{
      if(globalThis.__SM75_WORKBENCH__?.ready) {resolve();return;}
      const ready=()=>{if(!globalThis.__SM75_WORKBENCH__?.ready)return;clearTimeout(timer);document.removeEventListener("sm75-native-ready",ready);resolve();};
      const timer=setTimeout(()=>{document.removeEventListener("sm75-native-ready",ready);reject(Error("工作台初始化超时，请刷新后重试"));},45000);
      document.addEventListener("sm75-native-ready",ready);
    });
  }
  async function openHarness() {
    if (options.onConversation) {options.onConversation();return;}
    if (globalThis.__SM75_WORKBENCH__?.ready) return;
    if (!harnessBoot) {
      $("harnessState").textContent="正在连接工作台…";
      $("harnessState").hidden=false;
      $("root").hidden=false;
      harnessBoot=(async()=>{
        const profile = await resolveHarnessProfile();
        if (!profile) throw Error("请先创建运行配置");
        const data=await api("/console-api/harness/start",{profile});
        await loadNativeWorkbench(data.url);
        $("harnessState").hidden=true;
        $("startHarness").hidden=true;
      })();
    }
    try {
      await harnessBoot;
      if(!globalThis.__SM75_WORKBENCH__?.ready)
        throw Error("工作台组件已断开，请刷新页面重新连接");
    }
    catch(error) {
      harnessBoot = undefined;
      $("harnessState").textContent="工作台连接失败："+error.message;
      $("harnessState").hidden=false;
      $("startHarness").hidden=false;
      throw error;
    }
  }
  $("startHarness").onclick = run(async () => {
    if (nativeScriptsStarted && !globalThis.__SM75_WORKBENCH__?.ready) location.reload();
    else await openHarness();
  });
  $("stopHarness").onclick = run(async () => {
    await api("/console-api/harness/stop", {});
    location.replace("/");
    $("harnessState").hidden = false;
    $("harnessState").textContent = "DSH 已停止";
    $("startHarness").hidden = false;
  });
  void checkSession();
  const refreshTimer = setInterval(async () => {
    if (authState !== "authenticated" || refreshing || disposed) return;
    refreshing = true;
    try {
      await checkSession();
      if (authState !== "authenticated" || options.isActive?.() === false)
        return;
      await refresh();
      if (current === "chat" && !chatModelReady) await loadChatControls();
    } catch (e) {
      report(e);
    } finally {
      refreshing = false;
    }
  }, 5000);

  function syncModelChoices(main, draft) {
    const populate = (id, selected, rows) => {
      $(id).replaceChildren(
        new Option("请选择", ""),
        ...rows.map(
          (m) =>
            new Option(
              m.name +
                " · " +
                (m.weightGiB != null ? m.weightGiB + "G" : "体积未知") +
                " · " +
                (m.role === "draft"
                  ? "草稿模型"
                  : m.format === "kat"
                    ? "自定义"
                    : m.format.toUpperCase()),
              m.path,
            ),
        ),
      );
      if (selected && !rows.some((m) => m.path === selected))
        $(id).add(new Option(selected.split("/").at(-1), selected));
      $(id).value = selected || "";
    };
    populate(
      "mainModelSelect",
      main,
      catalog.filter((m) => m.role !== "draft"),
    );
    populate(
      "draftPath",
      draft,
      catalog.filter((m) => m.role === "draft"),
    );
  }
  async function refreshModels() {
    const rows = await api("/console-api/models");
    catalog = rows;
    const signature = JSON.stringify([rows, profiles, templateData]);
    if (signature === modelSignature) return;
    modelSignature = signature;
    $("modelList").replaceChildren(
      ...rows.map((m) => {
        const row = document.createElement("article"),
          label = document.createElement("h3"),
          info = document.createElement("p"),
          select = document.createElement("select"),
          button = document.createElement("button");
        row.className = "profile-card";
        label.textContent = m.name;
        info.textContent =
          (m.format === "kat" ? "自定义" : m.format.toUpperCase()) +
          " · " +
          (m.role === "draft" ? "草稿模型" : m.modelType || "本地模型");
        select.setAttribute("aria-label", m.name + " 配置模板");
        const suitable = profiles.filter((p) => p.format === m.format);
        select.add(new Option("选择运行配置模板", ""));
        for (const p of suitable) select.add(new Option(profileLabel(p), p.id));
        for (const t of templateData.personal.filter(
          (t) => t.profile.format === m.format,
        ))
          select.add(new Option(t.name, "personal:" + t.id));
        button.textContent = "配置并使用";
        button.disabled = m.role === "draft";
        button.onclick = run(async () => {
          if (!select.value) throw Error("请选择适用的运行配置模板");
          const d = await api("/console-api/models/use", {
            model: m.id,
            template: select.value,
            start: false,
          });
          active = d.profile;
          await load();
          page("profiles");
          $("notice").textContent = "已创建配置，请确认参数后启动。";
        });
        const details = document.createElement("details"),
          summary = document.createElement("summary"),
          path = document.createElement("p");
        summary.textContent = "文件位置";
        path.textContent = m.path;
        details.append(summary, path);
        row.append(label, info);
        if (m.role !== "draft") row.append(select, button);
        const remove = document.createElement("button");
        remove.textContent = "删除模型";
        remove.className = "danger";
        remove.onclick = run(async () => {
          if (
            !confirm(
              "删除模型文件？\n\n" +
                m.name +
                "\n" +
                m.path +
                "\n\n此操作永久删除该目录中的文件，关联模板保留。",
            )
          )
            return;
          remove.disabled = true;
          try {
            await api("/console-api/models/delete", { id: m.id, path: m.path });
            modelSignature = "";
            await refreshModels();
            $("notice").textContent = "模型文件已删除。";
          } finally {
            remove.disabled = false;
          }
        });
        row.append(details, remove);
        return row;
      }),
    );
  }
  $("importProfile").onclick = () => {
    $("importProfileError").textContent = "";
    $("importProfileDialog").showModal();
  };
  $("cancelImportProfile").onclick = () => {
    $("importProfileDialog").close();
  };
  $("importProfileFile").onchange = run(async () => {
    const f = $("importProfileFile").files[0];
    if (!f) return;
    if (f.size > 1048576) throw Error("配置文件不能超过 1 MiB");
    $("importProfileJson").value = await f.text();
  });
  $("confirmImportProfile").onclick = () => {
    try {
      const doc = JSON.parse($("importProfileJson").value);
      if (doc.schema && doc.schema !== "sm75-profile-v1")
        throw Error("不支持的配置格式版本");
      const source = doc.profile || doc;
      if (
        !source ||
        !Array.isArray(source.args) ||
        !source.args.length ||
        source.args.some((x) => typeof x !== "string" || x.includes("\0"))
      )
        throw Error("配置需要包含模型及字符串参数数组 args");
      if (!["fp8", "awq", "kat"].includes(source.format))
        throw Error("权重格式应为 fp8、awq 或 kat");
      if (
        source.args.some((x) => x === "--api-key" || x.startsWith("--api-key="))
      )
        throw Error("请移除分享文件中的 API Key，在本机设置中填写");
      if (
        source.binds !== undefined &&
        (!Array.isArray(source.binds) ||
          source.binds.some((x) => typeof x !== "string"))
      )
        throw Error("挂载 binds 必须是字符串数组");
      if (
        source.env !== undefined &&
        (!source.env ||
          typeof source.env !== "object" ||
          Array.isArray(source.env) ||
          Object.values(source.env).some((x) => typeof x !== "string"))
      )
        throw Error("环境变量 env 必须是字符串映射");
      const env = Object.fromEntries(
        Object.entries(source.env || {}).filter(
          ([k]) => !/key|token|secret|password|authorization/i.test(k),
        ),
      );
      const p = {
        id:
          "personal-" +
          Array.from(crypto.getRandomValues(new Uint8Array(8)), (x) =>
            x.toString(16).padStart(2, "0"),
          ).join(""),
        name: String(source.name || "导入配置"),
        format: source.format,
        args: [...source.args],
        binds: source.binds || [],
        env,
        ...(source.lmcache === undefined ? {} : {lmcache: structuredClone(source.lmcache)}),
        port: 8000,
        cacheRoot: globalSettings.cacheRoot,
        backend: "docker",
      };
      for (const [flag, value] of [
        ["--host", "0.0.0.0"],
        ["--port", "8000"],
      ]) {
        const i = p.args.indexOf(flag);
        if (i >= 0) p.args[i + 1] = value;
        else p.args.push(flag, value);
      }
      for (const flag of [
        "--speculative-config",
        "--kv-transfer-config",
        "--compilation-config",
        "--hf-overrides",
      ]) {
        const i = p.args.indexOf(flag);
        if (i >= 0) JSON.parse(p.args[i + 1]);
      }
      validateLMCacheProfile(p);
      closeEditor();
      editProfile(p, false, true);
      $("profileDefault").checked = false;
      $("importProfileDialog").close();
      $("notice").textContent = "配置已载入，请确认本机模型与草稿模型后保存。";
    } catch (e) {
      $("importProfileError").textContent = e.message;
    }
  };
  async function shareProfile(p, download) {
    const clean = structuredClone(p);
    for (const k of Object.keys(clean.env || {}))
      if (/key|token|secret|password|authorization/i.test(k))
        delete clean.env[k];
    clean.args = clean.args.filter(
      (v, i, a) => v !== "--api-key" && a[i - 1] !== "--api-key",
    );
    const text = JSON.stringify(
      { schema: "sm75-profile-v1", profile: clean },
      null,
      2,
    );
    if (download) {
      const url = URL.createObjectURL(
        new Blob([text], { type: "application/json" }),
      );
      const a = document.createElement("a");
      a.href = url;
      a.download = p.id + ".json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } else {
      const input = document.createElement("textarea");
      input.value = text;
      globalThis.document.body.append(input);
      input.select();
      const ok = globalThis.document.execCommand("copy");
      input.remove();
      if (!ok) throw Error("复制未成功，请使用导出配置");
    }
    $("notice").textContent = download
      ? "完整配置已导出（不含 API Key）。"
      : "完整配置已复制（不含 API Key）。";
  }
  $("saveApiKey").onclick = run(async () => {
    await api("/console-api/api-access", { key: $("modelApiKey").value });
    $("modelApiKey").value = "";
    $("apiKeyStatus").textContent = "API Key 已保存，下次启动模型生效。";
  });
  function closeEditor() {
    const editor = $("profileEditor");
    editor.hidden = true;
    $("profiles").append(editor);
    for (const c of document.querySelectorAll(".profile-card.editing"))
      c.classList.remove("editing");
    document.querySelector(".import-draft")?.remove();
    profilesSignature = "";
  }
  async function editProfile(p, copy = false, imported = false) {
    if (!(await resolveDraft())) return;
    editorDirty = false;
    if (!imported) {
      active = p.id;
      $("active").value = active;
    $("active").title = profiles.find(p => p.id === active)?.name || "当前运行配置";
    }
    fillProfile(p);
    featureBackup = {};
    if (copy) {
      $("pid").value =
        "personal-" +
        Array.from(crypto.getRandomValues(new Uint8Array(8)), (x) =>
          x.toString(16).padStart(2, "0"),
        ).join("");
      $("profileName").value = profileLabel(p) + " · 副本";
      $("profileDefault").checked = false;
    }
    for (const c of document.querySelectorAll(".profile-card.editing"))
      c.classList.remove("editing");
    let card = Array.from($("profileCards").children).find(
      (c) => c.dataset.profile === p.id,
    );
    if (imported) {
      card = document.createElement("article");
      card.className = "profile-card import-draft";
      const title = document.createElement("h3");
      title.textContent = p.name + " · 待保存";
      card.append(title);
      $("profileCards").prepend(card);
    }
    if (card) {
      card.classList.add("editing");
    }
    $("profileList").hidden = false;
    $("profileEditor").hidden = false;
    requestAnimationFrame(() =>
      $("profileEditor").scrollIntoView({ behavior: "smooth", block: "start" }),
    );
    syncFeatures();
    page("profiles");
  }
  async function refreshProfileCards() {
    if (!$("profileEditor").hidden) return;
    const states = await Promise.all(
      profiles.map(async (p) => [
        p.id,
        await api("/console-api/profiles/" + p.id + "/status"),
      ]),
    );
    // A user may reopen the editor while the status requests are in flight.
    // Replacing its parent card now would detach the live form.
    if (!$("profileEditor").hidden) return;
    profileState = Object.fromEntries(states);
    const signature = JSON.stringify([profiles, profileState]);
    if (signature === profilesSignature) return;
    profilesSignature = signature;
    $("profileCards").replaceChildren(
      ...profiles.map((p) => {
        const card = document.createElement("article");
        card.className = "profile-card";
        card.dataset.profile = p.id;
        const title = document.createElement("h3");
        title.textContent = profileLabel(p);
        const arg = (k) => {
          const i = p.args.indexOf(k);
          return i >= 0 ? p.args[i + 1] : null;
        };
        let spec = {};
        try {
          spec = JSON.parse(arg("--speculative-config") || "{}");
        } catch {}
        const description = document.createElement("p");
        description.textContent = [
          p.args[0].split("/").at(-1),
          p.format.toUpperCase(),
          spec.method === "dflash"
            ? "DFLASH2"
            : spec.method === "mtp"
              ? "MTP"
              : "普通推理",
        ].join(" · ");
        const facts = document.createElement("p");
        facts.textContent =
          "上下文 " +
          (arg("--max-model-len") || "自动") +
          " · 并发 " +
          (arg("--max-num-seqs") || "自动") +
          " · TP " +
          (arg("--tensor-parallel-size") || "1") +
          " · 电源 " +
          (p.power?.mode === "pstate"
            ? "P-State 常驻"
            : "休眠 " + (arg("--auto-sleep-idle-timeout") || "0") + " 分");
        const st = document.createElement("p");
        st.className = profileState[p.id]?.running ? "ok" : "";
        st.textContent = profileState[p.id]?.running ? "运行中" : "未运行";
        const actions = document.createElement("div");
        actions.className = "actions";
        for (const [label, fn] of [
          [
            profileState[p.id]?.running ? "停止" : "启动",
            async () => {
              await api(
                "/console-api/profiles/" +
                  p.id +
                  "/" +
                  (profileState[p.id]?.running ? "stop" : "start"),
                {},
              );
              await refreshProfileCards();
            },
          ],
          ["编辑", () => editProfile(profiles.find(row => row.id === p.id))],
          ["复制", () => editProfile(profiles.find(row => row.id === p.id), true)],
          ["复制完整配置", () => shareProfile(profiles.find(row => row.id === p.id), false)],
          ["导出配置", () => shareProfile(profiles.find(row => row.id === p.id), true)],
        ]) {
          const b = document.createElement("button");
          b.textContent = label;
          b.onclick = run(fn);
          actions.append(b);
        }
        card.append(title, description, facts, st, actions);
        return card;
      }),
    );
  }
  $("backProfiles").onclick = async () => {
    if (await resolveDraft()) closeEditor();
  };
  $("createProfile").onclick = run(() => {
    const p = profiles.find((p) => p.id === active) || profiles[0];
    if (!p) throw Error("请先导入基础配置");
    editProfile(p, true);
  });
  $("localModelsTab").onclick = () => {
    $("localModelsView").hidden = false;
    $("downloadModelsView").hidden = true;
  };
  $("downloadModelsTab").onclick = () => {
    $("localModelsView").hidden = true;
    $("downloadModelsView").hidden = false;
  };
  $("taskToggle").onclick = () => {
    $("taskDrawer").hidden = !$("taskDrawer").hidden;
  };
  $("taskClose").onclick = () => {
    $("taskDrawer").hidden = true;
  };
  $("closeLogs").onclick = () => $("logDialog").close();
  function putArg(key, value) {
    $("args").value = JSON.stringify(setArg(JSON.parse($("args").value), key, value));
  }
  function argValue(key) {
    return readArgument(JSON.parse($("args").value), key);
  }
  function readCpuCompatibility(args) {
    try { return readContextSettings(args); }
    catch { return {cpuKvMode: "custom", cpuKvGiB: ""}; }
  }
  function syncFeatures() {
    const a = JSON.parse($("args").value),
      spec = argValue("--speculative-config");
    $("accelEnabled").checked = !!spec;
    let method = "base";
    try {
      method = JSON.parse(spec || "{}").method || "base";
    } catch {}
    $("chooseMtp").classList.toggle("selected", method === "mtp");
    $("chooseDflash").classList.toggle("selected", method === "dflash");
    $("draftChoiceLabel").hidden = method !== "dflash";
    $("sleepIdle").value =
      Number(argValue("--auto-sleep-idle-timeout")) > 0
        ? Number(argValue("--auto-sleep-idle-timeout"))
        : 30;
    const contextState = readCpuCompatibility(a);
    $("offloadEnabled").checked = contextState.cpuKvMode === "native";
    $("offloadGiB").value = contextState.cpuKvGiB || "8";
    $("offloadBody").hidden = true;
    $("prefixEnabled").checked = !a.includes("--no-enable-prefix-caching");
    const apiUrl = new URL(location.href);
    apiUrl.port = "8000";
    apiUrl.pathname = "/v1";
    apiUrl.search = "";
    apiUrl.hash = "";
    $("editorPreview").textContent =
      "模型 API：" +
      apiUrl.href +
      "\n编译缓存：" +
      globalSettings.cacheRoot +
      "\n" +
      JSON.stringify(a, null, 2);
  }
  function contextArgsForModel(modelPath) {
    // Validate/apply a pending form against its original model before changing
    // paths. The shared retargeter resets the old model-specific RoPE overrides.
    contextEditor.flush();
    return retargetContextModel(JSON.parse($("args").value), modelPath,
      catalog.find(model => model.path === modelPath)?.context || {});
  }
  $("mainModelSelect").onchange = run(() => {
    const previousModel = JSON.parse($("args").value)[0];
    try {
      $("args").value = JSON.stringify(contextArgsForModel($("mainModelSelect").value));
      contextEditor.refresh(true);
      renderParameters();
      fillCombo();
    } catch (error) {
      $("mainModelSelect").value = previousModel;
      throw error;
    }
  });
  $("chooseMtp").onclick = () => {
    $("variant").value = "mtp";
    $("setAcceleration").click();
  };
  $("chooseDflash").onclick = () => {
    $("variant").value = "dflash";
    $("draftChoiceLabel").hidden = false;
    if ($("draftPath").value) $("setAcceleration").click();
  };
  $("draftPath").onchange = () => {
    $("variant").value = "dflash";
    $("setAcceleration").click();
  };
  $("accelEnabled").addEventListener("change", () => {
    $("accelOptions").hidden = !$("accelEnabled").checked;
  });
  $("accelEnabled").onchange = () => {
    if (!$("accelEnabled").checked) {
      featureBackup.spec = argValue("--speculative-config");
      featureBackup.scheduler = argValue("--scheduler-cls");
      putArg("--speculative-config", null);
      putArg("--scheduler-cls", null);
    } else if (featureBackup.spec) {
      putArg("--speculative-config", featureBackup.spec);
      putArg("--scheduler-cls", featureBackup.scheduler);
    } else {
      putArg(
        "--speculative-config",
        JSON.stringify({ method: "mtp", num_speculative_tokens: 5 }),
      );
      putArg(
        "--scheduler-cls",
        "vllm.v1.core.sched.scheduler_sm75.SM75Scheduler",
      );
    }
    renderParameters();
  };
  $("sleepIdle").onchange = $("sleepTarget").onchange = () => {
    if ($("powerMode").value !== "sleep") return;
    putArg("--auto-sleep-idle-timeout", Number($("sleepIdle").value) || 30);
    putArg("--auto-sleep-offload-target", $("sleepTarget").value || "exit");
    renderParameters();
  };
  const COMBO_SEQS = { single: 4, coding: 8, multi: 16 };
  const COMBO_UTIL = { fp8: 0.92, awq: 0.9, kat: 0.9 };
  const COMBO_SCHED = "vllm.v1.core.sched.scheduler_sm75.SM75Scheduler";
  function comboModelRows() {
    return (Array.isArray(catalog) ? catalog : []).filter(
      (m) => m.role !== "draft",
    );
  }
  let comboDetected = false;
  async function comboDetect() {
    if (comboDetected) return;
    comboDetected = true;
    try {
      const h = await api("/console-api/hardware");
      const g = (h.gpus || [])[0];
      if (!localStorage.getItem("sm75-combo-vram") && g && g.totalMiB)
        $("comboVram").value = Math.round(g.totalMiB / 1024);
      if (
        !$("comboTp").dataset.touched &&
        (h.gpus || []).length &&
        !argValue("--tensor-parallel-size")
      )
        $("comboTp").value = (h.gpus || []).length;
      syncComboBodies();
    } catch {}
  }
  function syncComboBodies() {
    const tp = Number($("comboTp").value) || 0,
      vr = Number($("comboVram").value) || 0;
    $("comboVramTotal").textContent = tp && vr ? tp * vr + " GiB" : "—";
    const acc = $("comboAccel").value;
    $("comboMtpRow").hidden = acc !== "mtp";
    $("comboDflashRow").hidden = acc !== "dflash";
    $("offloadBody").hidden = true;
  }
  function comboFillModels() {
    const rows = comboModelRows(),
      sel = $("comboModel");
    sel.replaceChildren(
      ...rows.map(
        (m) =>
          new Option(
            m.name +
              " · " +
              (m.weightGiB != null ? m.weightGiB + "G" : "体积未知") +
              " · " +
              (m.format === "kat"
                ? "自定义"
                : String(m.format || "").toUpperCase()),
            m.path,
          ),
      ),
    );
    if (!rows.length) {
      sel.replaceChildren(new Option("请先到模型库下载/导入模型", ""));
      return;
    }
    let cur = "";
    try {
      cur = JSON.parse($("args").value)[0] || "";
    } catch {}
    sel.value = rows.some((m) => m.path === cur) ? cur : rows[0].path;
  }
  function fillCombo() {
    if (!$("comboTp")) return;
    comboFillModels();
    $("comboTp").value =
      argValue("--tensor-parallel-size") || $("comboTp").value || 4;
    if (!$("comboVram").dataset.touched)
      $("comboVram").value =
        localStorage.getItem("sm75-combo-vram") || $("comboVram").value || 16;
    const seqs = Number(argValue("--max-num-seqs") || 4);
    $("comboScene").value =
      seqs >= 16 ? "multi" : seqs >= 8 ? "coding" : "single";
    $("comboKv").value = argValue("--kv-cache-dtype") || "fp8_e4m3";
    let spec = {};
    try {
      spec = JSON.parse(argValue("--speculative-config") || "{}");
    } catch {}
    $("comboAccel").value = spec.method || "none";
    const tok = Number(spec.num_speculative_tokens) || 0;
    $("comboMtpTokens").value = String(
      spec.method === "mtp" && tok >= 1 && tok <= 8 ? tok : 5,
    );
    $("comboDraftTokens").value = String(
      spec.method === "dflash" && tok >= 1 && tok <= 8 ? tok : 7,
    );
    const cpu = readCpuCompatibility(JSON.parse($("args").value));
    $("comboCpuKv").value = cpu.cpuKvMode === "native" ? "on" : "off";
    $("offloadGiB").value = cpu.cpuKvGiB || "8";
    syncComboBodies();
    comboDetect();
  }
  $("comboTp").oninput = $("comboVram").oninput = () => {
    $("comboTp").dataset.touched = "1";
    $("comboVram").dataset.touched = "1";
    if ($("comboVram").value)
      localStorage.setItem("sm75-combo-vram", $("comboVram").value);
    syncComboBodies();
  };
  $("comboAccel").onchange = $("comboCpuKv").onchange = syncComboBodies;
  $("comboApply").onclick = run(async () => {
    if (!Array.isArray(catalog) || !catalog.length) {
      try {
        catalog = await api("/console-api/models");
      } catch {}
    }
    const tp = Math.max(1, Number($("comboTp").value) || 1),
      vram = Math.max(1, Number($("comboVram").value) || 1);
    const scene = $("comboScene").value,
      accel = $("comboAccel").value,
      kvq = $("comboKv").value;
    const m = (catalog || []).find((x) => x.path === $("comboModel").value);
    if (!m)
      throw Error("未选择模型：请先到模型库下载/导入模型，再在组合里选择");
    let draftPath = "";
    if (accel === "dflash") {
      draftPath = $("draftPath").value;
      const d = (catalog || []).find((x) => x.path === draftPath);
      if (!d) throw Error("请先到模型库下载 DFlash2 草稿模型，并在下方选中");
    }
    const seqs = COMBO_SEQS[scene],
      util = COMBO_UTIL[m.format === "fp8" ? "fp8" : "awq"],
      batch = 8192;
    let args = contextArgsForModel(m.path);
    const put = (key, value) => {args = setArg(args, key, value);};
    const previousSpec = JSON.parse(readArgument(args, "--speculative-config", "{}"));
    put("--tensor-parallel-size", tp);
    put("--max-num-seqs", seqs);
    put("--max-num-batched-tokens", batch);
    put("--gpu-memory-utilization", util);
    put("--kv-cache-dtype", kvq);
    if (accel === "none") {
      put("--speculative-config", null);
      put("--scheduler-cls", null);
    } else if (accel === "mtp") {
      put(
        "--speculative-config",
        JSON.stringify({
          ...(previousSpec.method === "mtp" ? previousSpec : {}),
          method: "mtp",
          num_speculative_tokens: Number($("comboMtpTokens").value) || 5,
        }),
      );
      put("--scheduler-cls", COMBO_SCHED);
    } else {
      put(
        "--speculative-config",
        JSON.stringify({
          ...(previousSpec.method === "dflash" && previousSpec.model === draftPath ? previousSpec : {}),
          method: "dflash",
          model: draftPath,
          num_speculative_tokens: Number($("comboDraftTokens").value) || 7,
          draft_tensor_parallel_size: tp,
          kv_cache_dtype: "auto",
          attention_backend: "FLASHINFER",
          draft_sample_method: "probabilistic",
        }),
      );
      put("--scheduler-cls", COMBO_SCHED);
    }
    $("args").value = JSON.stringify(args, null, 2);
    $("format").value = m.format === "fp8" ? "fp8" : "awq";
    contextEditor.refresh(true);
    renderParameters();
    syncFeatures();
    syncPower();
    fillCombo();
    $("notice").textContent =
      "组合已应用：" +
      tp +
      " 卡×" +
      vram +
      "G · " +
      m.name +
      " · " +
      { single: "单人", coding: "Coding", multi: "多人" }[scene] +
      " · 上下文保持 " +
      (readArgument(args, "--max-model-len") || "auto") +
      " · seqs " +
      seqs +
      (accel !== "none" ? " · " + accel.toUpperCase() : "");
  });
  function syncPower() {
    const p = $("powerMode").value === "pstate";
    $("pstateFields").hidden = !p;
    $("sleepFields").hidden = p;
  }
  $("powerMode").onchange = () => {
    if ($("powerMode").value === "sleep") {
      putArg("--auto-sleep-idle-timeout", Number($("sleepIdle").value) || 30);
      putArg("--auto-sleep-offload-target", $("sleepTarget").value || "exit");
    } else {
      putArg("--auto-sleep-idle-timeout", "0");
    }
    syncPower();
    renderParameters();
  };
  $("prefixEnabled").onchange = () => {
    putArg("--enable-prefix-caching", null);
    putArg("--no-enable-prefix-caching", null);
    putArg(
      $("prefixEnabled").checked
        ? "--enable-prefix-caching"
        : "--no-enable-prefix-caching",
      true,
    );
    renderParameters();
  };
  $("offloadEnabled").onchange = run(() => {
    const args = JSON.parse($("args").value), state = readContextSettings(args);
    if (state.cpuKvMode === "custom") throw Error("请在高级参数中编辑自定义缓存连接器");
    $("args").value = JSON.stringify(applyContextSettings(args, {...state,
      cpuKvMode: $("offloadEnabled").checked ? "native" : "off", cpuKvGiB: $("offloadGiB").value}));
    renderParameters();
  });
  $("offloadGiB").onchange = () => {
    if ($("offloadEnabled").checked) $("offloadEnabled").onchange();
  };

  let templateData = { recommended: [], personal: [] };
  function showSecurity(info) {
    syncLoginMode(info.accountConfigured);
    $("adminUsername").value = info.username;
    $("allowedNetworks").value = info.networks.join("\n");
    $("securityStatus").textContent = info.accountConfigured
      ? "管理员账号已启用，之后使用账号密码登录。"
      : "请先创建管理员账号；创建成功后，首次登录 token 将停止用于 Web 登录。";
    $("saveAdmin").textContent = info.accountConfigured ? "更新管理员账号" : "创建管理员账号";
    $("networkPeer").textContent = "当前连接地址：" + info.peer + "（白名单必须包含此地址）";
  }
  $("saveAdmin").onclick = run(async () => {
    if ($("adminPassword").value !== $("adminPasswordConfirm").value) throw Error("两次新密码不一致");
    const info = await api("/console-api/security", { action: "account", username: $("adminUsername").value.trim(), password: $("adminPassword").value, currentPassword: $("adminCurrentPassword").value });
    showSecurity(info);
    for (const id of ["adminPassword", "adminPasswordConfirm", "adminCurrentPassword"]) $(id).value = "";
    $("notice").textContent = "管理员账号已保存；其他登录会话已退出。";
  });
  $("saveNetworks").onclick = run(async () => {
    showSecurity(await api("/console-api/security", { action: "networks", networks: $("allowedNetworks").value.split(/[\n,]+/).map(s => s.trim()).filter(Boolean), currentPassword: $("adminCurrentPassword").value }));
    $("adminCurrentPassword").value = "";
    $("notice").textContent = "网段白名单已保存。";
  });
  async function loadSettings(s) {
    showSecurity(await api("/console-api/security"));
    await refreshSampling();
    $("provider").value = s.source;
    $("settingModels").value = s.modelRoot;
    $("settingCache").value = s.cacheRoot;
    $("defaultProfile").replaceChildren(
      new Option("不指定", ""),
      ...profiles.map((p) => new Option(profileLabel(p), p.id)),
    );
    $("defaultProfile").value = s.defaultProfile;
    $("autoStart").checked = s.autoStart;
    templateData = await api("/console-api/templates");
    $("templateSelect").replaceChildren(
      ...templateData.recommended.map(
        (t) => new Option(t.name, "base:" + t.id),
      ),
      ...templateData.personal.map(
        (t) => new Option(t.name + " · v" + t.revision, "personal:" + t.id),
      ),
    );
  }
  $("saveSettings").onclick = run(async () => {
    globalSettings = await api("/console-api/settings", {
      modelRoot: $("settingModels").value,
      cacheRoot: $("settingCache").value,
      defaultProfile: $("defaultProfile").value,
      autoStart: $("autoStart").checked,
    });
    $("notice").textContent =
      "设置已保存。模型下载与编译缓存统一使用全局目录，正在运行的模型下次启动时生效。";
  });
  $("registerModel").onclick = run(async () => {
    await api("/console-api/models/register", { path: $("localModel").value });
    await refreshModels();
  });
  function editedProfile() {
    if (contextEditor.flush()) renderParameters();
    if (lmcacheEditor.flush()) renderParameters();
    return validateLMCacheProfile({
      ...editorBase,
      name: $("profileName").value,
      id: $("pid").value,
      format: $("format").value,
      port: 8000,
      cacheRoot: globalSettings.cacheRoot,
      args: JSON.parse($("args").value),
      binds: JSON.parse($("binds").value),
      backend: editorBase?.backend || "docker",
      power: {
        mode: $("powerMode").value,
        idleSeconds: Math.max(1, Number($("pstateIdle").value) || 1),
        util: Number($("pstateUtil").value) || 0,
        confirm: Number($("pstateConfirm").value) || 0,
        low: Number($("pstateLow").value),
        high: Number($("pstateHigh").value),
        poll: Number($("pstatePoll").value) || 5,
        gpus: "0,1,2,3",
      },
    });
  }
  $("saveTemplate").onclick = run(async () => {
    await api("/console-api/templates", {
      name: $("templateName").value,
      profile: editedProfile(),
    });
    await loadSettings(await api("/console-api/settings"));
    $("notice").textContent = "个人模板已保存，包含当前自定义参数。";
  });
  $("applyTemplate").onclick = run(async () => {
    const [kind, id] = $("templateSelect").value.split(":");
    if (kind === "base") {
      contextEditor.flush();
      lmcacheEditor.flush();
      $("args").value = JSON.stringify(
        (
          await api("/console-api/templates/apply", {
            id,
            args: JSON.parse($("args").value),
          })
        ).args,
        null,
        2,
      );
    } else {
      const p = templateData.personal.find((t) => t.id === id).profile;
      editorBase = structuredClone(p);
      $("args").value = JSON.stringify(p.args, null, 2);
      $("binds").value = JSON.stringify(p.binds || [], null, 2);
      $("cacheRoot").value = globalSettings.cacheRoot || p.cacheRoot;
      $("format").value = p.format;
      fillProfilePowerFields(document, p);
      syncPower();
    }
    contextEditor.refresh(true);
    lmcacheEditor.refresh(true);
    renderParameters();
    page("profiles");
    $("notice").textContent = "模板已填入，请检查并保存运行配置。";
  });

  // Keep dashboard actions in the page toolbar while retaining their original handlers.
  let monitorControlObserver, monitorSizeObserver;
  let monitorSpecPending = false;
  $("monitorFrame").addEventListener("load", () => {
    const frame = $("monitorFrame"),
      doc = frame.contentDocument,
      bar = $("monitorControls");
    monitorControlObserver?.disconnect();
    monitorSizeObserver?.disconnect();
    bar.replaceChildren();
    const source = doc?.querySelector("header .controls");
    if (!source) return;
    const controls = source.cloneNode(true);
    bar.append(controls);
    for (const id of ["window", "pause", "specToggle"]) {
      const original = doc.getElementById(id),
        proxy = controls.querySelector("#" + id);
      proxy.id = "monitor-" + id;
      if (id === "window")
        proxy.onchange = () => {
          original.value = proxy.value;
          original.dispatchEvent(
            new frame.contentWindow.Event("change", { bubbles: true }),
          );
        };
      else if (id === "specToggle") proxy.onclick = run(async () => {
        if (monitorSpecPending) return;
        const specUrl = route("spec");
        monitorSpecPending = true;
        proxy.disabled = true;
        proxy.textContent = "切换中…";
        try {
          const state = await toggleMonitorSpec((payload) => api(specUrl, payload));
          const enabled = state.desired_enabled ?? state.enabled;
          original.textContent = (enabled ? "开" : "关") + (state.pending ? " · 待生效" : "");
          original.title = enabled ? "点击关闭投机解码（不卸显存）" : "点击开启投机解码";
          $("notice").textContent = state.pending ? "投机切换已提交，等待引擎生效。" : "投机解码已" + (enabled ? "开启。" : "关闭（草稿权重仍驻留显存）。");
        } catch (error) {
          throw Error("投机切换失败：" + error.message);
        } finally {
          monitorSpecPending = false;
          sync();
        }
      });
      else proxy.onclick = () => original.click();
    }
    const sync = () => {
      for (const id of ["status", "pause", "specToggle"]) {
        if (id === "specToggle" && monitorSpecPending) continue;
        const original = doc.getElementById(id),
          proxy = controls.querySelector(
            id === "status" ? "#status" : "#monitor-" + id,
          );
        proxy.textContent = original.textContent;
        proxy.className = original.className;
        proxy.title = original.title;
        if ("disabled" in original) proxy.disabled = original.disabled;
      }
    };
    monitorControlObserver = new MutationObserver(sync);
    monitorControlObserver.observe(source, {
      subtree: true,
      childList: true,
      attributes: true,
      characterData: true,
    });
    sync();
    const style = doc.createElement("style");
    style.textContent =
      "header{display:none!important}main>section{margin-top:12px}.card{padding:14px}.section-head{margin-bottom:8px}.secondary{margin-top:8px;padding-top:8px}.chart{height:200px}";
    doc.head.append(style);
    const sections = doc.querySelectorAll("main>section");
    if (sections.length > 1) sections[1].after(sections[0]);
    const fit = () => {
      frame.style.height =
        Math.ceil(
          doc.querySelector("main").getBoundingClientRect().height + 24,
        ) + "px";
    };
    monitorSizeObserver = new ResizeObserver(fit);
    monitorSizeObserver.observe(doc.querySelector("main"));
    fit();
  });
  const gpuHistory = new Map();

  async function refreshHardware() {
    const data = await api("/console-api/hardware"),
      box = $("hardwareCards");
    const el = (tag, cls, text) => {
      const n = document.createElement(tag);
      n.className = cls;
      if (text !== undefined) n.textContent = text;
      return n;
    };
    const num = (v, unit = "", digits = 0) =>
      Number.isFinite(v) ? v.toFixed(digits) + unit : "—";
    const rate = (v) =>
      Number.isFinite(v)
        ? v >= 1024
          ? (v / 1024).toFixed(2) + " MiB/s"
          : v.toFixed(0) + " KiB/s"
        : "—";
    const cards = data.gpus.map((g) => {
      const card = el("article", "gpu-card");
      card.setAttribute("aria-label", `GPU ${g.index} ${g.name}`);
      const head = el("div", "gpu-head"),
        title = el("div", "gpu-identity");
      title.title = g.bdf || "总线地址未知";
      title.tabIndex = 0;
      title.append(el("strong", "", `GPU ${g.index}`), el("span", "", g.name));
      const state = el(
        "span",
        "gpu-state" + (g.pstate === "P8" ? " sleeping" : ""),
        g.pstate || "未知",
      );
      head.append(title, state);
      card.append(head);
      const memory =
        Number.isFinite(g.usedMiB) && g.totalMiB > 0
          ? (100 * g.usedMiB) / g.totalMiB
          : null;
      const meters = el("div", "gpu-meters");
      for (const [name, value, caption, cls] of [
        ["GPU 利用率", g.util, num(g.util, "%"), "compute"],
        [
          "显存占用",
          memory,
          `${num(g.usedMiB != null ? g.usedMiB / 1024 : null, "", 1)} / ${num(g.totalMiB != null ? g.totalMiB / 1024 : null, " GiB", 1)}`,
          "memory",
        ],
      ]) {
        const m = el(
            "div",
            "gpu-meter " + cls + " gpu-level-" + gpuLevel(value),
          ),
          label = el("div", "gpu-meter-label");
        label.append(el("span", "", name), el("strong", "", caption));
        m.title =
          "占用：低于 50% 绿 · 50–74% 黄 · 75–89% 橙 · 90% 起红；颜色表示负载，不代表故障";
        const bar = el("div", "gpu-track"),
          fill = el("div", "gpu-fill");
        fill.style.width = Math.max(0, Math.min(100, value ?? 0)) + "%";
        if (value == null) bar.classList.add("unknown");
        bar.setAttribute("role", "progressbar");
        bar.setAttribute("aria-label", name);
        if (value != null) bar.setAttribute("aria-valuenow", value.toFixed(1));
        bar.setAttribute("aria-valuemin", "0");
        bar.setAttribute("aria-valuemax", "100");
        bar.append(fill);
        m.append(label, bar);
        meters.append(m);
      }
      card.append(meters);
      const stats = el("dl", "gpu-stats");
      for (const [label, value] of [
        ["功耗", num(g.power, " W", 1)],
        ["温度", num(g.temp, " °C")],
        ["核心频率", num(g.coreMHz, " MHz")],
        ["显存频率", num(g.memoryMHz, " MHz")],
        ["PCIe 当前", `${g.pcieGen ?? "—"} ×${g.pcieWidth ?? "—"}`],
        ["设备上限", `${g.pcieMaxGen ?? "—"} ×${g.pcieMaxWidth ?? "—"}`],
        ["RX · 进入 GPU", rate(g.rxKBps)],
        ["TX · 离开 GPU", rate(g.txKBps)],
      ]) {
        const item = el("div", "");
        item.append(
          el("dt", "", label),
          el(
            "dd",
            label === "温度"
              ? "gpu-temperature gpu-level-" + gpuLevel(g.temp, [60, 75, 85])
              : "",
            value,
          ),
        );
        if (label === "温度")
          item.title =
            "温度分级：低于 60°C 绿 · 60–74°C 黄 · 75–84°C 橙 · 85°C 起红";
        stats.append(item);
      }
      card.append(stats);
      const history = (gpuHistory.get(g.uuid || g.index) || []).filter(
        (p) => p.time >= data.time - 300,
      );
      if (history.at(-1)?.time !== data.time) {
        history.push({ time: data.time, util: g.util, memory });
        if (history.length > 60) history.shift();
      }
      gpuHistory.set(g.uuid || g.index, history);
      const chart = el("div", "gpu-history");
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 400 44");
      svg.setAttribute("preserveAspectRatio", "none");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", "GPU 利用率和显存占用趋势，最近五分钟");
      for (const [key, color] of [
        ["util", "#62e0b6"],
        ["memory", "#70bbff"],
      ]) {
        let points = [];
        const flush = () => {
          if (!points.length) return;
          const line = document.createElementNS(svg.namespaceURI, "polyline");
          line.setAttribute("points", points.join(" "));
          line.setAttribute("fill", "none");
          line.setAttribute("stroke", color);
          line.setAttribute("stroke-width", "1.8");
          svg.append(line);
          points = [];
        };
        history.forEach((p, i) => {
          if (i && p.time - history[i - 1].time > 12) flush();
          if (!Number.isFinite(p[key])) {
            flush();
            return;
          }
          points.push(
            `${400 * (1 - (data.time - p.time) / 300)},${42 - Math.max(0, Math.min(100, p[key])) * 0.4}`,
          );
        });
        flush();
      }
      chart.append(svg);
      const foot = el("div", "gpu-footer");
      chart.title = "绿：利用率 · 蓝：显存 · 最近 5 分钟";
      card.append(chart);
      return card;
    });
    box.replaceChildren(...cards);
    $("hardwareStamp").textContent =
      "每 5 秒更新 · " + new Date(data.time * 1000).toLocaleTimeString();
  }
  $("p2pCapability").onclick = run(async () => {
    const button = $("p2pCapability");
    button.disabled = true;
    button.textContent = "检测中…";
    try {
      const d = await api("/console-api/hardware?p2p=1");
      $("p2pResult").textContent =
        d.p2p.length === 0
          ? "P2P 无法判定：至少需要两张 GPU"
          : d.p2p.some((p) => p.read === null || p.write === null)
            ? "P2P 无法判定"
            : d.p2p.every((p) => p.read === 0 && p.write === 0)
              ? "P2P 有效 · 所有卡对驱动读写支持"
              : "P2P 部分或全部卡对不可用";
      function table(parent, heads, rows) {
        const t = document.createElement("table"),
          head = t.createTHead().insertRow();
        for (const text of heads) {
          const cell = document.createElement("th");
          cell.textContent = text;
          head.append(cell);
        }
        const body = t.createTBody();
        for (const row of rows) {
          const tr = body.insertRow();
          for (const text of row) {
            const td = tr.insertCell();
            td.textContent = String(text ?? "未知");
            if (text === "支持") td.className = "ok";
            if (text === "不支持") td.className = "bad";
          }
        }
        $(parent).replaceChildren(t);
      }
      $("topologyRead").textContent = (
        d.topology?.read?.output ||
        d.topology?.read?.error ||
        "P2P 读取查询不可用"
      ).replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
      $("topologyMatrix").textContent = (
        d.topology?.matrix?.output ||
        d.topology?.matrix?.error ||
        "拓扑查询不可用"
      ).replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
      $("topologyRaw").textContent = Object.values(d.topology || {})
        .map((v) => v.command + "\n" + (v.output || v.error))
        .join("\n\n")
        .replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
      $("p2pEvidence").textContent =
        "采样时间：" +
        new Date(d.time * 1000).toLocaleString() +
        "。P2P 表示驱动能力，尚未执行 CUDA 卡间带宽测试，不能视为 vLLM 正在使用 P2P 的证据。";
    } finally {
      button.disabled = false;
      button.textContent = "检测 P2P";
    }
  });

  function profileLabel(p) {
    if (p.name)
      return p.name
        .replaceAll("MTP5", "MTP")
        .replaceAll("草稿加速", "DFLASH2")
        .replaceAll("多令牌预测", "MTP");
    const a = p.args || [],
      i = a.indexOf("--speculative-config");
    let mode = "普通推理";
    try {
      const m = JSON.parse(a[i + 1] || "{}").method;
      mode = m === "dflash" ? "DFLASH2" : m === "mtp" ? "MTP" : "普通推理";
    } catch {}
    return (
      (p.id.includes("awq") ? "低显存长上下文" : "单人长上下文") + " · " + mode
    );
  }
  const paramNames = {
    "--tensor-parallel-size": "张量并行卡数",
    "--pipeline-parallel-size": "流水线并行数",
    "--max-model-len": "最大上下文长度",
    "--max-num-seqs": "最大并发请求",
    "--max-num-batched-tokens": "批处理 Token 上限",
    "--gpu-memory-utilization": "显存利用比例",
    "--served-model-name": "对外模型名称",
    "--dtype": "计算精度",
    "--kv-cache-dtype": "KV 缓存精度",
    "--kv-cache-memory-bytes": "GPU KV 缓存字节数",
    "--block-size": "缓存块大小",
    "--auto-sleep-idle-timeout": "空闲休眠时间（分钟）",
    "--auto-sleep-offload-target": "休眠模式",
    "--speculative-config": "推测解码",
    "--kv-transfer-config": "CPU KV 缓存",
    "--compilation-config": "编译与计算图",
    "--enable-prefix-caching": "前缀缓存",
    "--enable-auto-tool-choice": "自动工具选择",
    "--tool-call-parser": "工具调用解析器",
    "--reasoning-parser": "推理内容解析器",
    "--scheduler-cls": "调度器",
    "--hf-overrides": "模型配置覆盖",
    "--disable-custom-all-reduce": "关闭原生自定义规约",
    "--attention-backend": "注意力后端",
    "--generation-config": "生成配置来源",
    "--mamba-cache-mode": "混合架构缓存模式",
    "--async-scheduling": "异步调度",
    "--port": "模型 API 端口",
    "--host": "监听地址",
  };
  const leafNames = {
    method: "加速方式",
    model: "草稿模型路径",
    num_speculative_tokens: "推测 Token 数",
    draft_tensor_parallel_size: "草稿并行卡数",
    max_model_len: "最大上下文",
    kv_cache_dtype: "缓存精度",
    attention_backend: "注意力后端",
    draft_sample_method: "草稿采样方式",
    kv_connector: "缓存连接器",
    kv_role: "缓存角色",
    kv_connector_extra_config: "缓存设置",
    cpu_bytes_to_use: "CPU 缓存容量（字节）",
    spec_name: "缓存实现",
    cudagraph_capture_sizes: "计算图捕获批次",
    cudagraph_mode: "计算图模式",
    dtype: "计算精度",
  };
  function readParamRows() {
    const args = JSON.parse($("args").value),
      rows = [];
    for (let i = 1; i < args.length; ) {
      const token = args[i++], equal = token.indexOf("="),
        key = equal > 0 ? token.slice(0, equal) : token,
        values = equal > 0 ? [token.slice(equal + 1)] : [];
      while (i < args.length && !/^--?[a-zA-Z]/.test(args[i]))
        values.push(args[i++]);
      rows.push({ key, values });
    }
    return { model: args[0] || "", rows };
  }
  function renderParameters() {
    const state = readParamRows(),
      box = $("parameterFields");
    const specRow = state.rows.find((r) => r.key === "--speculative-config");
    let spec = {};
    try {
      spec = JSON.parse(specRow?.values[0] || "{}");
    } catch {}
    $("variant").value = spec.method || "base";
    syncModelChoices(state.model, spec.model);
    $("draftPath").value = spec.model || "";
    box.replaceChildren();
    const sections = new Map();
    function section(name, open = true) {
      if (sections.has(name)) return sections.get(name);
      const d = document.createElement("details"),
        title = document.createElement("summary"),
        grid = document.createElement("div");
      d.className = "parameter-section";
      d.open = open;
      title.textContent = name;
      grid.className = "parameter-grid";
      d.append(title, grid);
      box.append(d);
      sections.set(name, grid);
      return grid;
    }
    const core = section("模型与资源");
    const commit = () => {
      $("args").value = JSON.stringify([
        state.model,
        ...state.rows
          .filter((r) => !r.disabled)
          .flatMap((r) => [r.key, ...r.values]),
      ]);
      contextEditor.refresh();
      lmcacheEditor.refresh();
    };
    const toolKeys = new Set([
      "--enable-auto-tool-choice", "--no-enable-auto-tool-choice",
      "--tool-call-parser", "--tool-parser-plugin", "--chat-template",
      "--chat-template-content-format", "--reasoning-parser",
    ]);
    const tools = section("工具调用与对话解析");
    const note = document.createElement("p");
    note.className = "wide";
    note.textContent = "解析器需与模型及当前引擎支持列表匹配；留空沿用引擎默认。保存后在模型下次启动时生效。";
    tools.append(note);
    for (const [key, label, hint] of [
      ["--enable-auto-tool-choice", "自动工具选择", ""],
      ["--tool-call-parser", "工具调用解析器", "例如 qwen3_coder"],
      ["--reasoning-parser", "推理内容解析器", "填写模型支持的解析器"],
      ["--chat-template", "对话模板", "引擎容器内的模板路径或模板内容"],
      ["--chat-template-content-format", "模板消息内容格式", "auto / string / openai"],
      ["--tool-parser-plugin", "工具解析器插件", "引擎容器内已安装的插件路径"],
    ]) {
      const row = state.rows.find((r) => r.key.split("=")[0] === key);
      const control = document.createElement("input");
      const labelNode = document.createElement("label");
      labelNode.textContent = label;
      control.setAttribute("aria-label", label);
      if (key === "--enable-auto-tool-choice") {
        control.type = "checkbox";
        control.setAttribute("role", "switch");
        control.checked = !!row && !state.rows.some((r) => r.key === "--no-enable-auto-tool-choice");
      } else {
        control.type = "text";
        control.value = row ? (row.key.includes("=") ? row.key.slice(row.key.indexOf("=") + 1) : row.values[0] || "") : "";
        control.placeholder = hint;
      }
      control.onchange = () => {
        state.rows = state.rows.filter((r) => r.key.split("=")[0] !== key &&
          !(key === "--enable-auto-tool-choice" && r.key === "--no-enable-auto-tool-choice"));
        if (control.type === "checkbox") {
          if (control.checked) state.rows.push({key, values: []});
        } else if (control.value.trim()) {
          state.rows.push({key, values: [control.value]});
        }
        commit();
        editorDirty = true;
      };
      labelNode.append(control);
      tools.append(labelNode);
    }
    function leaf(parent, label, value, set) {
      if (value && typeof value === "object" && !Array.isArray(value)) {
        for (const [k, v] of Object.entries(value)) {
          if (label === "推测解码" && ["method", "model"].includes(k)) continue;
          leaf(parent, label + " · " + (leafNames[k] || k), v, (x) => {
            value[k] = x;
            set(value);
          });
        }
        return;
      }
      const bytes =
        label === "GPU KV 缓存字节数" || label.endsWith("CPU 缓存容量（字节）");
      const l = document.createElement("label");
      l.textContent = bytes
        ? label
            .replace("GPU KV 缓存字节数", "GPU KV 缓存容量（GiB）")
            .replace("CPU 缓存容量（字节）", "CPU 缓存容量（GiB）")
        : label;
      let input;
      const options = label.endsWith("加速方式")
        ? ["dflash", "mtp"]
        : label === "休眠模式"
          ? ["exit", "reload", "cpu"]
          : label === "KV 缓存精度" || label.endsWith(" · 缓存精度")
            ? ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
            : label.endsWith("计算精度")
              ? ["auto", "float16", "bfloat16", "float32"]
              : label.endsWith("计算图模式")
                ? ["NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE"]
                : label.endsWith("草稿采样方式")
                  ? ["probabilistic", "greedy"]
                  : label.endsWith("缓存角色")
                    ? ["kv_both", "kv_producer", "kv_consumer"]
                    : label === "混合架构缓存模式"
                      ? ["none", "align", "all"]
                      : label.endsWith("注意力后端") ||
                          label.endsWith(" · backend")
                        ? ["FLASHINFER", "TORCH_SDPA", "TRITON_ATTN"]
                        : label === "生成配置来源"
                          ? ["vllm", "auto"]
                          : null;
      if (options) {
        input = document.createElement("select");
        for (const o of new Set([String(value), ...options]))
          input.add(
            new Option(
              {
                dflash: "DFLASH2",
                mtp: "多令牌预测（MTP）",
                exit: "退出引擎，按需唤醒",
                reload: "卸载后重新加载",
                cpu: "保留到内存",
              }[o] || o,
              o,
            ),
          );
        input.value = String(value);
      } else {
        input = document.createElement("input");
        input.type = typeof value === "boolean" ? "checkbox" : "text";
        if (input.type === "checkbox") {
          input.setAttribute("role", "switch");
          input.checked = value;
        } else
          input.value = Array.isArray(value)
            ? JSON.stringify(value)
            : String(value);
      }
      if (bytes) {
        input.type = "number";
        input.min = "0";
        input.step = "any";
        input.value = String(Number(value) / 1073741824);
        input.title = "1 GiB = 1,073,741,824 字节，支持小数";
      }
      input.onchange = run(() => {
        if (bytes) {
          const n = Number(input.value),
            b = Math.round(n * 1073741824);
          if (
            !input.value.trim() ||
            !Number.isFinite(n) ||
            n < 0 ||
            !Number.isSafeInteger(b)
          )
            throw Error("请填写有效的 GiB 容量");
          set(typeof value === "number" ? b : String(b));
          commit();
          return;
        }
        const x =
          typeof value === "boolean"
            ? input.checked
            : Array.isArray(value)
              ? JSON.parse(input.value)
              : typeof value === "number"
                ? Number(input.value)
                : input.value;
        if (typeof x === "number" && !Number.isFinite(x))
          throw Error("请填写有效数字");
        set(x);
        commit();
      });
      const scalar = bytes ? Number(value) / 1073741824 : Number(value);
      const numeric =
        !options &&
        typeof value !== "boolean" &&
        !Array.isArray(value) &&
        (typeof value === "number" ||
          (String(value).trim() !== "" && Number.isFinite(scalar)) ||
          /最大上下文/.test(label));
      const arrayPreset =
        Array.isArray(value) && label.endsWith("计算图捕获批次");
      if (numeric || arrayPreset) {
        let values = bytes
          ? [1, 2, 4, 8, 16, 32, 64]
          : /显存利用比例/.test(label)
            ? [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95]
            : /上下文|批处理 Token/.test(label)
              ? [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
              : /休眠时间/.test(label)
                ? [1, 5, 10, 15, 30, 60, 120]
                : /缓存块大小/.test(label)
                  ? [16, 32, 64, 128]
                  : [1, 2, 4, 8, 16, 32, 64, 128];
        if (arrayPreset)
          values = [1, 2, 4, 8, 16, 32, 64].map((n) => JSON.stringify([n]));
        const picker = document.createElement("select");
        picker.setAttribute("aria-label", l.textContent + " · 常用值");
        if (/最大上下文/.test(label)) picker.add(new Option("自动", "auto"));
        for (const n of values) picker.add(new Option(String(n), String(n)));
        picker.add(new Option("自定义", "custom"));
        const originalChange = input.onchange;
        input.onchange = run(() => {
          if (!arrayPreset) {
            const n = Number(input.value);
            if (!input.value.trim() || !Number.isFinite(n) || n < 0)
              throw Error("请填写有效的非负数值");
            if (!bytes && !/比例|factor|theta|partial_rotary/.test(label) && !Number.isInteger(n))
              throw Error("该参数需要整数");
            if (/显存利用比例/.test(label) && (n <= 0 || n > 1))
              throw Error("显存利用比例需要大于 0 且不超过 1");
          }
          originalChange();
        });
        const current = input.value;
        picker.value = Array.from(picker.options).some(
          (o) => o.value === current,
        )
          ? current
          : "custom";
        input.hidden = picker.value !== "custom";
        input.setAttribute("aria-label", l.textContent + " · 自定义");
        if (!arrayPreset) {
          input.type = "number";
          input.step = /比例|GiB/.test(l.textContent) ? "any" : "1";
          input.min = "0";
        }
        picker.onchange = () => {
          input.hidden = picker.value !== "custom";
          if (picker.value === "custom") {
            input.focus();
            return;
          }
          if (picker.value === "auto") {
            set("auto");
            commit();
            return;
          }
          input.value = picker.value;
          input.onchange();
        };
        const wrap = document.createElement("div");
        wrap.className = "parameter-choice";
        wrap.append(picker, input);
        l.append(wrap);
      } else l.append(input);
      parent.append(l);
    }

    state.rows.forEach((row) => {
      const key = row.key.split("=")[0];
      if (usesDedicatedCacheField(JSON.parse($("args").value), key)) return;
      if (toolKeys.has(key)) return;
      if (
        [
          "--port",
          "--host",
          "--enable-prefix-caching",
          "--no-enable-prefix-caching",
        ].includes(row.key)
      )
        return;
      const group = document.createElement("div");
      group.className = "parameter-row";
      const fields = document.createElement("div");
      fields.className = "parameter-inputs";
      const label = paramNames[row.key] || "自定义参数（" + row.key + "）";
      if (!row.values.length) {
        const l = document.createElement("label");
        l.textContent = label;
        const toggle = document.createElement("input");
        toggle.type = "checkbox";
        toggle.setAttribute("role", "switch");
        toggle.checked = true;
        toggle.onchange = () => {
          row.disabled = !toggle.checked;
          commit();
        };
        l.append(toggle);
        fields.append(l);
      } else if (row.values.length === 1) {
        let value = row.values[0];
        try {
          if (value.startsWith("{")) value = JSON.parse(value);
        } catch {}
        leaf(
          fields,
          label,
          value,
          (x) =>
            (row.values = [
              typeof x === "object" ? JSON.stringify(x) : String(x),
            ]),
        );
      } else
        row.values.forEach((v, i) =>
          leaf(fields, label + " · " + (i + 1), v, (x) => (row.values[i] = x)),
        );
      const remove = document.createElement("button");
      remove.textContent = "×";
      remove.title = "移除 " + label;
      remove.setAttribute("aria-label", "移除 " + label);
      remove.onclick = () => {
        state.rows.splice(state.rows.indexOf(row), 1);
        commit();
        renderParameters();
      };
      group.append(fields, remove);
      let target;
      if (/speculative|scheduler/.test(row.key)) target = section("推测解码");
      else if (/kv-|block-size|prefix|mamba/.test(row.key))
        target = section("KV 缓存");
      else if (/sleep/.test(row.key)) {
        if ($("powerMode").value !== "sleep") return;
        target = section("自动休眠");
      } else if (
        /compilation|hf-overrides|parser|tool|attention|generation|custom-all|async/.test(
          row.key,
        )
      )
        target = section("高级参数", false);
      else target = paramNames[row.key] ? core : section("高级参数", false);
      if (row.values[0]?.startsWith("{")) group.classList.add("wide");
      target.append(group);
    });
    syncFeatures();
    contextEditor.refresh();
    lmcacheEditor.refresh();
  }
  $("addParameter").onclick = run(() => {
    const key = $("newParamKey").value.trim();
    if (!/^--[a-z][a-z0-9-]*$/.test(key)) throw Error("参数名使用 --name 格式");
    const a = JSON.parse($("args").value);
    if (["--api-key", "--host", "--port", "--model"].includes(key))
      throw Error("该参数由服务或专用配置项管理，请在对应位置修改");
    if (readParamRows().rows.some((r) => r.key.split("=")[0] === key))
      throw Error("该参数已存在，请编辑现有参数");
    if ($("newParamValue").value.includes("\0")) throw Error("参数值不能包含空字符");
    a.push(key);
    if ($("newParamValue").value) a.push($("newParamValue").value);
    $("args").value = JSON.stringify(a);
    editorDirty = true;
    $("newParamKey").value = "";
    $("newParamValue").value = "";
    renderParameters();
    $("notice").textContent = "参数已添加，请保存运行配置；模型下次启动时生效。";
  });
  $("newProfile").onclick = () => {
    $("pid").value =
      "personal-" +
      Array.from(crypto.getRandomValues(new Uint8Array(8)), (x) =>
        x.toString(16).padStart(2, "0"),
      ).join("");
    $("profileName").value = "我的长上下文配置";
    $("apiPort").value = 8000;
  };
  let searchPageNumber = 1,
    searchGeneration = 0,
    downloadSelection = null;
  async function searchModelPage(page = 1) {
    const generation = ++searchGeneration,
      provider = $("provider").value,
      q = $("searchQuery").value.trim();
    $("searchResults").textContent = "搜索中…";
    $("downloadConfirm").hidden = true;
    downloadSelection = null;
    const result = await api(
      "/console-api/model-search?" + new URLSearchParams({ provider, q, page }),
    );
    if (generation !== searchGeneration) return;
    searchPageNumber = page;
    $("searchPage").textContent =
      `第 ${page} 页${result.total !== undefined ? " · 共 " + result.total + " 个模型" : ""}`;
    $("searchPrev").disabled = page === 1;
    $("searchNext").disabled = !result.more;
    $("searchResults").replaceChildren(
      ...result.items.map((m) => {
        const row = document.createElement("div");
        row.className = "model-result";
        const name = document.createElement("strong");
        name.textContent = m.id;
        const info = document.createElement("p");
        info.textContent = `${m.name !== m.id ? m.name + " · " : ""}下载 ${m.downloads ?? "未知"} · ${(m.tags || []).slice(0, 4).join(" / ")}`;
        const b = document.createElement("button");
        b.textContent = "选择下载";
        b.onclick = run(async () => {
          downloadSelection = { provider, repo: m.id };
          $("repo").value = m.id;
          $("downloadSummary").textContent =
            (provider === "modelscope" ? "ModelScope" : "Hugging Face") +
            " · " +
            m.id;
          const s = await api("/console-api/settings");
          $("downloadPath").textContent =
            "下载到：" +
            s.modelRoot +
            "/" +
            provider +
            "/" +
            m.id.replace("/", "--");
          $("downloadConfirm").hidden = false;
          $("downloadConfirm").scrollIntoView({ block: "nearest" });
        });
        row.append(name, info, b);
        return row;
      }),
    );
    if (!result.items.length)
      $("searchResults").textContent = "未找到模型，请更换关键词。";
  }
  $("searchModels").onclick = run(() => searchModelPage(1));
  $("searchQuery").onkeydown = (e) => {
    if (e.key === "Enter") run(() => searchModelPage(1))();
  };
  $("provider").onchange = run(async () => {
    ++searchGeneration;
    $("searchResults").replaceChildren();
    $("downloadConfirm").hidden = true;
    downloadSelection = null;
    if ($("searchQuery").value.trim()) await searchModelPage(1);
  });
  $("searchPrev").onclick = run(() => searchModelPage(searchPageNumber - 1));
  $("searchNext").onclick = run(() => searchModelPage(searchPageNumber + 1));
  $("cancelDownloadSelection").onclick = () => {
    $("downloadConfirm").hidden = true;
    downloadSelection = null;
  };

  $("setAcceleration").onclick = run(() => {
    const args = JSON.parse($("args").value);
    const put = (k, v) => {
      const i = args.indexOf(k);
      if (i >= 0) args.splice(i, 2);
      if (v !== null) args.push(k, v);
    };
    const mode = $("variant").value;
    if (mode === "base") {
      put("--speculative-config", null);
      put("--scheduler-cls", null);
    } else {
      const i = args.indexOf("--speculative-config"),
        old = i >= 0 ? JSON.parse(args[i + 1]) : {},
        same = old.method === mode;
      const spec = {
        ...(same ? old : {}),
        method: mode,
        num_speculative_tokens: same
          ? old.num_speculative_tokens
          : mode === "mtp"
            ? 5
            : 7,
      };
      if (mode === "dflash") {
        if (!$("draftPath").value) throw Error("请填写匹配的草稿模型路径");
        spec.model = $("draftPath").value;
        const ti = args.indexOf("--tensor-parallel-size");
        spec.draft_tensor_parallel_size = ti >= 0 ? Number(args[ti + 1]) : 1;
      }
      put("--speculative-config", JSON.stringify(spec));
      put("--scheduler-cls", "vllm.v1.core.sched.scheduler_sm75.SM75Scheduler");
    }
    $("args").value = JSON.stringify(args);
    renderParameters();
  });

  return () => {
    disposed = true;
    stopLiveSummary();
    headerSizeObserver.disconnect();
    headerContentObserver.disconnect();
    fontScaleObserver.disconnect();
    cancelAnimationFrame(headerLayoutFrame);
    ++authGeneration;
    stopLogs();
    globalThis.removeEventListener("beforeunload", beforeUnload);
    globalThis.removeEventListener("pageshow", onRestore);
    globalThis.removeEventListener("focus", onRestore);
    document.removeEventListener("sm75-native-settings", onNativeSettings);
    document.removeEventListener("sm75-native-navigate", onNativeNavigate);
    globalThis.removeEventListener("popstate", onPop);
    monitorSizeObserver?.disconnect();
    monitorControlObserver?.disconnect();
    clearInterval(refreshTimer);
    abort?.abort();
  };
}
if (globalThis.document.getElementById("login")) mountConsole();

// Capture the profile and cancellation signal before starting any upload.
export async function uploadChatImages(files, profile, signal, request = globalThis.fetch) {
  signal.throwIfAborted();
  if (files.length > 4) throw Error("每条消息最多发送 4 张图片");
  if (files.some(file => file.size > 20 * 1024 * 1024)) throw Error("单张图片不能超过 20 MiB");
  return Promise.all(files.map(async file => {
    const response = await request("/console-api/attachments?profile=" + encodeURIComponent(profile), {
      method: "POST", body: file, signal,
    });
    const payload = await response.json();
    if (!response.ok) throw Error(payload.error || "图片上传失败");
    signal.throwIfAborted();
    return payload.id;
  }));
}

// Read the engine state before toggling; never infer success from a button label.
export async function toggleMonitorSpec(request) {
  const before = await request();
  if (!before.spec_configured) throw Error("引擎未启用可切换的投机配置");
  if (before.pending) throw Error("上一次切换尚未生效，请稍后再试");
  const enabled = !(before.desired_enabled ?? before.enabled);
  await request({ enabled });
  const after = await request();
  if ((after.desired_enabled ?? after.enabled) !== enabled)
    throw Error("引擎状态未改变，请检查引擎日志");
  return after;
}
