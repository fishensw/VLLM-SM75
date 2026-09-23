// Shared with the monitoring cards: load and temperature have different limits.
export function metricLevel(value, limits = [50, 75, 90]) {
  return !Number.isFinite(value) ? "unknown"
    : value >= limits[2] ? "high"
    : value >= limits[1] ? "elevated"
    : value >= limits[0] ? "medium" : "low";
}

export function engineStatus(data = {}) {
  if (data.running === false) return { text: "模型未运行", tone: "unknown", power: "" };
  if (data.running !== true) return { text: "状态待确认", tone: "unknown", power: "" };
  const idle = data.powerState?.mode === "idle";
  const power = data.powerMode !== "pstate" ? ""
    : idle ? "P8已进入"
    : data.powerState?.mode === "active" ? "驱动自动"
    : "P8待生效";
  return { text: data.fresh ? "模型运行" : "运行·待采样",
    tone: data.fresh ? "low" : "medium", power };
}

export function mountLiveSummary(
  host,
  { profile = () => "", enabled = () => true } = {},
) {
  const root = host.shadowRoot || host.attachShadow({ mode: "open" });
  root.innerHTML = `<style>
    :host{display:block;min-width:0;max-width:100%;color:var(--text,var(--dsw-alias-label-primary,inherit));font:400 calc(14px * var(--ui-fs,1))/1.5 var(--dsw-font-family,system-ui)}
    :host([hidden]){display:none}
    .bar{display:flex;flex-wrap:nowrap;align-items:center;gap:8px;min-height:36px;box-sizing:border-box;overflow:visible;white-space:nowrap}
    .status,.item{display:inline-flex;flex-wrap:nowrap;align-items:baseline;gap:2px;flex:0 0 auto;white-space:nowrap}
    .status{align-items:center;gap:5px;padding-right:6px;font-size:calc(15px * var(--ui-fs,1));border-right:1px solid var(--line,var(--dsw-alias-border-l3,#ccd3dc))}
    .dot{width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0}
    .label,.unit,.power{color:var(--muted,var(--dsw-alias-label-secondary,#8993a2))}
    .power:not(:empty)::before{content:'·';margin-right:5px}
    .power.idle{color:var(--green,var(--dsw-alias-state-success-primary,#087f69))}
    b{display:inline-block;min-inline-size:var(--value-width,3ch);font:700 calc(16px * var(--ui-fs,1))/1.5 var(--dsw-font-family,system-ui);font-variant-numeric:tabular-nums;text-align:right}
    .label,.unit{font-size:calc(14px * var(--ui-fs,1))}
    .item[data-metric=temperature]{--value-width:2ch}
    [data-level=low]{color:var(--gpu-low,var(--dsw-alias-state-success-primary,#087f69))}
    [data-level=medium]{color:var(--gpu-medium,var(--dsw-alias-state-warn-primary,#8a6500))}
    [data-level=elevated]{color:var(--gpu-elevated,#c2680a)}
    [data-level=high]{color:var(--gpu-high,var(--dsw-alias-state-error-primary,#c62828))}
    [data-level=unknown]{color:var(--muted,var(--dsw-alias-label-secondary,#8993a2))}
    :host-context(body[data-ds-dark-theme]){--gpu-low:#58e0bc;--gpu-medium:#f5d35c;--gpu-elevated:#ffab55;--gpu-high:#ff7373;--green:#62e0b6}
    .bar:focus-visible{outline:2px solid var(--blue,var(--dsw-alias-link,#4285c5));outline-offset:-2px}
  </style><div class="bar" role="group" tabindex="0" aria-label="引擎运行状态与实时指标"><span class="status"><span class="dot" aria-hidden="true"></span><span class="engine">状态待确认</span><span class="power"></span></span></div>`;
  const bar = root.querySelector(".bar"),
    status = root.querySelector(".status"),
    engine = root.querySelector(".engine"),
    power = root.querySelector(".power");
  const fields = [
    ["temperature", "GPU", "°C", "可见 GPU 最高温度"],
    ["power", "功耗", "W", "可见 GPU 总功耗"],
    ["utilization", "核心", "%", "可见 GPU 中最高核心利用率；非核心频率"],
    ["memory", "显存", "%", "可见 GPU 中最高显存占用率（已用 / 总量），与 KV 缓存占用不同"],
    [
      "prefill",
      "P",
      "tok/s",
      "最近采样间隔内本地计算输入 token 吞吐，非单请求速度",
    ],
    [
      "decode",
      "D",
      "tok/s",
      "最近采样间隔内引擎全局生成吞吐，非单请求速度",
    ],
    [
      "kv",
      "KV",
      "%",
      "引擎 GPU KV 缓存占用率，并非当前对话已用上下文比例",
    ],
    [
      "cacheHit",
      "命中",
      "%",
      "最近最多 60 秒前缀缓存查询命中率；无查询显示 —",
    ],
  ];
  const cells = fields.map(([key, label, unit, title]) => {
    const cell = document.createElement("span");
    cell.className = "item";
    cell.dataset.metric = key;
    cell.title = title;
    cell.innerHTML = `<span class="label"></span><b>—</b><span class="unit"></span>`;
    cell.querySelector(".label").textContent = label;
    cell.querySelector(".unit").textContent = unit;
    bar.append(cell);
    return { key, cell, value: cell.querySelector("b"), title };
  });
  let disposed = false,
    timer,
    controller,
    generation = 0;
  function render(data = {}) {
    const state = engineStatus(data);
    engine.textContent = state.text;
    status.dataset.level = state.tone;
    power.textContent = state.power;
    power.classList.toggle("idle", data.running === true && data.powerState?.mode === "idle");
    status.title = [data.profile, "引擎全局指标 · 每 5 秒更新",
      data.sampledAt ? "采样：" + new Date(data.sampledAt).toLocaleTimeString() : "等待采样",
      data.running === true && !data.fresh ? "等待新采样，当前吞吐与缓存数据不可用" : "",
      Number.isFinite(data.powerState?.idle_remaining) && data.powerState?.mode === "active"
        ? `距空闲 P8 约 ${Math.max(0, data.powerState.idle_remaining)} 秒` : ""].filter(Boolean).join("\n");
    for (const { key, cell, value, title } of cells) {
      value.textContent = Number.isFinite(data[key])
        ? data[key].toLocaleString(undefined, {
            maximumFractionDigits:
              key === "temperature" || key === "power" ? 0 : 1,
          })
        : "—";
      if (["temperature", "kv", "utilization", "memory"].includes(key))
        value.dataset.level = metricLevel(data[key], key === "temperature" ? [60, 75, 85] : [50, 75, 90]);
      else if (!Number.isFinite(data[key])) value.dataset.level = "unknown";
      else delete value.dataset.level;
      cell.title =
        title +
        (key === "cacheHit" && data.cacheSeconds
          ? `（实际窗口 ${data.cacheSeconds} 秒）`
          : "");
      if ((key === "temperature" || key === "power") && data.gpus?.length)
        cell.title +=
          "\n" +
          data.gpus
            .map(
              (g) =>
                `GPU ${g.index}: ${g.temp ?? "—"} °C / ${g.power ?? "—"} W`,
            )
            .join("\n");
      if ((key === "utilization" || key === "memory") && data.gpus?.length)
        cell.title += "\n" + data.gpus.map((g) =>
          `GPU ${g.index}: ${Number.isFinite(g[key]) ? g[key].toFixed(1) : "—"}%`
        ).join("\n");
    }
  }
  async function tick() {
    if (disposed) return;
    const current = generation;
    const active = enabled();
    if (active && !document.hidden) {
      const selected = profile();
      const request = new AbortController();
      controller = request;
      const timeout = setTimeout(() => request.abort(), 4000);
      try {
        const r = await fetch(
          "/console-api/live-summary" +
            (selected ? "?profile=" + encodeURIComponent(selected) : ""),
          { cache: "no-store", signal: request.signal },
        );
        if (!r.ok) throw Error("summary unavailable");
        const data = await r.json();
        if (!disposed && current === generation && selected === profile() && enabled()) render(data);
      } catch {
        if (!disposed && current === generation) render();
      } finally {
        clearTimeout(timeout);
      }
    } else render();
    if (!disposed && current === generation) timer = setTimeout(tick, 5000);
  }
  function refresh() {
    if (disposed) return;
    generation++;
    clearTimeout(timer);
    controller?.abort();
    render();
    void tick();
  }
  // The authenticated page owns visibility. Sampling only changes values.
  render();
  host.hidden = false;
  document.addEventListener("visibilitychange", refresh);
  tick();
  const stop = () => {
    disposed = true;
    generation++;
    clearTimeout(timer);
    controller?.abort();
    document.removeEventListener("visibilitychange", refresh);
  };
  stop.refresh = refresh;
  return stop;
}
