import test from "node:test";
import assert from "node:assert/strict";
import { engineStatus, metricLevel, mountLiveSummary } from "../public/live-summary.js";

test("compact engine and power status still distinguish stale sampling", () => {
  assert.deepEqual(engineStatus({running: true, fresh: true, powerMode: "pstate", powerState: {mode: "idle"}}),
    {text: "模型运行", tone: "low", power: "P8已进入"});
  assert.deepEqual(engineStatus({running: true, fresh: false, powerMode: "pstate", powerState: {mode: "active"}}),
    {text: "运行·待采样", tone: "medium", power: "驱动自动"});
  assert.equal(engineStatus({running: true, fresh: true, powerMode: "pstate"}).power, "P8待生效");
  assert.equal(engineStatus({running: true, fresh: true, powerMode: "sleep"}).power, "");
  assert.deepEqual(engineStatus({running: false, powerMode: "pstate", powerState: {mode: "idle"}}),
    {text: "模型未运行", tone: "unknown", power: ""});
  assert.deepEqual(engineStatus(), {text: "状态待确认", tone: "unknown", power: ""});
});

test("metric colors retain the load and temperature thresholds", () => {
  assert.deepEqual([null, 49.9, 50, 75, 90].map(v => metricLevel(v)), ["unknown", "low", "medium", "elevated", "high"]);
  assert.deepEqual([59.9, 60, 75, 85].map(v => metricLevel(v, [60, 75, 85])), ["low", "medium", "elevated", "high"]);
});

function element() {
  return {
    dataset: {}, children: [], parts: {}, textContent: "", title: "",
    classList: {toggle() {}},
    append(node) {this.children.push(node);},
    querySelector(key) {return this.parts[key];},
    set innerHTML(value) {
      this.html = value;
      if (value.includes('class="bar"')) {
        for (const key of [".bar", ".status", ".engine", ".power"]) this.parts[key] = element();
      } else {
        for (const key of [".label", "b", ".unit"]) this.parts[key] = element();
      }
    },
  };
}

test("sampling updates all eight metrics in place and retains their units and unavailable state", async () => {
  const previousDocument = globalThis.document, previousFetch = globalThis.fetch;
  const listeners = new Map(), requests = [];
  const root = element(), host = {shadowRoot: root, hidden: true};
  let sample = {running: true, fresh: true, temperature: 65, power: 280, utilization: 90,
    memory: 50.5, prefill: 1234.5, decode: 24.6, kv: 75, cacheHit: 80, powerMode: "pstate", powerState: {mode: "active"}};
  let stop;
  try {
    globalThis.document = {hidden: false, createElement: element,
      addEventListener: (name, fn) => listeners.set(name, fn),
      removeEventListener: name => listeners.delete(name)};
    globalThis.fetch = async (url, options) => {requests.push({url, options}); return {ok: true, json: async () => sample};};
    stop = mountLiveSummary(host, {profile: () => "active model"});
    await new Promise(setImmediate);
    const cells = root.parts[".bar"].children;
    assert.equal(host.hidden, false);
    assert.equal(requests[0].url, "/console-api/live-summary?profile=active%20model");
    assert.equal(requests[0].options.cache, "no-store");
    assert.deepEqual(cells.map(cell => cell.dataset.metric), ["temperature", "power", "utilization", "memory", "prefill", "decode", "kv", "cacheHit"]);
    assert.deepEqual(cells.map(cell => cell.parts[".unit"].textContent), ["°C", "W", "%", "%", "tok/s", "tok/s", "%", "%"]);
    assert.equal(cells[4].parts.b.textContent, (1234.5).toLocaleString(undefined, {maximumFractionDigits: 1}));
    assert.equal(cells[0].parts.b.dataset.level, "medium");
    assert.equal(cells[2].parts.b.dataset.level, "high");
    assert.equal(cells[6].parts.b.dataset.level, "elevated");
    sample = {running: true, fresh: false, power: 98, sampledAt: 1};
    stop.refresh();
    await new Promise(setImmediate);
    assert.equal(root.parts[".bar"].children, cells);
    assert.equal(cells.length, 8);
    assert.equal(cells[1].parts.b.textContent, "98");
    assert.equal(cells[4].parts.b.textContent, "—");
    assert.equal(cells[4].parts.b.dataset.level, "unknown");
    assert.equal(root.parts[".engine"].textContent, "运行·待采样");
    assert.match(root.parts[".status"].title, /等待新采样/);
  } finally {
    stop?.();
    if (previousDocument === undefined) delete globalThis.document; else globalThis.document = previousDocument;
    globalThis.fetch = previousFetch;
  }
  assert.equal(listeners.size, 0);
});
