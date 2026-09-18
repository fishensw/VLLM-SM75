import test from "node:test";
import assert from "node:assert/strict";
import { liveSummary } from "../live-summary.mjs";
import { metrics } from "../telemetry.mjs";

const now = 100000;
const point = (t, hits, queries, tokens = 100) => ({
  t,
  prefill: 120,
  decode: 24,
  kv: 37,
  m: metrics.parsePrometheus(
    `vllm:prefix_cache_hits_total ${hits}\nvllm:prefix_cache_queries_total ${queries}\nvllm:generation_tokens_total ${tokens}`,
  ),
});
const hw = {
  time: now / 1000,
  gpus: [
    { index: 0, temp: 60, power: 70, util: 20, usedMiB: 8192, totalMiB: 16384 },
    { index: 1, temp: 65, power: 80, util: 80, usedMiB: 12288, totalMiB: 32768 },
  ],
};
test("header uses recent cache deltas, maximum temperature and complete total power", () => {
  const data = liveSummary(
    [
      point(now - 90000, 0, 1),
      point(now - 10000, 100, 1000),
      point(now, 130, 1040),
    ],
    hw,
    { now },
  );
  assert.equal(data.cacheHit, 75);
  assert.equal(data.cacheSeconds, 10);
  assert.equal(data.temperature, 65);
  assert.equal(data.power, 150);
  assert.equal(data.utilization, 80);
  assert.equal(data.memory, 50);
  assert.equal(data.gpus[1].memory, 37.5);
  assert.equal(data.kv, 37);
  assert.equal(data.decode, 24);
});
test("stale, failed or missing engine samples never appear live; hardware stays independent", () => {
  for (const rows of [[], [point(now - 16000, 0, 0)]]) {
    const data = liveSummary(rows, hw, { now });
    assert.equal(data.fresh, false);
    assert.equal(data.decode, null);
    assert.equal(data.power, 150);
  }
  assert.equal(
    liveSummary([point(now, 0, 0)], hw, { now, failed: true }).kv,
    null,
  );
  assert.equal(liveSummary([], { ...hw, time: 1 }, { now }).power, null);
  assert.equal(liveSummary([], { ...hw, time: 1 }, { now }).utilization, null);
  assert.equal(liveSummary([], { ...hw, time: 1 }, { now }).memory, null);
});
test("no cache queries, resets and scrape gaps cannot produce a misleading hit rate", () => {
  const pairs = [
    [point(now - 5000, 10, 20), point(now, 10, 20)],
    [point(now - 5000, 10, 20, 100), point(now, 1, 2, 1)],
    [point(now - 5000, 10, 20), { ...point(now, 15, 30), gap: true }],
  ];
  for (const rows of pairs)
    assert.equal(liveSummary(rows, hw, { now }).cacheHit, null);
});
test("partial GPU readings do not masquerade as total power or maximum temperature", () => {
  const data = liveSummary(
    [],
    { ...hw, gpus: [hw.gpus[0], { index: 1, temp: null, power: null }] },
    { now },
  );
  assert.equal(data.power, null);
  assert.equal(data.temperature, null);
  assert.equal(data.utilization, null);
  assert.equal(data.memory, null);
});
