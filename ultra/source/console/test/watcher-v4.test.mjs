import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

// Exercise the shipped bundle's fold and registration without installing the
// Harness runtime. Zod validation and the host's cache storage are not mocked
// into the fold; the identity parser only isolates registration from Zod.
const bundle = readFileSync(new URL('../plugins/dsh-watcher/lib/dsh-watcher.js', import.meta.url), 'utf8');
function between(start, end) {
  const from = bundle.indexOf(start);
  const to = bundle.indexOf(end, from + start.length);
  assert(from >= 0 && to > from, `Missing bundle section: ${start}`);
  return bundle.slice(from, to);
}
const engine = between('//#region src/insights/engine.mjs', '//#region src/insights/projection.ts');
const registration = between('function installProjection(ctx)', '//#region src/dsh-watcher.ts');
const installProjection = new Function('createHash', 'stateSchema', 'viewSchema',
  `${engine}\n${registration}\nreturn installProjection;`)(createHash, { parse: value => value }, {});
let projection;
installProjection({
  sessionProjections: { register: unit => { projection = unit; } },
  inject() {},
});

function fixture() {
  let state = projection.init({ id: 'synthetic-session' }, 0);
  let seq = 0;
  const events = [];
  function emit(type, data, time = seq * 10) {
    const event = { type, data, time, seq: seq++ };
    events.push(event);
    state = projection.apply(state, event);
    return event;
  }
  emit('request/header', { header: { config: { provider: 'local', model: 'fixture' } } });
  emit('turn/start', { turn: 1 });
  function result(format, { failed = true, content = 'synthetic tool result', error, args = { command: 'synthetic command' } } = {}) {
    const step = seq;
    const callId = `call-${step}`;
    const start = seq * 10;
    const call = emit('tool/call', { turn: 1, step, callId, name: 'bash', arguments: args }, start);
    const resultContent = [{ type: 'text', text: content }];
    const result = { toolCallId: callId, content: resultContent, isError: failed };
    const message = format === 4
      ? { id: `result-${step}`, role: 'tool', source: { kind: 'tool', callId }, ...result }
      : { id: `result-${step}`, role: 'user', source: { kind: 'tool', callId }, content: [{ type: 'tool-result', ...result }] };
    const event = emit('tool/result', { turn: 1, step, message, ...(error ? { error } : {}) }, start + 25);
    return { call, event };
  }
  return { get state() { return state; }, events, emit, result };
}

for (const format of [3, 4]) {
  test(`V${format} tool errors without error metadata count in every statistics scope`, () => {
    const f = fixture();
    f.result(format);
    assert.equal(f.state.tools.length, 0);
    for (const stats of [f.state.totals, f.state.models[0], f.state.turn.stats]) {
      assert.equal(stats.tools, 1);
      assert.equal(stats.toolErrors, 1);
      assert.equal(stats.toolMs, 25);
      assert.equal(stats.bashMs, 25);
    }
    assert.match(f.state.previousFailure.resultHash, /^[a-f0-9]{64}$/);
    assert.equal(projection.wire.view(f.state).totals.toolErrors, 1);
  });
}

test('V3 and V4 results have the same failure fingerprint across migrated logs', () => {
  const f = fixture();
  const first = f.result(3);
  const hash = f.state.previousFailure.resultHash;
  f.result(4);
  assert.equal(f.state.previousFailure.resultHash, hash);
  f.result(4);
  assert.equal(f.state.totals.toolErrors, 3);
  assert.equal(f.state.findingCount, 1);
  assert.equal(f.state.findings[0].kind, 'repeated-failure');
  assert.equal(f.state.findings[0].seqs[0], first.call.seq);
  assert.equal(f.state.findings[0].seqs.length, 3);
});

test('changed error results and successful results break the repeated failure chain', () => {
  const f = fixture();
  f.result(4);
  f.result(4);
  f.result(4, { content: 'a distinct error result' });
  assert.equal(f.state.previousFailure.seqs.length, 1);
  assert.equal(f.state.findingCount, 0);
  f.result(4, { failed: false });
  assert.equal(f.state.previousFailure, null);
  f.result(4);
  f.result(4);
  assert.equal(f.state.previousFailure.seqs.length, 2);
  assert.equal(f.state.findingCount, 0);
  assert.equal(f.state.totals.toolErrors, 5);
});

test('legacy error metadata remains supported and participates in the failure fingerprint', () => {
  const f = fixture();
  f.result(3, { failed: false, error: { code: 'E_ONE' } });
  const firstHash = f.state.previousFailure.resultHash;
  assert.equal(f.state.totals.toolErrors, 1);
  f.result(3, { failed: false, error: { code: 'E_TWO' } });
  assert.notEqual(f.state.previousFailure.resultHash, firstHash);
  assert.equal(f.state.previousFailure.seqs.length, 1);
});

test('duplicate events cannot count the same V4 result twice', () => {
  const f = fixture();
  const { event } = f.result(4);
  assert.equal(projection.apply(f.state, event), f.state);
  assert.equal(f.state.totals.toolErrors, 1);
});

test('persisted state and wire views retain only tool fingerprints, never prompt or tool bodies', () => {
  const f = fixture();
  const prompt = 'PRIVATE_SYNTHETIC_PROMPT_MARKER';
  const argument = 'PRIVATE_SYNTHETIC_TOOL_ARGUMENT_MARKER';
  const result = 'PRIVATE_SYNTHETIC_TOOL_RESULT_MARKER';
  f.emit('user/message', { message: { role: 'user', content: [{ type: 'text', text: prompt }] } });
  for (let i = 0; i < 3; i++) f.result(4, { args: { command: argument }, content: result });
  const persisted = JSON.stringify(f.state);
  const wire = JSON.stringify(projection.wire.view(f.state));
  for (const marker of [prompt, argument, result]) {
    assert(!persisted.includes(marker));
    assert(!wire.includes(marker));
  }
  assert.equal(f.state.findingCount, 1);
  assert.match(f.state.previousFailure.signature, /^[a-f0-9]{64}$/);
  f.emit('user/message', { message: { role: 'user', content: [{ type: 'text', text: prompt }] } });
  assert.equal(f.state.previousFailure, null);
});

test('projection version invalidates V3-only cached folds while preserving the wire schema', () => {
  // Harness uses stateVersion to discard persisted rows and replay the log.
  // Verify the actual registered unit opts into that invalidation and replay
  // produces correct V4 statistics; this does not emulate cache storage.
  assert.equal(projection.key, 'watcherInsights');
  assert.equal(projection.stateVersion, 3);
  assert.notEqual(projection.stateVersion, 1);
  const f = fixture();
  f.result(4);
  f.result(4);
  f.result(4);
  const replayed = f.events.reduce(projection.apply, projection.init({ id: 'synthetic-session' }, 0));
  assert.deepEqual(replayed, f.state);
  assert.equal(replayed.totals.toolErrors, 3);
  assert.equal(replayed.findingCount, 1);
  assert.equal(projection.wire.view(replayed).version, 1);
});


test('native header order refreshes the current turn route after turn and step start', () => {
  let state = projection.init({ id: 'native-order' }, 0);
  let seq = 0;
  const emit = (type, data) => {
    state = projection.apply(state, { type, data, seq: seq++, time: seq * 10 });
  };
  emit('turn/start', { turn: 1 });
  emit('step/start', { turn: 1, step: 1 });
  assert.equal(state.turn.route.model, 'unknown');
  emit('request/header', { header: { config: { provider: 'sm75-local', model: 'audit-model', reasoningEffort: 'high' } } });
  assert.deepEqual(state.turn.route, { provider: 'sm75-local', model: 'audit-model', effort: 'high' });
  assert.deepEqual(state.open.route, state.turn.route);
  emit('assistant/message', { turn: 1, step: 1, message: { source: { provider: 'sm75-local', model: 'audit-model' } }, usage: { inputTokens: 12, outputTokens: 8 } });
  emit('turn/end', { turn: 1 });
  assert.equal(projection.wire.view(state).turn.route.model, 'audit-model');
  assert.equal(state.models[0].model, 'audit-model');
  assert.equal(state.models[0].calls, 1);
  assert.equal(state.totals.tokens, 20);
});

test('settled source corrects the current turn route when no request header was available', () => {
  let state = projection.init({ id: 'source-route' }, 0);
  const events = [
    { type: 'turn/start', data: { turn: 1 } },
    { type: 'step/start', data: { turn: 1, step: 1 } },
    { type: 'assistant/message', data: { turn: 1, step: 1, message: { source: { provider: 'local', model: 'source-model' } } } },
  ];
  for (const [seq, event] of events.entries()) state = projection.apply(state, { ...event, seq, time: seq * 10 });
  assert.deepEqual(state.turn.route, { provider: 'local', model: 'source-model' });
  assert.equal(state.models[0].model, state.turn.route.model);
});
