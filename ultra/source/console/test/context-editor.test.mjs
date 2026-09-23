import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {argValue, GiB} from '../public/context-config.js';

// Load the shipped app module, resolving its browser module URLs to local files.
// Its top-level auto-mount is inactive; the exported form binding is unchanged.
const appURL = new URL('../public/app.js', import.meta.url);
const source = fs.readFileSync(appURL, 'utf8')
  .replace('from "/live-summary.js"', `from ${JSON.stringify(new URL('../public/live-summary.js', import.meta.url).href)}`)
  .replace('from "./context-config.js"', `from ${JSON.stringify(new URL('../public/context-config.js', import.meta.url).href)}`)
  .replace('from "./lmcache-config.js"', `from ${JSON.stringify(new URL('../public/lmcache-config.js', import.meta.url).href)}`);
const previousDocument = globalThis.document;
let bindContextEditor;
try {
  globalThis.document = {getElementById: () => null};
  ({bindContextEditor} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64')));
} finally {
  if (previousDocument === undefined) delete globalThis.document;
  else globalThis.document = previousDocument;
}
const html = fs.readFileSync(new URL('../public/index.html', import.meta.url), 'utf8');
const metadata = {source: 'root', originalMaxPositionEmbeddings: 65536, maxModelLen: 65536, ropeTheta: 1000000};
function fixture(initial = ['model'], modelMetadata = metadata) {
  let args = structuredClone(initial), applied = 0, dirty = 0;
  const elements = new Map();
  for (const [, id] of html.matchAll(/\bid="([^"]+)"/g)) elements.set(id, {
    value: '', checked: false, hidden: false, disabled: false, textContent: '', listeners: {},
    custom: {disabled: true}, addEventListener(type, fn) {this.listeners[type] = fn;},
    querySelector() {return this.custom;},
  });
  const get = id => {assert(elements.has(id), `Shipped HTML must contain ${id}`); return elements.get(id);};
  const binding = bindContextEditor({getElementById: get}, {
    getArgs: () => args, setArgs: value => {args = value;}, getMetadata: () => modelMetadata,
    onApply: () => {applied++;}, onDirty: () => {dirty++;},
  });
  binding.refresh();
  return {get, binding, get args() {return args;}, get applied() {return applied;}, get dirty() {return dirty;},
    replace(value) {args = value;},
    input(id, value) {
      if (id === 'ctxYarnEnabled') get(id).checked = value;
      else get(id).value = String(value);
      get('contextSettings').listeners.input(); // no blur or change event
    },
  };
}

test('unblurred context form saves independent fractional GPU and CPU capacities and preserves unknown fields', () => {
  const kv = {kv_connector: 'OffloadingConnector', kv_role: 'kv_both', keep: 'connector-data',
    kv_connector_extra_config: {spec_name: 'CPUOffloadingSpec', cpu_bytes_to_use: 8 * GiB, keep: 17}};
  const f = fixture(['model', '--max-model-len=32768', '--gpu-memory-utilization', '0.9',
    '--kv-cache-memory-bytes=' + 3.25 * GiB, '--kv-transfer-config', JSON.stringify(kv),
    '--hf-overrides', '{"unrelated":true}', '--custom', 'unchanged']);
  assert.equal(f.get('ctxGpuKvGiB').value, '3.25');
  assert.equal(f.get('ctxCpuKvMode').value, 'native');
  f.input('ctxGpuKvGiB', '4.5');
  f.input('ctxCpuKvGiB', '10.25');
  assert.equal(f.applied, 0);
  assert(f.binding.flush());
  assert.equal(argValue(f.args, '--kv-cache-memory-bytes'), String(4.5 * GiB));
  const cpu = JSON.parse(argValue(f.args, '--kv-transfer-config'));
  assert.equal(cpu.kv_connector_extra_config.cpu_bytes_to_use, 10.25 * GiB);
  assert.equal(cpu.kv_connector_extra_config.keep, 17);
  assert.equal(cpu.keep, 'connector-data');
  assert.equal(argValue(f.args, '--gpu-memory-utilization'), '0.9');
  assert.equal(argValue(f.args, '--max-model-len'), '32768');
  assert.equal(JSON.parse(argValue(f.args, '--hf-overrides')).unrelated, true);
  assert.equal(argValue(f.args, '--custom'), 'unchanged');
  f.input('ctxGpuKvGiB', '');
  f.get('ctxApply').onclick();
  assert.equal(argValue(f.args, '--kv-cache-memory-bytes'), null);
  assert.equal(f.get('ctxCpuKvGiB').value, '10.25');
});

test('1M derives its multiplier from actual model length and preserves independent budgets and draft limits', () => {
  const draft = {method: 'dflash', model: 'draft', max_model_len: 32768};
  const f = fixture(['model', '--speculative-config', JSON.stringify(draft), '--kv-cache-memory-bytes', String(2 * GiB)]);
  f.get('ctxOneMillion').onclick();
  assert.equal(f.get('ctxMaxModelLen').value, '1048576');
  assert.equal(f.get('ctxYarnFactor').value, '16');
  assert.equal(f.get('ctxYarnOriginal').value, '65536');
  assert.equal(f.get('ctxGpuKvGiB').value, '2');
  assert.equal(f.get('ctxCpuKvMode').value, 'off');
  assert.deepEqual(JSON.parse(argValue(f.args, '--speculative-config')), draft);
  assert.match(f.get('contextWarnings').textContent, /草稿/);
  f.input('ctxYarnEnabled', false);
  f.binding.flush();
  assert.equal(argValue(f.args, '--max-model-len'), '65536');
  assert.equal(f.get('ctxYarnEnabled').checked, false);
});

test('unknown original length requires an explicit value and layered RoPE remains outside the generic wizard', () => {
  const f = fixture(['model'], {});
  assert.throws(() => f.get('ctxOneMillion').onclick(), /原生上下文长度/);
  assert.deepEqual(f.args, ['model']);
  f.input('ctxYarnEnabled', true);
  f.input('ctxYarnOriginal', '131072');
  f.get('ctxOneMillion').onclick();
  assert.equal(f.get('ctxYarnFactor').value, '8');
  const layered = fixture(['model'], {...metadata, layeredRope: true});
  assert.equal(layered.get('ctxYarnEnabled').disabled, true);
  assert.equal(layered.get('ctxOneMillion').disabled, true);
  assert.throws(() => layered.get('ctxOneMillion').onclick(), /分层 RoPE/);
  assert.deepEqual(layered.args, ['model']);
});

test('refresh preserves pending edits, while explicit profile/template replacement resets them', () => {
  const f = fixture(['model', '--max-model-len', '32768']);
  f.input('ctxMaxModelLen', '16384');
  f.replace([...f.args, '--seed', '42']);
  f.binding.refresh();
  assert.equal(f.get('ctxMaxModelLen').value, '16384');
  f.binding.flush();
  assert.equal(argValue(f.args, '--max-model-len'), '16384');
  assert.equal(argValue(f.args, '--seed'), '42');
  f.input('ctxMaxModelLen', '999');
  f.replace(['imported', '--max-model-len=8192', '--kv-cache-memory-bytes=' + 1.5 * GiB]);
  f.binding.refresh(true);
  assert.equal(f.get('ctxMaxModelLen').value, '8192');
  assert.equal(f.get('ctxGpuKvGiB').value, '1.5');
  assert.equal(f.binding.flush(), false);
});

test('custom connectors survive unrelated edits and invalid capacity never writes partial arguments', () => {
  const custom = {kv_connector: 'CustomConnector', opaque: {setting: 'retain'}};
  const f = fixture(['model', '--kv-transfer-config', JSON.stringify(custom)]);
  assert.equal(f.get('ctxCpuKvMode').value, 'custom');
  assert.equal(f.get('ctxCpuKvFields').hidden, true);
  f.input('ctxGpuKvGiB', '-1');
  const before = [...f.args];
  assert.throws(() => f.binding.flush(), /GPU KV 容量/);
  assert.deepEqual(f.args, before);
  f.input('ctxGpuKvGiB', '2.5');
  f.binding.flush();
  assert.deepEqual(JSON.parse(argValue(f.args, '--kv-transfer-config')), custom);
  assert.equal(argValue(f.args, '--kv-cache-memory-bytes'), String(2.5 * GiB));
});


test('invalid advanced JSON remains recoverable through parameter editing instead of writing stale form values', () => {
  const f = fixture(['model', '--hf-overrides', '{invalid']);
  assert.match(f.get('contextWarnings').textContent, /无法读取上下文配置/);
  f.input('ctxMaxModelLen', '131072');
  assert.throws(() => f.binding.flush(), /合法 JSON/);
  f.replace(['model', '--max-model-len', '8192', '--hf-overrides', '{"keep":true}']);
  f.binding.refresh();
  assert.equal(f.get('ctxMaxModelLen').value, '8192');
  assert.doesNotMatch(f.get('contextWarnings').textContent, /无法读取/);
  f.input('ctxMaxModelLen', '16384');
  f.binding.flush();
  assert.equal(argValue(f.args, '--max-model-len'), '16384');
  assert.deepEqual(JSON.parse(argValue(f.args, '--hf-overrides')), {keep: true});
});
