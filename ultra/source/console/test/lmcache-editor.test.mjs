import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {argValue, GiB} from '../public/context-config.js';
import {validateLMCacheProfile} from '../public/lmcache-config.js';

const source = fs.readFileSync(new URL('../public/app.js', import.meta.url), 'utf8')
  .replace('from "/live-summary.js"', `from ${JSON.stringify(new URL('../public/live-summary.js', import.meta.url).href)}`)
  .replace('from "./context-config.js"', `from ${JSON.stringify(new URL('../public/context-config.js', import.meta.url).href)}`)
  .replace('from "./lmcache-config.js"', `from ${JSON.stringify(new URL('../public/lmcache-config.js', import.meta.url).href)}`);
const previousDocument = globalThis.document;
let bindLMCacheEditor, bindContextEditor;
try {
  globalThis.document = {getElementById: () => null};
  ({bindLMCacheEditor, bindContextEditor} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64')));
} finally {
  if (previousDocument === undefined) delete globalThis.document;
  else globalThis.document = previousDocument;
}
const html = fs.readFileSync(new URL('../public/index.html', import.meta.url), 'utf8');
const native = {kv_connector: 'OffloadingConnector', kv_role: 'kv_both',
  kv_connector_extra_config: {spec_name: 'CPUOffloadingSpec', cpu_bytes_to_use: 8 * GiB}};
function fixture(initial = {args: ['model']}, beforeApply = () => {}) {
  let profile = {port: 8000, ...structuredClone(initial)}, applies = 0;
  const elements = new Map();
  for (const [, id] of html.matchAll(/\bid="([^"]+)"/g)) elements.set(id, {
    value: '', checked: false, hidden: false, textContent: '', listeners: {}, custom: {disabled: true},
    addEventListener(type, fn) {(this.listeners[type] ||= []).push(fn);}, querySelector() {return this.custom;},
  });
  const get = id => {assert(elements.has(id), `Shipped HTML must contain ${id}`); return elements.get(id);};
  const document = {getElementById: id => elements.get(id) || null};
  const context = bindContextEditor(document, {getArgs: () => profile.args, setArgs: args => {profile = {...profile, args};}});
  const binding = bindLMCacheEditor(document, {
    getProfile: () => profile, setProfile: next => {profile = next; context.refresh(true);},
    beforeApply: () => {beforeApply(); context.flush();}, onApply: () => {applies++;},
  });
  context.refresh();
  binding.refresh();
  const event = (id, type) => {for (const callback of get(id).listeners[type] || []) callback();};
  const render = () => {context.refresh(); binding.refresh();};
  return {get, binding, context, get profile() {return profile;}, get applies() {return applies;},
    replace(next) {profile = next;}, render,
    reload(next) {profile = next; context.refresh(true); binding.refresh(true);},
    save() { // Same flush/render sequence used by both Save profile and Save template.
      if (context.flush()) render();
      if (binding.flush()) render();
      return structuredClone(validateLMCacheProfile(profile));
    },
    input(id, value) {
      if (id === 'ctxYarnEnabled') get(id).checked = value;
      else get(id).value = String(value);
      if (id === 'cacheBackend') {
        event(id, 'change'); event('lmcacheSettings', 'change');
      } else if (id.startsWith('ctxCpu')) {
        event('nativeCacheFields', 'input'); event('lmcacheSettings', 'input');
      } else event(id.startsWith('ctx') ? 'contextSettings' : 'lmcacheSettings', 'input');
    },
  };
}

function enable(f) {f.input('cacheBackend', 'lmcache'); f.input('lmcacheChunkSize', '192');}

test('one save commits unblurred LMCache fields and reload selects its cache scheme', () => {
  const args = ['model', '--kv-cache-memory-bytes', String(3.25 * GiB), '--max-model-len=32768'];
  const f = fixture({args});
  assert.equal(f.get('cacheBackend').value, 'default');
  assert.equal(f.get('nativeCacheFields').hidden, false);
  assert.equal(f.get('lmcacheFields').hidden, true);
  f.input('cacheBackend', 'lmcache');
  assert.equal(f.get('nativeCacheFields').hidden, true);
  assert.equal(f.get('lmcacheFields').hidden, false);
  assert.throws(() => f.save(), /chunk size/);
  assert.equal(f.profile.lmcache, undefined);
  f.input('lmcacheChunkSize', '192');
  f.input('lmcacheCpuGiB', '5.25');
  f.input('lmcacheDiskPath', '/mnt/synthetic/cache');
  f.input('lmcacheMinFreeDiskGiB', '2.5');
  const saved = f.save();
  assert.equal(f.applies, 0);
  assert.deepEqual(saved.lmcache, {enabled: true, cpuGiB: 5.25, chunkSize: 192,
    diskPath: '/mnt/synthetic/cache', minFreeDiskGiB: 2.5, port: 5555, httpPort: 5556});
  assert.deepEqual(saved.args, args);
  const restored = fixture(JSON.parse(JSON.stringify(saved)));
  assert.equal(restored.get('cacheBackend').value, 'lmcache');
  assert.equal(restored.get('lmcacheCpuGiB').value, '5.25');
  assert.equal(restored.get('lmcacheDiskPath').value, '/mnt/synthetic/cache');
});

test('selecting LMCache replaces known native CPU KV only after candidate validation', () => {
  const f = fixture({args: ['model', '--kv-transfer-config', JSON.stringify(native), '--kv-cache-memory-bytes=5368709120']});
  enable(f);
  assert.match(f.get('lmcacheStatus').textContent, /保存时.*替换原生 CPU KV/);
  f.input('lmcacheCpuGiB', '-1');
  assert.throws(() => f.save(), /CPU/);
  assert.deepEqual(JSON.parse(argValue(f.profile.args, '--kv-transfer-config')), native);
  f.input('lmcacheCpuGiB', '8');
  const saved = f.save();
  assert.equal(saved.lmcache.enabled, true);
  assert.equal(argValue(saved.args, '--kv-transfer-config'), null);
  assert.equal(argValue(saved.args, '--kv-cache-memory-bytes'), '5368709120');
  assert.equal(f.applies, 0);
});

test('custom connectors and speculative drafts are never deleted by a scheme switch', () => {
  for (const args of [
    ['model', '--kv-transfer-config={"kv_connector":"Custom","opaque":true}'],
    ['model', '--kv-offloading-backend=custom', '--kv-offloading-size=8'],
    ['model', '--speculative-config={"method":"dflash","model":"draft"}'],
    ['model', '--kv-transfer-config', JSON.stringify(native), '--speculative-config={"method":"mtp"}'],
  ]) {
    const f = fixture({args}), before = structuredClone(f.profile);
    enable(f);
    assert.throws(() => f.save(), /自定义|投机/);
    assert.deepEqual(f.profile, before);
    assert.match(f.get('lmcacheStatus').textContent, /自定义|投机/);
  }
  const custom = ['model', '--kv-transfer-config={"kv_connector":"Custom","opaque":true}'];
  const f = fixture({args: custom});
  assert.equal(f.get('cacheBackend').value, 'custom');
  f.input('cacheBackend', 'default');
  assert.throws(() => f.save(), /自定义/);
  assert.equal(argValue(f.profile.args, '--kv-transfer-config'), argValue(custom, '--kv-transfer-config'));
});

test('default to LMCache and back preserves both input drafts across refresh and saves', () => {
  const f = fixture({args: ['model', '--kv-transfer-config', JSON.stringify(native)]});
  f.input('ctxCpuKvGiB', '10.25');
  enable(f);
  f.input('lmcacheCpuGiB', '4.5');
  f.input('lmcacheDiskPath', '/mnt/synthetic/pending');
  f.render();
  f.input('cacheBackend', 'default');
  assert.equal(f.get('ctxCpuKvGiB').value, '10.25');
  assert.equal(f.get('lmcacheCpuGiB').value, '4.5');
  let saved = f.save();
  assert.equal(saved.lmcache.enabled, false);
  assert.equal(JSON.parse(argValue(saved.args, '--kv-transfer-config')).kv_connector_extra_config.cpu_bytes_to_use, 10.25 * GiB);
  f.input('cacheBackend', 'lmcache');
  saved = f.save();
  assert.equal(saved.lmcache.cpuGiB, 4.5);
  assert.equal(argValue(saved.args, '--kv-transfer-config'), null);
  f.render();
  f.input('cacheBackend', 'default');
  assert.equal(f.get('ctxCpuKvMode').value, 'native');
  assert.equal(f.get('ctxCpuKvGiB').value, '10.25');
  saved = f.save();
  assert.equal(saved.lmcache.enabled, false);
  assert.equal(JSON.parse(argValue(saved.args, '--kv-transfer-config')).kv_connector_extra_config.cpu_bytes_to_use, 10.25 * GiB);
  assert.equal(saved.lmcache.diskPath, '/mnt/synthetic/pending');
});

test('disabled LMCache draft survives template serialization without constraining speculative defaults', () => {
  const f = fixture({args: ['model', '--speculative-config={"method":"mtp"}']});
  f.input('cacheBackend', 'lmcache');
  f.input('lmcacheCpuGiB', '4.5');
  f.input('lmcacheDiskPath', '/mnt/synthetic/pending');
  f.input('cacheBackend', 'default');
  const restored = fixture(JSON.parse(JSON.stringify(f.save())));
  assert.equal(restored.get('cacheBackend').value, 'default');
  assert.equal(restored.get('lmcacheCpuGiB').value, '4.5');
  assert.equal(restored.get('lmcacheChunkSize').value, '');
  assert.equal(restored.get('lmcacheDiskPath').value, '/mnt/synthetic/pending');
});

test('explicit profile/template replacement resets drafts while pending edits survive ordinary refresh', () => {
  const f = fixture();
  enable(f);
  f.input('lmcacheCpuGiB', '9.5');
  f.input('lmcacheChunkSize', '384');
  f.replace({...f.profile, args: ['model', '--max-model-len=16384']});
  f.render();
  assert.equal(f.get('lmcacheCpuGiB').value, '9.5');
  const saved = f.save();
  assert.equal(argValue(saved.args, '--max-model-len'), '16384');
  assert.equal(saved.lmcache.chunkSize, 384);
  f.input('lmcacheCpuGiB', '999');
  f.reload({args: ['other-model']});
  assert.equal(f.get('cacheBackend').value, 'default');
  assert.equal(f.get('lmcacheCpuGiB').value, '8');
  assert.equal(f.get('lmcacheChunkSize').value, '');
  assert.equal(f.binding.flush(), false);
});

test('pending context errors and sleep conflicts block the single save without deleting native KV', () => {
  const args = ['model', '--kv-transfer-config', JSON.stringify(native)];
  const f = fixture({args});
  f.input('ctxGpuKvGiB', '-1');
  enable(f);
  assert.throws(() => f.save(), /GPU KV/);
  assert.equal(f.profile.lmcache, undefined);
  assert.deepEqual(f.profile.args, args);
  const sleeping = fixture({args, power: {mode: 'sleep'}});
  enable(sleeping);
  assert.match(sleeping.get('lmcacheStatus').textContent, /P-State/);
  assert.throws(() => sleeping.save(), /休眠/);
  assert.deepEqual(sleeping.profile.args, args);
});
