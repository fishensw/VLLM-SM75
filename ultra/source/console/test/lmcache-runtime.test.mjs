import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import net from 'node:net';
import {EventEmitter, once} from 'node:events';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {ManagedLMCache, lmcachePlan, preflightLMCache, checkLMCacheDisk} from '../lmcache-runtime.mjs';
import {Standalone} from '../standalone.mjs';

const tick = () => new Promise(resolve => setImmediate(resolve));
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const deferred = () => {let resolve; const promise = new Promise(r => {resolve = r;}); return {promise, resolve};};
function fakeChild(name) {
  const child = Object.assign(new EventEmitter(), {name, pid: 999999, exitCode: null, signalCode: null});
  child.finish = (code = 0) => {child.exitCode = code; child.emit('exit', code, null);};
  return child;
}
function profile(changes = {}) {
  return {id: 'cache-audit', backend: 'native', args: ['/models/synthetic', '--max-model-len', '8192'], port: 8000,
    lmcache: {enabled: true, cpuGiB: 0.25, chunkSize: 1664, diskPath: '', minFreeDiskGiB: 0, port: 15555, httpPort: 19000}, ...changes};
}
function managed(overrides = {}) {
  return new ManagedLMCache({logRoot: os.tmpdir(), portCheck: async () => {},
    launch: async () => fakeChild('cache'), terminate: async child => child.finish(),
    fetcher: async () => ({ok: true, json: async () => ({status: 'healthy'})}),
    timeoutMs: 50, pollMs: 1, ...overrides});
}

test('disabled LMCache does not probe, inspect disk, open ports or launch a helper', async () => {
  const p = profile({lmcache: {enabled: false, diskPath: 'unfinished input'}});
  const fail = () => assert.fail('disabled cache must not perform runtime work');
  assert.equal(lmcachePlan(p), null);
  assert.equal(await preflightLMCache(p, {platform: 'unsupported', probe: fail}), null);
  assert.equal(await managed({launch: fail, portCheck: fail, fetcher: fail}).start(p), null);
  assert.deepEqual(p.args, ['/models/synthetic', '--max-model-len', '8192']);
});
test('preflight propagates missing dependencies without starting anything', async () => {
  let probes = 0;
  await assert.rejects(preflightLMCache(profile(), {platform: 'linux', probe: async (_p, plan) => {
    probes++; assert.equal(plan.summary.storageBackend, null); throw Error('missing optional compiled extension');
  }}), /missing optional compiled extension/);
  assert.equal(probes, 1);
});
test('readiness requires healthy LMCache JSON, not an arbitrary HTTP 200', async () => {
  let requests = 0;
  const cache = managed({timeoutMs: 1000, fetcher: async url => {
    assert.match(url, /\/healthcheck$/); const n = ++requests;
    return {ok: n !== 2, json: async () => ({status: n < 4 ? 'unhealthy' : 'healthy'})};
  }});
  const runtime = await cache.start(profile());
  assert.equal(requests, 4); assert.equal(runtime.child, cache.child); await cache.stop();
});
test('readiness failure terminates the helper and allows a later retry', async () => {
  const calls = [];
  const cache = managed({fetcher: async () => ({ok: true, json: async () => ({status: 'listening'})}),
    terminate: async child => {calls.push(child); child.finish();}});
  await assert.rejects(cache.start(profile()), /未.*就绪|时限/);
  assert.equal(calls.length, 1); assert.equal(cache.child, null);
  cache.fetcher = async () => ({ok: true, json: async () => ({status: 'healthy'})});
  await cache.start(profile()); await cache.stop(); assert.equal(calls.length, 2);
});
test('helper exit during readiness fails and clears ownership', async () => {
  const child = fakeChild('cache');
  const cache = managed({launch: async () => child, fetcher: async () => {child.finish(1); throw Error('closed');}});
  await assert.rejects(cache.start(profile()), /提前退出/); assert.equal(cache.child, null);
});
test('occupied ports and spawn failure leave no owned cache process', async () => {
  const occupied = managed({portCheck: async () => {throw Error('occupied');}, launch: () => assert.fail('must not spawn')});
  await assert.rejects(occupied.start(profile()), /occupied/); assert.equal(occupied.child, null);
  const failed = managed({launch: async () => {throw Error('spawn failed');}});
  await assert.rejects(failed.start(profile()), /spawn failed/); assert.equal(failed.child, null);
});
test('concurrent stop calls both wait for the same process termination', async () => {
  const release = deferred(); let terminated = 0;
  const cache = managed({terminate: async child => {terminated++; await release.promise; child.finish();}});
  await cache.start(profile()); const a = cache.stop(); let secondDone = false;
  const b = cache.stop().then(() => {secondDone = true;});
  await tick(); assert.equal(secondDone, false, 'second stop must wait for pending termination');
  release.resolve(); await Promise.all([a, b]); assert.equal(terminated, 1);
});
test('disk namespace separates KV dtype, attention backend and changed model metadata', t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-lmcache-identity-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  fs.writeFileSync(path.join(directory, 'config.json'), '{"model_type":"synthetic","revision":1}');
  const p = profile({args: [directory, '--kv-cache-dtype', 'fp8'], env: {VLLM_ATTENTION_BACKEND: 'SM75'}});
  const first = lmcachePlan(p).namespace;
  assert.equal(lmcachePlan(structuredClone(p)).namespace, first);
  assert.notEqual(lmcachePlan({...p, args: [directory, '--kv-cache-dtype', 'auto']}).namespace, first);
  assert.notEqual(lmcachePlan({...p, env: {VLLM_ATTENTION_BACKEND: 'OTHER'}}).namespace, first);
  fs.writeFileSync(path.join(directory, 'config.json'), '{"model_type":"synthetic","revision":2}');
  assert.notEqual(lmcachePlan(p).namespace, first);
});
test('disk preparation uses a namespace and a free-space gate, not a quota', {skip: process.platform !== 'linux'}, t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-lmcache-disk-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const p = profile(); p.lmcache.diskPath = path.join(directory, 'new', 'cache');
  checkLMCacheDisk(p, {prepare: true}); const plan = lmcachePlan(p);
  assert.equal(fs.statSync(plan.diskPath).isDirectory(), true); assert.deepEqual(fs.readdirSync(plan.diskPath), []);
  assert.equal(plan.summary.diskQuota, 'external-filesystem-quota-required');
  const adapter = JSON.parse(plan.serverArgs[plan.serverArgs.indexOf('--l2-adapter') + 1]);
  assert.deepEqual(adapter, {type: 'fs', base_path: plan.diskPath, use_odirect: false});
  p.lmcache.minFreeDiskGiB = 2 ** 22; assert.throws(() => checkLMCacheDisk(p), /剩余空间/);
});

function standaloneHarness() {
  const order = [];
  const s = new Standalone(os.tmpdir(), 'synthetic-key', {preflight: async () => {order.push('preflight');}});
  s.prepareCache = () => ({});
  const engine = fakeChild('engine'), helper = fakeChild('cache');
  s.lmcache = managed({launch: async () => {order.push('cache-start'); return helper;},
    terminate: async child => {order.push('cache-stop'); child.finish();}});
  s.launch = async () => {order.push('engine-start'); return engine;};
  s.terminate = async child => {if (child) {order.push(child.name + '-stop'); child.finish();}};
  return {s, order, engine, helper};
}
test('standalone preflight failure does not launch a cache or engine', async () => {
  const {s, order} = standaloneHarness(); s.preflightLMCache = async () => {throw Error('missing dependency');};
  await assert.rejects(s.start(profile()), /missing dependency/);
  assert.deepEqual(order, []); assert.equal(s.engine, null); assert.equal(s.lmcache.child, null); assert.equal(s.starting, false);
});
test('engine spawn failure cleans a successfully started cache', async () => {
  const {s, order} = standaloneHarness();
  s.launch = async () => {order.push('engine-start'); throw Error('engine spawn failed');};
  await assert.rejects(s.start(profile()), /engine spawn failed/);
  assert.deepEqual(order, ['preflight', 'cache-start', 'engine-start', 'cache-stop']);
  assert.equal(s.engine, null); assert.equal(s.lmcache.child, null);
});
test('normal stop waits for engine exit before stopping its cache', async () => {
  const {s, order, engine} = standaloneHarness(); await s.start(profile()); const release = deferred();
  s.terminate = async child => {assert.equal(child, engine); order.push('engine-stop'); await release.promise; child.finish();};
  const stopped = s.stop(profile()); await tick(); assert.equal(order.includes('cache-stop'), false);
  release.resolve(); await stopped;
  assert.equal(order.indexOf('engine-stop') < order.indexOf('cache-stop'), true); assert.equal(s.lmcache.child, null);
});
test('helper failure stops its corresponding running engine', async () => {
  const {s, order, helper, engine} = standaloneHarness(); await s.start(profile()); helper.finish(1); await tick();
  assert.equal(engine.exitCode, 0); assert.equal(order.filter(x => x === 'engine-stop').length, 1); await s.close();
});
test('helper failure during engine spawn cannot be missed', async () => {
  const {s, helper, engine} = standaloneHarness(); const release = deferred();
  s.launch = async () => {helper.finish(1); await release.promise; return engine;};
  const start = s.start(profile()); await tick(); release.resolve();
  await assert.rejects(start, /LMCache|缓存/);
  assert.notEqual(engine.exitCode, null, 'new engine must be terminated after helper failure'); assert.equal(s.engine, null);
});
test('manager close waits for detached cache cleanup after engine exit', async () => {
  const {s, engine, helper} = standaloneHarness(); await s.start(profile()); const release = deferred();
  s.lmcache.terminate = async child => {assert.equal(child, helper); await release.promise; child.finish();};
  engine.finish(); let closed = false; const closing = s.close().then(() => {closed = true;});
  await tick(); assert.equal(closed, false, 'manager must await cache termination');
  release.resolve(); await closing; assert.equal(helper.exitCode, 0);
});
test('manager close stops an owned cache even without an engine', async () => {
  const {s, helper} = standaloneHarness(); s.lmcache.child = helper; await s.close();
  assert.equal(helper.exitCode, 0); assert.equal(s.lmcache.child, null);
});
test('disabled standalone retains its no-helper lifecycle', async () => {
  const {s, order, engine} = standaloneHarness(); s.preflightLMCache = () => assert.fail('must not preflight');
  await s.start(profile({lmcache: {enabled: false}})); assert.deepEqual(order, ['engine-start']);
  assert.equal(s.engine.child, engine); await s.close(); assert.equal(order.includes('cache-start'), false);
});
async function freePort() {
  const server = net.createServer(); await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port; await new Promise(resolve => server.close(resolve)); return port;
}
test('real manager API rejects missing LMCache without disrupting a running native engine', {skip: process.platform !== 'linux'}, async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-lmcache-api-')), engineFile = path.join(root, 'synthetic-engine');
  fs.writeFileSync(engineFile, '#!' + process.execPath + '\nsetInterval(() => {}, 1000);\nprocess.on("SIGTERM", () => process.exit(0));\n', {mode: 0o700});
  const port = await freePort(), proxyPort = await freePort();
  const manager = spawn(process.execPath, [fileURLToPath(new URL('../server.mjs', import.meta.url))], {
    env: {...process.env, SM75_SINGLE_CONTAINER: '0', SM75_CONSOLE_ROOT: root,
      SM75_CONSOLE_HOST: '127.0.0.1', SM75_CONSOLE_PORT: String(port), SM75_HARNESS_PROXY_PORT: String(proxyPort),
      SM75_VLLM_BIN: engineFile, SM75_PYTHON: path.join(root, 'missing-python'), PATH: root}, stdio: 'ignore'});
  const base = `http://127.0.0.1:${port}`;
  try {
    let ready = false;
    for (let i = 0; i < 100; i++) {try {ready = (await fetch(base)).ok; if (ready) break;} catch {} await delay(40);}
    assert(ready, 'isolated API must start');
    const headers = {Authorization: 'Bearer ' + fs.readFileSync(path.join(root, 'key'), 'utf8').trim(), 'Content-Type': 'application/json'};
    const old = {id: 'old-engine', backend: 'native', format: 'fp8', cacheRoot: path.join(root, 'compile'), port: 18001,
      args: ['/synthetic/model'], power: {mode: 'pstate'}};
    const next = {...old, ...profile({id: 'new-cache-engine', port: 18002})};
    const post = (url, body) => fetch(base + url, {method: 'POST', headers, body: JSON.stringify(body)});
    for (const p of [old, next]) assert.equal((await post('/console-api/profiles', p)).status, 200);
    const started = await post('/console-api/profiles/old-engine/start', {});
    assert.equal(started.status, 200, await started.text());
    const rejected = await post('/console-api/profiles/new-cache-engine/start', {});
    assert.equal(rejected.status, 400); assert.match((await rejected.json()).error, /LMCache.*启动检查失败/);
    const status = await (await post('/console-api/profiles/old-engine/start', {})).json();
    assert.equal(status.running, true, 'old process must remain owned and running');
    await post('/console-api/profiles/old-engine/stop', {});
  } finally {
    if (manager.exitCode === null) {const stopped = once(manager, 'exit'); manager.kill(); await stopped;}
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test('a replacement helper cannot start until prior cache termination completes', async () => {
  const release = deferred(); let launches = 0;
  const cache = managed({launch: async () => fakeChild('cache-' + ++launches),
    terminate: async child => {await release.promise; child.finish();}});
  await cache.start(profile()); const stopped = cache.stop();
  const starting = cache.start(profile()); await tick(); assert.equal(launches, 1);
  release.resolve(); await stopped; await starting; assert.equal(launches, 2); await cache.stop();
});

test('closing during engine spawn cancels the start and leaves no helper or engine', async () => {
  const {s, engine, helper} = standaloneHarness(); const release = deferred();
  s.launch = async () => {await release.promise; return engine;};
  const started = s.start(profile()); await tick(); const rejected = assert.rejects(started, /关闭|LMCache|缓存/);
  const closing = s.close(); await tick(); release.resolve();
  await rejected; await closing;
  assert.notEqual(engine.exitCode, null); assert.notEqual(helper.exitCode, null);
  assert.equal(s.engine, null); assert.equal(s.lmcache.child, null);
});

test('inherited layout environment and installed runtime fingerprint isolate disk namespaces', t => {
  const old = process.env.VLLM_ATTENTION_BACKEND;
  t.after(() => {if (old === undefined) delete process.env.VLLM_ATTENTION_BACKEND; else process.env.VLLM_ATTENTION_BACKEND = old;});
  const p = profile(); p.lmcache.diskPath = '/synthetic/cache';
  process.env.VLLM_ATTENTION_BACKEND = 'synthetic-a';
  const a = lmcachePlan(p, {runtimeFingerprint: 'runtime-a'});
  assert.equal(lmcachePlan(p, {runtimeFingerprint: 'runtime-a'}).namespace, a.namespace);
  assert.notEqual(lmcachePlan(p, {runtimeFingerprint: 'runtime-b'}).namespace, a.namespace);
  process.env.VLLM_ATTENTION_BACKEND = 'synthetic-b';
  assert.notEqual(lmcachePlan(p, {runtimeFingerprint: 'runtime-a'}).namespace, a.namespace);
  assert.equal(lmcachePlan({...p, env: {VLLM_ATTENTION_BACKEND: 'synthetic-a'}}, {runtimeFingerprint: 'runtime-a'}).namespace, a.namespace);
  assert.match(lmcachePlan(p).summary.diskPath, /<runtime-isolated>$/);
  assert.equal(a.summary.diskPath, a.diskPath);
});
test('management credentials do not become cache namespace inputs or public summaries', t => {
  const key = 'VLLM_TEST_API_KEY', old = process.env[key];
  t.after(() => {if (old === undefined) delete process.env[key]; else process.env[key] = old;});
  process.env[key] = 'synthetic-credential-a'; const a = lmcachePlan(profile());
  process.env[key] = 'synthetic-credential-b'; const b = lmcachePlan(profile());
  assert.equal(a.namespace, b.namespace); assert.equal(JSON.stringify(b).includes('synthetic-credential'), false);
});
test('sleep and VMM configuration are rejected before probing the runtime', async () => {
  const bad = [profile({power: {mode: 'sleep'}}),
    ...[['--enable-sleep-mode'], ['--enable-cumem-allocator'], ['--auto-sleep-idle-timeout=1']]
      .map(flags => profile({args: ['/models/synthetic', ...flags]})),
    profile({env: {VLLM_AUTO_SLEEP_IDLE_TIMEOUT: '1'}}),
    profile({env: {PYTORCH_ALLOC_CONF: 'expandable_segments:True'}}),
    profile({env: {PYTORCH_CUDA_ALLOC_CONF: 'max_split_size_mb:64,expandable_segments:True'}})];
  for (const p of bad) await assert.rejects(preflightLMCache(p, {platform: 'linux', probe: () => assert.fail('must reject before probe')}), /休眠|allocator|expandable/);
});
test('inherited sleep flags are checked even when the CLI timeout is zero', async t => {
  const names = ['VLLM_AUTO_SLEEP_IDLE_TIMEOUT', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'];
  const previous = Object.fromEntries(names.map(name => [name, process.env[name]]));
  t.after(() => {for (const name of names) {if (previous[name] === undefined) delete process.env[name]; else process.env[name] = previous[name];}});
  for (const name of names) delete process.env[name];
  const p = profile({args: ['/models/synthetic', '--auto-sleep-idle-timeout', '0']});
  process.env.VLLM_AUTO_SLEEP_IDLE_TIMEOUT = '2';
  await assert.rejects(preflightLMCache(p, {platform: 'linux', probe: () => assert.fail()}), /休眠/);
  delete process.env.VLLM_AUTO_SLEEP_IDLE_TIMEOUT;
  for (const name of names.slice(1)) {
    process.env[name] = 'expandable_segments:True';
    await assert.rejects(preflightLMCache(p, {platform: 'linux', probe: () => assert.fail()}), /expandable_segments/);
    delete process.env[name];
  }
  let checked = 0;
  await preflightLMCache({...p, power: {mode: 'pstate'}}, {platform: 'linux', probe: async () => {checked++; return {ok: true};}});
  assert.equal(checked, 1);
});
test('standalone passes the checked runtime identity into its managed cache namespace', async () => {
  const {s} = standaloneHarness(); const start = s.lmcache.start.bind(s.lmcache); let fingerprint;
  s.preflightLMCache = async () => ({runtimeFingerprint: 'verified-runtime'});
  s.lmcache.start = (p, options) => {fingerprint = options.runtimeFingerprint; return start(p, options);};
  await s.start(profile()); assert.equal(fingerprint, 'verified-runtime'); await s.close();
});

test('stop during a pending healthy response cancels startup and waits for termination', async () => {
  const health = deferred(), terminated = deferred();
  const child = fakeChild('cache'); let stopping = false;
  const cache = managed({launch: async () => child,
    fetcher: async () => {await health.promise; return {ok: true, json: async () => ({status: 'healthy'})};},
    terminate: async item => {stopping = true; await terminated.promise; item.finish();}});
  const starting = cache.start(profile());
  const rejected = assert.rejects(starting, /取消/);
  await tick();
  const stopped = cache.stop(); await tick(); assert.equal(stopping, true);
  health.resolve(); await tick();
  assert.equal(cache.child, null);
  terminated.resolve(); await Promise.all([rejected, stopped]);
  assert.equal(child.exitCode, 0);
});

async function nativeRaceFixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-lmcache-race-'));
  const helperFile = path.join(root, 'fake-python'), engineFile = path.join(root, 'fake-vllm');
  const gate = name => path.join(root, name);
  fs.writeFileSync(helperFile, `#!${process.execPath}
const fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const root=__dirname, argv=process.argv.slice(2);
if(argv[0]?.endsWith('lmcache_probe.py')) {
  console.log('SM75_LMCACHE_PROBE='+JSON.stringify({ok:true,runtimeFingerprint:'synthetic-test-only'}));
  process.exit(0);
}
if(argv[0]!=='-m') process.exit(2);
const server=http.createServer((req,res)=>{res.setHeader('Content-Type','application/json');
  res.end(JSON.stringify({status:fs.existsSync(path.join(root,'ready'))?'healthy':'starting'}));});
server.listen(Number(argv[argv.indexOf('--http-port')+1]),'127.0.0.1',()=>{
  fs.writeFileSync(path.join(root,'helper-pid'),String(process.pid));
  fs.writeFileSync(path.join(root,'helper-started'),'1');
});
process.on('SIGTERM',()=>{
  fs.writeFileSync(path.join(root,'helper-stopping'),'1');
  const timer=setInterval(()=>{if(fs.existsSync(path.join(root,'allow-helper-exit'))){
    clearInterval(timer);server.closeAllConnections();server.close(()=>process.exit(0));
  }},10);
});
`, {mode: 0o700});
  fs.writeFileSync(engineFile, `#!${process.execPath}
const fs=require('node:fs'),path=require('node:path');
fs.writeFileSync(path.join(__dirname,'engine-started'),String(process.pid));
setInterval(()=>{},1000);
process.on('SIGTERM',()=>process.exit(0));
`, {mode: 0o700});
  const port = await freePort(), proxyPort = await freePort();
  const p = profile({id: 'native-race', format: 'fp8', cacheRoot: path.join(root, 'compile'), power: {mode: 'pstate'}});
  p.lmcache.port = await freePort(); p.lmcache.httpPort = await freePort();
  const manager = spawn(process.execPath, [fileURLToPath(new URL('../server.mjs', import.meta.url))], {
    env: {...process.env, SM75_SINGLE_CONTAINER: '0', SM75_CONSOLE_ROOT: root,
      SM75_CONSOLE_HOST: '127.0.0.1', SM75_CONSOLE_PORT: String(port), SM75_HARNESS_PROXY_PORT: String(proxyPort),
      SM75_VLLM_BIN: engineFile, SM75_PYTHON: helperFile, PATH: root}, stdio: 'ignore'});
  t.after(async () => {
    fs.writeFileSync(gate('ready'), '1'); fs.writeFileSync(gate('allow-helper-exit'), '1');
    if (manager.exitCode === null) {
      const exited = once(manager, 'exit'); manager.kill('SIGTERM');
      const timer = setTimeout(() => manager.kill('SIGKILL'), 10000);
      await exited; clearTimeout(timer);
    }
    // On assertion failure also clean only these synthetic child PIDs.
    for (const name of ['helper-pid', 'engine-started']) if (fs.existsSync(gate(name))) {
      try {process.kill(Number(fs.readFileSync(gate(name))), 'SIGKILL');} catch (error) {if (error.code !== 'ESRCH') throw error;}
    }
    fs.rmSync(root, {recursive: true, force: true});
  });
  const base = `http://127.0.0.1:${port}`;
  async function until(predicate) {
    for (let i = 0; i < 150; i++) {if (await predicate()) return; await delay(20);}
    assert.fail('synthetic lifecycle condition did not complete');
  }
  await until(() => fetch(base).then(r => r.ok, () => false));
  const headers = {Authorization: 'Bearer ' + fs.readFileSync(gate('key'), 'utf8').trim(), 'Content-Type': 'application/json'};
  const post = (route, body = {}) => fetch(base + '/console-api/' + route, {
    method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(6000)});
  const response = await post('profiles', p); assert.equal(response.status, 200, await response.text());
  return {root, gate, manager, post, until};
}

test('native stop rejects pending starts and holds the model lock through delayed cache cleanup', {skip: process.platform !== 'linux'}, async t => {
  const {gate, post, until} = await nativeRaceFixture(t);
  const started = post('profiles/native-race/start');
  await until(() => fs.existsSync(gate('helper-started')));
  const busyStop = await post('profiles/native-race/stop');
  assert.equal(busyStop.status, 400);
  assert.match((await busyStop.json()).error, /操作进行中/);
  assert.equal(fs.existsSync(gate('helper-stopping')), false);
  assert.equal(fs.existsSync(gate('engine-started')), false);
  fs.writeFileSync(gate('ready'), '1');
  const response = await started; assert.equal(response.status, 200, await response.text());
  await until(() => fs.existsSync(gate('engine-started')));
  let stopDone = false;
  const stopped = post('profiles/native-race/stop').then(r => {stopDone = true; return r;});
  await until(() => fs.existsSync(gate('helper-stopping')));
  assert.equal(stopDone, false);
  const busyStart = await post('profiles/native-race/start');
  assert.equal(busyStart.status, 400);
  assert.match((await busyStart.json()).error, /操作进行中/);
  fs.writeFileSync(gate('allow-helper-exit'), '1');
  assert.equal((await stopped).status, 200);
  // The lock must also be released, and a subsequent real fake-engine cycle works.
  const restarted = await post('profiles/native-race/start');
  assert.equal(restarted.status, 200, await restarted.text());
  assert.equal((await post('profiles/native-race/stop')).status, 200);
});

test('native SIGTERM cancels pending cache readiness without using the public stop lock', {skip: process.platform !== 'linux'}, async t => {
  const {gate, manager, post, until} = await nativeRaceFixture(t);
  const started = post('profiles/native-race/start').catch(error => ({aborted: true, error}));
  await until(() => fs.existsSync(gate('helper-started')));
  const exited = once(manager, 'exit'); manager.kill('SIGTERM');
  await until(() => fs.existsSync(gate('helper-stopping')));
  fs.writeFileSync(gate('ready'), '1'); fs.writeFileSync(gate('allow-helper-exit'), '1');
  await exited;
  const response = await started;
  if (!response.aborted) assert.equal(response.status, 400);
  assert.equal(fs.existsSync(gate('engine-started')), false);
  const pid = Number(fs.readFileSync(gate('helper-pid')));
  assert.throws(() => process.kill(pid, 0), {code: 'ESRCH'});
});
