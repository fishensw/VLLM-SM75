import test from 'node:test';
import assert from 'node:assert/strict';
import {isLMCacheEnabled, validateLMCacheProfile, lmcacheConnectorArgs} from '../public/lmcache-config.js';
import {argValue, GiB} from '../public/context-config.js';

function profile(lmcache = {}, args = ['model']) {
  return {id: 'synthetic', backend: 'docker', port: 8000, args,
    lmcache: {enabled: true, cpuGiB: 8, chunkSize: 192, diskPath: '', minFreeDiskGiB: 4,
      port: 5555, httpPort: 5556, ...lmcache}};
}

test('managed MP arguments are ephemeral and preserve GPU, context and unrelated flags', () => {
  const p = profile({cpuGiB: 6.25}, ['model', '--max-model-len=32768',
    '--kv-cache-memory-bytes', String(2.5 * GiB), '--hf-overrides', '{"keep":true}',
    '--no-enable-prefix-caching', '--mamba-cache-mode=all', '--seed', '0']);
  const before = structuredClone(p);
  assert.equal(validateLMCacheProfile(p), p); // Docker metadata does not identify the real launch path.
  const args = lmcacheConnectorArgs(p);
  assert.deepEqual(p, before);
  assert.equal(argValue(args, '--max-model-len'), '32768');
  assert.equal(argValue(args, '--kv-cache-memory-bytes'), String(2.5 * GiB));
  assert.equal(argValue(args, '--hf-overrides'), '{"keep":true}');
  assert.equal(argValue(args, '--seed'), '0');
  assert(args.includes('--enable-prefix-caching'));
  assert(!args.some(x => x.startsWith('--no-enable-prefix-caching')));
  assert(!args.some(x => x.startsWith('--disable-hybrid-kv-cache-manager')));
  assert.equal(argValue(args, '--mamba-cache-mode'), 'align');
  assert.deepEqual(JSON.parse(argValue(args, '--kv-transfer-config')), {
    kv_connector: 'LMCacheMPConnector', kv_role: 'kv_both',
    kv_connector_module_path: 'lmcache.integration.vllm.lmcache_mp_connector',
    kv_connector_extra_config: {'lmcache.mp.host': 'tcp://127.0.0.1', 'lmcache.mp.port': 5555,
      'lmcache.mp.isolated_ipc': false},
  });
});

test('disabled drafts and profiles without LMCache preserve their original connector unchanged', () => {
  for (const p of [{args: ['model', '--kv-transfer-config', '{"opaque":true}']},
    profile({enabled: false, chunkSize: null, diskPath: 'partially entered', cpuGiB: null},
      ['model', '--speculative-config={"method":"dflash"}'])]) {
    const before = structuredClone(p);
    assert(!isLMCacheEnabled(p));
    assert.equal(validateLMCacheProfile(p), p);
    const args = lmcacheConnectorArgs(p);
    assert.deepEqual(args, p.args);
    assert.notEqual(args, p.args);
    assert.deepEqual(p, before);
  }
  for (const config of [null, [], true, {enabled: 'true'}, {enabled: 1}]) {
    assert.throws(() => validateLMCacheProfile({args: ['model'], lmcache: config}), /布尔/);
  }
});

test('required chunk size and numeric capacities cannot be guessed from coercible or unsafe values', () => {
  for (const chunkSize of [undefined, null, '', '192', false, 0, -1, 1.5, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1])
    assert.throws(() => validateLMCacheProfile(profile({chunkSize})), /chunk size/);
  for (const cpuGiB of [undefined, null, '', '8', true, 0, -1, 1e-20, 1e12, Infinity, NaN])
    assert.throws(() => validateLMCacheProfile(profile({cpuGiB})), /CPU/);
  for (const minFreeDiskGiB of [null, '', '4', false, -1, Infinity, 1e12])
    assert.throws(() => validateLMCacheProfile(profile({minFreeDiskGiB})), /剩余空间/);
  assert.equal(validateLMCacheProfile(profile({minFreeDiskGiB: 0, cpuGiB: 0.25})).lmcache.cpuGiB, 0.25);
});

test('disk is optional, otherwise requires an absolute subdirectory without traversal or control characters', () => {
  for (const diskPath of ['', '/mnt/cache/lmcache', '/mnt/cache/with space/'])
    assert.equal(validateLMCacheProfile(profile({diskPath})).lmcache.diskPath, diskPath);
  for (const diskPath of [null, '/','///', 'relative/cache', 'C:\\cache', '/mnt/../cache', '/mnt/line\ncache', '/mnt/\0cache'])
    assert.throws(() => validateLMCacheProfile(profile({diskPath})), /绝对子目录/);
});

test('managed RPC and HTTP ports must be distinct and cannot consume the model API port', () => {
  for (const port of [5556, 8000, 0, 1023, 65536, 5000.5, '5555', true])
    assert.throws(() => validateLMCacheProfile(profile({port})), /端口/);
  assert.throws(() => validateLMCacheProfile(profile({httpPort: 8000})), /端口/);
  assert.equal(validateLMCacheProfile(profile({port: 65534, httpPort: 65535})).lmcache.port, 65534);
});

test('native/custom connectors and speculative decoding require explicit removal, with no mutation on rejection', () => {
  for (const conflict of [
    ['--kv-transfer-config', '{"kv_connector":"OffloadingConnector"}'],
    ['--kv-transfer-config={"kv_connector":"OpaqueConnector","preserve":true}'],
    ['--kv-offloading-size', '8'], ['--kv-offloading-backend=lmcache'],
    ['--speculative-config', '{"method":"dflash"}'], ['--speculative-config={"method":"mtp"}'],
  ]) {
    const p = profile({}, ['model', ...conflict]), before = structuredClone(p);
    assert.throws(() => lmcacheConnectorArgs(p), /互斥|投机/);
    assert.deepEqual(p, before);
  }
});

test('legacy hybrid-manager disabling is rejected even when supplied with a false-looking value', () => {
  for (const flag of ['--disable-hybrid-kv-cache-manager', '--disable-hybrid-kv-cache-manager=true',
    '--disable-hybrid-kv-cache-manager=1', '--disable-hybrid-kv-cache-manager=false']) {
    assert.throws(() => lmcacheConnectorArgs(profile({}, ['model', flag])), /移除 --disable-hybrid/);
  }
});

test('LMCache rejects memory-releasing sleep and incompatible allocators without restricting disabled drafts', () => {
  const conflicts = [
    {power: {mode: 'sleep'}},
    {args: ['model', '--enable-sleep-mode']},
    {args: ['model', '--enable-sleep-mode=false']},
    {args: ['model', '--enable-cumem-allocator=true']},
    {args: ['model', '--auto-sleep-idle-timeout', '30']},
    {args: ['model', '--auto-sleep-idle-timeout=0.5']},
    {env: {VLLM_AUTO_SLEEP_IDLE_TIMEOUT: '10'}},
    {env: {PYTORCH_ALLOC_CONF: 'max_split_size_mb:128,expandable_segments:True'}},
    {env: {PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments: True'}},
  ];
  for (const conflict of conflicts) {
    const p = {...profile(), ...conflict};
    assert.throws(() => validateLMCacheProfile(p), /请选 P-State/);
    assert.equal(validateLMCacheProfile({...p, lmcache: {...p.lmcache, enabled: false}}).lmcache.enabled, false);
  }
  const valid = {...profile(), power: {mode: 'pstate'}, args: ['model', '--auto-sleep-idle-timeout=0'],
    env: {VLLM_AUTO_SLEEP_IDLE_TIMEOUT: '0', PYTORCH_ALLOC_CONF: 'expandable_segments:False'}};
  assert.equal(validateLMCacheProfile(valid), valid);
});
