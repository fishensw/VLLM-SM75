import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import {fileURLToPath} from 'node:url';
import {createHash, randomUUID} from 'node:crypto';
import {execFile, spawn} from 'node:child_process';
import {promisify} from 'node:util';
import {isLMCacheEnabled, validateLMCacheProfile, lmcacheConnectorArgs} from './public/lmcache-config.js';

const exec = promisify(execFile);
const here = path.dirname(fileURLToPath(import.meta.url));
export const LMCACHE_VERSION = '0.5.5';
export const lmcacheEnvironment = {LMCACHE_TRACK_USAGE: 'false', LMCACHE_REQUEST_TELEMETRY_TYPE: 'noop'};

// Isolate opaque hybrid pages by model files, complete engine settings and the
// reviewed implementation versions. Never use the compilation-cache directory.
export function lmcachePlan(profile, {runtimeFingerprint = ""} = {}) {
  if (!isLMCacheEnabled(profile)) return null;
  validateLMCacheProfile(profile);
  const c = profile.lmcache;
  const args = lmcacheConnectorArgs(profile);
  let modelIdentity = [];
  try {
    modelIdentity = fs.readdirSync(args[0]).filter(n => /\.(json|safetensors|bin)$/.test(n)).sort().map(n => {
      const f = path.join(args[0], n), s = fs.statSync(f);
      return [n, s.size, s.mtimeMs, n === 'config.json' ? fs.readFileSync(f, 'utf8') : null];
    });
  } catch { /* The engine will report missing model files. */ }
  const inheritedLayoutEnv = Object.fromEntries(Object.entries({...process.env, ...profile.env})
    .filter(([key]) => /^(VLLM_|LMCACHE_|FLASH|SM75_|PYTORCH_|TORCH_|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES)/.test(key) && !/(KEY|TOKEN|PASSWORD|SECRET)/i.test(key))
    .sort(([a], [b]) => a.localeCompare(b)));
  const namespace = createHash('sha256').update(JSON.stringify({
    schema: 1, lmcache: LMCACHE_VERSION, vllm: '0.30.0-sm75-v0.1.6',
    args, env: inheritedLayoutEnv, modelIdentity, runtimeFingerprint,
  })).digest('hex');
  const diskPath = c.diskPath ? path.posix.join(c.diskPath, namespace) : '';
  const serverArgs = ['-m', 'lmcache.v1.multiprocess.http_server',
    '--host', '127.0.0.1', '--port', String(c.port),
    '--http-host', '127.0.0.1', '--http-port', String(c.httpPort),
    '--chunk-size', String(c.chunkSize), '--separate-object-groups',
    '--no-isolated-ipc', '--l1-size-gb', String(c.cpuGiB),
    '--l1-init-size-gb', '1', '--eviction-policy', 'LRU'];
  if (diskPath) serverArgs.push('--l2-adapter', JSON.stringify({type: 'fs', base_path: diskPath, use_odirect: false}));
  return {bin: process.env.SM75_PYTHON || 'python3', serverArgs, engineArgs: args,
    diskPath, namespace, env: lmcacheEnvironment,
    healthURL: `http://127.0.0.1:${c.httpPort}/healthcheck`,
    summary: {version: LMCACHE_VERSION, cpuGiB: c.cpuGiB, chunkSize: c.chunkSize,
      diskPath: runtimeFingerprint ? diskPath : (c.diskPath ? path.posix.join(c.diskPath, '<runtime-isolated>') : ''), minFreeDiskGiB: c.minFreeDiskGiB, diskQuota: 'external-filesystem-quota-required',
      ports: [c.port, c.httpPort], storageBackend: diskPath ? 'fs' : null}};
}

export function checkLMCacheDisk(profile, {prepare = false, plan = lmcachePlan(profile)} = {}) {
  if (!plan?.diskPath) return;
  let parent = profile.lmcache.diskPath;
  while (!fs.existsSync(parent)) {
    const next = path.posix.dirname(parent);
    if (next === parent) throw Error('LMCache 磁盘目录不可访问');
    parent = next;
  }
  if (!fs.statSync(parent).isDirectory()) throw Error('LMCache 磁盘路径不是目录');
  fs.accessSync(parent, fs.constants.W_OK | fs.constants.X_OK);
  const disk = fs.statfsSync(parent);
  if (Number(disk.bavail) * Number(disk.bsize) < profile.lmcache.minFreeDiskGiB * 2 ** 30)
    throw Error('LMCache 磁盘剩余空间低于配置的启动门槛');
  if (prepare) {
    fs.mkdirSync(plan.diskPath, {recursive: true, mode: 0o700});
    const test = path.join(plan.diskPath, `.sm75-write-check-${randomUUID()}`);
    const fd = fs.openSync(test, 'wx', 0o600);
    fs.closeSync(fd);
    fs.unlinkSync(test);
  }
}

export async function probeLMCache(profile, plan = lmcachePlan(profile)) {
  const payload = {serverArgs: plan.serverArgs.slice(2), engineArgs: plan.engineArgs};
  try {
    const {stdout} = await exec(plan.bin, [path.join(here, 'lmcache_probe.py'), JSON.stringify(payload)], {
      timeout: 45000, maxBuffer: 1024 * 1024,
      env: {...process.env, ...profile.env, ...lmcacheEnvironment},
    });
    const line = stdout.split('\n').find(l => l.startsWith('SM75_LMCACHE_PROBE='));
    if (!line) throw Error('依赖检查未返回结果');
    const result = JSON.parse(line.slice('SM75_LMCACHE_PROBE='.length));
    if (!result.ok) throw Error(result.error || '依赖检查失败');
    return result;
  } catch (error) {
    // No inherited environment or CLI credentials are included in diagnostics.
    throw Error('LMCache 启动检查失败，请安装固定的 0.5.5 cu129 扩展并检查配置：' + (error.stdout?.split('\n').find(l => l.startsWith('SM75_LMCACHE_PROBE=')) || error.message).slice(0, 1200));
  }
}

export async function preflightLMCache(profile, {standalone = false, probe = probeLMCache, platform = process.platform, ownedPorts = []} = {}) {
  if (!isLMCacheEnabled(profile)) return null;
  validateLMCacheProfile(profile);
  if (!standalone && profile.backend !== 'native') throw Error('受管 LMCache 当前支持 ultra 单容器或 native 运行方式；独立 Docker 配置请使用文档中的手动启动方式');
  if (platform !== 'linux') throw Error('LMCache 受管服务需要 Linux CUDA 环境');
  const env = {...process.env, ...profile.env};
  if (Number(env.VLLM_AUTO_SLEEP_IDLE_TIMEOUT || 0) > 0 ||
      [env.PYTORCH_ALLOC_CONF, env.PYTORCH_CUDA_ALLOC_CONF].some(v => /expandable_segments\s*:\s*true/i.test(v || '')))
    throw Error('LMCache 尚未验收显存休眠或 expandable_segments；请关闭继承的自动休眠和可扩展显存分配配置');
  checkLMCacheDisk(profile);
  for (const port of [profile.lmcache.port, profile.lmcache.httpPort])
    if (!ownedPorts.includes(port)) await assertFreePort(port);
  return probe(profile, lmcachePlan(profile));
}

async function assertFreePort(port) {
  await new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', () => reject(Error(`LMCache 端口 ${port} 已被占用，未连接未知服务`)));
    server.listen(port, '127.0.0.1', () => server.close(resolve));
  });
}

export async function launchLMCacheProcess(bin, args, env, log) {
  const fd = fs.openSync(log, 'a');
  try {
    return await new Promise((resolve, reject) => {
      const child = spawn(bin, args, {env: {...process.env, ...env}, detached: true, stdio: ['ignore', fd, fd]});
      child.once('error', reject);
      child.once('spawn', () => resolve(child));
    });
  } finally { fs.closeSync(fd); }
}
export async function terminateLMCacheProcess(child) {
  if (child.exitCode !== null || child.signalCode != null) return;
  const exited = new Promise(resolve => child.once('exit', resolve));
  try { process.kill(-child.pid, 'SIGTERM'); } catch (error) { if (error.code !== 'ESRCH') throw error; return; }
  let timer;
  await Promise.race([exited, new Promise(resolve => {timer = setTimeout(resolve, 10000);})]);
  clearTimeout(timer);
  if (child.exitCode === null && child.signalCode == null) {
    try { process.kill(-child.pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
    await exited;
  }
}

export class ManagedLMCache {
  constructor({launch = launchLMCacheProcess, terminate = terminateLMCacheProcess, logRoot, fetcher = fetch, timeoutMs = 45000, pollMs = 200, portCheck = assertFreePort}) {
    Object.assign(this, {launch, terminate, logRoot, fetcher, timeoutMs, pollMs, portCheck});
    this.child = null;
    this.stopping = null;
    this.ports = [];
  }
  async start(profile, options = {}) {
    const plan = lmcachePlan(profile, options);
    if (!plan) return null;
    if (this.stopping) await this.stopping.promise;
    if (this.child) throw Error('LMCache 服务尚未停止');
    await this.portCheck(profile.lmcache.port);
    await this.portCheck(profile.lmcache.httpPort);
    checkLMCacheDisk(profile, {prepare: true, plan});
    const child = await this.launch(plan.bin, plan.serverArgs, {...profile.env, ...plan.env}, path.join(this.logRoot, profile.id + '.lmcache.log'));
    this.child = child;
    this.ports = [profile.lmcache.port, profile.lmcache.httpPort];
    try {
      const deadline = Date.now() + this.timeoutMs;
      while (Date.now() < deadline) {
        if (this.child !== child) throw Error('LMCache 启动已取消');
        if (child.exitCode !== null || child.signalCode != null) throw Error('LMCache 服务提前退出，请查看缓存服务日志');
        let healthy = false;
        try {
          const response = await this.fetcher(plan.healthURL, {signal: AbortSignal.timeout(1500)});
          healthy = response.ok && (await response.json()).status === 'healthy';
        } catch { /* Bound readiness wait; never count a TCP socket as ready. */ }
        // stop() may revoke ownership while an in-flight health response still
        // succeeds. Cancellation belongs outside the transient HTTP catch.
        if (this.child !== child) throw Error('LMCache 启动已取消');
        if (child.exitCode !== null || child.signalCode != null) throw Error('LMCache 服务提前退出，请查看缓存服务日志');
        if (healthy) return {child, plan};
        await new Promise(resolve => setTimeout(resolve, this.pollMs));
      }
      throw Error('LMCache 服务未在时限内就绪，请查看缓存服务日志');
    } catch (error) {
      await this.stop(child);
      throw error;
    }
  }
  async stop(expected = this.child || this.stopping?.child) {
    if (!expected) return;
    if (this.stopping?.child === expected) return this.stopping.promise;
    if (this.child !== expected) return;
    this.child = null;
    this.ports = [];
    const promise = Promise.resolve().then(() => this.terminate(expected));
    const stopping = {child: expected, promise};
    this.stopping = stopping;
    try { await promise; }
    finally { if (this.stopping === stopping) this.stopping = null; }
  }
}
