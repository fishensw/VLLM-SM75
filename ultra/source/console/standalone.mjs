import {isLMCacheEnabled} from './public/lmcache-config.js';
import {ManagedLMCache, preflightLMCache, lmcacheEnvironment} from './lmcache-runtime.mjs';
import { managedHarnessConfig, harnessPatch, legacyHarnessConfig } from "./harness-config.mjs";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { cacheLayout, containerCache, flashinferWorkspace } from "./config.mjs";

export function writeHarnessFile(file, text) {
  // The tool user owns this directory. Do not follow a replaced destination
  // symlink when the manager writes configuration on a later restart.
  const temporary = path.join(path.dirname(file), `.sm75-${randomUUID()}`);
  const fd = fs.openSync(temporary, "wx", 0o600);
  try {
    fs.writeFileSync(fd, text);
    fs.fchownSync(fd, 1000, 1000);
    fs.renameSync(temporary, file);
  } finally {
    fs.closeSync(fd);
    try {
      fs.unlinkSync(temporary);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
}

export class Standalone {
  constructor(root, key, options = {}) {
    this.preflightLMCache = options.preflight || preflightLMCache;
    this.root = root;
    this.key = key;
    this.engine = null;
    this.harness = null;
    this.supervisor = null;
    this.lmcache = new ManagedLMCache({launch: (...args) => this.launch(...args), terminate: child => this.terminate(child), logRoot: root});
    this.cacheCleanup = null;
    this.closing = false;
    this.startFinished = null;
  }
  status(p) {
    return {
      running: !!this.engine && this.engine.id === p.id,
      name: p.id,
      backend: "in-container",
    };
  }
  powerState() {
    try {
      return JSON.parse(fs.readFileSync("/tmp/pstate-state.json", "utf8"));
    } catch {
      return null;
    }
  }
  engineKey() {
    try {
      const k = JSON.parse(
        fs.readFileSync(path.join(this.root, "api-access.json"), "utf8"),
      ).key;
      return k || this.key;
    } catch {
      return this.key;
    }
  }
  async launch(bin, args, env, log, identity = {}) {
    const fd = fs.openSync(log, "a");
    try {
      return await new Promise((resolve, reject) => {
        const child = spawn(bin, args, {
          env: { ...process.env, ...env },
          stdio: ["ignore", fd, fd],
          detached: true,
          ...identity,
        });
        child.once("error", reject);
        child.once("spawn", () => resolve(child));
      });
    } finally {
      fs.closeSync(fd);
    }
  }
  async terminate(child) {
    if (!child || child.exitCode !== null || child.signalCode != null) return;
    const gone = new Promise((resolve) => child.once("exit", resolve));
    try {
      process.kill(-child.pid, "SIGTERM");
    } catch {
      return;
    }
    let timer;
    await Promise.race([
      gone,
      new Promise((resolve) => (timer = setTimeout(resolve, 30000))),
    ]);
    clearTimeout(timer);
    if (child.exitCode === null && child.signalCode == null) {
      try {
        process.kill(-child.pid, "SIGKILL");
      } catch {}
      await gone;
    }
  }
  prepareCache(p) {
    const layout = cacheLayout(p.cacheRoot, p.format);
    for (const [key, target] of Object.entries(layout)) {
      fs.mkdirSync(target, { recursive: true });
      const link = containerCache[key];
      fs.mkdirSync(path.dirname(link), { recursive: true });
      let st;
      try {
        st = fs.lstatSync(link);
      } catch {}
      if (st?.isSymbolicLink()) {
        if (fs.readlinkSync(link) === target) continue;
        fs.unlinkSync(link);
      } else if (st) {
        if (!st.isDirectory() || fs.readdirSync(link).length)
          throw Error("缓存路径存在未迁移内容：" + link);
        fs.rmdirSync(link);
      }
      fs.symlinkSync(target, link, "dir");
    }
    return {
      ...layout,
      FLASHINFER_WORKSPACE_BASE: flashinferWorkspace(p.cacheRoot, true),
    };
  }
  async start(p) {
    if (this.closing) throw Error("管理服务正在关闭");
    if (this.starting) throw Error("模型启动进行中");
    if (this.engine) {
      if (this.engine.id === p.id) return this.status(p);
      throw Error("请先停止当前模型");
    }
    this.starting = true;
    let finishStart;
    this.startFinished = new Promise(resolve => {finishStart = resolve;});
    let cacheRuntime, engineChild, cacheFailed = false;
    try {
      if (this.cacheCleanup) { await this.cacheCleanup; this.cacheCleanup = null; }
      if (isLMCacheEnabled(p)) {
        const checked = await this.preflightLMCache(p, {standalone: true});
        if (this.closing) throw Error("管理服务正在关闭");
        cacheRuntime = await this.lmcache.start(p, {runtimeFingerprint: checked?.runtimeFingerprint});
        cacheRuntime.child.once("exit", () => {
          cacheFailed = true;
          if (this.lmcache.child === cacheRuntime.child && this.engine?.child === engineChild)
            this.terminate(engineChild).catch(() => {});
        });
        if (cacheRuntime.child.exitCode !== null || cacheRuntime.child.signalCode != null)
          throw Error("LMCache 服务在模型启动前退出");
      }
      const preparedCache = this.prepareCache(p);
      const args = [...(cacheRuntime?.plan.engineArgs || p.args)];
      const i = args.indexOf("--port");
      if (i >= 0) args[i + 1] = String(p.port);
      else args.push("--port", String(p.port));
      if (p.power?.mode !== "pstate") {
        try {
          fs.unlinkSync("/tmp/pstate-state.json");
        } catch {}
      }
      if (p.power?.mode === "pstate") {
        const ai = args.indexOf("--auto-sleep-idle-timeout");
        if (ai >= 0) args[ai + 1] = "0";
        if (!fs.existsSync("/usr/local/nvidia/lib64/libnvidia-api.so.1"))
          throw Error(
            "P-State 需要挂载宿主 libnvidia-api.so.1 到 /usr/local/nvidia/lib64/",
          );
        const senv = {
          PSTATE_GPUS: p.power.gpus || "0,1,2,3",
          PSTATE_IDLE_TIMEOUT: String(
            Math.max(
              1,
              Math.round(
                p.power.idleSeconds ??
                  (p.power.idleMinutes != null ? p.power.idleMinutes * 60 : 1),
              ),
            ),
          ),
          PSTATE_UTIL: String(p.power.util ?? 5),
          PSTATE_CONFIRM: String(p.power.confirm ?? 60),
          PSTATE_LOW: String(p.power.low ?? 8),
          PSTATE_HIGH: String(p.power.high ?? 16),
          PSTATE_POLL: String(p.power.poll ?? 5),
        };
        this.supervisor = await this.launch(
          "bash",
          ["/opt/vllm-sm75/pstate-supervisor.sh"],
          senv,
          path.join(this.root, p.id + ".pstate.log"),
        );
      }
      if (this.closing) throw Error("管理服务正在关闭");
      if (cacheFailed) throw Error("LMCache 服务在模型启动前退出");
      const child = await this.launch(
        "vllm",
        ["serve", ...args],
        {
          MALLOC_ARENA_MAX: "2",
          NCCL_P2P_LEVEL: process.env.NCCL_P2P_LEVEL ?? "SYS",
          ...p.env,
          ...(cacheRuntime ? lmcacheEnvironment : {}),
          ...containerCache,
          FLASHINFER_WORKSPACE_BASE:
            preparedCache?.FLASHINFER_WORKSPACE_BASE || "/root",
          VLLM_API_KEY: this.engineKey(),
          VLLM_MONITOR: p.env?.VLLM_MONITOR ?? "0",
        },
        path.join(this.root, p.id + ".log"),
      );
      engineChild = child;
      if (this.closing || (cacheRuntime && (cacheFailed || cacheRuntime.child.exitCode !== null || cacheRuntime.child.signalCode != null))) {
        await this.terminate(child);
        throw Error("LMCache 服务在模型启动期间退出，已停止本次模型进程");
      }
      this.engine = { id: p.id, child };
      child.once("exit", async () => {
        if (cacheRuntime) {
          this.cacheCleanup = this.lmcache.stop(cacheRuntime.child);
          this.cacheCleanup.catch(() => {});
        }
        if (this.engine?.child === child) this.engine = null;
        if (this.supervisor) {
          const s = this.supervisor;
          this.supervisor = null;
          try {
            process.kill(-s.pid, "SIGINT");
          } catch {}
          try {
            await this.terminate(s);
          } catch {}
          try {
            fs.unlinkSync("/tmp/pstate-state.json");
          } catch {}
        }
      });
      return { started: true, backend: "in-container" };
    } catch (error) {
      if (cacheRuntime) await this.lmcache.stop(cacheRuntime.child);
      const supervisor = this.supervisor;
      this.supervisor = null;
      if (supervisor) await this.terminate(supervisor);
      throw error;
    } finally {
      this.starting = false;
      finishStart();
    }
  }
  async stop(p) {
    if (this.starting) throw Error("模型启动进行中");
    if (this.engine?.id !== p.id) return;
    if (this.supervisor) {
      const s = this.supervisor;
      this.supervisor = null;
      try {
        process.kill(-s.pid, "SIGINT");
      } catch {}
      await this.terminate(s);
    }
    try {
      fs.unlinkSync("/tmp/pstate-state.json");
    } catch {}
    if (this.engine?.id === p.id) await this.terminate(this.engine.child);
    await this.lmcache.stop();
    await this.cacheCleanup;
  }
  logs(p, offset = 0) {
    const file = path.join(this.root, p.id + ".log");
    if (!fs.existsSync(file))
      return { text: "", offset: 0, size: 0, reset: true };
    const size = fs.statSync(file).size,
      cap = 8 * 1024 * 1024;
    let start = Number(offset) || 0;
    const reset = start <= 0 || start > size;
    if (reset) start = Math.max(0, size - cap);
    start = Math.max(start, size - cap);
    const length = size - start;
    if (length <= 0) return { text: "", offset: size, size, reset: false };
    const fd = fs.openSync(file, "r");
    try {
      const buf = Buffer.alloc(length);
      fs.readSync(fd, buf, 0, length, start);
      return { text: buf.toString(), offset: size, size, reset };
    } finally {
      fs.closeSync(fd);
    }
  }
  harnessLog() {
    return path.join(this.root, "harness-process.log");
  }
  async startHarness(p) {
    if (this.harness) {
      if (this.harness.id !== p.id) throw Error("请先停止工作区再切换模型");
      return;
    }
    const home = "/dsh/home",
      work = "/dsh/workspace";
    for (const dir of [home, work]) {
      fs.mkdirSync(dir, { recursive: true });
      fs.chownSync(dir, 1000, 1000);
      fs.chmodSync(dir, 0o700);
    }
    const config = managedHarnessConfig(p);
    const file = path.join(home, "settings.yaml");
    const { createRequire } = await import("node:module");
    const require = createRequire("/opt/harness/node_modules/anchor.cjs");
    const legacy = legacyHarnessConfig(file, config, require("yaml").parse);
    if (legacy) writeHarnessFile(file, JSON.stringify(legacy, null, 2));
    const overlay = path.join(home, "sm75-plugins.patch.json");
    writeHarnessFile(overlay, JSON.stringify(harnessPatch(config), null, 2));
    const dshArgs = [
      "/opt/harness/node_modules/@deepseek-ai/dsh/lib/bin.js",
      "--profile",
      "web",
      "--patch",
      overlay,
      "--no-open",
      "--host",
      "127.0.0.1",
      "--port",
      "3085",
    ];
    const dshEnv = {
      DSH_HOME: home,
      HOME: work,
      DSH_TELEMETRY_DISABLED: "1",
      SM75_ENGINE_KEY: this.engineKey(),
      CUDA_VISIBLE_DEVICES: "",
    };
    const child = await this.launchHarness(dshArgs, dshEnv, work);
    this.harness = { id: p.id, child };
    child.once("exit", () => {
      if (this.harness?.child === child) this.harness = null;
    });
  }
  async launchHarness(args, env, cwd) {
    // DSH can execute user tools. Never fall back to the manager's root identity.
    // Pass only its explicit environment, not management credentials inherited
    // from the container. The engine API key is intentionally included above.
    const childEnv = { ...env };
    for (const name of ["PATH", "LANG", "LC_ALL", "TZ", "NODE_EXTRA_CA_CERTS"])
      if (process.env[name] !== undefined) childEnv[name] = process.env[name];
    try {
      return await this.launch(
        "/usr/local/bin/node",
        args,
        childEnv,
        this.harnessLog(),
        { uid: 1000, gid: 1000, cwd, env: childEnv },
      );
    } catch (error) {
      if (error?.code === "EPERM")
        throw new Error(
          "DSH 无法切换到 UID/GID 1000；请检查容器降权能力和工作目录权限，禁止以 root 回退启动",
          { cause: error },
        );
      throw error;
    }
  }
  async stopHarness() {
    await this.terminate(this.harness?.child);
  }
  async close() {
    this.closing = true;
    if (this.starting) {
      await this.lmcache.stop();
      await this.startFinished;
    }
    if (this.supervisor) {
      const s = this.supervisor;
      this.supervisor = null;
      try {
        process.kill(-s.pid, "SIGINT");
        await this.terminate(s);
      } catch {}
    }
    await Promise.all([this.terminate(this.engine?.child), this.stopHarness()]);
    await this.lmcache.stop();
    if (this.cacheCleanup) await this.cacheCleanup;
  }
}
