import {ManagedLMCache, preflightLMCache, lmcachePlan, lmcacheEnvironment, terminateLMCacheProcess} from './lmcache-runtime.mjs';
import {modelContext, retargetContextModel} from './public/context-config.js';
import { harnessVersion } from "./apply-branding.mjs";
import { harnessPath } from "./harness-config.mjs";
import { Attachments } from "./attachments.mjs";
import { Auth, equalSecret } from "./auth.mjs";
import { switchModel } from "./switch-model.mjs";
import { validateSampling } from "./sampling.mjs";
import { modelSamplingDefaults } from "./sampling-defaults.mjs";
import { deleteModelFiles } from "./model-delete.mjs";
import { OriginalTests } from "./original-tests.mjs";
import { chatConfig } from "./chat-config.mjs";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import crypto from "node:crypto";
import {
  makeCommand,
  validateProfile,
  cacheLayout,
  recommend,
} from "./config.mjs";
import { liveSummary } from "./live-summary.mjs";
import { Telemetry } from "./telemetry.mjs";
import { Store, presets, applyPreset } from "./store.mjs";
import { Standalone, writeHarnessFile } from "./standalone.mjs";
import { searchModels } from "./model-search.mjs";
import { JobLedger } from "./job-ledger.mjs";
const exec = promisify(execFile),
  here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(
  process.env.SM75_CONSOLE_ROOT || path.join(here, "data"),
);
fs.mkdirSync(root, { recursive: true, mode: 0o700 });
const store = new Store(root);
const secretPath = path.join(root, "key");
const existingInstall = fs.existsSync(secretPath);
const auth = new Auth(root, {
  ttlSeconds: Number(process.env.SM75_SESSION_TTL_SECONDS || 43200),
  trustedProxies: (process.env.SM75_TRUSTED_PROXIES || "")
    .split(",")
    .filter(Boolean),
});
const key = auth.key;
const accessFile = path.join(root, "api-access.json");
if (!fs.existsSync(accessFile))
  fs.writeFileSync(
    accessFile,
    JSON.stringify({
      key: existingInstall ? key : crypto.randomBytes(32).toString("base64url"),
    }),
    { mode: 0o600, flag: "wx" },
  );
const engineKey = () => {
  try {
    const k = JSON.parse(
      fs.readFileSync(path.join(root, "api-access.json"), "utf8"),
    ).key;
    return k || key;
  } catch {
    return key;
  }
};
const syncEngineKeyFile = () => {
  try {
    fs.writeFileSync(path.join(root, "engine-key.current"), engineKey(), {
      mode: 0o600,
    });
  } catch {}
};
syncEngineKeyFile();
const originalTests = new OriginalTests(root, store, engineKey);
const standalone =
  process.env.SM75_SINGLE_CONTAINER === "1" ? new Standalone(root, key) : null;
const attachments = new Attachments(root);
const profilesPath = path.join(root, "profiles.json");
let profiles = fs.existsSync(profilesPath)
  ? JSON.parse(fs.readFileSync(profilesPath))
  : [];
const ledger = new JobLedger(store),
  jobs = ledger.jobs;
let locks = new Set(),
  native = new Map();
const nativeCaches = new Map();
const pendingStarts = new Set();
let shuttingDown = false;
const history = new Map();
const telemetry = new Telemetry(path.join(root, "metrics"));
const host = process.env.SM75_CONSOLE_HOST || "127.0.0.1",
  port = Number(process.env.SM75_CONSOLE_PORT || 1616);
const harnessName = "sm75-v015-test-harness",
  harnessPort = Number(process.env.SM75_HARNESS_PORT || 3085),
  proxyPort = Number(process.env.SM75_HARNESS_PROXY_PORT || 1617);
let harnessCookie = "",
  harnessAuthTask = null;
async function nativeHarnessCookie() {
  if (harnessCookie) return harnessCookie;
  if (!harnessAuthTask)
    harnessAuthTask = (async () => {
      const log = standalone
        ? {
            stdout: fs.existsSync(standalone.harnessLog())
              ? fs.readFileSync(standalone.harnessLog(), "utf8")
              : "",
            stderr: "",
          }
        : await docker(["logs", "--tail", "80", harnessName]);
      const tokens = [
        ...(log.stdout + log.stderr).matchAll(/[?&]token=([A-Za-z0-9_-]+)/g),
      ];
      const token = tokens.at(-1)?.[1];
      if (!token) throw Error("Harness 尚未就绪");
      const r = await fetch(
        `http://127.0.0.1:${harnessPort}/?token=${encodeURIComponent(token)}`,
        { redirect: "manual", signal: AbortSignal.timeout(5000) },
      );
      harnessCookie = r.headers
        .getSetCookie()
        .map((c) => c.split(";")[0])
        .join("; ");
      if (!harnessCookie) throw Error("Harness 认证握手未完成");
      return harnessCookie;
    })().finally(() => (harnessAuthTask = null));
  return harnessAuthTask;
}
const json = (res, data, code = 200) => {
  res.writeHead(code, {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(data));
};
async function body(req) {
  let b = "";
  for await (const c of req) {
    b += c;
    if (b.length > 1_000_000) throw Error("请求过大");
  }
  return JSON.parse(b || "{}");
}
const equal = equalSecret;
function logged(req) {
  return !!auth.principal(req);
}
function save() {
  fs.writeFileSync(profilesPath + ".tmp", JSON.stringify(profiles, null, 2), {
    mode: 0o600,
  });
  fs.renameSync(profilesPath + ".tmp", profilesPath);
}
function profile(id) {
  const p = profiles.find((p) => p.id === id);
  if (!p) throw Error("配置不存在");
  return p;
}
const docker = (args) =>
  exec("docker", args, { maxBuffer: 4 * 1024 * 1024, timeout: 15000 });
async function inspect(name) {
  try {
    return JSON.parse((await docker(["inspect", name])).stdout)[0];
  } catch {
    return null;
  }
}
async function managed(id) {
  const name = makeCommand(profile(id), key).name,
    c = await inspect(name);
  if (c && c.Config.Labels?.["sm75.managed"] !== "v015-candidate")
    throw Error("拒绝操作非候选容器");
  return { name, c };
}
function launchJob(
  kind,
  bin,
  args,
  env = {},
  container = null,
  download = null,
) {
  const j = ledger.create(kind, { container, download });
  const c = spawn(bin, args, {
    env: { ...process.env, ...env },
    windowsHide: true,
  });
  j.child = c;
  j.command = { bin, args, env };
  const record = (d) => {
    j.log = (j.log + d.toString()).slice(-64000);
    ledger.persist(j);
  };
  c.stdout.on("data", record);
  c.stderr.on("data", record);
  c.on("error", (e) => {
    j.state = "failed";
    j.finished = Date.now();
    record(e.message);
  });
  c.on("close", (code) => {
    j.exitCode = code;
    if (j.state !== "cancelled") j.state = code === 0 ? "complete" : "failed";
    if (code === 0 && j.state === "complete" && j.download?.destination) {
      try {
        store.registerModel(j.download.destination, j.download.repo);
      } catch (e) {
        j.state = "failed";
        record(e.message);
      }
    }
    j.finished = Date.now();
    ledger.persist(j);
  });
  return j.id;
}
async function launchDownload(repo, provider, options = {}) {
  if (locks.has("model-delete")) throw Error("模型删除进行中");
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo || ""))
    throw Error("模型 ID 应为 owner/name");
  if (!["huggingface", "modelscope"].includes(provider))
    throw Error("请选择下载源");
  const duplicate = [...jobs.values()].find(
    (j) =>
      j.kind === "download" &&
      j.state === "running" &&
      j.download?.repo === repo &&
      j.download?.provider === provider,
  );
  if (duplicate) return { id: duplicate.id };
  const modelRoot = options.modelRoot || store.settings().modelRoot;
  fs.mkdirSync(modelRoot, { recursive: true });
  const destination = path.join(modelRoot, provider, repo.replace("/", "--"));
  fs.mkdirSync(destination, { recursive: true });
  const py =
    provider === "modelscope"
      ? "from modelscope import snapshot_download; snapshot_download(model_id=repo,local_dir=dest)"
      : "from huggingface_hub import snapshot_download; snapshot_download(repo_id=repo,local_dir=dest)";
  const script = `import sys\nrepo=sys.argv[1]\ndest='/downloads/'+repo.replace('/','--')\n${py}`;
  if (standalone) {
    const localScript = script.replace(
      "dest='/downloads/'+repo.replace('/','--')",
      "dest=sys.argv[2]",
    );
    const sdkRoot = path.join(modelRoot, ".sdk-cache");
    const id = launchJob(
      "download",
      "python3",
      ["-c", localScript, repo, destination],
      {
        HF_HOME: path.join(sdkRoot, "huggingface"),
        HF_HUB_CACHE: path.join(sdkRoot, "huggingface", "hub"),
        HF_XET_CACHE: path.join(sdkRoot, "huggingface", "xet"),
        MODELSCOPE_CACHE: path.join(sdkRoot, "modelscope"),
      },
      null,
      { repo, provider, modelRoot, destination },
    );
    return { id };
  }
  const container = "sm75-v015-test-download-" + crypto.randomUUID();
  await docker([
    "create",
    "--rm",
    "--name",
    container,
    "--label",
    "sm75.managed=v015-candidate",
    "--volume",
    `${modelRoot}:/downloads`,
    "--entrypoint",
    "python3",
    "local/vllm-sm75:v015-candidate-20260912",
    "-c",
    script,
    repo,
  ]);
  const id = launchJob(
    "download",
    "docker",
    ["start", "--attach", container],
    {},
    container,
    { repo, provider, modelRoot, destination },
  );
  return { id };
}
let hardwareSnapshot = null,
  hardwareAt = 0,
  hardwarePending = null;
async function hardware(p2p = false) {
  if (!p2p && hardwareSnapshot && Date.now() - hardwareAt < 3000)
    return hardwareSnapshot;
  if (!p2p && hardwarePending) return hardwarePending;
  const read = async () => {
    const r = await exec(
      "python3",
      [path.join(here, "hardware.py"), ...(p2p ? ["--p2p"] : [])],
      { timeout: 3000 },
    );
    const result = JSON.parse(r.stdout);
    if (!p2p) {
      hardwareSnapshot = result;
      hardwareAt = Date.now();
    }
    return result;
  };
  if (p2p) return read();
  hardwarePending = read();
  try {
    return await hardwarePending;
  } finally {
    hardwarePending = null;
  }
}
async function gpu() {
  try {
    return (await hardware()).gpus;
  } catch {
    return [];
  }
}
async function status(p) {
  if (standalone) return standalone.status(p);
  if (p.backend === "native")
    return { running: native.has(p.id), name: makeCommand(p, key).name };
  const { name, c } = await managed(p.id);
  return {
    name,
    running: !!c?.State.Running,
    state: c?.State.Status || "not-created",
  };
}
function endpoint(p) {
  return `http://127.0.0.1:${p.port}`;
}
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
async function waitHarnessReady() {
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    try {
      const cookie = await nativeHarnessCookie();
      if (standalone) {
        const response = await fetch(`http://127.0.0.1:${harnessPort}/sm75/ready`, {
          headers: {cookie}, signal: AbortSignal.timeout(5000),
        });
        if (!response.ok || !(await response.json()).ready) throw Error("Harness model setup pending");
      }
      return;
    } catch {
      await sleep(500);
    }
  }
  throw Error("Harness 启动未就绪，请查看日志后重试");
}
function unifiedProfile(input) {
  const p = structuredClone(input);
  if (standalone) {
    p.port = 8000;
    const args = p.args;
    for (const [flag, value] of [
      ["--host", "0.0.0.0"],
      ["--port", "8000"],
    ]) {
      const i = args.indexOf(flag);
      if (i >= 0) args[i + 1] = value;
      else args.push(flag, value);
    }
    p.cacheRoot = store.settings().cacheRoot;
  }
  return p;
}
const weightSizeCache = new Map();
function weightGiB(dir) {
  try {
    const real = fs.realpathSync(dir),
      st = fs.statSync(real),
      hit = weightSizeCache.get(real);
    if (hit && hit.mtime === st.mtimeMs) return hit.gib;
    let total = 0;
    const walk = (d, depth) => {
      if (depth > 5) return;
      let ents;
      try {
        ents = fs.readdirSync(d, { withFileTypes: true });
      } catch {
        return;
      }
      for (const e of ents) {
        if (e.name === ".git") continue;
        const fp = path.join(d, e.name);
        if (e.isDirectory()) walk(fp, depth + 1);
        else if (/\.(safetensors|bin|pt|pth|gguf|onnx)$/i.test(e.name)) {
          try {
            total += fs.statSync(fp).size;
          } catch {}
        }
      }
    };
    walk(real, 0);
    const gib = Math.round((total / 2 ** 30) * 10) / 10;
    weightSizeCache.set(real, { mtime: st.mtimeMs, gib });
    return gib;
  } catch {
    return null;
  }
}
function modelCatalog() {
  const found = new Map();
  const add = (dir, name, format, role = "main") => {
    try {
      const actual = fs.realpathSync(dir),
        file = path.join(actual, "config.json");
      if (!fs.existsSync(file)) return;
      const cfg = JSON.parse(fs.readFileSync(file));
      const quant = String(cfg.quantization_config?.quant_method || "");
      const kind =
        format ||
        (quant.includes("awq") ? "awq" : quant.includes("fp8") ? "fp8" : "kat");
      const old = found.get(actual);
      found.set(actual, {
        ...old,
        id: crypto
          .createHash("sha256")
          .update(actual)
          .digest("hex")
          .slice(0, 16),
        name: name || old?.name || path.basename(actual),
        path: dir,
        format: kind,
        modelType: cfg.model_type,
        context: modelContext(cfg),
        weightGiB: weightGiB(actual),
        role: old?.role === "main" ? "main" : role,
      });
    } catch {}
  };
  for (const p of profiles) {
    add(p.args[0], null, p.format);
    const i = p.args.indexOf("--speculative-config");
    try {
      const spec = JSON.parse(p.args[i + 1]);
      if (spec.model) add(spec.model, null, null, "draft");
    } catch {}
  }
  for (const m of store.list("model")) add(m.path, m.name, m.format, m.role);
  const root = store.settings().modelRoot;
  fs.mkdirSync(root, { recursive: true });
  for (const d of fs.readdirSync(root, { withFileTypes: true }))
    if (d.isDirectory()) add(path.join(root, d.name));
  return [...found.values()];
}
async function start(p) {
  if (shuttingDown) throw Error("管理服务正在关闭");
  p = unifiedProfile(p);
  if (locks.has("model-switch") || locks.has(p.id) || locks.has("model-delete"))
    throw Error("操作进行中");
  locks.add(p.id);
  locks.add("model-switch");
  let completeStart;
  const finished = new Promise(resolve => {completeStart = resolve;});
  pendingStarts.add(finished);
  try {
    const st = await status(p);
    if (st.running) return st;
    // Check optional dependencies and disk before switchModel can stop the current engine.
    const cacheCheck = await preflightLMCache(p, {standalone: !!standalone,
      ownedPorts: standalone?.lmcache.child ? standalone.lmcache.ports : []});
    if (shuttingDown) throw Error("管理服务正在关闭");
    if (standalone)
      return await switchModel(p, {
        current: standalone.engine,
        stop: (old) => standalone.stop(old),
        start: (target) => standalone.start(target),
        gpus: gpu,
      });
    const devices = await gpu();
    if (p.backend !== "native" && !devices.length)
      throw Error("无法读取 GPU 状态，未启动模型");
    const busy = devices.filter((g) => g.usedMiB > 1000);
    if (busy.length) throw Error("GPU 正被使用，候选服务未启动");
    const cmd = makeCommand(p, key, {
      mkdir: true,
      vllmBin: process.env.SM75_VLLM_BIN,
    });
    if (p.backend === "native") {
      const cache = new ManagedLMCache({logRoot: root});
      nativeCaches.set(p.id, cache);
      let cacheFailed = false, engineChild, log, cacheRuntime;
      try {
        if (shuttingDown) throw Error("管理服务正在关闭");
        cacheRuntime = await cache.start(p, {runtimeFingerprint: cacheCheck?.runtimeFingerprint});
        if (!cacheRuntime) nativeCaches.delete(p.id);
        if (cacheRuntime) cacheRuntime.child.once("exit", () => {
          cacheFailed = true;
          if (engineChild && cache.child === cacheRuntime.child && native.get(p.id) === engineChild)
            terminateLMCacheProcess(engineChild).catch(() => {});
        });
        if (shuttingDown) throw Error("管理服务正在关闭");
        if (cacheRuntime && (cacheRuntime.child.exitCode !== null || cacheRuntime.child.signalCode != null))
          throw Error("LMCache 服务在模型启动前退出");
        log = fs.openSync(path.join(root, p.id + ".log"), "a");
        await new Promise((resolve, reject) => {
          const c = spawn(cmd.bin, cmd.args, {
            env: { ...process.env, ...cmd.env, ...(cacheRuntime ? lmcacheEnvironment : {}) },
            detached: !!cacheRuntime,
            stdio: ["ignore", log, log],
          });
          engineChild = c;
          c.once("spawn", () => {
            if (shuttingDown || (cacheRuntime && (cacheFailed || cacheRuntime.child.exitCode !== null || cacheRuntime.child.signalCode != null))) {
              const stopped = cacheRuntime ? terminateLMCacheProcess(c) : new Promise(resolve => {c.once("exit", resolve); c.kill("SIGTERM");});
              stopped.then(() => reject(Error("管理服务关闭或 LMCache 在模型启动期间退出")), reject);
              return;
            }
            native.set(p.id, c);
            resolve();
          });
          c.once("exit", () => {
            native.delete(p.id);
            cache.stop().catch(() => {}).finally(() => {
              if (nativeCaches.get(p.id) === cache) nativeCaches.delete(p.id);
            });
          });
          c.once("error", (error) => {
            native.delete(p.id);
            fs.appendFileSync(
              path.join(root, p.id + ".log"),
              `启动失败：${error.message}\n`,
            );
            reject(error);
          });
        });
      } catch (error) {
        await cache.stop();
        nativeCaches.delete(p.id);
        throw error;
      } finally {
        if (log !== undefined) fs.closeSync(log);
      }
    } else {
      const { c } = await managed(p.id);
      if (c) await docker(["rm", cmd.name]);
      await exec(cmd.bin, cmd.args, {
        env: { ...process.env, ...cmd.env },
        timeout: 30000,
      });
    }
    return { started: true };
  } finally {
    pendingStarts.delete(finished);
    completeStart();
    locks.delete(p.id);
    locks.delete("model-switch");
  }
}
async function stop(p) {
  if (standalone) return standalone.stop(p);
  if (p.backend === "native") {
    if (locks.has("model-switch") || locks.has(p.id) || locks.has("model-delete"))
      throw Error("操作进行中");
    locks.add(p.id);
    locks.add("model-switch");
    try {
      const child = native.get(p.id), cache = nativeCaches.get(p.id);
      if (cache && child) await terminateLMCacheProcess(child);
      else child?.kill("SIGTERM");
      if (cache) {
        await cache.stop();
        if (nativeCaches.get(p.id) === cache) nativeCaches.delete(p.id);
      }
      return;
    } finally {
      locks.delete(p.id);
      locks.delete("model-switch");
    }
  }
  const { name, c } = await managed(p.id);
  if (c?.State.Running) await docker(["stop", "--time", "10", name]);
}
async function proxy(req, res, p, suffix, payload) {
  const abort = new AbortController();
  res.on("close", () => {
    if (!res.writableEnded) abort.abort();
  });
  const upstream = await fetch(endpoint(p) + suffix, {
    method: payload ? "POST" : "GET",
    headers: {
      Authorization: `Bearer ${engineKey()}`,
      "Content-Type": "application/json",
    },
    body: payload ? JSON.stringify(payload) : undefined,
    signal: abort.signal,
  });
  res.writeHead(upstream.status, {
    "Content-Type": upstream.headers.get("content-type") || "text/plain",
    "Cache-Control": "no-store",
  });
  if (upstream.body)
    for await (const chunk of upstream.body) {
      if (res.destroyed) break;
      res.write(chunk);
    }
  res.end();
}
const server = http.createServer(async (req, res) => {
  try {
    if (!auth.networkAllowed(req)) return json(res, { error: "当前地址不在允许网段内" }, 403);
    const u = new URL(req.url, "http://local");
    if (u.pathname.startsWith("/brand/")) {
      const name = u.pathname.slice(7);
      if (!/^[a-z0-9.-]+$/i.test(name))
        return json(res, { error: "路径不存在" }, 404);
      const file = path.join(here, "branding", name);
      if (!fs.existsSync(file)) return json(res, { error: "路径不存在" }, 404);
      const type = name.endsWith(".svg")
        ? "image/svg+xml"
        : name.endsWith(".ico")
          ? "image/x-icon"
          : name.endsWith(".png")
            ? "image/png"
            : name.endsWith(".webmanifest")
              ? "application/manifest+json"
              : null;
      if (!type) return json(res, { error: "路径不存在" }, 404);
      res.writeHead(200, { "Content-Type": type, "Cache-Control": "no-cache" });
      return res.end(fs.readFileSync(file));
    }
    if (u.pathname.startsWith("/bench-app/")) {
      if (!logged(req)) return json(res, { error: "请先登录" }, 401);
      auth.track(req, res);
      if (!auth.originAllowed(req))
        return json(res, { error: "来源不匹配" }, 403);
      const m = u.pathname.match(/^\/bench-app\/(speedtest|sql)(\/.*)$/);
      if (!m) return json(res, { error: "路径不存在" }, 404);
      const p = profile(
        u.searchParams.get("profile") ||
          store.settings().defaultProfile ||
          profiles[0]?.id,
      );
      return await originalTests.handle(req, res, {
        type: m[1],
        suffix: m[2] + (u.search || ""),
        profile: p,
        ip: uHost(req),
      });
    }
    if (u.pathname === "/dsh" || u.pathname === "/dsh/") {
      res.writeHead(308, {Location: "/" + u.search, "Cache-Control": "no-store"});
      return res.end();
    }
    if (
      u.pathname.startsWith("/dsh/") ||
      u.pathname.startsWith("/harness-ui/") ||
      u.pathname.startsWith("/plugins/") ||
      u.pathname.startsWith("/assets/") ||
      ["/favicon.svg", "/manifest.webmanifest"].includes(u.pathname) ||
      u.pathname.startsWith("/api/")
    ) {
      return bridge.emit("request", req, res);
    }
    if (u.pathname.startsWith("/console-api/"))
      u.pathname = u.pathname.replace("/console-api/", "/api/");
    if (u.pathname === "/api/login" && req.method === "POST") {
      if (!auth.originAllowed(req))
        return json(res, { error: "来源不匹配" }, 403);
      const payload = await body(req),
        result = payload.username !== undefined
          ? await auth.loginAccount(payload.username, payload.password, req)
          : auth.login(payload.token, req);
      if (result.cookie) res.setHeader("Set-Cookie", result.cookie);
      if (result.status === 429) res.setHeader("Retry-After", "60");
      return json(
        res,
        result.status === 200
          ? { ok: true, expiresAt: result.expiresAt }
          : { error: result.error },
        result.status,
      );
    }
    if (u.pathname === "/api/session" && req.method === "GET") {
      const principal = auth.principal(req);
      return json(res, { authenticated: !!principal, principal, accountConfigured: !!auth.policy.account });
    }
    if (u.pathname === "/api/logout" && req.method === "POST") {
      if (!auth.originAllowed(req))
        return json(res, { error: "来源不匹配" }, 403);
      auth.revoke(auth.sessionId(req));
      res.setHeader("Set-Cookie", auth.cookie("", req, 0));
      return json(res, { ok: true });
    }
    if (u.pathname === "/api/session/stream" && req.method === "GET") {
      if (!logged(req)) return json(res, { error: "请先登录" }, 401);
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-store",
      });
      res.write('data: {"authenticated":true}\n\n');
      auth.track(req, res);
      const heartbeat = setInterval(() => {
        try {
          if (!auth.principal(req))
            res.end('data: {"authenticated":false}\n\n');
          else res.write(": keepalive\n\n");
        } catch {
          res.destroy();
        }
      }, 15000);
      heartbeat.unref();
      res.on("close", () => clearInterval(heartbeat));
      return;
    }
    if (
      u.pathname === "/console-ui" ||
      u.pathname === "/" ||
      u.pathname === "/app.js" ||
      u.pathname === "/live-summary.js" ||
      u.pathname === "/context-config.js" ||
      u.pathname === "/lmcache-config.js" ||
      u.pathname === "/style.css"
    ) {
      const f = {
        "/console-ui": "index.html",
        "/": "index.html",
        "/app.js": "app.js",
        "/live-summary.js": "live-summary.js",
        "/context-config.js": "context-config.js",
        "/lmcache-config.js": "lmcache-config.js",
        "/style.css": "style.css",
      }[u.pathname];
      res.writeHead(200, {
        "Content-Type": f.endsWith(".html")
          ? "text/html; charset=utf-8"
          : f.endsWith(".js")
            ? "text/javascript"
            : "text/css",
        "Cache-Control": "no-store",
      });
      return res.end(fs.readFileSync(path.join(here, "public", f)));
    }
    if (!logged(req)) return json(res, { error: "请先登录" }, 401);
    auth.track(req, res);
    if (req.method !== "GET" && !auth.originAllowed(req))
      return json(res, { error: "来源不匹配" }, 403);
    if (u.pathname === "/api/attachments" && req.method === "POST") {
      const p = profile(u.searchParams.get("profile"));
      if (!chatConfig(p).input.includes("image")) return json(res, {error:"当前模型未启用图片输入"},422);
      try { return json(res, await attachments.upload(req,p.id)); }
      catch(error) { return json(res,{error:error.message},400); }
    }
    const attachmentPath = u.pathname.match(/^\/api\/attachments\/([a-f0-9]{32})$/);
    if (attachmentPath && req.method === "GET") {
      try {
        const {meta,data}=attachments.read(attachmentPath[1]);
        res.writeHead(200,{"Content-Type":meta.mime,"Cache-Control":"private, no-store","X-Content-Type-Options":"nosniff"});
        return res.end(data);
      } catch { return json(res,{error:"附件不存在"},404); }
    }
    const b = req.method === "POST" ? await body(req) : {};
    if (u.pathname === "/api/security") {
      if (req.method === "GET") return json(res, auth.securityInfo(req));
      if (req.method === "POST") {
        try { return json(res, await auth.updateSecurity(b, req)); }
        catch (error) { return json(res, { error: error.message }, 400); }
      }
      return json(res, { error: "方法不支持" }, 405);
    }
    if (u.pathname === "/api/workspace-file" && req.method === "GET") {
      const raw = (u.search.match(/[?&]path=([^&]*)/) || [])[1];
      const rel = raw ? decodeURIComponent(raw) : "";
      const roots = ["/dsh/workspace/", "/dsh/home/"];
      const abs = fs.realpathSync(
        path.normalize(rel.startsWith("/") ? rel : path.join(roots[0], rel)),
      );
      if (!roots.some((r) => abs.startsWith(r))) throw Error("路径不允许");
      if (!fs.existsSync(abs) || !fs.statSync(abs).isFile())
        throw Error("文件不存在");
      const st = fs.statSync(abs),
        name = path.basename(abs);
      res.writeHead(200, {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": `attachment; filename="${name.replace(/[^\x20-\x7e]/g, "_")}"; filename*=UTF-8''${encodeURIComponent(name)}`,
        "Content-Length": st.size,
        "Cache-Control": "no-store",
      });
      const stream = fs.createReadStream(abs);
      stream.on("error", () => res.destroy());
      res.on("close", () => stream.destroy());
      stream.pipe(res);
      return;
    }
    if (u.pathname === "/api/benchmarks/extension") {
      const type = u.searchParams.get("type") === "sql" ? "sql" : "speedtest",
        dir = path.join(root, "extensions", type),
        manifest = path.join(dir, "manifest.json");
      let setting = store.get("extension", type) || { enabled: false };
      const installing = [...jobs.values()].find(
        (j) => j.kind === "benchmark-install-" + type && j.state === "running",
      );
      if (req.method === "POST") {
        if (typeof b.enabled !== "boolean") throw Error("开关格式错误");
        if (
          !b.enabled &&
          [...jobs.values()].some(
            (j) => j.kind === "benchmark" && j.state === "running",
          )
        )
          throw Error("请先取消正在运行的测试");
        setting = store.put("extension", type, { enabled: b.enabled });
        if (!b.enabled) originalTests.stop(type);
        if (b.enabled && !originalTests.ready(type) && !installing)
          launchJob("benchmark-install-" + type, "python3", [
            path.join(
              here,
              type === "sql"
                ? "install-sql-benchmark.py"
                : "install-benchmark.py",
            ),
            dir,
          ]);
      }
      return json(res, {
        ...setting,
        installed: originalTests.ready(type),
        installing: [...jobs.values()].some(
          (j) =>
            j.kind === "benchmark-install-" + type && j.state === "running",
        ),
        manifest: fs.existsSync(manifest)
          ? JSON.parse(fs.readFileSync(manifest))
          : null,
      });
    }
    if (u.pathname === "/api/benchmarks") {
      if (req.method === "POST") {
        const type = b.type === "sql" ? "sql" : "speedtest";
        if (!store.get("extension", type)?.enabled)
          throw Error("请先开启测速扩展");
        const extension = path.join(root, "extensions", type);
        if (!fs.existsSync(path.join(extension, "manifest.json")))
          throw Error("测速组件尚未部署完成");
        if (
          [...jobs.values()].some(
            (j) => j.kind === "benchmark" && j.state === "running",
          )
        )
          throw Error("已有测试正在运行");
        const p = profile(b.profile);
        if (!(await status(p)).running) throw Error("请先启动当前模型");
        if (
          type === "sql" &&
          (!Array.isArray(b.questions) ||
            !b.questions.length ||
            b.questions.length > 25 ||
            b.questions.some((x) => !Number.isInteger(x) || x < 1 || x > 25))
        )
          throw Error("SQL 题号范围 1–25");
        const lengths = type === "sql" ? [32] : b.lengths;
        if (
          !Array.isArray(lengths) ||
          !lengths.length ||
          lengths.length > 8 ||
          lengths.some((x) => !Number.isInteger(x) || x < 32 || x > 131072)
        )
          throw Error("输入长度范围 32–131072，最多 8 组");
        for (const [field, min, max] of [
          ["output", 16, 4096],
          ["concurrency", 1, 16],
          ["repeats", 1, 5],
          ["timeout", 30, 1800],
        ])
          if (!Number.isInteger(b[field]) || b[field] < min || b[field] > max)
            throw Error(field + " 超出范围");
        if (lengths.length * b.concurrency * b.repeats > 128)
          throw Error("单次最多 128 个请求");
        const ci = p.args.indexOf("--max-model-len"),
          context = Number(p.args[ci + 1]);
        if (
          ci >= 0 &&
          Number.isFinite(context) &&
          Math.max(...lengths) + b.output + 512 > context
        )
          throw Error("输入与输出超过配置上下文上限");
        const models = await fetch(endpoint(p) + "/v1/models", {
          headers: { Authorization: `Bearer ${engineKey()}` },
          signal: AbortSignal.timeout(5000),
        }).then((r) => r.json());
        const model = models.data?.[0]?.id;
        if (!model) throw Error("模型尚未就绪");
        const id = crypto.randomUUID(),
          dir = path.join(root, "benchmarks");
        fs.mkdirSync(dir, { recursive: true });
        const config = {
          id,
          type,
          questions: type === "sql" ? [...new Set(b.questions)] : undefined,
          profile: p.id,
          profileSnapshot: p,
          lengths,
          output: b.output,
          concurrency: b.concurrency,
          repeats: b.repeats,
          timeout: b.timeout,
          thinking: !!b.thinking,
          warmup: b.warmup !== false,
          model,
          url: endpoint(p) + "/v1/chat/completions",
          created: Date.now(),
          source: JSON.parse(
            fs.readFileSync(path.join(extension, "manifest.json")),
          ),
        };
        const file = path.join(dir, id + ".json");
        fs.writeFileSync(file, JSON.stringify(config), { mode: 0o600 });
        const job = launchJob(
          "benchmark",
          type === "sql" ? process.execPath : "python3",
          [
            path.join(
              here,
              type === "sql"
                ? "sql-benchmark-runner.mjs"
                : "benchmark-runner.py",
            ),
            extension,
            file,
          ],
          { SM75_BENCH_KEY: engineKey(), PYTHONUNBUFFERED: "1" },
        );
        store.put("benchmark", id, { ...config, job });
        return json(res, { id, job });
      }
      return json(
        res,
        store
          .list("benchmark")
          .sort((a, b) => b.created - a.created)
          .map((v) => {
            const f = path.join(
              root,
              "benchmarks",
              v.id + ".json.results.json",
            );
            return {
              ...v,
              state: jobs.get(v.job)?.state || "interrupted",
              results: fs.existsSync(f) ? JSON.parse(fs.readFileSync(f)) : null,
            };
          }),
      );
    }
    if (u.pathname === "/api/model-search")
      return json(
        res,
        await searchModels(
          u.searchParams.get("provider"),
          u.searchParams.get("q"),
          u.searchParams.get("page") || 1,
        ),
      );
    if (u.pathname === "/api/api-access") {
      const file = path.join(root, "api-access.json");
      if (req.method === "POST") {
        if (
          typeof b.key !== "string" ||
          b.key.length < 8 ||
          b.key.length > 512 ||
          /[\s\x00-\x1f]/.test(b.key)
        )
          throw Error("API Key 需要 8–512 位且不含空白");
        fs.writeFileSync(file, JSON.stringify({ key: b.key }), { mode: 0o600 });
        fs.chmodSync(file, 0o600);
        syncEngineKeyFile();
        return json(res, { configured: true, restartRequired: true });
      }
      return json(res, { configured: fs.existsSync(file) });
    }
    if (u.pathname === "/api/settings") {
      return json(
        res,
        req.method === "POST"
          ? store.saveSettings(b, profiles)
          : store.settings(),
      );
    }
    if (u.pathname === "/api/templates") {
      if (req.method === "POST")
        return json(
          res,
          store.saveTemplate(b.name, validateProfile(b.profile), b.id),
        );
      return json(res, {
        recommended: presets,
        personal: store.list("template"),
      });
    }
    if (u.pathname === "/api/templates/apply" && req.method === "POST")
      return json(res, { args: applyPreset(b.args, b.id) });
    if (u.pathname === "/api/sampling") {
      const data = store.get("sampling", "main") || {
        templates: [],
        models: {},
      };
      if (req.method === "POST") {
        if (b.action === "template") {
          if (
            typeof b.name !== "string" ||
            !b.name.trim() ||
            b.name.length > 80
          )
            throw Error("填写模板名称（最多 80 字）");
          const params = validateSampling(b.params);
          const old = data.templates.find((t) => t.name === b.name.trim());
          if (old) old.params = params;
          else
            data.templates.push({
              id: crypto.randomUUID(),
              name: b.name.trim(),
              params,
            });
        } else if (b.action === "model") {
          if (typeof b.model !== "string" || !b.model.includes("/"))
            throw Error("选择模型");
          data.models[b.model] = validateSampling(b.params);
        } else throw Error("无效操作");
        store.put("sampling", "main", data);
        if (standalone) {
          fs.mkdirSync("/dsh/home", { recursive: true });
          writeHarnessFile("/dsh/home/sm75-sampling.json", JSON.stringify(data));
        }
      }
      let configured = [];
      if (standalone)
        try {
          const response = await fetch(`http://127.0.0.1:${harnessPort}/sm75/configured-models`, {
            headers: {cookie: await nativeHarnessCookie()}, signal: AbortSignal.timeout(5000),
          });
          if (!response.ok) throw Error("Harness model settings unavailable");
          const snapshot = await response.json();
          configured = snapshot.models.map(({model, config, ...row}) => ({
            ...row, defaults: modelSamplingDefaults(row.provider, config, model, profiles, standalone.engine?.id),
          }));
        } catch {}
      return json(res, { ...data, configured });
    }
    if (u.pathname === "/api/models/delete" && req.method === "POST") {
      if (locks.size) throw Error("模型操作进行中，请稍后重试");
      locks.add("model-delete");
      try {
        const model = modelCatalog().find((m) => m.id === b.id);
        if (!model) throw Error("模型不存在");
        const running = [];
        for (const p of profiles)
          if ((await status(p)).running) running.push(p);
        const result = deleteModelFiles(model, b.path, {
          running,
          downloads: [...jobs.values()]
            .filter((j) => j.state === "running" && j.download)
            .map((j) => j.download.destination),
          protectedPaths: [
            root,
            store.settings().modelRoot,
            store.settings().cacheRoot,
          ],
        });
        store.db
          .prepare("DELETE FROM records WHERE kind=? AND id=?")
          .run("model", model.id);
        return json(res, result);
      } finally {
        locks.delete("model-delete");
      }
    }
    if (u.pathname === "/api/models/register" && req.method === "POST")
      return json(res, store.registerModel(b.path, b.name));
    if (u.pathname === "/api/models/use" && req.method === "POST") {
      const model = modelCatalog().find((m) => m.id === b.model);
      if (!model) throw Error("本地模型不存在");
      const original = b.template?.startsWith("personal:")
        ? store.get("template", b.template.slice(9))?.profile
        : profile(b.template);
      if (!original) throw Error("模板不存在");
      if (model.format !== original.format)
        throw Error("模型格式与配置模板不匹配");
      let p;
      if (
        original.args[0] === model.path &&
        profiles.some((p) => p.id === original.id)
      ) {
        p = profile(original.id);
      } else {
        p = unifiedProfile({
          ...structuredClone(original),
          id: "personal-" + crypto.randomUUID().slice(0, 8),
          name: model.name + " · " + (original.name || "我的配置"),
        });
        p.args = retargetContextModel(p.args, model.path, model.context);
        validateProfile(p);
        profiles.push(p);
        save();
      }
      const i = p.args.indexOf("--speculative-config");
      if (i >= 0) {
        const spec = JSON.parse(p.args[i + 1]);
        if (
          spec.method === "dflash" &&
          !fs.existsSync(path.join(spec.model || "", "config.json"))
        )
          throw Error("请先在模型库下载或导入匹配的 DFLASH2 草稿模型");
      }
      return json(res, { profile: p.id, ...(b.start ? await start(p) : {}) });
    }
    if (u.pathname === "/api/power/p8" && req.method === "POST") {
      const cur = standalone?.engine?.id;
      const prof = cur ? profiles.find((p) => p.id === cur) : null;
      if (!prof || prof.power?.mode !== "pstate")
        throw Error("当前运行配置不是 P-State 模式");
      fs.writeFileSync(
        "/tmp/pstate-command.json",
        JSON.stringify({ action: "p8", at: Date.now() }),
        { mode: 0o600 },
      );
      return json(res, { ok: true });
    }
    if (u.pathname === "/api/power") {
      let pst = standalone ? standalone.powerState() : null;
      const cur = standalone?.engine?.id;
      const prof = cur ? profiles.find((p) => p.id === cur) : null;
      if (!prof || prof.power?.mode !== "pstate") pst = null;
      return json(res, {
        state: pst,
        nvapi: fs.existsSync("/usr/local/nvidia/lib64/libnvidia-api.so.1"),
      });
    }
    if (u.pathname === "/api/profiles") {
      if (req.method === "POST") {
        const p = validateProfile(unifiedProfile(b));
        const old = profiles.findIndex((x) => x.id === p.id);
        if (old >= 0 && (await status(profiles[old])).running)
          throw Error("先停止候选模型再修改配置");
        if (old >= 0) profiles[old] = p;
        else profiles.push(p);
        save();
      }
      return json(res, profiles);
    }
    if (u.pathname === "/api/live-summary" && req.method === "GET") {
      const requested = u.searchParams.get("profile");
      const id =
        requested || standalone?.engine?.id || store.settings().defaultProfile;
      const p = id ? profiles.find((p) => p.id === id) : null;
      if (requested && !p) return json(res, { error: "运行配置不存在" }, 404);
      const hw = await hardware().catch(() => null);
      const processStatus = p ? await status(p).catch(() => null) : { running: false };
      const result = liveSummary(p ? telemetry.read(p.id) : [], hw, {
        failed: p
          ? processStatus?.running !== true || telemetry.failed.has(p.id) ||
            (standalone && standalone.engine?.id !== p.id)
          : true,
      });
      res.setHeader("Cache-Control", "no-store");
      return json(res, { ...result, profile: p?.name || p?.id || null,
        running: processStatus?.running ?? null,
        powerMode: p?.power?.mode || null,
        powerState: processStatus?.running && standalone?.engine?.id === p?.id && p?.power?.mode === "pstate"
          ? standalone.powerState() : null,
      });
    }
    if (u.pathname === "/api/hardware")
      return json(res, await hardware(u.searchParams.get("p2p") === "1"));
    if (u.pathname === "/api/system")
      return json(res, {
        gpus: await gpu(),
        root,
        version: JSON.parse(fs.readFileSync(path.join(here, "package.json"), "utf8")).version,
        scope: standalone
          ? "单容器内模型与工作区进程"
          : "仅管理 sm75-v015-test-* 候选容器",
      });
    if (u.pathname === "/api/jobs") {
      if (req.method === "POST" && b.cancel) {
        const j = jobs.get(b.cancel);
        if (!j) throw Error("任务不存在");
        if (j.state === "running") {
          j.state = "cancelled";
          if (j.container) {
            const c = await inspect(j.container);
            if (c) {
              if (c.Config.Labels?.["sm75.managed"] !== "v015-candidate")
                throw Error("任务容器归属不符");
              await docker(["rm", "--force", j.container]);
            }
          }
          j.state = "cancelled";
          j.child?.kill("SIGTERM");
          j.finished = Date.now();
          ledger.persist(j);
        }
      }
      if (req.method === "POST" && b.retry) {
        const j = jobs.get(b.retry);
        if (!j || !["failed", "cancelled", "interrupted"].includes(j.state))
          throw Error("仅重试失败、已取消或中断任务");
        return json(
          res,
          j.download
            ? await launchDownload(
                j.download.repo,
                j.download.provider,
                j.download,
              )
            : (() => {
                if (!j.command) throw Error("请从原功能页面重新启动任务");
                return {
                  id: launchJob(
                    j.kind,
                    j.command.bin,
                    j.command.args,
                    j.command.env,
                    j.container,
                  ),
                };
              })(),
        );
      }
      return json(res, ledger.list());
    }
    if (u.pathname === "/api/models") return json(res, modelCatalog());
    const chatPath = u.pathname.match(/^\/api\/chats\/([a-z0-9-]+)$/);
    if (chatPath) {
      const p = profile(chatPath[1]),
        dir = path.join(root, "chats");
      fs.mkdirSync(dir, { recursive: true });
      const file = path.join(dir, p.id + ".json");
      if (req.method === "POST") {
        attachments.validate(b.messages, p.id);
        fs.writeFileSync(
          file + ".tmp",
          JSON.stringify({ messages: b.messages, updated: Date.now() }),
          { mode: 0o600 },
        );
        fs.renameSync(file + ".tmp", file);
      }
      return json(
        res,
        fs.existsSync(file)
          ? JSON.parse(fs.readFileSync(file))
          : { messages: [] },
      );
    }
    if (u.pathname === "/api/download" && req.method === "POST")
      return json(res, await launchDownload(b.repo, b.provider));
    const match = u.pathname.match(/^\/api\/profiles\/([a-z0-9-]+)\/(.+)$/);
    if (match) {
      const p = profile(match[1]),
        action = match[2];
      if (action === "status") return json(res, await status(p));
      if (action === "history")
        return json(res, { points: telemetry.read(p.id) });
      if (action === "preview" && standalone)
        return json(res, {
          bin: "vllm",
          args: ["serve", ...(lmcachePlan(p)?.engineArgs || p.args), "--port", String(p.port)],
          lmcache: lmcachePlan(p)?.summary || null,
          cache: cacheLayout(p.cacheRoot, p.format),
          backend: "in-container",
        });
      if (action === "preview") {
        const c = makeCommand(p, "<managed>", {
          vllmBin: process.env.SM75_VLLM_BIN,
        });
        return json(res, {
          bin: c.bin,
          args: c.args,
          lmcache: lmcachePlan(p)?.summary || null,
          cache: cacheLayout(p.cacheRoot, p.format),
        });
      }
      if (action === "start" && req.method === "POST")
        return json(res, await start(p));
      if (action === "stop" && req.method === "POST") {
        await stop(p);
        return json(res, { ok: true });
      }
      if (action === "logs" && standalone) {
        const offset = Number(u.searchParams.get("offset") || 0);
        return json(res, standalone.logs(p, offset));
      }
      if (action === "logs") {
        const { name, c } = await managed(p.id);
        const r = c ? await docker(["logs", name]) : {};
        return json(res, {
          text: c ? r.stdout + r.stderr : "尚未创建",
          offset: 0,
          size: 0,
          reset: true,
        });
      }
      if (action === "metrics") return await proxy(req, res, p, "/metrics");
      if (action === "monitor") {
        const html = fs
          .readFileSync(
            path.join(
              here,
              "../vllm/entrypoints/serve/instrumentator/dashboard.html",
            ),
            "utf8",
          )
          .replaceAll("'/metrics'", `'/console-api/profiles/${p.id}/metrics'`)
          .replaceAll(
            "'/monitor/spec_decode'",
            `'/console-api/profiles/${p.id}/spec'`,
          )
          .replace(
            "</head>",
            `<style>main{max-width:none;padding:0 0 24px}header>div:first-child{display:none}header{justify-content:flex-end;margin:0 0 16px}h2{font-size:16px}section{margin-top:20px}.card{border-radius:12px}</style></head>`,
          )
          .replace(
            "<script>",
            `<script>globalThis.SM75_HISTORY_URL='/console-api/profiles/${p.id}/history';`,
          );
        res.writeHead(200, {
          "Content-Type": "text/html; charset=utf-8",
          "Cache-Control": "no-store",
        });
        return res.end(html);
      }
      if (action === "chat" && req.method === "POST")
        return await proxy(req, res, p, "/v1/chat/completions", {
          ...b,
          messages: attachments.expand(b.messages,p.id,chatConfig(p).input.includes("image")),
          stream: true,
          stream_options: { include_usage: true },
        });
      if (action === "chat-config") return json(res, chatConfig(p));
      if (action === "models") return await proxy(req, res, p, "/v1/models");
      if (action === "spec") {
        if (req.method === "GET") {
          const r = await fetch(endpoint(p) + "/monitor/spec_decode", {
            signal: AbortSignal.timeout(8000),
          });
          const d = await r.json();
          return json(
            res,
            {
              ...d,
              desired_enabled:
                typeof d.desired_enabled === "boolean"
                  ? d.desired_enabled
                  : !!d.enabled,
              pending: !!d.pending,
              auth_required: false,
            },
            r.status,
          );
        }
        const r = await fetch(endpoint(p) + "/monitor/spec_decode", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-api-key-hash": crypto
              .createHash("sha256")
              .update(engineKey())
              .digest("hex"),
          },
          body: JSON.stringify(b),
          signal: AbortSignal.timeout(8000),
        });
        const d = await r.json();
        return json(
          res,
          {
            ...d,
            desired_enabled:
              typeof d.desired_enabled === "boolean"
                ? d.desired_enabled
                : !!d.enabled,
          },
          r.status,
        );
      }
    }
    if (standalone && u.pathname === "/api/harness")
      return json(res, {
        running: !!standalone.harness,
        version: harnessVersion(),
        backend: "in-container",
      });
    if (standalone && u.pathname === "/api/harness/install")
      return json(res, {
        installed: fs.existsSync(
          "/opt/harness/node_modules/@deepseek-ai/dsh/lib/bin.js",
        ),
      });
    if (
      standalone &&
      u.pathname === "/api/harness/stop" &&
      req.method === "POST"
    ) {
      await standalone.stopHarness();
      harnessCookie = "";
      return json(res, { ok: true });
    }
    if (
      standalone &&
      u.pathname === "/api/harness/start" &&
      req.method === "POST"
    ) {
      harnessCookie = "";
      await standalone.startHarness(profile(b.profile));
      await waitHarnessReady();
      return json(res, { url: "/harness-ui/" });
    }
    if (u.pathname === "/api/harness" && req.method === "GET")
      return json(res, {
        version: null,
        supported: false,
        message: "独立旧版 Harness 构建已移除；请使用 ultra 容器内工作台",
        running: !!(await inspect(harnessName))?.State.Running,
        home: path.join(root, "harness/home"),
      });
    if (u.pathname === "/api/harness/stop" && req.method === "POST") {
      const c = await inspect(harnessName);
      if (c?.Config.Labels?.["sm75.managed"] === "v015-candidate")
        await docker(["stop", "--time", "10", harnessName]);
      return json(res, { ok: true });
    }
    if (["/api/harness/start", "/api/harness/install"].includes(u.pathname) && req.method === "POST")
      return json(res, {error: "独立旧版 Harness 构建已移除；请使用 ultra 容器内工作台"}, 409);
    return json(res, { error: "接口不存在" }, 404);
  } catch (e) {
    if (!res.headersSent) json(res, { error: e.message }, 400);
    else res.end();
  }
});
server.listen(port, host, async () => {
  console.log(
    `SM75 console http://${host}:${port}; login token file: ${secretPath}`,
  );
  const settings = store.settings();
  if (settings.autoStart)
    try {
      await start(profile(settings.defaultProfile));
      store.put("startup", "last", {
        state: "started",
        time: Date.now(),
        profile: settings.defaultProfile,
      });
    } catch (e) {
      store.put("startup", "last", {
        state: "failed",
        time: Date.now(),
        error: e.message,
      });
      console.error("默认模型启动失败:", e.message);
    }
});
function uHost(req) {
  return (
    process.env.SM75_CONSOLE_PUBLIC_HOST ||
    new URL(`http://${req.headers.host}`).hostname
  );
}
// Separate browser origin for the native Harness UI. Only authenticated console sessions pass.
const bridge = http.createServer(async (req, res) => {
  if (!auth.networkAllowed(req)) return json(res, { error: "当前地址不在允许网段内" }, 403);
  if (!logged(req)) {
    if (
      req.method === "GET" &&
      (req.headers.accept || "").includes("text/html")
    ) {
      res.writeHead(302, {
        Location: "/",
        "Cache-Control": "no-store",
      });
      return res.end();
    }
    return json(res, { error: "请先登录 SM75 控制台" }, 401);
  }
  auth.track(req, res);
  if (!auth.originAllowed(req)) {
    res.writeHead(403);
    return res.end();
  }
  let cookie;
  try {
    cookie = await nativeHarnessCookie();
  } catch {
    res.writeHead(503);
    return res.end("Harness 正在启动，请稍后刷新");
  }
  const upstream = http.request(
    {
      host: "127.0.0.1",
      port: harnessPort,
      path: harnessPath(req.url),
      method: req.method,
      headers: {
        ...req.headers,
        authorization: "",
        host: `127.0.0.1:${harnessPort}`,
        "accept-encoding": "identity",
        cookie,
        ...(req.headers.origin
          ? { origin: `http://127.0.0.1:${harnessPort}` }
          : {}),
      },
    },
    (r) => {
      const headers = { ...r.headers, "cache-control": "no-store" };
      delete headers["set-cookie"];
      if (r.statusCode === 401 || r.statusCode === 403) harnessCookie = "";
      if (
        (r.headers["content-type"] || "").includes("text/html") &&
        r.statusCode === 200
      ) {
        const chunks = [];
        r.on("data", (chunk) => chunks.push(chunk));
        r.on("end", () => {
          delete headers["content-length"];
          delete headers["content-encoding"];
          res.writeHead(r.statusCode, headers);
          res.end(Buffer.concat(chunks).toString("utf8"));
        });
      } else {
        res.writeHead(r.statusCode, headers);
        r.pipe(res);
      }
    },
  );
  upstream.on("error", () => {
    if (!res.headersSent) res.writeHead(503);
    res.end("Harness 正在启动，请稍后刷新");
  });
  req.pipe(upstream);
  res.on("close", () => upstream.destroy());
});
bridge.on("upgrade", async (req, socket, head) => {
  if (!logged(req) || !auth.originAllowed(req, true)) {
    socket.destroy();
    return;
  }
  auth.track(req, socket);
  let cookie;
  try {
    cookie = await nativeHarnessCookie();
  } catch {
    socket.destroy();
    return;
  }
  const up = http.request({
    host: "127.0.0.1",
    port: harnessPort,
    path: harnessPath(req.url),
    headers: {
      ...req.headers,
      authorization: "",
      host: `127.0.0.1:${harnessPort}`,
      "accept-encoding": "identity",
      cookie,
      ...(req.headers.origin
        ? { origin: `http://127.0.0.1:${harnessPort}` }
        : {}),
    },
  });
  up.on("upgrade", (r, s, h) => {
    socket.write(
      `HTTP/1.1 ${r.statusCode} Switching Protocols\r\n` +
        Object.entries(r.headers)
          .filter(([k]) => k !== "set-cookie")
          .map(([k, v]) => `${k}: ${v}`)
          .join("\r\n") +
        "\r\n\r\n",
    );
    if (h.length) socket.write(h);
    if (head.length) s.write(head);
    s.pipe(socket);
    socket.pipe(s);
    s.on("error", () => socket.destroy());
    socket.on("error", () => s.destroy());
  });
  up.on("response", (r) => {
    socket.end(
      `HTTP/1.1 ${r.statusCode} Upstream rejected upgrade\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`,
    );
    r.resume();
  });
  up.setTimeout(10000, () => up.destroy());
  socket.on("close", () => up.destroy());
  up.on("error", () => socket.destroy());
  up.end();
});
server.on("upgrade", (req, socket, head) => {
  const m = req.url.match(/^\/bench-app\/(speedtest|sql)(\/.*)$/);
  if (m) {
    if (!logged(req) || !auth.originAllowed(req, true)) {
      socket.destroy();
      return;
    }
    auth.track(req, socket);
    return originalTests.upgrade(req, socket, head, m[1], m[2]);
  }
  bridge.emit("upgrade", req, socket, head);
});
let collecting = false;
const collectTimer = setInterval(async () => {
  if (collecting) return;
  collecting = true;
  try {
    for (const p of profiles) {
      try {
        if (!(await status(p)).running) continue;
        const r = await fetch(endpoint(p) + "/metrics", {
          signal: AbortSignal.timeout(2500),
        });
        if (!r.ok) throw Error("metrics");
        telemetry.observe(p.id, await r.text());
      } catch {
        telemetry.miss(p.id);
      }
    }
  } finally {
    collecting = false;
  }
}, 5000);
process.on("SIGTERM", async () => {
  if (shuttingDown) return;
  shuttingDown = true;
  clearInterval(collectTimer);
  for (const id of auth.connections.keys()) auth.closeConnections(id);
  bridge.close();
  server.close();
  originalTests.close();
  if (standalone) standalone.closing = true;
  for (const [id, cache] of nativeCaches) if (!native.has(id)) await cache.stop();
  await Promise.all([...pendingStarts]);
  if (standalone) await standalone.close();
  for (const p of profiles) telemetry.flush(p.id);
  for (const [id, c] of native) {
    const cache = nativeCaches.get(id);
    if (cache) { await terminateLMCacheProcess(c); await cache.stop(); }
    else c.kill();
  }
  for (const cache of nativeCaches.values()) await cache.stop();
  bridge.closeAllConnections();
  server.closeAllConnections();
  process.exit(0);
});
