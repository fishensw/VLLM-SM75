import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import { contains, validateNetworks } from "./network-policy.mjs";
const derive = promisify(crypto.scrypt);

const hash = (value) => crypto.createHash("sha256").update(value).digest("hex");
export function equalSecret(a, b) {
  return (
    typeof a === "string" &&
    typeof b === "string" &&
    crypto.timingSafeEqual(Buffer.from(hash(a)), Buffer.from(hash(b)))
  );
}

// Link a completely written file into place: concurrent first starts never read
// an empty key and never overwrite an administrator's existing credential.
export function ensureSecret(file) {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const temp = `${file}.${process.pid}.${crypto.randomUUID()}.tmp`;
  if (!fs.existsSync(file)) {
    fs.writeFileSync(temp, crypto.randomBytes(32).toString("base64url"), {
      mode: 0o600,
      flag: "wx",
    });
    try {
      fs.linkSync(temp, file);
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
    } finally {
      fs.unlinkSync(temp);
    }
  }
  const key = fs.readFileSync(file, "utf8").trim();
  if (!key || key.length > 4096 || /[\x00-\x20\x7f]/.test(key))
    throw Error("登录凭据文件无效，请使用本地恢复工具");
  fs.chmodSync(file, 0o600);
  return key;
}

function atomicJson(file, value) {
  const temp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(value), { mode: 0o600 });
  fs.renameSync(temp, file);
}

export class Auth {
  constructor(
    root,
    { ttlSeconds = 43200, now = Date.now, trustedProxies = [] } = {},
  ) {
    if (
      !Number.isInteger(ttlSeconds) ||
      ttlSeconds < 60 ||
      ttlSeconds > 2592000
    )
      throw Error("会话有效期必须为 60–2592000 秒");
    this.root = root;
    this.keyPath = path.join(root, "key");
    this.key = ensureSecret(this.keyPath);
    this.file = path.join(root, "web-sessions.json");
    this.now = now;
    this.ttl = ttlSeconds * 1000;
    this.trustedProxies = new Set(trustedProxies);
    this.connections = new Map();
    this.attempts = new Map();
    let saved = { sessions: [] };
    if (fs.existsSync(this.file)) {
      try {
        saved = JSON.parse(fs.readFileSync(this.file, "utf8"));
      } catch {
        throw Error("会话存储损坏，请备份后使用本地恢复工具");
      }
    }
    this.policyPath = path.join(root, "web-admin.json");
    this.policyRaw = fs.existsSync(this.policyPath) ? fs.readFileSync(this.policyPath, "utf8") : "";
    this.policy = this.policyRaw ? JSON.parse(this.policyRaw) : { account: null, networks: [] };
    validateNetworks(this.policy.networks);
    this.passwordJobs = 0;
    this.sessions = new Map(
      saved.keyHash === hash(this.key) && (saved.policyHash || hash("")) === hash(this.policyRaw)
        ? saved.sessions.filter(
            ([id, s]) => /^[a-f0-9]{64}$/.test(id) && s.expiresAt > now(),
          )
        : [],
    );
    this.persist();
  }
  persist() {
    atomicJson(this.file, {
      schema: 1,
      policyHash: hash(this.policyRaw),
      keyHash: hash(this.key),
      sessions: [...this.sessions],
    });
  }
  refreshPolicy() {
    const raw = fs.existsSync(this.policyPath) ? fs.readFileSync(this.policyPath, "utf8") : "";
    if (raw !== this.policyRaw) {
      const policy = raw ? JSON.parse(raw) : { account: null, networks: [] };
      validateNetworks(policy.networks);
      this.policyRaw = raw; this.policy = policy;
      for (const id of this.connections.keys()) this.closeConnections(id);
      this.sessions.clear(); this.persist();
    }
  }
  networkAllowed(req) {
    this.refreshPolicy();
    return !this.policy.networks.length || this.policy.networks.some(n => contains(n, req.socket.remoteAddress));
  }
  securityInfo(req) {
    return { accountConfigured: !!this.policy.account, username: this.policy.account?.username || "", networks: this.policy.networks, peer: req.socket.remoteAddress || "" };
  }
  async checkPassword(password) {
    const a = this.policy.account;
    if (!a || typeof password !== "string" || password.length > 256 || this.passwordJobs >= 4) return false;
    this.passwordJobs++;
    try { return equalSecret((await derive(password, a.salt, 64)).toString("hex"), a.digest); }
    finally { this.passwordJobs--; }
  }
  async loginAccount(username, password, req) {
    this.refreshKey();
    if (!this.networkAllowed(req)) return { status: 403, error: "当前地址不在允许网段内" };
    const attempt = this.attempts.get(req.socket.remoteAddress || "unknown");
    if (attempt && attempt.until > this.now() && attempt.count >= 10) return { status: 429, error: "尝试过于频繁，请稍后再试" };
    const version = this.policyRaw;
    const valid = await this.checkPassword(password);
    this.refreshKey();
    return this.login(null, req, valid && version === this.policyRaw && username === this.policy.account?.username);
  }
  savePolicy(policy, req) {
    atomicJson(this.policyPath, policy);
    this.policyRaw = fs.readFileSync(this.policyPath, "utf8"); this.policy = policy;
    const keep = this.sessionId(req);
    for (const id of [...this.sessions.keys()]) if (id !== keep) this.revoke(id);
    this.persist();
    // Close live connections whose peer is no longer allowed, even for the current session.
    for (const [id, streams] of this.connections) for (const stream of streams)
      if (stream.sm75Peer && !this.networkAllowed({socket:{remoteAddress:stream.sm75Peer}})) stream.destroy();
  }
  async updateSecurity(payload, req) {
    if (!this.principal(req)) throw Error("请先登录");
    const version = this.policyRaw;
    if (this.policy.account && !(await this.checkPassword(payload.currentPassword))) throw Error("当前密码错误");
    this.refreshKey();
    if (version !== this.policyRaw || !this.principal(req)) throw Error("账户设置已变化，请重新登录");
    if (payload.action === "account") {
      const { username, password } = payload;
      if (typeof username !== "string" || !/^[a-zA-Z0-9_.-]{3,64}$/.test(username)) throw Error("账号需为 3–64 位字母、数字、下划线、点或短横线");
      if (typeof password !== "string" || password.length < 12 || password.length > 256) throw Error("密码长度需为 12–256 位");
      const salt = crypto.randomBytes(16).toString("hex");
      const digest = (await derive(password, salt, 64)).toString("hex");
      this.refreshKey();
      if (version !== this.policyRaw || !this.principal(req)) throw Error("账户设置已变化，请重试");
      this.savePolicy({ ...this.policy, account: { username, salt, digest } }, req);
    } else if (payload.action === "networks") {
      const networks = validateNetworks(payload.networks);
      if (networks.length && !networks.some(n => contains(n, req.socket.remoteAddress))) throw Error("白名单必须包含当前连接地址，避免锁定登录");
      this.savePolicy({ ...this.policy, networks }, req);
    } else throw Error("未知账户设置操作");
    return this.securityInfo(req);
  }
  refreshKey() {
    this.refreshPolicy();
    const key = fs.readFileSync(this.keyPath, "utf8").trim();
    if (key !== this.key) {
      this.key = ensureSecret(this.keyPath);
      for (const id of this.connections.keys()) this.closeConnections(id);
      this.sessions.clear();
      this.persist();
    }
  }
  secure(req) {
    return (
      !!req.socket.encrypted ||
      (this.trustedProxies.has(req.socket.remoteAddress) &&
        req.headers["x-forwarded-proto"] === "https")
    );
  }
  originAllowed(req, required = false) {
    const origin = req.headers.origin;
    if (!origin) return !required;
    return (
      origin === `${this.secure(req) ? "https" : "http"}://${req.headers.host}`
    );
  }
  cookie(value, req, maxAge = Math.floor(this.ttl / 1000)) {
    return `sm75_session=${value}; HttpOnly; SameSite=Strict; Path=/; Max-Age=${maxAge}${this.secure(req) ? "; Secure" : ""}`;
  }
  sessionId(req) {
    const raw = (req.headers.cookie || "").match(
      /(?:^|;\s*)sm75_session=([^;]+)/,
    )?.[1];
    return raw ? hash(raw) : null;
  }
  principal(req) {
    this.refreshKey();
    if (!this.networkAllowed(req)) return null;
    if (!this.policy.account && equalSecret(req.headers.authorization, `Bearer ${this.key}`))
      return { id: "local-admin", method: "token" };
    const id = this.sessionId(req),
      session = this.sessions.get(id);
    if (!session) return null;
    if (session.expiresAt <= this.now()) {
      this.revoke(id);
      return null;
    }
    return {
      id: "local-admin",
      method: "session",
      expiresAt: session.expiresAt,
    };
  }
  login(token, req, accountVerified = false) {
    this.refreshKey();
    if (!this.networkAllowed(req)) return { status: 403, error: "当前地址不在允许网段内" };
    const address = req.socket.remoteAddress || "unknown",
      now = this.now();
    for (const [ip, value] of this.attempts)
      if (value.until <= now) this.attempts.delete(ip);
    const attempt = this.attempts.get(address) || {
      count: 0,
      until: now + 60000,
    };
    if (attempt.count >= 10)
      return { status: 429, error: "尝试过于频繁，请稍后再试" };
    if (!(this.policy.account ? accountVerified : equalSecret(token, this.key))) {
      attempt.count++;
      if (this.attempts.size >= 4096 && !this.attempts.has(address))
        return { status: 429, error: "尝试过于频繁，请稍后再试" };
      this.attempts.set(address, attempt);
      return { status: 401, error: "登录凭据错误" };
    }
    this.attempts.delete(address);
    this.revoke(this.sessionId(req));
    for (const [id, session] of this.sessions)
      if (session.expiresAt <= now) this.revoke(id);
    if (this.sessions.size >= 256)
      this.revoke(this.sessions.keys().next().value);
    const value = crypto.randomBytes(32).toString("base64url"),
      expiresAt = now + this.ttl;
    this.sessions.set(hash(value), { expiresAt });
    this.persist();
    return { status: 200, cookie: this.cookie(value, req), expiresAt };
  }
  closeConnections(id) {
    const connections = this.connections.get(id);
    this.connections.delete(id);
    for (const stream of connections || []) stream.destroy();
  }
  revoke(id) {
    if (!id) return;
    this.closeConnections(id);
    if (this.sessions.delete(id)) this.persist();
  }
  track(req, stream) {
    const id = this.sessionId(req),
      session = this.sessions.get(id);
    if (!session) return;
    let connections = this.connections.get(id);
    if (!connections) this.connections.set(id, (connections = new Set()));
    connections.add(stream);
    stream.sm75Peer = req.socket.remoteAddress;
    const timer = setTimeout(
      () => this.revoke(id),
      Math.max(1, session.expiresAt - this.now()),
    );
    timer.unref();
    stream.once("close", () => {
      clearTimeout(timer);
      connections.delete(stream);
      if (!connections.size) this.connections.delete(id);
    });
  }
}
