import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

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
    this.sessions = new Map(
      saved.keyHash === hash(this.key)
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
      keyHash: hash(this.key),
      sessions: [...this.sessions],
    });
  }
  refreshKey() {
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
    if (equalSecret(req.headers.authorization, `Bearer ${this.key}`))
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
  login(token, req) {
    this.refreshKey();
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
    if (!equalSecret(token, this.key)) {
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
