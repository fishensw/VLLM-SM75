import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { ensureSecret } from "./auth.mjs";
const root = path.resolve(
  process.env.SM75_CONSOLE_ROOT ||
    fileURLToPath(new URL("./data", import.meta.url)),
);
const file = path.join(root, "key");
switch (process.argv[2]) {
  case "show":
    process.stdout.write(ensureSecret(file) + "\n");
    break;
  case "recover":
  case "reset": {
    if (process.argv[2] === "recover") {
      const policy = path.join(root, "web-admin.json");
      if (fs.existsSync(policy)) fs.renameSync(policy, policy + ".recovery-" + Date.now());
    }
    const tmp = `${file}.${process.pid}.tmp`;
    fs.mkdirSync(root, { recursive: true, mode: 0o700 });
    fs.writeFileSync(tmp, crypto.randomBytes(32).toString("base64url"), {
      mode: 0o600,
      flag: "wx",
    });
    fs.renameSync(tmp, file);
    // Remove corrupted session state as well; a running manager observes the new key.
    fs.writeFileSync(
      path.join(root, "web-sessions.json"),
      JSON.stringify({ sessions: [] }),
      { mode: 0o600 },
    );
    console.log(
      "已轮换 Web 登录 token（recover 同时清除账户和网段限制）；使用 auth-cli.mjs show 在本机读取。引擎 API key 未修改。",
    );
    break;
  }
  default:
    throw Error("用法：node auth-cli.mjs show|reset|recover（仅宿主/容器管理终端）");
}
