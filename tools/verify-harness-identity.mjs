// Disposable Linux container only: no GPU, production data, or model required.
import assert from "node:assert/strict";
import fs from "node:fs";
import { Standalone } from "/opt/sm75-workbench/console/standalone.mjs";

assert.equal(process.env.SM75_DISPOSABLE_AUDIT, "1");
fs.mkdirSync("/tmp/identity-audit", { recursive: true, mode: 0o700 });
fs.writeFileSync("/tmp/identity-audit/manager-secret", "disposable", { mode: 0o600 });
const runtime = new Standalone("/tmp/identity-audit", "disposable-engine-key");
let ok = false;
try {
  await runtime.startHarness({ id: "audit", args: ["audit-model"], port: 8000 });
  const child = runtime.harness.child;
  const status = fs.readFileSync(`/proc/${child.pid}/status`, "utf8");
  assert.match(status, /Uid:\s+1000\s+1000\s+1000\s+1000/);
  assert.match(status, /Gid:\s+1000\s+1000\s+1000\s+1000/);
  assert.match(status, /CapEff:\s+0+\s/);
  const probe = await runtime.launchHarness(["-e", `
    const fs=require('node:fs');
    if (process.env.SM75_MANAGEMENT_TEST_SECRET) process.exit(4);
    try { fs.readFileSync('/tmp/identity-audit/manager-secret'); process.exit(2); }
    catch(e) { process.exit(e.code==='EACCES' ? 0 : 3); }
  `], {}, "/tmp");
  const code = await new Promise(resolve => probe.once("exit", resolve));
  assert.equal(code, 0, "tool UID must not read manager-only data");
  for (let attempt = 0; attempt < 40; attempt++) {
    if (child.exitCode !== null) throw Error("DSH exited before ready");
    try {
      const log = fs.readFileSync(runtime.harnessLog(), "utf8");
      const token = [...log.matchAll(/[?&]token=([A-Za-z0-9_-]+)/g)].at(-1)?.[1];
      if (!token) throw Error("waiting for DSH handshake");
      const login = await fetch(`http://127.0.0.1:3085/?token=${encodeURIComponent(token)}`, { redirect: "manual", signal: AbortSignal.timeout(1000) });
      const cookie = login.headers.getSetCookie().map(c => c.split(";")[0]).join("; ");
      const response = await fetch("http://127.0.0.1:3085/", { headers: { cookie }, signal: AbortSignal.timeout(1000) });
      if (response.ok) { ok = true; break; }
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  if (!ok) console.error(fs.readFileSync(runtime.harnessLog(), "utf8").slice(-6000).replace(/token=[A-Za-z0-9_-]+/g, "token=[redacted]"));
  assert(ok, "DSH HTTP did not become ready");
} finally {
  await runtime.close();
}
assert.equal(runtime.harness, null);
const secret = "/tmp/identity-audit/manager-secret";
fs.unlinkSync("/dsh/home/sm75-plugins.patch.json");
fs.symlinkSync(secret, "/dsh/home/sm75-plugins.patch.json");
try {
  await runtime.startHarness({ id: "audit", args: ["audit-model"], port: 8000 });
  assert.equal(fs.lstatSync("/dsh/home/sm75-plugins.patch.json").isSymbolicLink(), false);
  assert.equal(fs.readFileSync(secret, "utf8"), "disposable");
} finally { await runtime.close(); }
fs.unlinkSync("/dsh/home/settings.yaml");
fs.symlinkSync(secret, "/dsh/home/settings.yaml");
await assert.rejects(runtime.startHarness({ id: "audit", args: ["audit-model"], port: 8000 }), { code: "ELOOP" });
assert.equal(fs.readFileSync(secret, "utf8"), "disposable");
console.log("PASS DSH UID/GID 1000, zero capabilities, credential isolation, HTTP startup/shutdown, config symlink rejection");
