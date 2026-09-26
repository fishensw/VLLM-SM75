import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { Standalone } from "../standalone.mjs";
import fs from "node:fs";
test("one container admits only one engine even for simultaneous different profiles", async () => {
  const s = new Standalone("/tmp", "key");
  s.prepareCache = () => {};
  let finish;
  s.launch = () => new Promise((resolve) => (finish = resolve));
  const a = s.start({ id: "a", args: ["model"], port: 8015 });
  await assert.rejects(
    s.start({ id: "b", args: ["model"], port: 8016 }),
    /启动进行中/,
  );
  const child = new EventEmitter();
  finish(child);
  await a;
  await assert.rejects(
    s.start({ id: "b", args: ["model"], port: 8016 }),
    /停止当前/,
  );
  child.emit("exit");
  assert.equal(s.engine, null);
});
test("failed process launch releases the start lock", async () => {
  const s = new Standalone("/tmp", "key");
  s.prepareCache = () => {};
  s.launch = async () => {
    throw Error("spawn failed");
  };
  await assert.rejects(
    s.start({ id: "a", args: ["model"], port: 8015 }),
    /spawn failed/,
  );
  assert.equal(s.starting, false);
  assert.equal(s.engine, null);
});

test("engine launch failure also cleans the previously started power supervisor", async (t) => {
  const original = fs.existsSync;
  t.mock.method(
    fs,
    "existsSync",
    (p) => p === "/usr/local/nvidia/lib64/libnvidia-api.so.1" || original(p),
  );
  const s = new Standalone("/tmp", "key");
  s.prepareCache = () => {};
  const supervisor = { pid: 123 };
  let calls = 0,
    terminated;
  s.launch = async () => {
    if (++calls === 1) return supervisor;
    throw Error("engine spawn failed");
  };
  s.terminate = async (child) => {
    terminated = child;
  };
  await assert.rejects(
    s.start({
      id: "a",
      args: ["model"],
      port: 8015,
      power: { mode: "pstate" },
    }),
    /engine spawn failed/,
  );
  assert.equal(terminated, supervisor);
  assert.equal(s.supervisor, null);
  assert.equal(s.starting, false);
});

test("stopping another profile does not stop the active engine or its supervisor", async () => {
  const s = new Standalone("/tmp", "key"),
    supervisor = { pid: 123 };
  s.engine = { id: "active" };
  s.supervisor = supervisor;
  s.terminate = async () =>
    assert.fail("unrelated process must not be touched");
  await s.stop({ id: "other" });
  assert.equal(s.supervisor, supervisor);
  assert.equal(s.engine.id, "active");
});

test("DSH refuses a root fallback when dropping identity is denied", async () => {
  const s = new Standalone("/tmp", "key");
  let calls = 0;
  s.launch = async (_bin, _args, _env, _log, identity) => {
    calls++;
    assert.equal(identity.uid, 1000);
    assert.equal(identity.gid, 1000);
    throw Object.assign(Error("denied"), { code: "EPERM" });
  };
  await assert.rejects(s.launchHarness([], {}, "/tmp"), /禁止以 root 回退/);
  assert.equal(calls, 1);
});

test("DSH gets an explicit environment without inherited management credentials", async (t) => {
  const previous = process.env.SM75_MANAGEMENT_TEST_SECRET;
  process.env.SM75_MANAGEMENT_TEST_SECRET = "private";
  t.after(() => {
    if (previous === undefined) delete process.env.SM75_MANAGEMENT_TEST_SECRET;
    else process.env.SM75_MANAGEMENT_TEST_SECRET = previous;
  });
  const s = new Standalone("/tmp", "key");
  s.launch = async (_bin, _args, env, _log, identity) => {
    assert.equal(identity.env.SM75_MANAGEMENT_TEST_SECRET, undefined);
    assert.equal(identity.env.SM75_ENGINE_KEY, "engine-only");
    assert.equal(identity.env, env);
    return { pid: 123 };
  };
  assert.equal(
    (await s.launchHarness([], { SM75_ENGINE_KEY: "engine-only" }, "/tmp")).pid,
    123,
  );
});

test('single-container engine launch inherits NCCL default and explicit overrides', async () => {
  const previous = process.env.NCCL_P2P_LEVEL;
  try {
    for (const [containerValue, profileEnv, expected] of [
      [undefined, {}, 'SYS'], ['PIX', {}, 'PIX'],
      ['PIX', {NCCL_P2P_LEVEL:'PHB'}, 'PHB'],
    ]) {
      if (containerValue === undefined) delete process.env.NCCL_P2P_LEVEL;
      else process.env.NCCL_P2P_LEVEL = containerValue;
      const manager = new Standalone('/tmp', 'test-key');
      manager.prepareCache = () => ({});
      const child = new EventEmitter();
      manager.launch = async (_bin, _args, env) => {
        assert.equal(env.NCCL_P2P_LEVEL, expected);
        return child;
      };
      await manager.start({id:'nccl', args:['model'], port:8015, env:profileEnv});
      child.emit('exit', 0);
    }
  } finally {
    if (previous === undefined) delete process.env.NCCL_P2P_LEVEL;
    else process.env.NCCL_P2P_LEVEL = previous;
  }
});
