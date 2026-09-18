import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { EventEmitter } from "node:events";
import { flashinferWorkspace } from "../config.mjs";
import { Standalone } from "../standalone.mjs";

test(
  "FlashInfer workspace preserves compiled artifacts across manager instances",
  { skip: process.platform === "win32" },
  () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "sm75-flashinfer-"));
    try {
      const first = flashinferWorkspace(root, true);
      fs.writeFileSync(
        path.join(first, ".cache/flashinfer/compiled.so"),
        "fixture",
      );
      const second = flashinferWorkspace(root, true);
      assert.equal(
        fs.readFileSync(
          path.join(second, ".cache/flashinfer/compiled.so"),
          "utf8",
        ),
        "fixture",
      );
      assert.equal(
        fs.realpathSync(path.join(second, ".cache/flashinfer")),
        path.join(root, "shared/flashinfer"),
      );
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  },
);
test("ultra passes the persistent FlashInfer workspace into the engine", async () => {
  const s = new Standalone("/tmp", "test-key");
  let env;
  s.prepareCache = () => ({
    FLASHINFER_WORKSPACE_BASE: "/data/cache/shared/flashinfer-home",
  });
  s.launch = async (bin, args, values) => {
    env = values;
    return new EventEmitter();
  };
  await s.start({ id: "test", port: 8000, args: ["model"] });
  assert.equal(
    env.FLASHINFER_WORKSPACE_BASE,
    "/data/cache/shared/flashinfer-home",
  );
});
test("shutdown does not signal or wait for a child already terminated by signal", async (t) => {
  const s = new Standalone("/tmp", "test-key");
  t.mock.method(process, "kill", () =>
    assert.fail("exited child must not be signalled"),
  );
  const child = Object.assign(new EventEmitter(), {
    pid: 12345,
    exitCode: null,
    signalCode: "SIGTERM",
  });
  await s.terminate(child);
  assert.equal(child.listenerCount("exit"), 0);
});
