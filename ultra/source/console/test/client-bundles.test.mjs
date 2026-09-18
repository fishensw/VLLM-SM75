import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {verifyClientBundles} from '../verify-client-bundles.mjs';

test('client bundle gate rejects corrupted embedded worker strings before packaging', () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-bundles-'));
  try {
    const dir = path.join(base, '@deepseek-ai', 'preview', 'lib');
    fs.mkdirSync(dir, {recursive: true});
    const file = path.join(dir, 'client.js');
    fs.writeFileSync(file, 'const worker = "text\\u001ctext";');
    assert.deepEqual(verifyClientBundles(base), {checked: 1, failed: 0});
    fs.writeFileSync(file, 'const worker = "text\ntext";');
    assert.throws(() => verifyClientBundles(base), /Invalid client bundles/);
  } finally { fs.rmSync(base, {recursive: true, force: true}); }
});
