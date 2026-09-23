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

function releaseFixture(base) {
  const write = (name, content) => {
    const file = path.join(base, name);
    fs.mkdirSync(path.dirname(file), {recursive: true});
    fs.writeFileSync(file, content);
  };
  write('@deepseek-ai/dsh/package.json', JSON.stringify({version: '0.1.7-alpha.2'}));
  for (const name of ['sm75-brand', 'dsh-watcher']) {
    write(`${name}/package.json`, JSON.stringify({name, type: 'module', dsh: {client: {platform: 'web'}}}));
    write(`${name}/lib/client.js`, `window.__ModuleLoader__.load({id: '${name}', factory: () => ({})});`);
  }
  write('sm75-workbench/package.json', JSON.stringify({name: 'sm75-workbench', type: 'module', main: 'lib/index.js'}));
  write('sm75-workbench/lib/client.js', 'export const legacyClient = true;');
  return write;
}

test('release bundle gate accepts Watcher without the retired usage package and checks server-only leftover syntax', () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-release-bundles-'));
  try {
    const write = releaseFixture(base);
    assert.equal(fs.existsSync(path.join(base, '@deepseek-ai/dsh-client-ui-token-usage')), false);
    assert.deepEqual(verifyClientBundles(base), {checked: 3, failed: 0});
    write('sm75-workbench/lib/client.js', 'export const broken = ;');
    assert.throws(() => verifyClientBundles(base), /Invalid client bundles:[\s\S]*sm75-workbench/);
    fs.unlinkSync(path.join(base, 'sm75-workbench/lib/client.js'));
    assert.deepEqual(verifyClientBundles(base), {checked: 2, failed: 0});
  } finally {fs.rmSync(base, {recursive: true, force: true});}
});

test('release bundle gate still rejects an incorrect custom package or browser module identity', () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-module-identity-'));
  try {
    const write = releaseFixture(base);
    write('sm75-brand/package.json', JSON.stringify({name: 'other-brand', type: 'module'}));
    assert.throws(() => verifyClientBundles(base), /Custom client module identity mismatch: sm75-brand/);
    write('sm75-brand/package.json', JSON.stringify({name: 'sm75-brand', type: 'module'}));
    write('dsh-watcher/lib/client.js', "window.__ModuleLoader__.load({id: 'wrong-watcher', factory: () => ({})});");
    assert.throws(() => verifyClientBundles(base), /Custom client module identity mismatch: dsh-watcher/);
  } finally {fs.rmSync(base, {recursive: true, force: true});}
});
