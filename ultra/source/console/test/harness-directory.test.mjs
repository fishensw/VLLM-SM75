import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {listWorkspaceDirectory, renderDirectory} from '../plugins/sm75-workspace-tools/lib/directory.js';

const modules = process.env.SM75_HARNESS_MODULES;
test('native directory listing preserves workspace boundaries with the real Harness filesystem', {skip: !modules && 'Set SM75_HARNESS_MODULES to the installed Harness node_modules for integration'}, async () => {
  const require = createRequire(path.join(modules, 'sm75-runtime-check.cjs'));
  const load = name => import(pathToFileURL(require.resolve(name)).href);
  const [{Context}, {default: Projections}, {default: Policy}, {SandboxedFileSystem}] = await Promise.all([
    load('@deepseek-ai/cordis'), load('@deepseek-ai/dsh-session-projection'),
    load('@deepseek-ai/dsh-sandbox-policy'), load('@deepseek-ai/dsh-fs-sandbox'),
  ]);
  const base = await fs.mkdtemp(path.join(os.tmpdir(), 'sm75-native-fs-'));
  const workspace = path.join(base, 'work'), outside = path.join(base, 'work-other');
  const ctx = new Context(), fibers = [];
  try {
    await fs.mkdir(workspace);
    await fs.mkdir(outside);
    await fs.mkdir(path.join(workspace, 'empty'));
    await fs.writeFile(path.join(workspace, 'hello.txt'), 'hello');
    await fs.writeFile(path.join(outside, 'private.txt'), 'not part of the workspace');
    await fs.symlink(outside, path.join(workspace, 'outside-link'), 'dir');
    await fs.symlink(path.join(workspace, 'empty'), path.join(workspace, 'inside-link'), 'dir');
    fibers.push(await ctx.plugin(Projections));
    fibers.push(await ctx.plugin(Policy, {mode: 'read-only', workspaceRoot: workspace}));
    fibers.push(await ctx.plugin(SandboxedFileSystem, {cwd: workspace}));
    const exec = {agent: {session: {header: {cwd: workspace}}}, signal: new AbortController().signal};
    const result = await listWorkspaceDirectory(ctx.fs, {}, exec);
    assert.equal(ctx.fs.sandboxMode, 'read-only');
    assert.equal(result.total, 4);
    assert.equal(result.entries.find(row => row.name === 'empty').type, 'directory');
    assert.deepEqual(result.entries.find(row => row.name === 'outside-link'), {name: 'outside-link', type: 'outside-link'});
    assert.equal(JSON.stringify(result).includes('private.txt'), false);
    assert.equal((await listWorkspaceDirectory(ctx.fs, {path: 'inside-link'}, exec)).total, 0);
    for (const target of ['..', '../work-other', outside, 'outside-link', 'outside-link/../work-other']) {
      await assert.rejects(listWorkspaceDirectory(ctx.fs, {path: target}, exec), {code: 'FS_SANDBOX_DENIED'});
    }
    await assert.rejects(listWorkspaceDirectory(ctx.fs, {}, {signal: exec.signal}), {code: 'FS_SANDBOX_DENIED'});
    await assert.rejects(listWorkspaceDirectory(ctx.fs, {limit: 201}, exec), /1 到 200/);
    await assert.rejects(listWorkspaceDirectory(ctx.fs, {path: 'missing'}, exec), {code: 'FS_NOT_FOUND'});
    await assert.rejects(listWorkspaceDirectory(ctx.fs, {path: 'hello.txt'}, exec), {code: 'FS_NOT_DIRECTORY'});
    const page = await listWorkspaceDirectory(ctx.fs, {offset: 1, limit: 2}, exec);
    assert.deepEqual(page.entries, result.entries.slice(1, 3));
    assert.equal(page.hasMore, true);
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(listWorkspaceDirectory(ctx.fs, {}, {...exec, signal: controller.signal}), {name: 'AbortError'});
  } finally {
    for (const fiber of fibers.reverse()) await fiber.dispose();
    await fs.rm(base, {recursive: true, force: true});
  }
});

test('directory output preserves unusual filenames as data and reports pagination', () => {
  const [result] = renderDirectory({path: '.', entries: [{name: 'line\nbreak', type: 'file'}], total: 2, offset: 0, hasMore: true});
  assert.match(result.text, /"line\\nbreak"/);
  assert.match(result.text, /offset=1/);
});
