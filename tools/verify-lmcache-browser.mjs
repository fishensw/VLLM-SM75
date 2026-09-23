#!/usr/bin/env node
/** Real LMCache UI audit using fresh synthetic profiles in a disposable CPU-only container. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {spawn, spawnSync} from 'node:child_process';
import {once} from 'node:events';

assert.equal(process.env.SM75_DISPOSABLE_AUDIT, '1', 'Requires SM75_DISPOSABLE_AUDIT=1');
assert.equal(process.platform, 'linux', 'Only use the disposable Linux container');
assert(fs.existsSync('/.dockerenv'), 'A disposable container is required');
assert(!fs.readdirSync('/dev').some(name => /^nvidia/.test(name)), 'GPU devices must not be available');
const argv = process.argv.slice(2);
const option = (name, fallback) => {const i = argv.indexOf(name); return i < 0 ? fallback : argv[i + 1];};
for (let i = 0; i < argv.length; i += 2) assert(['--source-root', '--console-root', '--browser-root', '--output'].includes(argv[i]) && argv[i + 1], 'Unknown or missing CLI argument');
const sourceRoot = path.resolve(option('--source-root', path.join(path.dirname(fileURLToPath(import.meta.url)), '..')));
const consoleRoot = path.resolve(option('--console-root', path.join(sourceRoot, 'ultra/source/console')));
const browserRoot = path.resolve(option('--browser-root', '/work/browser-audit'));
const runId = new Date().toISOString().replace(/[:.]/g, '-');
const auditRoot = '/work/lmcache-browser-audit';
const data = path.join(auditRoot, 'runs', runId, 'data');
const output = path.resolve(option('--output', path.join(auditRoot, 'evidence', runId)));
assert(output.startsWith('/work/'), 'Evidence must stay inside the disposable /work mount');
assert(fs.existsSync(path.join(consoleRoot, 'server.mjs')), 'Console source is missing');
assert(!fs.existsSync(data), 'Every run must use a new synthetic data directory');
fs.mkdirSync(data, {recursive: true, mode: 0o700});
fs.mkdirSync(output, {recursive: true});
const {chromium} = await import(pathToFileURL(path.join(browserRoot, 'node_modules/playwright/index.mjs')));
const {argValue, GiB} = await import(pathToFileURL(path.join(consoleRoot, 'public/context-config.js')));
const base = 'http://127.0.0.1:19269';
const checks = [], pageErrors = [], consoleErrors = [], requests = [], forbidden = [], secrets = [], layouts = [], hardwareFailures = [];
let manager, browser, page, cookie = '', failure, allowStart = false;
const startFailures = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function clean(value) {let text = String(value); for (const secret of secrets) if (secret) text = text.split(secret).join('[redacted]'); return text;}
function check(name, detail = {}) {checks.push({name, ...detail}); console.log(JSON.stringify({passed: name, ...detail}));}
async function until(name, fn, timeout = 20000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    if (manager && manager.exitCode !== null) throw Error(`Manager exited: ${manager.exitCode}`);
    try {if (await fn()) return;} catch {}
    await sleep(100);
  }
  throw Error('Timed out: ' + name);
}
async function api(route) {
  const response = await fetch(base + '/console-api/' + route, {headers: {Cookie: cookie}});
  assert.equal(response.status, 200, route + ' must succeed');
  return response.json();
}
const saved = async (id = 'lmcache-audit') => (await api('profiles')).find(row => row.id === id);
async function edit(id = 'lmcache-audit') {
  const editor = page.locator('#profileEditor');
  if (!(await editor.isVisible())) await page.locator(`[data-profile="${id}"]`).getByRole('button', {name: '编辑', exact: true}).click();
  await editor.waitFor({state: 'visible'});
  await page.locator('#lmcacheEnabled').waitFor({state: 'visible'});
}
async function save() {
  const id = await page.locator('#pid').inputValue();
  const response = page.waitForResponse(r => new URL(r.url()).pathname === '/console-api/profiles' && r.request().method() === 'POST');
  await page.locator('#save').click();
  const result = await response;
  assert.equal(result.status(), 200, 'UI profile save must succeed');
  await page.locator('#profileEditor').waitFor({state: 'hidden'});
  return saved(id);
}
async function rejectedSave(pattern) {
  const id = await page.locator('#pid').inputValue(), before = await saved(id);
  const posts = requests.filter(r => r.path === '/console-api/profiles' && r.method === 'POST').length;
  await page.locator('#save').click();
  await until('validation error', async () => pattern.test(await page.locator('#notice').innerText()));
  assert.deepEqual(await saved(id), before);
  assert.equal(requests.filter(r => r.path === '/console-api/profiles' && r.method === 'POST').length, posts);
}
async function snapshot(name, width) {
  await page.setViewportSize({width, height: 1000});
  await page.locator('#lmcacheSettings').scrollIntoViewIfNeeded();
  await sleep(150);
  const layout = await page.locator('#lmcacheSettings').evaluate(section => {
    const bounds = section.getBoundingClientRect();
    const fields = [...section.querySelectorAll('input,select,button')].filter(node => node.checkVisibility()).map(node => {
      const b = node.getBoundingClientRect();
      return {id: node.id, x: b.x, y: b.y, right: b.right, bottom: b.bottom, width: b.width, height: b.height};
    });
    const overlaps = [];
    for (let i = 0; i < fields.length; i++) for (let j = i + 1; j < fields.length; j++) {
      const a = fields[i], b = fields[j];
      if (Math.min(a.right, b.right) - Math.max(a.x, b.x) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.y, b.y) > 1) overlaps.push([a.id, b.id]);
    }
    return {viewport: innerWidth, documentWidth: document.documentElement.scrollWidth,
      section: {left: bounds.left, right: bounds.right, width: bounds.width}, fields, overlaps};
  });
  layouts.push({name, ...layout});
  assert.equal(layout.overlaps.length, 0, 'LMCache controls must not overlap');
  assert(layout.fields.every(field => field.x >= -1 && field.right <= width + 1 && field.width >= 12 && field.height >= 12), 'LMCache fields must fit the viewport');
  await page.screenshot({path: path.join(output, name + '.png'), fullPage: true});
  await page.locator('#lmcacheSettings').screenshot({path: path.join(output, name + '-panel.png')});
  check('readable LMCache layout: ' + name, {width, controls: layout.fields.length});
}
async function stop(child) {
  if (!child || child.exitCode !== null) return;
  const exited = once(child, 'exit');
  child.kill('SIGTERM');
  await Promise.race([exited, sleep(5000)]);
  if (child.exitCode === null) {child.kill('SIGKILL'); await exited;}
}
try {
  for (const port of [19269, 19270, 19271]) {
    const occupied = await fetch(`http://127.0.0.1:${port}/`, {signal: AbortSignal.timeout(400)}).then(() => true, () => false);
    assert(!occupied, `Audit port ${port} is occupied`);
  }
  // Do not attempt this preflight UI test in an environment where LMCache could launch.
  const dependency = spawnSync('python3', ['-c', 'import importlib.util; assert importlib.util.find_spec("lmcache") is None'], {encoding: 'utf8'});
  assert.equal(dependency.status, 0, 'Requires Python with LMCache absent; GPU/service execution is outside this audit');
  const model = path.join(data, 'models/main');
  fs.mkdirSync(model, {recursive: true});
  fs.writeFileSync(path.join(model, 'config.json'), JSON.stringify({model_type: 'qwen3_8',
    quantization_config: {quant_method: 'fp8'}, text_config: {max_position_embeddings: 262144,
      rope_parameters: {rope_type: 'default', rope_theta: 10000000}}}));
  const native = {kv_connector: 'OffloadingConnector', kv_role: 'kv_both',
    kv_connector_extra_config: {spec_name: 'CPUOffloadingSpec', cpu_bytes_to_use: 8 * GiB, keep: 7}};
  const args = [model, '--served-model-name', 'synthetic-lmcache', '--tensor-parallel-size', '4',
    '--max-model-len', '262144', '--kv-cache-memory-bytes', String(2.5 * GiB),
    '--hf-overrides', '{"text_config":{"custom_field":17}}'];
  const original = {id: 'lmcache-audit', name: 'LMCache 隔离验收', backend: 'native', port: 8000,
    format: 'fp8', cacheRoot: path.join(data, 'compile'), args: [...args, '--kv-transfer-config', JSON.stringify(native)]};
  const custom = {...original, id: 'custom-audit', name: '自定义连接器验收', args: [...args, '--kv-transfer-config', '{"kv_connector":"OpaqueConnector","opaque":true}']};
  const speculative = {...original, id: 'spec-audit', name: '投机互斥验收', args: [...args, '--speculative-config', '{"method":"mtp","num_speculative_tokens":2}']};
  fs.writeFileSync(path.join(data, 'profiles.json'), JSON.stringify([original, custom, speculative]), {mode: 0o600});
  const log = fs.openSync(path.join(data, 'manager-private.log'), 'a', 0o600);
  manager = spawn(process.execPath, [path.join(consoleRoot, 'server.mjs')], {env: {...process.env,
    CUDA_VISIBLE_DEVICES: '', SM75_SINGLE_CONTAINER: '1', SM75_CONSOLE_ROOT: data,
    SM75_CONSOLE_HOST: '127.0.0.1', SM75_CONSOLE_PORT: '19269', SM75_HARNESS_PROXY_PORT: '19270', SM75_HARNESS_PORT: '19271'}, stdio: ['ignore', log, log]});
  fs.closeSync(log);
  await until('manager readiness', async () => (await fetch(base + '/')).ok);
  const token = fs.readFileSync(path.join(data, 'key'), 'utf8').trim(); secrets.push(token);
  const access = path.join(data, 'api-access.json');
  if (fs.existsSync(access)) secrets.push(JSON.parse(fs.readFileSync(access)).key);
  const login = await fetch(base + '/console-api/login', {method: 'POST', headers: {Origin: base, 'Content-Type': 'application/json'}, body: JSON.stringify({token})});
  assert.equal(login.status, 200);
  cookie = login.headers.get('set-cookie').split(';')[0]; secrets.push(cookie);
  browser = await chromium.launch({headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}, locale: 'zh-CN'});
  const separator = cookie.indexOf('=');
  await context.addCookies([{name: cookie.slice(0, separator), value: cookie.slice(separator + 1), url: base, httpOnly: true, sameSite: 'Strict'}]);
  await context.route('**/*', async route => {
    const request = route.request(), url = new URL(request.url());
    if ((url.origin !== base && !['data:', 'blob:'].includes(url.protocol)) ||
        (request.method() !== 'GET' && (/\/(start|install|restart)(?:\/|$)/.test(url.pathname) || url.pathname.startsWith('/console-api/harness')) && !(allowStart && url.pathname === '/console-api/profiles/lmcache-audit/start'))) {
      forbidden.push({path: url.pathname, method: request.method()});
      return route.abort('blockedbyclient');
    }
    return route.continue();
  });
  page = await context.newPage();
  page.on('pageerror', error => pageErrors.push(clean(error.message)));
  page.on('console', message => {if (message.type() === 'error') consoleErrors.push({text: clean(message.text()), path: clean(message.location().url)});});
  page.on('request', request => {const url = new URL(request.url()); if (url.origin === base) requests.push({path: url.pathname, method: request.method()});});
  page.on('response', async response => {
    const url = new URL(response.url());
    if (url.origin === base && url.pathname === '/console-api/hardware' && response.status() === 400) {
      const body = await response.json().catch(() => ({}));
      hardwareFailures.push({path: url.pathname, status: response.status(), reason: clean(body.error || '')});
    }
  });
  await page.goto(base + '/?page=profiles', {waitUntil: 'domcontentloaded'});
  await edit();
  assert.equal(await page.locator('#lmcacheChunkSize').inputValue(), '');
  await page.locator('#lmcacheEnabled').check();
  await rejectedSave(/chunk size/);
  check('chunk size requires explicit input and invalid values never reach profile storage');
  await page.locator('#lmcacheChunkSize').fill('192'); // Synthetic form fixture, not a suggested model value.
  await page.locator('#lmcacheCpuGiB').fill('5.25');
  await page.locator('#lmcacheDiskPath').fill(path.join(data, 'disk-cache'));
  await page.locator('#lmcacheMinFreeDiskGiB').fill('0');
  await rejectedSave(/互斥/);
  assert.deepEqual(JSON.parse(argValue((await saved()).args, '--kv-transfer-config')), native);
  await page.locator('#lmcacheReplaceNative').click();
  assert.equal(await page.locator('#ctxCpuKvMode').inputValue(), 'off');
  await page.locator('#lmcacheCpuGiB').fill('6.25');
  assert(await page.locator('#lmcacheCpuGiB').evaluate(node => node === document.activeElement));
  let enabled = await save();
  assert.equal(enabled.lmcache.cpuGiB, 6.25);
  assert.equal(enabled.lmcache.chunkSize, 192);
  assert.equal(enabled.lmcache.diskPath, path.join(data, 'disk-cache'));
  assert.equal(argValue(enabled.args, '--kv-transfer-config'), null);
  assert.equal(argValue(enabled.args, '--kv-cache-memory-bytes'), String(2.5 * GiB));
  assert.equal(argValue(enabled.args, '--max-model-len'), '262144');
  assert.equal(JSON.parse(argValue(enabled.args, '--hf-overrides')).text_config.custom_field, 17);
  check('explicit native CPU replacement preserves context/GPU/HF settings and saves unblurred LMCache input');
  await edit();
  assert.equal(await page.locator('#lmcacheCpuGiB').inputValue(), '6.25');
  await page.reload({waitUntil: 'domcontentloaded'}); await edit();
  assert.equal(await page.locator('#lmcacheEnabled').isChecked(), true);
  assert.equal(await page.locator('#lmcacheChunkSize').inputValue(), '192');
  assert.equal(await page.locator('#lmcacheCpuGiB').inputValue(), '6.25');
  check('saved LMCache configuration survives immediate reopening and refresh');
  await page.locator('#lmcacheSettings details > summary').click();
  await snapshot('desktop', 1440); await snapshot('narrow', 420);
  await page.setViewportSize({width: 1440, height: 1000});
  await page.locator('#lmcacheCpuGiB').fill('7.5');
  await page.locator('#templateSelect').selectOption('base:single-long');
  await page.locator('#applyTemplate').click();
  assert.equal(await page.locator('#lmcacheCpuGiB').inputValue(), '7.5');
  enabled = await save();
  assert.equal(enabled.lmcache.cpuGiB, 7.5);
  assert.equal(argValue(enabled.args, '--max-num-seqs'), '1');
  check('base preset flushes pending LMCache input while preserving its independent configuration');
  await edit();
  await page.locator('details.template-save > summary').click();
  await page.locator('#templateName').fill('LMCache synthetic template');
  await page.locator('#lmcacheCpuGiB').fill('9.25');
  const templateResponse = page.waitForResponse(r => new URL(r.url()).pathname === '/console-api/templates' && r.request().method() === 'POST');
  await page.locator('#saveTemplate').click();
  assert.equal((await templateResponse).status(), 200);
  const templates = await api('templates');
  const personal = templates.personal.find(row => row.name === 'LMCache synthetic template');
  assert.equal(personal.profile.lmcache.cpuGiB, 9.25);
  await page.locator('#lmcacheCpuGiB').fill('11');
  await page.locator('#templateSelect').selectOption('personal:' + personal.id);
  await page.locator('#applyTemplate').click();
  assert.equal(await page.locator('#lmcacheCpuGiB').inputValue(), '9.25');
  enabled = await save();
  check('personal template captures unblurred LMCache configuration and restores it');
  const downloadPromise = page.waitForEvent('download');
  await page.locator('[data-profile="lmcache-audit"]').getByRole('button', {name: '导出配置', exact: true}).click();
  const download = await downloadPromise;
  const exported = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
  assert.deepEqual(exported.profile.lmcache, enabled.lmcache);
  await page.locator('#importProfile').click();
  await page.locator('#importProfileJson').fill(JSON.stringify(exported));
  await page.locator('#confirmImportProfile').click();
  await page.locator('#profileEditor').waitFor({state: 'visible'});
  assert.equal(await page.locator('#lmcacheCpuGiB').inputValue(), '9.25');
  const imported = await save();
  assert.notEqual(imported.id, enabled.id);
  assert.deepEqual(imported.lmcache, enabled.lmcache);
  check('real profile export and import preserve LMCache fields');
  await edit();
  await page.locator('#lmcacheEnabled').uncheck();
  const disabled = await save();
  assert.equal(disabled.lmcache.enabled, false);
  assert.equal(disabled.lmcache.cpuGiB, 9.25);
  await edit();
  assert.equal(await page.locator('#lmcacheEnabled').isChecked(), false);
  assert.equal(await page.locator('#lmcacheCpuGiB').inputValue(), '9.25');
  await page.locator('#lmcacheEnabled').check(); await save();
  check('disabling and re-enabling LMCache preserves independent field values');
  await edit();
  await page.locator('#powerMode').selectOption('sleep');
  await rejectedSave(/请选 P-State/);
  await page.locator('#powerMode').selectOption('pstate');
  const resumed = await save();
  assert.equal(resumed.power.mode, 'pstate');
  assert.equal(argValue(resumed.args, '--auto-sleep-idle-timeout'), '0');
  check('memory-releasing sleep is rejected and switching to P-State can be saved immediately');
  for (const [id, pattern] of [['custom-audit', /互斥/], ['spec-audit', /投机/]]) {
    await edit(id);
    await page.locator('#lmcacheEnabled').check(); await page.locator('#lmcacheChunkSize').fill('192');
    assert.equal(await page.locator('#lmcacheReplaceNative').isVisible(), false);
    await rejectedSave(pattern);
    await page.reload({waitUntil: 'domcontentloaded'}); // Discard only unsaved synthetic draft.
  }
  check('custom connectors and speculative configurations are rejected without silent removal');
  allowStart = true;
  const startResponse = page.waitForResponse(r => new URL(r.url()).pathname === '/console-api/profiles/lmcache-audit/start');
  await page.locator('[data-profile="lmcache-audit"]').getByRole('button', {name: '启动', exact: true}).click();
  const started = await startResponse;
  allowStart = false;
  assert.equal(started.status(), 400);
  const reason = clean((await started.json()).error || '');
  assert.match(reason, /LMCache 启动检查失败/);
  assert.match(reason, /No package metadata was found for lmcache|No module named ['"]lmcache|PackageNotFoundError/);
  startFailures.push({path: '/console-api/profiles/lmcache-audit/start', status: started.status(), reason});
  await until('visible dependency failure', async () => /LMCache 启动检查失败/.test(await page.locator('#notice').innerText()));
  assert.equal((await api('profiles/lmcache-audit/status')).running, false);
  assert(!fs.existsSync(path.join(data, 'lmcache-audit.lmcache.log')), 'Dependency failure must happen before starting helper');
  check('missing LMCache dependency is reported in the real UI before launching cache/model processes');
  assert.deepEqual(forbidden, [], 'No external or unauthorized service start requests');
  assert.deepEqual(pageErrors, [], 'No JavaScript exceptions');
  for (const item of hardwareFailures) assert.match(item.reason, /No module named ['"]pynvml['"]/);
  const unexpected = consoleErrors.filter(row => !((row.path.endsWith('/console-api/hardware') && hardwareFailures.length) ||
    (row.path.endsWith('/console-api/profiles/lmcache-audit/start') && startFailures.length)));
  assert.deepEqual(unexpected, [], 'No unexpected browser console errors');
  check('no JavaScript exceptions; cache/model and Harness remain stopped');
  fs.writeFileSync(path.join(output, 'saved-enabled-profile.json'), JSON.stringify(await saved(), null, 2));
} catch (error) {
  failure = clean(error.stack || error.message);
  if (page) {
    await page.screenshot({path: path.join(output, 'failure.png'), fullPage: true}).catch(() => {});
    fs.writeFileSync(path.join(output, 'failure-page.txt'), clean(await page.locator('body').innerText().catch(() => '')));
  }
  console.error(JSON.stringify({failure: clean(error.message)}));
  process.exitCode = 1;
} finally {
  await browser?.close();
  await stop(manager);
  const log = path.join(data, 'manager-private.log');
  if (fs.existsSync(log)) fs.writeFileSync(path.join(output, 'manager.log'), clean(fs.readFileSync(log, 'utf8')));
  const result = {passed: !failure, checks, failure: failure || null, pageErrors, consoleErrors, hardwareFailures, startFailures, forbidden, requests, layouts,
    sourceRoot, consoleRoot, data, base, engineStarted: false, harnessStarted: false, completedAt: new Date().toISOString()};
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({passed: !failure, checks: checks.length, output, pageErrors: pageErrors.length, forbiddenRequests: forbidden.length}));
}
