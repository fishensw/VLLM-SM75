#!/usr/bin/env node
/** Disposable, CPU-only real-browser integration audit. Never run on an installation. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

assert.equal(process.env.SM75_DISPOSABLE_AUDIT, '1', 'Requires explicit disposable-audit guard');
assert.equal(process.platform, 'linux');
assert(fs.existsSync('/.dockerenv'), 'Only run inside the isolated audit container');
assert(!fs.existsSync('/dev/nvidia0'), 'This audit must not have GPU access');
const source = '/opt/sm75-workbench/console';
const browserRoot = '/work/browser-audit';
assert(fs.existsSync(path.join(source, 'server.mjs')));
assert(fs.existsSync(path.join(browserRoot, 'node_modules/playwright/index.mjs')));
const { chromium } = await import(pathToFileURL(path.join(browserRoot, 'node_modules/playwright/index.mjs')));
const runId = new Date().toISOString().replace(/[:.]/g, '-');
const evidence = `/work/evidence/browser-${runId}`;
const data = `${browserRoot}/runs/${runId}/data`;
const base = 'http://127.0.0.1:1616';
fs.mkdirSync(evidence, { recursive: true });
fs.mkdirSync(data, { recursive: true, mode: 0o700 });
const checks = [];
const errors = [];
const requests = [];
const ws = [];
const mockCalls = [];
const transportFailures = [];
const watcherVisuals = [];
const auditNonce = `browser-${runId}`;
const auditReply = `AUDIT_MOCK_REPLY: ${auditNonce}`;
let browser, page, manager, cookie = '', restarting = false, hardwareFailure, expectedConsoleErrors = 0;
const sensitive = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function clean(value) {
  let text = String(value).replace(/([?&]token=)[A-Za-z0-9_-]+/g, '$1[redacted]');
  for (const secret of sensitive) if (secret) text = text.split(secret).join('[redacted]');
  return text;
}
function check(name, detail = {}) {
  checks.push({ name, ...detail });
  console.log(JSON.stringify({ passed: name, ...detail }));
}
async function waitUntil(label, fn, timeout = 60000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    if (manager && manager.exitCode !== null) throw Error(`Manager exited (${manager.exitCode}); see redacted log`);
    try { const value = await fn(); if (value) return value; } catch {}
    await sleep(250);
  }
  throw Error(`Timed out: ${label}`);
}
async function api(route, body) {
  const response = await fetch(base + route, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { Cookie: cookie, Origin: base, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) throw Error(`${route}: HTTP ${response.status}`);
  return response;
}
async function dump(page, name) {
  fs.writeFileSync(path.join(evidence, `${name}.txt`), clean(await page.locator('body').innerText()));
  await page.screenshot({ path: path.join(evidence, `${name}.png`), fullPage: true });
}
async function stopChild(child) {
  if (!child || child.exitCode !== null) return;
  const stopped = once(child, 'exit');
  child.kill('SIGTERM');
  await Promise.race([stopped, sleep(5000).then(() => { if (child.exitCode === null) child.kill('SIGKILL'); })]);
}
const mock = http.createServer(async (req, res) => {
  if (req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(req.url?.endsWith('/models') ? { object: 'list', data: [{ id: 'audit-model', object: 'model', owned_by: 'audit' }] } : { status: 'ok' }));
  }
  if (req.method !== 'POST' || req.url !== '/v1/chat/completions') { res.writeHead(404); return res.end(); }
  try {
    let body = ''; for await (const chunk of req) body += chunk;
    const payload = JSON.parse(body);
    mockCalls.push({ model: payload.model, stream: payload.stream === true, messages: payload.messages?.length ?? 0,
      temperature: payload.temperature, top_k: payload.top_k, max_tokens: payload.max_tokens,
      tools: payload.tools?.length ?? 0, auditNonce: (JSON.stringify(payload.messages?.filter(message => message.role === 'user')) || '').includes(auditNonce) });
    const reply = mockCalls.at(-1).auditNonce ? auditReply : 'AUDIT_BACKGROUND_REPLY';
    if (!payload.stream) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ id: 'chatcmpl-audit', object: 'chat.completion', model: 'audit-model', choices: [{ index: 0, message: { role: 'assistant', content: reply }, finish_reason: 'stop' }], usage: { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 } }));
    }
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
    const write = (delta, finish_reason = null, usage) => res.write(`data: ${JSON.stringify({ id: 'chatcmpl-audit', object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model: 'audit-model', choices: [{ index: 0, delta, finish_reason }], ...(usage ? { usage } : {}) })}\n\n`);
    write({ role: 'assistant', content: '' });
    await sleep(100); write({ content: reply });
    await sleep(100); write({}, 'stop', { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 });
    res.end('data: [DONE]\n\n');
  } catch { res.writeHead(500); res.end('synthetic mock error'); }
});
try {
  for (const port of [1616, 1617, 3085, 8000]) {
    try { await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(500) }); throw Error(`Audit port ${port} is already occupied`); }
    catch (error) { if (error.message.startsWith('Audit port')) throw error; }
  }
  if (fs.existsSync('/dsh/home')) {
    const saved = fs.existsSync('/dsh/identity-home') ? `/dsh/browser-home-${runId}` : '/dsh/identity-home';
    fs.renameSync('/dsh/home', saved);
    check('previous synthetic home preserved', { destination: saved });
  }
  fs.mkdirSync('/dsh/home', { recursive: true, mode: 0o700 });
  fs.chownSync('/dsh/home', 1000, 1000);
  const legacy = {
    'ui-theme': { fontSize: 24, preference: 'light' }, 'locale': { preference: 'en' },
    'agent-default-model': { provider: 'legacy-local', model: 'legacy-model', reasoningEffort: 'maximum' },
    'llm-pi-ai': { providers: { 'legacy-local': { api: 'openai-completions', baseURL: 'http://127.0.0.1:8000/v1', models: [{ id: 'legacy-model', contextWindow: 8192 }] } } },
  };
  fs.writeFileSync('/dsh/home/settings.yaml', JSON.stringify(legacy), { mode: 0o600 });
  fs.chownSync('/dsh/home/settings.yaml', 1000, 1000);
  fs.writeFileSync(path.join(data, 'profiles.json'), JSON.stringify([{ id: 'audit', name: 'Synthetic audit', args: ['audit-model', '--max-model-len', '8192'], port: 8000, format: 'fp8', backend: 'native', cacheRoot: path.join(data, 'cache') }]));
  await new Promise((resolve, reject) => { mock.once('error', reject); mock.listen(8000, '127.0.0.1', resolve); });
  const output = fs.openSync(path.join(data, 'manager-private.log'), 'a', 0o600);
  manager = spawn(process.execPath, [path.join(source, 'server.mjs')], {
    env: { ...process.env, CUDA_VISIBLE_DEVICES: '', SM75_SINGLE_CONTAINER: '1', SM75_CONSOLE_ROOT: data,
      SM75_CONSOLE_HOST: '127.0.0.1', SM75_CONSOLE_PORT: '1616', SM75_HARNESS_PROXY_PORT: '1617', SM75_HARNESS_PORT: '3085', DSH_HOME: '/dsh/home' },
    stdio: ['ignore', output, output],
  });
  fs.closeSync(output);
  await waitUntil('manager', async () => (await fetch(base + '/')).ok);
  const token = fs.readFileSync(path.join(data, 'key'), 'utf8').trim(); sensitive.push(token);
  sensitive.push(JSON.parse(fs.readFileSync(path.join(data, 'api-access.json'), 'utf8')).key);
  const login = await fetch(base + '/console-api/login', { method: 'POST', headers: { Origin: base, 'Content-Type': 'application/json' }, body: JSON.stringify({ token }) });
  assert.equal(login.status, 200);
  cookie = login.headers.get('set-cookie').split(';')[0]; sensitive.push(cookie);
  await api('/console-api/harness/start', { profile: 'audit' });
  await waitUntil('native Harness readiness', async () => (await fetch('http://127.0.0.1:3085/sm75/ready')).ok, 120000);
  check('real manager and Harness started with synthetic profile');
  browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, locale: 'en-US' });
  const separator = cookie.indexOf('=');
  await context.addCookies([{ name: cookie.slice(0, separator), value: cookie.slice(separator + 1), url: base, httpOnly: true, sameSite: 'Strict' }]);
  page = await context.newPage();
  page.on('pageerror', error => errors.push({ kind: 'javascript', message: clean(error.message), restarting }));
  page.on('console', msg => { if (msg.type() === 'error') errors.push({ kind: 'console', message: clean(msg.text()), location: clean(msg.location().url || ''), restarting }); });
  page.on('response', async response => {
    const url = new URL(response.url());
    if (url.origin === base) requests.push({ path: url.pathname, status: response.status(), restarting });
    if (url.origin === base && url.pathname === '/console-api/hardware' && response.status() === 400) {
      const body = await response.json().catch(() => ({}));
      hardwareFailure = { path: url.pathname, status: response.status(), reason: clean(body.error || '') };
    }
  });
  page.on('requestfailed', request => { const url = new URL(request.url()); if (url.origin === base) transportFailures.push({ path: url.pathname, error: request.failure()?.errorText, restarting }); });
  page.on('websocket', socket => {
    const item = { path: new URL(socket.url()).pathname, receivedFrames: 0, closed: false, errors: [] }; ws.push(item);
    socket.on('framereceived', () => { item.receivedFrames++; });
    socket.on('socketerror', error => item.errors.push(clean(error))); socket.on('close', () => { item.closed = true; });
  });
  await page.goto(base + '/dsh/', { waitUntil: 'domcontentloaded' });
  await waitUntil('native page', async () => (await page.locator('body').innerText()).length > 80);
  await sleep(2500);
  const notice = page.getByRole('button', { name: 'Continue', exact: true });
  await notice.waitFor({ state: 'visible', timeout: 15000 });
  await notice.click();
  await notice.waitFor({ state: 'hidden', timeout: 15000 });
  const later = page.getByRole('button', { name: 'Configure later', exact: true });
  if (await later.isVisible().catch(() => false)) await later.click();
  await dump(page, '01-native-home');
  check('native page loaded', { title: await page.title(), fontSize: await page.evaluate(() => document.body.style.getPropertyValue('--dsh-content-font-size')) });
  if (process.argv.includes('--probe')) {
    console.log(JSON.stringify({ probe: evidence, controls: await page.locator('button,input,textarea,[contenteditable=true]').evaluateAll(nodes => nodes.slice(0,80).map(n => ({ tag: n.tagName, role: n.getAttribute('role'), label: n.getAttribute('aria-label'), placeholder: n.getAttribute('placeholder'), text: (n.innerText || '').slice(0,80) }))) }));
  } else {
    assert.equal(await page.evaluate(() => document.body.style.getPropertyValue('--dsh-content-font-size')), '24px');
    assert(await page.getByText('工作台', { exact: true }).count() > 0);
    const usage = await api('/token-usage.json?range=all'); assert(Array.isArray((await usage.json()).rows));
    const workspaceRoot = '/dsh/workspace/deepseek-harness';
    const workspacePaths = () => fs.readdirSync(workspaceRoot).filter(name => fs.statSync(path.join(workspaceRoot, name)).isDirectory()).sort();
    const workspaceBefore = workspacePaths();
    assert(workspaceBefore.includes('Default workspace'));
    assert(await page.getByText('Default workspace', { exact: true }).count() > 0);
    check('legacy 24px theme and token usage preserved');
    assert.equal(await page.title(), '工作台');
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await page.getByText('Models', { exact: true }).first().click();
    await page.getByRole('button', { name: 'Add model provider', exact: true }).click();
    await page.getByText('Custom model API', { exact: true }).click();
    await page.getByLabel('Provider ID', { exact: true }).fill('audit-added');
    await page.getByRole('textbox', { name: 'Base URL', exact: true }).fill('http://127.0.0.1:8000/v1');
    await page.getByRole('combobox', { name: 'API protocol', exact: true }).selectOption('openai-completions');
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByRole('textbox', { name: 'Model ID 1', exact: true }).fill('audit-added-model');
    await page.getByRole('button', { name: 'Create provider', exact: true }).click();
    await page.getByRole('button', { name: /^Edit audit-added$/ }).waitFor({ state: 'visible', timeout: 20000 });
    await dump(page, '02-native-models-saved');
    const require = createRequire('/opt/harness/node_modules/@deepseek-ai/dsh/package.json');
    const yaml = require('yaml');
    const persisted = () => yaml.parse(fs.readFileSync('/dsh/home/profiles/web/cordis.patch.yml', 'utf8'));
    function assertProfile() {
      const rows = persisted();
      const llm = rows.find(row => row.id === 'llm-pi-ai').config.providers;
      assert(llm['legacy-local'] && llm['sm75-local']);
      assert.equal(llm['audit-added'].baseURL, 'http://127.0.0.1:8000/v1');
      assert.equal(llm['audit-added'].api, 'openai-completions');
      assert.equal(llm['audit-added'].models[0].id, 'audit-added-model');
      const selected = rows.find(row => row.id === 'agent-default-model').config;
      assert.equal(selected.provider, 'sm75-local'); assert.equal(selected.model, 'audit-model');
      assert.equal(selected.reasoningEffort, undefined);
      assert.equal(rows.find(row => row.id === 'ui-theme').config.fontSize, 24);
    }
    assertProfile();
    check('native Models writes provider; legacy provider and theme survive; stale effort cleared');
    const oldSockets = ws.length;
    const successes = requests.filter(row => row.path.startsWith('/dsh/api/') && row.status === 200).length;
    restarting = true;
    await api('/console-api/harness/stop', {});
    await api('/console-api/harness/start', { profile: 'audit' });
    await waitUntil('same-page WebSocket receives data', () => ws.slice(oldSockets).some(socket => socket.path === '/dsh/api/remote.mux' && socket.receivedFrames > 0 && !socket.closed && socket.errors.length === 0), 60000);
    await waitUntil('same-page RPC recovery', () => requests.filter(row => row.path.startsWith('/dsh/api/') && row.status === 200).length > successes, 60000);
    await page.getByRole('button', { name: /^Edit audit-added$/ }).waitFor({ state: 'visible', timeout: 20000 });
    assertProfile();
    assert.deepEqual(workspacePaths(), workspaceBefore);
    assert(await page.getByText('Default workspace', { exact: true }).count() > 0);
    check('default workspace path and title survive restart', { paths: workspaceBefore.map(name => path.join(workspaceRoot, name)) });
    await dump(page, '03-reconnected-models');
    await sleep(1000);
    restarting = false;
    check('Harness restart recovers the same page and preserves native model edits', { websocketGenerations: ws.length });
    await api('/console-api/sampling', { action: 'model', model: 'sm75-local/audit-model', params: { temperature: 0.37, top_k: 17, max_tokens: 128 } });
    await page.getByRole('button', { name: 'Close', exact: true }).click();
    await page.getByRole('button', { name: 'New session', exact: true }).first().click();
    const composer = page.locator('[contenteditable="true"]').first();
    await composer.waitFor({ state: 'visible', timeout: 20000 });
    await composer.fill(`Return the synthetic audit reply for ${auditNonce}. Do not use tools.`);
    const beforeSend = mockCalls.length;
    await page.getByRole('button', { name: 'Send message', exact: true }).click();
    await page.locator('p').filter({ hasText: auditReply }).first().waitFor({ state: 'visible', timeout: 60000 });
    assert(mockCalls.slice(beforeSend).some(call => call.auditNonce && call.tools > 0 && call.model === 'audit-model' && call.stream && call.temperature === 0.37 && call.top_k === 17 && call.max_tokens === 128));
    const watcher = page.getByRole('button', { name: /^Watcher/ });
    await watcher.waitFor({ state: 'visible', timeout: 10000 });
    await watcher.click();
    await page.locator('[data-dsh-watcher-panel]').waitFor({ state: 'visible', timeout: 10000 });
    const inspectWatcher = async theme => {
      const panel = page.locator('[data-dsh-watcher-panel]');
      await panel.getByText('audit-model', { exact: true }).waitFor({ state: 'visible', timeout: 10000 });
      // Sample after the native 180 ms entrance animation has settled.
      await sleep(400);
      const surfaces = await page.locator('.QCz6mq_menu, .QCz6mq_workPicture').evaluateAll(elements => elements.map(element => {
        const style = getComputedStyle(element);
        const color = style.backgroundColor;
        const alpha = color.startsWith('rgba(') ? Number(color.slice(5, -1).split(',').at(-1)) : color.startsWith('rgb(') ? 1 : color === 'transparent' ? 0 : null;
        return { surface: element.className, color, alpha, opacity: Number(style.opacity), overlay: style.getPropertyValue('--dsw-alias-bg-overlay').trim() };
      }));
      assert.equal(surfaces.length, 2);
      for (const surface of surfaces) { assert.equal(surface.alpha, 1); assert.equal(surface.opacity, 1); }
      assert.equal(await page.locator('body').evaluate(element => element.hasAttribute('data-ds-dark-theme')), theme === 'dark');
      watcherVisuals.push({ theme, model: 'audit-model', surfaces });
    };
    await inspectWatcher('light');
    await dump(page, '04-mock-conversation-watcher');
    await watcher.click();
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await page.getByText('General', { exact: true }).first().click();
    await page.getByRole('button', { name: 'Dark', exact: true }).click();
    await page.getByRole('button', { name: 'Close', exact: true }).click();
    await watcher.click();
    await inspectWatcher('dark');
    await dump(page, '04b-mock-conversation-watcher-dark');
    await watcher.click();
    check('Watcher displays the settled model and opaque light/dark panels', { themes: watcherVisuals.map(row => row.theme) });
    check('real UI conversation completes against mock SSE with saved sampling parameters', { requests: mockCalls.length });
    await page.getByRole('button', { name: '使用统计', exact: true }).click();
    await page.frameLocator('iframe[title="使用统计"][src="/token-usage"]').getByText('模型用量占比', { exact: false }).waitFor({ state: 'visible', timeout: 20000 });
    await dump(page, '05-token-usage');
    const report = await (await api('/token-usage.json?range=all')).json();
    assert(report.rows.some(row => row.model === 'audit-model' && row.totalTokens > 0));
    const v4Logs = fs.readdirSync('/dsh/home/sessions', { recursive: true }).filter(file => String(file).endsWith('session.v4.jsonl.zstd'));
    assert(v4Logs.length > 0, 'Real UI session must produce a native V4 log');
    check('V4 session contributes to the embedded token usage dashboard', { logs: v4Logs.length, totalTokens: report.totals.totalTokens });
    const expectedRestartConsole = row => row.kind === 'console' && row.restarting && row.message === `WebSocket connection to 'ws://127.0.0.1:1616/dsh/api/remote.mux' failed: Connection closed before receiving a handshake response`;
    const expectedHardwareConsole = row => row.kind === 'console' && row.location === base + '/console-api/hardware' && row.message === 'Failed to load resource: the server responded with a status of 400 (Bad Request)';
    assert.equal(errors.filter(row => !expectedRestartConsole(row) && !expectedHardwareConsole(row)).length, 0, 'Unexpected browser JavaScript/console errors');
    expectedConsoleErrors = errors.filter(row => expectedRestartConsole(row) || expectedHardwareConsole(row)).length;
    assert.equal(requests.filter(row => row.status >= 400 && !(row.path === '/console-api/hardware' && row.status === 400) && !(row.restarting && row.path.startsWith('/dsh/api/') && row.status === 503)).length, 0, 'Unexpected local HTTP failures');
    assert.equal(transportFailures.filter(row => !(row.restarting && row.path.startsWith('/dsh/api/') && ['net::ERR_CONNECTION_RESET', 'net::ERR_EMPTY_RESPONSE', 'net::ERR_ABORTED'].includes(row.error))).length, 0, 'Unexpected local transport failures');
    assert(ws.slice(oldSockets).some(row => row.path === '/dsh/api/remote.mux' && row.receivedFrames > 0 && !row.closed && row.errors.length === 0));
    check('native resources, RPC and WebSocket pass; expected restart handshake errors recorded');
    if (hardwareFailure) assert.match(hardwareFailure.reason, /python|pynvml|NVML|ENOENT|not found/i);
    check('CPU-only hardware limit recorded', { hardwareProbe: requests.find(row => row.path === '/console-api/hardware')?.status, gpuInferenceTested: false });
  }
  fs.writeFileSync(path.join(evidence, 'browser-report.json'), JSON.stringify({ mode: process.argv.includes('--probe') ? 'probe' : 'audit', checks, errors, requests, transportFailures, ws, mockCalls, watcherVisuals, hardwareFailure, limitation: 'Synthetic CPU-only browser and mock API audit; no model or GPU inference. The CPU-only container lacks the production CUDA/NVML environment; the exact hardware endpoint failure is recorded and is not a passing hardware check.' }, null, 2));
  console.log(JSON.stringify({ evidence, checks: checks.length, javascriptErrors: errors.filter(row => row.kind === 'javascript').length, consoleErrors: errors.filter(row => row.kind === 'console').length, expectedConsoleErrors, ws: ws.length }));
} catch (error) {
  if (page) await dump(page, 'failure-page').catch(() => {});
  fs.writeFileSync(path.join(evidence, 'failure.json'), JSON.stringify({ error: clean(error.stack), checks, errors, requests, transportFailures, ws, mockCalls, watcherVisuals, hardwareFailure }, null, 2));
  console.error(clean(error.stack)); process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  try { if (cookie) await api('/console-api/harness/stop', {}); } catch {}
  await stopChild(manager);
  await new Promise(resolve => mock.close(resolve));
  for (const name of ['manager-private.log', 'harness-process.log']) {
    const file = path.join(data, name);
    if (fs.existsSync(file)) fs.writeFileSync(path.join(evidence, `${name.replace('-private','')}.redacted`), clean(fs.readFileSync(file, 'utf8')));
  }
}
