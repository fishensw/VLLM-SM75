import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import net from 'node:net';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const listen = server => new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server.address().port)));
const close = server => new Promise(resolve => server.close(resolve));
async function unusedPort() {const server = net.createServer(); const port = await listen(server); await close(server); return port;}

test('embedded workbench assets and websocket use native auth behind the management login', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-harness-http-'));
  const observed = [];
  const upstream = http.createServer((req, res) => {
    if (req.url === '/?token=fake-launch-token') {
      res.writeHead(302, {'set-cookie': 'native-fixture=local-cookie; HttpOnly; SameSite=Strict', location: './'});
      return res.end();
    }
    observed.push({url: req.url, cookie: req.headers.cookie, authorization: req.headers.authorization});
    res.writeHead(200, {'content-type': 'application/json', 'set-cookie': 'do-not-forward=fixture'});
    res.end(JSON.stringify({path: req.url}));
  });
  const sockets = new Set();
  upstream.on('connection', socket => {sockets.add(socket); socket.on('error', () => {}); socket.on('close', () => sockets.delete(socket));});
  upstream.on('upgrade', (req, socket) => {
    observed.push({url: req.url, cookie: req.headers.cookie, origin: req.headers.origin});
    socket.end('HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\n');
  });
  let child;
  try {
    const harnessPort = await listen(upstream), consolePort = await unusedPort(), proxyPort = await unusedPort();
    fs.writeFileSync(path.join(root, 'harness-process.log'), 'Fixture launch http://127.0.0.1/?token=fake-launch-token\n');
    child = spawn(process.execPath, [fileURLToPath(new URL('../server.mjs', import.meta.url))], {
      env: {...process.env, SM75_SINGLE_CONTAINER: '1', SM75_CONSOLE_ROOT: root,
        SM75_CONSOLE_HOST: '127.0.0.1', SM75_CONSOLE_PORT: String(consolePort),
        SM75_HARNESS_PORT: String(harnessPort), SM75_HARNESS_PROXY_PORT: String(proxyPort)},
      stdio: 'ignore',
    });
    const base = `http://127.0.0.1:${consolePort}`;
    let ready = false;
    for (let i = 0; i < 100; i++) {
      try {await fetch(base + '/console-api/session'); ready = true; break;} catch {await delay(30);}
    }
    assert(ready, 'fixture management server did not start');
    assert.equal((await fetch(base + '/assets/app.js')).status, 401);
    assert.equal((await fetch(base + '/harness-ui/')).status, 401);
    const key = fs.readFileSync(path.join(root, 'key'), 'utf8').trim();
    const headers = {authorization: `Bearer ${key}`};
    const embedded = await fetch(base + '/harness-ui/?embedded=1', {headers});
    assert.equal(embedded.status, 200);
    assert.deepEqual(await embedded.json(), {path: '/?embedded=1'});
    assert.equal((await fetch(base + '/harness-ui/', {headers: {...headers, origin: 'http://untrusted.invalid'}})).status, 403);
    const response = await fetch(base + '/assets/app.js?v=1', {headers});
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('set-cookie'), null);
    assert.deepEqual(await response.json(), {path: '/assets/app.js?v=1'});
    assert.equal(observed.at(-1).cookie, 'native-fixture=local-cookie');
    assert.equal(observed.at(-1).authorization, '');
    assert.equal((await fetch(base + '/dsh', {headers, redirect: 'manual'})).headers.get('location'), '/');
    const upgraded = await new Promise((resolve, reject) => {
      const socket = net.connect(consolePort, '127.0.0.1');
      let result = '';
      socket.setTimeout(5000, () => {socket.destroy(); reject(Error('upgrade timeout'));});
      socket.on('error', reject);
      socket.on('connect', () => socket.write(`GET /api/ws?fixture=1 HTTP/1.1\r\nHost: 127.0.0.1:${consolePort}\r\nOrigin: ${base}\r\nAuthorization: Bearer ${key}\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\n`));
      socket.on('data', chunk => {result += chunk; if (result.includes('\r\n\r\n')) {socket.destroy(); resolve(result);}});
    });
    assert.match(upgraded, /^HTTP\/1.1 101/);
    assert.equal(observed.at(-1).url, '/api/ws?fixture=1');
    assert.equal(observed.at(-1).origin, `http://127.0.0.1:${harnessPort}`);
    assert.equal(observed.at(-1).cookie, 'native-fixture=local-cookie');
  } finally {
    if (child) {const exited = new Promise(resolve => child.once('exit', resolve)); child.kill(); await exited;}
    for (const socket of sockets) socket.destroy();
    await close(upstream);
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test('retired independent Harness creation is rejected without launching a legacy Docker job', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-harness-retired-'));
  let child;
  try {
    const port = await unusedPort(), proxy = await unusedPort();
    child = spawn(process.execPath, [fileURLToPath(new URL('../server.mjs', import.meta.url))], {
      env: {...process.env, SM75_SINGLE_CONTAINER: '0', SM75_CONSOLE_ROOT: root,
        SM75_CONSOLE_HOST: '127.0.0.1', SM75_CONSOLE_PORT: String(port), SM75_HARNESS_PROXY_PORT: String(proxy)}, stdio: 'ignore'});
    const base = `http://127.0.0.1:${port}`;
    let ready = false;
    for (let i=0; i<100; i++) {try {await fetch(base + '/console-api/session'); ready = true; break;} catch {await delay(30);}}
    assert(ready);
    const headers = {authorization: 'Bearer ' + fs.readFileSync(path.join(root, 'key'), 'utf8').trim(), 'content-type': 'application/json'};
    for (const action of ['start', 'install']) {
      const response = await fetch(`${base}/console-api/harness/${action}`, {method: 'POST', headers, body: '{}'});
      assert.equal(response.status, 409);
      assert.match((await response.json()).error, /ultra/);
    }
  } finally {
    if (child) {const exited = new Promise(resolve => child.once('exit', resolve)); child.kill(); await exited;}
    fs.rmSync(root, {recursive: true, force: true});
  }
});
