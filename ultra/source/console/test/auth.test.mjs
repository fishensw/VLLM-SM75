import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {EventEmitter} from 'node:events';
import {Auth, ensureSecret} from '../auth.mjs';

const temp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-auth-'));
const request = (headers = {}, address = '127.0.0.1') => ({headers: {host: 'lan:1615', ...headers}, socket: {remoteAddress: address}});
test('first credential is strong, stable and existing installations are preserved', () => {
  const root = temp(), file = path.join(root, 'key');
  const key = ensureSecret(file);
  assert.equal(Buffer.from(key, 'base64url').length, 32);
  assert.equal(ensureSecret(file), key);
  fs.writeFileSync(file, 'existing-admin-token');
  assert.equal(new Auth(root).key, 'existing-admin-token');
  fs.writeFileSync(file, '');
  assert.throws(() => new Auth(root), /凭据文件无效/);
});
test('session survives manager restart, expires absolutely, and stores only hashes', () => {
  const root = temp(); let clock = 1000;
  const auth = new Auth(root, {now: () => clock, ttlSeconds: 60});
  const login = auth.login(auth.key, request());
  const cookie = login.cookie.split(';')[0], raw = cookie.split('=')[1];
  assert(!fs.readFileSync(auth.file, 'utf8').includes(raw));
  const restarted = new Auth(root, {now: () => clock, ttlSeconds: 60});
  assert.equal(restarted.principal(request({cookie})).id, 'local-admin');
  clock += 60001;
  assert.equal(restarted.principal(request({cookie})), null);
});
test('logout closes streams and key rotation revokes old sessions without model operations', () => {
  const root = temp(), auth = new Auth(root);
  const cookie = auth.login(auth.key, request()).cookie.split(';')[0], req = request({cookie});
  const stream = new EventEmitter(); let destroyed = false;
  stream.destroy = () => {destroyed = true; stream.emit('close');};
  auth.track(req, stream); auth.revoke(auth.sessionId(req));
  assert(destroyed); assert.equal(auth.principal(req), null);
  const next = auth.login(auth.key, request()).cookie.split(';')[0];
  fs.writeFileSync(auth.keyPath, 'new-local-admin-token');
  assert.equal(auth.principal(request({cookie: next})), null);
  assert.equal(auth.login('new-local-admin-token', request()).status, 200);
});
test('LAN HTTP and explicitly trusted HTTPS proxy have correct cookies and origin checks', () => {
  const auth = new Auth(temp(), {trustedProxies: ['127.0.0.2']});
  assert(!auth.cookie('x', request()).includes('Secure'));
  assert(!auth.secure(request({'x-forwarded-proto': 'https'})));
  const proxy = request({'x-forwarded-proto': 'https', origin: 'https://lan:1615'}, '127.0.0.2');
  assert(auth.cookie('x', proxy).includes('Secure'));
  assert(auth.originAllowed(proxy));
  assert(!auth.originAllowed(request({origin: 'https://other.invalid'})));
  assert(!auth.originAllowed(request(), true));
});
test('bounded failed login attempts are rate limited and recover', () => {
  let now = 0; const auth = new Auth(temp(), {now: () => now});
  for (let i = 0; i < 10; i++) assert.equal(auth.login('wrong', request()).status, 401);
  assert.equal(auth.login(auth.key, request()).status, 429);
  now = 60001; assert.equal(auth.login(auth.key, request()).status, 200);
});
