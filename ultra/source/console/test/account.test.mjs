import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {Auth} from '../auth.mjs';
import {contains, validateNetworks} from '../network-policy.mjs';
const req=(cookie='',ip='192.168.1.5',extra={})=>({headers:{host:'lan:1615',cookie,...extra},socket:{remoteAddress:ip}});
const password='test-admin-password-123';
test('bootstrap, account authentication, restart, credential changes and local recovery',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-account-'));
 try {
  let a=new Auth(root); const token=a.key;
  const cookie=a.login(token,req()).cookie.split(';')[0];const current=req(cookie);
  const other=a.login(token,req()).cookie.split(';')[0];
  await assert.rejects(a.updateSecurity({action:'account',username:'admin',password},req()),/先登录/);
  await assert.rejects(a.updateSecurity({action:'account',username:'admin',password:'short'},current),/密码长度/);
  await a.updateSecurity({action:'account',username:'admin',password},current);
  assert(a.principal(current)); assert.equal(a.principal(req(other)),null);
  assert.equal(a.login(token,req()).status,401);
  assert.equal(a.principal(req('',undefined,{authorization:'Bearer '+token})),null);
  assert.equal((await a.loginAccount('admin','bad',req())).status,401);
  assert.equal((await a.loginAccount('wrong',password,req())).status,401);
  assert.equal((await a.loginAccount('admin',password,req())).status,200);
  assert(!fs.readFileSync(a.policyPath,'utf8').includes(password));
  a=new Auth(root);assert(a.principal(current));assert.equal(a.securityInfo(current).username,'admin');
  const logged=(await a.loginAccount('admin',password,req())).cookie.split(';')[0];
  await assert.rejects(a.updateSecurity({action:'account',username:'admin2',password},current),/当前密码/);
  await a.updateSecurity({action:'account',username:'admin2',password:password+'2',currentPassword:password},current);
  assert.equal(a.principal(req(logged)),null);
  assert.equal((await a.loginAccount('admin',password,req())).status,401);
  assert.equal((await a.loginAccount('admin2',password+'2',req())).status,200);
  execFileSync(process.execPath,[fileURLToPath(new URL('../auth-cli.mjs',import.meta.url)),'recover'],{env:{...process.env,SM75_CONSOLE_ROOT:root}});
  assert.equal(a.principal(current),null);assert.equal(a.securityInfo(current).accountConfigured,false);
  assert.notEqual(a.key,token);assert.equal(a.login(a.key,req()).status,200);
 } finally {fs.rmSync(root,{recursive:true,force:true});}
});
test('CIDR policy persists, rejects spoofed headers and prevents accidental lockout',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-network-'));
 try {
  const a=new Auth(root), current=req(a.login(a.key,req()).cookie.split(';')[0]);
  await a.updateSecurity({action:'account',username:'admin',password},current);
  await assert.rejects(a.updateSecurity({action:'networks',networks:['10.0.0.0/8'],currentPassword:password},current),/当前连接地址/);
  await a.updateSecurity({action:'networks',networks:['192.168.1.0/24','fd00::/64'],currentPassword:password},current);
  assert(a.networkAllowed(req('', '::ffff:192.168.1.6')));
  assert(a.networkAllowed(req('', 'fd00::123')));
  assert(!a.networkAllowed(req('', '192.168.2.5',{'x-forwarded-for':'192.168.1.5'})));
  assert.equal((await a.loginAccount('admin',password,req('','10.0.0.1'))).status,403);
  const b=new Auth(root);assert(b.principal(current));assert(!b.networkAllowed(req('','10.0.0.1')));
  await a.updateSecurity({action:'networks',networks:[],currentPassword:password},current);
  assert(a.networkAllowed(req('','10.0.0.1')));
 }finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('IPv4 and IPv6 network boundaries and invalid ranges',()=>{
 assert(contains('192.168.1.0/24','192.168.1.255'));assert(!contains('192.168.1.0/24','192.168.2.0'));
 assert(contains('::/0','2001:db8::1'));assert(contains('::1','::1'));assert(!contains('::1','::2'));
 assert(!contains('0.0.0.0/0','::1'));assert(contains('fd00::/64','fd00::ffff'));assert(!contains('fd00::/64','fd00:0:0:1::1'));
 for(const n of ['192.168.1.0/33','::/129','bad','1.1.1.1/-1','1.1.1.1/']) assert.throws(()=>validateNetworks([n]));
});
