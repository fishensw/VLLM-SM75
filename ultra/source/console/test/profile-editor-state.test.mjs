import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(new URL('../public/app.js', import.meta.url), 'utf8')
  .replace('from "/live-summary.js"', 'from ' + JSON.stringify(new URL('../public/live-summary.js', import.meta.url).href))
  .replace('from "./context-config.js"', 'from ' + JSON.stringify(new URL('../public/context-config.js', import.meta.url).href))
  .replace('from "./lmcache-config.js"', 'from ' + JSON.stringify(new URL('../public/lmcache-config.js', import.meta.url).href));
const previousDocument = globalThis.document;
let fillProfilePowerFields, usesDedicatedCacheField;
try {
  globalThis.document = {getElementById: () => null};
  ({fillProfilePowerFields, usesDedicatedCacheField} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64')));
} finally {
  if (previousDocument === undefined) delete globalThis.document;
  else globalThis.document = previousDocument;
}

function powerForm() {
  const elements = Object.fromEntries([
    'powerMode', 'pstateIdle', 'pstateUtil', 'pstateConfirm', 'pstateLow', 'pstateHigh',
    'pstatePoll', 'sleepIdle', 'sleepTarget', 'profileName', 'pid', 'profileDefault',
  ].map(id => [id, {value: 'original:' + id, checked: true}]));
  return {elements, document: {getElementById: id => elements[id]}};
}

test('personal template restores P-State settings after a sleep profile without replacing target identity', () => {
  const {elements, document} = powerForm();
  fillProfilePowerFields(document, {args: ['model', '--auto-sleep-idle-timeout=12', '--auto-sleep-offload-target=exit'],
    power: {mode: 'sleep'}});
  assert.equal(elements.powerMode.value, 'sleep');
  assert.equal(elements.sleepIdle.value, 12);
  const template = {args: ['other-model'], lmcache: {enabled: true},
    power: {mode: 'pstate', idleSeconds: 9, util: 4, confirm: 0, low: 8, high: 16, poll: 3}};
  fillProfilePowerFields(document, template);
  assert.deepEqual(Object.fromEntries(['powerMode', 'pstateIdle', 'pstateUtil', 'pstateConfirm', 'pstateLow', 'pstateHigh', 'pstatePoll']
    .map(id => [id, elements[id].value])), {
    powerMode: 'pstate', pstateIdle: 9, pstateUtil: 4, pstateConfirm: 0, pstateLow: 8, pstateHigh: 16, pstatePoll: 3,
  });
  assert.equal(elements.sleepIdle.value, 30);
  assert.equal(elements.profileName.value, 'original:profileName');
  assert.equal(elements.pid.value, 'original:pid');
  assert.equal(elements.profileDefault.checked, true);
});

test('legacy or absent power settings reset all controls instead of retaining the previous profile', () => {
  const {elements, document} = powerForm();
  fillProfilePowerFields(document, {args: ['model'], power: {mode: 'pstate', idleMinutes: 2, util: 0}});
  assert.equal(elements.pstateIdle.value, 120);
  assert.equal(elements.pstateUtil.value, 0);
  fillProfilePowerFields(document, {args: ['model']});
  assert.equal(elements.powerMode.value, 'pstate');
  assert.equal(elements.pstateIdle.value, 1);
  assert.equal(elements.pstateUtil.value, 5);
  assert.equal(elements.pstateConfirm.value, 60);
  assert.equal(elements.pstateLow.value, 8);
  assert.equal(elements.pstateHigh.value, 16);
  assert.equal(elements.pstatePoll.value, 5);
  assert.equal(elements.sleepTarget.value, 'exit');
});

test('custom offloading backend and capacity remain in advanced fields so the connector can be removed', () => {
  for (const args of [
    ['model', '--kv-offloading-backend=custom', '--kv-offloading-size=8'],
    ['model', '--kv-offloading-backend', 'custom'],
  ]) {
    assert.equal(usesDedicatedCacheField(args, '--kv-offloading-backend'), false);
    assert.equal(usesDedicatedCacheField(args, '--kv-offloading-size'), false);
    assert.equal(usesDedicatedCacheField(args, '--kv-transfer-config'), false);
  }
  for (const args of [
    ['model', '--kv-offloading-backend=native', '--kv-offloading-size=8'],
    ['model', '--kv-offloading-size=8'],
  ]) {
    assert.equal(usesDedicatedCacheField(args, '--kv-offloading-backend'), true);
    assert.equal(usesDedicatedCacheField(args, '--kv-offloading-size'), true);
  }
  assert.equal(usesDedicatedCacheField(['model'], '--max-model-len'), true);
  assert.equal(usesDedicatedCacheField(['model'], '--kv-cache-memory-bytes'), true);
});
