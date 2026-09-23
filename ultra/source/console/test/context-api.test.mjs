import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {fileURLToPath} from 'node:url';
import {argValue, modelContext, readContextSettings, applyContextSettings, oneMillionContext,retargetContextModel} from '../public/context-config.js';
import {makeCommand} from '../config.mjs';
import {managedHarnessConfig} from '../harness-config.mjs';

test('context model metadata, save/reload and executable argv preserve YaRN and independent KV budgets', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-context-api-'));
  const model = path.join(root, 'model');
  fs.mkdirSync(model);
  const config = {model_type: 'qwen3_8', text_config: {model_type: 'qwen3_8_text', max_position_embeddings: 262144,
    rope_parameters: {rope_type: 'default', rope_theta: 10000000, partial_rotary_factor: 0.25, mrope_section: [11, 11, 10]}}};
  fs.writeFileSync(path.join(model, 'config.json'), JSON.stringify(config));
  const base = 'http://127.0.0.1:19268';
  const child = spawn(process.execPath, [fileURLToPath(new URL('../server.mjs', import.meta.url))], {
    env: {...process.env, SM75_SINGLE_CONTAINER: '1', SM75_CONSOLE_ROOT: root, SM75_CONSOLE_HOST: '127.0.0.1', SM75_CONSOLE_PORT: '19268'}, stdio: 'ignore',
  });
  try {
    let ready = false;
    for (let i = 0; i < 100; i++) {
      try { ready = (await fetch(base + '/')).ok; if (ready) break; } catch {}
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    assert(ready, 'manager must start');
    const headers = {Authorization: 'Bearer ' + fs.readFileSync(path.join(root, 'key'), 'utf8').trim(), 'Content-Type': 'application/json'};
    const metadata = modelContext(config);
    const originalArgs = [model, '--tensor-parallel-size', '4', '--seed', '42', '--hf-overrides', '{"architectures":["Synthetic"],"text_config":{"custom_field":17}}'];
    const values = {...oneMillionContext(readContextSettings(originalArgs, metadata), metadata), gpuKvGiB: '2.5', cpuKvMode: 'native', cpuKvGiB: '16.5'};
    const args = applyContextSettings(originalArgs, values, metadata);
    const input = {id: 'long-context', args, port: 8000, backend: 'native', format: 'fp8', cacheRoot: path.join(root, 'cache')};
    let response = await fetch(base + '/console-api/profiles', {method: 'POST', headers, body: JSON.stringify(input)});
    assert.equal(response.status, 200, await response.text());
    const saved = (await (await fetch(base + '/console-api/profiles', {headers})).json()).find(p => p.id === input.id);
    const command = makeCommand(saved, 'synthetic-engine-key');
    assert.equal(command.args[0], 'serve');
    assert.equal(argValue(saved.args, '--max-model-len'), '1048576');
    assert.equal(argValue(saved.args, '--kv-cache-memory-bytes'), String(2.5 * 2 ** 30));
    assert.equal(JSON.parse(argValue(saved.args, '--kv-transfer-config')).kv_connector_extra_config.cpu_bytes_to_use, 16.5 * 2 ** 30);
    const hf = JSON.parse(argValue(saved.args, '--hf-overrides'));
    assert.equal(hf.text_config.max_position_embeddings, 1048576);
    assert.deepEqual(hf.text_config.rope_parameters.mrope_section, [11, 11, 10]);
    assert.equal(hf.text_config.custom_field, 17);
    assert.equal(argValue(command.args.slice(1), '--hf-overrides'), argValue(saved.args, '--hf-overrides'));
    assert.equal(managedHarnessConfig(saved).provider.models[0].contextWindow, 1048576);
    const catalog = await (await fetch(base + '/console-api/models', {headers})).json();
    assert.deepEqual(catalog.find(row => row.path === model).context, metadata);
    const module = await fetch(base + '/context-config.js');
    assert.equal(module.status, 200);
    assert.match(module.headers.get('content-type'), /javascript/);
    assert.match(await module.text(), /export function applyContextSettings/);
    const otherModel = path.join(root, 'other-model');
    fs.mkdirSync(otherModel);
    fs.writeFileSync(path.join(otherModel, 'config.json'), JSON.stringify({model_type: 'llama',
      max_position_embeddings: 32768, rope_theta: 500000, quantization_config: {quant_method: 'fp8'}}));
    const registered = await (await fetch(base + '/console-api/models/register', {method: 'POST', headers, body: JSON.stringify({path: otherModel})})).json();
    const use = await fetch(base + '/console-api/models/use', {method: 'POST', headers,
      body: JSON.stringify({model: registered.id, template: input.id, start: false})});
    assert.equal(use.status, 200, await use.clone().text());
    const switchedId = (await use.json()).profile;
    const switched = (await (await fetch(base + '/console-api/profiles', {headers})).json()).find(row => row.id === switchedId);
    assert.equal(switched.args[0], otherModel);
    assert.equal(argValue(switched.args, '--max-model-len'), '32768');
    assert.equal(JSON.parse(argValue(switched.args, '--hf-overrides')).text_config.rope_parameters, undefined);
    assert.equal(argValue(switched.args, '--kv-cache-memory-bytes'), String(2.5 * 2 ** 30));

    const bad = {...input, args: [...args, '--kv-offloading-size=32']};
    response = await fetch(base + '/console-api/profiles', {method: 'POST', headers, body: JSON.stringify(bad)});
    assert.equal(response.status, 400);
    assert.match((await response.json()).error, /不能.*同时/);
    const custom = {kv_connector: 'OffloadingConnector', kv_role: 'kv_consumer', kv_connector_extra_config: {custom: 7}};
    response = await fetch(base + '/console-api/profiles', {method: 'POST', headers, body: JSON.stringify({...input, id: 'custom-cache', args: [model, '--kv-transfer-config=' + JSON.stringify(custom)]})});
    assert.equal(response.status, 200);
    const customSaved = (await response.json()).find(p => p.id === 'custom-cache');
    assert.deepEqual(JSON.parse(argValue(customSaved.args, '--kv-transfer-config')), custom);
  } finally {
    if (child.exitCode === null) {const stopped = once(child, 'exit'); child.kill(); await stopped;}
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test('Harness accepts explicit 1M token suffix and leaves auto context to the engine', () => {
  const profile = {args: ['model', '--max-model-len=1M'], port: 8000};
  assert.equal(managedHarnessConfig(profile).provider.models[0].contextWindow, 1048576);
  profile.args.push('--served-model-name', 'first', '--served-model-name=actual-served');
  assert.equal(managedHarnessConfig(profile).defaultModel.model, 'actual-served');
  profile.args = ['model', '--max-model-len', 'auto'];
  assert.equal(managedHarnessConfig(profile).provider.models[0].contextWindow, undefined);
});


test('disabling YaRN also restores native length when serving cap is between native and extended limits', () => {
  const metadata = {source: 'root', maxModelLen: 262144, originalMaxPositionEmbeddings: 262144,
    ropeParameters: {rope_type: 'default', rope_theta: 1000000}};
  const base = ['model'];
  const enabled = applyContextSettings(base, {...oneMillionContext(readContextSettings(base, metadata), metadata), maxModelLen: '524288'}, metadata);
  const disabled = applyContextSettings(enabled, {...readContextSettings(enabled, metadata), yarnEnabled: false}, metadata);
  assert.equal(argValue(disabled, '--max-model-len'), '262144');
  assert.equal(JSON.parse(argValue(disabled, '--hf-overrides')).max_position_embeddings, undefined);
});


test('changing models clears old YaRN fields and rechecks the serving cap while retaining KV budgets', () => {
  const metadata = modelContext({text_config: {max_position_embeddings: 262144,
    rope_parameters: {rope_type: 'default', rope_theta: 10000000, mrope_section: [11, 11, 10]}}});
  const base = ['model-a', '--hf-overrides', '{"other":17,"text_config":{"unchanged":true}}'];
  const enabled = applyContextSettings(base, {...oneMillionContext(readContextSettings(base, metadata), metadata), gpuKvGiB: '2.5', cpuKvMode: 'native', cpuKvGiB: '16'}, metadata);
  const next = retargetContextModel(enabled, 'model-b', {source: 'root', maxModelLen: 32768, originalMaxPositionEmbeddings: 32768, ropeTheta: 500000});
  assert.equal(next[0], 'model-b');
  assert.equal(argValue(next, '--max-model-len'), '32768');
  assert.deepEqual(JSON.parse(argValue(next, '--hf-overrides')), {other: 17, text_config: {unchanged: true}});
  assert.equal(argValue(next, '--kv-cache-memory-bytes'), String(2.5 * 2 ** 30));
  assert.equal(argValue(next, '--kv-transfer-config'), argValue(enabled, '--kv-transfer-config'));
  assert.deepEqual(retargetContextModel(enabled, 'model-a', metadata), enabled);
  assert.equal(argValue(retargetContextModel(enabled, 'unknown-model'), '--max-model-len'), 'auto');
});


test('retargeting also discards old default or layered RoPE overrides after YaRN was disabled', () => {
  for (const rope of [{rope_type: 'default', rope_theta: 1234567}, {full_attention: {rope_type: 'default', rope_theta: 1234567}}]) {
    const args = ['a', '--hf-overrides', JSON.stringify({other: 17, partial_rotary_factor: 0.5, text_config: {rope_parameters: rope, rope_theta: 1234567, partial_rotary_factor: 0.5, max_position_embeddings: 262144}})];
    const next = retargetContextModel(args, 'b', {source: 'root', maxModelLen: 32768, ropeTheta: 500000});
    assert.deepEqual(JSON.parse(argValue(next, '--hf-overrides')), {other: 17});
  }
});


test('YaRN retains an explicit auto serving cap so GPU capacity can still determine the runtime length', () => {
  const metadata = {source: 'root', maxModelLen: 262144, originalMaxPositionEmbeddings: 262144, ropeTheta: 1000000};
  const base = ['model'];
  const output = applyContextSettings(base, {...readContextSettings(base, metadata), yarnEnabled: true, yarnFactor: '4', maxModelLen: 'auto'}, metadata);
  assert.equal(argValue(output, '--max-model-len'), 'auto');
  assert.equal(JSON.parse(argValue(output, '--hf-overrides')).max_position_embeddings, 1048576);
});
