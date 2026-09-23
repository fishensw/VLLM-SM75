import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GiB, argValue, setArg, parseTokenCount, modelContext, readContextSettings,
  oneMillionContext, applyContextSettings, validateContextArgs,
} from '../public/context-config.js';

// Representative multimodal Qwen configuration: only text_config determines
// the language model's position limit and rotary parameters.
const qwen = {
  model_type: 'qwen3_5', max_position_embeddings: 16384, rope_theta: 500000,
  text_config: {
    model_type: 'qwen3_5_text', max_position_embeddings: 262144,
    rope_parameters: {
      rope_type: 'default', rope_theta: 10000000,
      partial_rotary_factor: 0.25, mrope_section: [11, 11, 10],
    },
  },
};
const metadata = () => modelContext(qwen);
const hf = args => JSON.parse(argValue(args, '--hf-overrides', '{}'));
const kv = args => JSON.parse(argValue(args, '--kv-transfer-config', '{}'));
const flagCount = (args, key) => args.filter(x => x === key || x.startsWith(key + '=')).length;
function apply(args = ['model'], changes = {}, model = metadata()) {
  return applyContextSettings(args, {...readContextSettings(args, model), ...changes}, model);
}
const million = {maxModelLen: '1M', yarnEnabled: true, yarnOriginal: '262144', yarnFactor: '4'};

test('text_config takes precedence over root and its rotary metadata is detached', () => {
  const result = metadata();
  assert.equal(result.source, 'text_config');
  assert.equal(result.modelType, 'qwen3_5_text');
  assert.equal(result.maxModelLen, 262144);
  assert.equal(result.originalMaxPositionEmbeddings, 262144);
  assert.equal(result.ropeTheta, 10000000);
  assert.equal(result.ropeParameters.partial_rotary_factor, 0.25);
  result.ropeParameters.mrope_section[0] = 999;
  assert.deepEqual(qwen.text_config.rope_parameters.mrope_section, [11, 11, 10]);
});

test('root-only models retain their own position and theta metadata', () => {
  const model = modelContext({model_type: 'llama', max_position_embeddings: 131072, rope_theta: 500000});
  assert.equal(model.source, 'root');
  const result = apply(['llama'], {...million, yarnOriginal: '131072', yarnFactor: '8'}, model);
  assert.equal(hf(result).max_position_embeddings, 1048576);
  assert.equal(hf(result).rope_parameters.rope_theta, 500000);
  assert.equal(hf(result).text_config, undefined);
});

test('1M shortcut means 1048576 tokens and factor 4 from native 262144', () => {
  const values = oneMillionContext(readContextSettings(['model'], metadata()), metadata());
  assert.equal(values.maxModelLen, '1048576');
  assert.equal(values.yarnOriginal, '262144');
  assert.equal(values.yarnFactor, '4');
  assert.equal(values.yarnEnabled, true);
  const result = applyContextSettings(['model'], values, metadata());
  assert.equal(argValue(result, '--max-model-len'), '1048576');
  assert.equal(hf(result).text_config.max_position_embeddings, 1048576);
  assert.equal(hf(result).text_config.rope_parameters.factor, 4);
  assert.equal(hf(result).max_position_embeddings, undefined);
  assert(!result.some(value => value.includes('ALLOW_LONG_MAX_MODEL_LEN')));
});

test('decimal m and binary M have different CLI meanings', () => {
  assert.equal(parseTokenCount('1m'), 1000000);
  assert.equal(parseTokenCount('1M'), 1048576);
  assert.equal(parseTokenCount('1.5M'), 1572864);
  assert.equal(parseTokenCount('1.5k'), 1500);
  assert.equal(parseTokenCount('1K'), 1024);
  assert.equal(parseTokenCount('-1', {auto: true}), null);
  assert.equal(parseTokenCount('auto', {auto: true}), null);
});

test('fractional YaRN factors expand the real position limit while respecting a smaller serving cap', () => {
  const result = apply(['model'], {...million, yarnFactor: '1.5', maxModelLen: '300000'});
  assert.equal(hf(result).text_config.rope_parameters.factor, 1.5);
  assert.equal(hf(result).text_config.max_position_embeddings, 393216);
  assert.equal(argValue(result, '--max-model-len'), '300000');
  const automatic = apply(['model'], {...million, yarnFactor: '1.5', maxModelLen: 'auto'});
  assert.equal(argValue(automatic, '--max-model-len'), 'auto');
  assert.equal(hf(automatic).text_config.max_position_embeddings, 393216);
  assert.throws(() => apply(['model'], {...million, yarnFactor: '1.5', maxModelLen: '393217'}), /超过/);
});

test('native and explicit mRoPE, theta, partial rotary and unrelated HF fields survive enabling', () => {
  const overrides = {
    architectures: ['Qwen3_5ForConditionalGeneration'], root_extension: {enabled: true},
    text_config: {
      layer_types: ['linear_attention', 'full_attention'], partial_rotary_factor: 0.5,
      rope_parameters: {rope_theta: 900000, partial_rotary_factor: 0.75, mrope_section: [12, 10, 10]},
    },
  };
  const args = ['model', '--hf-overrides', JSON.stringify(overrides), '--tensor-parallel-size', '4'];
  const before = JSON.stringify(args);
  const result = apply(args, million);
  const out = hf(result);
  assert.deepEqual(out.architectures, overrides.architectures);
  assert.deepEqual(out.root_extension, overrides.root_extension);
  assert.deepEqual(out.text_config.layer_types, overrides.text_config.layer_types);
  assert.equal(out.text_config.partial_rotary_factor, 0.5);
  assert.deepEqual(out.text_config.rope_parameters, {
    rope_type: 'yarn', rope_theta: 900000, partial_rotary_factor: 0.75,
    mrope_section: [12, 10, 10], factor: 4, original_max_position_embeddings: 262144,
  });
  assert.equal(argValue(result, '--tensor-parallel-size'), '4');
  assert.equal(JSON.stringify(args), before, 'the input argument list is not mutated');
});

test('old rope_scaling/type input migrates at the same text-model location', () => {
  const args = ['model', '--hf-overrides', JSON.stringify({text_config: {
    rope_scaling: {type: 'linear', factor: 2, rope_theta: 123456}, custom_field: 'keep',
  }})];
  const text = hf(apply(args, million)).text_config;
  assert.equal(text.rope_scaling, undefined);
  assert.equal(text.rope_parameters.type, undefined);
  assert.equal(text.rope_parameters.rope_type, 'yarn');
  assert.equal(text.rope_parameters.rope_theta, 123456);
  assert.equal(text.custom_field, 'keep');
});

test('ambiguous or wrong root RoPE overrides are not moved silently into text_config', () => {
  for (const overrides of [
    {rope_parameters: {rope_type: 'default'}},
    {rope_parameters: {rope_type: 'default'}, text_config: {rope_parameters: {rope_type: 'default'}}},
  ]) {
    assert.throws(() => apply(['model', '--hf-overrides', JSON.stringify(overrides)], million), /顶层|text_config/);
  }
});

test('unknown native length remains unknown and cannot activate the 1M shortcut', () => {
  const model = modelContext({model_type: 'custom'});
  assert.equal(model.maxModelLen, null);
  assert.equal(model.originalMaxPositionEmbeddings, null);
  const values = readContextSettings(['model'], model);
  assert.equal(values.yarnOriginal, '');
  assert.throws(() => oneMillionContext(values, model), /原生/);
  assert.throws(() => apply(['model'], {yarnEnabled: true, yarnOriginal: '', yarnFactor: '4'}, model), /原生/);
});

test('layered RoPE blocks the generic shortcut and preserves its model-specific structure', () => {
  const model = modelContext({max_position_embeddings: 131072, rope_parameters: {
    full_attention: {rope_type: 'default', rope_theta: 1000000},
    sliding_attention: {rope_type: 'default', rope_theta: 10000},
  }});
  const values = readContextSettings(['model'], model);
  assert.equal(values.yarnEditable, false);
  assert(values.warnings.some(value => value.includes('分层')));
  assert.throws(() => oneMillionContext(values, model), /分层/);
  assert.throws(() => apply(['model'], million, model), /分层/);
  assert.deepEqual(model.ropeParameters.full_attention, {rope_type: 'default', rope_theta: 1000000});
});

test('disabling an added YaRN override restores native length and native rotary fields', () => {
  const enabled = apply(['model', '--hf-overrides', JSON.stringify({text_config: {attention_bias: false}, unrelated: 7})], million);
  const disabled = apply(enabled, {yarnEnabled: false});
  assert.equal(argValue(disabled, '--max-model-len'), '262144');
  assert.equal(hf(disabled).text_config.max_position_embeddings, undefined);
  assert.deepEqual(hf(disabled).text_config.rope_parameters, qwen.text_config.rope_parameters);
  assert.equal(hf(disabled).text_config.attention_bias, false);
  assert.equal(hf(disabled).unrelated, 7);
});

test('disabling YaRN preserves explicit theta/mRoPE overrides rather than replacing them with defaults', () => {
  const override = {rope_theta: 123456, partial_rotary_factor: 0.5, mrope_section: [10, 11, 11]};
  const args = ['model', '--hf-overrides', JSON.stringify({text_config: {rope_parameters: override}})];
  const result = apply(apply(args, million), {yarnEnabled: false});
  const rope = hf(result).text_config.rope_parameters;
  for (const [key, value] of Object.entries(override)) assert.deepEqual(rope[key], value, key);
  assert.equal(rope.rope_type, 'default');
  assert.equal(rope.factor, undefined);
});

test('disabling YaRN does not increase a manually reduced serving limit', () => {
  const enabled = apply(['model'], {...million, maxModelLen: '131072'});
  const disabled = apply(enabled, {yarnEnabled: false});
  assert.equal(argValue(disabled, '--max-model-len'), '131072');
});

test('GPU GiB is per GPU and CPU GiB is one independent offloading budget', () => {
  const args = ['model', '--tensor-parallel-size', '4', '--gpu-memory-utilization', '0.88'];
  const result = apply(args, {gpuKvGiB: '1.25', cpuKvMode: 'native', cpuKvGiB: '2.5'});
  assert.equal(Number(argValue(result, '--kv-cache-memory-bytes')), 1.25 * GiB);
  assert.equal(kv(result).kv_connector, 'OffloadingConnector');
  assert.equal(kv(result).kv_role, 'kv_both');
  assert.equal(kv(result).kv_connector_extra_config.spec_name, 'CPUOffloadingSpec');
  assert.equal(kv(result).kv_connector_extra_config.cpu_bytes_to_use, 2.5 * GiB);
  assert.equal(argValue(result, '--gpu-memory-utilization'), '0.88');
  assert.equal(argValue(result, '--max-model-len'), 'auto');
  const values = readContextSettings(result, metadata());
  assert.equal(values.gpuKvGiB, '1.25');
  assert.equal(values.cpuKvGiB, '2.5');
  assert(values.warnings.some(value => value.includes('不能相加')));
  assert(values.warnings.some(value => value.includes('每卡')));
});

test('GPU automatic and CPU disabled remove only their own settings', () => {
  const enabled = apply(['model', '--cpu-offload-gb', '3'], {gpuKvGiB: '2', cpuKvMode: 'native', cpuKvGiB: '4'});
  const result = apply(enabled, {gpuKvGiB: '', cpuKvMode: 'off'});
  assert.equal(argValue(result, '--kv-cache-memory-bytes'), null);
  assert.equal(argValue(result, '--kv-transfer-config'), null);
  assert.equal(argValue(result, '--cpu-offload-gb'), '3', 'weight offload is a different feature');
});

test('native offloading updates retain connector options unrelated to capacity', () => {
  const config = {kv_connector: 'OffloadingConnector', kv_role: 'kv_both', engine_id: 'audit',
    kv_connector_extra_config: {spec_name: 'CPUOffloadingSpec', cpu_bytes_to_use: GiB, block_size: 1664}};
  const result = apply(['model', '--kv-transfer-config', JSON.stringify(config)], {cpuKvMode: 'native', cpuKvGiB: '3'});
  assert.equal(kv(result).engine_id, 'audit');
  assert.equal(kv(result).kv_connector_extra_config.block_size, 1664);
  assert.equal(kv(result).kv_connector_extra_config.cpu_bytes_to_use, 3 * GiB);
});

test('unknown connectors and custom roles/specs are preserved until explicitly disabled', () => {
  const custom = [
    {kv_connector: 'LMCacheConnectorV1', kv_connector_extra_config: {cpu_bytes_to_use: 99, arbitrary: ['keep']}},
    {kv_connector: 'OffloadingConnector', kv_role: 'kv_consumer'},
    {kv_connector: 'OffloadingConnector', kv_connector_extra_config: {spec_name: 'RemoteOffloadingSpec'}},
  ];
  for (const config of custom) {
    const raw = JSON.stringify(config), args = ['model', '--kv-transfer-config', raw];
    assert.equal(readContextSettings(args, metadata()).cpuKvMode, 'custom');
    const result = apply(args, {gpuKvGiB: '1'});
    assert.equal(argValue(result, '--kv-transfer-config'), raw);
    assert.throws(() => apply(args, {cpuKvMode: 'native', cpuKvGiB: '2'}), /自定义/);
    assert.equal(argValue(apply(args, {cpuKvMode: 'off'}), '--kv-transfer-config'), null);
  }
});

test('native shortcut and equals syntax become one explicit CPU connector without stale flags', () => {
  const args = ['model', '--kv-offloading-size=12.5', '--kv-offloading-backend=native'];
  assert.equal(readContextSettings(args).cpuKvGiB, '12.5');
  const result = apply(args, {cpuKvMode: 'native', cpuKvGiB: '6'});
  assert.equal(argValue(result, '--kv-offloading-size'), null);
  assert.equal(argValue(result, '--kv-offloading-backend'), null);
  assert.equal(kv(result).kv_connector_extra_config.cpu_bytes_to_use, 6 * GiB);
  assert.equal(flagCount(result, '--kv-transfer-config'), 1);
});

test('non-native shortcut is not silently converted into the native connector', () => {
  const args = ['model', '--kv-offloading-size', '8', '--kv-offloading-backend', 'lmcache'];
  assert.equal(readContextSettings(args).cpuKvMode, 'custom');
  assert.equal(argValue(apply(args, {gpuKvGiB: '1'}), '--kv-offloading-backend'), 'lmcache');
  assert.throws(() => apply(args, {cpuKvMode: 'native', cpuKvGiB: '8'}), /自定义/);
});

test('repeated and equals flags use the final value and are deduplicated by editing', () => {
  const args = ['model', '--max-model-len=32k', '--max-model-len', '64K',
    '--kv-cache-memory-bytes=1G', '--kv-cache-memory-bytes', '2G', '--trust-remote-code'];
  assert.equal(readContextSettings(args).maxModelLen, '64K');
  assert.equal(readContextSettings(args).gpuKvGiB, '2');
  const result = apply(args, {maxModelLen: '128K', gpuKvGiB: '3'});
  assert.equal(flagCount(result, '--max-model-len'), 1);
  assert.equal(flagCount(result, '--kv-cache-memory-bytes'), 1);
  assert.equal(argValue(result, '--max-model-len'), '131072');
  assert(result.includes('--trust-remote-code'));
  assert.equal(result[0], 'model');
});

test('removing malformed flag does not swallow the next option and consumes numeric -1 values', () => {
  assert.deepEqual(setArg(['model', '--max-model-len', '--tp', '4'], '--max-model-len', null), ['model', '--tp', '4']);
  assert.deepEqual(setArg(['model', '--max-model-len', '-1', '--tp', '4'], '--max-model-len', 'auto'), ['model', '--tp', '4', '--max-model-len', 'auto']);
});

test('invalid token counts are rejected rather than rounded or made automatic', () => {
  for (const value of ['0', '-2', '1.25', '1MiB', 'NaN', 'Infinity', '9007199254740992', true])
    assert.throws(() => parseTokenCount(value), undefined, String(value));
});

test('missing max-model-len value is rejected before invoking vLLM', () => {
  assert.throws(() => validateContextArgs(['model', '--max-model-len']));
  assert.throws(() => validateContextArgs(['model', '--max-model-len', '--tp', '4']));
});

test('bad JSON, removed rope flag and conflicting CPU definitions fail validation', () => {
  for (const raw of ['{', '[]', 'null', '2']) assert.throws(() => validateContextArgs(['model', '--hf-overrides', raw]), /JSON/);
  assert.throws(() => validateContextArgs(['model', '--rope-scaling', '{"factor":4}']), /移除/);
  assert.throws(() => validateContextArgs(['model', '--kv-offloading-size', '8', '--kv-transfer-config', '{"kv_connector":"OffloadingConnector"}']), /同时/);
});

test('invalid YaRN ranges and memory capacities fail without mutating source arguments', () => {
  const args = ['model', '--tp', '4'];
  for (const patch of [
    {...million, yarnFactor: '0.5'}, {...million, yarnFactor: 'Infinity'},
    {...million, yarnOriginal: '1.5'}, {...million, yarnFactor: '9007199254740992'},
    {gpuKvGiB: '-1'}, {gpuKvGiB: '0'}, {gpuKvGiB: '1073741824'},
    {cpuKvMode: 'native', cpuKvGiB: ''}, {cpuKvMode: 'native', cpuKvGiB: 'NaN'}, {cpuKvMode: 'invalid'},
  ]) assert.throws(() => apply(args, patch), undefined, JSON.stringify(patch));
  assert.deepEqual(args, ['model', '--tp', '4']);
});

for (const [name, args] of [
  ['boolean YaRN factor', ['model', '--hf-overrides', JSON.stringify({rope_parameters: {rope_type: 'yarn', factor: true}})]],
  ['array YaRN factor', ['model', '--hf-overrides', JSON.stringify({rope_parameters: {rope_type: 'yarn', factor: [4]}})]],
  ['boolean CPU byte budget', ['model', '--kv-transfer-config', JSON.stringify({kv_connector: 'OffloadingConnector', kv_connector_extra_config: {cpu_bytes_to_use: true}})]],
]) test(`${name} is rejected instead of numeric coercion`, () => {
  assert.throws(() => validateContextArgs(args));
});

test('speculative draft warning is explicit and draft limits remain untouched', () => {
  const draft = JSON.stringify({method: 'dflash', model: '/models/draft', num_speculative_tokens: 7, max_model_len: 32768});
  const args = ['model', '--speculative-config=' + draft];
  const values = readContextSettings(args, metadata());
  assert(values.warnings.some(value => value.includes('草稿') && value.includes('尚未验收')));
  const result = apply(args, million);
  assert.equal(argValue(result, '--speculative-config'), draft);
  assert.equal(flagCount(result, '--speculative-config'), 1);
});


test('1M shortcut supports a fractional factor and refuses to shorten a longer native context', () => {
  const model = modelContext({max_position_embeddings: 786432});
  const values = oneMillionContext(readContextSettings(['model'], model), model);
  assert.equal(Number(values.yarnFactor), 4 / 3);
  const out = applyContextSettings(['model'], values, model);
  assert.equal(hf(out).max_position_embeddings, 1048576);
  const longer = modelContext({max_position_embeddings: 2097152});
  assert.throws(() => oneMillionContext(readContextSettings(['model'], longer), longer), /超过/);
});

test('disabling the extension restores a model whose native rotary scaling was already non-default', () => {
  const model = modelContext({max_position_embeddings: 131072, rope_parameters: {
    rope_type: 'yarn', factor: 4, original_max_position_embeddings: 32768,
    beta_fast: 32, beta_slow: 1, rope_theta: 500000,
  }});
  const extended = apply(['model'], {...million, yarnOriginal: '32768', yarnFactor: '32'}, model);
  const restored = apply(extended, {yarnEnabled: false}, model);
  assert.equal(argValue(restored, '--max-model-len'), '131072');
  assert.deepEqual(hf(restored).rope_parameters, model.ropeParameters);
  assert.equal(hf(restored).max_position_embeddings, undefined);
});

test('editing repeated HF and transfer options retains the final JSON and emits one copy', () => {
  const transfer = {kv_connector: 'OffloadingConnector', kv_connector_extra_config: {cpu_bytes_to_use: GiB, retained: 'last'}};
  const args = ['model', '--hf-overrides={"ignored":true}', '--hf-overrides', '{"retained":true}',
    '--kv-transfer-config={"kv_connector":"OtherConnector"}', '--kv-transfer-config', JSON.stringify(transfer)];
  const out = apply(args, {...million, cpuKvMode: 'native', cpuKvGiB: '2'});
  assert.equal(hf(out).ignored, undefined);
  assert.equal(hf(out).retained, true);
  assert.equal(flagCount(out, '--hf-overrides'), 1);
  assert.equal(flagCount(out, '--kv-transfer-config'), 1);
  assert.equal(kv(out).kv_connector_extra_config.retained, 'last');
  assert.equal(kv(out).kv_connector_extra_config.cpu_bytes_to_use, 2 * GiB);
});

test('applying the same visible context settings twice is stable', () => {
  const once = apply(['model', '--tp', '4'], {...million, gpuKvGiB: '1.5', cpuKvMode: 'native', cpuKvGiB: '4'});
  const twice = apply(once);
  assert.deepEqual(twice, once);
});
