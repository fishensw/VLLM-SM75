// Shared by the browser editor and server validation. All capacities remain separate:
// CPU offload is a prefix-cache tier, not additional active attention memory.
export const GiB = 2 ** 30;
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const clone = value => JSON.parse(JSON.stringify(value));
const positive = (value, label, integer = false) => {
  const n = Number(value);
  if (!['number', 'string'].includes(typeof value) || value === '' || value == null || !Number.isFinite(n) || n <= 0 || (integer && !Number.isSafeInteger(n)))
    throw Error(`${label}须为${integer ? '正整数' : '正数'}`);
  return n;
};

export function argValue(args, key, fallback = null) {
  let result = fallback;
  for (let i = 1; i < args.length; i++) {
    if (args[i] === key) result = args[i + 1];
    else if (args[i].startsWith(key + '=')) result = args[i].slice(key.length + 1);
  }
  return result;
}
export function setArg(args, key, value) {
  const out = [args[0]];
  for (let i = 1; i < args.length; i++) {
    if (args[i] === key) {
      if (i + 1 < args.length && !/^--?[A-Za-z]/.test(args[i + 1])) i++;
    } else if (!args[i].startsWith(key + '=')) out.push(args[i]);
  }
  if (value !== null && value !== undefined && value !== false)
    out.push(key, ...(value === true ? [] : [String(value)]));
  return out;
}
export function parseTokenCount(value, {auto = false, label = '上下文长度'} = {}) {
  if (auto && ['auto', '-1', '', null, undefined].includes(value)) return null;
  const match = String(value).trim().match(/^(\d+(?:\.\d+)?)([kKmMgG]?)$/);
  if (!match) throw Error(`${label}须为正整数或 k/K/m/M 后缀值`);
  const suffix = match[2], power = {k: 1, m: 2, g: 3}[suffix.toLowerCase()] || 0;
  const n = Number(match[1]) * (suffix === suffix.toUpperCase() ? 1024 : 1000) ** power;
  return positive(n, label, true);
}
function jsonArg(args, key) {
  const raw = argValue(args, key);
  if (raw === null) return {};
  let parsed;
  try { parsed = JSON.parse(raw); } catch { throw Error(`${key} 须为合法 JSON 对象`); }
  if (!object(parsed)) throw Error(`${key} 须为合法 JSON 对象`);
  return parsed;
}
function layered(rope) {
  return object(rope) && !('rope_type' in rope) && !('type' in rope) &&
    Object.values(rope).some(v => object(v));
}
export function modelContext(config) {
  const source = object(config.text_config) ? 'text_config' : 'root';
  const text = source === 'text_config' ? config.text_config : config;
  const rope = text.rope_parameters ?? text.rope_scaling ?? null;
  const length = Number(text.max_position_embeddings ?? text.n_positions ?? text.max_seq_len);
  const original = Number(rope?.original_max_position_embeddings ?? text.original_max_position_embeddings ?? length);
  return {
    source, modelType: text.model_type || config.model_type || null,
    maxModelLen: Number.isSafeInteger(length) && length > 0 ? length : null,
    originalMaxPositionEmbeddings: Number.isSafeInteger(original) && original > 0 ? original : null,
    ropeTheta: rope?.rope_theta ?? text.rope_theta ?? null,
    ropeParameters: object(rope) ? clone(rope) : null,
    layeredRope: layered(rope),
  };
}
function ropeLocation(hf, metadata) {
  const nested = object(hf.text_config) && (hf.text_config.rope_parameters || hf.text_config.rope_scaling);
  const root = hf.rope_parameters || hf.rope_scaling;
  if (nested && root) throw Error('同时存在顶层和 text_config RoPE 覆盖，请先在高级参数中明确位置');
  const source = nested ? 'text_config' : root ? 'root' : metadata.source || 'root';
  const section = source === 'text_config' ? (hf.text_config || {}) : hf;
  return {source, section, rope: section.rope_parameters ?? section.rope_scaling ?? null};
}
function cpuCache(args) {
  const hasTransfer = argValue(args, '--kv-transfer-config') !== null;
  const shortcut = argValue(args, '--kv-offloading-size');
  const kv = hasTransfer ? jsonArg(args, '--kv-transfer-config') : {};
  const extra = kv.kv_connector_extra_config || {};
  const native = hasTransfer && (!kv.kv_connector || kv.kv_connector === 'OffloadingConnector') &&
    (!extra.spec_name || extra.spec_name === 'CPUOffloadingSpec') && (!kv.kv_role || kv.kv_role === 'kv_both');
  if (hasTransfer) return {mode: native ? 'native' : 'custom', kv,
    gib: native ? String((extra.cpu_bytes_to_use ?? 8 * GiB) / GiB) : ''};
  if (shortcut !== null) return {mode: ['native', null].includes(argValue(args, '--kv-offloading-backend')) ? 'native' : 'custom', kv: {}, gib: String(shortcut)};
  return {mode: 'off', kv: {}, gib: '8'};
}
export function readContextSettings(args, metadata = {}) {
  const hf = jsonArg(args, '--hf-overrides');
  const location = ropeLocation(hf, metadata);
  const rope = location.rope || {};
  const yarn = (rope.rope_type ?? rope.type) === 'yarn';
  const cpu = cpuCache(args);
  const gpu = argValue(args, '--kv-cache-memory-bytes');
  const warnings = ['CPU KV 卸载扩展前缀缓存复用；活跃请求仍须装入 GPU KV，CPU 与 GPU 容量不能相加推算上下文。'];
  if (gpu !== null) warnings.push('显式 GPU KV 配额按每卡设置，会覆盖显存利用率对 KV 缓存的自动分配；减小配额可能降低可承载上下文。');
  if (argValue(args, '--speculative-config') !== null) warnings.push('投机草稿的上下文与 RoPE 必须匹配；超过草稿上限的长请求请先关闭 DFlash/MTP，或使用单独验证过的长上下文草稿。此处不会自动扩展草稿，也不保证自动跳过越界，1M 与投机组合尚未验收。');
  if (cpu.mode === 'custom') warnings.push('已有自定义 KV 连接器，保留原配置；如需改为原生 CPU KV，请先关闭自定义连接器。');
  const editable = !metadata.layeredRope && !layered(rope);
  if (!editable) warnings.push('模型使用分层 RoPE，请在高级参数中按模型说明设置，通用 YaRN 向导不覆盖分层结构。');
  return {
    maxModelLen: String(argValue(args, '--max-model-len', 'auto')),
    yarnEnabled: yarn,
    yarnOriginal: String(rope.original_max_position_embeddings ?? metadata.originalMaxPositionEmbeddings ?? ''),
    yarnFactor: String(rope.factor ?? '1'),
    gpuKvGiB: gpu === null ? '' : String(parseTokenCount(gpu, {label: 'GPU KV 字节数'}) / GiB),
    cpuKvMode: cpu.mode, cpuKvGiB: cpu.gib, warnings, yarnEditable: editable,
  };
}
export function oneMillionContext(values, metadata = {}) {
  const original = positive(values.yarnOriginal || metadata.originalMaxPositionEmbeddings, '模型原生上下文长度', true);
  const target = 1048576;
  if (original > target) throw Error('模型原生上下文已超过 1M，无需用此快捷项扩展');
  if (metadata.layeredRope || values.yarnEditable === false) throw Error('分层 RoPE 请使用模型专用配置');
  return {...values, maxModelLen: String(target), yarnEnabled: true,
    yarnOriginal: String(original), yarnFactor: String(target / original)};
}
function capacity(value, label) {
  const n = positive(value, label);
  const bytes = Math.round(n * GiB);
  if (!Number.isSafeInteger(bytes) || bytes <= 0) throw Error(`${label}超出可用范围`);
  return bytes;
}
export function applyContextSettings(args, values, metadata = {}) {
  let out = [...args];
  let target = parseTokenCount(values.maxModelLen, {auto: true});
  const hf = jsonArg(out, '--hf-overrides');
  const loc = ropeLocation(hf, metadata);
  const previous = loc.rope || {};
  const wasYarn = (previous.rope_type ?? previous.type) === 'yarn';
  if (values.yarnEnabled) {
    if (metadata.layeredRope || layered(previous)) throw Error('分层 RoPE 不支持通用 YaRN 向导，请保留模型专用配置');
    if (metadata.source === 'text_config' && loc.source === 'root' && loc.rope)
      throw Error('此模型需要 text_config 内的 RoPE，请先移除错误的顶层覆盖');
    const original = positive(values.yarnOriginal, 'YaRN 原生上下文长度', true);
    const factor = positive(values.yarnFactor, 'YaRN 倍率');
    if (factor < 1) throw Error('YaRN 倍率须不小于 1');
    const extended = Math.floor(original * factor);
    if (!Number.isSafeInteger(extended)) throw Error('YaRN 扩展长度超出范围');
    if (target !== null && target > extended) throw Error('上下文上限不能超过 YaRN 原生长度 × 倍率');
    // Merge the actual text model's native fields: mRoPE sections, partial
    // rotary factors and theta must survive the replacement of rope parameters.
    const rope = {...(metadata.ropeParameters || {}), ...previous,
      rope_type: 'yarn', factor, original_max_position_embeddings: original};
    delete rope.type;
    if (rope.rope_theta == null && metadata.ropeTheta != null) rope.rope_theta = metadata.ropeTheta;
    const section = loc.source === 'text_config' ? (hf.text_config ||= {}) : hf;
    section.rope_parameters = rope;
    delete section.rope_scaling;
    // v0.30 treats YaRN max_position_embeddings as already expanded. Setting
    // the real model limit avoids using the global ALLOW_LONG escape hatch.
    section.max_position_embeddings = extended;
  } else if (wasYarn) {
    const section = loc.source === 'text_config' ? hf.text_config : hf;
    const oldExtended = Math.floor(Number(previous.original_max_position_embeddings) * Number(previous.factor));
    const base = metadata.maxModelLen || Number(previous.original_max_position_embeddings);
    const remaining = {...previous};
    for (const key of ['rope_type', 'type', 'factor', 'original_max_position_embeddings', 'beta_fast', 'beta_slow', 'mscale', 'mscale_all_dim', 'attention_factor', 'truncate']) delete remaining[key];
    const restored = {...(metadata.ropeParameters || {}), ...remaining};
    if (Object.keys(restored).length && !restored.rope_type && !restored.type) restored.rope_type = 'default';
    delete section.rope_scaling;
    if (Object.keys(restored).length) section.rope_parameters = restored;
    else delete section.rope_parameters;
    if (section.max_position_embeddings === oldExtended) delete section.max_position_embeddings;
    if (target !== null && Number.isSafeInteger(base) && base > 0 && target > base) target = base;
    if (loc.source === 'text_config' && !Object.keys(section).length) delete hf.text_config;
  }
  out = setArg(out, '--hf-overrides', Object.keys(hf).length ? JSON.stringify(hf) : null);
  out = setArg(out, '--max-model-len', target === null ? 'auto' : target);
  out = setArg(out, '--kv-cache-memory-bytes', String(values.gpuKvGiB ?? '').trim() === '' ? null : capacity(values.gpuKvGiB, '每卡 GPU KV 容量'));
  const previousCpu = cpuCache(out);
  if (!['off', 'native', 'custom'].includes(values.cpuKvMode)) throw Error('CPU KV 模式无效');
  if (values.cpuKvMode !== 'custom') {
    if (values.cpuKvMode === 'native' && previousCpu.mode === 'custom') throw Error('请先关闭已有自定义 KV 连接器，再启用原生 CPU KV');
    out = setArg(setArg(out, '--kv-offloading-size', null), '--kv-offloading-backend', null);
    if (values.cpuKvMode === 'off') out = setArg(out, '--kv-transfer-config', null);
    else {
      const kv = previousCpu.kv;
      const next = {...kv, kv_connector: 'OffloadingConnector', kv_role: 'kv_both',
        kv_connector_extra_config: {...kv.kv_connector_extra_config, spec_name: 'CPUOffloadingSpec', cpu_bytes_to_use: capacity(values.cpuKvGiB, 'CPU KV 容量')}};
      out = setArg(out, '--kv-transfer-config', JSON.stringify(next));
    }
  }
  validateContextArgs(out);
  return out;
}
// A model choice is a boundary for model-specific YaRN fields. Keep resource
// budgets, but do not carry the old model's native mRoPE/theta into a new model.
export function retargetContextModel(args, modelPath, metadata = {}) {
  if (args[0] === modelPath) return [...args];
  let out = [...args];
  const hf = jsonArg(out, '--hf-overrides');
  for (const section of [hf, hf.text_config].filter(object)) {
    for (const field of ['rope_parameters', 'rope_scaling', 'rope_theta', 'partial_rotary_factor',
      'original_max_position_embeddings', 'max_position_embeddings']) delete section[field];
  }
  if (object(hf.text_config) && !Object.keys(hf.text_config).length) delete hf.text_config;
  out = setArg(out, '--hf-overrides', Object.keys(hf).length ? JSON.stringify(hf) : null);
  const previous = parseTokenCount(argValue(out, '--max-model-len'), {auto: true});
  const maximum = metadata.maxModelLen;
  if (previous !== null) out = setArg(out, '--max-model-len',
    Number.isSafeInteger(maximum) && maximum > 0 ? Math.min(previous, maximum) : 'auto');
  out[0] = modelPath;
  validateContextArgs(out);
  return out;
}

export function validateContextArgs(args) {
  const length = argValue(args, '--max-model-len');
  if (length === undefined || length === '') throw Error('--max-model-len 缺少参数值');
  if (length !== null) parseTokenCount(length, {auto: true});
  const gpu = argValue(args, '--kv-cache-memory-bytes');
  if (gpu !== null) parseTokenCount(gpu, {label: 'GPU KV 字节数'});
  if (argValue(args, '--rope-scaling') !== null)
    throw Error('vLLM 0.30 已移除 --rope-scaling，请通过 YaRN 配置或 --hf-overrides 设置 rope_parameters');
  const hf = jsonArg(args, '--hf-overrides');
  if (hf.text_config !== undefined && !object(hf.text_config)) throw Error('text_config 覆盖须为 JSON 对象');
  for (const section of [hf, hf.text_config].filter(object)) {
    for (const field of ['rope_parameters', 'rope_scaling'])
      if (section[field] !== undefined && section[field] !== null && !object(section[field]))
        throw Error(`${field} 须为 JSON 对象`);
    for (const rope of [section.rope_parameters, section.rope_scaling].filter(object)) {
      if ((rope.rope_type ?? rope.type) !== 'yarn') continue;
      if (positive(rope.factor, 'YaRN 倍率') < 1) throw Error('YaRN 倍率须不小于 1');
      if (rope.original_max_position_embeddings !== undefined)
        positive(rope.original_max_position_embeddings, 'YaRN 原生上下文长度', true);
    }
  }
  const kv = jsonArg(args, '--kv-transfer-config');
  if (kv.kv_connector !== undefined && (typeof kv.kv_connector !== 'string' || !kv.kv_connector.trim()))
    throw Error('KV 连接器名称须为非空字符串');
  if (kv.kv_connector_extra_config !== undefined && !object(kv.kv_connector_extra_config))
    throw Error('KV 连接器扩展配置须为 JSON 对象');
  const cpu = cpuCache(args);
  const size = argValue(args, '--kv-offloading-size');
  if (size !== null) positive(size, 'CPU KV 容量');
  if (size !== null && argValue(args, '--kv-transfer-config') !== null)
    throw Error('CPU KV 简写参数不能与 --kv-transfer-config 同时设置');
  if (cpu.mode === 'native' && cpu.kv.kv_connector_extra_config?.cpu_bytes_to_use !== undefined)
    positive(cpu.kv.kv_connector_extra_config.cpu_bytes_to_use, 'CPU KV 字节数', true);
  return args;
}
