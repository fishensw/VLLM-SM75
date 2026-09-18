import fs from "node:fs";
import path from "node:path";

const neutral = { temperature: 1, top_p: 1, top_k: 0, min_p: 0, presence_penalty: 0, repetition_penalty: 1 };
const arg = (p, key) => {
  const index = p.args.indexOf(key);
  return index < 0 ? undefined : p.args[index + 1];
};

// Display metadata only: never persist these values as user overrides.
export function modelSamplingDefaults(provider, config, model, profiles, activeId) {
  const unknown = { values: {}, source: "服务未提供默认值" };
  if (provider !== "sm75-local") return unknown;
  let url;
  try { url = new URL(config.baseURL); } catch { return unknown; }
  if (!["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) return unknown;
  const matches = profiles.filter(p => Array.isArray(p.args)
    && (arg(p, "--served-model-name") || p.args[0]) === model.id
    && Number(p.port) === Number(url.port || 80));
  const profile = matches.find(p => p.id === activeId) || (matches.length === 1 ? matches[0] : null);
  if (!profile) return unknown;
  // The deployment owner's supplied Qwen3.8-27B model guidance. Match the
  // actual model path as served aliases may retain an older model name.
  const modelName = path.basename(profile.args[0]).replace(/_+/g, ".");
  const recommendations = /^qwen3[.-]8-27b(?:[-.]|$)/i.test(modelName) ? {
    thinking: { temperature: 1, top_p: .95, top_k: 20, min_p: 0, presence_penalty: 0, repetition_penalty: 1 },
    nonThinking: { temperature: .7, top_p: .8, top_k: 20, min_p: 0, presence_penalty: 1.5, repetition_penalty: 1 },
  } : undefined;
  const mode = arg(profile, "--generation-config") || "auto";
  let generation = {};
  if (mode !== "vllm") {
    try {
      generation = JSON.parse(fs.readFileSync(path.join(
        mode === "auto" ? (arg(profile, "--hf-config-path") || profile.args[0]) : mode,
        "generation_config.json"), "utf8"));
    } catch {}
  }
  let overrides = {};
  try { overrides = JSON.parse(arg(profile, "--override-generation-config") || "{}"); } catch {}
  Object.assign(generation, overrides);
  const values = { ...neutral };
  for (const key of Object.keys(neutral))
    // vLLM get_diff_sampling_param does not import presence_penalty from
    // generation_config; that recommendation must be sent in the request.
    if (key !== "presence_penalty" && Number.isFinite(generation[key])) values[key] = generation[key];
  // Pinned DSH custom-model catalog defaults to 32768. pi-ai reduces this
  // request budget to the remaining context (including its safety margin).
  values.max_tokens = model.maxTokens ?? config.defaultMaxTokens ?? 32768;
  const serverCap = mode === "auto" || mode === "vllm"
    ? overrides.max_new_tokens : generation.max_new_tokens;
  if (Number.isFinite(serverCap) && serverCap > 0)
    values.max_tokens = Math.min(values.max_tokens, serverCap);
  return {
    values,
    ...(recommendations ? { recommendations, recommendationModel: "Qwen3.8-27B" } : {}),
    source: mode === "vllm" ? "vLLM 默认值（含启动覆盖配置）" : "模型生成配置及 vLLM 默认值",
    outputSource: "工作台输出上限（受服务端及剩余上下文限制）",
    outputNote: "最大输出为默认上限，实际随剩余上下文调整；Top K 为 0 或 -1 时不限制。",
  };
}
