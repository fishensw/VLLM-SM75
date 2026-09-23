import {argValue, parseTokenCount} from './public/context-config.js';
import fs from 'node:fs';
import {chatConfig, harnessThinking, thinkingBudgets} from './chat-config.mjs';

export function harnessPath(url) {
  return url.replace(/^\/(?:dsh|harness-ui)(?=\/|\?|$)/, '') || '/';
}

export function managedHarnessConfig(profile) {
  const args = profile.args;
  const model = argValue(args, '--served-model-name', args[0]);
  const contextWindow = parseTokenCount(argValue(args, '--max-model-len'), {auto: true});
  return {
    provider: {
      apiKeyEnv: 'SM75_ENGINE_KEY', api: 'openai-completions', thinkingBudgets,
      baseURL: `http://127.0.0.1:${profile.port}/v1`,
      compat: {supportsDeveloperRole: false, maxTokensField: 'max_tokens'},
      models: [{id: model, input: chatConfig(profile).input, ...harnessThinking(profile),
        ...(contextWindow > 0 ? {contextWindow} : {})}],
    },
    defaultModel: {provider: 'sm75-local', model},
  };
}

export function harnessPatch(config) {
  return [
    {id: 'ui-brand-official', disabled: true},
    {id: 'system-prompt', config: {personaPrefix: '你是 Ultra 工作台中的智能体。除非用户明确要求其他语言，默认使用简体中文回应，包括进度、工具失败和权限说明。代码、命令及错误标识保留原文；不得把工具失败描述为任务已完成。'}},
    // The container has no desktop XDG directory service. This config is only
    // the native first-use directory policy, not a Workspace registry edit.
    // Native initializeDefault reuses existing workspaces without renaming them.
    {id: 'workspace-controller', config: {documentsDirectory: '/dsh/workspace'}},
    {insert: [
      {id: 'sm75-workbench', name: 'sm75-workbench', config},
      {id: 'sm75-brand', name: 'sm75-brand'},
      {id: 'dsh-watcher', name: 'dsh-watcher'},
      {id: 'sm75-workspace-tools', name: 'sm75-workspace-tools'},
    ]},
  ];
}

// A legacy settings document is imported asynchronously by Harness once. Update
// only the same managed values before that one import so its interleaved write
// cannot restore a previous engine. Never recreate the obsolete document.
export function legacyHarnessConfig(file, config, parse, owner = 1000) {
  let stat;
  try { stat = fs.lstatSync(file); } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
  if (!stat.isFile() || (process.platform !== 'win32' && stat.uid !== owner))
    throw Error('Legacy Harness settings must be a regular file owned by the Harness user');
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    const opened = fs.fstatSync(fd);
    if (opened.ino !== stat.ino || opened.dev !== stat.dev)
      throw Error('Legacy Harness settings changed during migration');
    const old = parse(fs.readFileSync(fd, 'utf8')) || {};
    if (typeof old !== 'object' || Array.isArray(old)) throw Error('Invalid legacy Harness settings');
    return {...old,
      'llm-pi-ai': {...old['llm-pi-ai'], providers: {...old['llm-pi-ai']?.providers, 'sm75-local': config.provider}},
      'agent-default-model': {...config.defaultModel},
    };
  } finally { fs.closeSync(fd); }
}
