// The managed local provider is written through Harness's native settings
// service. A CLI config override would prevent subsequent Models-page edits.
export const inject = ['settings', 'webServer'];
export function apply(ctx, config = {}) {
  let ready = false;
  let failed = false;
  let stage = 'waiting-for-plugins';
  let failure;
  let pending;
  let retryAfter = 0;
  const send = (res, value, status = 200) => {
    res.writeHead(status, {'content-type': 'application/json', 'cache-control': 'no-store'});
    res.end(JSON.stringify(value));
  };
  ctx.webServer.register({kind: 'exact', path: '/sm75/ready', handler: (_req, res) => {
    // A new open can retry a released writer lock without restarting Harness.
    // Never remove a lock: only its operator or owner can establish ownership.
    if (failed && failure?.code === 'CONFIG_LOCK_TIMEOUT' && Date.now() >= retryAfter)
      void initialize();
    send(res, {ready, failed, stage, ...(failure ? {failure} : {})}, ready ? 200 : 503);
  }});
  ctx.webServer.register({kind: 'exact', path: '/sm75/configured-models', handler: (_req, res) => {
    const section = ctx.settings.describe().find(row => row.ns === 'llm-pi-ai');
    const models = [];
    for (const [provider, value] of Object.entries(section?.value?.providers || {}))
      for (const model of value.models || []) models.push({
        id: `${provider}/${model.id}`, name: model.id, provider,
        // Deliberately whitelist presentation fields; never return credentials.
        api: model.api || value.api,
        model: {id: model.id, reasoning: model.reasoning, maxTokens: model.maxTokens},
        config: {api: value.api, defaultMaxTokens: value.defaultMaxTokens,
          ...(provider === 'sm75-local' ? {baseURL: config.provider?.baseURL} : {})},
      });
    send(res, {models});
  }});
  const settle = async () => {
    await ctx.root.loader.await();
    if (!config.provider || !config.defaultModel) throw Error('SM75 managed model configuration is missing');
    stage = 'local-provider';
    await ctx.settings.mutate('llm-pi-ai', [{op: 'set', path: ['providers', 'sm75-local'], value: config.provider}]);
    stage = 'default-model';
    await ctx.settings.replace('agent-default-model', config.defaultModel);
    ready = true;
    stage = 'ready';
  };
  const initialize = () => {
    if (pending) return pending;
    failed = false;
    failure = undefined;
    stage = 'waiting-for-plugins';
    pending = settle().catch(error => {
    failure = {stage, kind:error?.name || 'Error',
      code:typeof error?.code==='string' && /^[A-Z_]+$/.test(error.code) ? error.code : 'SETUP_FAILED'};
    // Diagnostics describe the failing stage, never provider values or secrets.
    const message=String(error?.message || '');
    if(message.startsWith('atomic-write: timed out waiting for the writer lock at ')) {
      failure.code='CONFIG_LOCK_TIMEOUT';
      failure.reason='工作台配置写入锁被占用，请检查锁的持有进程';
    }
    if(/^(?:Configuration|No configurable plugin entry|Plugin entry|Config field|Service |Cannot access|Missing dependency)/.test(message))
      failure.reason=message.split('\n')[0].slice(0,200);
    failed = true;
    retryAfter = Date.now() + 3000;
    ctx.logger.error('SM75 local model setup failed at ' + stage + ' (' + failure.code + ')');
    }).finally(() => { pending = undefined; });
    return pending;
  };
  void initialize();
}
