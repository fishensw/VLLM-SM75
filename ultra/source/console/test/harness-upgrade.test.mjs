import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import {pathToFileURL} from 'node:url';
import {harnessPatch, managedHarnessConfig, legacyHarnessConfig, harnessPath} from '../harness-config.mjs';
import {apply as managedPlugin} from '../plugins/sm75-workbench/lib/index.js';
import {installNativePlugins} from '../install-native-plugins.mjs';
import {applyBranding} from '../apply-branding.mjs';
import {applyUnifiedShell} from '../native-shell.mjs';

const profile = {id: 'test', port: 8000, args: ['/models/test', '--served-model-name', 'local-test', '--max-model-len', '8192']};

test('native managed model update retains unrelated providers and edits without a CLI config override', async () => {
  const config = managedHarnessConfig(profile);
  const providers = {other: {models: [{id: 'remote'}]}, 'sm75-local': {apiKey: 'obsolete-test-secret'}};
  let selection = {provider: 'other', model: 'old', reasoningEffort: 'high'};
  const routes = new Map();
  managedPlugin({root: {loader: {await: async () => {}}},
    webServer: {register: route => routes.set(route.path, route.handler)},
    settings: {
      mutate: async (ns, [op]) => {assert.equal(ns, 'llm-pi-ai'); assert.deepEqual(op.path, ['providers', 'sm75-local']); providers['sm75-local'] = op.value;},
      replace: async (ns, value) => {assert.equal(ns, 'agent-default-model'); selection = value;},
      describe: () => [{ns: 'llm-pi-ai', value: {providers}}],
    }, logger: {error: () => assert.fail('unexpected setup failure')},
  }, config);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(providers.other, {models: [{id: 'remote'}]});
  assert.equal(providers['sm75-local'].apiKey, undefined);
  assert.equal(providers['sm75-local'].models[0].contextWindow, 8192);
  assert.deepEqual(selection, {provider: 'sm75-local', model: 'local-test'});
  let response;
  routes.get('/sm75/configured-models')({}, {writeHead(){}, end: body => {response = JSON.parse(body);}});
  assert.equal(response.models.length, 2);
  assert.equal(JSON.stringify(response).includes('apiKey'), false);
  const patch = harnessPatch(config);
  assert.equal(patch.some(row => ['llm-pi-ai', 'agent-default-model'].includes(row.id)), false);
  assert(patch.some(row => row.id === 'ui-brand-official' && row.disabled));
  assert.deepEqual(patch.find(row => row.id === 'workspace-controller'), {
    id: 'workspace-controller', config: {documentsDirectory: '/dsh/workspace'},
  });
  assert.equal(patch.some(row => row.id === 'workspace'), false);
});

test('managed readiness exposes a safe failing stage without leaking provider configuration', async()=>{
 const routes=new Map(),messages=[];
 managedPlugin({root:{loader:{await:async()=>{}}},
   webServer:{register:route=>routes.set(route.path,route.handler)},
   settings:{mutate:async()=>{throw Error('synthetic error containing a private provider value');},
     replace:()=>assert.fail('default model must not be written after provider failure')},
   logger:{error:message=>messages.push(message)},
 },managedHarnessConfig(profile));
 await new Promise(resolve=>setImmediate(resolve));
 let code,body;
 routes.get('/sm75/ready')({}, {writeHead:value=>{code=value;},end:value=>{body=JSON.parse(value);}});
 assert.equal(code,503);assert.equal(body.failed,true);assert.equal(body.ready,false);
 assert.equal(body.failure.stage,'local-provider');
 assert.equal(body.failure.code,'SETUP_FAILED');
 assert.equal(JSON.stringify(body).includes('private provider value'),false);
 assert.equal(messages.join('').includes('private provider value'),false);
});


test('released writer lock retries once per pending initialization and retains safe diagnostics',async t=>{
 t.mock.timers.enable({apis:['Date'],now:1000});
 const routes=new Map();let blocked=true,writes=0,defaultWrites=0;
 managedPlugin({root:{loader:{await:async()=>{}}},
  webServer:{register:route=>routes.set(route.path,route.handler)},
  settings:{mutate:async()=>{writes++;if(blocked)throw Error('atomic-write: timed out waiting for the writer lock at /private/profile/package.json.lock');},
   replace:async()=>{defaultWrites++;}},logger:{error(){}}},managedHarnessConfig(profile));
 const read=()=>{let result;routes.get('/sm75/ready')({}, {writeHead(){},end:body=>{result=JSON.parse(body);}});return result;};
 await new Promise(resolve=>setImmediate(resolve));
 const failure=read();assert.equal(failure.failure.code,'CONFIG_LOCK_TIMEOUT');
 assert.equal(JSON.stringify(failure).includes('/private'),false);
 read();assert.equal(writes,1);assert.equal(defaultWrites,0);
 blocked=false;t.mock.timers.tick(3001);
 read();read();read();
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(read().ready,true);assert.equal(writes,2);assert.equal(defaultWrites,1);
 assert.equal(read().failure,undefined);
});

test('one-time legacy input preserves unrelated settings, does not recreate absent input, and rejects a symlink', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-harness-legacy-'));
  try {
    const file = path.join(root, 'settings.yaml');
    const config = managedHarnessConfig(profile);
    const owner = process.getuid?.() ?? 1000;
    assert.equal(legacyHarnessConfig(file, config, JSON.parse, owner), null);
    fs.writeFileSync(file, JSON.stringify({'ui-theme': {fontSize: 20}, 'llm-pi-ai': {providers: {other: {models: []}}}, 'agent-default-model': {provider: 'other', model: 'old', reasoningEffort: 'high'}}));
    const migrated = legacyHarnessConfig(file, config, JSON.parse, owner);
    assert.deepEqual(migrated['ui-theme'], {fontSize: 20});
    // The upstream legacy import may run after the managed plugin. Its model
    // selection must match the native replacement, including omitted effort.
    assert.deepEqual(migrated['agent-default-model'], config.defaultModel);
    assert.equal(Object.hasOwn(migrated['agent-default-model'], 'reasoningEffort'), false);
    assert.deepEqual(migrated['llm-pi-ai'].providers.other, {models: []});
    assert.equal(migrated['llm-pi-ai'].providers['sm75-local'].models[0].id, 'local-test');
    if (process.platform !== 'win32') {
      fs.unlinkSync(file);
      fs.symlinkSync(path.join(root, 'another-file'), file);
      assert.throws(() => legacyHarnessConfig(file, config, JSON.parse, owner), /regular file/);
    }
  } finally {fs.rmSync(root, {recursive: true, force: true});}
});

test('Harness path prefix preserves nested assets and query without matching similarly named paths', () => {
  assert.equal(harnessPath('/dsh/assets/app.js?v=1'), '/assets/app.js?v=1');
  assert.equal(harnessPath('/dsh/api/ws?session=sample'), '/api/ws?session=sample');
  assert.equal(harnessPath('/dsh/'), '/');
  assert.equal(harnessPath('/harness-ui/'), '/');
  assert.equal(harnessPath('/harness-ui/assets/app.js?v=1'), '/assets/app.js?v=1');
  assert.equal(harnessPath('/dsh-extra/api/ws'), '/dsh-extra/api/ws');
});

test('installer retains native layout behavior and loads the sampling adapter through portable file URLs', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sm75-harness-install-'));
  const write = (name, text) => {const target = path.join(root, name); fs.mkdirSync(path.dirname(target), {recursive: true}); fs.writeFileSync(target, text);};
  try {
    write('@deepseek-ai/dsh-hmr/package.json','{"version":"0.1.7-alpha.2"}');
    write('@deepseek-ai/dsh-hmr/lib/index.js','function watchConfig(){const watcher = watch(target.root, {\n\t\t...watchOptions,\n\t\tdepth: target.depth,\n});}');
    write('@deepseek-ai/dsh/package.json', JSON.stringify({version: '0.1.7-alpha.2', dependencies: {}}));
    for (const name of ['dsh-client-ui-layout', 'dsh-llm-pi-ai'])
      write(`@deepseek-ai/${name}/package.json`, JSON.stringify({version: '0.1.7-alpha.2', type: 'module'}));
    write('@deepseek-ai/dsh-web-frontend/dist/index.html', '<html><head><title>DeepSeek Harness</title><link href="./favicon.svg"></head></html>');
    write('@deepseek-ai/dsh-web-frontend/dist/manifest.webmanifest', '{}');
    const marker = '...options.sessionId === void 0 ? {} : { sessionId: String(options.sessionId) },';
    write('@deepseek-ai/dsh-llm-pi-ai/lib/index.js', `export const buildCall=(options, model)=>({${marker}});`);
    for (const name of ['chat', 'brand-official', 'settings'])
      write(`@deepseek-ai/dsh-client-ui-${name}/lib/client.js`, `// native ${name}`);
    write('@deepseek-ai/dsh-client-ui-sidebar/lib/client.js', 'function SidebarRoot({wide}) {return wide;}');
    write('@deepseek-ai/dsh-client-ui-conversation/lib/client.js', 'function ConversationHeader({value}) {return value;}');
    write('@deepseek-ai/dsh-client-ui-settings-general/lib/client.js', 'function SettingsRoot(props) {return props.value;} function GeneralSection({value}) {return value;}');
    write('@deepseek-ai/dsh-client-ui-theme/lib/client.js', `const zh={"fontSize.title":"字号大小","fontSize.description":"仅影响会话内容的字号"};const en={"fontSize.title":"Font size","fontSize.description":"Only affects conversation content"};
      const fontControls={disabled: fontSize >= 26};
      const FONT_SIZE_FIELD="fontSize";
      class ThemeRuntime {
        constructor(host) {this.host=host;this.ctx={logger:{warn(){}}};this.fontSize=14;this.preference="dark";this.published=[];}
        setFontSize(px) {if(this.fontSize===px)return;this.fontSize=px;this.host.set(FONT_SIZE_FIELD, px);this.publish();}
        adopt() {const section = this.host.getSnapshot().value;if (section === void 0) return;
          if(this.preference === section.preference && this.fontSize === section.fontSize)return;
          this.preference=section.preference;this.fontSize = section.fontSize;this.publish();}
        publish(){this.published.push(this.fontSize);}
      }
      globalThis.ThemeRuntime=ThemeRuntime;
    `);
    write('@deepseek-ai/dsh-client-ui-open-in-app/lib/client.js', 'const endpoint="/open-in-app/apps";');
    const nativeLayout = 'function AppFrame({value}) {return value;} const productTitle = "DeepSeek Harness"; function updateTitle(title) { document.title = title === undefined ? productTitle : `${title} — ${productTitle}`; }';
    write('@deepseek-ai/dsh-client-ui-layout/lib/client.js', nativeLayout + '\nfunction owner() {\nconst layout = new LayoutController(instance.actions, id => true);\nreturn layout;}');
    write('@deepseek-ai/dsh-client-ui-token-usage/package.json', '{}');
    write('sm75-workbench/lib/client.js', '// obsolete inverse embedding');
    const manifestBefore = JSON.parse(fs.readFileSync(path.join(root, '@deepseek-ai/dsh/package.json'), 'utf8'));
    manifestBefore.dependencies['@deepseek-ai/dsh-client-ui-token-usage'] = '0.1.6';
    write('@deepseek-ai/dsh/package.json', JSON.stringify(manifestBefore));
    installNativePlugins(root);
    assert.equal(fs.existsSync(path.join(root, '@deepseek-ai/dsh-client-ui-token-usage')), false);
    assert.equal(fs.existsSync(path.join(root, 'sm75-workbench/lib/client.js')), false);
    assert.equal(JSON.parse(fs.readFileSync(path.join(root, 'sm75-workbench/package.json'), 'utf8')).exports['./client'], undefined);
    assert.equal(JSON.parse(fs.readFileSync(path.join(root, '@deepseek-ai/dsh/package.json'), 'utf8')).dependencies['@deepseek-ai/dsh-client-ui-token-usage'], undefined);
    const once = fs.readFileSync(path.join(root, '@deepseek-ai/dsh-llm-pi-ai/lib/index.js'), 'utf8');
    applyBranding(root);
    assert.equal(fs.readFileSync(path.join(root, '@deepseek-ai/dsh-llm-pi-ai/lib/index.js'), 'utf8'), once);
    const llmFile = path.join(root, '@deepseek-ai/dsh-llm-pi-ai/lib/index.js');
    // Load the generated ESM rather than only parsing it. On Windows the real
    // console source may be UNC; on Linux it is an absolute POSIX path.
    assert.equal(typeof (await import(pathToFileURL(llmFile).href)).buildCall, 'function');
    const consoleRoot = path.join(root, 'console # 空间');
    fs.mkdirSync(path.join(consoleRoot, 'branding'), {recursive: true});
    fs.writeFileSync(path.join(consoleRoot, 'branding/favicon.svg'), '<svg/>');
    fs.writeFileSync(path.join(consoleRoot, 'sampling.mjs'), 'export function dshSampling(provider, model) { return {probe: provider + "/" + model.id}; }');
    // Reinstall an earlier candidate with its raw path import. Special URL
    // characters expose missing escaping, on both Windows and Linux runners.
    const rawImport = `import {dshSampling} from ${JSON.stringify(path.join(consoleRoot, 'sampling.mjs'))};\n`;
    fs.writeFileSync(llmFile, rawImport + once.slice(once.indexOf('\n') + 1));
    applyBranding(root, consoleRoot);
    const repaired = fs.readFileSync(llmFile, 'utf8');
    assert(repaired.startsWith(`import {dshSampling} from ${JSON.stringify(pathToFileURL(path.join(consoleRoot, 'sampling.mjs')).href)};\n`));
    const loaded = await import(pathToFileURL(llmFile).href + '?reinstalled');
    assert.deepEqual(loaded.buildCall({provider: 'synthetic', sessionId: 42}, {id: 'model'}), {sessionId: '42', probe: 'synthetic/model'});
    applyBranding(root, consoleRoot);
    assert.equal(fs.readFileSync(llmFile, 'utf8'), repaired);
    for (const name of ['chat', 'brand-official', 'settings'])
      assert.equal(fs.readFileSync(path.join(root, `@deepseek-ai/dsh-client-ui-${name}/lib/client.js`), 'utf8'), `// native ${name}`);
    const layoutFile = path.join(root, '@deepseek-ai/dsh-client-ui-layout/lib/client.js');
    const patchedLayout = fs.readFileSync(layoutFile, 'utf8');
    assert.match(patchedLayout, /sm75-unified-shell:AppFrame/);
    assert.match(fs.readFileSync(path.join(root, '@deepseek-ai/dsh-client-ui-sidebar/lib/client.js'),'utf8'), /sm75-unified-shell:SidebarRoot/);
    assert.match(fs.readFileSync(path.join(root, '@deepseek-ai/dsh-client-ui-open-in-app/lib/client.js'),'utf8'), /harness-ui\/open-in-app\/apps/);
    const shellFiles=['dsh-client-ui-layout','dsh-client-ui-sidebar','dsh-client-ui-conversation','dsh-client-ui-settings-general','dsh-client-ui-theme']
      .map(name=>path.join(root,'@deepseek-ai',name,'lib/client.js'));
    assert.match(fs.readFileSync(path.join(root,'@deepseek-ai/dsh-client-ui-theme/lib/client.js'),'utf8'), /全局字号/);
    const themeSandbox={fontSize:17};
    vm.runInNewContext(fs.readFileSync(path.join(root,'@deepseek-ai/dsh-client-ui-theme/lib/client.js'),'utf8'),themeSandbox);
    assert.match(fs.readFileSync(path.join(root,'@deepseek-ai/dsh-client-ui-theme/lib/client.js'),'utf8'), /disabled: fontSize >= 17/);
    let accepted=14;
    const writes=[];
    const runtime=new themeSandbox.ThemeRuntime({getSnapshot:()=>({value:{preference:'dark',fontSize:accepted}}),set:(_key,value)=>new Promise((resolve,reject)=>writes.push({value,resolve,reject}))});
    for(const size of [15,16,17])runtime.setFontSize(size);
    accepted=15;runtime.adopt();writes[0].resolve(true);await Promise.resolve();
    assert.equal(runtime.fontSize,17);
    accepted=16;runtime.adopt();writes[1].resolve(true);await Promise.resolve();
    assert.equal(runtime.fontSize,17);
    accepted=17;runtime.adopt();writes[2].resolve(true);await runtime.sm75FontWrite;
    assert.equal(runtime.fontSize,17);assert.equal(runtime.sm75FontPending,false);
    for(const size of [16,15,14,13,12])runtime.setFontSize(size);
    for(const item of writes.slice(3)){accepted=item.value;runtime.adopt();item.resolve(true);await Promise.resolve();assert.equal(runtime.fontSize,12);}
    await runtime.sm75FontWrite;assert.equal(runtime.fontSize,12);
    runtime.setFontSize(13);accepted=12;runtime.adopt();writes.at(-1).resolve(false);await runtime.sm75FontWrite;
    assert.equal(runtime.fontSize,12);assert.equal(runtime.sm75FontPending,false);
    runtime.setFontSize(13);writes.at(-1).reject(Error('synthetic transport failure'));assert.equal(await runtime.sm75FontWrite,false);
    assert.equal(runtime.fontSize,12);assert.equal(runtime.sm75FontPending,false);

    const shellBefore=shellFiles.map(file=>fs.readFileSync(file,'utf8'));
    applyUnifiedShell(root);
    assert.deepEqual(shellFiles.map(file=>fs.readFileSync(file,'utf8')),shellBefore);
    assert.equal(fs.readFileSync(layoutFile,'utf8'),patchedLayout);
    for(const [name,fn,method] of [
      ['dsh-client-ui-conversation','ConversationHeader','header'],
      ['dsh-client-ui-settings-general','SettingsRoot','settings'],
      ['dsh-client-ui-settings-general','GeneralSection','generalSettings'],
    ]) {
      const passed=[];
      const sandbox={__SM75_NATIVE_SHELL__:{[method]:(props,native)=>{passed.push(props);return native(props);}}};
      vm.runInNewContext(fs.readFileSync(path.join(root,'@deepseek-ai',name,'lib/client.js'),'utf8'),sandbox);
      const props={value:123};
      assert.equal(sandbox[fn](props),123);assert.equal(passed[0],props);
    }
    const page = {document: {title: ''}, __SM75_NATIVE_SHELL__: {frame: props => props.value}};
    vm.runInNewContext(patchedLayout, page);
    assert.equal(page.AppFrame({value: 42}), 42);
    page.updateTitle();
    assert.equal(page.document.title, '工作台');
    page.updateTitle('Synthetic session');
    assert.equal(page.document.title, 'Synthetic session — 工作台');
    assert.equal(JSON.parse(fs.readFileSync(path.join(root, 'sm75-brand/package.json'))).name, 'sm75-brand');
    fs.writeFileSync(layoutFile, nativeLayout + '\n' + nativeLayout);
    assert.throws(() => applyBranding(root), /title anchor changed or is ambiguous/);
    fs.writeFileSync(layoutFile, nativeLayout);
    write('@deepseek-ai/dsh-client-ui-layout/package.json', '{"version":"0.1.6"}');
    assert.throws(() => applyBranding(root), /dsh-client-ui-layout adaptation requires/);
    write('@deepseek-ai/dsh-client-ui-layout/package.json', '{"version":"0.1.7-alpha.2"}');
    write('@deepseek-ai/dsh/package.json', '{"version":"0.1.5-rc.1"}');
    assert.throws(() => applyBranding(root), /requires 0.1.7-alpha.2/);
  } finally {fs.rmSync(root, {recursive: true, force: true});}
});


function browserPluginHarness(slotRows = [], configurePage = () => {}) {
  const source=fs.readFileSync(new URL('../plugins/sm75-brand/lib/client.js',import.meta.url),'utf8');
  let plugin,themeChange,currentTheme='dark';
  const components=new Map(),cleanups=[],events=[],panels=[],themes=[],subscriptions=new Set(),listeners=new Map(),styles=new Map();
  const React={
    Fragment:Symbol('fragment'),
    createElement:(type,props,...children)=>({type,props:{...props,children:children.length===1?children[0]:children}}),
    useEffect:effect=>{cleanups.push(effect());},
    useRef:()=>({current:{getBoundingClientRect:()=>({width:1200})}}),
    useLayoutEffect:effect=>{cleanups.push(effect());},
    useState:initial=>[typeof initial==='function'?initial():initial,()=>{}],
    useSyncExternalStore:(subscribe,snapshot)=>{subscriptions.add(subscribe(()=>{}));return snapshot();},
  };
  const page={document:{dispatchEvent:event=>{events.push(event);listeners.get(event.type)?.(event);},
    addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name),
    querySelector:()=>({dataset:{page:"harness"}}),
    getElementById:id=>({id}),documentElement:{style:{setProperty:(name,value)=>styles.set(name,value)}}},
    localStorage:{getItem:()=> 'dark'},CustomEvent:class {constructor(type,{detail}={}){this.type=type;this.detail=detail;}},
    ResizeObserver:class{observe(){} disconnect(){}},
    window:{__ModuleLoader__:{load:item=>{plugin=item.factory(name=>name==='react'?React:{createPortal:(node,target)=>({portal:target.id,node})});}}}};
  configurePage(page);
  vm.runInNewContext(source,page);
  const context={
    slots:{inject:(_key,fn)=>fn(),register:({name,id},component)=>components.set(id||name,component),entries:name=>name==='settings.general.item'?slotRows:[],subscribe:()=>()=>{},getVersion:()=>0},
    effect:effect=>cleanups.push(effect()),
    layout:{sm75ActivePanel:()=>panels.length?panels.at(-1):null,selectPanel:id=>panels.push(id)},
    theme:{getTheme:()=>({active:{colorScheme:currentTheme},fontSize:18}),setTheme:mode=>{currentTheme=mode;themes.push(mode);}},
    on:(_name,fn)=>{themeChange=fn;return()=>{themeChange=null;};},
  };
  plugin.apply(context);
  return {page,context,components,events,panels,themes,styles,listeners,React,themeChange:snapshot=>themeChange(snapshot),
    cleanup:()=>{for(const cleanup of [...cleanups.reverse(),...subscriptions])if(typeof cleanup==='function')cleanup();}};
}

test('native API is ready without a settings launcher and routes single-level panels and horizontal settings',()=>{
  const app=browserPluginHarness(),api=app.page.__SM75_WORKBENCH__;
  assert.equal(api.ready,false);
  const shell=app.page.__SM75_NATIVE_SHELL__;
  const frame=shell.frame({useStore:select=>select({layoutInfo:{viewportWidth:1200}}),
    usePanelInfo:select=>select({activePanelId:null}),actions:{setViewportWidth(){}},renderSlot:()=>null});
  frame.type(frame.props);
  assert.equal(api.ready,false);
  const settings=shell.settings({useSections:select=>select([]),useOnboardingSteps:select=>select([]),
    useSessions:select=>select({phase:'ready',byId:{}}),useConnectionState:select=>select('connected'),
    reconnect:()=>{},renderSlot:()=>null});
  settings.type(settings.props);
  assert.equal(api.ready,true);
  assert.equal(app.components.has('settings.launcher'),false);
  assert(app.events.some(e=>e.type==='sm75-native-ready'));
  assert.equal(app.styles.get('--ui-fs'),String(18/14));
  // A redundant null selection previously aborted native workspace restore.
  api.selectPanel(null);assert.deepEqual(app.panels,[]);
  api.selectPanel('sm75-usage');assert.deepEqual(app.panels,['sm75-usage']);
  api.selectPanel('sm75-usage');assert.deepEqual(app.panels,['sm75-usage']);
  api.selectPanel(null);assert.deepEqual(app.panels,['sm75-usage',null]);
  api.selectPanel(null);assert.deepEqual(app.panels,['sm75-usage',null]);
  for(const page of ['appearance','assistant','connections',null])assert.doesNotThrow(()=>api.selectSettings(page));
  assert.throws(()=>api.selectSettings('nested-page'),/Unknown workbench settings page/);
  api.openSettings();
  assert.equal(app.events.find(e=>e.type==='sm75-native-settings').detail.id,'appearance');
  api.setTheme('invalid');api.setTheme('light');
  assert.deepEqual(app.themes,['light']);
  app.themeChange({active:{colorScheme:'light'},fontSize:18});
  assert(app.events.some(e=>e.type==='sm75-set-theme'));
  app.cleanup();assert.equal(app.page.__SM75_WORKBENCH__,undefined);assert.equal(api.ready,false);
});

test('sidebar panel switching supports an already-loaded layout without the new state reader',()=>{
  const app=browserPluginHarness(),api=app.page.__SM75_WORKBENCH__;
  delete app.context.layout.sm75ActivePanel;
  api.selectPanel(null);assert.deepEqual(app.panels,[]);
  api.selectPanel('sm75-usage');api.selectPanel('sm75-usage');
  api.selectPanel('plugins');api.selectPanel(null);api.selectPanel(null);
  assert.deepEqual(app.panels,['sm75-usage','plugins',null]);
  // A failed native action must not advance the compatibility state.
  const select=app.context.layout.selectPanel;
  app.context.layout.selectPanel=()=>{throw Error('Native action failed');};
  assert.throws(()=>api.selectPanel('plugins'),/Native action failed/);
  app.context.layout.selectPanel=select;
  api.selectPanel('plugins');
  assert.deepEqual(app.panels,['sm75-usage','plugins',null,'plugins']);
  app.cleanup();
});

test('native conversation header moves its original children to the public toolbar without a second header element',()=>{
  const app=browserPluginHarness(),shell=app.page.__SM75_NATIVE_SHELL__;
  let received;
  const children=[{title:'Original title'},{tabs:'Original tabs'}];
  const props={usePanelInfo:selector=>selector({activePanelId:null}),value:42};
  const entry=shell.header(props,p=>{received=p;return app.React.createElement('header',{className:'old-two-row-shell'},...children);});
  const output=entry.type(entry.props);
  assert.equal(received,props);
  assert.equal(output.portal,'workspaceToolbar');
  assert.equal(output.node.type,'div');
  assert.equal(output.node.props.className,'sm75-conversation-header');
  assert.equal(output.node.props.hidden,false);
  assert.equal(output.node.props.children[0],children[0]);
  assert.equal(output.node.props.children[1],children[1]);
  app.cleanup();
});

test('native settings render original sections into the chosen content tab and retain onboarding routing',()=>{
  const app=browserPluginHarness(),api=app.page.__SM75_WORKBENCH__,calls=[];
  const rows=['general','models','agent-presets','plugins'].map(id=>({id,label:id}));
  const props={
    useSections:selector=>selector(rows),useOnboardingSteps:selector=>selector([]),
    useSessions:selector=>selector({phase:'ready',byId:{}}),
    useConnectionState:selector=>selector('connected'),reconnect:()=>{},
    renderSlot:(name,owner,options)=>{calls.push({name,owner,options});return {name,owner,options};},
  };
  const render=()=>{const entry=app.page.__SM75_NATIVE_SHELL__.settings(props);return entry.type(entry.props);};
  api.selectSettings('connections');
  const output=render();
  assert.equal(output.props.children[0].portal,'nativeSettingsSurface');
  assert.deepEqual(calls.filter(c=>c.name==='settings.section').map(c=>c.options.only),['models']);
  calls.length=0;api.selectSettings('assistant');render();
  assert.deepEqual(calls.filter(c=>c.name==='settings.section').map(c=>c.options.only),['general','agent-presets','plugins']);
  calls.length=0;api.selectSettings(null);render();
  assert.equal(calls.some(c=>c.name==='settings.section'),false);
  calls.length=0;
  props.useOnboardingSteps=selector=>selector([{id:'native-onboarding'}]);
  render();
  const onboarding=calls.find(c=>c.name==='settings.onboarding');
  assert.equal(onboarding.options.only,'native-onboarding');
  onboarding.owner.openSection('models');
  assert.equal(app.events.find(e=>e.type==='sm75-native-settings').detail.id,'connections');
  app.cleanup();
});

test('leaving settings preserves visited native section keys and completed onboarding while hiding their content',()=>{
  const app=browserPluginHarness(),api=app.page.__SM75_WORKBENCH__,calls=[],states=[];
  let cursor=0,pending=[];
  app.React.useState=initial=>{
    const at=cursor++;
    if(!(at in states))states[at]=typeof initial==='function'?initial():initial;
    return [states[at],value=>{states[at]=typeof value==='function'?value(states[at]):value;}];
  };
  app.React.useEffect=effect=>pending.push(effect);
  const props={
    useSections:selector=>selector([{id:'general'},{id:'models'}]),
    useOnboardingSteps:selector=>selector([{id:'welcome'}]),
    useSessions:selector=>selector({phase:'ready',byId:{}}),
    useConnectionState:selector=>selector('connected'),reconnect:()=>{},
    renderSlot:(name,owner,options)=>{calls.push({name,owner,options});return {name,owner,options};},
  };
  const render=()=>{
    cursor=0;pending=[];calls.length=0;
    const entry=app.page.__SM75_NATIVE_SHELL__.settings(props);
    const output=entry.type(entry.props);
    for(const effect of pending)effect();
    return output;
  };
  api.selectSettings('connections');
  const first=render().props.children[0].node;
  const firstSection=first.props.children[1][0];
  assert.equal(firstSection.props.key,'models');assert.equal(firstSection.props.hidden,false);
  calls.find(call=>call.name==='settings.onboarding').owner.complete();
  api.selectSettings(null);
  const hidden=render().props.children[0].node.props.children[1][0];
  assert.equal(hidden.props.key,'models');assert.equal(hidden.props.hidden,true);
  assert(calls.some(call=>call.name==='settings.section'&&call.options.only==='models'));
  assert.equal(calls.some(call=>call.name==='settings.onboarding'),false);
  api.selectSettings('appearance');
  const switched=render().props.children[0].node.props.children[1];
  assert.deepEqual(switched.map(row=>[row.props.key,row.props.hidden]),[['general',false],['models',true]]);
  api.selectSettings('connections');
  const restored=render().props.children[0].node.props.children[1].find(row=>row.props.key==='models');
  assert.equal(restored.props.hidden,false);
  app.cleanup();
});

test('conversation toolbar hides on a native utility panel without discarding original header subscriptions',()=>{
  const app=browserPluginHarness(),props={usePanelInfo:select=>select({activePanelId:'sm75-usage'})};
  let renders=0;
  const entry=app.page.__SM75_NATIVE_SHELL__.header(props,()=>{
    renders++;
    return app.React.createElement('header',null,app.React.createElement('button',null,'Native tool'));
  });
  const output=entry.type(entry.props);
  assert.equal(renders,1);
  assert.equal(output.portal,'workspaceToolbar');
  assert.equal(output.node.props.hidden,true);
  assert.equal(output.node.props.children.type,'button');
  app.cleanup();
});

test('background native initialization cannot show onboarding outside the workbench route',()=>{
  const app=browserPluginHarness(),calls=[],states=[];
  let cursor=0,pending=[];
  app.page.document.querySelector=()=>({dataset:{page:'profiles'}});
  app.React.useState=initial=>{
    const at=cursor++;
    if(!(at in states))states[at]=typeof initial==='function'?initial():initial;
    return [states[at],value=>{states[at]=typeof value==='function'?value(states[at]):value;}];
  };
  app.React.useEffect=effect=>pending.push(effect);
  const props={
    useSections:select=>select([]),useOnboardingSteps:select=>select([{id:'welcome'}]),
    useSessions:select=>select({phase:'ready',byId:{}}),
    useConnectionState:select=>select('connected'),reconnect:()=>{},
    renderSlot:(name,owner,options)=>{calls.push({name,owner,options});return {name};},
  };
  const render=()=>{
    cursor=0;pending=[];calls.length=0;
    const entry=app.page.__SM75_NATIVE_SHELL__.settings(props);
    const output=entry.type(entry.props);
    for(const effect of pending)effect();
    return output;
  };
  render();assert.equal(calls.some(c=>c.name==='settings.onboarding'),false);
  app.listeners.get('sm75-route-change')({detail:{page:'harness'}});
  render();assert.equal(calls.some(c=>c.name==='settings.onboarding'),true);
  app.listeners.get('sm75-route-change')({detail:{page:'settings'}});
  render();assert.equal(calls.some(c=>c.name==='settings.onboarding'),false);
  app.cleanup();
});

test('appearance and assistant divide original General rows without replacing permission actions or stores',()=>{
  const ids=['permission','language','appearance','font-size','transcript-view','performance-usage','developer-tools','composer-enter','link-opening','current-version'];
  const rows=ids.map((id,order)=>({options:{id,order}}));
  const app=browserPluginHarness(rows),calls=[];
  const props={renderSlot:(name,owner,options)=>{const value={name,owner,options};calls.push(value);return value;}};
  const render=()=>{const entry=app.page.__SM75_NATIVE_SHELL__.generalSettings(props);return entry.type(entry.props);};
  app.page.__SM75_WORKBENCH__.selectSettings('appearance');
  const appearance=render().props.children;
  assert.deepEqual(appearance.filter(row=>!row.props.hidden).map(row=>row.props['data-native-setting-item']),
    ['language','appearance','font-size','transcript-view','performance-usage','current-version']);
  assert.deepEqual(calls.map(call=>call.options.only),ids);
  calls.length=0;app.page.__SM75_WORKBENCH__.selectSettings('assistant');
  const assistant=render().props.children;
  assert.deepEqual(assistant.filter(row=>!row.props.hidden).map(row=>row.props['data-native-setting-item']),
    ['permission','developer-tools','composer-enter','link-opening']);
  assert.deepEqual(assistant.map(row=>row.props.key),appearance.map(row=>row.props.key));
  assert(calls.every(call=>call.name==='settings.general.item'));
  app.cleanup();
});


test('native font scaling covers lazy styles and trajectory tokens exactly once without touching foreign sheets or content tokens',()=>{
  const declaration=entries=>{
    const values=new Map(Object.entries(entries).map(([key,value])=>[key,{value,priority:key==='font-size'?'important':''}]));
    return {
      [Symbol.iterator]:function*(){yield* values.keys();},
      getPropertyValue:key=>values.get(key)?.value||'',
      getPropertyPriority:key=>values.get(key)?.priority||'',
      setProperty:(key,value,priority='')=>values.set(key,{value,priority}),
    };
  };
  const fixed=declaration({'font-size':'13px','padding':'12px','line-height':'20px'});
  const legacy=declaration({'font-size':'calc(14px * var(--dsh-font-scale,1))','line-height':'calc(22px * var(--dsh-font-scale,1))'});
  const content=declaration({'font-size':'var(--dsh-content-font-size, 14px)'});
  const tokens=declaration({
    '--dsw-font-xs-13':'13px/20px var(--dsw-font-family)',
    '--dsw-font-xs-13-font-size':'13px',
    '--dsw-font-xs-13-line-height':'20px',
    '--dsh-content-font-size-secondary':'13px',
    '--dsw-font-markdown-base':'var(--dsh-content-font-size, 14px) / calc(24px + var(--dsh-content-font-delta)) var(--dsw-font-family)',
  });
  const font=declaration({font:'600 12px/16px var(--dsw-font-family)'});
  const foreign=declaration({'font-size':'15px'}),link=declaration({'font-size':'14px'});
  const tag=plugin=>({tagName:'STYLE',dataset:{plugin,pluginCss:plugin+'/example.css'}});
  const sheets=[
    {ownerNode:tag('@deepseek-ai/dsh-client-ui-sidebar'),cssRules:[{style:fixed},{style:legacy},{style:content},{cssRules:[{style:tokens},{style:font}]}]},
    {ownerNode:tag('unrelated-plugin'),cssRules:[{style:foreign}]},
    {ownerNode:{tagName:'LINK',dataset:{sm75Native:'true'},href:'https://example.test/harness-ui/style.css'},cssRules:[{style:link}]},
  ];
  let refresh,disconnected=false,load;
  const app=browserPluginHarness([],page=>{
    page.document.styleSheets=sheets;
    page.document.head={addEventListener:(_name,fn)=>{load=fn;},removeEventListener:()=>{load=undefined;}};
    page.location={href:'https://example.test/',origin:'https://example.test'};page.URL=URL;
    page.MutationObserver=class{constructor(fn){refresh=fn;}observe(){}disconnect(){disconnected=true;}};
  });
  const expected='calc(13px * var(--ui-fs, 1))';
  assert.equal(fixed.getPropertyValue('font-size'),expected);
  assert.equal(fixed.getPropertyPriority('font-size'),'important');
  assert.equal(fixed.getPropertyValue('padding'),'12px');
  assert.equal(fixed.getPropertyValue('line-height'),'calc(20px * var(--ui-fs, 1))');
  assert.equal(legacy.getPropertyValue('font-size'),'calc(14px * var(--ui-fs,1))');
  assert.equal(legacy.getPropertyValue('line-height'),'calc(22px * var(--ui-fs,1))');
  assert.equal(content.getPropertyValue('font-size'),'var(--dsh-content-font-size, 14px)');
  assert.equal(tokens.getPropertyValue('--dsw-font-xs-13'),expected+'/calc(20px * var(--ui-fs, 1)) var(--dsw-font-family)');
  assert.equal(tokens.getPropertyValue('--dsw-font-xs-13-font-size'),expected);
  assert.equal(tokens.getPropertyValue('--dsw-font-xs-13-line-height'),'calc(20px * var(--ui-fs, 1))');
  assert.equal(tokens.getPropertyValue('--dsh-content-font-size-secondary'),expected);
  assert(tokens.getPropertyValue('--dsw-font-markdown-base').startsWith('var(--dsh-content-font-size,'));
  assert.equal(font.getPropertyValue('font'),'600 calc(12px * var(--ui-fs, 1))/calc(16px * var(--ui-fs, 1)) var(--dsw-font-family)');
  assert.equal(foreign.getPropertyValue('font-size'),'15px');
  assert.equal(link.getPropertyValue('font-size'),'calc(14px * var(--ui-fs, 1))');
  refresh();assert.equal(fixed.getPropertyValue('font-size'),expected);
  const lazy=declaration({'font-size':'17px'});
  sheets.push({ownerNode:tag('dsh-watcher'),cssRules:[{style:lazy}]});
  load();assert.equal(lazy.getPropertyValue('font-size'),'calc(17px * var(--ui-fs, 1))');
  for(const size of [12,14,17]){
    app.themeChange({active:{colorScheme:'dark'},fontSize:size});
    assert.equal(app.styles.get('--ui-fs'),String(size/14));
  }
  app.cleanup();
  assert.equal(disconnected,true);assert.equal(load,undefined);
  assert.equal(fixed.getPropertyValue('font-size'),'13px');
  assert.equal(font.getPropertyValue('font'),'600 12px/16px var(--dsw-font-family)');
  assert.equal(tokens.getPropertyValue('--dsh-content-font-size-secondary'),'13px');
  assert.equal(lazy.getPropertyValue('font-size'),'17px');
});
