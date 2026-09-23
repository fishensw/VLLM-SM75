import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../plugins/dsh-watcher/lib/client.js', import.meta.url), 'utf8');
// These are the dsh-v0.1.7-alpha.2 public ui-primitives exports the shipped
// Watcher uses, not permissive mocks accepting every requested property.
const primitiveNames = ['StateDot', 'writeClipboard', 'IconCheckOutlineRegular', 'IconCopyOutlineRegular',
  'TerminalBlock', 'ReadBlock', 'DiffBlock', 'JsonTree', 'MarkdownText',
  'IconChevronRightOutlineRegular', 'useAnchoredPosition', 'Pill', 'IconRefreshOutlineRegular'];
function fixture() {
  let client, header, usageMain, stateIndex = 0, projectionKeys = [];
  const remote = {session: {list: async () => ({items: []})}};
  const state = [];
  const element = (type, props = {}) => {
    assert(['string', 'function'].includes(typeof type), 'JSX component must resolve to a real native export');
    return {type, props};
  };
  const react = {
    createElement: element,
    useMemo: factory => factory(),
    useRef: value => ({current: value}),
    useEffect() {}, useLayoutEffect() {},
    useState(initial) {
      const index = stateIndex++;
      if (!(index in state)) state[index] = typeof initial === 'function' ? initial() : initial;
      return [state[index], value => {state[index] = typeof value === 'function' ? value(state[index]) : value;}];
    },
  };
  const primitives = Object.fromEntries(primitiveNames.map(name => [name, name === 'useAnchoredPosition' ? () => null : name]));
  const dependencies = {
    react, 'react-dom': {createPortal: node => node},
    'react/jsx-runtime': {jsx: element, jsxs: element},
    '@deepseek-ai/dsh-client-ui-primitives': new Proxy(primitives, {
      get: (values, key) => {assert(key in values, `Unknown native primitive: ${String(key)}`); return values[key];},
    }),
  };
  const styles = [];
  const document = {querySelector: () => null, body: {},
    createElement: () => ({dataset: {}}), head: {appendChild: tag => styles.push(tag)},
  };
  vm.runInNewContext(source, {window: {__ModuleLoader__: {load: entry => {
    assert.equal(entry.id, 'dsh-watcher');
    client = entry.factory(name => {assert(name in dependencies, `Unexpected native dependency: ${name}`); return dependencies[name];});
  }}}, document});
  const chat = {
    legacy: {nodes: [], turnTimings: new Map(), runningCalls: []},
    timeline: {turnOrder: [], turns: new Map()},
    locations: {getStep: () => []}, nodes: {get: () => undefined},
  };
  const views = new Map([['chat', chat]]);
  const lifecycle = {blank: false, running: false, hasMore: false, loadingOlder: false};
  const statuses = new Map();
  const sessionId = 'synthetic-session';
  const session = {getSnapshot: () => lifecycle, loadOlder: async () => {}};
  client.apply({
    slots: {inject: (_name, mount) => mount(), register: (options, Component) => {
      if (options.id === 'dsh-watcher') header = {options, Component};
    }},
    inject: (_services, mount) => mount({remote, slots: {inject: (_name, register) => register(), register: (options, Component) => {if (options.key === 'sm75-usage') usageMain = Component;}}}),
    sessions: {binding: id => {assert.equal(id, sessionId); return {session};}},
    uiConversation: {events: {register() {}}, binding: id => {
      assert.equal(id, sessionId); return {snapshot: {getSnapshot: () => ({views})}};
    }},
  });
  const hooks = {
    useConversation: select => select({views}),
    useSession: select => select(lifecycle),
    useSessionStatus: select => select(statuses),
    useProjection: key => {projectionKeys.push(key); return key === 'sessionStats' ? {turns: 2, steps: 4} : undefined;},
  };
  const injected = header.options.inject(sessionId);
  return {
    client, statuses, sessionId, lifecycle, injected, styles, remote,
    usageMain: () => usageMain,
    render() {
      stateIndex = 0; projectionKeys = [];
      const ready = header.Component({...hooks, ...injected, sessionId});
      const tree = ready.type(ready.props); // Runs the shipped ReadyWatcher, including all native hooks.
      assert.deepEqual(projectionKeys, ['sessionStats', 'watcherInsights']);
      return tree;
    },
  };
}
function nodes(tree) {
  if (!tree || typeof tree !== 'object') return [];
  const children = tree.props?.children;
  return [tree, ...(Array.isArray(children) ? children : [children]).flatMap(nodes)];
}

test('shipped Watcher renders and opens using the complete 0.1.7 public hook contract', () => {
  const f = fixture();
  assert.equal(f.client.clientContract.harnessVersion, '0.1.7-alpha.2');
  assert.deepEqual(Array.from(f.client.clientContract.standardHooks), ['useConversation', 'useSession', 'useSessionStatus', 'useProjection']);
  assert.deepEqual(Array.from(f.client.clientContract.primitives), primitiveNames);
  // Check every compiled primitive reference, including inspector branches that
  // are not mounted in the empty-session render below.
  const referenced = [...new Set(Array.from(source.matchAll(/_deepseek_ai_dsh_client_ui_primitives\.([A-Za-z0-9_]+)/g), match => match[1]))];
  assert.deepEqual(referenced.sort(), [...primitiveNames].sort());
  const button = nodes(f.render()).find(node => node.type === 'button' && node.props.title === 'Watcher');
  assert(button);
  assert.equal(button.props['aria-expanded'], false);
  assert.match(button.props['aria-label'], /待命/);
  button.props.onClick();
  const rendered = nodes(f.render());
  assert(rendered.some(node => node.props.role === 'dialog' && node.props['aria-label'] === 'Watcher 工作图'));
  assert.equal(rendered.find(node => node.type === 'button' && node.props.title === 'Watcher').props['aria-expanded'], true);
});

test('Watcher reads the selected session pending request from SessionStatus, and clears it when settled', () => {
  const f = fixture();
  const label = () => nodes(f.render()).find(node => node.type === 'button' && node.props.title === 'Watcher').props['aria-label'];
  f.statuses.set('another-session', {pendingInteraction: {key: 'other', kind: 'approval', sessionId: 'another-session'}});
  assert.match(label(), /待命/);
  f.statuses.set(f.sessionId, {pendingInteraction: {key: 'request-1', kind: 'approval', sessionId: f.sessionId}});
  assert.match(label(), /等待确认/);
  f.statuses.set(f.sessionId, {pendingInteraction: undefined});
  assert.doesNotMatch(label(), /等待确认/);
});

test('Watcher history loader still uses native session paging and conversation snapshots', async () => {
  const f = fixture();
  const result = await f.injected.loadAllHistory(new AbortController().signal);
  assert.equal(result.kind, 'complete');
  assert.equal(result.pages, 0);
});


test('Watcher registers opaque theme surfaces for the opened work picture', () => {
  const f = fixture();
  nodes(f.render()).find(node => node.type === 'button' && node.props.title === 'Watcher').props.onClick();
  const rendered = nodes(f.render());
  const css = f.styles.find(tag => tag.dataset.pluginCss === 'dsh-watcher/Watcher.module.css')?.textContent;
  assert(css, 'the shipped client must inject its panel styles');
  for (const surface of ['QCz6mq_menu', 'QCz6mq_workPicture']) {
    assert(rendered.some(node => node.props.className?.split(/\s+/).includes(surface)), `${surface} must be mounted`);
    const rule = new RegExp(`\\.${surface}\\{([^{}]+)\\}`).exec(css)?.[1];
    assert(rule, `${surface} must have a base style rule`);
    // Native specific-menu is translucent in both themes. Use the public overlay
    // surface with theme/system fallbacks; the real browser suite checks opacity.
    assert.match(rule, /background:var\(--dsw-alias-bg-overlay,var\(--dsw-alias-bg-layer-2,Canvas\)\)/);
    assert.doesNotMatch(rule, /background:var\(--dsw-specific-menu\)/);
  }
});

test('Watcher usage navigation mounts its native Insights view', () => {
  const f = fixture();
  const view = f.usageMain()();
  assert.equal(view.type.name, 'InsightsSettings');
  assert.equal(view.props.remote, f.remote);
});
