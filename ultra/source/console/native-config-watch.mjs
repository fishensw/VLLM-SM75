import fs from 'node:fs';
import path from 'node:path';

// Harness 0.1.7-alpha.2 watches a config file's parent before filtering change
// events. A protected unrelated sibling can then disable HMR and config edits.
export function applyNativeConfigWatch(base) {
  const root=path.join(base,'@deepseek-ai/dsh-hmr');
  const version=JSON.parse(fs.readFileSync(path.join(root,'package.json'),'utf8')).version;
  if(version!=='0.1.7-alpha.2')throw Error('Harness config watcher requires 0.1.7-alpha.2');
  const file=path.join(root,'lib/index.js');
  let source=fs.readFileSync(file,'utf8');
  const marker='/* sm75-exact-config-watch */';
  if(source.includes(marker))return;
  const anchor='const watcher = watch(target.root, {';
  if(source.split(anchor).length!==2)throw Error('Harness config watcher anchor changed');
  source=source.replace(anchor,marker+`
  // Retain only the requested file and ancestors needed for missing parents
  // and atomic replacement. Never open or watch unrelated sibling files.
  const configWatchPaths = new Set();
  for (let current of [resolve(filename), target.filename]) {
    while (!configWatchPaths.has(current)) {
      configWatchPaths.add(current);
      const parent = dirname(current);
      if (parent === current) break;
      current = parent;
    }
  }
  const watcher = watch(target.root, {`);
  const options='...watchOptions,\n\t\tdepth: target.depth,';
  if(source.split(options).length!==2)throw Error('Harness config watcher options changed');
  source=source.replace(options,'...watchOptions,\n\t\tignored: observed => !configWatchPaths.has(resolve(observed)),\n\t\tdepth: target.depth,');
  fs.writeFileSync(file,source);
}
