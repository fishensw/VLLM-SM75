import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {assertHarnessVersion} from './apply-branding.mjs';

export function verifyClientBundles(base = '/opt/harness/node_modules') {
  const scope = path.join(base, '@deepseek-ai');
  // Only these custom packages publish a DSH browser module. sm75-workbench
  // is server-only; a remaining client file still receives the syntax check.
  const customClients = ['sm75-brand', 'dsh-watcher'];
  if (fs.existsSync(path.join(scope, 'dsh/package.json'))) {
    assertHarnessVersion(base);
    for (const name of customClients) {
      const directory = path.join(base, name);
      const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'package.json'), 'utf8'));
      const source = fs.readFileSync(path.join(directory, 'lib/client.js'), 'utf8');
      const moduleId = source.match(/\bid:\s*["']([^"']+)["']/)?.[1];
      if (manifest.name !== name || moduleId !== name)
        throw Error(`Custom client module identity mismatch: ${name}`);
    }
  }
  const packages = fs.readdirSync(scope).map(name => '@deepseek-ai/' + name)
    .concat(['sm75-workbench', ...customClients]);
  let checked = 0;
  const errors = [];
  for (const name of packages) {
    const file = ['lib/client.js', 'client.js'].map(p => path.join(base, name, p))
      .find(p => fs.existsSync(p));
    if (!file) continue;
    checked++;
    const result = spawnSync(process.execPath, ['--check', file], {encoding: 'utf8'});
    if (result.status !== 0) errors.push(name + ': ' + (result.error?.message || result.stderr.slice(-500)));
  }
  if (!checked) throw Error('No client bundles found for syntax validation');
  if (errors.length) throw Error('Invalid client bundles:\n' + errors.join('\n'));
  return {checked, failed: 0};
}

if (process.argv[1] && fs.existsSync(process.argv[1]) && fs.realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(verifyClientBundles(process.argv[2])));
}
