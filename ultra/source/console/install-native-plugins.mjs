import {applyNativeConfigWatch} from './native-config-watch.mjs';
import {applyUnifiedShell} from "./native-shell.mjs";
import {applyBranding, assertHarnessVersion} from './apply-branding.mjs';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
// Avoid fs.cpSync's native recursive-copy crash on a Windows UNC source. These
// are packaged, regular source files; reject links instead of dereferencing.
function copyPackage(source, target) {
  fs.mkdirSync(target, {recursive: true});
  for (const entry of fs.readdirSync(source, {withFileTypes: true})) {
    const from = path.join(source, entry.name), to = path.join(target, entry.name);
    if (entry.isDirectory()) copyPackage(from, to);
    else if (entry.isFile()) fs.copyFileSync(from, to);
    else throw Error(`Unexpected non-regular plugin source: ${entry.name}`);
  }
}
export function installNativePlugins(base = '/opt/harness/node_modules', consoleRoot = here) {
  assertHarnessVersion(base);
  const source = path.join(consoleRoot, 'plugins');
  const packages = [
    ['sm75-workbench', 'sm75-workbench'],
    ['sm75-brand', 'sm75-brand'],
    ['dsh-watcher', 'dsh-watcher'],
    ['sm75-workspace-tools', 'sm75-workspace-tools'],
  ];
  const manifest = path.join(base, '@deepseek-ai/dsh/package.json');
  const app = JSON.parse(fs.readFileSync(manifest, 'utf8'));
  app.dependencies ||= {};
  const retiredUsage = '@deepseek-ai/dsh-client-ui-token-usage';
  delete app.dependencies[retiredUsage];
  fs.rmSync(path.join(base, retiredUsage), {recursive: true, force: true});
  // The managed model plugin is server-only; reinstall removes its old UI.
  fs.rmSync(path.join(base, 'sm75-workbench/lib/client.js'), {force: true});
  for (const [dir, name] of packages) {
    const target = path.join(base, name);
    fs.mkdirSync(path.dirname(target), {recursive: true});
    copyPackage(path.join(source, dir), target);
    const pkg = JSON.parse(fs.readFileSync(path.join(target, 'package.json'), 'utf8'));
    if (pkg.name !== name) throw Error(`Custom plugin package name mismatch: ${name}`);
    app.dependencies[name] = pkg.version;
  }
  fs.writeFileSync(manifest, JSON.stringify(app, null, 2));
  applyBranding(base, consoleRoot);
  applyUnifiedShell(base);
  applyNativeConfigWatch(base);
  return {installed: packages.map(([, name]) => name)};
}
if (process.argv[1] && fs.existsSync(process.argv[1]) && fs.realpathSync(process.argv[1]) === fileURLToPath(import.meta.url))
  console.log(JSON.stringify(installNativePlugins(process.argv[2], process.argv[3])));
