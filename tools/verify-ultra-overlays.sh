#!/usr/bin/env bash
# Run only in a disposable no-GPU container with a fresh locked Harness install.
set -euo pipefail
test "${SM75_DISPOSABLE_AUDIT:-}" = 1
mkdir -p /tmp/ultra-overlay-audit
tar -xf /audit/ultra-overlay-audit.tar -C /tmp/ultra-overlay-audit
base=/opt/harness/node_modules/@deepseek-ai
python3 /tmp/ultra-overlay-audit/overlay/font-scale.py
node /opt/sm75-workbench/console/install-native-plugins.mjs
node /opt/sm75-workbench/console/verify-client-bundles.mjs
find "$base" /opt/harness/node_modules/sm75-workbench /opt/harness/node_modules/sm75-brand /opt/harness/node_modules/dsh-watcher \
    -type f \( -name '*.js' -o -name '*.mjs' -o -name '*.css' -o -name '*.html' -o -name '*.json' -o -name '*.webmanifest' \) \
    -print0 | sort -z | xargs -0 sha256sum > /tmp/branding-first.sha256
python3 /tmp/ultra-overlay-audit/overlay/font-scale.py
node /opt/sm75-workbench/console/install-native-plugins.mjs
sha256sum -c --quiet /tmp/branding-first.sha256
node --input-type=module <<'JS'
import assert from 'node:assert/strict';
import fs from 'node:fs';
const base='/opt/harness/node_modules/@deepseek-ai';
const version='0.1.7-alpha.2';
let count=0;
const lock=JSON.parse(fs.readFileSync('/tmp/ultra-overlay-audit/harness/package-lock.json','utf8'));
const native=Object.keys(lock.packages).filter(name => /^node_modules\/@deepseek-ai\/dsh(?:-[^/]+)?$/.test(name));
for (const entry of native) {
  const name=entry.replace('node_modules/@deepseek-ai/','');
  const metadata=JSON.parse(fs.readFileSync(`${base}/${name}/package.json`,'utf8'));
  assert.equal(metadata.version,version,name);
  count++;
}
assert(count > 200, 'expected the complete native Harness package graph');
const layout=fs.readFileSync(base+'/dsh-client-ui-layout/lib/client.js','utf8');
assert(layout.includes('shell.overlay'), 'native additive layout slot must remain available');
assert(!layout.includes('sm75.statusbar'), 'obsolete bundled layout must not replace upstream');
const theme=fs.readFileSync(base+'/dsh-client-ui-theme/lib/client.js','utf8');
assert(theme.includes('--dsh-font-scale'), 'native font rules must be scalable');
assert(theme.includes('fontSize >= 26'), 'ultra font range must remain available');
console.log(`PASS ${count} exact-version DSH packages, native layout, font rules, repeated installer`);
JS
