#!/usr/bin/env bash
# Run only in a disposable no-GPU container. Replays the full-image overlay order.
set -euo pipefail
test "${SM75_DISPOSABLE_AUDIT:-}" = 1
mkdir -p /tmp/ultra-overlay-audit
tar -xf /audit/ultra-overlay-audit.tar -C /tmp/ultra-overlay-audit
base=/opt/harness/node_modules/@deepseek-ai
for mapping in \
    dsh-theme:dsh-client-ui-theme \
    dsh-layout:dsh-client-ui-layout \
    dsh-chat:dsh-client-ui-chat \
    dsh-sidebar-right:dsh-client-ui-sidebar-right \
    dsh-preview:dsh-client-ui-sidebar-documentpreview \
    dsh-deliverables:dsh-client-ui-deliverables; do
    cp -a "/tmp/ultra-overlay-audit/overlay/${mapping%%:*}/." "$base/${mapping#*:}/lib/"
done
python3 /tmp/ultra-overlay-audit/overlay/font-scale.py
node /opt/sm75-workbench/console/install-native-plugins.mjs
find "$base" -type f -name '*.js' -print0 | sort -z | xargs -0 sha256sum > /tmp/branding-first.sha256
node /opt/sm75-workbench/console/install-native-plugins.mjs
sha256sum -c --quiet /tmp/branding-first.sha256
node --input-type=module <<'JS'
import assert from 'node:assert/strict';
import fs from 'node:fs';
const base='/opt/harness/node_modules/@deepseek-ai';
const layout=fs.readFileSync(base+'/dsh-client-ui-layout/lib/client.js','utf8');
const chat=fs.readFileSync(base+'/dsh-client-ui-chat/lib/client.js','utf8');
assert(layout.includes('const productTitle = "工作台"'));
assert(layout.includes('"sm75.statusbar"'));
assert(chat.includes('用量与耗时统计中'));
assert(!chat.includes('if (stats.steps === 0 && !hasTokens) return null;'));
console.log('PASS full-build overlay order, statusbar, branding, chat state, repeated installer');
JS
