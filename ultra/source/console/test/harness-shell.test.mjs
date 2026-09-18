import test from 'node:test';
import assert from 'node:assert/strict';
import {integrateHarnessHtml} from '../harness-shell.mjs';
test('DSH retains root and native assets in same document with shared navigation',()=>{const source='<html><head><script src="./assets/app.js"></script></head><body><div id="root"></div></body></html>';const html=integrateHarnessHtml(source);assert.ok(html.includes('<div id="root"></div>'));assert.ok(html.includes('src="./assets/app.js"'));assert.ok(html.includes('<base href="/">'));assert.ok(html.includes('/?page=profiles'));assert.ok(html.includes('aria-current="page"'));assert.equal(/<iframe/i.test(html),false);});
