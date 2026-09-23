import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import {applyNativeConfigWatch} from '../native-config-watch.mjs';

test('native config watch excludes unreadable siblings while retaining missing parents and atomic replacement',()=>{
 const base=fs.mkdtempSync(path.join(os.tmpdir(),'sm75-config-watch-'));
 const root=path.join(base,'@deepseek-ai/dsh-hmr');
 fs.mkdirSync(path.join(root,'lib'),{recursive:true});
 fs.writeFileSync(path.join(root,'package.json'),JSON.stringify({version:'0.1.7-alpha.2'}));
 const file=path.join(root,'lib/index.js');
 fs.writeFileSync(file,'function optionsFor(filename,target,watchOptions){const watcher = watch(target.root, {\n\t\t...watchOptions,\n\t\tdepth: target.depth,\n});return watcher;}');
 try{
  applyNativeConfigWatch(base);
  const source=fs.readFileSync(file,'utf8');
  applyNativeConfigWatch(base);
  assert.equal(fs.readFileSync(file,'utf8'),source);
  const page={resolve:path.resolve,dirname:path.dirname,watch:(root,options)=>({root,options})};
  vm.runInNewContext(source,page);
  const filename=path.join(base,'home/profiles/web/cordis.patch.yml');
  const canonical=path.join(base,'canonical/profiles/web/cordis.patch.yml');
  const result=page.optionsFor(filename,{filename:canonical,root:path.join(base,'canonical'),depth:2},{ignorePermissionErrors:false});
  const ignored=result.options.ignored;
  assert.equal(ignored(canonical),false);
  assert.equal(ignored(path.dirname(canonical)),false);
  assert.equal(ignored(path.join(base,'canonical')),false);
  assert.equal(ignored(filename),false);
  assert.equal(ignored(canonical+'.private-backup'),true);
  assert.equal(ignored(path.join(base,'canonical','root-only-backup')),true);
  assert.equal(ignored(path.join(base,'canonical','unrelated/subdir/file')),true);
  assert.equal(result.options.ignorePermissionErrors,false);
  assert.equal(result.options.depth,2);
  fs.writeFileSync(path.join(root,'package.json'),'{"version":"unexpected"}');
  assert.throws(()=>applyNativeConfigWatch(base),/requires 0.1.7-alpha.2/);
 }finally{fs.rmSync(base,{recursive:true,force:true});}
});
