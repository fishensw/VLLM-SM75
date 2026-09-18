import {applyBranding} from './apply-branding.mjs';
import fs from 'node:fs';
import path from 'node:path';
const base='/opt/harness/node_modules',source='/opt/sm75-workbench/console/plugins';
const packages=[['sm75-workbench','sm75-workbench'],['dsh-watcher','dsh-watcher'],['token-usage','@deepseek-ai/dsh-client-ui-token-usage']];
const manifest=path.join(base,'@deepseek-ai/dsh/package.json'),app=JSON.parse(fs.readFileSync(manifest));
for(const [dir,name]of packages){const target=path.join(base,name);fs.mkdirSync(path.dirname(target),{recursive:true});fs.cpSync(path.join(source,dir),target,{recursive:true});const pkg=JSON.parse(fs.readFileSync(path.join(target,'package.json')));app.dependencies[name]=pkg.version;}
fs.writeFileSync(manifest,JSON.stringify(app,null,2));

applyBranding(base);
