import {DatabaseSync} from 'node:sqlite';
import path from 'node:path';
import fs from 'node:fs';
import crypto from 'node:crypto';

export const presets = [
 {id:'single-long',name:'单人长上下文',values:{'--max-num-seqs':'1','--max-num-batched-tokens':'8192'}},
 {id:'multi-user',name:'多人并发',values:{'--max-num-seqs':'16','--max-num-batched-tokens':'8192'}},
 {id:'coding',name:'Coding',values:{'--max-num-seqs':'8','--max-num-batched-tokens':'8192'}},
 {id:'power-idle',name:'低功耗待机',values:{'--auto-sleep-idle-timeout':'30','--auto-sleep-offload-target':'exit'}},
];
// Patches preserve every advanced argument not explicitly selected by a preset.
export function applyPreset(args,id){const preset=presets.find(p=>p.id===id);if(!preset)throw Error('模板不存在');const out=[...args];for(const [flag,value] of Object.entries(preset.values)){const i=out.indexOf(flag);if(i>=0)out.splice(i,2,flag,value);else out.push(flag,value);}return out;}
export class Store {
 constructor(root){fs.mkdirSync(root,{recursive:true,mode:0o700});this.db=new DatabaseSync(path.join(root,'workbench.sqlite'));this.db.exec('PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS records(kind TEXT NOT NULL,id TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(kind,id))');this.root=root;}
 get(kind,id){const r=this.db.prepare('SELECT value FROM records WHERE kind=? AND id=?').get(kind,id);return r?JSON.parse(r.value):null;}
 list(kind){return this.db.prepare('SELECT value FROM records WHERE kind=? ORDER BY id').all(kind).map(r=>JSON.parse(r.value));}
 put(kind,id,value){this.db.prepare('INSERT INTO records VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET value=excluded.value').run(kind,id,JSON.stringify(value));return value;}
 settings(){return this.get('settings','main')||{modelRoot:process.env.SM75_MODEL_ROOT||path.join(this.root,'models'),cacheRoot:path.join(this.root,'cache'),defaultProfile:'',autoStart:false,defaultChatProfile:'',source:'huggingface'};}
 saveSettings(input,profiles){const s={...this.settings(),...input};for(const k of ['modelRoot','cacheRoot']){if(typeof s[k]!=='string'||!path.isAbsolute(s[k])||s[k].includes('\0'))throw Error('模型与缓存目录须为绝对路径');}for(const k of ['modelRoot','cacheRoot']){s[k]=path.normalize(s[k].trim());if(s[k]===path.parse(s[k]).root)throw Error('请选择具体子目录，例如 /data/model 或 /data/cache');let parent=s[k];while(!fs.existsSync(parent)&&path.dirname(parent)!==parent)parent=path.dirname(parent);if(!fs.statSync(parent).isDirectory())throw Error('目录路径不能指向文件');fs.accessSync(parent,fs.constants.W_OK);}if(s.modelRoot===s.cacheRoot)throw Error('模型目录与编译缓存目录必须分开设置');for(const k of ['defaultProfile','defaultChatProfile'])if(s[k]&&!profiles.some(p=>p.id===s[k]))throw Error('默认配置不存在');if(typeof s.autoStart!=='boolean'||!['huggingface','modelscope'].includes(s.source))throw Error('设置格式无效');if(s.autoStart&&!s.defaultProfile)throw Error('自动启动需要指定默认配置');return this.put('settings','main',s);}
 saveTemplate(name,profile,id){if(typeof name!=='string'||!name.trim()||name.length>100)throw Error('填写模板名称');const old=id?this.get('template',id):null;if(id&&!old)throw Error('模板不存在');const value={id:old?.id||crypto.randomUUID(),name:name.trim(),revision:(old?.revision||0)+1,profile:structuredClone(profile),updated:Date.now()};this.put('template-version',`${value.id}:${value.revision}`,value);return this.put('template',value.id,value);}
 registerModel(dir,name){const real=fs.realpathSync(dir);if(!fs.statSync(real).isDirectory()||!fs.existsSync(path.join(real,'config.json')))throw Error('模型目录必须包含 config.json');const config=JSON.parse(fs.readFileSync(path.join(real,'config.json')));const id=crypto.createHash('sha256').update(real).digest('hex').slice(0,16);const old=this.get('model',id)||{};return this.put('model',id,{...old,id,name:name||old.name||path.basename(real),path:real,source:'local',modelType:config.model_type||old.modelType||null,registered:old.registered||Date.now()});}
 close(){this.db.close();}
}
