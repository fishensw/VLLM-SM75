import fs from 'node:fs';
import path from 'node:path';
const real=p=>{try{return fs.realpathSync(p);}catch{return path.resolve(p);}};
const inside=(parent,child)=>{const rel=path.relative(parent,child);return rel===''||(!rel.startsWith('..'+path.sep)&&rel!=='..'&&!path.isAbsolute(rel));};
export function deleteModelFiles(model,confirmedPath,{running=[],downloads=[],protectedPaths=[]}={}){
 if(confirmedPath!==model.path)throw Error('模型路径已变化，请刷新后重试');
 const target=real(model.path);
 if(target===path.parse(target).root||!fs.statSync(target).isDirectory()||!fs.existsSync(path.join(target,'config.json')))throw Error('拒绝删除非模型目录');
 for(const p of protectedPaths)if(p&&inside(target,real(p)))throw Error('不能删除模型根目录、缓存根目录或工作台数据目录');
 for(const p of running){const refs=[p.args[0]];const i=p.args.indexOf('--speculative-config');if(i>=0){const spec=JSON.parse(p.args[i+1]);if(spec.model)refs.push(spec.model);}for(const ref of refs)if(ref&&inside(target,real(ref)))throw Error('模型正在使用，请先停止使用它的模型服务（含草稿模型）');}
 for(const dest of downloads)if(dest&&(inside(target,real(dest))||inside(real(dest),target)))throw Error('模型正在下载，请先停止下载');
 // Remove only this registered/catalogued model directory; do not follow file symlinks into shared caches.
 fs.rmSync(target,{recursive:true,force:false});
 return {deleted:true,id:model.id,path:model.path};
}
