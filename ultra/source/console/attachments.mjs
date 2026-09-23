import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
const exec=promisify(execFile);
export class Attachments {
 constructor(root){this.dir=path.join(root,'attachments');fs.mkdirSync(this.dir,{recursive:true,mode:0o700});}
 file(id){if(!/^[a-f0-9]{32}$/.test(id))throw Error('附件 ID 无效');return path.join(this.dir,id);}
 async upload(req,profile){
  const chunks=[];let size=0;
  for await(const c of req){size+=c.length;if(size>20*1024*1024)throw Error('单张图片不能超过 20 MiB');chunks.push(c);}
  if(!size)throw Error('图片为空');
  const id=crypto.randomBytes(16).toString('hex'),file=this.file(id);
  fs.writeFileSync(file,Buffer.concat(chunks),{mode:0o600,flag:'wx'});
  try {
   const script='from PIL import Image;import sys,json,warnings;warnings.simplefilter("error");im=Image.open(sys.argv[1]);assert im.format in ["PNG","JPEG","WEBP","GIF"],"unsupported image";assert im.width*im.height<=40000000,"too many pixels";d={"mime":Image.MIME[im.format],"width":im.width,"height":im.height};im.verify();print(json.dumps(d))';
   const {stdout}=await exec(process.env.SM75_PYTHON || 'python3',['-c',script,file],{timeout:15000,maxBuffer:4096});
   const meta={id,profile,size,...JSON.parse(stdout),created:Date.now()};
   fs.writeFileSync(file+'.json',JSON.stringify(meta),{mode:0o600,flag:'wx'});return meta;
  }catch {fs.unlinkSync(file);throw Error('图片无法解码或格式不支持，请使用 PNG/JPEG/WebP/GIF（最多 4000 万像素）');}
 }
 read(id,profile){const file=this.file(id),meta=JSON.parse(fs.readFileSync(file+'.json'));if(profile&&meta.profile!==profile)throw Error('附件不属于当前模型会话');return {meta,data:fs.readFileSync(file)};}
 validate(messages,profile){
  if(!Array.isArray(messages)||messages.length>200)throw Error('无效会话');
  for(const m of messages){
   if(!m||typeof m!=='object'||!['user','assistant','system'].includes(m.role))throw Error('无效会话角色');
   if(typeof m.content==='string')continue;
   if(m.role!=='user'||!Array.isArray(m.content)||m.content.length>5)throw Error('无效多模态消息');
   for(const part of m.content){if(!part||typeof part!=='object')throw Error('无效多模态消息');if(part.type==='text'&&typeof part.text==='string')continue;if(part.type!=='image'||typeof part.attachmentId!=='string')throw Error('无效图片消息');this.read(part.attachmentId,profile);}
  }
 }
 expand(messages,profile,vision){
  this.validate(messages,profile);
  return messages.map(m=>typeof m.content==='string'?m:{...m,content:m.content.map(part=>{
   if(part.type==='text')return part;
   if(!vision)throw Error('当前模型未启用图片输入');
   const {meta,data}=this.read(part.attachmentId,profile);
   return {type:'image_url',image_url:{url:`data:${meta.mime};base64,${data.toString('base64')}`}};
  })});
 }
}
