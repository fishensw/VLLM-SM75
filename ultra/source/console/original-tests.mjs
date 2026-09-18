import fs from 'node:fs';import path from 'node:path';import http from 'node:http';import {spawn} from 'node:child_process';import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
export class OriginalTests{
 constructor(root,store,key){this.root=root;this.store=store;this.key=key;this.keyFn=typeof key==='function'?key:()=>key;this.processes=new Map();this.starting=new Map();}
 port(type){return type==='sql'?18002:18001;}
 ready(type){try{return JSON.parse(fs.readFileSync(path.join(this.root,'extensions',type,'manifest.json'))).original_ui===1;}catch{return false;}}
 async start(type,ip){if(this.processes.has(type))return;if(this.starting.has(type))return this.starting.get(type);const task=(async()=>{const dir=path.join(this.root,'extensions',type),fd=fs.openSync(path.join(dir,'original-server.log'),'a');const c=spawn('python3',[path.join(here,'original-test-server.py'),type,dir],{env:{...process.env,SM75_ENGINE_KEY:this.keyFn(),SM75_ENGINE_KEY_FILE:path.join(this.root,'engine-key.current'),SM75_CONTAINER_IP:ip},stdio:['ignore',fd,fd]});fs.closeSync(fd);let error;c.on('error',e=>error=e);c.once('exit',()=>this.processes.delete(type));this.processes.set(type,c);for(let i=0;i<80;i++){if(error||c.exitCode!==null)throw Error('原项目启动失败，请查看扩展日志');try{await fetch('http://127.0.0.1:'+this.port(type)+'/');return;}catch{await new Promise(r=>setTimeout(r,100));}}throw Error('原项目启动超时');})().finally(()=>this.starting.delete(type));this.starting.set(type,task);return task;}
 stop(type){const p=this.processes.get(type);p?.kill('SIGTERM');this.processes.delete(type);}
 close(){for(const type of this.processes.keys())this.stop(type);}
 async handle(req,res,{type,suffix,profile,ip}){
  if(!this.store.get('extension',type)?.enabled||!this.ready(type)){res.writeHead(409);res.end('请先启用并部署测试组件');return;}
  await this.start(type,ip);
  const apiPath=suffix.match(/^\/model\/(v1\/(?:models|chat\/completions))$/);
  if(apiPath){const upstream=http.request({host:'127.0.0.1',port:8000,path:'/'+apiPath[1],method:req.method,headers:{'Content-Type':'application/json',Authorization:'Bearer '+this.keyFn()}},r=>{res.writeHead(r.statusCode,{'Content-Type':r.headers['content-type']||'application/json','Cache-Control':'no-store'});r.pipe(res);});upstream.on('error',()=>{if(!res.headersSent)res.writeHead(503);res.end('模型尚未就绪');});req.pipe(upstream);res.on('close',()=>upstream.destroy());return;}
  const prefix='/bench-app/'+type;
  const up=http.request({host:'127.0.0.1',port:this.port(type),path:suffix,method:req.method,headers:{...req.headers,host:'127.0.0.1:'+this.port(type),'accept-encoding':'identity'}},r=>{
   const headers={...r.headers,'Cache-Control':'no-store'};delete headers['set-cookie'];
   if((r.headers['content-type']||'').includes('text/html')){const chunks=[];r.on('data',c=>chunks.push(c));r.on('end',()=>{let html=Buffer.concat(chunks).toString('utf8');const i=profile.args.indexOf('--served-model-name'),model=i>=0?profile.args[i+1]:profile.args[0];
    if(type==='speedtest'){
     html=html.replace('</body>','<script>'+fs.readFileSync(path.join(here,'public','benchmark-sampling.js'),'utf8')+'</script></body>');
     html=html.replace('const savedMinLength =',"if(!localStorage.getItem('sm75-speed-defaults-v2')){for(const [k,v]of Object.entries({MinLength:512,MaxLength:131072,Step:128,StepMultiplier:2,OutputLength:512}))localStorage.setItem('llmPerfTest'+k,String(v));localStorage.setItem('sm75-speed-defaults-v2','1');}const savedMinLength =");
     for(const [name,value]of Object.entries({minLength:512,maxLength:131072,step:128,stepMultiplier:2,outputLength:512,timeout:30000,temperature:1,top_p:0.1,presence_penalty:0,frequency_penalty:0}))html=html.replace(new RegExp('let d_'+name+' = [^;]+;'),'let d_'+name+' = '+value+';');
     html=html.replace('backendUrlInput.value = `ws://${host}:${port}`;','backendUrlInput.value = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}'+prefix+'`;');
     html=html.replace(/let d_openai_apiUrl = '[^']*'/,"let d_openai_apiUrl = "+JSON.stringify('http://'+ip+':8000/v1/chat/completions')).replace(/let d_openai_modelName = '[^']*'/,'let d_openai_modelName = '+JSON.stringify(model)).replace("let d_api_key = '';","let d_api_key = 'sm75-managed';");
    }else{
     html=html.replaceAll('https://api.openai.com/v1/chat/completions',`http://${req.headers.host}${prefix}/model/v1/chat/completions`);
     html=html.replaceAll("fetch('/save_trace'","fetch('"+prefix+"/save_trace'").replaceAll("fetch('/save_summary'","fetch('"+prefix+"/save_summary'").replaceAll("fetch('/download_zip'","fetch('"+prefix+"/download_zip'");
     const init=`<script>window.addEventListener('load',()=>{const e=document.getElementById('api-endpoint');if(e&&(e.value==='https://api.openai.com/v1/chat/completions'||!e.value))e.value=location.origin+${JSON.stringify(prefix+'/model/v1/chat/completions')};});</script>`;html=html.replace('</body>',init+'</body>');
    }
    delete headers['content-length'];delete headers['content-encoding'];res.writeHead(r.statusCode,headers);res.end(html);
   });}else{res.writeHead(r.statusCode,headers);r.pipe(res);}
  });up.on('error',()=>{if(!res.headersSent)res.writeHead(503);res.end('测试组件尚未就绪');});req.pipe(up);res.on('close',()=>up.destroy());
 }
 upgrade(req,socket,head,type,suffix){if(!this.store.get('extension',type)?.enabled||!this.processes.has(type)){socket.destroy();return;}const up=http.request({host:'127.0.0.1',port:this.port(type),path:suffix,headers:{...req.headers,host:'127.0.0.1:'+this.port(type)}});up.on('upgrade',(r,s,h)=>{socket.write('HTTP/1.1 101 Switching Protocols\r\n'+Object.entries(r.headers).map(([k,v])=>k+': '+v).join('\r\n')+'\r\n\r\n');if(h.length)socket.write(h);if(head.length)s.write(head);s.pipe(socket);socket.pipe(s);s.on('error',()=>socket.destroy());socket.on('error',()=>s.destroy());socket.on('close',()=>s.destroy());});up.on('error',()=>socket.destroy());up.on('response',r=>{r.resume();socket.destroy();});up.end();}
}
