import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const [root,file,mode]=process.argv.slice(2),cfg=JSON.parse(fs.readFileSync(file)),here=path.dirname(fileURLToPath(import.meta.url));
const html=fs.readFileSync(path.join(root,'sql_benchmark.html'),'utf8');
const between=(a,b)=>{const i=html.indexOf(a),j=html.indexOf(b,i);if(i<0||j<0)throw Error('上游接口结构变化');return html.slice(i,j);};
const questions=html.match(/const questionsList = (.*);/)[0];
const children=new Set();process.on('SIGTERM',()=>{for(const c of children)c.kill();process.exit(143);});
async function executeSql(sql){return new Promise((resolve,reject)=>{const c=spawn('python3',[path.join(here,'sql-query.py'),root],{windowsHide:true});children.add(c);let out='',err='';const timer=setTimeout(()=>c.kill(),15000);c.stdout.on('data',d=>out+=d);c.stderr.on('data',d=>err+=d);c.on('error',reject);c.on('close',code=>{clearTimeout(timer);children.delete(c);if(code)return reject(Error(err||'SQL 查询超时'));try{resolve(JSON.parse(out));}catch(e){reject(e);}});c.stdin.end(sql);});}
const context=vm.createContext({console,executeSql,dbSchema:JSON.parse(fs.readFileSync(path.join(root,'schema.json'))),setTimeout,clearTimeout,AbortController,crypto:globalThis.crypto,btoa,Uint8Array,fetch:(url,options)=>{const payload=JSON.parse(options.body);payload.chat_template_kwargs={enable_thinking:cfg.thinking};return fetch(url,{...options,body:JSON.stringify(payload)});}});
vm.runInContext(questions+'\n'+between('function formatSchema(schema)', 'async function executeSql(sql)')+'\n'+between('function extractThinking(content)','async function saveTraceToServer(')+'\nglobalThis.core={questionsList,verifyResults,isCorrect,runQuestionBenchmark};',context,{timeout:1000});
const {core}=context,rows=[];
for(let round=0;round<(cfg.repeats||1);round++)for(const q of core.questionsList.filter(q=>cfg.questions.includes(q.id))){
 const trace=[],started=Date.now();let result;
 try{result=mode==='reference'?{queryResult:await executeSql(q.sql),generatedSql:q.sql}:await core.runQuestionBenchmark(q,cfg.url,process.env.SM75_BENCH_KEY,cfg.model,cfg.output,0,null,cfg.timeout*1000,3,new AbortController().signal,()=>{},step=>trace.push(step),()=>{});}
 catch(e){result={error:e.message};}
 const check=result.queryResult?core.verifyResults(q,result.queryResult):null;
 rows.push({question:q.id,title:q.question,round:round+1,success:!result.error&&core.isCorrect(check),elapsed_ms:Date.now()-started,...result,check,trace});
 fs.writeFileSync(file+'.results.tmp',JSON.stringify({rows,updated:Date.now()}));fs.renameSync(file+'.results.tmp',file+'.results.json');console.log('SQL',q.id,rows.at(-1).success?'pass':'fail',result.error||'');
}
if(rows.some(r=>!r.success))process.exitCode=2;
