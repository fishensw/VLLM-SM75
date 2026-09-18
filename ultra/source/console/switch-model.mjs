export async function switchModel(target,{current,stop,start,gpus,sleep=ms=>new Promise(r=>setTimeout(r,ms)),attempts=120}){
 if(current?.id===target.id)return {running:true,name:target.id};
 if(current)await stop(current);
 for(let i=0;i<attempts;i++){
  const cards=await gpus();if(!cards.length)throw Error('无法读取 GPU 状态，未启动模型');
  if(cards.every(g=>g.usedMiB<=1000))return start(target);
  if(!current)throw Error('GPU 被其他服务占用，未启动模型');
  await sleep(500);
 }
 throw Error('旧模型显存尚未释放，新配置未启动，请查看日志');
}
