(()=>{
 const fields={temperature:'temperatureInput',top_p:'topPInput',presence_penalty:'presencePenaltyInput',frequency_penalty:'frequencyPenaltyInput'};
 async function apply(){
  const profile=new URLSearchParams(location.search).get('profile');if(!profile)return;
  const r=await fetch('/console-api/profiles/'+encodeURIComponent(profile)+'/chat-config');if(!r.ok)return;
  const config=await r.json();
  for(const [key,id]of Object.entries(fields)){const input=document.getElementById(id);if(!input)continue;
   if(Number.isFinite(config.sampling?.[key])){input.value=config.sampling[key];input.title='默认值来自'+config.source+'，可手动调整';}
  }
 }
 const init=()=>{apply().catch(()=>{});document.getElementById('modelName')?.addEventListener('change',()=>apply().catch(()=>{}));document.getElementById('resetParams')?.addEventListener('click',()=>apply().catch(()=>{}));};
 if(document.readyState==='complete')init();else window.addEventListener('load',init);
})();
