export async function searchModels(provider,query,page=1,fetcher=fetch){
 if(!['huggingface','modelscope'].includes(provider))throw Error('请选择下载源');
 query=String(query||'').trim();if(!query||query.length>160)throw Error('请输入模型关键词');
 page=Number(page);if(!Number.isInteger(page)||page<1||page>20)throw Error('页码范围 1–20');
 const size=12;
 if(provider==='huggingface'){
  const u=new URL('https://huggingface.co/api/models');u.search=new URLSearchParams({search:query,sort:'downloads',direction:'-1',limit:String(page*size+1)});
  const r=await fetcher(u,{signal:AbortSignal.timeout(20000)});if(!r.ok)throw Error('Hugging Face 搜索失败：'+r.status);const data=await r.json();if(!Array.isArray(data))throw Error('搜索结果格式无效');
  return {provider,page,more:data.length>page*size,items:data.slice((page-1)*size,page*size).map(m=>({id:m.id,name:m.id,downloads:m.downloads??null,likes:m.likes??null,task:m.pipeline_tag||'',tags:m.tags||[]}))};
 }
 const r=await fetcher('https://www.modelscope.cn/api/v1/dolphin/models',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({PageNumber:page,PageSize:size,Name:query,SortBy:'Default',Target:'',Criterion:[],SingleCriterion:[]}),signal:AbortSignal.timeout(20000)});
 if(!r.ok)throw Error('ModelScope 搜索失败：'+r.status);const d=await r.json(),data=d.Data?.Model;if(d.Code!==200||!Array.isArray(data?.Models))throw Error('ModelScope 搜索结果格式无效');
 return {provider,page,total:data.TotalCount,more:page*size<data.TotalCount,items:data.Models.map(m=>({id:m.Path+'/'+m.Name,name:m.ChineseName||m.Name,downloads:m.Downloads??null,likes:m.Stars??null,task:(m.Tasks||[]).join(' / '),tags:m.Libraries||[]}))};
}
