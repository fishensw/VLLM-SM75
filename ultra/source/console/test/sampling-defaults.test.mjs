import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";
import { modelSamplingDefaults } from "../sampling-defaults.mjs";

test("workbench defaults respect vllm/auto, startup overrides and output caps", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "sampling-defaults-"));
  try {
    fs.writeFileSync(path.join(dir, "generation_config.json"), JSON.stringify({temperature:.6,top_k:20,max_new_tokens:100}));
    const p = {id:"fp8",port:8000,args:[dir,"--served-model-name","local-model","--generation-config","vllm"]};
    const config = {baseURL:"http://127.0.0.1:8000/v1"};
    const model = {id:"local-model"};
    const get = () => modelSamplingDefaults("sm75-local",config,model,[p],p.id);
    assert.deepEqual(get().values,{temperature:1,top_p:1,top_k:0,min_p:0,presence_penalty:0,repetition_penalty:1,max_tokens:32768});
    p.args[p.args.length-1]="auto";
    assert.equal(get().values.temperature,.6);
    // auto model max_new_tokens is not a hard cap on DSH's explicit budget.
    assert.equal(get().values.max_tokens,32768);
    p.args.push("--override-generation-config",JSON.stringify({temperature:0,max_new_tokens:4096}));
    assert.equal(get().values.temperature,0);
    assert.equal(get().values.max_tokens,4096);
    model.maxTokens=2048;
    assert.equal(get().values.max_tokens,2048);
    p.args=[dir,"--served-model-name","local-model","--generation-config",dir];
    assert.equal(get().values.max_tokens,100);
  } finally { fs.rmSync(dir,{recursive:true,force:true}); }
});

test("same model alias resolves active profile; unrelated or ambiguous services remain unknown", () => {
  const a={id:"a",port:8000,args:["/missing","--served-model-name","model","--generation-config","vllm"]};
  const b={...a,id:"b",args:[...a.args,"--override-generation-config",'{"temperature":0.4}']};
  const config={baseURL:"http://127.0.0.1:8000/v1"}, model={id:"model"};
  assert.equal(modelSamplingDefaults("sm75-local",config,model,[a,b],"b").values.temperature,.4);
  assert.deepEqual(modelSamplingDefaults("sm75-local",config,model,[a,b]).values,{});
  assert.deepEqual(modelSamplingDefaults("remote",config,model,[a],"a").values,{});
  assert.deepEqual(modelSamplingDefaults("sm75-local",{baseURL:"https://example.com/v1"},model,[a],"a").values,{});
  assert.deepEqual(modelSamplingDefaults("sm75-local",config,{id:"other"},[a],"a").values,{});
});

test("Qwen3.8-27B recommendations are separate from vllm service defaults and model scoped", () => {
  const profile={id:"fp8",port:8000,args:["/models/Qwen3___8-27B-FP8","--served-model-name","old-alias","--generation-config","vllm"]};
  const get=()=>modelSamplingDefaults("sm75-local",{baseURL:"http://127.0.0.1:8000/v1"},{id:"old-alias"},[profile],"fp8");
  const result=get();
  assert.equal(result.values.top_p,1);
  assert.equal(result.values.top_k,0);
  assert.deepEqual(result.recommendations.thinking,{temperature:1,top_p:.95,top_k:20,min_p:0,presence_penalty:0,repetition_penalty:1});
  assert.deepEqual(result.recommendations.nonThinking,{temperature:.7,top_p:.8,top_k:20,min_p:0,presence_penalty:1.5,repetition_penalty:1});
  assert.equal(result.recommendations.thinking.max_tokens,undefined);
  profile.args[0]="/models/another-model";
  assert.equal(get().recommendations,undefined);
});

test("displayed defaults remain placeholders: saving untouched fields sends no overrides", () => {
  const source=fs.readFileSync(new URL('../public/app.js',import.meta.url),'utf8');
  const code=source.slice(source.indexOf('  function paintModelSampling()'),source.indexOf('  $("samplingModel").onchange'));
  const fields={temperature:"",top_p:"",top_k:"",min_p:"",presence_penalty:"",repetition_penalty:"",max_tokens:""};
  const elements={samplingModel:{value:"local/model"},modelSamplingDefaultInfo:{},modelSamplingRecommendations:{},modelSamplingRecommendation:{value:"thinking"},modelSamplingRecommendationInfo:{}};
  for(const key of Object.keys(fields)) elements['modelSampling-'+key]={value:""};
  const data={models:{},configured:[{id:"local/model",defaults:{values:{temperature:0,top_k:0,max_tokens:32768},source:"vLLM",outputSource:"工作台"}}]};
  const scope={samplingData:data,chatFields:fields,$:id=>elements[id]};
  vm.createContext(scope);vm.runInContext(code,scope);
  scope.paintModelSampling();
  assert.equal(elements['modelSampling-temperature'].placeholder,'服务默认 0');
  assert.equal(elements['modelSampling-max_tokens'].placeholder,'服务默认 32768');
  assert.equal(JSON.stringify(scope.modelSamplingValues()),'{}');
  data.models['local/model']={temperature:.8};scope.paintModelSampling();
  assert.equal(elements['modelSampling-temperature'].value,.8);
  assert.equal(JSON.stringify(scope.modelSamplingValues()),'{"temperature":0.8}');
  data.configured[0].defaults.recommendations={thinking:{temperature:1,top_p:.95,top_k:20,min_p:0,presence_penalty:0,repetition_penalty:1},nonThinking:{temperature:.7,top_p:.8,top_k:20,min_p:0,presence_penalty:1.5,repetition_penalty:1}};
  elements['modelSampling-max_tokens'].value=2048;
  elements.modelSamplingRecommendation.value='nonThinking';
  scope.applyModelRecommendation();
  assert.equal(scope.modelSamplingValues().presence_penalty,1.5);
  assert.equal(scope.modelSamplingValues().temperature,.7);
  assert.equal(scope.modelSamplingValues().max_tokens,2048);
  elements.modelSamplingRecommendation.value='thinking';scope.applyModelRecommendation();
  assert.equal(scope.modelSamplingValues().presence_penalty,0);
  assert.equal(scope.modelSamplingValues().top_p,.95);
  elements.samplingModel.value='remote/model';scope.paintModelSampling();
  assert.equal(elements['modelSampling-temperature'].placeholder,'默认值未提供');
  assert.equal(JSON.stringify(scope.modelSamplingValues()),'{}');
});
