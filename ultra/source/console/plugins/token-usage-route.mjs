/**
 * token-usage-route — host-plane HTTP routes on the Harness web server itself.
 *
 * Registers exact routes:
 *   /token-usage.json  -> live per-model aggregate usage JSON
 *   /token-usage       -> minimal dashboard page (optional direct view)
 *
 * This replaces the old companion file server on :3082. The routes expose
 * ONLY the aggregate token/cost report; no file browsing, no session
 * transcript access.
 *
 * Reads both log generations in place:
 *   - v0 logs: `session.jsonl.zstd`, usage on `assistant/chunk` events;
 *   - v3 logs: `session.v3.jsonl.zstd`, usage on `assistant/attempt` /
 *     `assistant/message` stream records (settled `assistant/message.usage`
 *     preferred, so a message's embedded stream is never counted twice).
 */
import fs from 'node:fs'
import path from 'node:path'
import { zstdDecompressSync } from 'node:zlib'

export const name = 'token-usage-route'
export const inject = ['webServer']

const SESSIONS_ROOT = path.join(process.env.DSH_HOME || '/dsh/home', 'sessions')
const ZSTD_MAGIC = 0xFD2FB528

/** USD per 1M tokens; quota in USD. */
const MODEL_PRICING = {}

function scanZstdFrames(buf) {
  const frames = []
  let offset = 0
  while (offset < buf.length) {
    const start = offset
    if (buf.length - offset < 4) break
    if (buf.readUInt32LE(offset) !== ZSTD_MAGIC) throw new Error(`bad zstd magic at byte ${offset}`)
    offset += 4
    if (offset >= buf.length) break
    const descriptor = buf.readUInt8(offset++)
    if ((descriptor & 0x18) !== 0) throw new Error(`reserved zstd frame bits at byte ${offset - 1}`)
    const contentSizeFlag = descriptor >>> 6
    const singleSegment = (descriptor & 0x20) !== 0
    const checksum = (descriptor & 0x04) !== 0
    const dictionaryFlag = descriptor & 0x03
    const dictionaryBytes = dictionaryFlag === 3 ? 4 : dictionaryFlag
    const contentSizeBytes = contentSizeFlag === 0 ? (singleSegment ? 1 : 0) : (1 << contentSizeFlag)
    offset += (singleSegment ? 0 : 1) + dictionaryBytes + contentSizeBytes
    for (;;) {
      if (buf.length - offset < 3) throw new Error(`truncated zstd block at byte ${offset}`)
      const blockHeader = buf.readUIntLE(offset, 3)
      offset += 3
      const lastBlock = (blockHeader & 1) !== 0
      const blockType = (blockHeader >>> 1) & 0x03
      const blockSize = blockHeader >>> 3
      if (blockType === 3) throw new Error(`reserved zstd block type at byte ${offset - 3}`)
      offset += blockType === 1 ? 1 : blockSize
      if (lastBlock) break
    }
    if (checksum) offset += 4
    frames.push({ start, end: offset })
  }
  return frames
}

function decodeZstdFile(file) {
  const buf = fs.readFileSync(file)
  const frames = scanZstdFrames(buf)
  let text = ''
  for (const frame of frames) text += zstdDecompressSync(buf.subarray(frame.start, frame.end)).toString('utf8')
  return text
}

function sessionFiles() {
  const out = []
  const walk = (dir) => {
    let best = null
    let bestVersion = -1
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        walk(full)
        continue
      }
      const match = /^session(\.v(\d+))?\.jsonl\.zstd$/.exec(entry.name)
      if (match === null) continue
      const version = match[2] === undefined ? 0 : Number(match[2])
      if (version > bestVersion) {
        bestVersion = version
        best = full
      }
    }
    if (best !== null) out.push(best)
  }
  if (fs.existsSync(SESSIONS_ROOT)) walk(SESSIONS_ROOT)
  return out
}

/**
 * Provider usage records carried by one durable session event.
 * v0 keeps usage on `assistant/chunk`; v3 carries the exact stream on
 * `assistant/attempt` / `assistant/message`, with the settled usage also on
 * `assistant/message.usage` (preferred there to avoid counting its stream twice).
 */
function usagesOf(event) {
  const data = event.data
  if (event.type === 'assistant/chunk') {
    const chunk = data?.chunk
    return chunk?.type === 'usage' && typeof chunk.usage === 'object' ? [chunk.usage] : []
  }
  if (event.type !== 'assistant/message' && event.type !== 'assistant/attempt') return []
  if (event.type === 'assistant/message' && typeof data.usage === 'object' && data.usage !== null) {
    return [data.usage]
  }
  const stream = data?.stream
  if (!Array.isArray(stream)) return []
  const out = []
  for (const record of stream) {
    const chunk = record?.chunk
    if (record?.type === 'chunk' && chunk?.type === 'usage' && typeof chunk.usage === 'object') out.push(chunk.usage)
  }
  return out
}

function rangeWindow(range, now = new Date()) {
  switch (range) {
    case '24h': return { from: now.getTime() - 24 * 60 * 60 * 1000, to: Number.POSITIVE_INFINITY }
    case 'yesterday': {
      const start = new Date(now); start.setHours(0, 0, 0, 0)
      const to = start.getTime(); const from = to - 24 * 60 * 60 * 1000
      return { from, to }
    }
    case '7d': return { from: now.getTime() - 7 * 24 * 60 * 60 * 1000, to: Number.POSITIVE_INFINITY }
    case '30d': return { from: now.getTime() - 30 * 24 * 60 * 60 * 1000, to: Number.POSITIVE_INFINITY }
    default: return null
  }
}

function dayKey(time) {
  const d = new Date(time)
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${d.getFullYear()}-${m}-${day}`
}

export function tokenUsageReport(range) {
  const totals = new Map()
  const ensure = (provider, model) => {
    const key = `${provider}::${model}`
    if (!totals.has(key)) totals.set(key, {
      provider, model, sessions: new Set(), requests: 0,
      uncachedInputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0,
      outputTokens: 0, reasoningTokens: 0,
    })
    return totals.get(key)
  }
  const window = rangeWindow(range)
  const inRange = (time) => window === null || (time >= window.from && time < window.to)
  const activeSessions = new Set()
  const daily = new Map()
  let messageCount = 0
  let files = 0
  let skipped = 0
  for (const file of sessionFiles()) {
    files += 1
    let text
    try { text = decodeZstdFile(file) } catch { skipped += 1; continue }
    let current = null
    const seen = new Set()
    for (const line of text.split('\n')) {
      if (line.length === 0) continue
      let event
      try { event = JSON.parse(line) } catch { continue }
      if (event.type === 'request/header') {
        const config = event.data?.header?.config ?? {}
        current = { provider: config.provider ?? 'unknown', model: config.model ?? 'unknown' }
        if (typeof event.time === 'number' && inRange(event.time)) {
          seen.add(`${current.provider}::${current.model}`)
          ensure(current.provider, current.model).requests += 1
        }
        continue
      }
      if ((event.type === 'user/message' || event.type === 'assistant/message') && typeof event.time === 'number' && inRange(event.time)) {
        messageCount += 1
      }
      if (typeof event.time === 'number' && !inRange(event.time)) continue
      for (const u of usagesOf(event)) {
        const provider = current?.provider ?? 'unknown'
        const model = current?.model ?? 'unknown'
        const t = ensure(provider, model)
        const cacheRead = Number(u.cacheReadTokens ?? 0)
        const cacheWrite = Number(u.cacheWriteTokens ?? 0)
        const prompt = Number(u.inputTokens ?? 0)
        t.uncachedInputTokens += Math.max(0, prompt - cacheRead)
        t.cacheReadTokens += cacheRead
        t.cacheWriteTokens += cacheWrite
        t.outputTokens += Number(u.outputTokens ?? 0)
        t.reasoningTokens += Number(u.reasoningTokens ?? 0)
        activeSessions.add(file)
        const day = dayKey(event.time)
        let dayRow = daily.get(day)
        if (dayRow === undefined) {
          dayRow = { date: day, total: 0, models: new Map() }
          daily.set(day, dayRow)
        }
        const usageTotal = Math.max(0, prompt - cacheRead) + cacheRead + cacheWrite + Number(u.outputTokens ?? 0)
        dayRow.total += usageTotal
        dayRow.models.set(model, (dayRow.models.get(model) ?? 0) + usageTotal)
      }
    }
    for (const key of seen) totals.get(key)?.sessions.add(file)
  }
  const rows = [...totals.values()].map((t) => {
    const row = {
      ...t, sessions: t.sessions.size,
      billedInputTokens: t.uncachedInputTokens + t.cacheReadTokens + t.cacheWriteTokens,
      totalTokens: t.uncachedInputTokens + t.cacheReadTokens + t.cacheWriteTokens + t.outputTokens,
    }
    const price = MODEL_PRICING[t.model]
    if (price !== undefined) {
      row.cost = (t.uncachedInputTokens / 1e6) * price.input
        + (t.cacheReadTokens / 1e6) * price.cacheRead
        + (t.outputTokens / 1e6) * price.output
      row.quota = price.quota
      row.remaining = price.quota - row.cost
    }
    return row
  }).sort((a, b) => b.totalTokens - a.totalTokens)
  let billedInputTokens = 0
  let outputTokens = 0
  let totalTokens = 0
  let cost = 0
  for (const r of rows) {
    billedInputTokens += r.billedInputTokens
    outputTokens += r.outputTokens
    totalTokens += r.totalTokens
    if (typeof r.cost === 'number') cost += r.cost
  }
  const dailyRows = [...daily.values()].sort((a, b) => a.date.localeCompare(b.date)).map((d) => ({
    date: d.date,
    total: d.total,
    models: Object.fromEntries([...d.models.entries()].sort((a, b) => b[1] - a[1])),
  }))
  const activeDays = dailyRows.length
  const localToday = dayKey(Date.now())
  let streak = 0
  const cursor = new Date(); cursor.setHours(0, 0, 0, 0)
  if (!daily.has(localToday)) cursor.setDate(cursor.getDate() - 1)
  for (;;) {
    const key = dayKey(cursor.getTime())
    if (daily.has(key)) {
      streak += 1
      cursor.setDate(cursor.getDate() - 1)
    } else {
      break
    }
  }
  const top = rows[0]
  const topShare = top === undefined || totalTokens === 0 ? 0 : top.totalTokens / totalTokens
  const summary = {
    totalTokens,
    cost,
    sessionCount: activeSessions.size,
    messageCount,
    activeDays,
    currentStreakDays: streak,
    topModel: top?.model ?? null,
    topModelShare: topShare,
  }
  return { generatedAt: new Date().toISOString(), range: range ?? 'all', from: window?.from, to: window?.to, files, skipped, rows, daily: dailyRows, summary, totals: { billedInputTokens, outputTokens, totalTokens, cost } }
}

const HTML = `<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token 使用统计 · DeepSeek Harness</title>
<style>
:root{--dsw-alias-label-primary:#1a1a1a;--dsw-alias-label-secondary:#667085;--dsw-alias-border-l3:#e4e7ec;--dsw-alias-bg-layer-2:#fff;--dsw-static-blue-450:#2563eb;--dsw-static-neutral-bluish-400:#94a3b8}body{font-family:system-ui,'Segoe UI',sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}
header{background:#101828;color:#fff;padding:14px 24px;display:flex;align-items:center;gap:14px}
header h1{font-size:17px;margin:0;font-weight:600}
header span{color:#98a2b3;font-size:12px}
main{max-width:1180px;margin:18px auto;padding:0 16px}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0 14px}
select,button{background:#fff;color:#1a1a1a;border:1px solid #e4e7ec;border-radius:8px;padding:6px 10px;font-size:12px;cursor:pointer}
button:hover{background:#f2f4f7}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-bottom:14px}
.card{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:12px 14px;min-width:0}
.card b{font-size:19px;display:block;margin-bottom:3px;white-space:nowrap}.card-inline{grid-column:span 2;display:flex;align-items:center;justify-content:space-between;gap:10px}.card-inline b{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card small{color:#667085;font-size:12px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;align-items:stretch}
.panel{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:10px 12px;min-width:0;height:230px;display:flex;flex-direction:column}
.panel h3{margin:0 0 10px;font-size:13px;font-weight:600}
.piebox{display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap;flex:1;min-height:0}.barCol{flex:1;min-width:0;display:flex;flex-direction:column}.barStack{flex:1;min-height:0;position:relative;border-radius:4px;overflow:hidden;width:50%;margin:0 auto}.barSeg{position:absolute;left:0;right:0}.barLabel{font-size:14px;line-height:22px;font-weight:500;color:#1a1a1a;text-align:center;margin-top:6px;white-space:nowrap}.barLabelHidden{visibility:hidden}
.legend{display:flex;flex-direction:column;gap:8px;min-width:0;flex:1}
.legend div{display:flex;flex-direction:column;gap:2px;font-size:12px;min-width:0}.legend div span:first-child{min-width:0;overflow-wrap:anywhere}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e4e7ec;border-radius:10px;overflow:hidden;font-size:11px;table-layout:fixed}
th:nth-child(1),td:nth-child(1){width:24%}th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){width:7%}th:nth-child(4),td:nth-child(4),th:nth-child(5),td:nth-child(5){width:12%}th:nth-child(6),td:nth-child(6),th:nth-child(7),td:nth-child(7){width:9%}th:nth-child(8),td:nth-child(8){width:20%;text-align:center}th,td{padding:6px 8px;text-align:right;font-size:11px;border-bottom:1px solid #eef1f4;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}th{background:#f9fafb;color:#475467;font-weight:500}td:first-child{text-align:left;white-space:normal;overflow:visible;text-overflow:clip;word-break:break-word}
</style>
<header><h1>📊 Token 使用统计</h1><span>本地会话聚合 · 每 10 秒自动刷新</span></header>
<main>
<div class="bar">
<select id="range" onchange="load()"><option value="all">全部</option><option value="24h">24小时</option><option value="yesterday">昨天</option><option value="7d">近7天</option><option value="30d">近30天</option></select>
<button onclick="load()">刷新</button><span id="meta" style="color:#667085;font-size:12px"></span>
</div>
<div class="cards" id="cards"></div>
<div class="charts">
<div class="panel"><h3>按天 Token 趋势</h3><div id="bar" style="flex:1;min-height:0;display:flex;gap:6px;align-items:stretch"></div></div>
<div class="panel"><h3>模型用量占比</h3><div class="piebox"><svg id="pie" width="160" height="160"></svg><div class="legend" id="legend"></div></div></div>
</div>
<table id="tbl"></table>
</main>
<script>
const fmt=n=>{if(n>=1e9)return (n/1e9).toFixed(2)+'B';if(n>=1e6)return (n/1e6).toFixed(2)+'M';if(n>=1e3)return (n/1e3).toFixed(1)+'K';return String(n)};
const big=n=>{if(n>=1e8)return (n/1e8).toFixed(1)+'亿';if(n>=1e4)return (n/1e4).toFixed(1)+'万';return fmt(n)};
const money=n=>n<.01?'$'+n.toFixed(4):'$'+n.toFixed(2);
const colors={'deepseek-v4-flash':'var(--dsw-static-blue-450)','deepseek-v4-pro':'#a78bfa'};
let lastData=null;
const colorFor=(m,i)=>colors[m]||['#94a3b8','#0ea5e9','#f59e0b','#10b981'][i%4];
function barChart(d){
 const box=document.getElementById('bar');const days=d.daily.slice(-31);if(!days.length){box.innerHTML='';return}
 const max=Math.max(...days.map(x=>x.total),1);const every=Math.max(1,Math.ceil(days.length/8));
 box.innerHTML=days.map((day,i)=>{
  const models=Object.entries(day.models||{}).sort((a,b)=>a[0].localeCompare(b[0])).reverse();let bottom=0;
  const segs=models.map(([model,tokens])=>{const h=tokens/max*100;const html='<div class="barSeg" style="bottom:'+bottom.toFixed(2)+'%;height:'+h.toFixed(2)+'%;background:'+colorFor(model,Object.keys(day.models).indexOf(model))+'"></div>';bottom+=h;return html}).join('');
  const show=i%every===0||i===days.length-1;
  return '<div class="barCol"><div class="barStack">'+segs+'</div><div class="barLabel'+(show?'':' barLabelHidden')+'">'+day.date.slice(5).replace('-','/')+'</div></div>';
 }).join('');
}
function pieChart(d){
 const svg=document.getElementById('pie');const cx=80,cy=80,r=66,ir=46;const rows=d.rows.filter(r=>r.totalTokens>0);const total=rows.reduce((a,r)=>a+r.totalTokens,0)||1;
 let start=-Math.PI/2,html='';
 for(const [i,r] of rows.entries()){const ang=r.totalTokens/total*Math.PI*2;const end=start+ang;const x1=cx+r*Math.cos(start),y1=cy+r*Math.sin(start),x2=cx+r*Math.cos(end),y2=cy+r*Math.sin(end);const large=ang>Math.PI?1:0;html+='<path d="M'+cx+','+cy+' L'+x1+','+y1+' A'+r+','+r+' 0 '+large+' 1 '+x2+','+y2+' Z" fill="'+colorFor(r.model,i)+'"/>';start=end;}
 html+='<circle cx="'+cx+'" cy="'+cy+'" r="'+ir+'" fill="var(--dsw-alias-bg-layer-2)"/>';html+='<text x="'+cx+'" y="'+(cy-4)+'" text-anchor="middle" font-size="13" font-weight="600" fill="var(--dsw-alias-label-primary)">'+big(total)+'</text><text x="'+cx+'" y="'+(cy+13)+'" text-anchor="middle" font-size="10" fill="var(--dsw-alias-label-secondary)">tokens</text>';
 document.getElementById('legend').innerHTML=rows.map((r,i)=>'<div><span><span class="dot" style="background:'+colorFor(r.model,i)+'"></span>'+r.model+'</span><span>'+big(r.totalTokens)+' · '+((r.totalTokens/total)*100).toFixed(1)+'%</span></div>').join('');
 svg.innerHTML=html;
}
async function load(){
 const range=document.getElementById('range').value;const d=await (await fetch('/token-usage.json?range='+range,{cache:'no-store'})).json();
 const s=d.summary||{};const t=d.totals;
 document.getElementById('meta').textContent='更新于 '+new Date(d.generatedAt).toLocaleTimeString()+' · '+document.getElementById('range').selectedOptions[0].text+' · 扫描 '+d.files+' 个会话';
 const cards=[['tokens 用量',big(s.totalTokens??t.totalTokens),false],['总计费用',money(s.cost??0),false],['会话数量',s.sessionCount??0,false],['消息数量',s.messageCount??0,false],['活跃天数',s.activeDays??0,false],['当前连续天数',s.currentStreakDays??0,false],['最常用模型',s.topModel? s.topModel+' · '+((s.topModelShare??0)*100).toFixed(1)+'%':'—',true]];
 document.getElementById('cards').innerHTML=cards.map(x=>x[2]?'<div class="card card-inline"><small>'+x[0]+'</small><b>'+x[1]+'</b></div>':'<div class="card"><b>'+x[1]+'</b><small>'+x[0]+'</small></div>').join('');
 document.getElementById('tbl').innerHTML='<tr>'+['模型','会话','请求','输入','读缓存','输出','推理','总计费用'].map(x=>'<th>'+x+'</th>').join('')+'</tr>'+d.rows.map(r=>'<tr><td>'+r.model+'</td><td>'+r.sessions+'</td><td>'+r.requests+'</td><td>'+fmt(r.uncachedInputTokens)+'</td><td>'+fmt(r.cacheReadTokens)+'</td><td>'+fmt(r.outputTokens)+'</td><td>'+fmt(r.reasoningTokens)+'</td><td style="text-align:center">'+big(r.totalTokens)+(r.cost===undefined?'':'<br>'+money(r.cost))+'</td></tr>').join('');
 lastData=d;barChart(d);pieChart(d);
}
load();setInterval(load,10000);window.addEventListener('resize',()=>{if(lastData)barChart(lastData)});
</script>
</main>
`

export function apply(ctx) {
  ctx.webServer.register({
    kind: 'exact',
    path: '/token-usage.json',
    handler: async (req, res) => {
      try {
        const range = new URL(req.url ?? '/', 'http://x').searchParams.get('range')
        const report = tokenUsageReport(range)
        res.writeHead(200, { 'content-type': 'application/json; charset=utf-8' })
        res.end(JSON.stringify(report, null, 2))
      } catch (error) {
        res.writeHead(500, { 'content-type': 'text/plain; charset=utf-8' })
        res.end(`token-usage report failed: ${String(error && error.message || error)}`)
      }
    },
  })
  ctx.webServer.register({
    kind: 'exact',
    path: '/token-usage',
    handler: async (_req, res) => {
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      res.end(HTML)
    },
  })
}
