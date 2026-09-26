import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

export async function initializeNext(root, consoleDir, selected) {
  const {Store} = await import(pathToFileURL(path.join(consoleDir, 'store.mjs')));
  const {validateProfile} = await import(pathToFileURL(path.join(consoleDir, 'config.mjs')));
  fs.mkdirSync(root, {recursive:true, mode:0o700});
  const marker = path.join(root, 'next-initialized.json');
  if (fs.existsSync(marker)) return;
  const existing = fs.readdirSync(root).filter(n => n !== 'cache');
  if (existing.length) throw Error('Next initialization requires a dedicated empty console directory; existing data will not be overwritten');
  if (selected.args[0] !== '--model') throw Error('Unexpected selected-config model layout');
  const profile = validateProfile({
    id:'flash-next-tp8', name:'Flash-Next TP8 · MTP5 · 256K · 并发4',
    backend:'native', format:'awq', port:8000, cacheRoot:'/data/cache',
    args:selected.args.slice(1),
    env:{OMP_NUM_THREADS:'1', VLLM_SM75_QWEN38_HC_GEMV:'1', VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:'134217728', VLLM_MONITOR:'0'},
    power:{mode:'sleep', gpus:'0,1,2,3,4,5,6,7'},
  });
  const store = new Store(root);
  try {
    fs.writeFileSync(path.join(root, 'profiles.json'), JSON.stringify([profile], null, 2), {flag:'wx',mode:0o600});
    store.put('settings','main', {...store.settings(), modelRoot:'/models', cacheRoot:'/data/cache',
      defaultProfile:profile.id, defaultChatProfile:profile.id, autoStart:true});
    fs.writeFileSync(marker, JSON.stringify({profile:profile.id}), {flag:'wx',mode:0o600});
  } finally {store.close();}
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const consoleDir='/opt/sm75-workbench/console';
  process.env.SM75_CONSOLE_ROOT ||= '/data';
  process.env.SM75_SINGLE_CONTAINER='1';
  process.env.NCCL_P2P_LEVEL ??= 'SYS';
  process.env.SM75_CONSOLE_HOST ||= '0.0.0.0';
  process.env.SM75_CONSOLE_PORT ||= '1615';
  await initializeNext(process.env.SM75_CONSOLE_ROOT, consoleDir,
    JSON.parse(fs.readFileSync('/opt/flashnext/selected-config.json','utf8')));
  await import(pathToFileURL(path.join(consoleDir,'server.mjs')));
}
