// Use the native provider's canonical targets and containment semantics. This
// read-only helper never starts a shell or changes the active sandbox policy.
export async function listWorkspaceDirectory(fs, args, exec) {
  const fail = (message, code = 'FS_SANDBOX_DENIED') => {throw Object.assign(new Error(message), {code});};
  const cwd = exec.agent?.session?.header?.cwd;
  if (typeof cwd !== 'string' || !cwd.trim()) fail('请先选择当前会话的工作区，再列出目录。');
  const requested = args.path ?? '.';
  const offset = args.offset ?? 0, limit = args.limit ?? 100;
  if (typeof requested !== 'string' || !requested.trim()) fail('目录路径不能为空。', 'FS_NOT_FOUND');
  if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 200)
    fail('offset 必须是非负整数，limit 必须是 1 到 200 的整数。', 'FS_IO_ERROR');
  exec.signal?.throwIfAborted();
  const root = await fs.resolve(cwd, {signal: exec.signal});
  const target = await fs.resolve(requested, {cwd, signal: exec.signal});
  if (!fs.contains(root, target)) fail('只能列出当前工作区内的目录；该路径或符号链接指向工作区之外。');
  const all = await fs.listDir(target, exec.signal);
  exec.signal?.throwIfAborted();
  // Outside links may be named, but do not expose their resolved paths, types,
  // sizes or contents. Following one is rejected by the same containment check.
  const entries = all.slice(offset, offset + limit).map(entry => fs.contains(root, entry.target)
    ? {name: entry.name, type: entry.type, ...(Number.isFinite(entry.size) ? {size: entry.size} : {})}
    : {name: entry.name, type: 'outside-link'});
  return {path: target.displayPath, entries, total: all.length, offset, hasMore: offset + entries.length < all.length};
}

export function renderDirectory(value) {
  const labels = {directory: '目录', file: '文件', other: '其他', 'outside-link': '工作区外链接'};
  const rows = value.entries.map(entry => `${labels[entry.type] || '其他'}\t${JSON.stringify(entry.name)}`);
  return [{type: 'text', text: `目录：${value.path}\n${rows.join('\n') || '（无条目）'}\n共 ${value.total} 项，当前从第 ${value.offset + 1} 项显示。${value.hasMore ? ` 后续请使用 offset=${value.offset + value.entries.length}。` : ''}`}];
}
