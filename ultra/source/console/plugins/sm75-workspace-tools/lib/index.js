import {defineTool} from '@deepseek-ai/dsh-tools';
import {FsError} from '@deepseek-ai/dsh-fs';
import {listWorkspaceDirectory, renderDirectory} from './directory.js';

export const inject = ['fs', 'tools', 'systemPrompt'];
export function apply(ctx) {
  ctx.tools.register(defineTool({
    name: 'list_directory',
    description: '列出当前会话工作区内目录的直属条目，包含文件、子目录和空目录。无需命令行沙盒；不读取文件内容，不遍历工作区之外。',
    parameters: {
      path: {type: 'string', description: '相对当前工作区的目录路径，默认 .。'},
      offset: {type: 'integer', description: '分页起点，从 0 开始，默认 0。'},
      limit: {type: 'integer', description: '本页最多条目数，1–200，默认 100。'},
    },
    timeoutMs: 30000,
    isConcurrencySafe: () => true,
    output: {
      schema: {type: 'object', additionalProperties: false, properties: {
        path: {type: 'string', required: true},
        entries: {type: 'array', required: true, items: {type: 'object', additionalProperties: false, properties: {
          name: {type: 'string', required: true}, type: {type: 'string', required: true}, size: {type: 'number'},
        }}},
        total: {type: 'integer', required: true}, offset: {type: 'integer', required: true}, hasMore: {type: 'boolean', required: true},
      }},
      render: (_args, value) => renderDirectory(value),
    },
    async execute(args, exec) {
      try {return await listWorkspaceDirectory(ctx.fs, args, exec);}
      catch (error) {
        if (error instanceof FsError || !error.code?.startsWith('FS_')) throw error;
        throw new FsError(error.message, error.code, {cause: error});
      }
    },
  }));
  ctx.systemPrompt.section({
    name: 'sm75:workspace-tools', order: ctx.systemPrompt.getSectionOrder('TOOL_GLOB') - 1,
    text: ({scope}) => ctx.tools.get('list_directory', scope) === undefined ? '' :
      '工作区目录列表优先使用 list_directory；它包含空目录且不依赖命令行沙盒。文件查找使用 glob，内容搜索使用 grep，读取文件使用 read。' +
      '除非用户明确要求其他语言，回复、进度和错误说明均使用简体中文。' +
      '若命令工具报告 SANDBOX_UNAVAILABLE、missing Linux sandbox runtime 或 sandbox runner 不可用，请说明“当前系统缺少可用的命令执行沙盒，暂时不能运行该命令”，继续用可用的原生文件工具完成文件操作。' +
      '这不是工作区访问权限不足；不要为绕过缺失的沙盒而请求扩大工作区权限、切换 danger-full-access 或关闭安全限制，也不要声称命令或测试已成功执行。',
  });
}
