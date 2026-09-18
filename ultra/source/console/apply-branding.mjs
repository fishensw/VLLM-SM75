import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const assets = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "branding",
);
export function applyBranding(base = "/opt/harness/node_modules") {
  const pkg = (name) => path.join(base, "@deepseek-ai", name);
  const replace = (file, before, after) => {
    const text = fs
      .readFileSync(file, "utf8")
      .replaceAll("VLLM-SM75 模型工作台", "工作台")
      .replaceAll("VLLM-SM75 Workbench", "Workbench");
    if (!text.includes(before) && !text.includes(after))
      throw Error("品牌适配位置发生变化: " + file);
    fs.writeFileSync(file, text.replaceAll(before, after));
  };
  const dist = path.join(pkg("dsh-web-frontend"), "dist");
  replace(
    path.join(dist, "index.html"),
    "<title>DeepSeek Harness</title>",
    "<title>工作台</title>",
  );
  fs.copyFileSync(
    path.join(assets, "favicon.svg"),
    path.join(dist, "favicon.svg"),
  );
  const index = path.join(dist, "index.html");
  fs.writeFileSync(
    index,
    fs
      .readFileSync(index, "utf8")
      .replace(
        /(?:\/brand\/|\.\/|\/)?favicon\.svg(?:\?v=[^"'\s>]*)?/g,
        "/brand/favicon.svg",
      ),
  );
  const file = path.join(dist, "manifest.webmanifest"),
    manifest = JSON.parse(fs.readFileSync(file));
  manifest.name = "工作台";
  manifest.short_name = "工作台";
  manifest.icons = [
    { src: "/brand/icon-192.png", sizes: "192x192", type: "image/png" },
    { src: "/brand/icon-512.png", sizes: "512x512", type: "image/png" },
    {
      src: "/brand/maskable-512.png",
      sizes: "512x512",
      type: "image/png",
      purpose: "maskable",
    },
  ];
  fs.writeFileSync(file, JSON.stringify(manifest, null, 2));
  fs.copyFileSync(
    path.join(assets, "client.js"),
    path.join(pkg("dsh-client-ui-brand-official"), "lib/client.js"),
  );
  replace(
    path.join(pkg("dsh-client-ui-layout"), "lib/client.js"),
    'const productTitle = "DeepSeek Harness"',
    'const productTitle = "工作台"',
  );
  replace(
    path.join(pkg("dsh-client-ui-chat"), "lib/client.js"),
    "if (stats.steps === 0 && !hasTokens) return null;",
    'if (stats.steps === 0 && !hasTokens) return (0, react_jsx_runtime.jsx)("div", {className: StatsPills_module_css_default.root, "data-composer-stats": true, children: "用量与耗时统计中 · 首个步骤结束后显示"});',
  );
  const conversation = path.join(
    pkg("dsh-client-ui-conversation"),
    "lib/client.js",
  );
  replace(
    conversation,
    '"hero.headline": "探索未至之境"',
    '"hero.headline": "工作台"',
  );
  replace(
    conversation,
    '"hero.headline": "Into the Unknown"',
    '"hero.headline": "Workbench"',
  );
  replace(
    conversation,
    '"hero.preview": "预览版"',
    '"hero.preview": "本地开发版"',
  );
  const welcome = path.join(
    pkg("dsh-client-ui-settings-models"),
    "lib/client.js",
  );
  replace(welcome, 'welcomeTitle: "内测声明"', 'welcomeTitle: "工作台"');
  replace(
    welcome,
    'welcomeTitle: "Internal Testing Notice"',
    'welcomeTitle: "Workbench"',
  );
  const copy = fs
    .readFileSync(welcome, "utf8")
    .replace(
      /welcomeBody: "(?:[^"\\]|\\.)*"/g,
      (match) =>
        "welcomeBody: " +
        JSON.stringify(
          /[\u4e00-\u9fff]/.test(match)
            ? "下载或导入模型，选择场景参数，即可启动推理、对话与任务。性能、用量和测试统一在工作台查看。Chat / Agent 功能基于 DeepSeek Harness。"
            : "Download or import a model, choose a configuration, and start inference, chat or tasks. Monitor performance, usage and tests in one workbench. Chat and Agent capabilities are powered by DeepSeek Harness.",
        ),
    );
  const welcomeRegistration =
    /ctx\.slots\.inject\("settings\.onboarding", \(\) => ctx\.slots\.register\(\{\s*name: "settings\.onboarding",\s*id: "welcome-notice",[\s\S]*?\}, WelcomeNotice\)\);/;
  const noWelcome =
    "// SM75: enter the application directly; no welcome notice.";
  if (!welcomeRegistration.test(copy) && !copy.includes(noWelcome))
    throw Error("欢迎弹窗注册位置发生变化");
  fs.writeFileSync(welcome, copy.replace(welcomeRegistration, noWelcome));

  const llm = path.join(pkg("dsh-llm-pi-ai"), "lib/index.js");
  let llmText = fs.readFileSync(llm, "utf8");
  if (!llmText.includes("import {dshSampling}"))
    llmText =
      "import {dshSampling} from '/opt/sm75-workbench/console/sampling.mjs';\n" +
      llmText;
  const marker =
    "...options.sessionId === void 0 ? {} : { sessionId: String(options.sessionId) },";
  if (!llmText.includes("...dshSampling(options.provider, model)")) {
    if (!llmText.includes(marker)) throw Error("DSH 生成参数适配位置变化");
    llmText = llmText.replace(
      marker,
      marker + "\n ...dshSampling(options.provider, model),",
    );
  }
  fs.writeFileSync(llm, llmText);
  const sidebar = path.join(pkg("dsh-client-ui-sidebar"), "lib/client.js");
  let nav = fs.readFileSync(sidebar, "utf8");
  if (!nav.includes("const inWorkbench ="))
    nav = nav.replace(
      "const panels = usePanels((snapshot) => snapshot);",
      'const panels = usePanels((snapshot) => snapshot);\n const inWorkbench = usePanelInfo(info => info.activePanelId === null || info.activePanelId === "conversation");',
    );
  nav = nav.replace(
    'children: renderSlot("sidebar.workspaces", {',
    'children: inWorkbench && renderSlot("sidebar.workspaces", {',
  );
  if (!nav.includes('(id === "conversation" ?'))
    nav = nav.replace(
      "info.activePanelId === id",
      '(id === "conversation" ? info.activePanelId === null || info.activePanelId === id : info.activePanelId === id)',
    );
  fs.writeFileSync(sidebar, nav);
  const settingsClient = path.join(
    pkg("dsh-client-ui-settings"),
    "lib/client.js",
  );
  fs.writeFileSync(
    settingsClient,
    fs
      .readFileSync(settingsClient, "utf8")
      .replace(
        'const persistence = ctx.remote.$host.isLoopback ? \"host\" : \"memory\";',
        'const persistence = (globalThis.location?.pathname.startsWith(\"/dsh/\") || ctx.remote.$host.isLoopback) ? \"host\" : \"memory\";',
      ),
  );
  const general = path.join(
    pkg("dsh-client-ui-settings-general"),
    "lib/client.js",
  );
  let generalText = fs
    .readFileSync(general, "utf8")
    .replaceAll("系统及设置", "设置");
  fs.writeFileSync(general, generalText);
  let hero = fs
    .readFileSync(conversation, "utf8")
    .replace(
      'children: t("hero.preview")',
      'style: {display:"none"}, children: ""',
    );
  fs.writeFileSync(conversation, hero);
}
if (process.argv[1] === fileURLToPath(import.meta.url))
  applyBranding(process.argv[2]);
