import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const supportedHarnessVersion = "0.1.7-alpha.2";

export function harnessVersion(base = "/opt/harness/node_modules") {
  try { return JSON.parse(fs.readFileSync(path.join(base, "@deepseek-ai/dsh/package.json"), "utf8")).version; }
  catch { return null; }
}

export function assertHarnessVersion(base) {
  const version = harnessVersion(base);
  if (version !== supportedHarnessVersion)
    throw Error(`Harness adaptation requires ${supportedHarnessVersion}; installed ${version ?? "unknown"}`);
}

export function applyBranding(base = "/opt/harness/node_modules", consoleRoot = here) {
  assertHarnessVersion(base);
  const pkg = (name) => path.join(base, "@deepseek-ai", name);
  for (const name of ["dsh-client-ui-layout", "dsh-llm-pi-ai"]) {
    const version = JSON.parse(fs.readFileSync(path.join(pkg(name), "package.json"), "utf8")).version;
    if (version !== supportedHarnessVersion)
      throw Error(`Harness ${name} adaptation requires ${supportedHarnessVersion}; installed ${version}`);
  }
  const dist = path.join(pkg("dsh-web-frontend"), "dist");
  const index = path.join(dist, "index.html");
  let html = fs.readFileSync(index, "utf8");
  if (!/<title>(DeepSeek Harness|工作台)<\/title>/.test(html))
    throw Error("Harness HTML title anchor changed");
  html = html.replace(/<title>(DeepSeek Harness|工作台)<\/title>/, "<title>工作台</title>")
    .replace(/(?:\/brand\/|\.\/|\/)?favicon\.svg(?:\?v=[^"'\s>]*)?/g, "/brand/favicon.svg");
  fs.writeFileSync(index, html);
  fs.copyFileSync(path.join(consoleRoot, "branding/favicon.svg"), path.join(dist, "favicon.svg"));
  const manifestFile = path.join(dist, "manifest.webmanifest");
  const manifest = JSON.parse(fs.readFileSync(manifestFile, "utf8"));
  manifest.name = manifest.short_name = "工作台";
  manifest.icons = [
    { src: "/brand/icon-192.png", sizes: "192x192", type: "image/png" },
    { src: "/brand/icon-512.png", sizes: "512x512", type: "image/png" },
    { src: "/brand/maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
  ];
  fs.writeFileSync(manifestFile, JSON.stringify(manifest, null, 2));

  // The official compiled layout bakes DSH_CLIENT_TITLE into one constant.
  // Replace only that product name; keep DocumentTitle's dynamic session title
  // and every other native layout feature. Refuse changed or ambiguous output.
  const layout = path.join(pkg("dsh-client-ui-layout"), "lib/client.js");
  const nativeTitle = 'const productTitle = "DeepSeek Harness"';
  const localTitle = 'const productTitle = "工作台"';
  const layoutSource = fs.readFileSync(layout, "utf8");
  if (layoutSource.split(nativeTitle).length + layoutSource.split(localTitle).length !== 3)
    throw Error("Harness layout product title anchor changed or is ambiguous");
  fs.writeFileSync(layout, layoutSource.replace(nativeTitle, localTitle));

  // One additional server-side sampling seam. No official bundle is replaced;
  // only the checked title constant above and the call-site below are patched.
  // pi-ai exposes onPayload, but Harness has no provider setting for the extended
  // OpenAI sampling fields. Verify the exact release and injection point at build time.
  const llm = path.join(pkg("dsh-llm-pi-ai"), "lib/index.js");
  let source = fs.readFileSync(llm, "utf8");
  const injection = "...dshSampling(options.provider, model),";
  const marker = "...options.sessionId === void 0 ? {} : { sessionId: String(options.sessionId) },";
  const samplingImport = `import {dshSampling} from ${JSON.stringify(pathToFileURL(path.join(consoleRoot, "sampling.mjs")).href)};\n`;
  if (!source.includes(injection)) {
    if (source.split(marker).length !== 2) throw Error("Harness sampling adapter anchor changed");
    source = source.replace(marker, marker + "\n        " + injection);
    source = samplingImport + source;
  } else {
    // Earlier candidates wrote an OS path. Normalize only our own first-line
    // import so reinstall also repairs Windows drive/UNC paths and relocation.
    const ownImport = /^import \{dshSampling\} from "(?:[^"\\\r\n]|\\[^\r\n])*";\r?\n/;
    if (!ownImport.test(source)) throw Error("Harness sampling adapter import anchor changed");
    source = source.replace(ownImport, () => samplingImport);
  }
  fs.writeFileSync(llm, source);
}

if (process.argv[1] && fs.existsSync(process.argv[1]) && fs.realpathSync(process.argv[1]) === fileURLToPath(import.meta.url))
  applyBranding(process.argv[2]);
