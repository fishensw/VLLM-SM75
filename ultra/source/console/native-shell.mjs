import fs from "node:fs";
import path from "node:path";

// Fixed-release adapter: retain each owner's service, store and child-slot
// declarations; replace only visual shell components at render time.
export function applyUnifiedShell(base) {
  for (const [pkg, name, renderer] of [
    ["dsh-client-ui-layout", "AppFrame", "frame"],
    ["dsh-client-ui-sidebar", "SidebarRoot", "sidebar"],
    ["dsh-client-ui-conversation", "ConversationHeader", "header"],
    ["dsh-client-ui-settings-general", "SettingsRoot", "settings"],
    ["dsh-client-ui-settings-general", "GeneralSection", "generalSettings"],
  ]) {
    const file = path.join(base, "@deepseek-ai", pkg, "lib/client.js");
    let source = fs.readFileSync(file, "utf8");
    const marker = "/* sm75-unified-shell:" + name + " */";
    if (source.includes(marker)) continue;
    const anchor = "function " + name + "(";
    if (source.split(anchor).length !== 2) throw Error("Harness unified shell anchor changed: " + name);
    source = source.replace(anchor, marker + "\nfunction " + name + "(props) {\n"
      + "  const shell = globalThis.__SM75_NATIVE_SHELL__;\n"
      + "  if (!shell) throw Error('SM75 unified shell is not ready');\n"
      + "  return shell." + renderer + "(props, Native" + name + ");\n}\nfunction Native" + name + "(");
    fs.writeFileSync(file, source);
  }
  // Read the owner's live store before issuing panel navigation: reselecting
  // the already-active conversation must not abort initial workspace restore.
  const layoutFile=path.join(base,"@deepseek-ai/dsh-client-ui-layout","lib/client.js");
  let layoutSource=fs.readFileSync(layoutFile,"utf8");
  if (!layoutSource.includes("/* sm75-active-panel-reader */")) {
    const anchor=/(const layout = new LayoutController\(instance\.actions,[^\n]+\);)/;
    if ([...layoutSource.matchAll(new RegExp(anchor.source,"g"))].length!==1)
      throw Error("Harness panel reader anchor changed");
    layoutSource=layoutSource.replace(anchor,"$1\n/* sm75-active-panel-reader */\nlayout.sm75ActivePanel = () => instance.getSnapshot().panelInfo.activePanelId;");
    fs.writeFileSync(layoutFile,layoutSource);
  }
  // This native control now scales the entire shared interface. Keep its
  // original settings store and actions, but describe the actual scope.
  const themeFile=path.join(base,"@deepseek-ai/dsh-client-ui-theme","lib/client.js");
  let theme=fs.readFileSync(themeFile,"utf8");
  for (const [pattern, value] of [
    [/("fontSize\.title"\s*:\s*)"(?:字号大小|全局字号)"/g,"全局字号"],
    [/("fontSize\.description"\s*:\s*)"(?:仅影响会话内容的字号|调整工作台与会话的字号|统一调整侧栏、表单、会话与使用统计的字号)"/g,"统一调整侧栏、表单、会话与使用统计的字号"],
    [/("fontSize\.title"\s*:\s*)"(?:Font size|Global font size)"/g,"Global font size"],
    [/("fontSize\.description"\s*:\s*)"(?:Only affects conversation content|Adjusts workbench and conversation text|Adjusts sidebar, forms, conversations and usage statistics)"/g,"Adjusts sidebar, forms, conversations and usage statistics"],
  ]) {
    if ([...theme.matchAll(pattern)].length!==1) throw Error("Harness global font copy anchor changed");
    theme=theme.replace(pattern, (_match, prefix)=>prefix+JSON.stringify(value));
  }
  // Older local bundles widened this UI-only range. Use the upstream 12..17
  // control and validation range; profile values are never rewritten here.
  theme=theme.replaceAll("disabled: fontSize >= 26","disabled: fontSize >= 17")
    .replaceAll("px > 26","px > 17")
    .replaceAll("outside 12..26","outside 12..17")
    .replaceAll("[FONT_SIZE_FIELD]: Schema.number().step(1).min(12).max(26)","[FONT_SIZE_FIELD]: Schema.number().step(1).min(12).max(17)")
    .replaceAll("parsed >= 12 && parsed <= 26","parsed >= 12 && parsed <= 17");
  const fontWriteMarker="/* sm75-font-write-settlement */";
  if (!theme.includes(fontWriteMarker)) {
    const write="this.host.set(FONT_SIZE_FIELD, px);";
    const adopt=/(const section = this\.host\.getSnapshot\(\)\.value;\s*if \(section === void 0\) return;)/;
    for (const needle of [write,"this.fontSize === section.fontSize","this.fontSize = section.fontSize;"])
      if (theme.split(needle).length!==2) throw Error("Harness font settlement anchor changed");
    if (!adopt.test(theme)) throw Error("Harness font adoption anchor changed");
    theme=theme.replace(write, fontWriteMarker+`
        const generation = this.sm75FontGeneration = (this.sm75FontGeneration || 0) + 1;
        this.sm75FontPending = true;
        const finish = () => {
          if (generation !== this.sm75FontGeneration) return;
          this.sm75FontPending = false;
          this.adopt();
        };
        this.sm75FontWrite = Promise.resolve(this.host.set(FONT_SIZE_FIELD, px)).then(
          accepted => { finish(); return accepted; },
          error => { finish(); this.ctx.logger?.warn?.("Unable to save global font size", error); return false; }
        );`)
      .replace(adopt,"$1\n        const fontSize = this.sm75FontPending ? this.fontSize : section.fontSize;")
      .replace("this.fontSize === section.fontSize","this.fontSize === fontSize")
      .replace("this.fontSize = section.fontSize;","this.fontSize = fontSize;");
  }
  fs.writeFileSync(themeFile,theme);
  const file=path.join(base,"@deepseek-ai/dsh-client-ui-open-in-app","lib/client.js");
  let source=fs.readFileSync(file,"utf8");
  const route='"/open-in-app/';
  if (!source.includes('"/harness-ui/open-in-app/')) {
    if (!source.includes(route)) throw Error("Harness open-in-app route anchor changed");
    source=source.replaceAll(route,'"/harness-ui/open-in-app/');
    fs.writeFileSync(file,source);
  }

}
