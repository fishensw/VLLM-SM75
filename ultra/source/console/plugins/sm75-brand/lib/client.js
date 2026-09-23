window.__ModuleLoader__.load({
  id: "sm75-brand",
  factory: (require) => {
    const React = require("react");
    const {createPortal} = require("react-dom");
    const h = React.createElement;
    let ctx, settingsPage = null, activePanel;
    const settingsListeners = new Set();
    const mountedOwners = new Set();
    function mounted(owner) {
      mountedOwners.add(owner);
      const api=globalThis.__SM75_WORKBENCH__;
      if(api && mountedOwners.has("frame") && mountedOwners.has("settings") && !api.ready) {
        api.ready=true;
        emit("sm75-native-ready");
      }
      return ()=>{
        mountedOwners.delete(owner);
        if(globalThis.__SM75_WORKBENCH__===api)api.ready=false;
      };
    }
    const settingsPages = new Set(["appearance", "assistant", "connections"]);
    const settingsTab = id => id === "general" ? "appearance" : id === "models" ? "connections" : "assistant";
    const appearanceItems = new Set(["language","appearance","font-size","transcript-view","performance-usage","current-version"]);
    const sectionVisible = (id,page) => id==="general"
      ? page==="appearance" || page==="assistant" : settingsTab(id)===page;
    const subscribeSettings = listener => {settingsListeners.add(listener);return () => settingsListeners.delete(listener);};
    const selectSettings = page => {
      if (page !== null && !settingsPages.has(page)) throw Error("Unknown workbench settings page: " + page);
      if (settingsPage === page) return;
      settingsPage = page;
      for (const listener of settingsListeners) listener();
    };
    const emit = (name, detail) => document.dispatchEvent(new CustomEvent(name, {detail}));
    const portal = (node, id) => {
      const target = document.getElementById(id);
      return target ? createPortal(node, target) : null;
    };
    function installNativeFontScale() {
      // Native module CSS is tagged by the upstream loader. Restrict this
      // compatibility layer to those sheets; console/third-party sheets keep
      // their own typography. CSS variables update typography without zoom
      // or per-element style overrides, retaining native layout measurement.
      if (!document.head) return () => {};
      const changed = new Map();
      const pixels = /^(?:\d+(?:\.\d+)?|\.\d+)px$/;
      const fontHead = /^((?:(?:normal|italic|oblique|small-caps|bold|bolder|lighter|[1-9]00)\s+)*)(\d+(?:\.\d+)?px)(?:\s*\/\s*(\d+(?:\.\d+)?px))?(?=\s|\/)/;
      const scaled = value => `calc(${value} * var(--ui-fs, 1))`;
      const ownSheet = sheet => {
        const node = sheet.ownerNode;
        if (!node) return false;
        const plugin = node.dataset?.plugin || "";
        if ((/^@deepseek-ai\/dsh-/.test(plugin) || plugin === "dsh-watcher")
          && node.dataset.pluginCss?.startsWith(plugin + "/")) return true;
        if (node.tagName === "STYLE" && node.dataset?.sm75Native === "true") return true;
        if (node.tagName === "LINK" && node.dataset?.sm75Native === "true") {
          try {
            const url = new URL(node.href, location.href);
            return url.origin === location.origin && url.pathname.startsWith("/harness-ui/");
          } catch {return false;}
        }
        return false;
      };
      const adaptStyle = style => {
        for (const property of Array.from(style)) {
          const original = style.getPropertyValue(property).trim();
          if (!original || original.includes("--ui-fs")) continue;
          let value;
          const typography = property === "font-size" || property === "line-height" || property === "font"
            || /^--dsw-font-/.test(property) || property === "--dsh-content-font-size-secondary";
          // Earlier local images already scaled typography with this token.
          // Migrate its reference, never multiply a migrated expression again.
          if (typography && original.includes("--dsh-font-scale")) {
            value = original.replaceAll("--dsh-font-scale", "--ui-fs");
          } else if (property === "font-size" || property === "line-height"
            || /^--dsw-font-.+-(?:font-size|line-height)$/.test(property)
            || property === "--dsh-content-font-size-secondary") {
            if (pixels.test(original)) value = scaled(original);
          } else if (property === "font" || /^--dsw-font-/.test(property)) {
            // Native trajectory uses font shorthand tokens. Change just the
            // size and pixel line-height, never weight, family or dynamic
            // --dsh-content-font-size expressions (already scaled upstream).
            if (fontHead.test(original)) value = original.replace(fontHead, (_match, prefix, size, line) => prefix + scaled(size) + (line ? "/" + scaled(line) : ""));
          }
          if (value === undefined) continue;
          const priority = style.getPropertyPriority(property);
          style.setProperty(property, value, priority);
          let records = changed.get(style);
          if (!records) changed.set(style, records = new Map());
          records.set(property, {original, value:style.getPropertyValue(property), priority});
        }
      };
      const adaptRules = rules => {
        for (const rule of rules) {
          if (rule.style) adaptStyle(rule.style);
          if (rule.cssRules) adaptRules(rule.cssRules);
        }
      };
      const refresh = () => {
        for (const sheet of document.styleSheets) {
          if (!ownSheet(sheet)) continue;
          try {adaptRules(sheet.cssRules);} catch { /* Non-readable optional sheets keep native styles. */ }
        }
      };
      const observer = new MutationObserver(refresh);
      observer.observe(document.head, {childList:true, subtree:true, characterData:true});
      document.head.addEventListener("load", refresh, true);
      refresh();
      return () => {
        observer.disconnect();
        document.head.removeEventListener("load", refresh, true);
        for (const [style, records] of changed) for (const [property, record] of records) {
          if (style.getPropertyValue(property) === record.value)
            style.setProperty(property, record.original, record.priority);
        }
        changed.clear();
      };
    }
    function Frame({useStore, usePanelInfo, actions, renderSlot}) {
      const layout = useStore(state => state.layoutInfo);
      const panel = usePanelInfo(info => info.activePanelId);
      activePanel = panel;
      React.useEffect(()=>mounted("frame"),[]);
      const ref = React.useRef(null);
      // Keep visited panels mounted so navigating to Watcher or plugins cannot
      // discard the conversation draft, scroll position or tool presentation.
      const activeKey = panel ?? "conversation";
      const [visited, setVisited] = React.useState(() => new Set(["conversation"]));
      React.useEffect(() => {
        setVisited(previous => previous.has(activeKey) ? previous : new Set([...previous, activeKey]));
      }, [activeKey]);
      const panelKeys = visited.has(activeKey) ? [...visited] : [...visited, activeKey];
      React.useLayoutEffect(() => {
        const el = ref.current;
        const measure = () => {
          const width = el.getBoundingClientRect().width;
          if (width > 0) actions.setViewportWidth(width);
        };
        measure();
        const observer = new ResizeObserver(measure);
        observer.observe(el);
        return () => observer.disconnect();
      }, [actions]);
      React.useEffect(() => {
        const entry = ctx.slots.entries("sidebar.panellist").find(e => e.options.id === panel);
        const label = typeof entry?.options.label === "function" ? entry.options.label() : entry?.options.label;
        emit("sm75-native-panel", {id: panel, label: label || "工作台"});
      }, [panel]);
      const available = layout.viewportWidth;
      const canShow = available >= 700;
      const width = canShow ? Math.min(Math.max(300, layout.rightbar ?? available * .45), available * .75) : 0;
      const track = layout.rightbarTrack ? width : 0;
      const resize = event => {
        if (event.button !== 0) return;
        event.preventDefault();
        const left = event.clientX, initial = width;
        const move = e => actions.setRightbar(initial + left - e.clientX);
        const end = () => {document.removeEventListener("pointermove", move);document.removeEventListener("pointerup", end);};
        document.addEventListener("pointermove", move);
        document.addEventListener("pointerup", end, {once:true});
      };
      return h("div", {className:"sm75-native-frame",ref,
        style:{gridTemplateColumns:"minmax(0,1fr) "+track+"px"},
        "data-rightbar-fullscreen":layout.rightbarFullscreen || undefined,
        "data-rightbar-instant":layout.rightbarInstant || undefined},
        renderSlot("sidebar", {collapsed:false,width:248}),
        h("div",{className:"sm75-native-main"},panelKeys.map(key =>
          h("div",{key,className:"sm75-native-panel",hidden:key!==activeKey,
            style:{height:"100%",minHeight:0}, "data-native-panel":key},
            renderSlot("main",{}, {entryKey:key})))) ,
        h("div",{className:"sm75-native-right","data-rightbar-col":""},
          renderSlot("rightbar",{width,viewportWidth:available,canShow})),
        layout.rightbarShown && !layout.rightbarFullscreen && width > 0
          ? h("div",{className:"sm75-native-resizer",style:{right:width+"px"},onPointerDown:resize}) : null,
        h("div",{className:"sm75-native-overlay","data-shell-overlay":""},renderSlot("shell.overlay",{})));
    }
    function Sidebar({startSession, usePanels, renderSlot}) {
      const panels = usePanels(value => value);
      React.useEffect(() => {emit("sm75-native-panels", panels.map(({id,label})=>({id,label})));}, [panels]);
      return h(React.Fragment,null,
        portal(h("div",{className:"sm75-workspace-browser"},
          h("button",{type:"button",className:"sm75-new-session",onClick:()=>{startSession();emit("sm75-native-navigate");}},"＋ 新会话"),
          h("div",{className:"sm75-workspace-list"},renderSlot("sidebar.workspaces",{wide:true,expandSidebar:()=>{}}))),"workspaceSidebar"),
        renderSlot("sidebar.settings",{wide:true}));
    }
    function ConversationHeader({native, originalProps}) {
      // Invoke the fixed-release visual component in this hook owner. Its
      // authorized child slots, Session subscriptions and actions are unchanged.
      const element = native(originalProps);
      const visible = originalProps.usePanelInfo(info => info.activePanelId === null);
      return portal(h("div",{...element.props,
        className:"sm75-conversation-header",hidden:!visible,
        "data-native-conversation-toolbar":""},element.props.children),"workspaceToolbar");
    }
    function Settings({useSections, useOnboardingSteps, useSessions, useConnectionState, reconnect, renderSlot}) {
      React.useEffect(()=>mounted("settings"),[]);
      const page = React.useSyncExternalStore(subscribeSettings,()=>settingsPage,()=>settingsPage);
      const [route,setRoute] = React.useState(()=>document.querySelector("main")?.dataset.page);
      React.useEffect(()=>{
        const follow=event=>setRoute(event.detail?.page ?? document.querySelector("main")?.dataset.page);
        document.addEventListener("sm75-route-change",follow);
        return ()=>document.removeEventListener("sm75-route-change",follow);
      },[]);
      const rows = useSections(value=>value);
      const steps = useOnboardingSteps(value=>value);
      const connection = useConnectionState(value=>value);
      const [visited, setVisited] = React.useState(() => new Set());
      const [requestedOnboarding, setRequestedOnboarding] = React.useState();
      const [completed, setCompleted] = React.useState(() => new Set());
      const onboardingActive = useSessions(state => {
        const main=Object.values(state.byId).find(session=>(session.retainedBy.mainView??0)>0);
        return state.phase==="ready" && (main===undefined || main.blank);
      });
      const step=requestedOnboarding!==undefined
        ? steps.find(item=>item.id===requestedOnboarding)
        : onboardingActive ? steps.find(item=>!completed.has(item.id)) : undefined;
      React.useEffect(()=>{if(!onboardingActive)setCompleted(new Set());},[onboardingActive]);
      React.useEffect(()=>{
        if(page!==null)setVisited(previous=>previous.has(page)?previous:new Set([...previous,page]));
      },[page]);
      React.useEffect(()=>{
        globalThis.__SM75_WORKBENCH__.reconnect=reconnect;
        emit("sm75-native-connection",{state:connection});
      },[connection,reconnect]);
      const openSection=id=>{
        const target=settingsTab(id);
        selectSettings(target);
        emit("sm75-native-settings",{id:target});
      };
      const close=()=>{selectSettings(null);emit("sm75-native-navigate");};
      const sections=rows.filter(row=>sectionVisible(row.id,page)
        || [...visited].some(tab=>sectionVisible(row.id,tab)));
      return h(React.Fragment,null,
        portal(h("div",{className:"sm75-native-settings","data-native-settings-page":page??""},
          page==="appearance" ? h("div",{className:"sm75-native-settings-actions"},renderSlot("settings.action",{})) : null,
          sections.map(row=>h("section",{key:row.id,hidden:!sectionVisible(row.id,page),
            className:"sm75-native-settings-section","data-native-settings-section":row.id},
            renderSlot("settings.section",{close},{only:row.id})))),
          "nativeSettingsSurface"),
        step!==undefined && route==="harness" ? portal(renderSlot("settings.onboarding",{
          stepId:step.id,explicit:requestedOnboarding!==undefined,
          complete:()=>{setRequestedOnboarding(undefined);setCompleted(previous=>new Set([...previous,step.id]));},
          openSection,
        },{only:step.id}),"nativeOverlays") : null);
    }
    function GeneralSettings({renderSlot}) {
      const page=React.useSyncExternalStore(subscribeSettings,()=>settingsPage,()=>settingsPage);
      React.useSyncExternalStore(
        listener=>ctx.slots.subscribe("settings.general.item",listener),
        ()=>ctx.slots.getVersion("settings.general.item"),
        ()=>ctx.slots.getVersion("settings.general.item"));
      const rows=ctx.slots.entries("settings.general.item")
        .slice().sort((a,b)=>(a.options.order??0)-(b.options.order??0));
      // The original General owner still renders every child with its original
      // injection and persistence. Only the row's visible destination changes.
      return h("div",{className:"sm75-native-general-settings"},
        rows.map(row=>{
          const id=row.options.id;
          const target=appearanceItems.has(id)?"appearance":"assistant";
          return h("div",{key:id,hidden:page!==target,"data-native-setting-item":id},
            renderSlot("settings.general.item",{},{only:id}));
        }));
    }
    function Mark({size=28,className}) {
      return h("img",{src:"/brand/favicon.svg",width:size,height:size,className,alt:"",style:{display:"block",flexShrink:0}});
    }
    function apply(context) {
      ctx=context;
      settingsPage=null;
      activePanel=undefined;
      mountedOwners.clear();
      const api={ready:false,
        fontSettled:async()=>{
          let pending,accepted;
          do {pending=ctx.theme.sm75FontWrite;accepted=await pending;}
          while(pending!==ctx.theme.sm75FontWrite);
          return accepted!==false;
        },
        panels:()=>ctx.slots.entries("sidebar.panellist").map(e=>({id:e.options.id,label:typeof e.options.label==="function"?e.options.label():e.options.label})),
        selectPanel:id=>{
          // Older already-loaded layouts have only the native select action.
          // The Frame subscription remains the authority for that interface.
          const selected=typeof ctx.layout.sm75ActivePanel==="function"
            ? ctx.layout.sm75ActivePanel() : activePanel;
          if(selected===id || (selected===undefined && id===null))return;
          ctx.layout.selectPanel(id);
          activePanel=id;
        },
        selectSettings,
        openSettings:()=>{selectSettings("appearance");emit("sm75-native-settings",{id:"appearance"});},
        setTheme:mode=>{
          if(["dark","light"].includes(mode)&&ctx.theme.getTheme().active.colorScheme!==mode)ctx.theme.setTheme(mode);
        }};
      globalThis.__SM75_WORKBENCH__=api;
      globalThis.__SM75_NATIVE_SHELL__={
        frame:props=>h(Frame,props),sidebar:props=>h(Sidebar,props),
        header:(props,native)=>h(ConversationHeader,{native,originalProps:props}),
        settings:props=>h(Settings,props),
        generalSettings:props=>h(GeneralSettings,props),
      };
      ctx.slots.inject("conversation.hero.brand.mark",()=>ctx.slots.register({name:"conversation.hero.brand.mark",priority:-100},Mark));
      ctx.effect(installNativeFontScale,"sm75 native global font scale");
      ctx.effect(()=>{
        const syncSnapshot=snapshot=>{
          document.dispatchEvent(new CustomEvent("sm75-set-theme",{detail:snapshot.active.colorScheme}));
          const size=Number(snapshot.fontSize);
          if(Number.isFinite(size)&&size>=12&&size<=26)document.documentElement.style.setProperty("--ui-fs",String(size/14));
        };
        const off=ctx.on("theme/change",syncSnapshot);
        const sync=event=>api.setTheme(event.detail);
        document.addEventListener("sm75-theme-mode",sync);
        api.setTheme(localStorage.getItem("sm75-color-mode")||"dark");
        syncSnapshot(ctx.theme.getTheme());
        return ()=>{off();document.removeEventListener("sm75-theme-mode",sync);settingsListeners.clear();settingsPage=null;api.ready=false;delete globalThis.__SM75_WORKBENCH__;delete globalThis.__SM75_NATIVE_SHELL__;};
      },"sm75 unified navigation, settings and theme");
    }
    return {apply,inject:["slots","layout","theme"]};
  },
});
