// Host-side stub for the token-usage client plugin. The plugin is purely a
// browser surface (Settings > 使用统计) backed by the host /token-usage route;
// the host row exists only so the loader resolves the package name and the
// client row (dsh.client: true) feeds window.__DSH_BOOT__. There is no host
// logic to run.
/** Host plugin body — no host-side behavior for this surface plugin. */
export function apply() {}
