import { isIP } from "node:net";
function address(value) {
  if (typeof value !== "string") throw Error("IP 地址无效");
  let ip = value.trim();
  if (ip.startsWith("::ffff:") && isIP(ip.slice(7)) === 4) ip = ip.slice(7);
  const bits = isIP(ip) === 4 ? 32 : isIP(ip) === 6 ? 128 : 0;
  if (!bits) throw Error("IP 地址无效");
  if (bits === 32) return { bits, value: ip.split(".").reduce((v, n) => (v << 8n) + BigInt(n), 0n) };
  if (ip.includes(".")) {
    const i = ip.lastIndexOf(":");
    const v = address(ip.slice(i + 1)).value;
    ip = ip.slice(0, i + 1) + (v >> 16n).toString(16) + ":" + (v & 65535n).toString(16);
  }
  const halves = ip.split("::"), left = halves[0] ? halves[0].split(":") : [], right = halves[1] ? halves[1].split(":") : [];
  const parts = halves.length === 2 ? [...left, ...Array(8-left.length-right.length).fill("0"), ...right] : left;
  return { bits, value: parts.reduce((v,n) => (v << 16n) + BigInt("0x"+n), 0n) };
}
export function parseNetwork(value) {
  const parts = value.trim().split("/");
  if (parts.length > 2) throw Error("网段格式无效");
  const ip = address(parts[0]);
  const prefix = parts.length === 1 ? ip.bits : /^\d+$/.test(parts[1]) ? Number(parts[1]) : -1;
  if (prefix < 0 || prefix > ip.bits) throw Error("网段前缀长度无效");
  return { ...ip, prefix };
}
export function contains(network, ip) {
  try {
    const n = parseNetwork(network), a = address(ip);
    const shift = BigInt(n.bits-n.prefix);
    return n.bits === a.bits && (n.value >> shift) === (a.value >> shift);
  } catch { return false; }
}
export function validateNetworks(values) {
  if (!Array.isArray(values) || values.length > 64) throw Error("最多配置 64 个白名单网段");
  return [...new Set(values.map(v => {
    if (typeof v !== "string" || v.length > 64) throw Error("网段格式无效");
    parseNetwork(v); return v.trim();
  }))];
}
