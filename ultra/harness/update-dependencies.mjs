// Run after npm install --package-lock-only; npm owns versions and integrity.
// Use --check in validation to require the committed inventory to match the lock.
import { readFileSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
if (args.length > 1 || (args.length === 1 && args[0] !== "--check")) {
  throw new Error("Usage: node update-dependencies.mjs [--check]");
}
const lock = JSON.parse(readFileSync(new URL("./package-lock.json", import.meta.url)));
const manifestUrl = new URL("./dependencies.json", import.meta.url);
const inventory = Object.entries(lock.packages)
  .filter(([location]) => location !== "")
  .map(([location, metadata]) => ({
    package: location.replace(/^node_modules\//, ""),
    version: metadata.version,
    license: metadata.license ?? null,
    resolved: metadata.resolved,
    integrity: metadata.integrity,
  }));
const expected = `${JSON.stringify(inventory, null, 2)}\n`;
if (args[0] === "--check") {
  if (readFileSync(manifestUrl, "utf8") !== expected) {
    throw new Error("dependencies.json is stale; run node update-dependencies.mjs");
  }
  console.log(`Dependency inventory matches ${inventory.length} locked packages.`);
} else {
  writeFileSync(manifestUrl, expected);
  console.log(`Wrote ${inventory.length} locked packages to dependencies.json.`);
}
