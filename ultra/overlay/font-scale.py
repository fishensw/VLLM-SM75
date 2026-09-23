#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re, glob

parser = argparse.ArgumentParser(description="Scale native Harness UI fonts without replacing bundles.")
parser.add_argument("--base", default="/opt/harness/node_modules/@deepseek-ai")
BASE = parser.parse_args().base
# Extend the existing ultra range without copying an old theme bundle.
# The server schema and the compiled client each inline some constants.
theme = Path(BASE) / "dsh-client-ui-theme"
if theme.exists():
    version = json.loads((theme / "package.json").read_text(encoding="utf-8"))["version"]
    if version != "0.1.7-alpha.2":
        raise RuntimeError(f"font range patch requires Harness 0.1.7-alpha.2, got {version}")
    replacements = {
        "index.js": [
            ("const FONT_SIZE_MAX = 17;", "const FONT_SIZE_MAX = 26;", 1),
            (".min(12).max(17).default(14)", ".min(12).max(26).default(14)", 2),
        ],
        "client.js": [
            (".min(12).max(17).default(14)", ".min(12).max(26).default(14)", 1),
            ("fontSize >= 17", "fontSize >= 26", 1),
            ("px > 17", "px > 26", 1),
            ("outside 12..17", "outside 12..26", 1),
            ("parsed <= 17", "parsed <= 26", 1),
            ("仅影响会话内容的字号", "调整工作台与会话的字号", 1),
            ("Only affects conversation content", "Adjusts workbench and conversation text", 1),
        ],
    }
    for name, rules in replacements.items():
        file = theme / "lib" / name
        source = file.read_text(encoding="utf-8")
        for old, new, expected in rules:
            if source.count(old) == expected and source.count(new) == 0:
                source = source.replace(old, new)
            elif source.count(old) != 0 or source.count(new) != expected:
                raise RuntimeError(f"Harness theme anchor changed in {name}: {old}")
        file.write_text(source, encoding="utf-8", newline="\n")

files = sorted(glob.glob(BASE + "/*/lib/client.js")) + sorted(glob.glob(BASE + "/dsh-web-frontend/dist/assets/*.css"))

fs = re.compile(r"font-size:\s*([0-9.]+)px")
lh = re.compile(r"line-height:\s*([0-9.]+)px")
fsh = re.compile(r"font:\s*([0-9.]+)px(?=[/ ])")

total_fs = total_lh = total_sh = 0
changed_files = 0

def rep_fs(m):
    return "font-size:calc(%spx * var(--dsh-font-scale,1))" % m.group(1)

def rep_lh(m):
    return "line-height:calc(%spx * var(--dsh-font-scale,1))" % m.group(1)

def rep_fsh(m):
    return "font:calc(%spx * var(--dsh-font-scale,1))" % m.group(1)

for f in files:
    t = open(f, encoding="utf-8").read()
    t2, n1 = fs.subn(rep_fs, t)
    t2, n2 = lh.subn(rep_lh, t2)
    t2, n3 = fsh.subn(rep_fsh, t2)
    if n1 or n2 or n3:
        open(f, "w", encoding="utf-8", newline="\n").write(t2)
        changed_files += 1
        total_fs += n1
        total_lh += n2
        total_sh += n3

print("files changed:", changed_files)
print("font-size:", total_fs, "line-height:", total_lh, "font shorthand:", total_sh)
