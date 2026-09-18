#!/usr/bin/env python3
import re, glob

BASE = "/opt/harness/node_modules/@deepseek-ai"
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
    if "--dsh-font-scale" in t:
        continue
    t2, n1 = fs.subn(rep_fs, t)
    t2, n2 = lh.subn(rep_lh, t2)
    t2, n3 = fsh.subn(rep_fsh, t2)
    if n1 or n2 or n3:
        open(f, "w", encoding="utf-8").write(t2)
        changed_files += 1
        total_fs += n1
        total_lh += n2
        total_sh += n3

print("files changed:", changed_files)
print("font-size:", total_fs, "line-height:", total_lh, "font shorthand:", total_sh)
