import re

data = open('logs/pafl_shared_pmlp2_qk.log').read()

# Val PPL
vals = []
seen = set()
for line in data.split('\n'):
    if 'shared_pmlp2_qk' not in line:
        continue
    m = re.search(r'val_ppl=([\d.]+)', line)
    if m and m.group(1) not in seen:
        vals.append(float(m.group(1)))
        seen.add(m.group(1))

# Extrap
extrap = []
seen_ext = set()
for line in data.split('\n'):
    if 'shared_pmlp2_qk' not in line or '512:' not in line:
        continue
    m = re.search(r'(512:[\d.]+ .* 8192:[\d.]+)', line)
    if m and m.group(1) not in seen_ext:
        seen_ext.add(m.group(1))
        m2 = re.search(r'512:([\d.]+).*8192:([\d.]+)', m.group(1))
        extrap.append((float(m2.group(1)), float(m2.group(2))))

print(f"pmlp2: {len(vals)-1}K iters")
for i, v in enumerate(vals):
    if i == 0: continue
    ext = ""
    if i <= len(extrap):
        e = extrap[i-1]
        ext = f"  512={e[0]}, 8192={e[1]}, ratio={e[1]/e[0]:.2f}x"
    print(f"  {i}K: val={v:.2f}{ext}")
