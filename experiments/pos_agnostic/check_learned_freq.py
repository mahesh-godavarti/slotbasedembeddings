#!/usr/bin/env python3
"""Compare RoPE vs jfixed vs their learned-frequency variants."""
import re

def get_vals_tqdm(logfile, model, interval=5):
    """Parse tqdm unique val_ppl values and extrap."""
    try:
        data = open(logfile, errors='replace').read()
    except FileNotFoundError:
        return {}, {}
    vals = []
    last = None
    for line in data.split('\n'):
        if model not in line:
            continue
        m = re.search(r'val_ppl=([\d.]+)', line)
        if m and m.group(1) != last:
            vals.append(float(m.group(1)))
            last = m.group(1)
    exts = []
    last_ext = None
    for line in data.split('\n'):
        if model not in line or '512:' not in line:
            continue
        m = re.search(r'512:([\d.]+).*8192:([\d.]+)', line)
        if m and m.group(0) != last_ext:
            last_ext = m.group(0)
            exts.append((float(m.group(1)), float(m.group(2))))
    val_result = {}
    for i, v in enumerate(vals):
        if i == 0:
            continue
        val_result[i * interval] = v
    ext_result = {}
    for i, e in enumerate(exts):
        ext_result[(i + 1) * interval] = e[1] / e[0]
    return val_result, ext_result

# All from constant lr=5e-4 base training logs only (first 100K)
rope, rope_ext = get_vals_tqdm('logs/pafl_rope_5e4_200k.log', 'rope')
jfixed, jfixed_ext = get_vals_tqdm('logs/pafl_jfixed_5e4_200k.log', 'joformer_fixed')
rope_lf, rope_lf_ext = get_vals_tqdm('logs/gpu0_rope_lf_sched.log', 'rope_lf')
jfixed_lf, jfixed_lf_ext = get_vals_tqdm('logs/gpu3_jfixed_lf_sched.log', 'joformer_fixed_lf')

models = {
    'RoPE': (rope, rope_ext),
    'jfixed': (jfixed, jfixed_ext),
    'rope_lf': (rope_lf, rope_lf_ext),
    'jfixed_lf': (jfixed_lf, jfixed_lf_ext),
}

names = list(models.keys())
header = f"{'Iter':>6}"
for n in names:
    header += f" | {n:>10}"
for n in names:
    header += f" | {n+' ext':>12}"
print(header)
print("-" * len(header))

for k in range(5, 105, 5):
    row = f"{k:>5}K"
    has_data = False
    for n in names:
        vals, _ = models[n]
        if k in vals:
            row += f" | {vals[k]:>10.2f}"
            has_data = True
        else:
            row += f" | {'':>10}"
    for n in names:
        _, exts = models[n]
        if k in exts:
            row += f" | {exts[k]:>11.2f}x"
            has_data = True
        else:
            row += f" | {'':>12}"
    if has_data:
        print(row)
