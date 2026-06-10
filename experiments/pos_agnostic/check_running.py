#!/usr/bin/env python3
"""Check status of all currently running experiments."""
import re
import sys

logs = [
    ('rope_lf (GPU 0)', 'logs/gpu0_rope_lf_sched.log', 'rope_lf'),
    ('jfixed_lf (GPU 3)', 'logs/gpu3_jfixed_lf_sched.log', 'joformer_fixed_lf'),
]

for name, logfile, model in logs:
    try:
        data = open(logfile).read()
    except FileNotFoundError:
        print(f'{name}: not started')
        continue
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
    print(f'{name}: {len(vals)-1} evals, latest val={vals[-1]:.2f}')
    if exts:
        e = exts[-1]
        print(f'  extrap: 512={e[0]}, 8192={e[1]}, ratio={e[1]/e[0]:.2f}x')
