import re

def get_data(logfile, total_offset=0):
    try:
        data = open(logfile).read().replace('\r', '\n')
    except FileNotFoundError:
        return {}, {}

    # Val PPLs
    ppls = []
    ppl_set = set()
    for line in data.split('\n'):
        m = re.search(r'val_ppl=([\d.]+)', line)
        if m:
            ppl = m.group(1)
            if ppl not in ppl_set:
                ppls.append(float(ppl))
                ppl_set.add(ppl)
    val = {}
    for i, ppl in enumerate(ppls):
        if i == 0: continue
        val[total_offset + i * 5] = ppl

    # Extrap
    extrap = {}
    for line in data.split('\n'):
        m = re.search(r'extrap iter (\d+): (.+)\]', line)
        if m:
            key = int(m.group(1))
            if key not in extrap:
                parts = {}
                for p in m.group(2).split():
                    l, v = p.split(':')
                    parts[int(l)] = float(v)
                extrap[total_offset + key // 1000] = parts

    return val, extrap

# pmlp v5 (1K eval interval)
pmlp_val = {}
pmlp_ext = {}
try:
    data = open('logs/pafl_shared_pmlp_qk_v5.log').read().replace('\r', '\n')
    ppls = []
    ppl_set = set()
    for line in data.split('\n'):
        m = re.search(r'val_ppl=([\d.]+)', line)
        if m:
            ppl = m.group(1)
            if ppl not in ppl_set:
                ppls.append(float(ppl))
                ppl_set.add(ppl)
    for i, ppl in enumerate(ppls):
        if i == 0: continue
        pmlp_val[i] = ppl  # 1K intervals

    for line in data.split('\n'):
        m = re.search(r'extrap iter (\d+): (.+)\]', line)
        if m:
            key = int(m.group(1)) // 1000
            if key not in pmlp_ext:
                parts = {}
                for p in m.group(2).split():
                    l, v = p.split(':')
                    parts[int(l)] = float(v)
                pmlp_ext[key] = parts
except FileNotFoundError:
    pass

print("=== pmlp v5 (GPU 0) ===")
for k in sorted(pmlp_val.keys()):
    ext_str = ""
    if k in pmlp_ext:
        e = pmlp_ext[k]
        ratio = e.get(8192, 0) / e.get(512, 1)
        ext_str = f"  512={e.get(512)}, 8192={e.get(8192)}, ratio={ratio:.2f}x"
    print(f"  {k}K: val={pmlp_val[k]:.2f}{ext_str}")

# cbd K=8 qkv (GPU 3, in gpu3_queue_v2.log after pemb_qk)
print("\n=== cbd K=8 qkv (GPU 3) ===")
try:
    data = open('logs/gpu3_queue_v2.log').read().replace('\r', '\n')
    # Find cbd_qkv lines only
    ppls = []
    ppl_set = set()
    in_cbd = False
    for line in data.split('\n'):
        if 'shared_cbd_qkv' in line:
            in_cbd = True
        elif 'shared_pemb_qk' in line:
            in_cbd = False
        if in_cbd:
            m = re.search(r'val_ppl=([\d.]+)', line)
            if m:
                ppl = m.group(1)
                if ppl not in ppl_set:
                    ppls.append(float(ppl))
                    ppl_set.add(ppl)
    for i, ppl in enumerate(ppls):
        if i == 0: continue
        print(f"  {i*5}K: val={ppl:.2f}")

    extrap_seen = {}
    for line in data.split('\n'):
        if 'shared_cbd_qkv' not in line:
            continue
        m = re.search(r'extrap iter (\d+): (.+)\]', line)
        if m:
            key = int(m.group(1))
            if key not in extrap_seen:
                parts = {}
                for p in m.group(2).split():
                    l, v = p.split(':')
                    parts[int(l)] = float(v)
                extrap_seen[key] = parts
    for k in sorted(extrap_seen):
        e = extrap_seen[k]
        ratio = e.get(8192, 0) / e.get(512, 1)
        print(f"  extrap {k//1000}K: 512={e.get(512)}, 8192={e.get(8192)}, ratio={ratio:.2f}x")
except FileNotFoundError:
    print("  log not found")

# GPU 1 status
print("\n=== GPU 1 queue ===")
try:
    data = open('logs/gpu1_queue.log').read().replace('\r', '\n')
    # Find which model is running
    for model in ['shared_pemb_qkv', 'joformer_fixed', 'random_ln_indep_qk', 'shared_cbd_qk']:
        lines = [l for l in data.split('\n') if model in l and 'val_ppl' in l]
        if lines:
            last = lines[-1]
            m = re.search(r'val_ppl=([\d.]+)', last)
            iter_m = re.search(r'(\d+)/\d+', last)
            if m and iter_m:
                print(f"  {model}: iter={iter_m.group(1)}, val_ppl={m.group(1)}")
except FileNotFoundError:
    print("  log not found")

# GPU 2 status
print("\n=== GPU 2 queue ===")
try:
    data = open('logs/gpu2_queue.log').read().replace('\r', '\n')
    for model in ['shared_cbd_qk', 'random_ln_indep_qkv']:
        lines = [l for l in data.split('\n') if model in l and 'val_ppl' in l]
        if lines:
            last = lines[-1]
            m = re.search(r'val_ppl=([\d.]+)', last)
            iter_m = re.search(r'(\d+)/\d+', last)
            if m and iter_m:
                print(f"  {model}: iter={iter_m.group(1)}, val_ppl={m.group(1)}")
except FileNotFoundError:
    print("  log not found")
