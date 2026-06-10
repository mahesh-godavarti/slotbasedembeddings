import re

def get_val_ppls(logfiles):
    result = {}
    for logfile in logfiles:
        try:
            data = open(logfile).read().replace('\r', '\n')
        except FileNotFoundError:
            continue
        seen = []
        seen_set = set()
        for line in data.split('\n'):
            m = re.search(r'val_ppl=([\d.]+)', line)
            if m:
                ppl = m.group(1)
                if ppl not in seen_set:
                    seen.append(float(ppl))
                    seen_set.add(ppl)
        for i, ppl in enumerate(seen):
            if i == 0: continue
            result[i * 5] = ppl
    return result

# pemb_qk: original 100K + 200K extension
pemb_orig = get_val_ppls(['logs/pafl_pemb_deti_v2.log'])
pemb_ext = get_val_ppls(['logs/gpu3_queue_v2.log'])
pemb = {}
for k, v in pemb_orig.items():
    pemb[k] = v
for k, v in pemb_ext.items():
    pemb[100 + k] = v  # extension starts from 100K

# random_qk 5e-4
random_qk = get_val_ppls(['logs/pafl_random_indep_5e4_200k.log'])

# RoPE 5e-4
rope = get_val_ppls(['logs/pafl_rope_5e4_200k.log'])

# Print matched
all_iters = sorted(set(pemb.keys()) & set(random_qk.keys()) & set(rope.keys()))
print(f"{'Iter':>6} | {'pemb_qk':>8} | {'random_qk':>9} | {'RoPE':>6} | {'pemb-rand':>9} | {'pemb-RoPE':>9}")
print("-" * 65)
for k in all_iters:
    pr = pemb[k] - random_qk[k]
    pe = pemb[k] - rope[k]
    print(f"{k:>5}K | {pemb[k]:>8.2f} | {random_qk[k]:>9.2f} | {rope[k]:>6.2f} | {pr:>+9.2f} | {pe:>+9.2f}")
