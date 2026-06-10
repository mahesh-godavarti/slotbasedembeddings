import re

def get_ordered_ppls(logfile):
    data = open(logfile).read().replace('\r', '\n')
    seen = []
    seen_set = set()
    for line in data.split('\n'):
        m = re.search(r'val_ppl=([\d.]+)', line)
        if m:
            ppl = m.group(1)
            if ppl not in seen_set:
                seen.append(float(ppl))
                seen_set.add(ppl)
    return seen

# pemb_qk: original 100K (5K eval) + 200K extension (5K eval)
pemb_orig = get_ordered_ppls('logs/pafl_pemb_deti_v2.log')
pemb_ext = get_ordered_ppls('logs/gpu3_queue_v2.log')
pemb = {}
for i, ppl in enumerate(pemb_orig):
    if i == 0: continue
    pemb[i*5] = ppl
for i, ppl in enumerate(pemb_ext):
    if i == 0: continue
    pemb[100 + i*5] = ppl

# cbd K=4: original 10K (5K eval) + continuation 90K (5K eval) + 200K extension (5K eval)
cbd_orig = get_ordered_ppls('logs/pafl_shared_cbd_qk_K4.log')  # first 10K
cbd_cont = get_ordered_ppls('logs/pafl_shared_cbd_qk_K4_100k.log')  # 10K->100K
cbd_ext = get_ordered_ppls('logs/gpu2_queue.log')  # 100K->200K

cbd = {}
for i, ppl in enumerate(cbd_orig):
    if i == 0: continue
    cbd[i*5] = ppl
for i, ppl in enumerate(cbd_cont):
    if i == 0: continue
    cbd[10 + i*5] = ppl
for i, ppl in enumerate(cbd_ext):
    if i == 0: continue
    cbd[100 + i*5] = ppl

# RoPE 5e-4
rope_ppls = get_ordered_ppls('logs/pafl_rope_5e4_200k.log')
rope = {}
for i, ppl in enumerate(rope_ppls):
    if i == 0: continue
    rope[i*5] = ppl

matched = sorted(set(pemb.keys()) & set(cbd.keys()) & set(rope.keys()))
print(f"{'Iter':>6} | {'pemb_qk':>8} | {'cbd_K4':>7} | {'RoPE':>6} | {'pemb-cbd':>8} | {'pemb-RoPE':>9} | {'cbd-RoPE':>8}")
print("-" * 75)
for k in matched:
    pc = pemb[k] - cbd[k]
    pr = pemb[k] - rope[k]
    cr = cbd[k] - rope[k]
    print(f"{k:>5}K | {pemb[k]:>8.2f} | {cbd[k]:>7.2f} | {rope[k]:>6.2f} | {pc:>+8.2f} | {pr:>+9.2f} | {cr:>+8.2f}")
