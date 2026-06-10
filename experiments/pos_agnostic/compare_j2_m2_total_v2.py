import re

def get_val_ppls(logfiles, total_offsets):
    """Extract val PPLs from multiple log files with total iteration offsets."""
    result = {}
    for logfile, offset in zip(logfiles, total_offsets):
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
            total = offset + i * 5
            result[total] = ppl
    return result

# j2 control: 5K frozen + 95K first + 100K second + 100K third
j2 = get_val_ppls([
    'logs/pafl_joformer2_from_frozen_control.log',
    'logs/pafl_j2_control_200k.log',
    'logs/pafl_j2_control_300k.log',
], [5, 100, 200])

# m2 control: 5K frozen + 40K first + 100K second + 100K third
m2 = get_val_ppls([
    'logs/pafl_monoidal2_from_frozen_control.log',
    'logs/pafl_m2_control_200k.log',
    'logs/pafl_m2_control_300k.log',
], [5, 45, 145])

matched = sorted(set(j2.keys()) & set(m2.keys()))
print("Total_K | j2_control | m2_control | gap(j2-m2)")
for t in matched:
    gap = j2[t] - m2[t]
    print(f"  {t}K | {j2[t]:.2f} | {m2[t]:.2f} | {gap:+.2f}")
