import re

# j2 control: 5K frozen, then 95K first phase, then 200K phase
# total = 5K + first_phase_iter  (for first phase)
# total = 100K + 200K_phase_iter (for 200K phase)
j2_first = []
data = open('logs/pafl_joformer2_from_frozen_control.log').read().replace('\r', '\n')
seen_set = set()
for line in data.split('\n'):
    m = re.search(r'val_ppl=([\d.]+)', line)
    if m:
        ppl = m.group(1)
        if ppl not in seen_set:
            j2_first.append(float(ppl))
            seen_set.add(ppl)

j2_second = []
data = open('logs/pafl_j2_control_200k.log').read().replace('\r', '\n')
seen_set = set()
for line in data.split('\n'):
    m = re.search(r'val_ppl=([\d.]+)', line)
    if m:
        ppl = m.group(1)
        if ppl not in seen_set:
            j2_second.append(float(ppl))
            seen_set.add(ppl)

j2_total = {}
for i, ppl in enumerate(j2_first):
    if i == 0: continue
    total = 5 + i * 5  # 5K frozen + i*5K continuation
    j2_total[total] = ppl
for i, ppl in enumerate(j2_second):
    if i == 0: continue
    total = 100 + i * 5  # 100K + i*5K
    j2_total[total] = ppl

# m2 control: 5K frozen, then 40K first phase, then 100K second phase
# total = 5K + first_phase_iter (for first phase)
# total = 45K + second_phase_iter (for second phase)
m2_first = []
data = open('logs/pafl_monoidal2_from_frozen_control.log').read().replace('\r', '\n')
seen_set = set()
for line in data.split('\n'):
    m = re.search(r'val_ppl=([\d.]+)', line)
    if m:
        ppl = m.group(1)
        if ppl not in seen_set:
            m2_first.append(float(ppl))
            seen_set.add(ppl)

m2_second = []
data = open('logs/pafl_m2_control_200k.log').read().replace('\r', '\n')
seen_set = set()
for line in data.split('\n'):
    m = re.search(r'val_ppl=([\d.]+)', line)
    if m:
        ppl = m.group(1)
        if ppl not in seen_set:
            m2_second.append(float(ppl))
            seen_set.add(ppl)

m2_total = {}
for i, ppl in enumerate(m2_first):
    if i == 0: continue
    total = 5 + i * 5
    m2_total[total] = ppl
for i, ppl in enumerate(m2_second):
    if i == 0: continue
    total = 45 + i * 5
    m2_total[total] = ppl

# Print matched
all_totals = sorted(set(j2_total.keys()) & set(m2_total.keys()))
print("Total_K | j2_control | m2_control | gap(j2-m2)")
for t in all_totals:
    gap = j2_total[t] - m2_total[t]
    print(f"  {t}K | {j2_total[t]:.2f} | {m2_total[t]:.2f} | {gap:+.2f}")
