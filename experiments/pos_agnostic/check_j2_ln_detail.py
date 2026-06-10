import re

# Original 5K run
data1 = open('logs/pafl_joformer2_h1_ln_consistent.log').read().replace('\r', '\n')
seen1 = []
seen_set1 = set()
for line in data1.split('\n'):
    m = re.search(r'val_ppl=([\d.]+)', line)
    if m:
        ppl = m.group(1)
        if ppl not in seen_set1:
            seen1.append(float(ppl))
            seen_set1.add(ppl)

# Continuation run
data2 = open('logs/pafl_joformer2_h1_ln_consistent_100k.log').read().replace('\r', '\n')
seen2 = []
seen_set2 = set()
for line in data2.split('\n'):
    m = re.search(r'val_ppl=([\d.]+)', line)
    if m:
        ppl = m.group(1)
        if ppl not in seen_set2:
            seen2.append(float(ppl))
            seen_set2.add(ppl)

print('Val PPL (total iters):')
# Original: 5K run
for i, ppl in enumerate(seen1):
    if i == 0: continue
    print(f'  {i*5}K: {ppl}')
# Continuation: starts from 5K
for i, ppl in enumerate(seen2):
    if i == 0: continue  # skip initial eval (same as end of original)
    total = 5 + i * 5
    print(f'  {total}K: {ppl}')

# Extrap from both
print('\nExtrap (total iters):')
for data, offset in [(data1, 0), (data2, 5)]:
    extrap_seen = {}
    for line in data.split('\n'):
        m = re.search(r'extrap iter (\d+): (.+)\]', line)
        if m:
            key = int(m.group(1))
            if key not in extrap_seen:
                extrap_seen[key] = m.group(2)
    for k in sorted(extrap_seen):
        total = offset + k // 1000
        print(f'  {total}K: {extrap_seen[k]}')
