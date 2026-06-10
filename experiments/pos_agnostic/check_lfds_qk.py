import re

# Original run
data1 = open('logs/pafl_shared_lfds_qk_h1.log').read().replace('\r', '\n')
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
data2 = open('logs/pafl_shared_lfds_qk_h1_100k.log').read().replace('\r', '\n')
seen2 = []
seen_set2 = set()
for line in data2.split('\n'):
    m = re.search(r'val_ppl=([\d.]+)', line)
    if m:
        ppl = m.group(1)
        if ppl not in seen_set2:
            seen2.append(float(ppl))
            seen_set2.add(ppl)

print('lfds_qk original run:')
for i, ppl in enumerate(seen1):
    if i == 0: continue
    print(f'  {i*5}K: {ppl}')

# Figure out where original ended
orig_end = (len(seen1) - 1) * 5
print(f'\nlfds_qk continuation (from ~{orig_end}K):')
for i, ppl in enumerate(seen2):
    if i == 0: continue
    total = orig_end + i * 5
    print(f'  {total}K total: {ppl}')

# Extrap from continuation
extrap_seen = {}
for line in data2.split('\n'):
    m = re.search(r'extrap iter (\d+): (.+)\]', line)
    if m:
        key = int(m.group(1))
        if key not in extrap_seen:
            extrap_seen[key] = m.group(2)
if extrap_seen:
    print('\nextrap (continuation iters):')
    for k in sorted(extrap_seen):
        total = orig_end + k // 1000
        print(f'  {k//1000}K cont ({total}K total): {extrap_seen[k]}')
