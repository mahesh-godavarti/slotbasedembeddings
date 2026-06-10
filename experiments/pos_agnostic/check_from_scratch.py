import re
import sys

for model, logfile in [('joformer2', 'logs/pafl_joformer2_from_scratch.log'),
                       ('monoidal2', 'logs/pafl_monoidal2_from_scratch.log')]:
    data = open(logfile).read().replace('\r', '\n')
    # Collect unique val_ppl values in order of first appearance
    seen = []
    seen_set = set()
    for line in data.split('\n'):
        m = re.search(r'val_ppl=([\d.]+)', line)
        if m:
            ppl = m.group(1)
            if ppl not in seen_set:
                seen.append(float(ppl))
                seen_set.add(ppl)
    print(f'{model}:')
    for i, ppl in enumerate(seen):
        if i == 0:
            continue  # skip iter 0
        print(f'  {i*5}K: {ppl}')

    # Extrap data
    extrap_seen = set()
    for line in data.split('\n'):
        m = re.search(r'extrap iter (\d+): (.+)\]', line)
        if m:
            key = m.group(1)
            if key not in extrap_seen:
                extrap_seen.add(key)
                print(f'  extrap {int(key)//1000}K: {m.group(2)}')
