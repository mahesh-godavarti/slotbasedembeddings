import re

logs = [
    ('joformer2 control (frozen)', 'logs/pafl_joformer2_from_frozen_control.log'),
    ('joformer2 slow (angle lr=5e-5)', 'logs/pafl_joformer2_from_frozen_slowboth.log'),
    ('monoidal2 control (frozen)', 'logs/pafl_monoidal2_from_frozen_control.log'),
    ('monoidal2 slow (angle lr=5e-5)', 'logs/pafl_monoidal2_from_frozen_slowboth.log'),
]

for model, logfile in logs:
    try:
        data = open(logfile).read().replace('\r', '\n')
    except FileNotFoundError:
        print(f'{model}: log not found\n')
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
    print(f'{model} val PPL:')
    for i, ppl in enumerate(seen):
        if i == 0:
            continue
        print(f'  {i*5}K: {ppl}')

    extrap_seen = {}
    for line in data.split('\n'):
        m = re.search(r'extrap iter (\d+): (.+)\]', line)
        if m:
            key = int(m.group(1))
            if key not in extrap_seen:
                extrap_seen[key] = m.group(2)
    if extrap_seen:
        for k in sorted(extrap_seen):
            print(f'  extrap {k//1000}K: {extrap_seen[k]}')
    print()
