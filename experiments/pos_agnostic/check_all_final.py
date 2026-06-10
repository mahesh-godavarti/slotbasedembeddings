import re

logs = [
    ('joformer2 from fixed 200K', 'logs/pafl_joformer2_from_fixed_200k_v3.log'),
    ('lf_qk 150K', 'logs/pafl_shared_lf_qk_h1_150k_v2.log'),
    ('lf_qk 200K', 'logs/pafl_shared_lf_qk_h1_200k.log'),
    ('joformer2 from frozen', 'logs/pafl_joformer2_from_frozen_v3.log'),
    ('monoidal2 from frozen', 'logs/pafl_monoidal2_from_frozen_v3.log'),
]

for model, logfile in logs:
    try:
        data = open(logfile).read().replace('\r', '\n')
    except FileNotFoundError:
        print(f'{model}: log not found')
        continue

    # Val PPL
    seen = []
    seen_set = set()
    for line in data.split('\n'):
        m = re.search(r'val_ppl=([\d.]+)', line)
        if m:
            ppl = m.group(1)
            if ppl not in seen_set:
                seen.append(float(ppl))
                seen_set.add(ppl)
    print(f'\n{model} val PPL:')
    for i, ppl in enumerate(seen):
        if i == 0:
            continue
        print(f'  {i*5}K: {ppl}')

    # Extrap
    extrap_seen = {}
    for line in data.split('\n'):
        m = re.search(r'extrap iter (\d+): (.+)\]', line)
        if m:
            key = int(m.group(1))
            if key not in extrap_seen:
                extrap_seen[key] = m.group(2)
    if extrap_seen:
        print(f'  extrap:')
        for k in sorted(extrap_seen):
            print(f'    {k//1000}K: {extrap_seen[k]}')
