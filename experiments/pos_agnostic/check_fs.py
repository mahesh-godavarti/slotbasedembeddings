import re

for model, logfile in [('fs_qk', 'logs/pafl_shared_fs_qk_h1.log'),
                       ('fs_qkv', 'logs/pafl_shared_fs_qkv_h1.log')]:
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
    print(f'{model}:')
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
