import re

for model, logfile in [('lf_qk slowangle', 'logs/pafl_shared_lf_qk_h1_slowangle.log'),
                       ('lf_qkv slowangle', 'logs/pafl_shared_lf_qkv_h1_slowangle.log')]:
    data = open(logfile).read().replace('\r', '\n')
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
    print(f'{model} val PPL:')
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
    print(f'{model} extrap:')
    for k in sorted(extrap_seen):
        print(f'  {k//1000}K: {extrap_seen[k]}')
    print()
