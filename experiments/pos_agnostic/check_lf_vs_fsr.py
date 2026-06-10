import re

for model, logfile in [('lf_qk', 'logs/pafl_shared_lf_qk_h1.log'),
                       ('fsr_qk', 'logs/pafl_shared_fsr_qk_h1.log')]:
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
