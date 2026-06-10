import re

for model, logfile in [('random_ln_qk (fixed freq)', 'logs/pafl_random_ln_qk.log'),
                       ('random_ln_indep_qk (fixed freq, indep)', 'logs/pafl_random_ln_indep_qk.log'),
                       ('random_glf_qk (learned freq)', 'logs/pafl_random_glf_qk.log')]:
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
    print(f'{model}:')
    for i, ppl in enumerate(seen):
        if i == 0:
            continue
        print(f'  {i*5}K: {ppl}')
    print()
