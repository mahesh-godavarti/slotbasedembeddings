import re

for model, logfile in [('RoPE 5e-4', 'logs/pafl_rope_5e4_200k.log'),
                       ('jfixed 5e-4', 'logs/pafl_jfixed_5e4_200k.log'),
                       ('random 5e-4', 'logs/pafl_random_indep_5e4_200k.log')]:
    try:
        data = open(logfile).read().replace('\r', '\n')
    except FileNotFoundError:
        print(f'{model}: not started yet\n')
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
        if i == 0: continue
        print(f'  {i*5}K: {ppl}')
    print()
