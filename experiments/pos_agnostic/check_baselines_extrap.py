import re

for model, logfile in [('RoPE', 'logs/pafl_rope_5e5_200k.log'),
                       ('jfixed', 'logs/pafl_jfixed_5e5_200k.log'),
                       ('random', 'logs/pafl_random_indep_5e5_200k.log')]:
    data = open(logfile).read().replace('\r', '\n')
    extrap_seen = {}
    for line in data.split('\n'):
        m = re.search(r'extrap iter (\d+): (.+)\]', line)
        if m:
            key = int(m.group(1))
            if key not in extrap_seen:
                extrap_seen[key] = m.group(2)
    print(f'{model}:')
    for k in sorted(extrap_seen):
        print(f'  {k//1000}K: {extrap_seen[k]}')
    print()
