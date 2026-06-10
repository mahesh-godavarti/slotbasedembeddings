import re

data = open('logs/pafl_joformer_fixed_clean.log').read().replace('\r', '\n')

# This log has both rope and joformer_fixed - need to find rope
# Actually let me check the rope clean log
for logfile in ['logs/pafl_rope_clean.log', 'logs/pafl_joformer_fixed_clean.log']:
    try:
        data = open(logfile).read().replace('\r', '\n')
    except FileNotFoundError:
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
    print(f'{logfile}:')
    for i, ppl in enumerate(seen):
        if i == 0:
            continue
        print(f'  {i*5}K: {ppl}')
