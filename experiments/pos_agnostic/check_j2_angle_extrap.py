import re

data = open('logs/pafl_j2_angle_200k.log').read().replace('\r', '\n')
extrap_seen = {}
for line in data.split('\n'):
    m = re.search(r'extrap iter (\d+): (.+)\]', line)
    if m:
        key = int(m.group(1))
        if key not in extrap_seen:
            extrap_seen[key] = m.group(2)
for k in sorted(extrap_seen):
    print(f'  {k//1000}K: {extrap_seen[k]}')
