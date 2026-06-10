import re

def get_extrap(logfile, total_offset):
    data = open(logfile).read().replace('\r', '\n')
    extrap_seen = {}
    for line in data.split('\n'):
        m = re.search(r'extrap iter (\d+): (.+)\]', line)
        if m:
            key = int(m.group(1))
            if key not in extrap_seen:
                extrap_seen[key] = m.group(2)
    result = {}
    for k, v in extrap_seen.items():
        total = total_offset + k // 1000
        # Parse 512:xx 8192:xx
        parts = {}
        for p in v.split():
            length, ppl = p.split(':')
            parts[int(length)] = float(ppl)
        result[total] = parts
    return result

# j2 control: first phase offset=5K, 200K phase offset=100K
j2_e1 = get_extrap('logs/pafl_joformer2_from_frozen_control.log', 5)
j2_e2 = get_extrap('logs/pafl_j2_control_200k.log', 100)
j2_extrap = {**j2_e1, **j2_e2}

# m2 control: first phase offset=5K, 200K phase offset=45K
m2_e1 = get_extrap('logs/pafl_monoidal2_from_frozen_control.log', 5)
m2_e2 = get_extrap('logs/pafl_m2_control_200k.log', 45)
m2_extrap = {**m2_e1, **m2_e2}

print("j2 control extrap:")
for t in sorted(j2_extrap):
    e = j2_extrap[t]
    r = e.get(8192, 0) / e.get(512, 1)
    print(f"  {t}K total: 512={e.get(512)}, 8192={e.get(8192)}, ratio={r:.2f}x")

print("\nm2 control extrap:")
for t in sorted(m2_extrap):
    e = m2_extrap[t]
    r = e.get(8192, 0) / e.get(512, 1)
    print(f"  {t}K total: 512={e.get(512)}, 8192={e.get(8192)}, ratio={r:.2f}x")
