import re

data = open('logs/pafl_1layer.log').read().replace('\r', '\n')

# Find all unique val_ppl per model
current_model = None
models = {}
for line in data.split('\n'):
    # Detect model name from tqdm prefix
    for name in ['rope', 'joformer_fixed', 'monoidal2', 'joformer2']:
        if line.startswith(name + ':') or line.startswith(name + ' '):
            current_model = name
            break
    if current_model:
        m = re.search(r'val_ppl=([\d.]+)', line)
        if m:
            ppl = m.group(1)
            if current_model not in models:
                models[current_model] = {'ppls': [], 'ppl_set': set(), 'extrap': {}}
            if ppl not in models[current_model]['ppl_set']:
                models[current_model]['ppls'].append(float(ppl))
                models[current_model]['ppl_set'].add(ppl)
        m = re.search(r'extrap iter (\d+): (.+)\]', line)
        if m:
            key = int(m.group(1))
            if key not in models[current_model]['extrap']:
                models[current_model]['extrap'][key] = m.group(2)

# Also check v1 log
try:
    data2 = open('logs/pafl_1layer_v1.log').read().replace('\r', '\n')
    for line in data2.split('\n'):
        for name in ['monoidal', 'joformer']:
            if line.startswith(name + ':') or line.startswith(name + ' '):
                if name == 'joformer' and 'joformer_fixed' in line:
                    continue
                if name == 'joformer' and 'joformer2' in line:
                    continue
                current_model = name
                break
        if current_model in ('monoidal', 'joformer'):
            m = re.search(r'val_ppl=([\d.]+)', line)
            if m:
                ppl = m.group(1)
                if current_model not in models:
                    models[current_model] = {'ppls': [], 'ppl_set': set(), 'extrap': {}}
                if ppl not in models[current_model]['ppl_set']:
                    models[current_model]['ppls'].append(float(ppl))
                    models[current_model]['ppl_set'].add(ppl)
            m = re.search(r'extrap iter (\d+): (.+)\]', line)
            if m:
                key = int(m.group(1))
                if key not in models[current_model]['extrap']:
                    models[current_model]['extrap'][key] = m.group(2)
except FileNotFoundError:
    pass

for name in ['rope', 'joformer_fixed', 'monoidal', 'joformer', 'monoidal2', 'joformer2']:
    if name not in models:
        continue
    m = models[name]
    print(f'{name}:')
    for i, ppl in enumerate(m['ppls']):
        if i == 0: continue
        print(f'  {i*5}K: {ppl}')
    for k in sorted(m['extrap']):
        print(f'  extrap {k//1000}K: {m["extrap"][k]}')
    print()
