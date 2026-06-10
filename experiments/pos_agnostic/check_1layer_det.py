import re

data = open('logs/pafl_1layer_v1.log').read().replace('\r', '\n')

for model_name in ['shared_det_qk', 'shared_detb_qk']:
    seen = []
    seen_set = set()
    in_model = False
    for line in data.split('\n'):
        if line.startswith(model_name + ':') or line.startswith(model_name + ' '):
            in_model = True
        elif any(line.startswith(m + ':') or line.startswith(m + ' ') for m in ['monoidal:', 'joformer:', 'shared_det_qk:', 'shared_detb_qk:'] if m != model_name + ':'):
            if in_model and not line.startswith(model_name):
                in_model = False
        if in_model:
            m = re.search(r'val_ppl=([\d.]+)', line)
            if m:
                ppl = m.group(1)
                if ppl not in seen_set:
                    seen.append(float(ppl))
                    seen_set.add(ppl)
    print(f'{model_name}:')
    for i, ppl in enumerate(seen):
        if i == 0: continue
        print(f'  {i*5}K: {ppl}')
    print()
