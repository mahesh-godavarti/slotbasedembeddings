import re

def get_ordered_ppls(logfiles, eval_interval=5, model_filter=None, offsets=None):
    """Extract val PPLs from one or more log files, return {total_iter_K: ppl}.
    offsets: optional dict {logfile: offset_k} to override auto-computed offset.
    """
    result = {}
    offset = 0
    for logfile in logfiles:
        if offsets and logfile in offsets:
            offset = offsets[logfile]
        try:
            data = open(logfile).read().replace('\r', '\n')
        except FileNotFoundError:
            continue

        # Try structured format first: [model] iter N: train PPL X, val PPL Y
        structured = []
        for line in data.split('\n'):
            if model_filter and f'[{model_filter}]' not in line:
                continue
            m = re.search(r'iter (\d+): train PPL [\d.]+, val PPL ([\d.]+)', line)
            if m:
                structured.append((int(m.group(1)), float(m.group(2))))

        if structured:
            for it, ppl in structured:
                k = offset + (it + 500) // 1000  # round to nearest K
                if k > 0:
                    result[k] = ppl
            last_it = structured[-1][0]
            offset += (last_it + 500) // 1000
        else:
            # Fallback: tqdm postfix val_ppl= unique values
            seen = []
            seen_set = set()
            for line in data.split('\n'):
                if model_filter and model_filter not in line:
                    continue
                m = re.search(r'val_ppl=([\d.]+)', line)
                if m:
                    ppl = m.group(1)
                    if ppl not in seen_set:
                        seen.append(float(ppl))
                        seen_set.add(ppl)
            for i, ppl in enumerate(seen):
                if i == 0: continue
                result[offset + i * eval_interval] = ppl
            if seen:
                offset += (len(seen) - 1) * eval_interval
    return result

models = {
    'RoPE': get_ordered_ppls(['logs/pafl_rope_5e4_200k.log'], model_filter='rope'),
    'jfixed': get_ordered_ppls(['logs/pafl_jfixed_5e4_200k.log', 'logs/gpu1_queue.log'],
                               model_filter='joformer_fixed',
                               offsets={'logs/gpu1_queue.log': 162}),
    'random_qk': get_ordered_ppls(['logs/pafl_random_indep_5e4_200k.log', 'logs/gpu1_queue.log'],
                                  model_filter='random_ln_indep_qk',
                                  offsets={'logs/gpu1_queue.log': 173}),
    'random_qkv': get_ordered_ppls(['logs/pafl_random_ln_indep_qkv.log', 'logs/gpu2_queue.log'],
                                   model_filter='random_ln_indep_qkv'),
    'pemb_qk': get_ordered_ppls(['logs/pafl_pemb_deti_v2.log', 'logs/gpu3_queue_v2.log'],
                                model_filter='shared_pemb_qk'),
    'pemb_qkv': get_ordered_ppls(['logs/pafl_shared_pemb_qkv.log', 'logs/gpu1_queue.log'],
                                 model_filter='shared_pemb_qkv'),
    'cbd_K4_qk': get_ordered_ppls(['logs/pafl_shared_cbd_qk_K4.log', 'logs/pafl_shared_cbd_qk_K4_100k.log',
                                    'logs/gpu2_queue.log'],
                                   model_filter='shared_cbd_qk'),
    'cbd_K8_qk': get_ordered_ppls(['logs/pafl_shared_cbd_qk_K8.log', 'logs/pafl_cbd_qk_K8_200k.log'],
                                  model_filter='shared_cbd_qk'),
    'cbd_K8_qkv': get_ordered_ppls(['logs/gpu3_queue_v2.log'],
                                   model_filter='shared_cbd_qkv'),
    'pmlp_qk': get_ordered_ppls(['logs/pafl_shared_pmlp_qk_v5.log'],
                                model_filter='shared_pmlp_qk', eval_interval=1),
}

# Print at 10K intervals
iters = list(range(10, 205, 10))

header = f"{'Iter':>6}"
for name in models:
    header += f" | {name:>10}"
print(header)
print("-" * len(header))

for k in iters:
    row = f"{k:>5}K"
    for name in models:
        if k in models[name]:
            row += f" | {models[name][k]:>10.2f}"
        else:
            row += f" | {'':>10}"
    if any(k in models[name] for name in models):
        print(row)
