import re

def parse_structured(logfile, model_filter, base_offset=100):
    """Parse log with [model] iter N: format. Handles multiple phases."""
    data = open(logfile).read()

    # Extract all (iter, val_ppl) pairs
    raw_vals = []
    for line in data.split('\n'):
        if f'[{model_filter}]' not in line:
            continue
        m = re.search(r'iter (\d+): train PPL [\d.]+, val PPL ([\d.]+)', line)
        if m:
            raw_vals.append((int(m.group(1)), float(m.group(2))))

    # Extract all extrap pairs
    raw_exts = []
    for line in data.split('\n'):
        if f'[{model_filter}]' not in line and model_filter not in line:
            continue
        m = re.search(r'extrap iter (\d+):.*512:([\d.]+).*8192:([\d.]+)', line)
        if m:
            raw_exts.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

    # Split into phases (iter resets to 0)
    def split_phases(items):
        phases = []
        current = []
        for i, item in enumerate(items):
            if i > 0 and item[0] <= items[i-1][0]:
                phases.append(current)
                current = []
            current.append(item)
        if current:
            phases.append(current)
        return phases

    val_phases = split_phases(raw_vals)
    ext_phases = split_phases(raw_exts)

    result = {}
    offset = base_offset
    for pi, phase in enumerate(val_phases):
        for it, ppl in phase:
            k = offset + (it + 500) // 1000
            result[k] = {'val': ppl}
        if phase:
            offset += (phase[-1][0] + 1) // 1000

    offset = base_offset
    for pi, phase in enumerate(ext_phases):
        for it, e512, e8192 in phase:
            k = offset + (it + 500) // 1000
            if k not in result:
                result[k] = {}
            result[k]['extrap'] = (e512, e8192)
        if pi < len(val_phases) and val_phases[pi]:
            offset += (val_phases[pi][-1][0] + 1) // 1000

    return result


def parse_tqdm(logfile, model_filter, base_offset=100, eval_interval=5, phase_sizes=None):
    """Parse log with tqdm val_ppl= format. Handles multi-phase via phase_sizes.
    phase_sizes: list of iters per phase in K, e.g. [50, 50] for two 50K phases.
    Tracks val_ppl transitions (not unique values) to handle duplicate PPLs correctly.
    """
    data = open(logfile).read()

    # Split file into phases using "Resuming from" markers
    sections = data.split('Resuming from:')
    if len(sections) <= 1:
        sections = [data]
    else:
        sections = sections[1:]  # drop preamble before first resume

    result = {}
    offset = base_offset
    for si, section in enumerate(sections):
        # Track val_ppl changes: each time the displayed value changes, it's a new eval
        vals = []
        last_ppl = None
        for line in section.split('\n'):
            if model_filter not in line:
                continue
            m = re.search(r'val_ppl=([\d.]+)', line)
            if m:
                ppl = m.group(1)
                if ppl != last_ppl:
                    vals.append(float(ppl))
                    last_ppl = ppl

        exts = []
        last_ext = None
        for line in section.split('\n'):
            if model_filter not in line or '512:' not in line:
                continue
            m = re.search(r'512:([\d.]+).*8192:([\d.]+)', line)
            if m:
                key = m.group(0)
                if key != last_ext:
                    last_ext = key
                    exts.append((float(m.group(1)), float(m.group(2))))

        for i, v in enumerate(vals):
            if i == 0:
                continue
            k = offset + i * eval_interval
            result[k] = {'val': v}

        for i, (e512, e8192) in enumerate(exts):
            k = offset + (i + 1) * eval_interval
            if k not in result:
                result[k] = {}
            result[k]['extrap'] = (e512, e8192)

        # Advance offset by phase size
        if phase_sizes and si < len(phase_sizes):
            offset += phase_sizes[si]
        elif vals:
            offset += (len(vals) - 1) * eval_interval

    return result


# Parse all schedule runs
models = {
    'random_qk': parse_structured('logs/launch_random_continue_gpu2.log', 'random_ln_indep_qk'),
    'cbd_K4_qk': parse_tqdm('logs/gpu2_queue_sched.log', 'shared_cbd_qk', phase_sizes=[50, 50]),
    'pmlp_qk': parse_tqdm('logs/gpu3_pmlp_sched.log', 'shared_pmlp_qk', phase_sizes=[50, 50]),
    'pemb_qk': parse_tqdm('logs/gpu1_pemb_sched.log', 'shared_pemb_qk', phase_sizes=[50, 50]),
    'cbd_K4_qkv': parse_tqdm('logs/gpu0_cbd_qkv_sched.log', 'shared_cbd_qkv', phase_sizes=[50, 50]),
    'pemb_qkv': parse_tqdm('logs/gpu2_pemb_qkv_sched.log', 'shared_pemb_qkv', phase_sizes=[50, 50]),
    'random_qkv': parse_tqdm('logs/gpu3_random_qkv_sched.log', 'random_ln_indep_qkv', phase_sizes=[50, 50]),
    'lf_qk': {**parse_structured('logs/pafl_shared_lf_qk_h1_150k_v2.log', 'shared_lf_qk'),
               **parse_structured('logs/pafl_shared_lf_qk_h1_200k.log', 'shared_lf_qk', base_offset=150)},
}

# Print table
names = list(models.keys())
header = f"{'Iter':>6}"
for n in names:
    header += f" | {n:>10}"
for n in names:
    header += f" | {n+' ext':>12}"
print("LR Schedule comparison: 100K@5e-4 + 50K@2e-4 + 50K@5e-5")
print(header)
print("-" * len(header))

for k in range(105, 205, 5):
    row = f"{k:>5}K"
    has_data = False
    for n in names:
        d = models[n].get(k)
        if d and 'val' in d:
            row += f" | {d['val']:>10.2f}"
            has_data = True
        else:
            row += f" | {'':>10}"
    for n in names:
        d = models[n].get(k)
        if d and 'extrap' in d:
            ratio = d['extrap'][1] / d['extrap'][0]
            row += f" | {ratio:>11.2f}x"
            has_data = True
        else:
            row += f" | {'':>12}"
    if has_data:
        print(row)
