#!/bin/bash
# Parse look-ahead JSON results file
# Usage: bash parse_results.sh [json_file]
FILE="${1:-/home/ubuntu/look_ahead5/look_ahead_results_latest.json}"

python3 -c "
import json, sys
with open('$FILE') as f:
    d = json.load(f)
print(f'Config: n_embed={d[\"config\"][\"n_embed\"]}, n_layers={d[\"config\"][\"n_layers\"]}, block_size={d[\"config\"][\"block_size\"]}')
print(f'Models: {d[\"config\"][\"models\"]}')
print()
for name, r in d['results'].items():
    print(f'=== {name} ===')
    print(f'  Val PPL: {r[\"val_ppl\"]:.2f}')
    if 'depth_results' in r:
        for k, v in sorted(r['depth_results'].items(), key=lambda x: (x[0]!='sequential', x[0])):
            if k == 'sequential':
                print(f'  Sequential K=1: {v[\"val_ppl\"]:.2f}')
            else:
                print(f'  Parallel K={k}: {v[\"val_ppl\"]:.2f}')
    if 'diagnostics' in r and r['diagnostics']:
        last = r['diagnostics'][-1]
        if 'empirical_L' in last:
            print(f'  L: {last[\"empirical_L\"]:.4f}')
        if 'avg_contraction_ratios' in last:
            ratios = [f'{x:.4f}' for x in last['avg_contraction_ratios']]
            print(f'  Contraction ratios: {ratios}')
    print()
"
