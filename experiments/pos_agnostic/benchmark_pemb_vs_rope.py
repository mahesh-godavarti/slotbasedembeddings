#!/usr/bin/env python3
"""Benchmark pemb vs RoPE with torch.compile."""

import torch
import time
import sys
import argparse
sys.path.insert(0, '.')
from models import GPTModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    batch_size = 4
    seq_lengths = [512, 1024, 2048, 4096, 8192]

    for model_name in ['rope', 'shared_pemb_qk', 'joformer_fixed']:
        print(f'\n{model_name} (torch.compile + SDPA):')
        print(f'{"Seq Len":>8} | {"Time (ms)":>12}')
        print('-' * 25)
        for seq_len in seq_lengths:
            torch.manual_seed(42)
            model = GPTModel(32000, 768, 16, 8, seq_len, dropout=0.0,
                           attn_config=model_name, window_size=999999)
            model = torch.compile(model.to(device).eval())
            idx = torch.randint(0, 32000, (batch_size, seq_len), device=device)
            targets = torch.randint(0, 32000, (batch_size, seq_len), device=device)
            # Warmup
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for _ in range(10):
                    model(idx, targets)
            torch.cuda.synchronize()
            # Timed
            start = time.time()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for _ in range(20):
                    model(idx, targets)
            torch.cuda.synchronize()
            t = (time.time() - start) / 20 * 1000
            print(f'{seq_len:>8} | {t:>10.1f}ms')
            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
