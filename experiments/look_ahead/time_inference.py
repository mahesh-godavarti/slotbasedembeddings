"""Compare wall-clock generation cost of D=1 K=1 C=2048 (look-ahead SA) vs N=6 C=1088 (roformer).

Random weights — we're measuring inference cost, not quality.

Generates 10000 tokens autoregressively (no KV cache; existing generate() reprocesses prefix
each step). Reports cumulative wall-clock at each 100-token boundary.
"""

import os
import sys
import time
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODEL_CLASSES


@torch.no_grad()
def time_generation(model, max_new_tokens, chunk, device, dtype, log_path):
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    chunk_times = []
    autocast = torch.amp.autocast(device_type='cuda', dtype=dtype)

    # Warm-up: one short generate to compile kernels
    with autocast:
        _ = model.generate(idx, 4)
    torch.cuda.synchronize()

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    n_chunks = max_new_tokens // chunk
    cum_start = time.perf_counter()
    last_t = cum_start

    with open(log_path, 'w') as f:
        f.write("chunk_end_token\tchunk_seconds\tcumulative_seconds\tprefix_len_at_chunk_end\n")
        f.flush()

        for c in range(n_chunks):
            with autocast:
                idx = model.generate(idx, chunk)
            torch.cuda.synchronize()
            now = time.perf_counter()
            chunk_dt = now - last_t
            cum_dt = now - cum_start
            last_t = now
            tokens_so_far = (c + 1) * chunk
            prefix_len = idx.shape[1]
            chunk_times.append(chunk_dt)
            f.write(f"{tokens_so_far}\t{chunk_dt:.4f}\t{cum_dt:.4f}\t{prefix_len}\n")
            f.flush()
            if (c + 1) % 5 == 0 or c == 0:
                print(f"  tokens={tokens_so_far:5d}  chunk={chunk_dt:7.3f}s  cum={cum_dt:8.2f}s  T={prefix_len}", flush=True)

    return chunk_times


def build_lookahead(C, block_size, vocab_size, n_head, dropout, device):
    cls = MODEL_CLASSES['block_head_corr_ffn_add']
    # D=1, K=1: n_layers=1 (=K), d_block=1 (=D)
    model = cls(
        vocab_size=vocab_size,
        n_embed=C,
        n_layers=1,                # K (n_iters)
        block_size=block_size,
        dropout=dropout,
        use_softmax=True,
        d_block=1,                 # D
        n_head=n_head,
    ).to(device)
    return model


def build_roformer(C, block_size, vocab_size, n_head, dropout, n_layers, device):
    cls = MODEL_CLASSES['roformer']
    model = cls(
        vocab_size=vocab_size,
        n_embed=C,
        n_layers=n_layers,
        block_size=block_size,
        dropout=dropout,
        use_softmax=True,
        n_head=n_head,
    ).to(device)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max_new_tokens', type=int, default=10000)
    p.add_argument('--chunk', type=int, default=100)
    p.add_argument('--block_size', type=int, default=10100)
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--n_head', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--model', choices=['lookahead', 'roformer', 'both'], default='both')
    p.add_argument('--logdir', type=str, default='/home/ubuntu/look_ahead8/logs')
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device(f'cuda:{args.gpu}')
    dtype = torch.bfloat16

    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}", flush=True)
    print(f"block_size={args.block_size}, vocab={args.vocab_size}, n_head={args.n_head}, dtype=bfloat16", flush=True)
    print(f"Generating {args.max_new_tokens} tokens, logging every {args.chunk}", flush=True)

    if args.model in ('lookahead', 'both'):
        print("\n=== D=1 K=1 C=2048 (block_head_corr_ffn_add) ===", flush=True)
        m = build_lookahead(C=2048, block_size=args.block_size, vocab_size=args.vocab_size,
                            n_head=args.n_head, dropout=args.dropout, device=device)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  params: {n_params:,}", flush=True)
        log = os.path.join(args.logdir, 'time_d1_k1_c2048.tsv')
        time_generation(m, args.max_new_tokens, args.chunk, device, dtype, log)
        print(f"  log: {log}", flush=True)
        del m
        torch.cuda.empty_cache()

    if args.model in ('roformer', 'both'):
        print("\n=== N=6 C=1088 (roformer) ===", flush=True)
        m = build_roformer(C=1088, block_size=args.block_size, vocab_size=args.vocab_size,
                           n_head=args.n_head, dropout=args.dropout, n_layers=6, device=device)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  params: {n_params:,}", flush=True)
        log = os.path.join(args.logdir, 'time_n6_c1088.tsv')
        time_generation(m, args.max_new_tokens, args.chunk, device, dtype, log)
        print(f"  log: {log}", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
