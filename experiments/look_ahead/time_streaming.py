"""Time streaming (KV-cached) autoregressive decode.

Usage:
    python time_streaming.py --model lookahead --gpu 0
    python time_streaming.py --model roformer  --gpu 1
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_streaming import StreamingRoFormer, StreamingLookAhead


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['lookahead', 'roformer'], required=True)
    p.add_argument('--n_embed', type=int, required=True, help='C')
    p.add_argument('--n_layers', type=int, required=True, help='D for lookahead, N for roformer')
    p.add_argument('--max_new_tokens', type=int, default=10000)
    p.add_argument('--chunk', type=int, default=100)
    p.add_argument('--max_seq_len', type=int, default=10100)
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--n_head', type=int, default=16)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--logdir', type=str, default='/home/ubuntu/look_ahead8/logs')
    p.add_argument('--tag', type=str, default='')
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device(f'cuda:{args.gpu}')
    dtype = torch.bfloat16

    if args.model == 'lookahead':
        name = f'd{args.n_layers}_k1_c{args.n_embed}'
        m = StreamingLookAhead(args.vocab_size, n_embed=args.n_embed, d_block=args.n_layers,
                                block_size=args.max_seq_len, n_head=args.n_head).to(device)
    else:
        name = f'n{args.n_layers}_c{args.n_embed}'
        m = StreamingRoFormer(args.vocab_size, n_embed=args.n_embed, n_layers=args.n_layers,
                              block_size=args.max_seq_len, n_head=args.n_head).to(device)
    m.eval()

    n_params = sum(pr.numel() for pr in m.parameters())
    print(f"GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}", flush=True)
    print(f"[{name}] params: {n_params:,}", flush=True)
    print(f"[{name}] decode {args.max_new_tokens} tokens, chunk={args.chunk}, max_seq_len={args.max_seq_len}", flush=True)

    log_path = os.path.join(args.logdir, f"stream_{name}{args.tag}.tsv")
    fh = open(log_path, 'w')
    fh.write("tokens_so_far\tchunk_seconds\tcumulative_seconds\tprefix_len\n")
    fh.flush()

    def on_chunk(tokens_so_far, chunk_dt, cum_dt, prefix_len):
        fh.write(f"{tokens_so_far}\t{chunk_dt:.4f}\t{cum_dt:.4f}\t{prefix_len}\n")
        fh.flush()
        if tokens_so_far % 500 == 0 or tokens_so_far == args.chunk:
            print(f"[{name}] tokens={tokens_so_far:5d} chunk={chunk_dt:7.4f}s cum={cum_dt:8.2f}s T={prefix_len}", flush=True)

    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Warm-up
    _ = m.generate(prompt, max_new_tokens=8, max_seq_len=args.max_seq_len,
                   dtype=dtype, on_chunk=None, chunk=10**9)
    torch.cuda.synchronize()

    _ = m.generate(prompt, max_new_tokens=args.max_new_tokens, max_seq_len=args.max_seq_len,
                   dtype=dtype, on_chunk=on_chunk, chunk=args.chunk)

    fh.close()
    print(f"[{name}] log: {log_path}", flush=True)


if __name__ == '__main__':
    main()
