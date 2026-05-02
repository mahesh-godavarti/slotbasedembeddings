"""End-to-end verification: load a real trained checkpoint into BOTH the original
BlockHeadCorrFFNAddModel and the StreamingLookAhead, run them on the same input,
and assert logits match.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODEL_CLASSES
from models_streaming import StreamingLookAhead, StreamingRoFormer
from blocks import RoFormer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model', required=True, help="model name (e.g. 'block_head_corr_ffn_add' or 'roformer')")
    p.add_argument('--n_embed', type=int, required=True)
    p.add_argument('--n_layers', type=int, required=True, help='for D>1 look-ahead: n_layers = d_block * n_iters_train')
    p.add_argument('--d_block', type=int, default=1)
    p.add_argument('--block_size', type=int, required=True)
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--n_head', type=int, default=16)
    p.add_argument('--T', type=int, default=64, help='test sequence length (≤ block_size)')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    print(f"Loaded {args.ckpt}")
    print(f"  iter={ck.get('iter')}, val_ppl={ck.get('val_ppl')}")

    if args.model == 'roformer':
        ref = RoFormer(args.vocab_size, args.n_embed, args.n_layers, args.block_size,
                       dropout=0.0, use_softmax=True, n_head=args.n_head)
        stream = StreamingRoFormer(args.vocab_size, args.n_embed, args.n_layers, args.block_size,
                                    dropout=0.0, use_softmax=True, n_head=args.n_head)
    else:
        cls = MODEL_CLASSES[args.model]
        ref = cls(vocab_size=args.vocab_size, n_embed=args.n_embed, n_layers=args.n_layers,
                  block_size=args.block_size, dropout=0.0, use_softmax=True,
                  d_block=args.d_block, n_head=args.n_head)
        stream = StreamingLookAhead(args.vocab_size, args.n_embed, d_block=args.d_block,
                                     block_size=args.block_size, dropout=0.0, use_softmax=True,
                                     n_head=args.n_head)

    ref_missing, ref_unexpected = ref.load_state_dict(sd, strict=True)
    stream_missing, stream_unexpected = stream.load_state_dict(sd, strict=True)
    print(f"  ref load:    missing={list(ref_missing)}, unexpected={list(ref_unexpected)}")
    print(f"  stream load: missing={list(stream_missing)}, unexpected={list(stream_unexpected)}")
    ref.to(device).eval()
    stream.to(device).eval()

    n_params = sum(p.numel() for p in stream.parameters())
    print(f"  params: {n_params:,}")

    torch.manual_seed(0)
    idx = torch.randint(0, args.vocab_size, (1, args.T), device=device)

    with torch.no_grad():
        if args.model == 'roformer':
            ref_logits, _ = ref(idx)
        else:
            ref_logits, _ = ref.forward_sequential(idx, seq_k=1)
        stream_logits = stream.streaming_forward_logits(idx, max_seq_len=args.block_size,
                                                         dtype=torch.float32)

    diff = (ref_logits - stream_logits).abs()
    print(f"\n  ref logits norm: {ref_logits.abs().max().item():.3f}")
    print(f"  max |Δ|:         {diff.max().item():.3e}")
    print(f"  mean|Δ|:         {diff.mean().item():.3e}")
    print(f"  rel max:         {(diff.max() / (ref_logits.abs().max() + 1e-9)).item():.3e}")

    ref_top = ref_logits.argmax(-1)
    stream_top = stream_logits.argmax(-1)
    agree = (ref_top == stream_top).float().mean().item()
    print(f"  argmax agreement: {agree*100:.1f}% ({(ref_top==stream_top).sum().item()}/{ref_top.numel()})")

    if diff.max().item() < 1e-2:
        print("\n  PASS: streaming output matches reference within tolerance.")
    else:
        print(f"\n  FAIL: max diff {diff.max().item():.3e} exceeds tolerance.")


if __name__ == '__main__':
    main()
