"""Verify state_dict compatibility and numerical equivalence:

  StreamingRoFormer.streaming_forward_logits(idx)  ==  RoFormer.forward(idx)
  StreamingLookAhead.streaming_forward_logits(idx) ==  BlockHeadCorrFFNAddModel.forward_sequential(idx, seq_k=1)

Random init (seeded), small dims, CPU, fp32.

If max-abs logit diff is small (< ~1e-4) we have a drop-in replacement.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import RoFormer
from models import MODEL_CLASSES
from models_streaming import StreamingRoFormer, StreamingLookAhead


def verify_roformer():
    print("\n=== StreamingRoFormer  ↔  RoFormer ===")
    torch.manual_seed(0)
    vocab_size = 64
    n_embed = 32
    n_layers = 3
    block_size = 16
    n_head = 2
    T = 10

    ref = RoFormer(vocab_size, n_embed, n_layers, block_size, dropout=0.0,
                   use_softmax=True, n_head=n_head).eval()
    stream = StreamingRoFormer(vocab_size, n_embed, n_layers, block_size, dropout=0.0,
                               use_softmax=True, n_head=n_head).eval()

    # Copy state_dict from reference into streaming model.
    missing, unexpected = stream.load_state_dict(ref.state_dict(), strict=True)
    print(f"  load_state_dict (strict=True): missing={list(missing)}, unexpected={list(unexpected)}")

    idx = torch.randint(0, vocab_size, (1, T))

    with torch.no_grad():
        ref_logits, _ = ref(idx)                                    # (1, T, V)
        stream_logits = stream.streaming_forward_logits(idx, max_seq_len=block_size, dtype=torch.float32)

    diff = (ref_logits - stream_logits).abs()
    print(f"  ref_logits shape: {tuple(ref_logits.shape)}, stream shape: {tuple(stream_logits.shape)}")
    print(f"  max |Δ|: {diff.max().item():.3e}")
    print(f"  mean|Δ|: {diff.mean().item():.3e}")
    print(f"  rel    : {(diff.max() / (ref_logits.abs().max() + 1e-9)).item():.3e}")
    assert diff.max().item() < 1e-3, f"RoFormer streaming mismatch: max diff = {diff.max().item()}"
    print("  PASS")


def verify_lookahead():
    print("\n=== StreamingLookAhead  ↔  BlockHeadCorrFFNAddModel.forward_sequential(seq_k=1) ===")
    torch.manual_seed(0)
    vocab_size = 64
    n_embed = 32
    n_iters = 3              # K_train (must be > 1 to exercise forward_sequential's body)
    block_size = 16
    n_head = 2
    T = 10

    cls = MODEL_CLASSES['block_head_corr_ffn_add']
    ref = cls(vocab_size=vocab_size, n_embed=n_embed, n_layers=n_iters,
              block_size=block_size, dropout=0.0, use_softmax=True, d_block=1,
              n_head=n_head).eval()

    stream = StreamingLookAhead(vocab_size, n_embed, d_block=1, block_size=block_size,
                                 dropout=0.0, use_softmax=True, n_head=n_head).eval()

    missing, unexpected = stream.load_state_dict(ref.state_dict(), strict=True)
    print(f"  load_state_dict (strict=True): missing={list(missing)}, unexpected={list(unexpected)}")

    idx = torch.randint(0, vocab_size, (1, T))

    with torch.no_grad():
        ref_logits, _ = ref.forward_sequential(idx, seq_k=1)        # (1, T, V)
        stream_logits = stream.streaming_forward_logits(idx, max_seq_len=block_size, dtype=torch.float32)

    diff = (ref_logits - stream_logits).abs()
    print(f"  ref_logits shape: {tuple(ref_logits.shape)}, stream shape: {tuple(stream_logits.shape)}")
    print(f"  max |Δ|: {diff.max().item():.3e}")
    print(f"  mean|Δ|: {diff.mean().item():.3e}")
    print(f"  rel    : {(diff.max() / (ref_logits.abs().max() + 1e-9)).item():.3e}")
    assert diff.max().item() < 1e-3, f"LookAhead streaming mismatch: max diff = {diff.max().item()}"
    print("  PASS")


def verify_lookahead_d2():
    print("\n=== StreamingLookAhead (D=2)  ↔  BlockHeadCorrFFNAddModel(d_block=2).forward_sequential ===")
    torch.manual_seed(0)
    vocab_size = 64
    n_embed = 32
    d_block = 2
    n_iters_train = 2
    n_layers = d_block * n_iters_train  # codebase: n_layers = d_block * n_iters
    block_size = 16
    n_head = 2
    T = 10

    cls = MODEL_CLASSES['block_head_corr_ffn_add']
    ref = cls(vocab_size=vocab_size, n_embed=n_embed, n_layers=n_layers,
              block_size=block_size, dropout=0.0, use_softmax=True, d_block=d_block,
              n_head=n_head).eval()

    stream = StreamingLookAhead(vocab_size, n_embed, d_block=d_block, block_size=block_size,
                                 dropout=0.0, use_softmax=True, n_head=n_head).eval()

    missing, unexpected = stream.load_state_dict(ref.state_dict(), strict=True)
    print(f"  load_state_dict (strict=True): missing={list(missing)}, unexpected={list(unexpected)}")

    idx = torch.randint(0, vocab_size, (1, T))

    with torch.no_grad():
        ref_logits, _ = ref.forward_sequential(idx, seq_k=1)
        stream_logits = stream.streaming_forward_logits(idx, max_seq_len=block_size, dtype=torch.float32)

    diff = (ref_logits - stream_logits).abs()
    print(f"  max |Δ|: {diff.max().item():.3e}")
    print(f"  mean|Δ|: {diff.mean().item():.3e}")
    assert diff.max().item() < 1e-3, f"D=2 streaming mismatch: max diff = {diff.max().item()}"
    print("  PASS")


if __name__ == '__main__':
    verify_roformer()
    verify_lookahead()
    verify_lookahead_d2()
    print("\nAll verifications passed.")
