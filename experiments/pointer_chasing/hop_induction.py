#!/usr/bin/env python3
"""k-hop induction heads task, adapted from chsanford/hop-induction-heads.

Task: given a random character sequence, at each position find the previous
occurrence of the same character and return what follows. Iterate k times.

This is the standard task for demonstrating depth separation in transformers.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os

# Import our blocks for the BPTT model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import RoFormerBlock, FeedForward

# Use HuggingFace GPT2 for transformer baselines (same as original paper)
from transformers import GPT2Model, GPT2Config


class InductionHopsTask:
    """k-hop induction heads task from Sanford et al."""

    def __init__(self, seq_len=100, char_tokens=4, max_hops=3, rng=None):
        self.seq_len = seq_len
        self.char_tokens = char_tokens
        self.max_hops = max_hops
        self.num_tokens = char_tokens + max_hops + 2  # chars + hop indicators + blank + DNE
        self.rng = rng if rng is not None else np.random.RandomState(42)

        self.BLANK = self.num_tokens - 1
        self.DNE = self.num_tokens - 2  # does not exist

    def get_batch(self, batch_size, hops=None):
        """Generate a batch. Returns (inputs, targets) both (B, seq_len-1)."""
        inputs = torch.zeros((batch_size, self.seq_len - 1), dtype=torch.long)
        targets = torch.zeros((batch_size, self.seq_len - 1), dtype=torch.long)

        for i in range(batch_size):
            inp, tgt = self._generate_one(hops=hops)
            inputs[i] = inp
            targets[i] = tgt

        return inputs, targets

    def _generate_one(self, hops=None):
        """Generate one example."""
        # Random character sequence (no consecutive repeats)
        chars = list(range(self.max_hops, self.max_hops + self.char_tokens))
        seq = [self.rng.choice(chars)]
        for _ in range(self.seq_len - 1):
            remaining = [c for c in chars if c != seq[-1]]
            seq.append(self.rng.choice(remaining))

        # Compute hop results for all positions
        hop_results = [seq[:]]  # 0-hop = identity
        for h in range(self.max_hops + 1):
            last = hop_results[-1]
            new = []
            for pos in range(self.seq_len):
                if last[pos] == self.DNE:
                    new.append(self.DNE)
                else:
                    # Find previous occurrence of last[pos] in original seq
                    prev_idx = -1
                    for j in range(pos - 1, -1, -1):
                        if seq[j] == last[pos]:
                            prev_idx = j
                            break
                    if prev_idx == -1 or prev_idx + 1 >= self.seq_len:
                        new.append(self.DNE)
                    else:
                        new.append(seq[prev_idx + 1])
            hop_results.append(new)

        # Pick number of hops
        if hops is None:
            num_hops = self.rng.randint(0, self.max_hops + 1)
        else:
            num_hops = hops

        # Input: [hop_indicator, seq[0], seq[1], ..., seq[-3]]
        # Target: [blank, hop_result[0], hop_result[1], ..., hop_result[-3]]
        input_seq = [num_hops] + seq[:-2]
        target_seq = [self.BLANK] + hop_results[num_hops][:-2]

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


class GPT2Baseline(nn.Module):
    """GPT2 transformer baseline (same as original paper)."""

    def __init__(self, vocab_size, dim_embedding, seq_length, depth, headcount):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=seq_length,
            n_embd=dim_embedding,
            n_layer=depth,
            n_head=headcount,
            use_cache=False,
        )
        self.backbone = GPT2Model(config)
        self.head = nn.Linear(dim_embedding, vocab_size, bias=False)

    def forward(self, xs):
        h = self.backbone(xs).last_hidden_state
        return self.head(h)


class LookAheadD1BPTT(nn.Module):
    """D=1 look-ahead with BPTT for hop induction task."""

    def __init__(self, vocab_size, n_embed, block_size, n_head=4, z_residual=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)
        self.block = RoFormerBlock(n_embed, block_size, 0.0, use_softmax=True, n_head=n_head)
        self.corr_ffn = FeedForward(n_embed, 0.0)
        self.ln_corr = nn.LayerNorm(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
        self.n_embed = n_embed
        self.z_residual = z_residual

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding(idx)
        px_list = []

        # Position 0
        zero = torch.zeros(B, 1, self.n_embed, device=device)
        corr_0 = self.corr_ffn(self.ln_corr(zero + tok_emb[:, 0:1, :]))
        if self.z_residual:
            px_list.append(tok_emb[:, 0:1, :] + zero + corr_0)
        else:
            px_list.append(tok_emb[:, 0:1, :] + corr_0)

        for t in range(1, T):
            px_so_far = torch.cat(px_list, dim=1)
            z = self.block(px_so_far)
            z_prev = z[:, -1:, :]

            corr_t = self.corr_ffn(self.ln_corr(z_prev + tok_emb[:, t:t+1, :]))
            if self.z_residual:
                px_list.append(tok_emb[:, t:t+1, :] + z_prev + corr_t)
            else:
                px_list.append(tok_emb[:, t:t+1, :] + corr_t)

        px_full = torch.cat(px_list, dim=1)
        z_final = self.block(px_full)
        logits = self.head(self.ln_f(z_final))
        return logits


def train_and_eval(model, task, n_iters, batch_size, lr, device, eval_every=500,
                   warmup_steps=1000, max_hops=None):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def lr_func(t):
        if t <= warmup_steps:
            return (t + 1) / warmup_steps
        else:
            return max(0, (n_iters - t) / (n_iters - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    for it in range(1, n_iters + 1):
        model.train()
        inputs, targets = task.get_batch(batch_size)
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_inputs, eval_targets = task.get_batch(256)
                eval_inputs, eval_targets = eval_inputs.to(device), eval_targets.to(device)
                eval_logits = model(eval_inputs)
                eval_error = (eval_logits.argmax(dim=-1) != eval_targets).float().mean().item()

                # Per-hop errors
                hop_errors = {}
                for h in range(task.max_hops + 1):
                    h_inputs, h_targets = task.get_batch(256, hops=h)
                    h_inputs, h_targets = h_inputs.to(device), h_targets.to(device)
                    h_logits = model(h_inputs)
                    h_error = (h_logits.argmax(dim=-1) != h_targets).float().mean().item()
                    hop_errors[h] = h_error

                hop_str = ' '.join(f'{h}h:{e:.3f}' for h, e in hop_errors.items())
                print(f"  iter {it:6d}: loss={loss.item():.4f} err={eval_error:.4f} | {hop_str}")

    # Final per-hop eval
    model.eval()
    final_errors = {}
    with torch.no_grad():
        for h in range(task.max_hops + 1):
            h_inputs, h_targets = task.get_batch(1024, hops=h)
            h_inputs, h_targets = h_inputs.to(device), h_targets.to(device)
            h_logits = model(h_inputs)
            h_error = (h_logits.argmax(dim=-1) != h_targets).float().mean().item()
            final_errors[h] = h_error
    return final_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--char_tokens', type=int, default=4)
    parser.add_argument('--max_hops', type=int, default=3)
    parser.add_argument('--n_embed', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_iters', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--z_residual', action='store_true')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    rng = np.random.RandomState(args.seed)
    task = InductionHopsTask(
        seq_len=args.seq_len, char_tokens=args.char_tokens,
        max_hops=args.max_hops, rng=rng
    )

    block_size = args.seq_len + 10
    print(f"Hop induction: seq_len={args.seq_len}, char_tokens={args.char_tokens}, max_hops={args.max_hops}")
    print(f"Vocab: {task.num_tokens}, n_embed={args.n_embed}, n_head={args.n_head}, device={device}")
    print()

    # Determine which models to run
    if args.run:
        run_models = [m.strip() for m in args.run.split(',')]
    else:
        run_models = [f'N{n}' for n in range(1, 7)] + ['bptt']

    for model_name in run_models:
        if model_name.startswith('N'):
            depth = int(model_name[1:])
            print(f"=== GPT2 depth={depth} ===")
            model = GPT2Baseline(
                task.num_tokens, args.n_embed, args.seq_len,
                depth=depth, headcount=args.n_head
            )
        elif model_name == 'bptt':
            print(f"=== D=1 Look-Ahead BPTT ===")
            model = LookAheadD1BPTT(
                task.num_tokens, args.n_embed, block_size,
                n_head=args.n_head, z_residual=args.z_residual
            )
        elif model_name == 'bptt_zresid':
            print(f"=== D=1 Look-Ahead BPTT + z_residual ===")
            model = LookAheadD1BPTT(
                task.num_tokens, args.n_embed, block_size,
                n_head=args.n_head, z_residual=True
            )
        else:
            print(f"Unknown model: {model_name}")
            continue

        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")

        final_errors = train_and_eval(
            model, task, n_iters=args.n_iters, batch_size=args.batch_size,
            lr=args.lr, device=device, warmup_steps=args.warmup, max_hops=args.max_hops
        )

        print(f"  Final per-hop errors:")
        for h, e in final_errors.items():
            status = "PASS" if e < 0.05 else "FAIL"
            print(f"    {h}-hop: error={e:.4f} [{status}]")
        print()


if __name__ == '__main__':
    main()
