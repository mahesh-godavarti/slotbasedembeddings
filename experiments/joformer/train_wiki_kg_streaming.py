#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Mahesh Godavarti. All Rights Reserved.
#
# License: This software is provided for non-commercial research purposes only.
# Any commercial use, including but not limited to use in a product, service,
# or for-profit research, is strictly prohibited without explicit written
# permission from the copyright holder.
#
# Patent Pending: Certain aspects of this software are the subject of a
# pending patent application.
#
# Contact: m@qalaxia.com
# -----------------------------------------------------------------------------
#
# train_wiki_kg_streaming.py — Joint KG + wiki text training with memmap
#
# Extends train_wiki_streaming.py with knowledge graph training using the
# native angle-gap mechanism from models E/H/I in kg_text_experiment.py.
# Relations are NOT tokenized — they are learned angle parameters inserted
# as gaps in the cumulative angle sum. Text PPL is the primary metric.
#
# Usage:
#   python train_wiki_kg_streaming.py preprocess --wiki_path PATH --vocab_size 16000
#   python train_wiki_kg_streaming.py train --data_dir joformer/data_kg ...
#   python train_wiki_kg_streaming.py auto --wiki_path PATH --vocab_size 16000 ...

import argparse
import json
import math
import os
import pickle
import random
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_wiki import (
    MODEL_CLASSES, FeedForward,
    build_rotation_matrix, apply_rotation, apply_inverse_rotation,
    JoFormerLearnedAttention, JoFormerLearnedBlock,
)

# ---------------------------------------------------------------------------
# KG Data Loading (adapted from exp8/word_experiment.py)
# ---------------------------------------------------------------------------

def load_wordnet_synonyms(path):
    """Load WordNet synonyms file. Format: word syn1 syn2 ... per line."""
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if len(words) < 2:
                continue
            head = words[0]
            for syn in words[1:]:
                triples.append((head, "synonym_of", syn))
    return triples


def load_framenet(path):
    """Load FrameNet relations. Format: word rel1 rel2 ... per line."""
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if len(words) < 2:
                continue
            head = words[0]
            for related in words[1:]:
                triples.append((head, "frame_related", related))
    return triples


def load_bats_analogies(bats_dir):
    """Load BATS 3.0 analogies from directory."""
    triples = []
    for category_dir in sorted(os.listdir(bats_dir)):
        cat_path = os.path.join(bats_dir, category_dir)
        if not os.path.isdir(cat_path):
            continue
        for fname in sorted(os.listdir(cat_path)):
            if not fname.endswith('.txt'):
                continue
            rel_name = fname.replace('.txt', '').strip()
            if '[' in rel_name and ']' in rel_name:
                rel_name = rel_name[rel_name.index('[') + 1:rel_name.index(']')].strip()
            rel_name = rel_name.replace(' ', '_').replace('-', '_').lower()
            fpath = os.path.join(cat_path, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        head, tail = parts[0].strip(), parts[1].strip()
                        for t in tail.split('/'):
                            t = t.strip()
                            if t:
                                triples.append((head, rel_name, t))
    return triples


def load_google_analogies(path):
    """Load Google analogy questions. Format: : category then A B C D lines."""
    triples = []
    current_rel = "unknown"
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith(':'):
                current_rel = line[1:].strip().replace(' ', '_').lower()
                continue
            parts = line.split()
            if len(parts) == 4:
                a, b, c, d = parts
                triples.append((a, current_rel, b))
                triples.append((c, current_rel, d))
    return list(set(triples))


def load_word_analogies(path):
    """Load word analogies. Format: singular plural singular plural per line."""
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                s1, p1, s2, p2 = parts
                triples.append((s1, "plural_of", p1))
                triples.append((s2, "plural_of", p2))
    return list(set(triples))


def load_all_kg_triples(args):
    """Load KG triples from all available data sources."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'exp8', 'data')

    all_triples = []

    # WordNet synonyms
    wordnet_path = args.wordnet_path or os.path.join(data_dir, 'wordnet-synonyms.txt')
    if os.path.exists(wordnet_path):
        t = load_wordnet_synonyms(wordnet_path)
        print(f"  WordNet synonyms: {len(t):,} triples")
        all_triples.extend(t)

    # FrameNet
    framenet_path = args.framenet_path or os.path.join(data_dir, 'framenet.txt')
    if os.path.exists(framenet_path):
        t = load_framenet(framenet_path)
        print(f"  FrameNet: {len(t):,} triples")
        all_triples.extend(t)

    # BATS analogies
    bats_dir = args.bats_dir or os.path.join(data_dir, 'BATS_3.0')
    if os.path.exists(bats_dir):
        t = load_bats_analogies(bats_dir)
        print(f"  BATS analogies: {len(t):,} triples")
        all_triples.extend(t)

    # Google analogies
    google_path = args.google_path or os.path.join(data_dir, 'questions-words_for_training.txt')
    if os.path.exists(google_path):
        t = load_google_analogies(google_path)
        print(f"  Google analogies: {len(t):,} triples")
        all_triples.extend(t)

    # Word analogies
    analogies_path = args.analogies_path or os.path.join(data_dir, 'wordanalogies.txt')
    if os.path.exists(analogies_path):
        t = load_word_analogies(analogies_path)
        print(f"  Word analogies: {len(t):,} triples")
        all_triples.extend(t)

    # Deduplicate
    all_triples = list(set(all_triples))
    print(f"  Total unique triples: {len(all_triples):,}")

    return all_triples


# ---------------------------------------------------------------------------
# KGDataset (adapted from exp8/word_experiment.py, uses BPE tokenizer)
# ---------------------------------------------------------------------------

class KGDataset:
    """Dataset for KG training with BPE-encoded entities."""

    def __init__(self, encoded_triples, relations, pad_id=0):
        """
        Args:
            encoded_triples: list of dicts with keys:
                'head_ids': list of int (BPE token IDs)
                'tail_ids': list of int
                'rel': str (relation name)
            relations: list of unique relation names
            pad_id: padding token ID
        """
        self.relations = relations
        self.rel_to_idx = {r: i for i, r in enumerate(relations)}
        self.pad_id = pad_id

        # Sort by total sequence length for length-bucketed batching
        self.encoded = sorted(encoded_triples,
                              key=lambda e: len(e['head_ids']) + len(e['tail_ids']))
        self.n = len(self.encoded)

    def get_causal_batch(self, batch_size, device):
        """Get a length-bucketed causal batch with 50% direction flipping.

        Picks a random start index and takes a contiguous chunk of batch_size
        triples (which have similar lengths due to sorting), minimizing padding.

        Returns:
            token_ids: (B, T) BPE token IDs (head + tail concatenated)
            targets: (B, T) next-token targets, -100 for padding
            head_lens: list of int, length of head portion
            rel_names: list of str, relation names
            negate_angles: list of bool, True if direction was flipped
        """
        # Sample a contiguous chunk from length-sorted triples
        start = random.randint(0, self.n - batch_size)
        batch = self.encoded[start:start + batch_size]

        seqs = []
        head_lens = []
        negate_angles = []
        rel_names = []

        for b in batch:
            flip = random.random() < 0.5
            if flip:
                seq = b['tail_ids'] + b['head_ids']
                head_lens.append(len(b['tail_ids']))
            else:
                seq = b['head_ids'] + b['tail_ids']
                head_lens.append(len(b['head_ids']))
            seqs.append(seq)
            negate_angles.append(flip)
            rel_names.append(b['rel'])

        max_len = max(len(s) for s in seqs)

        token_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            token_ids[i, :len(seq)] = seq_t
            targets[i, :len(seq) - 1] = seq_t[1:]

        return (token_ids.to(device), targets.to(device),
                head_lens, rel_names, negate_angles)


# ---------------------------------------------------------------------------
# Phase 1: Preprocessing (constant memory)
# ---------------------------------------------------------------------------

def train_bpe_tokenizer_streaming(wiki_path, vocab_size, max_lines=None,
                                   kg_entity_words=None):
    """Train BPE tokenizer by streaming wiki file, optionally including KG entity words.

    Returns: (tokenizer, actual_vocab_size, line_count)
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                     encoding='utf-8') as f:
        tmp_path = f.name
        count = 0
        with open(wiki_path, 'r', encoding='utf-8') as src:
            for i, line in enumerate(src):
                if max_lines and i >= max_lines:
                    break
                stripped = line.strip()
                if stripped:
                    f.write(stripped + '\n')
                    count += 1

        # Append KG entity words so BPE learns to tokenize them well
        if kg_entity_words:
            for word in kg_entity_words:
                f.write(word + '\n')

    print(f"  Filtered {count:,} wiki lines + {len(kg_entity_words) if kg_entity_words else 0} KG words")

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>"],
    )
    tokenizer.train([tmp_path], trainer)
    os.unlink(tmp_path)

    actual_vocab_size = tokenizer.get_vocab_size()
    return tokenizer, actual_vocab_size, count


def tokenize_to_disk(tokenizer, wiki_path, output_bin_path, max_lines=None):
    """Tokenize wiki text line-by-line, writing int32 IDs to binary file."""
    total_tokens = 0
    with open(wiki_path, 'r', encoding='utf-8') as src, \
         open(output_bin_path, 'wb') as dst:
        for i, line in enumerate(tqdm(src, desc="Tokenizing", unit=" lines")):
            if max_lines and i >= max_lines:
                break
            stripped = line.strip()
            if stripped:
                enc = tokenizer.encode(stripped)
                ids = enc.ids
                if ids:
                    chunk = np.array(ids, dtype=np.int32)
                    dst.write(chunk.tobytes())
                    total_tokens += len(ids)
    return total_tokens


def encode_kg_triples(tokenizer, triples):
    """Encode KG triples using BPE tokenizer.

    Returns:
        encoded: list of dicts with 'head_ids', 'tail_ids', 'rel'
        relations: sorted list of unique relation names
    """
    relations = sorted(set(r for _, r, _ in triples))
    encoded = []
    for head, rel, tail in triples:
        head_ids = tokenizer.encode(head).ids
        tail_ids = tokenizer.encode(tail).ids
        if head_ids and tail_ids:
            encoded.append({
                'head_ids': head_ids,
                'tail_ids': tail_ids,
                'rel': rel,
            })
    return encoded, relations


def preprocess(args):
    """Run the full preprocessing pipeline: BPE training + tokenization + KG encoding."""
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    bin_path = os.path.join(data_dir, 'wiki_tokens.bin')
    meta_path = os.path.join(data_dir, 'wiki_tokens.meta')
    tok_path = os.path.join(data_dir, 'wiki_tokenizer.json')
    kg_path = os.path.join(data_dir, 'kg_triples.pkl')
    kg_meta_path = os.path.join(data_dir, 'kg_meta.json')

    print(f"Preprocessing wiki text: {args.wiki_path}")
    print(f"  max_lines={args.wiki_lines}, vocab_size={args.vocab_size}")
    print(f"  Output dir: {data_dir}")

    # Load KG triples for entity word extraction
    print("\n[0/3] Loading KG triples...")
    kg_triples = load_all_kg_triples(args)

    # Collect unique entity words for BPE training
    entity_words = set()
    for h, r, t in kg_triples:
        entity_words.add(h)
        entity_words.add(t)
    entity_words = sorted(entity_words)
    print(f"  Unique KG entities: {len(entity_words):,}")

    # Step 1: Train BPE tokenizer (streaming + KG entities)
    print("\n[1/3] Training BPE tokenizer...")
    t0 = time.time()
    tokenizer, actual_vocab_size, line_count = train_bpe_tokenizer_streaming(
        args.wiki_path, args.vocab_size, args.wiki_lines,
        kg_entity_words=entity_words
    )
    print(f"  Vocab size: {actual_vocab_size}, trained in {time.time()-t0:.1f}s")
    tokenizer.save(tok_path)
    print(f"  Tokenizer saved to {tok_path}")

    # Step 2: Tokenize wiki to binary (streaming)
    print("\n[2/3] Tokenizing corpus to binary...")
    t0 = time.time()
    total_tokens = tokenize_to_disk(
        tokenizer, args.wiki_path, bin_path, args.wiki_lines
    )
    dt = time.time() - t0
    file_size_gb = os.path.getsize(bin_path) / (1024**3)
    print(f"  {total_tokens:,} tokens written in {dt:.1f}s")
    print(f"  Binary file: {bin_path} ({file_size_gb:.2f} GB)")

    # Save wiki metadata
    meta = {
        'total_tokens': total_tokens,
        'vocab_size': actual_vocab_size,
        'source': os.path.abspath(args.wiki_path),
        'max_lines': args.wiki_lines,
        'line_count': line_count,
        'dtype': 'int32',
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    # Step 3: Encode KG triples
    print("\n[3/3] Encoding KG triples...")
    t0 = time.time()
    encoded_triples, relations = encode_kg_triples(tokenizer, kg_triples)
    print(f"  Encoded {len(encoded_triples):,} triples, {len(relations)} relations in {time.time()-t0:.1f}s")

    with open(kg_path, 'wb') as f:
        pickle.dump(encoded_triples, f)
    print(f"  KG triples saved to {kg_path}")

    kg_meta = {
        'n_triples': len(encoded_triples),
        'n_relations': len(relations),
        'relations': relations,
    }
    with open(kg_meta_path, 'w') as f:
        json.dump(kg_meta, f, indent=2)
    print(f"  KG metadata saved to {kg_meta_path}")

    print(f"\nPreprocessing complete.")
    return meta


# ---------------------------------------------------------------------------
# Phase 2: Training with memmap
# ---------------------------------------------------------------------------

def load_memmap_data(data_dir):
    """Load preprocessed text data as memory-mapped numpy array + KG data."""
    meta_path = os.path.join(data_dir, 'wiki_tokens.meta')
    bin_path = os.path.join(data_dir, 'wiki_tokens.bin')
    tok_path = os.path.join(data_dir, 'wiki_tokenizer.json')
    kg_path = os.path.join(data_dir, 'kg_triples.pkl')
    kg_meta_path = os.path.join(data_dir, 'kg_meta.json')

    with open(meta_path) as f:
        meta = json.load(f)

    total_tokens = meta['total_tokens']
    data = np.memmap(bin_path, dtype=np.int32, mode='r', shape=(total_tokens,))

    n = int(total_tokens * 0.9)
    train_data = data[:n]
    val_data = data[n:]

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tok_path)

    # Load KG data
    kg_dataset = None
    kg_meta = None
    if os.path.exists(kg_path) and os.path.exists(kg_meta_path):
        with open(kg_path, 'rb') as f:
            encoded_triples = pickle.load(f)
        with open(kg_meta_path) as f:
            kg_meta = json.load(f)
        pad_id = tokenizer.token_to_id("<PAD>") or 0
        kg_dataset = KGDataset(encoded_triples, kg_meta['relations'], pad_id=pad_id)

    return train_data, val_data, tokenizer, meta, kg_dataset, kg_meta


def get_batch(train_data, val_data, split, block_size, batch_size, device):
    """Random-access batch from memory-mapped data."""
    data = train_data if split == "train" else val_data
    n = len(data) - block_size
    ix = torch.randint(0, n, (batch_size,)).numpy()

    sequences = np.stack([data[i:i + block_size + 1] for i in ix])
    sequences = torch.from_numpy(sequences.astype(np.int64)).to(device)

    x = sequences[:, :block_size].contiguous()
    y = sequences[:, 1:block_size + 1].contiguous()
    return x, y


# ---------------------------------------------------------------------------
# KG-Capable Model Classes
# ---------------------------------------------------------------------------

# Set of models that support KG training
KG_MODEL_NAMES = {'joformer_learned_kg', 'joformer_fixed_kg', 'joformer_projected_kg'}


class JoFormerLearnedKG(nn.Module):
    """JoFormer-Learned with KG angle-gap mechanism (analog of Model E/E').

    Per-token learned angles with cumsum. For KG, a learned relation angle
    is inserted into the cumsum between head and tail tokens.
    Rotates Q, K, V (primed variant style).
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, n_relations=1):
        super().__init__()
        self.block_size = block_size
        self.n_embed = n_embed
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding_table = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList(
            [JoFormerLearnedBlock(n_embed, block_size, dropout, use_softmax)
             for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # KG: learned angle vector per relation
        self.relation_angles = nn.Parameter(torch.randn(n_relations, n_embed // 2) * 0.1)
        self.rel_to_idx = {}  # set after construction

    def _cumsum_angles_text(self, idx):
        """Right-cumsum angles for text."""
        raw_angles = self.angle_embedding_table(idx)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        return angles

    def _cumsum_angles_kg(self, token_ids, head_lens, rel_names, negate_angles):
        """Cumsum angles for KG with relation angle gap between head and tail.

        Builds extended sequence: [head_angles, rel_angle, tail_angles],
        does right-cumsum, then extracts token positions (skipping relation).
        """
        B, T = token_ids.shape
        device = token_ids.device
        raw_angles = self.angle_embedding_table(token_ids)  # (B, T, C//2)
        D = raw_angles.shape[-1]

        ext_angles = torch.zeros(B, T + 1, D, device=device)

        for i in range(B):
            h_len = head_lens[i]
            ext_angles[i, :h_len] = raw_angles[i, :h_len]

            rel_idx = self.rel_to_idx[rel_names[i]]
            rel_angle = self.relation_angles[rel_idx]
            if negate_angles[i]:
                rel_angle = -rel_angle
            ext_angles[i, h_len] = rel_angle

            t_len = T - h_len
            ext_angles[i, h_len + 1:h_len + 1 + t_len] = raw_angles[i, h_len:h_len + t_len]

        # Right-cumsum
        ext_cumsum = torch.flip(ext_angles, dims=(1,))
        ext_cumsum = torch.cumsum(ext_cumsum, dim=1)
        ext_cumsum = torch.flip(ext_cumsum, dims=(1,))

        # Extract token positions (skip relation)
        angles = torch.zeros(B, T, D, device=device)
        for i in range(B):
            h_len = head_lens[i]
            t_len = T - h_len
            angles[i, :h_len] = ext_cumsum[i, :h_len]
            angles[i, h_len:h_len + t_len] = ext_cumsum[i, h_len + 1:h_len + 1 + t_len]

        return angles

    def forward_text(self, idx, targets=None):
        """Text mode: causal next-token prediction."""
        B, T = idx.shape
        x = self.expander(self.token_embedding_table(idx))
        angles = self._cumsum_angles_text(idx)

        for block in self.blocks:
            x = block(x, angles)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, token_ids, targets, head_lens, rel_names, negate_angles):
        """KG mode: causal next-token prediction with angle-gap encoding."""
        B, T = token_ids.shape
        x = self.expander(self.token_embedding_table(token_ids))
        angles = self._cumsum_angles_kg(token_ids, head_lens, rel_names, negate_angles)

        for block in self.blocks:
            x = block(x, angles)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward(self, idx, targets=None):
        """Default forward = text mode (for compatibility with text-only eval)."""
        return self.forward_text(idx, targets)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward_text(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class JoFormerFixedKGAttention(nn.Module):
    """Fixed RoPE attention that accepts optional external angles for KG mode."""

    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.use_softmax = use_softmax
        max_len = block_size * 2  # KG sequences may exceed block_size
        self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x, angles=None):
        """
        x: (B, T, C)
        angles: (B, T, C//2) cumsum'd angles. If None, uses fixed RoPE.
        """
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        if angles is None:
            # Fixed RoPE angles
            angle1 = torch.arange(T, device=x.device)
            angle2 = torch.arange(C // 2, device=x.device)
            angle = torch.outer(angle1, angle2).unsqueeze(0)
            angle = torch.flip(angle, dims=(1,))
        else:
            angle = angles

        matrix = build_rotation_matrix(torch.cos(angle), torch.sin(angle))

        k = apply_rotation(k, matrix)
        q = apply_rotation(q, matrix)
        v = apply_rotation(v, matrix)

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        if self.use_softmax:
            wei = F.softmax(wei, dim=-1)
        else:
            wei = torch.log(torch.exp(wei) + 1)
            wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
        wei = self.dropout(wei)
        out = wei @ v

        out = apply_inverse_rotation(out, matrix)

        out = self.proj(out)
        out = self.dropout(out)
        return out


class JoFormerFixedKGBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.sa_head = JoFormerFixedKGAttention(n_embed, block_size, dropout, use_softmax)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles=None):
        x = x + self.sa_head(self.ln1(x), angles)
        x = x + self.ffn(self.ln2(x))
        return x


class JoFormerFixedKG(nn.Module):
    """JoFormer-Fixed with KG angle-gap mechanism (analog of Model H/H').

    Fixed base frequencies for text. For KG, relation angle is inserted into
    the cumsum between head and tail positions.
    Rotates Q, K, V (primed variant style).
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, n_relations=1):
        super().__init__()
        self.block_size = block_size
        self.n_embed = n_embed
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [JoFormerFixedKGBlock(n_embed, block_size, dropout, use_softmax)
             for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # Fixed base frequencies — must match the outer(pos, dim_idx) formula
        # used in JoFormerFixedAttention so text and KG share the same angle space
        self.register_buffer('base_freq', torch.arange(0, n_embed // 2).float())

        # KG: learned angle vector per relation
        self.relation_angles = nn.Parameter(torch.randn(n_relations, n_embed // 2) * 0.1)
        self.rel_to_idx = {}

    def _cumsum_angles_kg(self, T, head_lens, rel_names, device, negate_angles):
        """Cumsum angles for KG with relation angle gap."""
        B = len(head_lens)
        D = self.n_embed // 2

        ext_angles = torch.zeros(B, T + 1, D, device=device)

        for i in range(B):
            h_len = head_lens[i]
            ext_angles[i, :h_len] = self.base_freq.unsqueeze(0).expand(h_len, -1)

            rel_idx = self.rel_to_idx[rel_names[i]]
            rel_angle = self.relation_angles[rel_idx]
            if negate_angles[i]:
                rel_angle = -rel_angle
            ext_angles[i, h_len] = rel_angle

            t_len = T - h_len
            ext_angles[i, h_len + 1:h_len + 1 + t_len] = self.base_freq.unsqueeze(0).expand(t_len, -1)

        ext_cumsum = torch.flip(ext_angles, dims=(1,))
        ext_cumsum = torch.cumsum(ext_cumsum, dim=1)
        ext_cumsum = torch.flip(ext_cumsum, dims=(1,))

        angles = torch.zeros(B, T, D, device=device)
        for i in range(B):
            h_len = head_lens[i]
            t_len = T - h_len
            angles[i, :h_len] = ext_cumsum[i, :h_len]
            angles[i, h_len:h_len + t_len] = ext_cumsum[i, h_len + 1:h_len + 1 + t_len]

        return angles

    def forward_text(self, idx, targets=None):
        """Text mode: causal next-token prediction with fixed angles."""
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        # Use None angles so the attention layer computes fixed RoPE
        for block in self.blocks:
            x = block(x, angles=None)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, token_ids, targets, head_lens, rel_names, negate_angles):
        """KG mode: causal next-token prediction with angle-gap encoding."""
        B, T = token_ids.shape
        x = self.token_embedding_table(token_ids)
        angles = self._cumsum_angles_kg(T, head_lens, rel_names, token_ids.device, negate_angles)

        for block in self.blocks:
            x = block(x, angles)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward(self, idx, targets=None):
        return self.forward_text(idx, targets)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward_text(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class JoFormerProjectedKGBlock(nn.Module):
    """Projected-angle block that accepts optional external angles for KG mode."""

    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.sa_head = JoFormerLearnedAttention(n_embed, block_size, dropout, use_softmax)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.vector_proj = nn.Linear(n_embed, n_embed)
        self.angle_proj = nn.Sequential(
            nn.Linear(n_embed, 2 * n_embed),
            nn.GELU(),
            nn.Linear(2 * n_embed, n_embed // 2),
        )

    def forward(self, x, external_angles=None):
        """
        If external_angles is None: compute angles from residual stream (text mode).
        If external_angles is provided: use those instead (KG mode).
        """
        x_proj = self.vector_proj(x)

        if external_angles is not None:
            angles = external_angles
        else:
            raw_angles = self.angle_proj(x)
            angles = torch.flip(raw_angles, dims=(1,))
            angles = torch.cumsum(angles, dim=1)
            angles = torch.flip(angles, dims=(1,))

        x_proj = x_proj + self.sa_head(self.ln1(x_proj), angles)
        x_proj = x_proj + self.ffn(self.ln2(x_proj))
        return x_proj


class JoFormerProjectedKG(nn.Module):
    """JoFormer-Projected with KG angle-gap mechanism (analog of Model I/I').

    Per-layer MLP projects angles from residual stream. For KG, the projected
    angles are augmented with a relation angle gap in the cumsum.
    Rotates Q, K, V (primed variant style).
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, n_relations=1):
        super().__init__()
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layers = n_layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [JoFormerProjectedKGBlock(n_embed, block_size, dropout, use_softmax)
             for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # KG: learned angle vector per relation
        self.relation_angles = nn.Parameter(torch.randn(n_relations, n_embed // 2) * 0.1)
        self.rel_to_idx = {}

    def _cumsum_angles_kg(self, raw_angles, head_lens, rel_names, device, negate_angles):
        """Cumsum projected angles for KG with relation angle gap."""
        B, T, D = raw_angles.shape

        ext_angles = torch.zeros(B, T + 1, D, device=device)

        for i in range(B):
            h_len = head_lens[i]
            ext_angles[i, :h_len] = raw_angles[i, :h_len]

            rel_idx = self.rel_to_idx[rel_names[i]]
            rel_angle = self.relation_angles[rel_idx]
            if negate_angles[i]:
                rel_angle = -rel_angle
            ext_angles[i, h_len] = rel_angle

            t_len = T - h_len
            ext_angles[i, h_len + 1:h_len + 1 + t_len] = raw_angles[i, h_len:h_len + t_len]

        ext_cumsum = torch.flip(ext_angles, dims=(1,))
        ext_cumsum = torch.cumsum(ext_cumsum, dim=1)
        ext_cumsum = torch.flip(ext_cumsum, dims=(1,))

        angles = torch.zeros(B, T, D, device=device)
        for i in range(B):
            h_len = head_lens[i]
            t_len = T - h_len
            angles[i, :h_len] = ext_cumsum[i, :h_len]
            angles[i, h_len:h_len + t_len] = ext_cumsum[i, h_len + 1:h_len + 1 + t_len]

        return angles

    def forward_text(self, idx, targets=None):
        """Text mode: each block projects its own angles from residual stream."""
        B, T = idx.shape
        x = self.token_embedding_table(idx)

        for block in self.blocks:
            x = block(x, external_angles=None)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, token_ids, targets, head_lens, rel_names, negate_angles):
        """KG mode: project angles, insert relation gap, cumsum, attend."""
        B, T = token_ids.shape
        x = self.token_embedding_table(token_ids)

        for block in self.blocks:
            # Project raw angles from residual stream
            raw_angles = block.angle_proj(x)
            # Insert relation angle gap and cumsum
            angles = self._cumsum_angles_kg(raw_angles, head_lens, rel_names,
                                            token_ids.device, negate_angles)
            x = block(x, external_angles=angles)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward(self, idx, targets=None):
        return self.forward_text(idx, targets)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward_text(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Model registry: text-only baselines imported from train_wiki.py + KG models
KG_MODEL_CLASSES = {
    'joformer_learned_kg': JoFormerLearnedKG,
    'joformer_fixed_kg': JoFormerFixedKG,
    'joformer_projected_kg': JoFormerProjectedKG,
}

ALL_MODEL_CLASSES = {}
ALL_MODEL_CLASSES.update(MODEL_CLASSES)  # roformer, joformer_fixed, joformer_learned, joformer_projected
ALL_MODEL_CLASSES.update(KG_MODEL_CLASSES)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device,
                  eval_iters=20, kg_dataset=None, kg_batch_size=None,
                  is_kg_model=False):
    """Estimate train/val text loss + optional KG loss."""
    out = {}
    model.eval()

    # Text loss
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split,
                             block_size, batch_size, device)
            if is_kg_model:
                _, loss = model.forward_text(X, Y)
            else:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    # KG loss (only for KG models with KG data)
    if is_kg_model and kg_dataset is not None:
        kg_bs = kg_batch_size or batch_size
        kg_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            tok, tgt, hlens, rels, neg = kg_dataset.get_causal_batch(kg_bs, device)
            _, loss = model.forward_kg_causal(tok, tgt, hlens, rels, neg)
            kg_losses[k] = loss.item()
        out['kg'] = kg_losses.mean().item()

    model.train()
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model_name, model, train_data, val_data, args, device, tokenizer,
                kg_dataset=None, kg_meta=None):
    """Train a single model. Returns (val_loss, val_ppl, ppl_log)."""
    is_kg = model_name in KG_MODEL_NAMES

    # Set up relation mapping for KG models
    if is_kg and kg_meta is not None:
        model.rel_to_idx = {r: i for i, r in enumerate(kg_meta['relations'])}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters)
                 if args.cosine_decay else None)
    model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training {model_name}  ({n_params:,} parameters)")
    if is_kg and kg_dataset is not None:
        print(f"  KG: {len(kg_dataset.encoded):,} triples, {len(kg_dataset.relations)} relations, weight={args.kg_weight}")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    ppl_log = {"iter": [], "train_ppl": [], "val_ppl": []}
    if is_kg:
        ppl_log["kg_loss"] = []

    kg_bs = args.kg_batch_size or args.batch_size

    pbar = tqdm(range(args.max_iters), desc=model_name)
    for it in pbar:
        # Eval
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data,
                                   args.block_size, args.batch_size, device,
                                   kg_dataset=kg_dataset if is_kg else None,
                                   kg_batch_size=kg_bs,
                                   is_kg_model=is_kg)
            train_ppl = math.exp(losses['train'])
            val_ppl = math.exp(losses['val'])
            ppl_log["iter"].append(it)
            ppl_log["train_ppl"].append(round(train_ppl, 2))
            ppl_log["val_ppl"].append(round(val_ppl, 2))

            postfix = dict(train_loss=f"{losses['train']:.3f}",
                          val_loss=f"{losses['val']:.3f}",
                          val_ppl=f"{val_ppl:.2f}")

            msg = (f"  [{model_name}] iter {it}: "
                   f"train loss {losses['train']:.4f} (PPL {train_ppl:.2f}), "
                   f"val loss {losses['val']:.4f} (PPL {val_ppl:.2f})")

            if 'kg' in losses:
                ppl_log["kg_loss"].append(round(losses['kg'], 4))
                postfix['kg_loss'] = f"{losses['kg']:.3f}"
                msg += f", kg loss {losses['kg']:.4f}"

            pbar.set_postfix(**postfix)
            tqdm.write(msg)

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

            # Checkpoint
            if args.checkpoint_dir:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                path = os.path.join(args.checkpoint_dir, f"{model_name}_iter{it}.pt")
                torch.save({
                    'iter': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': losses['val'],
                }, path)

            # Generate sample
            if it > 0 and it % (args.eval_interval * 2) == 0:
                try:
                    model.eval()
                    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
                    generated = model.generate(prompt, args.generate_len)
                    text = tokenizer.decode(generated[0].cpu().tolist())
                    tqdm.write(f"  [{model_name}] sample: {text[:200]}")
                    model.train()
                except Exception as e:
                    tqdm.write(f"  [{model_name}] sample generation failed: {e}")
                    model.train()

        # Train step: text
        xb, yb = get_batch(train_data, val_data, "train",
                           args.block_size, args.batch_size, device)
        if is_kg:
            _, text_loss = model.forward_text(xb, yb)
        else:
            _, text_loss = model(xb, yb)

        # Train step: KG (only for KG models with KG data)
        if is_kg and kg_dataset is not None:
            tok, tgt, hlens, rels, neg = kg_dataset.get_causal_batch(kg_bs, device)
            _, kg_loss = model.forward_kg_causal(tok, tgt, hlens, rels, neg)
            loss = text_loss + args.kg_weight * kg_loss
        else:
            loss = text_loss

        # NaN detection
        if torch.isnan(loss):
            tqdm.write(f"  [{model_name}] NaN loss detected at iter {it}, stopping early.")
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

    # Final eval
    losses = estimate_loss(model, train_data, val_data,
                           args.block_size, args.batch_size, device,
                           kg_dataset=kg_dataset if is_kg else None,
                           kg_batch_size=kg_bs,
                           is_kg_model=is_kg)
    val_ppl = math.exp(losses['val'])

    msg = f"\n  [{model_name}] final val loss: {losses['val']:.4f} (PPL {val_ppl:.2f})"
    if 'kg' in losses:
        msg += f", kg loss: {losses['kg']:.4f}"
    print(msg)

    # Final generation sample
    try:
        model.eval()
        prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(prompt, args.generate_len)
        text = tokenizer.decode(generated[0].cpu().tolist())
        print(f"  [{model_name}] sample: {text[:300]}")
    except Exception as e:
        print(f"  [{model_name}] final sample generation failed: {e}")

    # Save final checkpoint
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        path = os.path.join(args.checkpoint_dir, f"{model_name}_final.pt")
        torch.save({
            'iter': args.max_iters,
            'model_state_dict': model.state_dict(),
            'val_loss': losses['val'],
        }, path)

    # Check relation angles changed (for KG models)
    if is_kg:
        rel_angles = model.relation_angles.data
        print(f"  [{model_name}] relation_angles norm: {rel_angles.norm():.4f}, "
              f"mean abs: {rel_angles.abs().mean():.4f}")

    return losses['val'], val_ppl, ppl_log, losses.get('kg', None)


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def run_training(args):
    """Load memmap data and train all requested models."""
    if args.smoke:
        args.max_iters = 50
        args.eval_interval = 25
        args.n_layers = 2
        args.n_embed = 64
        args.generate_len = 50

    if args.n_embed % 2 != 0:
        args.n_embed += 1
        print(f"Adjusted n_embed to {args.n_embed} (must be even)")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    attn_type = "softmax" if args.softmax else "normalized softplus"
    print(f"Config: n_embed={args.n_embed}, n_layers={args.n_layers}, "
          f"block_size={args.block_size}, batch_size={args.batch_size}, "
          f"lr={args.lr}, max_iters={args.max_iters}, attn={attn_type}, "
          f"kg_weight={args.kg_weight}")

    # Load memmap data + KG
    print(f"\nLoading preprocessed data from {args.data_dir}...")
    train_data, val_data, tokenizer, meta, kg_dataset, kg_meta = load_memmap_data(args.data_dir)
    actual_vocab_size = meta['vocab_size']
    print(f"Total tokens: {meta['total_tokens']:,}")
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
    print(f"Vocab size: {actual_vocab_size}")
    if kg_meta:
        print(f"KG: {kg_meta['n_triples']:,} triples, {kg_meta['n_relations']} relations")

    if len(val_data) < args.block_size + 1:
        print("WARNING: val data too small for block_size, reducing block_size")
        args.block_size = len(val_data) - 2

    # Train each model
    results = {}
    for model_name in args.models:
        torch.manual_seed(args.seed)
        is_kg = model_name in KG_MODEL_NAMES

        if is_kg:
            n_relations = kg_meta['n_relations'] if kg_meta else 1
            cls = KG_MODEL_CLASSES[model_name]
            model = cls(actual_vocab_size, args.n_embed, args.n_layers,
                        args.block_size, args.dropout, use_softmax=args.softmax,
                        n_relations=n_relations)
        else:
            cls = MODEL_CLASSES[model_name]
            model = cls(actual_vocab_size, args.n_embed, args.n_layers,
                        args.block_size, args.dropout, use_softmax=args.softmax)

        val_loss, val_ppl, ppl_log, kg_loss = train_model(
            model_name, model, train_data, val_data, args, device, tokenizer,
            kg_dataset=kg_dataset, kg_meta=kg_meta
        )
        result = {'val_loss': val_loss, 'val_ppl': val_ppl, 'ppl_curve': ppl_log}
        if kg_loss is not None:
            result['final_kg_loss'] = kg_loss
        results[model_name] = result

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    has_kg = any('final_kg_loss' in results[n] for n in args.models)
    if has_kg:
        print(f"{'Model':<25} {'Val Loss':>10} {'Val PPL':>10} {'KG Loss':>10}")
        print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    else:
        print(f"{'Model':<25} {'Val Loss':>10} {'Val PPL':>10}")
        print(f"{'-'*25} {'-'*10} {'-'*10}")
    for name in args.models:
        r = results[name]
        line = f"{name:<25} {r['val_loss']:>10.4f} {r['val_ppl']:>10.2f}"
        if has_kg:
            kg = r.get('final_kg_loss')
            line += f" {kg:>10.4f}" if kg is not None else f" {'N/A':>10}"
        print(line)
    print(f"{'='*60}")

    # Find best by text PPL
    best_name = min(results, key=lambda k: results[k]['val_loss'])
    print(f"\nBest model (text PPL): {best_name} (val PPL {results[best_name]['val_ppl']:.2f})")

    # Save results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        "config": {
            "n_embed": args.n_embed, "n_layers": args.n_layers,
            "block_size": args.block_size, "batch_size": args.batch_size,
            "lr": args.lr, "max_iters": args.max_iters,
            "vocab_size": actual_vocab_size, "models": args.models,
            "data_dir": args.data_dir,
            "total_tokens": meta['total_tokens'],
            "kg_weight": args.kg_weight,
            "kg_triples": kg_meta['n_triples'] if kg_meta else 0,
            "kg_relations": kg_meta['n_relations'] if kg_meta else 0,
        },
        "results": results,
        "timestamp": timestamp,
    }
    results_file = os.path.join(results_dir, f"joformer_kg_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    latest_file = os.path.join(results_dir, "joformer_kg_results_latest.json")
    with open(latest_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Latest results: {latest_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_training_args(parser):
    """Add training-specific arguments to a subparser."""
    parser.add_argument('--models', nargs='+',
                        default=['roformer', 'joformer_learned_kg'],
                        choices=list(ALL_MODEL_CLASSES.keys()),
                        help='Which models to train')
    parser.add_argument('--n_embed', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick test: 50 iters, small model')
    parser.add_argument('--generate_len', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--softmax', action='store_true',
                        help='Use softmax attention instead of normalized softplus')
    parser.add_argument('--cosine_decay', action='store_true',
                        help='Use cosine annealing LR schedule')
    # KG-specific args
    parser.add_argument('--kg_weight', type=float, default=1.0,
                        help='Weight of KG loss relative to text loss')
    parser.add_argument('--kg_batch_size', type=int, default=None,
                        help='KG batch size (default: same as batch_size)')


def add_data_args(parser):
    """Add data path arguments."""
    parser.add_argument('--wiki_path', type=str, default=None,
                        help='Path to wiki.en.txt')
    parser.add_argument('--wiki_lines', type=int, default=None,
                        help='Max lines to process (default: all)')
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--data_dir', type=str, default='joformer/data_kg',
                        help='Directory for preprocessed data')
    # KG data paths
    parser.add_argument('--wordnet_path', type=str, default=None)
    parser.add_argument('--framenet_path', type=str, default=None)
    parser.add_argument('--bats_dir', type=str, default=None)
    parser.add_argument('--google_path', type=str, default=None)
    parser.add_argument('--analogies_path', type=str, default=None)


def main():
    parser = argparse.ArgumentParser(
        description="Joint KG + wiki text training with memmap data"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # --- preprocess subcommand ---
    pp = subparsers.add_parser('preprocess', help='Tokenize wiki text + encode KG to binary')
    add_data_args(pp)

    # --- train subcommand ---
    tr = subparsers.add_parser('train', help='Train models from preprocessed data')
    tr.add_argument('--data_dir', type=str, default='joformer/data_kg')
    add_training_args(tr)

    # --- auto subcommand ---
    au = subparsers.add_parser('auto', help='Preprocess if needed, then train')
    add_data_args(au)
    add_training_args(au)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Default wiki path relative to this script
    if hasattr(args, 'wiki_path') and args.wiki_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.wiki_path = os.path.join(script_dir, '..', 'exp8', 'data', 'wiki.en.txt')

    # Smoke overrides for preprocessing
    if hasattr(args, 'smoke') and args.smoke:
        if args.wiki_lines is None:
            args.wiki_lines = 1000
        args.vocab_size = 2000

    if args.command == 'preprocess':
        preprocess(args)
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'auto':
        bin_path = os.path.join(args.data_dir, 'wiki_tokens.bin')
        if not os.path.exists(bin_path):
            print("Preprocessed data not found, running preprocessing...")
            preprocess(args)
        else:
            print(f"Using existing preprocessed data in {args.data_dir}")
        run_training(args)


if __name__ == '__main__':
    main()
