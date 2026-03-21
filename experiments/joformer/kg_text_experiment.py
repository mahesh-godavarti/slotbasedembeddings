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
# kg_text_experiment.py — Joint KG + Text: Angle-Gap vs Text-Linearized KG
#
# Compares 6 models (3 architectures x 2 KG methods) on wiki text with
# knowledge graph augmentation. Evaluates on a held-out test set (80/10/10).
#
# Models:
#   1. roformer_kg           — RoFormer + angle-gap KG (Q,K only)
#   2. roformer_text_kg      — RoFormer + text-linearized KG
#   3. joformer_fixed_kg     — JoFormer-Fixed + angle-gap KG (Q,K,V)
#   4. joformer_fixed_text_kg — JoFormer-Fixed + text-linearized KG
#   5. joformer_projected_kg — JoFormer-Projected + angle-gap KG (Q,K,V)
#   6. joformer_projected_text_kg — JoFormer-Projected + text-linearized KG
#
# Usage:
#   python kg_text_experiment.py --smoke
#   nohup python kg_text_experiment.py > kg_text_exp.log 2>&1 &

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
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
    RoFormer, JoFormerFixed, JoFormerProjected,
)
from train_wiki_kg_streaming import (
    JoFormerFixedKG, JoFormerProjectedKG, JoFormerProjectedMergedKG,
    JoFormerFixedKGAttention, JoFormerFixedKGBlock,
    KGDataset,
    load_all_kg_triples, encode_kg_triples,
)


# ---------------------------------------------------------------------------
# RoFormerKG — RoFormer with angle-gap KG (rotates Q,K only, NOT V)
# ---------------------------------------------------------------------------

class RoFormerKGAttention(nn.Module):
    """Fixed RoPE attention that rotates Q,K only (not V), with optional
    external angles for KG mode."""

    # Uses block_size * 2 for tril to handle KG sequences that may exceed block_size
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.use_softmax = use_softmax
        max_len = block_size * 2
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
        # V is NOT rotated (standard RoPE)

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        if self.use_softmax:
            wei = F.softmax(wei, dim=-1)
        else:
            wei = torch.log(torch.exp(wei) + 1)
            wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
        wei = self.dropout(wei)
        out = wei @ v
        # No inverse rotation (V was not rotated)

        out = self.proj(out)
        out = self.dropout(out)
        return out


class RoFormerKGBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout, use_softmax=False):
        super().__init__()
        self.sa_head = RoFormerKGAttention(n_embed, block_size, dropout, use_softmax)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles=None):
        x = x + self.sa_head(self.ln1(x), angles)
        x = x + self.ffn(self.ln2(x))
        return x


class RoFormerKG(nn.Module):
    """RoFormer with KG angle-gap mechanism.

    Standard RoPE (Q,K only) for text. For KG, relation angle is inserted
    into the cumsum between head and tail positions.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout,
                 use_softmax=False, n_relations=1):
        super().__init__()
        self.block_size = block_size
        self.n_embed = n_embed
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [RoFormerKGBlock(n_embed, block_size, dropout, use_softmax)
             for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # Fixed base frequencies (same as JoFormerFixedKG)
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
        """Text mode: causal next-token prediction with fixed RoPE."""
        B, T = idx.shape
        x = self.token_embedding_table(idx)
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


# ---------------------------------------------------------------------------
# Text-linearized KG preprocessing
# ---------------------------------------------------------------------------

def linearize_kg_to_text(triples):
    """Convert KG triples to linearized text sentences.

    Each triple becomes: "head relation tail ."
    Relations use plain text (no special tokens) since we use BPE.

    Returns list of sentence strings.
    """
    sentences = []
    for head, rel, tail in triples:
        # Convert relation name: underscore to space for BPE
        rel_text = rel.replace('_', ' ')
        sentences.append(f"{head} {rel_text} {tail} .")
    return sentences


def linearize_kg_to_disk(triples, tokenizer, output_bin_path):
    """Linearize KG triples as text, tokenize, and write to binary file.

    Returns (total_tokens, n_sentences).
    """
    sentences = linearize_kg_to_text(triples)
    random.shuffle(sentences)

    total_tokens = 0
    with open(output_bin_path, 'wb') as dst:
        for sent in sentences:
            enc = tokenizer.encode(sent)
            ids = enc.ids
            if ids:
                chunk = np.array(ids, dtype=np.int32)
                dst.write(chunk.tobytes())
                total_tokens += len(ids)

    return total_tokens, len(sentences)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _decode_entity(tokenizer, token_ids):
    """Decode token IDs back to text, extract lowercase words.

    The character-level tokenizer inserts spaces between characters,
    so we remove spaces first to reconstruct the original word(s).
    """
    text = tokenizer.decode(token_ids).replace(' ', '')
    return set(re.findall(r'[a-zA-Z]+', text.lower()))


def _load_wiki_word_freq(data_dir):
    """Load cached wiki word frequencies, or compute from wiki text file."""
    cache_path = os.path.join(data_dir, 'wiki_word_freq.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Compute from wiki text file
    from collections import Counter
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wiki_path = os.path.join(script_dir, '..', 'exp8', 'data', 'wiki.en.txt')
    print(f"  Computing wiki word frequencies from {wiki_path}...")
    word_counts = Counter()
    with open(wiki_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                words = re.findall(r'[a-zA-Z]+', line.lower())
                word_counts.update(words)
    with open(cache_path, 'wb') as f:
        pickle.dump(word_counts, f)
    print(f"  Cached {len(word_counts):,} unique words to {cache_path}")
    return word_counts


def _filter_kg_triples_by_wiki_freq(kg_test_triples, tokenizer, wiki_word_freq,
                                     min_freq=100):
    """Filter KG test triples for those where entity words are common in wiki.

    Returns triples where ALL words in both head and tail entities have
    wiki frequency >= min_freq. These are the triples that text training
    should help with — the model has seen these words often in wiki text.
    """
    filtered = []
    for triple in kg_test_triples:
        head_words = _decode_entity(tokenizer, triple['head_ids'])
        tail_words = _decode_entity(tokenizer, triple['tail_ids'])
        all_words = head_words | tail_words
        if not all_words:
            continue
        # All entity words must be common in wiki
        if all(wiki_word_freq.get(w, 0) >= min_freq for w in all_words):
            filtered.append(triple)
    return filtered


def load_data(data_dir, split_ratios=(0.8, 0.1, 0.1), kg_test_ratio=0.1):
    """Load preprocessed wiki data with 80/10/10 train/val/test split.

    Also loads KG data (angle-gap encoded triples + text-linearized tokens),
    split into train/test for KG evaluation.

    Cross-pollination eval:
    - KG→Text: wiki sentences with rare KG words (from cross_text_eval.bin)
    - Text→KG: held-out KG test triples with common wiki words (filtered)
    """
    from tokenizers import Tokenizer

    meta_path = os.path.join(data_dir, 'wiki_tokens.meta')
    bin_path = os.path.join(data_dir, 'wiki_tokens.bin')
    tok_path = os.path.join(data_dir, 'wiki_tokenizer.json')
    kg_path = os.path.join(data_dir, 'kg_triples.pkl')
    kg_meta_path = os.path.join(data_dir, 'kg_meta.json')
    kg_text_bin_path = os.path.join(data_dir, 'kg_text_tokens.bin')
    kg_text_meta_path = os.path.join(data_dir, 'kg_text_tokens.meta')

    with open(meta_path) as f:
        meta = json.load(f)

    total_tokens = meta['total_tokens']
    data = np.memmap(bin_path, dtype=np.int32, mode='r', shape=(total_tokens,))

    # 80/10/10 split
    n_train = int(total_tokens * split_ratios[0])
    n_val = int(total_tokens * (split_ratios[0] + split_ratios[1]))
    train_data = data[:n_train]
    val_data = data[n_train:n_val]
    test_data = data[n_val:]

    tokenizer = Tokenizer.from_file(tok_path)

    # Load KG angle-gap data, split into train/test
    kg_train_dataset = None
    kg_test_dataset = None
    kg_meta = None
    kg_test_triples = None  # raw list, kept for cross-eval filtering
    if os.path.exists(kg_path) and os.path.exists(kg_meta_path):
        with open(kg_path, 'rb') as f:
            encoded_triples = pickle.load(f)
        with open(kg_meta_path) as f:
            kg_meta = json.load(f)
        pad_id = tokenizer.token_to_id("<PAD>") or 0

        # Deterministic shuffle then split
        rng = random.Random(42)
        shuffled = list(encoded_triples)
        rng.shuffle(shuffled)
        n_kg_test = int(len(shuffled) * kg_test_ratio)
        kg_train_triples = shuffled[n_kg_test:]
        kg_test_triples = shuffled[:n_kg_test]

        kg_train_dataset = KGDataset(kg_train_triples, kg_meta['relations'], pad_id=pad_id)
        kg_test_dataset = KGDataset(kg_test_triples, kg_meta['relations'], pad_id=pad_id)

    # Load KG text-linearized data, split into train/test
    kg_text_train = None
    kg_text_test = None
    if os.path.exists(kg_text_bin_path) and os.path.exists(kg_text_meta_path):
        with open(kg_text_meta_path) as f:
            kg_text_meta = json.load(f)
        kg_text_tokens = kg_text_meta['total_tokens']
        kg_text_data = np.memmap(kg_text_bin_path, dtype=np.int32, mode='r',
                                 shape=(kg_text_tokens,))
        n_kg_text_test = int(kg_text_tokens * kg_test_ratio)
        kg_text_train = kg_text_data[:kg_text_tokens - n_kg_text_test]
        kg_text_test = kg_text_data[kg_text_tokens - n_kg_text_test:]

    # Load cross-pollination eval sets
    cross_text_data = None
    cross_kg_dataset = None

    # KG→Text: wiki sentences with rare KG words
    cross_text_bin = os.path.join(data_dir, 'cross_text_eval.bin')
    cross_text_meta_path = os.path.join(data_dir, 'cross_text_eval.meta')
    if os.path.exists(cross_text_bin) and os.path.exists(cross_text_meta_path):
        with open(cross_text_meta_path) as f:
            ct_meta = json.load(f)
        cross_text_data = np.memmap(cross_text_bin, dtype=np.int32, mode='r',
                                     shape=(ct_meta['total_tokens'],))

    # Load wiki word frequencies (needed for both cross-eval sets)
    rare_kg_words = None
    id_to_char = None
    wiki_word_freq = None
    if kg_test_triples is not None and kg_meta is not None:
        wiki_word_freq = _load_wiki_word_freq(data_dir)

        # KG→Text: build rare KG word set + char mapping for masked eval
        all_encoded = (kg_train_triples if kg_train_triples else []) + (kg_test_triples or [])
        rare_kg_words, id_to_char = _build_rare_kg_words(
            all_encoded, tokenizer, wiki_word_freq, max_freq=50)
        print(f"  KG→Text cross-eval: {len(rare_kg_words):,} rare KG words "
              f"(wiki freq <= 50), {len(id_to_char)} alpha token IDs")

        # Text→KG: filter KG test triples for common wiki words
        common_triples = _filter_kg_triples_by_wiki_freq(
            kg_test_triples, tokenizer, wiki_word_freq, min_freq=100)
        if common_triples:
            pad_id = tokenizer.token_to_id("<PAD>") or 0
            cross_kg_dataset = KGDataset(
                common_triples, kg_meta['relations'], pad_id=pad_id)
            print(f"  Text→KG cross-eval: {len(common_triples):,} / "
                  f"{len(kg_test_triples):,} test triples with common wiki words")

    return (train_data, val_data, test_data, tokenizer, meta,
            kg_train_dataset, kg_test_dataset, kg_meta,
            kg_text_train, kg_text_test,
            cross_text_data, cross_kg_dataset,
            rare_kg_words, id_to_char)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def get_batch(data, block_size, batch_size, device):
    """Random-access batch from memory-mapped data."""
    n = len(data) - block_size
    ix = torch.randint(0, n, (batch_size,)).numpy()
    sequences = np.stack([data[i:i + block_size + 1] for i in ix])
    sequences = torch.from_numpy(sequences.astype(np.int64)).to(device)
    x = sequences[:, :block_size].contiguous()
    y = sequences[:, 1:block_size + 1].contiguous()
    return x, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device,
                  eval_iters=20, kg_dataset=None, kg_batch_size=None,
                  is_kg_model=False, kg_text_data=None, is_text_kg_model=False):
    """Estimate train/val text loss + optional KG loss."""
    out = {}
    model.eval()

    # Text loss
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            if is_kg_model:
                _, loss = model.forward_text(X, Y)
            else:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    # KG angle-gap loss
    if is_kg_model and kg_dataset is not None:
        kg_bs = kg_batch_size or batch_size
        kg_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            tok, tgt, hlens, rels, neg = kg_dataset.get_causal_batch(kg_bs, device)
            _, loss = model.forward_kg_causal(tok, tgt, hlens, rels, neg)
            kg_losses[k] = loss.item()
        out['kg'] = kg_losses.mean().item()

    # KG text-linearized loss
    if is_text_kg_model and kg_text_data is not None:
        kg_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(kg_text_data, block_size, batch_size, device)
            _, loss = model(X, Y)
            kg_losses[k] = loss.item()
        out['kg'] = kg_losses.mean().item()

    model.train()
    return out


@torch.no_grad()
def evaluate_test(model, test_data, block_size, batch_size, device,
                  eval_iters=50, is_kg_model=False,
                  kg_test_dataset=None, kg_text_test=None, is_text_kg_model=False):
    """Evaluate on held-out test set. Returns dict with 'text' and optionally 'kg' loss."""
    model.eval()
    out = {}

    # Text test loss
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(test_data, block_size, batch_size, device)
        if is_kg_model:
            _, loss = model.forward_text(X, Y)
        else:
            _, loss = model(X, Y)
        losses[k] = loss.item()
    out['text'] = losses.mean().item()

    # KG angle-gap test loss
    if is_kg_model and kg_test_dataset is not None:
        kg_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            tok, tgt, hlens, rels, neg = kg_test_dataset.get_causal_batch(batch_size, device)
            _, loss = model.forward_kg_causal(tok, tgt, hlens, rels, neg)
            kg_losses[k] = loss.item()
        out['kg'] = kg_losses.mean().item()

    # KG text-linearized test loss
    if is_text_kg_model and kg_text_test is not None:
        kg_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(kg_text_test, block_size, batch_size, device)
            _, loss = model(X, Y)
            kg_losses[k] = loss.item()
        out['kg'] = kg_losses.mean().item()

    model.train()
    return out


def _build_rare_kg_words(kg_encoded_triples, tokenizer, wiki_word_freq, max_freq=50):
    """Build set of rare KG words: words in KG entities with wiki freq <= max_freq.

    Since the tokenizer is character-level, we also build a mapping from
    each character token ID to its character, for efficient word reconstruction.
    """
    # Extract all words from KG entities
    # Character-level tokenizer inserts spaces between chars, remove them first
    kg_words = set()
    for triple in kg_encoded_triples:
        head_text = tokenizer.decode(triple['head_ids']).replace(' ', '')
        tail_text = tokenizer.decode(triple['tail_ids']).replace(' ', '')
        kg_words.update(re.findall(r'[a-zA-Z]+', head_text.lower()))
        kg_words.update(re.findall(r'[a-zA-Z]+', tail_text.lower()))

    # Filter to rare words (wiki freq <= max_freq, length >= 3)
    rare = set()
    for w in kg_words:
        if len(w) >= 3 and wiki_word_freq.get(w, 0) <= max_freq:
            rare.add(w)

    # Build token_id -> character mapping for ASCII a-z only
    vocab = tokenizer.get_vocab()
    id_to_char = {}
    for ch, tid in vocab.items():
        if len(ch) == 1 and ch.isascii() and ch.isalpha():
            id_to_char[tid] = ch.lower()

    return rare, id_to_char


def _build_word_mask(targets, id_to_char, rare_words):
    """Build a boolean mask over target positions that are part of rare KG words.

    Reconstructs words from runs of alpha character tokens, checks each word
    against the rare set, and marks all character positions of matching words.

    Args:
        targets: (B, T) tensor of target token IDs
        id_to_char: dict mapping token ID -> lowercase character
        rare_words: set of rare KG word strings

    Returns:
        mask: (B, T) boolean tensor, True at positions belonging to rare KG words
    """
    B, T = targets.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=targets.device)
    targets_cpu = targets.cpu().numpy()

    for b in range(B):
        # Scan through sequence, collecting runs of alpha characters
        i = 0
        while i < T:
            tid = int(targets_cpu[b, i])
            if tid in id_to_char:
                # Start of a word — collect all alpha chars
                word_start = i
                chars = []
                while i < T and int(targets_cpu[b, i]) in id_to_char:
                    chars.append(id_to_char[int(targets_cpu[b, i])])
                    i += 1
                word = ''.join(chars)
                if word in rare_words:
                    mask[b, word_start:i] = True
            else:
                i += 1

    return mask


@torch.no_grad()
def evaluate_cross(model, block_size, batch_size, device, eval_iters=50,
                   is_kg_model=False, cross_text_data=None,
                   cross_kg_dataset=None,
                   rare_kg_words=None, id_to_char=None):
    """Evaluate cross-pollination.

    KG→Text: PPL on rare KG word tokens only (not full sentences).
             Computes per-token loss, masks to only rare-KG-word positions.
             If KG training helps, the model should predict these tokens better.
    Text→KG: PPL on held-out KG triples with common wiki words.
             If text training helps, the model should predict these better.
    """
    model.eval()
    out = {}

    # KG→Text: masked PPL on rare KG word tokens only
    if (cross_text_data is not None and len(cross_text_data) > block_size
            and rare_kg_words is not None and id_to_char is not None):
        total_loss = 0.0
        total_tokens = 0
        for k in range(eval_iters):
            X, Y = get_batch(cross_text_data, block_size, batch_size, device)
            # Get per-token logits
            if is_kg_model:
                logits, _ = model.forward_text(X, Y)
            else:
                logits, _ = model(X, Y)
            # Compute per-token loss (no reduction)
            # logits: (B, T, V), Y: (B, T)
            B, T, V = logits.shape
            per_token_loss = F.cross_entropy(
                logits.view(B * T, V), Y.view(B * T), reduction='none'
            ).view(B, T)
            # Build mask for rare KG word positions
            word_mask = _build_word_mask(Y, id_to_char, rare_kg_words)
            n_masked = word_mask.sum().item()
            if n_masked > 0:
                masked_loss = (per_token_loss * word_mask.float()).sum().item()
                total_loss += masked_loss
                total_tokens += n_masked
        if total_tokens > 0:
            out['cross_text'] = total_loss / total_tokens
            out['cross_text_n_tokens'] = total_tokens

    # Text→KG: PPL on held-out KG triples with common wiki words
    if is_kg_model and cross_kg_dataset is not None:
        kg_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            tok, tgt, hlens, rels, neg = cross_kg_dataset.get_causal_batch(
                batch_size, device)
            _, loss = model.forward_kg_causal(tok, tgt, hlens, rels, neg)
            kg_losses[k] = loss.item()
        out['cross_kg'] = kg_losses.mean().item()

    model.train()
    return out


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Angle-gap KG models
ANGLE_GAP_MODELS = {
    'roformer_kg': RoFormerKG,
    'joformer_fixed_kg': JoFormerFixedKG,
    'joformer_projected_kg': JoFormerProjectedKG,
    'joformer_projected_merged_kg': JoFormerProjectedMergedKG,
}

# Text-linearized KG models (reuse text-only model classes)
TEXT_KG_MODELS = {
    'roformer_text_kg': RoFormer,
    'joformer_fixed_text_kg': JoFormerFixed,
    'joformer_projected_text_kg': JoFormerProjected,
}

# Text-only baselines (no KG at all)
BASELINE_MODELS = {
    'roformer': RoFormer,
    'joformer_fixed': JoFormerFixed,
    'joformer_projected': JoFormerProjected,
}

ALL_MODELS = (list(BASELINE_MODELS.keys()) +
              list(ANGLE_GAP_MODELS.keys()) +
              list(TEXT_KG_MODELS.keys()))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model_name, model, train_data, val_data, args, device, tokenizer,
                kg_train_dataset=None, kg_meta=None, kg_text_train=None):
    """Train a single model. Returns (val_loss, val_ppl, ppl_log, kg_loss)."""
    is_kg = model_name in ANGLE_GAP_MODELS
    is_text_kg = model_name in TEXT_KG_MODELS

    # Set up relation mapping for angle-gap KG models
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
    if is_kg and kg_train_dataset is not None:
        print(f"  KG (angle-gap): {len(kg_train_dataset.encoded):,} train triples, "
              f"{len(kg_train_dataset.relations)} relations, weight={args.kg_weight}")
    if is_text_kg and kg_text_train is not None:
        print(f"  KG (text-linearized): {len(kg_text_train):,} train tokens, weight={args.kg_weight}")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    ppl_log = {"iter": [], "train_ppl": [], "val_ppl": []}
    if is_kg or is_text_kg:
        ppl_log["kg_loss"] = []

    kg_bs = args.kg_batch_size or args.batch_size

    pbar = tqdm(range(args.max_iters), desc=model_name)
    for it in pbar:
        # Eval
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data,
                                   args.block_size, args.batch_size, device,
                                   kg_dataset=kg_train_dataset if is_kg else None,
                                   kg_batch_size=kg_bs,
                                   is_kg_model=is_kg,
                                   kg_text_data=kg_text_train if is_text_kg else None,
                                   is_text_kg_model=is_text_kg)
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
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        if is_kg:
            _, text_loss = model.forward_text(xb, yb)
        else:
            _, text_loss = model(xb, yb)

        # Train step: KG (angle-gap)
        if is_kg and kg_train_dataset is not None:
            tok, tgt, hlens, rels, neg = kg_train_dataset.get_causal_batch(kg_bs, device)
            _, kg_loss = model.forward_kg_causal(tok, tgt, hlens, rels, neg)
            loss = text_loss + args.kg_weight * kg_loss
        # Train step: KG (text-linearized)
        elif is_text_kg and kg_text_train is not None:
            kg_xb, kg_yb = get_batch(kg_text_train, args.block_size, args.batch_size, device)
            _, kg_loss = model(kg_xb, kg_yb)
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
                           kg_dataset=kg_train_dataset if is_kg else None,
                           kg_batch_size=kg_bs,
                           is_kg_model=is_kg,
                           kg_text_data=kg_text_train if is_text_kg else None,
                           is_text_kg_model=is_text_kg)
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

    # Check relation angles changed (for angle-gap KG models)
    if is_kg:
        rel_angles = model.relation_angles.data
        print(f"  [{model_name}] relation_angles norm: {rel_angles.norm():.4f}, "
              f"mean abs: {rel_angles.abs().mean():.4f}")

    return losses['val'], val_ppl, ppl_log, losses.get('kg', None)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_kg(args, data_dir, tokenizer):
    """Preprocess KG data: encode triples for angle-gap + linearize for text-KG."""
    kg_path = os.path.join(data_dir, 'kg_triples.pkl')
    kg_meta_path = os.path.join(data_dir, 'kg_meta.json')
    kg_text_bin_path = os.path.join(data_dir, 'kg_text_tokens.bin')
    kg_text_meta_path = os.path.join(data_dir, 'kg_text_tokens.meta')

    # Load raw triples
    print("\nLoading KG triples...")
    kg_triples = load_all_kg_triples(args)

    if not kg_triples:
        print("  No KG triples found, skipping KG preprocessing.")
        return

    # 1. Encode for angle-gap models
    if not os.path.exists(kg_path):
        print("Encoding KG triples for angle-gap models...")
        t0 = time.time()
        encoded_triples, relations = encode_kg_triples(tokenizer, kg_triples)
        print(f"  Encoded {len(encoded_triples):,} triples, "
              f"{len(relations)} relations in {time.time()-t0:.1f}s")

        with open(kg_path, 'wb') as f:
            pickle.dump(encoded_triples, f)

        kg_meta = {
            'n_triples': len(encoded_triples),
            'n_relations': len(relations),
            'relations': relations,
        }
        with open(kg_meta_path, 'w') as f:
            json.dump(kg_meta, f, indent=2)
        print(f"  Saved to {kg_path}")
    else:
        print(f"  Angle-gap KG data already exists at {kg_path}")

    # 2. Linearize for text-KG models
    if not os.path.exists(kg_text_bin_path):
        print("Linearizing KG triples as text...")
        t0 = time.time()
        total_tokens, n_sentences = linearize_kg_to_disk(
            kg_triples, tokenizer, kg_text_bin_path)
        dt = time.time() - t0
        print(f"  {n_sentences:,} sentences, {total_tokens:,} tokens in {dt:.1f}s")

        kg_text_meta = {
            'total_tokens': total_tokens,
            'n_sentences': n_sentences,
            'n_source_triples': len(kg_triples),
            'dtype': 'int32',
        }
        with open(kg_text_meta_path, 'w') as f:
            json.dump(kg_text_meta, f, indent=2)
        print(f"  Saved to {kg_text_bin_path}")
    else:
        print(f"  Text-linearized KG data already exists at {kg_text_bin_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KG + Text experiment: Angle-Gap vs Text-Linearized KG")

    parser.add_argument('--data_dir', type=str, default='joformer/data_v8k',
                        help='Preprocessed data directory')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help='Which models to train')
    parser.add_argument('--n_embed', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_iters', type=int, default=200000)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='joformer/checkpoints_kg_text')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick test: 50 iters, small model')
    parser.add_argument('--generate_len', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--softmax', action='store_true', default=True,
                        help='Use softmax attention (default: True)')
    parser.add_argument('--cosine_decay', action='store_true')
    parser.add_argument('--kg_weight', type=float, default=1.0,
                        help='Weight for KG loss')
    parser.add_argument('--kg_batch_size', type=int, default=None,
                        help='Batch size for KG training (default: same as batch_size)')
    parser.add_argument('--eval_iters', type=int, default=20,
                        help='Number of eval iterations')
    parser.add_argument('--test_eval_iters', type=int, default=50,
                        help='Number of test eval iterations')

    # KG data paths (passed through to load_all_kg_triples)
    parser.add_argument('--wordnet_path', type=str, default=None)
    parser.add_argument('--framenet_path', type=str, default=None)
    parser.add_argument('--bats_dir', type=str, default=None)
    parser.add_argument('--google_path', type=str, default=None)
    parser.add_argument('--analogies_path', type=str, default=None)

    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke:
        args.max_iters = 50
        args.eval_interval = 25
        args.n_layers = 2
        args.n_embed = 64
        args.generate_len = 50
        args.eval_iters = 5
        args.test_eval_iters = 10

    # Ensure n_embed is even
    if args.n_embed % 2 != 0:
        args.n_embed += 1
        print(f"Adjusted n_embed to {args.n_embed} (must be even)")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: n_embed={args.n_embed}, n_layers={args.n_layers}, "
          f"block_size={args.block_size}, batch_size={args.batch_size}, "
          f"lr={args.lr}, max_iters={args.max_iters}, "
          f"kg_weight={args.kg_weight}, softmax={args.softmax}")
    print(f"Models: {args.models}")

    # Resolve data_dir relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(script_dir, os.path.basename(args.data_dir))

    # Check preprocessed data exists
    meta_path = os.path.join(args.data_dir, 'wiki_tokens.meta')
    if not os.path.exists(meta_path):
        print(f"ERROR: Preprocessed data not found at {args.data_dir}")
        print(f"  Expected: {meta_path}")
        sys.exit(1)

    # Load tokenizer for KG preprocessing
    from tokenizers import Tokenizer
    tok_path = os.path.join(args.data_dir, 'wiki_tokenizer.json')
    tokenizer = Tokenizer.from_file(tok_path)

    # Preprocess KG data if needed
    preprocess_kg(args, args.data_dir, tokenizer)

    # Load all data
    print(f"\nLoading data from {args.data_dir}...")
    (train_data, val_data, test_data, tokenizer, meta,
     kg_train_dataset, kg_test_dataset, kg_meta,
     kg_text_train, kg_text_test,
     cross_text_data, cross_kg_dataset,
     rare_kg_words, id_to_char) = load_data(args.data_dir)
    actual_vocab_size = meta['vocab_size']

    print(f"Wiki tokens: {meta['total_tokens']:,}")
    print(f"  Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}")
    print(f"Vocab size: {actual_vocab_size}")
    if kg_meta:
        kg_train_n = len(kg_train_dataset.encoded) if kg_train_dataset else 0
        kg_test_n = len(kg_test_dataset.encoded) if kg_test_dataset else 0
        print(f"KG (angle-gap): {kg_meta['n_triples']:,} total triples "
              f"({kg_train_n:,} train, {kg_test_n:,} test), "
              f"{kg_meta['n_relations']} relations")
    if kg_text_train is not None:
        print(f"KG (text-linearized): {len(kg_text_train):,} train tokens, "
              f"{len(kg_text_test):,} test tokens")
    if cross_text_data is not None:
        print(f"Cross-eval text (rare KG words): {len(cross_text_data):,} tokens")
    if cross_kg_dataset is not None:
        print(f"Cross-eval KG (common-wiki-word triples): {len(cross_kg_dataset.encoded):,} triples")

    if len(val_data) < args.block_size + 1:
        print("WARNING: val data too small, reducing block_size")
        args.block_size = len(val_data) - 2
    if len(test_data) < args.block_size + 1:
        print("WARNING: test data too small, reducing block_size")
        args.block_size = min(args.block_size, len(test_data) - 2)

    n_relations = kg_meta['n_relations'] if kg_meta else 1

    # Train each model
    results = {}
    trained_models = {}
    for model_name in args.models:
        torch.manual_seed(args.seed)

        if model_name in ANGLE_GAP_MODELS:
            cls = ANGLE_GAP_MODELS[model_name]
            model = cls(actual_vocab_size, args.n_embed, args.n_layers,
                        args.block_size, args.dropout,
                        use_softmax=args.softmax, n_relations=n_relations)
        elif model_name in TEXT_KG_MODELS:
            cls = TEXT_KG_MODELS[model_name]
            model = cls(actual_vocab_size, args.n_embed, args.n_layers,
                        args.block_size, args.dropout, use_softmax=args.softmax)
        else:
            cls = BASELINE_MODELS[model_name]
            model = cls(actual_vocab_size, args.n_embed, args.n_layers,
                        args.block_size, args.dropout, use_softmax=args.softmax)

        val_loss, val_ppl, ppl_log, kg_loss = train_model(
            model_name, model, train_data, val_data, args, device, tokenizer,
            kg_train_dataset=kg_train_dataset, kg_meta=kg_meta,
            kg_text_train=kg_text_train)

        results[model_name] = {
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'ppl_curve': ppl_log,
            'kg_loss': kg_loss,
        }
        trained_models[model_name] = model

    # Test evaluation
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")
    for model_name, model in trained_models.items():
        is_kg = model_name in ANGLE_GAP_MODELS
        is_text_kg = model_name in TEXT_KG_MODELS
        test_out = evaluate_test(model, test_data, args.block_size,
                                 args.batch_size, device,
                                 eval_iters=args.test_eval_iters,
                                 is_kg_model=is_kg,
                                 kg_test_dataset=kg_test_dataset if is_kg else None,
                                 kg_text_test=kg_text_test if is_text_kg else None,
                                 is_text_kg_model=is_text_kg)
        test_ppl = math.exp(test_out['text'])
        results[model_name]['test_loss'] = test_out['text']
        results[model_name]['test_ppl'] = test_ppl
        msg = f"  {model_name}: text test loss {test_out['text']:.4f} (PPL {test_ppl:.2f})"
        if 'kg' in test_out:
            kg_test_ppl = math.exp(test_out['kg'])
            results[model_name]['kg_test_loss'] = test_out['kg']
            results[model_name]['kg_test_ppl'] = kg_test_ppl
            msg += f", KG test loss {test_out['kg']:.4f} (PPL {kg_test_ppl:.2f})"
        print(msg)

    # Cross-pollination evaluation
    if cross_text_data is not None or cross_kg_dataset is not None:
        print(f"\n{'='*60}")
        print("CROSS-POLLINATION EVALUATION")
        print(f"{'='*60}")
        for model_name, model in trained_models.items():
            is_kg = model_name in ANGLE_GAP_MODELS
            cross_out = evaluate_cross(
                model, args.block_size, args.batch_size, device,
                eval_iters=args.test_eval_iters,
                is_kg_model=is_kg,
                cross_text_data=cross_text_data,
                cross_kg_dataset=cross_kg_dataset,
                rare_kg_words=rare_kg_words,
                id_to_char=id_to_char)
            msg = f"  {model_name}:"
            if 'cross_text' in cross_out:
                cross_text_ppl = math.exp(cross_out['cross_text'])
                results[model_name]['cross_text_ppl'] = cross_text_ppl
                msg += f" rare-KG-word text PPL {cross_text_ppl:.2f}"
            if 'cross_kg' in cross_out:
                cross_kg_ppl = math.exp(cross_out['cross_kg'])
                results[model_name]['cross_kg_ppl'] = cross_kg_ppl
                msg += f", common-word KG PPL {cross_kg_ppl:.2f}"
            print(msg)

    # Comparison table
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    header = (f"{'Model':<30} {'KG Method':<12} {'Val PPL':>8} {'Test PPL':>9} "
              f"{'Rare-KG PPL':>12} {'Common-KG PPL':>14} {'KG Test PPL':>12}")
    print(header)
    print(f"{'-'*30} {'-'*12} {'-'*8} {'-'*9} {'-'*12} {'-'*14} {'-'*12}")
    for name in args.models:
        r = results[name]
        if name in ANGLE_GAP_MODELS:
            kg_method = "angle-gap"
        elif name in TEXT_KG_MODELS:
            kg_method = "text-linear"
        else:
            kg_method = "none"
        cross_text_str = f"{r['cross_text_ppl']:.2f}" if r.get('cross_text_ppl') is not None else "N/A"
        cross_kg_str = f"{r['cross_kg_ppl']:.2f}" if r.get('cross_kg_ppl') is not None else "N/A"
        kg_test_ppl_str = f"{r['kg_test_ppl']:.2f}" if r.get('kg_test_ppl') is not None else "N/A"
        print(f"{name:<30} {kg_method:<12} {r['val_ppl']:>8.2f} {r['test_ppl']:>9.2f} "
              f"{cross_text_str:>12} {cross_kg_str:>14} {kg_test_ppl_str:>12}")
    print(f"{'='*100}")

    # Best model
    best_name = min(results, key=lambda k: results[k]['test_ppl'])
    print(f"\nBest model: {best_name} (test PPL {results[best_name]['test_ppl']:.2f})")

    # Save results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        "config": {
            "n_embed": args.n_embed, "n_layers": args.n_layers,
            "block_size": args.block_size, "batch_size": args.batch_size,
            "lr": args.lr, "max_iters": args.max_iters,
            "kg_weight": args.kg_weight, "models": args.models,
            "softmax": args.softmax, "dropout": args.dropout,
        },
        "results": {name: {k: v for k, v in r.items()} for name, r in results.items()},
        "timestamp": timestamp,
    }
    results_file = os.path.join(results_dir, f"kg_text_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    latest_file = os.path.join(results_dir, "kg_text_results_latest.json")
    with open(latest_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Latest results: {latest_file}")


if __name__ == '__main__':
    main()
