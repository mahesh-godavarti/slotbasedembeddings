# -----------------------------------------------------------------------------
# Exp 8: Word-Level KG+Text on Real Data
#
# Same architectural comparisons as Exp 7 (Models A-I with prime variants),
# but word-level tokenization on real data:
#   - Wikipedia text (wiki.en.txt)
#   - WordNet synonyms, FrameNet relations
#   - BATS 3.0 analogies, Google analogies, word analogies
#
# Seven evaluation tiers (same structure as Exp 7):
#   Tier 1: Memorization (facts seen in both KG and text)
#   Tier 2: Transfer (base in both, derived only in KG)
#   Tier 3: Generalization (base only, no derived)
#   Tier 4: KG-exclusive memorization (KG only, zero text)
#   Tier 5: KG-exclusive generalization (base in KG only, derived nowhere)
#   Tier 6: Text-exclusive memorization (text only, zero KG)
#   Tier 7: Text-exclusive generalization (base in text only, derived nowhere)
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import os
import tempfile
from datetime import datetime
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# ============================================================================
# Configuration
# ============================================================================

class Config:
    n_embed = 24
    n_layers = 1
    dropout = 0.2
    block_size = 64       # max text sequence length (words)
    kg_block_size = 24    # max KG sequence length (BPE: multi-token head + rel + multi-token tail)
    batch_size = 32
    max_iters = 5000
    lr = 5e-4
    eval_interval = 500
    eval_iters = 20
    n_seeds = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlm_mask_prob = 0.15

cfg = Config()

# ============================================================================
# Data Loading: Real-World Sources
# ============================================================================

def load_wiki_text(path, max_lines=1000000):
    """Load wiki.en.txt line-by-line, return list of word lists.

    Each line is whitespace-separated words, one sentence per line.
    """
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            words = line.strip().split()
            if len(words) >= 3:  # skip very short lines
                sentences.append(words)
    return sentences


def load_wordnet_synonyms(path):
    """Load WordNet synonyms file. Format: word syn1 syn2 ... per line.

    Returns list of (head, rel, tail) triples with rel='synonym_of'.
    """
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
    """Load FrameNet relations. Format: word rel1 rel2 ... per line.

    Returns list of (head, rel, tail) triples with rel='frame_related'.
    """
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
    """Load BATS 3.0 analogies. Directory structure:
    BATS_3.0/
      1_Inflectional_morphology/
        I01 [noun - plural_reg].txt  (word1\\tword2 per line)
      2_Derivational_morphology/
      3_Encyclopedic_semantics/
      4_Lexicographic_semantics/

    Returns list of (head, rel, tail) triples where rel is category-based.
    """
    triples = []
    for category_dir in sorted(os.listdir(bats_dir)):
        cat_path = os.path.join(bats_dir, category_dir)
        if not os.path.isdir(cat_path):
            continue
        for fname in sorted(os.listdir(cat_path)):
            if not fname.endswith('.txt'):
                continue
            # Extract relation name from filename, e.g. "I01 [noun - plural_reg]"
            rel_name = fname.replace('.txt', '').strip()
            # Simplify: use the bracket content as relation
            if '[' in rel_name and ']' in rel_name:
                rel_name = rel_name[rel_name.index('[')+1:rel_name.index(']')].strip()
            rel_name = rel_name.replace(' ', '_').replace('-', '_').lower()

            fpath = os.path.join(cat_path, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        head, tail = parts[0].strip(), parts[1].strip()
                        # BATS can have multiple answers separated by /
                        for t in tail.split('/'):
                            t = t.strip()
                            if t:
                                triples.append((head, rel_name, t))
    return triples


def load_google_analogies(path):
    """Load Google analogy questions. Format:
    : category_name
    A B C D  (A is to B as C is to D)

    Returns list of (head, rel, tail) triples.
    Each line gives two pairs sharing a relation.
    """
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
    # Deduplicate
    triples = list(set(triples))
    return triples


def load_word_analogies(path):
    """Load word analogies. Format: singular plural singular plural per line.

    Returns list of (head, rel, tail) triples with rel='plural_of'.
    """
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                s1, p1, s2, p2 = parts
                triples.append((s1, "plural_of", p1))
                triples.append((s2, "plural_of", p2))
    triples = list(set(triples))
    return triples


# ============================================================================
# Vocabulary and Tokenization (Word-Level)
# ============================================================================

def _collect_kg_relations(triples):
    """Collect unique relation names from triples."""
    return sorted(set(rel for _, rel, _ in triples))


class Vocabulary:
    """BPE subword vocabulary with special tokens."""

    def __init__(self):
        self.tokenizer = None       # HuggingFace BPE tokenizer
        self.word2idx = {}           # maps token string -> ID (for special/relation tokens)
        self.idx2word = {}           # maps ID -> token string (for special/relation tokens)
        self.size = 0
        self.kg_relations = {}

        # Special tokens — assigned IDs during build_bpe()
        self.PAD = None
        self.BOS = None
        self.EOS = None
        self.MASK = None
        self.UNK = None
        self._special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
        self._relation_tokens = []

    def add_relation(self, rel):
        """Add a KG relation token like <synonym_of>."""
        token = f"<{rel}>"
        if token not in self._relation_tokens:
            self._relation_tokens.append(token)
        inv_token = f"<inverse_{rel}>"
        if inv_token not in self._relation_tokens:
            self._relation_tokens.append(inv_token)

    def build_bpe(self, sentences, kg_entity_words, vocab_size=16000):
        """Train BPE tokenizer on wiki sentences + KG entity words."""
        all_special = self._special_tokens + self._relation_tokens

        # Write training corpus to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                         encoding='utf-8') as f:
            tmp_path = f.name
            for words in sentences:
                f.write(" ".join(words) + "\n")
            # Add KG words one per line so BPE learns subwords for them
            for w in kg_entity_words:
                f.write(w + "\n")

        # Train BPE
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=all_special,
        )
        tokenizer.train([tmp_path], trainer)
        os.unlink(tmp_path)

        self.tokenizer = tokenizer
        self.size = tokenizer.get_vocab_size()

        # Map special token names to IDs
        vocab_map = tokenizer.get_vocab()
        self.PAD = vocab_map["<PAD>"]
        self.BOS = vocab_map["<BOS>"]
        self.EOS = vocab_map["<EOS>"]
        self.MASK = vocab_map["<MASK>"]
        self.UNK = vocab_map["<UNK>"]

        # Map relation tokens
        for token_str in self._relation_tokens:
            rel_name = token_str[1:-1]  # strip < >
            self.kg_relations[rel_name] = vocab_map[token_str]
            self.word2idx[token_str] = vocab_map[token_str]

    def _add(self, token):
        """Look up a token in the BPE vocab (backward compat for linearize)."""
        if self.tokenizer is not None:
            vocab_map = self.tokenizer.get_vocab()
            if token in vocab_map:
                self.word2idx[token] = vocab_map[token]
                return vocab_map[token]
        return self.UNK

    def encode_sentence(self, words):
        """Encode list of words -> list of BPE token IDs."""
        text = " ".join(words)
        return self.tokenizer.encode(text).ids

    def encode_entity(self, entity):
        """Encode a KG entity word -> list of BPE token IDs (may be multi-token)."""
        return self.tokenizer.encode(entity).ids

    def encode_kg_triple(self, head, rel, tail):
        """Encode KG triple -> dict with head/tail as variable-length token lists."""
        head_ids = self.encode_entity(head)
        rel_id = self.kg_relations[rel]
        tail_ids = self.encode_entity(tail)
        return {
            "head": head_ids,
            "rel": rel,
            "rel_token": rel_id,
            "tail": tail_ids,
        }

    def decode(self, indices):
        """Decode token IDs back to string."""
        return self.tokenizer.decode(indices)


# ============================================================================
# Dataset Classes
# ============================================================================

class TextDataset:
    """Dataset for text next-token prediction. Samples word/subword sequences."""

    def __init__(self, sentences, vocab, block_size):
        self.vocab = vocab
        self.block_size = block_size

        self.encoded = []
        for item in sentences:
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], int):
                # Pre-encoded token IDs (from linearized KG)
                tokens = [vocab.BOS] + item + [vocab.EOS]
            else:
                # Word list — BPE encode
                tokens = [vocab.BOS] + vocab.encode_sentence(item) + [vocab.EOS]
            self.encoded.append(torch.tensor(tokens, dtype=torch.long))

        self.data = torch.cat(self.encoded)

    def get_batch(self, batch_size, device):
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        max_len = min(max(len(s) for s in batch), self.block_size + 1)
        x = torch.full((batch_size, max_len - 1), self.vocab.PAD, dtype=torch.long)
        y = torch.full((batch_size, max_len - 1), -100, dtype=torch.long)

        for i, seq in enumerate(batch):
            # Truncate if needed
            seq = seq[:max_len]
            seq_len = len(seq) - 1
            x[i, :seq_len] = seq[:-1]
            y[i, :seq_len] = seq[1:]

        return x.to(device), y.to(device)


class KGDataset:
    """Dataset for KG training with MLM or tail-prediction."""

    def __init__(self, triples, vocab, device, inverse_kg=False):
        self.triples = triples
        self.vocab = vocab
        self.device = device

        self.encoded = []
        for head, rel, tail in triples:
            enc = vocab.encode_kg_triple(head, rel, tail)
            enc["inverse"] = False
            self.encoded.append(enc)

        if inverse_kg:
            for head, rel, tail in triples:
                inv_rel = f"inverse_{rel}"
                enc = vocab.encode_kg_triple(tail, inv_rel, head)
                enc["inverse"] = True
                self.encoded.append(enc)

    def get_mlm_batch_flat(self, batch_size, device, mask_prob=0.15):
        """Get a batch for flat KG models (D/D'): relation is a token in the sequence."""
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        seqs = []
        for b in batch:
            seq = b["head"] + [b["rel_token"]] + b["tail"]
            seqs.append(seq)

        max_len = max(len(s) for s in seqs)

        tokens = torch.full((batch_size, max_len), self.vocab.PAD, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            tokens[i, :len(seq)] = seq_t

            mask = torch.rand(len(seq)) < mask_prob
            if mask.sum() == 0:
                mask[torch.randint(0, len(seq), (1,))] = True

            for j in range(len(seq)):
                if mask[j]:
                    targets[i, j] = seq_t[j]
                    tokens[i, j] = self.vocab.MASK

        rel_names = [batch[i]["rel"] for i in range(batch_size)]
        return tokens.to(device), targets.to(device), rel_names

    def get_mlm_batch_slotted(self, batch_size, device, mask_prob=0.15):
        """Get a batch for Model A/A': HEAD + REL + TAIL slots."""
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        seqs = []
        head_lens = []
        for b in batch:
            seq = b["head"] + [b["rel_token"]] + b["tail"]
            seqs.append(seq)
            head_lens.append(len(b["head"]))

        max_len = max(len(s) for s in seqs)

        tokens = torch.full((batch_size, max_len), self.vocab.PAD, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            tokens[i, :len(seq)] = seq_t

            mask = torch.rand(len(seq)) < mask_prob
            if mask.sum() == 0:
                mask[torch.randint(0, len(seq), (1,))] = True

            for j in range(len(seq)):
                if mask[j]:
                    targets[i, j] = seq_t[j]
                    tokens[i, j] = self.vocab.MASK

        rel_names = [batch[i]["rel"] for i in range(batch_size)]
        return tokens.to(device), targets.to(device), head_lens, rel_names

    def get_mlm_batch_native(self, batch_size, device, mask_prob=0.15):
        """Get a batch for native KG models (E/E'): only word tokens, no relation token."""
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        seqs = []
        head_lens = []
        for b in batch:
            seq = b["head"] + b["tail"]
            seqs.append(seq)
            head_lens.append(len(b["head"]))

        max_len = max(len(s) for s in seqs)

        char_tokens = torch.full((batch_size, max_len), self.vocab.PAD, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            char_tokens[i, :len(seq)] = seq_t

            mask = torch.rand(len(seq)) < mask_prob
            if mask.sum() == 0:
                mask[torch.randint(0, len(seq), (1,))] = True

            for j in range(len(seq)):
                if mask[j]:
                    targets[i, j] = seq_t[j]
                    char_tokens[i, j] = self.vocab.MASK

        rel_names = [batch[i]["rel"] for i in range(batch_size)]
        negate_angles = [batch[i].get("inverse", False) for i in range(batch_size)]
        return char_tokens.to(device), targets.to(device), head_lens, rel_names, negate_angles

    def get_causal_batch_native(self, batch_size, device):
        """Get a causal batch for E/E'. 50% flip direction."""
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        seqs = []
        head_lens = []
        negate_angles = []
        for b in batch:
            flip = random.random() < 0.5
            if flip:
                seq = b["tail"] + b["head"]
                head_lens.append(len(b["tail"]))
            else:
                seq = b["head"] + b["tail"]
                head_lens.append(len(b["head"]))
            seqs.append(seq)
            negate_angles.append(flip)

        max_len = max(len(s) for s in seqs)

        char_tokens = torch.full((batch_size, max_len), self.vocab.PAD, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            char_tokens[i, :len(seq)] = seq_t
            targets[i, :len(seq) - 1] = seq_t[1:]

        rel_names = [batch[i]["rel"] for i in range(batch_size)]
        return char_tokens.to(device), targets.to(device), head_lens, rel_names, negate_angles


# ============================================================================
# Linearization (for kg_as_text mode)
# ============================================================================

def linearize_kg_triples(triples, kg_relations_inverse):
    """Convert KG triples to linearized text strings with relation tokens."""
    sentences = []
    for head, rel, tail in triples:
        fwd = f"{head} <{rel}> {tail}"
        inv_rel = kg_relations_inverse.get(rel, f"inverse_{rel}")
        inv = f"{tail} <{inv_rel}> {head}"
        sentences.append(fwd)
        sentences.append(inv)
    return sentences


def _encode_linearized_string(s, vocab):
    """Encode a linearized KG string like 'Adam <synonym_of> Brian' into token IDs.

    Splits on <rel> tokens (which are special tokens), BPE-encodes word parts,
    and inserts relation token IDs directly.
    """
    import re
    # Split on angle-bracket tokens, keeping the delimiters
    parts = re.split(r'(<[^>]+>)', s)
    token_ids = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith('<') and part.endswith('>'):
            # Relation token — look up directly
            rel_name = part[1:-1]  # strip < >
            if rel_name in vocab.kg_relations:
                token_ids.append(vocab.kg_relations[rel_name])
            else:
                token_ids.append(vocab.UNK)
        else:
            # Word part — BPE encode
            token_ids.extend(vocab.encode_sentence(part.split()))
    return token_ids


# ============================================================================
# Rotation Utilities
# ============================================================================

def apply_rotation(x, angles):
    """Apply 2D rotation matrices to pairs of dimensions."""
    B, T, C = x.shape
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    x_pairs = x.reshape(B, T, C // 2, 2)
    x_even = x_pairs[..., 0]
    x_odd = x_pairs[..., 1]
    r_even = x_even * cos_a - x_odd * sin_a
    r_odd = x_even * sin_a + x_odd * cos_a
    result = torch.stack([r_even, r_odd], dim=-1)
    return result.reshape(B, T, C)


def apply_inverse_rotation(x, angles):
    """Apply inverse 2D rotation (negate angles)."""
    return apply_rotation(x, -angles)


# ============================================================================
# Shared Building Blocks
# ============================================================================

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class RotaryAttention(nn.Module):
    """Attention with external angles. Supports both commutative and operator-based modes."""

    def __init__(self, n_embed, block_size, dropout=0.2, rotate_v=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.rotate_v = rotate_v
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, angles, causal=True, pad_mask=None):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        k = apply_rotation(k, angles)
        q = apply_rotation(q, angles)

        if self.rotate_v:
            v = apply_rotation(v, angles)

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)

        if causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        if pad_mask is not None:
            pad_mask_k = pad_mask.unsqueeze(1)
            wei = wei.masked_fill(~pad_mask_k, float('-inf'))

        wei = torch.log(torch.exp(wei) + 1)
        wei = self.dropout(wei)
        out = wei @ v

        if self.rotate_v:
            out = apply_inverse_rotation(out, angles)

        out = self.proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout=0.2, rotate_v=False):
        super().__init__()
        self.sa_head = RotaryAttention(n_embed, block_size, dropout, rotate_v)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles, causal=True, pad_mask=None):
        x = x + self.sa_head(self.ln1(x), angles, causal, pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================================
# Model A/A': RoPE + Slot Angles, Native KG
# ============================================================================

class ModelA(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.slot_angles = nn.Parameter(torch.randn(3, n_embed // 2) * 0.1)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def _rope_angles(self, T, device):
        positions = torch.arange(T, device=device, dtype=torch.float)
        angles = torch.outer(positions, self.base_freq)
        return angles.unsqueeze(0)

    def _kg_angles(self, head_lens, seq_len, batch_size, rel_names, device):
        angles = torch.zeros(batch_size, seq_len, self.n_embed // 2, device=device)
        for i in range(batch_size):
            h_len = head_lens[i]
            rel_pos = h_len
            tail_start = h_len + 1
            for j in range(h_len):
                angles[i, j] = j * self.base_freq + self.slot_angles[0]
            angles[i, rel_pos] = 0 * self.base_freq + self.slot_angles[1]
            for j in range(seq_len - tail_start):
                if tail_start + j < seq_len:
                    angles[i, tail_start + j] = j * self.base_freq + self.slot_angles[2]
        return angles

    def forward_text(self, idx, targets=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._rope_angles(T, idx.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_kg(self, tokens, targets, head_lens, rel_names):
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._kg_angles(head_lens, T, B, rel_names, tokens.device)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model B/B': Standard RoPE, Linearized KG-as-Text
# ============================================================================

class ModelB(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def _rope_angles(self, T, device):
        positions = torch.arange(T, device=device, dtype=torch.float)
        angles = torch.outer(positions, self.base_freq)
        return angles.unsqueeze(0)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._rope_angles(T, idx.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits


# ============================================================================
# Model C/C': Per-Token Cumsum (Journey), Linearized KG-as-Text
# ============================================================================

class ModelC(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.expander(self.token_embedding(idx))
        pad_mask = (idx != 0)
        raw_angles = self.angle_embedding(idx)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits


# ============================================================================
# Model D/D': Per-Token Cumsum, Flat KG (Relation as Token)
# ============================================================================

class ModelD(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def _cumsum_angles(self, idx):
        raw_angles = self.angle_embedding(idx)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        return angles

    def forward_text(self, idx, targets=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.expander(self.token_embedding(idx))
        angles = self._cumsum_angles(idx)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_kg(self, tokens, targets):
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.expander(self.token_embedding(tokens))
        angles = self._cumsum_angles(tokens)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model E/E': Per-Token Cumsum + Relation Operator, Native KG
# ============================================================================

class ModelE(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.relation_angles = nn.Parameter(torch.randn(n_relations, n_embed // 2) * 0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.rel_to_idx = {}  # populated at runtime

    def _cumsum_angles_text(self, idx):
        raw_angles = self.angle_embedding(idx)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        return angles

    def _cumsum_angles_kg(self, char_tokens, head_lens, rel_names, device,
                          negate_angles=None):
        B, T = char_tokens.shape
        raw_char_angles = self.angle_embedding(char_tokens)
        ext_angles = torch.zeros(B, T + 1, self.n_embed // 2, device=device)
        for i in range(B):
            h_len = head_lens[i]
            rel_idx = self.rel_to_idx[rel_names[i]]
            ext_angles[i, :h_len] = raw_char_angles[i, :h_len]
            rel_angle = self.relation_angles[rel_idx]
            if negate_angles is not None and negate_angles[i]:
                rel_angle = -rel_angle
            ext_angles[i, h_len] = rel_angle
            t_len = T - h_len
            ext_angles[i, h_len + 1:h_len + 1 + t_len] = raw_char_angles[i, h_len:h_len + t_len]
        ext_cumsum = torch.flip(ext_angles, dims=(1,))
        ext_cumsum = torch.cumsum(ext_cumsum, dim=1)
        ext_cumsum = torch.flip(ext_cumsum, dims=(1,))
        angles = torch.zeros(B, T, self.n_embed // 2, device=device)
        for i in range(B):
            h_len = head_lens[i]
            t_len = T - h_len
            angles[i, :h_len] = ext_cumsum[i, :h_len]
            angles[i, h_len:h_len + t_len] = ext_cumsum[i, h_len + 1:h_len + 1 + t_len]
        return angles

    def forward_text(self, idx, targets=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.expander(self.token_embedding(idx))
        angles = self._cumsum_angles_text(idx)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_kg(self, char_tokens, targets, head_lens, rel_names, negate_angles=None):
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)
        x = self.expander(self.token_embedding(char_tokens))
        angles = self._cumsum_angles_kg(char_tokens, head_lens, rel_names, char_tokens.device,
                                        negate_angles=negate_angles)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, char_tokens, targets, head_lens, rel_names, negate_angles):
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)
        x = self.expander(self.token_embedding(char_tokens))
        angles = self._cumsum_angles_kg(char_tokens, head_lens, rel_names,
                                        char_tokens.device, negate_angles=negate_angles)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model F/F': Fixed-Angle RoPE, Flat KG (Relation as Token)
# ============================================================================

class ModelF(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def _rope_angles(self, T, device):
        positions = torch.arange(T, device=device, dtype=torch.float)
        angles = torch.outer(positions, self.base_freq)
        return angles.unsqueeze(0)

    def forward_text(self, idx, targets=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._rope_angles(T, idx.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_kg(self, tokens, targets):
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._rope_angles(T, tokens.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model G/G': Relation-Dependent Slot Angles, Native KG
# ============================================================================

class ModelG(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.slot_angles = nn.Parameter(torch.randn(n_relations, 3, n_embed // 2) * 0.1)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.rel_to_idx = {}  # populated at runtime

    def _rope_angles(self, T, device):
        positions = torch.arange(T, device=device, dtype=torch.float)
        angles = torch.outer(positions, self.base_freq)
        return angles.unsqueeze(0)

    def _kg_angles(self, head_lens, seq_len, batch_size, rel_names, device):
        angles = torch.zeros(batch_size, seq_len, self.n_embed // 2, device=device)
        for i in range(batch_size):
            h_len = head_lens[i]
            rel_pos = h_len
            tail_start = h_len + 1
            rel_idx = self.rel_to_idx[rel_names[i]]
            for j in range(h_len):
                angles[i, j] = j * self.base_freq + self.slot_angles[rel_idx, 0]
            angles[i, rel_pos] = 0 * self.base_freq + self.slot_angles[rel_idx, 1]
            for j in range(seq_len - tail_start):
                if tail_start + j < seq_len:
                    angles[i, tail_start + j] = j * self.base_freq + self.slot_angles[rel_idx, 2]
        return angles

    def forward_text(self, idx, targets=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._rope_angles(T, idx.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_kg(self, tokens, targets, head_lens, rel_names):
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._kg_angles(head_lens, T, B, rel_names, tokens.device)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model H/H': Fixed Angles + Relation Operator, Native KG
# ============================================================================

class ModelH(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.relation_angles = nn.Parameter(torch.randn(n_relations, n_embed // 2) * 0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.rel_to_idx = {}  # populated at runtime

    def _cumsum_angles_text(self, T, B, device):
        raw_angles = self.base_freq.unsqueeze(0).expand(T, -1)
        angles = torch.flip(raw_angles, dims=(0,))
        angles = torch.cumsum(angles, dim=0)
        angles = torch.flip(angles, dims=(0,))
        return angles.unsqueeze(0).expand(B, -1, -1)

    def _cumsum_angles_kg(self, T, head_lens, rel_names, device, negate_angles=None):
        B = len(head_lens)
        ext_angles = torch.zeros(B, T + 1, self.n_embed // 2, device=device)
        for i in range(B):
            h_len = head_lens[i]
            ext_angles[i, :h_len] = self.base_freq.unsqueeze(0).expand(h_len, -1)
            rel_idx = self.rel_to_idx[rel_names[i]]
            rel_angle = self.relation_angles[rel_idx]
            if negate_angles is not None and negate_angles[i]:
                rel_angle = -rel_angle
            ext_angles[i, h_len] = rel_angle
            t_len = T - h_len
            ext_angles[i, h_len + 1:h_len + 1 + t_len] = self.base_freq.unsqueeze(0).expand(t_len, -1)
        ext_cumsum = torch.flip(ext_angles, dims=(1,))
        ext_cumsum = torch.cumsum(ext_cumsum, dim=1)
        ext_cumsum = torch.flip(ext_cumsum, dims=(1,))
        angles = torch.zeros(B, T, self.n_embed // 2, device=device)
        for i in range(B):
            h_len = head_lens[i]
            t_len = T - h_len
            angles[i, :h_len] = ext_cumsum[i, :h_len]
            angles[i, h_len:h_len + t_len] = ext_cumsum[i, h_len + 1:h_len + 1 + t_len]
        return angles

    def forward_text(self, idx, targets=None):
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._cumsum_angles_text(T, B, idx.device)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_kg(self, char_tokens, targets, head_lens, rel_names, negate_angles=None):
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)
        x = self.token_embedding(char_tokens)
        angles = self._cumsum_angles_kg(T, head_lens, rel_names, char_tokens.device,
                                        negate_angles=negate_angles)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, char_tokens, targets, head_lens, rel_names, negate_angles):
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)
        x = self.token_embedding(char_tokens)
        angles = self._cumsum_angles_kg(T, head_lens, rel_names, char_tokens.device,
                                        negate_angles=negate_angles)
        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model I/I': Per-Layer Angle Computation, Native KG
# ============================================================================

class ModelI(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.angle_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embed, n_embed),
                nn.GELU(),
                nn.Linear(n_embed, n_embed // 2),
            )
            for _ in range(n_layers)
        ])
        self.relation_angles = nn.Parameter(torch.randn(n_relations, n_embed // 2) * 0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.rel_to_idx = {}  # populated at runtime

    def _cumsum_angles_text(self, raw_angles):
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        return angles

    def _cumsum_angles_kg(self, raw_angles, head_lens, rel_names, device,
                          negate_angles=None):
        B, T, D = raw_angles.shape
        ext_angles = torch.zeros(B, T + 1, D, device=device)
        for i in range(B):
            h_len = head_lens[i]
            rel_idx = self.rel_to_idx[rel_names[i]]
            ext_angles[i, :h_len] = raw_angles[i, :h_len]
            rel_angle = self.relation_angles[rel_idx]
            if negate_angles is not None and negate_angles[i]:
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
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        for l, block in enumerate(self.blocks):
            raw_angles = self.angle_projectors[l](x)
            angles = self._cumsum_angles_text(raw_angles)
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_kg(self, char_tokens, targets, head_lens, rel_names, negate_angles=None):
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)
        x = self.token_embedding(char_tokens)
        for l, block in enumerate(self.blocks):
            raw_angles = self.angle_projectors[l](x)
            angles = self._cumsum_angles_kg(raw_angles, head_lens, rel_names,
                                            char_tokens.device, negate_angles=negate_angles)
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, char_tokens, targets, head_lens, rel_names, negate_angles):
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)
        x = self.token_embedding(char_tokens)
        for l, block in enumerate(self.blocks):
            raw_angles = self.angle_projectors[l](x)
            angles = self._cumsum_angles_kg(raw_angles, head_lens, rel_names,
                                            char_tokens.device, negate_angles=negate_angles)
            x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Tier Split and Data Preparation
# ============================================================================

ALL_TIERS = [
    "memorization", "transfer", "generalization",
    "kg_exclusive_memorization", "kg_exclusive_generalization",
    "text_exclusive_memorization", "text_exclusive_generalization",
]


def split_data_into_tiers(triples, sentences, seed=42):
    """Split KG triples and text into 7 evaluation tiers."""
    rng = random.Random(seed)

    text_words = set()
    for words in sentences:
        text_words.update(w.lower() for w in words)

    rel_groups = defaultdict(list)
    for h, r, t in triples:
        rel_groups[r].append((h, r, t))

    train_triples = []
    eval_triples = []
    for rel, group in rel_groups.items():
        rng.shuffle(group)
        split_idx = max(1, int(len(group) * 0.8))
        train_triples.extend(group[:split_idx])
        eval_triples.extend(group[split_idx:])

    eval_data = {tier: [] for tier in ALL_TIERS}

    train_kg_pairs = set()
    for h, r, t in train_triples:
        train_kg_pairs.add((h.lower(), t.lower()))
        train_kg_pairs.add((t.lower(), h.lower()))

    for h, r, t in eval_triples:
        h_in_text = h.lower() in text_words
        t_in_text = t.lower() in text_words
        both_in_text = h_in_text and t_in_text
        pair_in_kg = (h.lower(), t.lower()) in train_kg_pairs

        if both_in_text and pair_in_kg:
            eval_data["memorization"].append({"head": h, "rel": r, "tail": t})
        elif both_in_text and not pair_in_kg:
            eval_data["transfer"].append({"head": h, "rel": r, "tail": t})
        elif not both_in_text and pair_in_kg:
            eval_data["kg_exclusive_memorization"].append({"head": h, "rel": r, "tail": t})
        elif not both_in_text and not pair_in_kg:
            eval_data["kg_exclusive_generalization"].append({"head": h, "rel": r, "tail": t})

    rng.shuffle(train_triples)
    for h, r, t in train_triples[:len(train_triples)//10]:
        if h.lower() in text_words and t.lower() in text_words:
            eval_data["generalization"].append({"head": h, "rel": r, "tail": t})

    kg_words = set()
    for h, r, t in triples:
        kg_words.add(h.lower())
        kg_words.add(t.lower())

    text_excl_pairs_mem = []
    text_excl_pairs_gen = []
    for words in sentences:
        lower_words = [w.lower() for w in words]
        text_only = [w for w in lower_words if w in text_words and w not in kg_words]
        if len(text_only) >= 2:
            h, t = text_only[0], text_only[1]
            if len(text_excl_pairs_mem) < 50:
                text_excl_pairs_mem.append({"head": h, "rel": "text_cooccurrence", "tail": t})
            elif len(text_excl_pairs_gen) < 50:
                text_excl_pairs_gen.append({"head": h, "rel": "text_cooccurrence", "tail": t})
            if len(text_excl_pairs_mem) >= 50 and len(text_excl_pairs_gen) >= 50:
                break

    eval_data["text_exclusive_memorization"] = text_excl_pairs_mem[:50]
    eval_data["text_exclusive_generalization"] = text_excl_pairs_gen[:50]

    for tier in eval_data:
        if len(eval_data[tier]) > 200:
            rng.shuffle(eval_data[tier])
            eval_data[tier] = eval_data[tier][:200]

    return train_triples, sentences, eval_data, text_words


def prepare_data(wiki_path, wiki_lines, vocab_size,
                 wordnet_path=None, framenet_path=None,
                 bats_dir=None, google_analogies_path=None,
                 word_analogies_path=None,
                 inverse_kg=False, kg_as_text=False, seed=42):
    """Prepare all data for Exp 8."""
    print("Loading data...")

    sentences = load_wiki_text(wiki_path, wiki_lines)
    print(f"  Wiki text: {len(sentences)} sentences")

    all_triples = []
    if wordnet_path and os.path.exists(wordnet_path):
        wn = load_wordnet_synonyms(wordnet_path)
        print(f"  WordNet: {len(wn)} triples")
        all_triples.extend(wn)
    if framenet_path and os.path.exists(framenet_path):
        fn = load_framenet(framenet_path)
        print(f"  FrameNet: {len(fn)} triples")
        all_triples.extend(fn)
    if bats_dir and os.path.exists(bats_dir):
        bt = load_bats_analogies(bats_dir)
        print(f"  BATS: {len(bt)} triples")
        all_triples.extend(bt)
    if google_analogies_path and os.path.exists(google_analogies_path):
        ga = load_google_analogies(google_analogies_path)
        print(f"  Google analogies: {len(ga)} triples")
        all_triples.extend(ga)
    if word_analogies_path and os.path.exists(word_analogies_path):
        wa = load_word_analogies(word_analogies_path)
        print(f"  Word analogies: {len(wa)} triples")
        all_triples.extend(wa)

    if not all_triples:
        print("WARNING: No KG triples loaded! Using placeholder.")
        all_triples = [("placeholder_h", "placeholder_rel", "placeholder_t")]

    print(f"  Total KG triples: {len(all_triples)}")

    train_triples, train_sentences, eval_data, text_words = split_data_into_tiers(
        all_triples, sentences, seed=seed)

    kg_relations_list = _collect_kg_relations(all_triples)
    if "text_cooccurrence" not in kg_relations_list:
        kg_relations_list.append("text_cooccurrence")
    kg_relations_list = sorted(set(kg_relations_list))
    kg_relations_inverse = {r: f"inverse_{r}" for r in kg_relations_list}

    vocab = Vocabulary()
    for rel in kg_relations_list:
        vocab.add_relation(rel)

    # Collect all unique KG entity words for BPE training
    kg_entity_words = set()
    for head, rel, tail in all_triples:
        kg_entity_words.add(head)
        kg_entity_words.add(tail)

    vocab.build_bpe(train_sentences, sorted(kg_entity_words), vocab_size=vocab_size)
    print(f"  Vocabulary size: {vocab.size}")

    text_dataset_base = TextDataset(train_sentences, vocab, cfg.block_size)

    linearized_kg_token_lists = []
    if kg_as_text:
        lin_strs = linearize_kg_triples(train_triples, kg_relations_inverse)
        for s in lin_strs:
            # Encode linearized string: split on <rel> tokens, BPE-encode word
            # parts, insert relation token IDs directly
            token_ids = _encode_linearized_string(s, vocab)
            linearized_kg_token_lists.append(token_ids)

    lin_sents = train_sentences + linearized_kg_token_lists if kg_as_text else train_sentences
    text_dataset_linearized = TextDataset(lin_sents, vocab, cfg.block_size)
    kg_dataset = KGDataset(train_triples, vocab, cfg.device, inverse_kg=inverse_kg)

    eval_prompts = _build_text_eval_prompts(eval_data, vocab)
    kg_eval_prompts = _build_kg_eval_prompts(eval_data)
    linearized_eval_prompts = None
    if kg_as_text:
        linearized_eval_prompts = _build_linearized_eval_prompts(eval_data, vocab, kg_relations_inverse)

    return (vocab, text_dataset_base, text_dataset_linearized, kg_dataset,
            eval_prompts, kg_eval_prompts, linearized_eval_prompts, kg_relations_list)


def _build_text_eval_prompts(eval_data, vocab):
    """Build text cloze prompts: head word -> predict tail."""
    prompts = []
    for tier, items in eval_data.items():
        for item in items:
            h, r, t = item["head"], item["rel"], item["tail"]
            prompts.append({
                "tier": tier, "prompt": f"{h} ", "target": t, "relation": r,
                "prompt_tokens": vocab.encode_entity(h),
                "target_tokens": vocab.encode_entity(t),
            })
    return prompts


def _build_kg_eval_prompts(eval_data):
    """Build KG evaluation prompts."""
    prompts = []
    for tier, items in eval_data.items():
        for item in items:
            prompts.append({
                "tier": tier, "head": item["head"], "rel": item["rel"],
                "tail": item["tail"], "relation": item["rel"],
            })
    return prompts


def _build_linearized_eval_prompts(eval_data, vocab, kg_relations_inverse):
    """Build linearized KG eval prompts."""
    prompts = []
    for tier, items in eval_data.items():
        for item in items:
            h, r, t = item["head"], item["rel"], item["tail"]
            inv_r = kg_relations_inverse.get(r, f"inverse_{r}")
            # BPE-encode entity parts, insert relation token ID directly
            h_tokens = vocab.encode_entity(h)
            t_tokens = vocab.encode_entity(t)
            rel_id = vocab.kg_relations.get(r, vocab.UNK)
            inv_rel_id = vocab.kg_relations.get(inv_r, vocab.UNK)
            prompts.append({
                "tier": tier, "prompt": f"{h} <{r}> ", "target": t, "relation": r,
                "prompt_tokens": h_tokens + [rel_id],
                "target_tokens": t_tokens,
            })
            prompts.append({
                "tier": tier, "prompt": f"{t} <{inv_r}> ", "target": h, "relation": inv_r,
                "prompt_tokens": t_tokens + [inv_rel_id],
                "target_tokens": h_tokens,
            })
    return prompts


# ============================================================================
# Evaluation
# ============================================================================

def _forward_kg_eval(model, tokens, targets, head_lens, rel_names, model_type):
    """Run forward_kg for the appropriate model type and return logits."""
    if model_type in ("A", "G"):
        logits, _ = model.forward_kg(tokens, targets, head_lens, rel_names)
    elif model_type in ("D", "F"):
        logits, _ = model.forward_kg(tokens, targets)
    elif model_type in ("E", "H", "I"):
        logits, _ = model.forward_kg(tokens, targets, head_lens, rel_names)
    return logits


def evaluate_model(model, eval_prompts, vocab, config, model_name="?"):
    """Evaluate a model on text cloze prompts. Word-level version."""
    model.eval()
    model.to(config.device)

    results = defaultdict(lambda: defaultdict(list))
    batch_size = config.batch_size

    processed = []
    for p in eval_prompts:
        pt = p["prompt_tokens"]
        tt = p["target_tokens"]
        if len(pt) > config.block_size:
            pt = pt[-config.block_size:]
        processed.append({
            "prompt_tokens": pt, "target_tokens": tt,
            "prompt_len": len(pt), "target_len": len(tt),
            "tier": p["tier"], "relation": p["relation"],
            "prompt": p["prompt"], "target": p["target"],
        })

    groups = defaultdict(list)
    for idx, pp in enumerate(processed):
        groups[(pp["prompt_len"], pp["target_len"])].append(idx)

    n = len(processed)
    total_log_probs = [0.0] * n
    log_prob_firsts = [None] * n
    log_prob_lasts = [None] * n
    generated = [[] for _ in range(n)]
    all_correct = [True] * n
    all_in_top5s = [True] * n

    with torch.no_grad():
        for (prompt_len, target_len), prompt_indices in groups.items():
            if target_len == 0:
                continue
            current_tokens_map = {}
            for pi in prompt_indices:
                current_tokens_map[pi] = processed[pi]["prompt_tokens"].copy()

            for t_idx in range(target_len):
                input_len = min(prompt_len + t_idx, config.block_size)
                for sb_start in range(0, len(prompt_indices), batch_size):
                    sb_indices = prompt_indices[sb_start:sb_start + batch_size]
                    B = len(sb_indices)
                    x = torch.zeros(B, input_len, dtype=torch.long, device=config.device)
                    for bi, pi in enumerate(sb_indices):
                        toks = current_tokens_map[pi][-config.block_size:]
                        x[bi, :len(toks)] = torch.tensor(toks, dtype=torch.long)
                    logits = model.predict_text(x)
                    for bi, pi in enumerate(sb_indices):
                        pp = processed[pi]
                        t_tok = pp["target_tokens"][t_idx]
                        step_logits = logits[bi, -1, :]
                        log_probs = F.log_softmax(step_logits, dim=0)
                        lp = log_probs[t_tok].item()
                        total_log_probs[pi] += lp
                        if t_idx == 0:
                            log_prob_firsts[pi] = lp
                        log_prob_lasts[pi] = lp
                        pred = torch.argmax(step_logits).item()
                        generated[pi].append(pred)
                        if pred != t_tok:
                            all_correct[pi] = False
                        if all_in_top5s[pi]:
                            top5 = torch.topk(step_logits, k=min(5, step_logits.shape[0])).indices.tolist()
                            if t_tok not in top5:
                                all_in_top5s[pi] = False
                        current_tokens_map[pi].append(t_tok)

    for pi, pp in enumerate(processed):
        tl = pp["target_len"]
        if tl > 0:
            ppl = np.exp(-total_log_probs[pi] / tl)
            fc_ppl = np.exp(-log_prob_firsts[pi]) if log_prob_firsts[pi] is not None else ppl
            lc_ppl = np.exp(-log_prob_lasts[pi]) if log_prob_lasts[pi] is not None else ppl
        else:
            ppl = fc_ppl = lc_ppl = 1.0
        hit1 = 1 if generated[pi] == pp["target_tokens"] else 0
        hit5 = 1 if all_in_top5s[pi] else 0
        results[pp["tier"]][pp["relation"]].append({
            "hit1": hit1, "hit5": hit5, "ppl": ppl,
            "first_char_ppl": fc_ppl, "last_char_ppl": lc_ppl,
            "full_correct": 1 if all_correct[pi] else 0,
            "prompt": pp["prompt"], "target": pp["target"],
        })

    summary = {}
    for tier in results:
        tr = {"hit1": [], "hit5": [], "ppl": [], "first_char_ppl": [], "last_char_ppl": [], "full_correct": []}
        for rel in results[tier]:
            for r in results[tier][rel]:
                for k in tr:
                    tr[k].append(r[k])
        summary[tier] = {
            "hit1": np.mean(tr["hit1"]), "hit5": np.mean(tr["hit5"]),
            "ppl": np.exp(np.mean(np.log(tr["ppl"]))) if tr["ppl"] else 1.0,
            "first_char_ppl": np.exp(np.mean(np.log(tr["first_char_ppl"]))) if tr["first_char_ppl"] else 1.0,
            "last_char_ppl": np.exp(np.mean(np.log(tr["last_char_ppl"]))) if tr["last_char_ppl"] else 1.0,
            "full_correct": np.mean(tr["full_correct"]), "n": len(tr["hit1"]),
        }

    relation_summary = {}
    for tier in results:
        relation_summary[tier] = {}
        for rel in results[tier]:
            rd = results[tier][rel]
            if rd:
                relation_summary[tier][rel] = {
                    "hit1": np.mean([r["hit1"] for r in rd]),
                    "hit5": np.mean([r["hit5"] for r in rd]),
                    "ppl": np.exp(np.mean([np.log(r["ppl"]) for r in rd])),
                    "full_correct": np.mean([r["full_correct"] for r in rd]),
                    "n": len(rd),
                }

    print(f"\n{'='*60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*60}")
    for tier in ALL_TIERS:
        if tier in summary:
            s = summary[tier]
            print(f"  {tier:>35s}: h@1={s['hit1']:.3f}  h@5={s['hit5']:.3f}  "
                  f"ppl={s['ppl']:.2f}  (n={s['n']})")

    model.train()
    return summary, relation_summary, results


def evaluate_model_kg(model, kg_eval_prompts, vocab, config, model_name="?", model_type="A"):
    """Evaluate KG model on completion: given head+relation, predict tail. Word-level."""
    model.eval()
    model.to(config.device)

    encoded_prompts = []
    for p in kg_eval_prompts:
        ht = vocab.encode_entity(p["head"])
        tt = vocab.encode_entity(p["tail"])
        encoded_prompts.append({
            "head_tokens": ht, "tail_tokens": tt,
            "head_len": len(ht), "tail_len": len(tt),
            "rel_name": p["rel"], "tier": p["tier"],
            "relation": p["relation"], "head": p["head"], "tail": p["tail"],
        })

    groups = defaultdict(list)
    for idx, ep in enumerate(encoded_prompts):
        groups[(ep["head_len"], ep["tail_len"])].append(idx)

    n = len(encoded_prompts)
    total_log_probs = [0.0] * n
    all_in_top5s = [True] * n
    hit1s = [1] * n

    results = defaultdict(lambda: defaultdict(list))
    batch_size = config.batch_size

    with torch.no_grad():
        for (head_len, tail_len), prompt_indices in groups.items():
            if model_type in ("E", "H", "I"):
                seq_len = head_len + tail_len
            else:
                seq_len = head_len + 1 + tail_len

            # Pseudo-perplexity: mask one tail position at a time
            for t_idx in range(tail_len):
                for sb_start in range(0, len(prompt_indices), batch_size):
                    sb = prompt_indices[sb_start:sb_start + batch_size]
                    B = len(sb)
                    tokens = torch.full((B, seq_len), vocab.PAD, dtype=torch.long)
                    targets = torch.full((B, seq_len), -100, dtype=torch.long)
                    hl_list, rn_list = [], []
                    for bi, pi in enumerate(sb):
                        ep = encoded_prompts[pi]
                        if model_type in ("E", "H", "I"):
                            seq = list(ep["head_tokens"]) + list(ep["tail_tokens"])
                            mask_pos = head_len + t_idx
                        else:
                            rt = vocab.kg_relations[ep["rel_name"]]
                            seq = list(ep["head_tokens"]) + [rt] + list(ep["tail_tokens"])
                            mask_pos = head_len + 1 + t_idx
                        tokens[bi, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                        targets[bi, mask_pos] = tokens[bi, mask_pos].item()
                        tokens[bi, mask_pos] = vocab.MASK
                        hl_list.append(head_len)
                        rn_list.append(ep["rel_name"])
                    tokens, targets = tokens.to(config.device), targets.to(config.device)
                    logits = _forward_kg_eval(model, tokens, targets, hl_list, rn_list, model_type)
                    mp = head_len + t_idx if model_type in ("E", "H", "I") else head_len + 1 + t_idx
                    for bi, pi in enumerate(sb):
                        ep = encoded_prompts[pi]
                        sl = logits[bi, mp, :]
                        lp = F.log_softmax(sl, dim=0)[ep["tail_tokens"][t_idx]].item()
                        total_log_probs[pi] += lp
                        top5 = torch.topk(sl, k=min(5, sl.shape[0])).indices.tolist()
                        if ep["tail_tokens"][t_idx] not in top5:
                            all_in_top5s[pi] = False

            # Simultaneous mask for hit@1
            for sb_start in range(0, len(prompt_indices), batch_size):
                sb = prompt_indices[sb_start:sb_start + batch_size]
                B = len(sb)
                tokens = torch.full((B, seq_len), vocab.PAD, dtype=torch.long)
                targets = torch.full((B, seq_len), -100, dtype=torch.long)
                hl_list, rn_list = [], []
                for bi, pi in enumerate(sb):
                    ep = encoded_prompts[pi]
                    if model_type in ("E", "H", "I"):
                        seq = list(ep["head_tokens"]) + list(ep["tail_tokens"])
                        for ti in range(tail_len):
                            pos = head_len + ti
                            targets[bi, pos] = seq[pos]
                            seq[pos] = vocab.MASK
                    else:
                        rt = vocab.kg_relations[ep["rel_name"]]
                        seq = list(ep["head_tokens"]) + [rt] + list(ep["tail_tokens"])
                        for ti in range(tail_len):
                            pos = head_len + 1 + ti
                            targets[bi, pos] = seq[pos]
                            seq[pos] = vocab.MASK
                    tokens[bi, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                    hl_list.append(head_len)
                    rn_list.append(ep["rel_name"])
                tokens, targets = tokens.to(config.device), targets.to(config.device)
                logits = _forward_kg_eval(model, tokens, targets, hl_list, rn_list, model_type)
                for bi, pi in enumerate(sb):
                    ep = encoded_prompts[pi]
                    for ti in range(tail_len):
                        mp = head_len + ti if model_type in ("E", "H", "I") else head_len + 1 + ti
                        if torch.argmax(logits[bi, mp, :]).item() != ep["tail_tokens"][ti]:
                            hit1s[pi] = 0
                            break

    for pi, ep in enumerate(encoded_prompts):
        tl = ep["tail_len"]
        ppl = np.exp(-total_log_probs[pi] / max(tl, 1))
        results[ep["tier"]][ep["relation"]].append({
            "hit1": hit1s[pi], "hit5": 1 if all_in_top5s[pi] else 0,
            "ppl": ppl, "first_char_ppl": ppl, "last_char_ppl": ppl,
            "head": ep["head"], "rel": ep["rel_name"], "tail": ep["tail"],
        })

    summary = {}
    for tier in results:
        tr = {"hit1": [], "hit5": [], "ppl": []}
        for rel in results[tier]:
            for r in results[tier][rel]:
                for k in tr:
                    tr[k].append(r[k])
        summary[tier] = {
            "hit1": np.mean(tr["hit1"]), "hit5": np.mean(tr["hit5"]),
            "ppl": np.exp(np.mean(np.log(tr["ppl"]))) if tr["ppl"] else 1.0,
            "n": len(tr["hit1"]),
        }

    relation_summary = {}
    for tier in results:
        relation_summary[tier] = {}
        for rel in results[tier]:
            rd = results[tier][rel]
            if rd:
                relation_summary[tier][rel] = {
                    "hit1": np.mean([r["hit1"] for r in rd]),
                    "hit5": np.mean([r["hit5"] for r in rd]),
                    "ppl": np.exp(np.mean([np.log(r["ppl"]) for r in rd])),
                    "n": len(rd),
                }

    print(f"\n{'='*60}")
    print(f"  KG Evaluation: {model_name}")
    print(f"{'='*60}")
    for tier in ALL_TIERS:
        if tier in summary:
            s = summary[tier]
            print(f"  {tier:>35s}: h@1={s['hit1']:.3f}  h@5={s['hit5']:.3f}  "
                  f"ppl={s['ppl']:.2f}  (n={s['n']})")

    model.train()
    return summary, relation_summary, results


def evaluate_model_kg_causal(model, kg_eval_prompts, vocab, config, model_name="?"):
    """Evaluate E/E'/H/H'/I/I' with causal KG. Word-level."""
    model.eval()
    model.to(config.device)

    results = defaultdict(lambda: defaultdict(list))
    batch_size = config.batch_size
    model_type_str = model_name.replace("'", "")

    eval_items = []
    for p in kg_eval_prompts:
        ht = vocab.encode_entity(p["head"])
        tt = vocab.encode_entity(p["tail"])
        for direction in ("forward", "backward"):
            if direction == "forward":
                seq = list(ht) + list(tt)
                ctx_len = len(ht)
                pred_tokens = list(tt)
                negate = False
            else:
                seq = list(tt) + list(ht)
                ctx_len = len(tt)
                pred_tokens = list(ht)
                negate = True
            eval_items.append({
                "seq": seq, "ctx_len": ctx_len, "pred_tokens": pred_tokens,
                "pred_len": len(pred_tokens), "negate": negate,
                "rel_name": p["rel"], "tier": p["tier"], "relation": p["relation"],
                "head": p["head"], "tail": p["tail"], "direction": direction,
            })

    groups = defaultdict(list)
    for idx, item in enumerate(eval_items):
        groups[len(item["seq"])].append(idx)

    with torch.no_grad():
        for seq_len, item_indices in groups.items():
            for sb_start in range(0, len(item_indices), batch_size):
                sb = item_indices[sb_start:sb_start + batch_size]
                B = len(sb)
                seq_t = torch.zeros(B, seq_len, dtype=torch.long, device=config.device)
                ctx_lens, rn_list, neg_list = [], [], []
                for bi, ii in enumerate(sb):
                    item = eval_items[ii]
                    seq_t[bi] = torch.tensor(item["seq"], dtype=torch.long)
                    ctx_lens.append(item["ctx_len"])
                    rn_list.append(item["rel_name"])
                    neg_list.append(item["negate"])

                pad_mask = (seq_t != 0)
                if model_type_str == "I":
                    x = model.token_embedding(seq_t)
                    for l, block in enumerate(model.blocks):
                        ra = model.angle_projectors[l](x)
                        angles = model._cumsum_angles_kg(ra, ctx_lens, rn_list,
                                                         config.device, negate_angles=neg_list)
                        x = block(x, angles, causal=True, pad_mask=pad_mask)
                else:
                    x = model.token_embedding(seq_t)
                    if hasattr(model, 'expander'):
                        x = model.expander(x)
                    if model_type_str == "H":
                        angles = model._cumsum_angles_kg(seq_t.shape[1], ctx_lens, rn_list,
                                                          config.device, negate_angles=neg_list)
                    else:
                        angles = model._cumsum_angles_kg(seq_t, ctx_lens, rn_list,
                                                          config.device, negate_angles=neg_list)
                    for block in model.blocks:
                        x = block(x, angles, causal=True, pad_mask=pad_mask)
                logits = model.lm_head(x)

                for bi, ii in enumerate(sb):
                    item = eval_items[ii]
                    total_lp = 0.0
                    all_top5 = True
                    all_top1 = True
                    for j in range(item["pred_len"]):
                        pred_pos = item["ctx_len"] - 1 + j
                        true_tok = item["pred_tokens"][j]
                        sl = logits[bi, pred_pos, :]
                        lps = F.log_softmax(sl, dim=0)
                        total_lp += lps[true_tok].item()
                        top5 = torch.topk(sl, k=min(5, sl.shape[0])).indices.tolist()
                        if true_tok not in top5:
                            all_top5 = False
                        if torch.argmax(sl).item() != true_tok:
                            all_top1 = False
                    ppl = np.exp(-total_lp / max(item["pred_len"], 1))
                    results[item["tier"]][item["relation"]].append({
                        "hit1": 1 if all_top1 else 0, "hit5": 1 if all_top5 else 0,
                        "ppl": ppl, "first_char_ppl": ppl, "last_char_ppl": ppl,
                        "head": item["head"], "rel": item["rel_name"], "tail": item["tail"],
                    })

    summary = {}
    for tier in results:
        tr = {"hit1": [], "hit5": [], "ppl": []}
        for rel in results[tier]:
            for r in results[tier][rel]:
                for k in tr:
                    tr[k].append(r[k])
        summary[tier] = {
            "hit1": np.mean(tr["hit1"]), "hit5": np.mean(tr["hit5"]),
            "ppl": np.exp(np.mean(np.log(tr["ppl"]))) if tr["ppl"] else 1.0,
            "n": len(tr["hit1"]),
        }

    relation_summary = {}
    for tier in results:
        relation_summary[tier] = {}
        for rel in results[tier]:
            rd = results[tier][rel]
            if rd:
                relation_summary[tier][rel] = {
                    "hit1": np.mean([r["hit1"] for r in rd]),
                    "hit5": np.mean([r["hit5"] for r in rd]),
                    "ppl": np.exp(np.mean([np.log(r["ppl"]) for r in rd])),
                    "n": len(rd),
                }

    print(f"\n{'='*60}")
    print(f"  KG Evaluation (causal): {model_name}")
    print(f"{'='*60}")
    for tier in ALL_TIERS:
        if tier in summary:
            s = summary[tier]
            print(f"  {tier:>35s}: h@1={s['hit1']:.3f}  h@5={s['hit5']:.3f}  "
                  f"ppl={s['ppl']:.2f}  (n={s['n']})")

    model.train()
    return summary, relation_summary, results


# ============================================================================
# Training
# ============================================================================

def train_model_text_only(model, text_dataset, config, name="?",
                          resume_optimizer_state=None):
    """Train text-only model (B/C) with next-token prediction."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
    model.to(config.device)
    model.train()
    losses_log = {"text": [], "iter": []}
    for it in tqdm(range(config.max_iters), desc=f"Model {name}"):
        x, y = text_dataset.get_batch(config.batch_size, config.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if it % config.eval_interval == 0:
            losses_log["iter"].append(it)
            losses_log["text"].append(loss.item())
            print(f"  [{name}] iter {it}, loss: {loss.item():.4f}")
    return losses_log, optimizer.state_dict()


def train_model_mixed(model, text_dataset, kg_dataset, config, name="?",
                      kg_batch_fn="native", resume_optimizer_state=None,
                      kg_only=False, causal_kg=False):
    """Train mixed text+KG model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
    model.to(config.device)
    model.train()
    losses_log = {"text": [], "kg": [], "iter": []}
    for it in tqdm(range(config.max_iters), desc=f"Model {name}"):
        if not kg_only:
            x, y = text_dataset.get_batch(config.batch_size, config.device)
            if hasattr(model, 'forward_text'):
                _, text_loss = model.forward_text(x, y)
            else:
                _, text_loss = model(x, y)
        if kg_batch_fn == "slotted":
            tokens, targets, head_lens, rel_names = kg_dataset.get_mlm_batch_slotted(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, kg_loss = model.forward_kg(tokens, targets, head_lens, rel_names)
        elif kg_batch_fn == "native" and causal_kg:
            ct, tgt, hl, rn, neg = kg_dataset.get_causal_batch_native(config.batch_size, config.device)
            _, kg_loss = model.forward_kg_causal(ct, tgt, hl, rn, neg)
        elif kg_batch_fn == "native":
            ct, tgt, hl, rn, neg = kg_dataset.get_mlm_batch_native(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, kg_loss = model.forward_kg(ct, tgt, hl, rn, neg)
        elif kg_batch_fn == "flat":
            tokens, targets, rel_names = kg_dataset.get_mlm_batch_flat(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, kg_loss = model.forward_kg(tokens, targets)
        if kg_only:
            loss = kg_loss
            text_loss = torch.tensor(0.0)
        else:
            loss = text_loss + kg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if it % config.eval_interval == 0:
            losses_log["iter"].append(it)
            losses_log["text"].append(text_loss.item())
            losses_log["kg"].append(kg_loss.item())
            print(f"  [{name}] iter {it}, text: {text_loss.item():.4f}, kg: {kg_loss.item():.4f}")
    return losses_log, optimizer.state_dict()


# ============================================================================
# Model Factory
# ============================================================================

MODEL_NAMES = ["A", "A'", "B", "B'", "C", "C'", "D", "D'", "E", "E'",
               "F", "F'", "G", "G'", "H", "H'", "I", "I'"]

LINEARIZED_MODELS = {"B", "B'", "C", "C'"}
SLOTTED_KG_MODELS = {"A", "A'", "G", "G'"}
NATIVE_KG_MODELS = {"E", "E'", "H", "H'", "I", "I'"}
FLAT_KG_MODELS = {"D", "D'", "F", "F'"}


def create_model(name, vocab_size, config, n_relations=8):
    """Create a model by name."""
    n_e = config.n_embed
    n_l = config.n_layers
    bs = config.block_size
    d = config.dropout

    if name == "A":    return ModelA(vocab_size, n_e, n_l, bs, d, rotate_v=False)
    elif name == "A'": return ModelA(vocab_size, n_e, n_l, bs, d, rotate_v=True)
    elif name == "B":  return ModelB(vocab_size, n_e, n_l, bs, d, rotate_v=False)
    elif name == "B'": return ModelB(vocab_size, n_e, n_l, bs, d, rotate_v=True)
    elif name == "C":  return ModelC(vocab_size, n_e, n_l, bs, d, rotate_v=False)
    elif name == "C'": return ModelC(vocab_size, n_e, n_l, bs, d, rotate_v=True)
    elif name == "D":  return ModelD(vocab_size, n_e, n_l, bs, d, rotate_v=False)
    elif name == "D'": return ModelD(vocab_size, n_e, n_l, bs, d, rotate_v=True)
    elif name == "E":  return ModelE(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=False)
    elif name == "E'": return ModelE(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=True)
    elif name == "F":  return ModelF(vocab_size, n_e, n_l, bs, d, rotate_v=False)
    elif name == "F'": return ModelF(vocab_size, n_e, n_l, bs, d, rotate_v=True)
    elif name == "G":  return ModelG(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=False)
    elif name == "G'": return ModelG(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=True)
    elif name == "H":  return ModelH(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=False)
    elif name == "H'": return ModelH(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=True)
    elif name == "I":  return ModelI(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=False)
    elif name == "I'": return ModelI(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=True)
    else: raise ValueError(f"Unknown model: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(wiki_path, wiki_lines, vocab_size, seed=42, models_to_run=None,
                   checkpoint_dir=None, load_checkpoints=False, resume_training=False,
                   kg_only=False, causal_kg=False, inverse_kg=False, kg_as_text=False,
                   wordnet_path=None, framenet_path=None, bats_dir=None,
                   google_analogies_path=None, word_analogies_path=None):
    """Run one experiment with a given seed."""
    if models_to_run is None:
        models_to_run = MODEL_NAMES

    print(f"\n{'#'*70}")
    print(f"# Experiment 8, seed={seed}")
    print(f"{'#'*70}\n")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    (vocab, text_base, text_linearized, kg_dataset,
     eval_prompts, kg_eval_prompts, linearized_eval_prompts,
     kg_relations_list) = prepare_data(
        wiki_path, wiki_lines, vocab_size,
        wordnet_path=wordnet_path, framenet_path=framenet_path,
        bats_dir=bats_dir, google_analogies_path=google_analogies_path,
        word_analogies_path=word_analogies_path,
        inverse_kg=inverse_kg, kg_as_text=kg_as_text, seed=seed)

    n_relations = len(kg_relations_list)
    rel_to_idx = {r: i for i, r in enumerate(kg_relations_list)}

    print(f"Vocabulary size: {vocab.size}")
    print(f"Text dataset (base): {len(text_base.data)} tokens")
    print(f"Text dataset (linearized): {len(text_linearized.data)} tokens")
    print(f"KG triples: {len(kg_dataset.triples)}")
    print(f"KG relations: {n_relations}")
    print(f"Eval prompts (text): {len(eval_prompts)}")
    print(f"Eval prompts (KG): {len(kg_eval_prompts)}")
    if linearized_eval_prompts:
        print(f"Eval prompts (linearized KG): {len(linearized_eval_prompts)}")

    # Print tier sizes
    tier_counts = defaultdict(int)
    for p in eval_prompts:
        tier_counts[p["tier"]] += 1
    for tier in ALL_TIERS:
        print(f"  {tier}: {tier_counts.get(tier, 0)} prompts")

    results = {}
    relation_results = {}
    kg_results = {}
    kg_relation_results = {}
    linearized_results = {}
    linearized_relation_results = {}

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for name in models_to_run:
        model = create_model(name, vocab.size, cfg, n_relations=n_relations)
        model.vocab = vocab

        # Set rel_to_idx on models that need it
        if hasattr(model, 'rel_to_idx'):
            model.rel_to_idx = rel_to_idx

        print(f"\nModel {name}: {count_parameters(model):,} params")

        safe_name = name.replace("'", "p")
        ckpt_path = os.path.join(checkpoint_dir, f"exp8_{safe_name}_seed{seed}.pt") if checkpoint_dir else None

        loaded = False
        resume_opt = None
        iters_done = 0
        if (load_checkpoints or resume_training) and ckpt_path and os.path.exists(ckpt_path):
            print(f"--- Loading Model {name} from {ckpt_path} ---")
            ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(cfg.device)
            iters_done = ckpt.get("iters_done", 0)
            if resume_training:
                resume_opt = ckpt.get("optimizer_state_dict", None)
                remaining = cfg.max_iters - iters_done
                if remaining <= 0:
                    print(f"  Already trained {iters_done} iters, skipping")
                    loaded = True
                else:
                    print(f"  Resuming from iter {iters_done}, {remaining} more")
            else:
                loaded = True

        if not loaded:
            remaining = cfg.max_iters - iters_done
            orig_max_iters = cfg.max_iters
            cfg.max_iters = remaining
            print(f"--- Training Model {name} ---")
            opt_state = None
            if name in LINEARIZED_MODELS:
                _, opt_state = train_model_text_only(model, text_linearized, cfg, name=name,
                                                     resume_optimizer_state=resume_opt)
            elif name in SLOTTED_KG_MODELS:
                _, opt_state = train_model_mixed(model, text_base, kg_dataset, cfg,
                                                  name=name, kg_batch_fn="slotted",
                                                  resume_optimizer_state=resume_opt, kg_only=kg_only)
            elif name in NATIVE_KG_MODELS:
                _, opt_state = train_model_mixed(model, text_base, kg_dataset, cfg,
                                                  name=name, kg_batch_fn="native",
                                                  resume_optimizer_state=resume_opt,
                                                  kg_only=kg_only, causal_kg=causal_kg)
            elif name in FLAT_KG_MODELS:
                _, opt_state = train_model_mixed(model, text_base, kg_dataset, cfg,
                                                  name=name, kg_batch_fn="flat",
                                                  resume_optimizer_state=resume_opt, kg_only=kg_only)
            cfg.max_iters = orig_max_iters
            total_iters = iters_done + remaining

            if ckpt_path:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt_state,
                    "iters_done": total_iters,
                    "model_name": name, "seed": seed,
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

        summary, rel_summary, _ = evaluate_model(model, eval_prompts, vocab, cfg, name)
        results[name] = summary
        relation_results[name] = rel_summary

        if kg_as_text and linearized_eval_prompts and name in LINEARIZED_MODELS:
            ls, lrs, _ = evaluate_model(model, linearized_eval_prompts, vocab, cfg, name)
            linearized_results[name] = ls
            linearized_relation_results[name] = lrs

        base_name = name.replace("'", "")
        if not kg_as_text and base_name in ("A", "D", "E", "F", "G", "H", "I"):
            if causal_kg and base_name in ("E", "H", "I"):
                ks, krs, _ = evaluate_model_kg_causal(model, kg_eval_prompts, vocab, cfg, model_name=name)
            else:
                ks, krs, _ = evaluate_model_kg(model, kg_eval_prompts, vocab, cfg,
                                                model_name=name, model_type=base_name)
            kg_results[name] = ks
            kg_relation_results[name] = krs

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (results, relation_results, kg_results, kg_relation_results,
            linearized_results, linearized_relation_results)


# ============================================================================
# Comparison Tables
# ============================================================================

def print_comparison_table(results, exp_name, models):
    print(f"\n{'='*140}")
    print(f"  COMPARISON TABLE -- {exp_name}")
    print(f"{'='*140}")
    header = f"{'Tier':<35} {'Metric':<12} "
    for m in models:
        header += f"{m:<12}"
    print(header)
    print(f"{'-'*140}")
    for tier in ALL_TIERS:
        for metric in ["hit1", "hit5", "full_correct", "ppl"]:
            line = f"{tier:<35} {metric:<12} "
            for m in models:
                if m in results and tier in results[m]:
                    v = results[m][tier].get(metric, float('nan'))
                else:
                    v = float('nan')
                line += f"{v:<12.2f}" if metric == "ppl" else f"{v:<12.3f}"
            print(line)
        print()


def print_kg_comparison_table(kg_results, exp_name, models):
    kg_models = [m for m in models if m.replace("'", "") in ("A", "D", "E", "F", "G", "H", "I")]
    if not kg_models:
        return
    print(f"\n{'='*120}")
    print(f"  KG EVALUATION TABLE -- {exp_name}")
    print(f"{'='*120}")
    header = f"{'Tier':<35} {'Metric':<8} "
    for m in kg_models:
        header += f"{m:<12}"
    print(header)
    print(f"{'-'*120}")
    for tier in ALL_TIERS:
        for metric in ["hit1", "hit5", "ppl"]:
            line = f"{tier:<35} {metric:<8} "
            for m in kg_models:
                if m in kg_results and tier in kg_results[m]:
                    v = kg_results[m][tier].get(metric, float('nan'))
                else:
                    v = float('nan')
                line += f"{v:<12.2f}" if metric == "ppl" else f"{v:<12.3f}"
            print(line)
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 8: Word-Level KG+Text")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--n_embed", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--load_checkpoints", action="store_true")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--kg_only", action="store_true")
    parser.add_argument("--causal_kg", action="store_true")
    parser.add_argument("--inverse_kg", action="store_true")
    parser.add_argument("--kg_as_text", action="store_true")
    # Data paths
    parser.add_argument("--wiki_path", default=os.path.expanduser(
        "~/AWS/fastText/fastText/wiki.en.txt"))
    parser.add_argument("--wiki_lines", type=int, default=1000000)
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--wordnet_path", default=os.path.expanduser(
        "~/AWS/python-scripts-to-experiment-with-fasttext-models/retrofitting/lexicons/wordnet-synonyms.txt"))
    parser.add_argument("--framenet_path", default=os.path.expanduser(
        "~/AWS/python-scripts-to-experiment-with-fasttext-models/retrofitting/lexicons/framenet.txt"))
    parser.add_argument("--bats_dir", default=os.path.expanduser(
        "~/AWS/measuring-regularities-in-word-embeddings/BATS_3.0"))
    parser.add_argument("--google_analogies", default=os.path.expanduser(
        "~/AWS/expRNN/expRNN/data/questions-words_for_training.txt"))
    parser.add_argument("--word_analogies", default=os.path.expanduser(
        "~/AWS/expRNN/expRNN/data/wordanalogies.txt"))
    parser.add_argument("--exp", default="8a")
    args = parser.parse_args()

    if args.n_embed is not None:
        cfg.n_embed = args.n_embed
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers

    if args.smoke:
        cfg.max_iters = 50
        cfg.n_seeds = 1
        cfg.eval_interval = 25
        args.wiki_lines = min(args.wiki_lines, 1000)
        args.vocab_size = min(args.vocab_size, 4000)
        models = args.models or MODEL_NAMES
    else:
        if args.iters is not None:
            cfg.max_iters = args.iters
        if args.seeds is not None:
            cfg.n_seeds = args.seeds
        models = args.models or MODEL_NAMES

    # Normalize model names
    normalized = []
    for m in models:
        if m.endswith("p") and len(m) == 2 and m[0] in "ABCDEFGHI":
            normalized.append(m[0] + "'")
        else:
            normalized.append(m)
    models = normalized

    print("=" * 70)
    print("  Exp 8: Word-Level KG+Text on Real Data")
    print("=" * 70)
    print(f"\nConfig: n_embed={cfg.n_embed}, n_layers={cfg.n_layers}, "
          f"max_iters={cfg.max_iters}, batch_size={cfg.batch_size}, "
          f"lr={cfg.lr}, device={cfg.device}")
    print(f"Wiki lines: {args.wiki_lines}, Vocab size: {args.vocab_size}")
    print(f"Models: {models}")

    all_results = {}
    seed_results_list = []
    seed_kg_results_list = []
    seed_lin_results_list = []

    for seed in range(cfg.n_seeds):
        (res, rel_res, kg_res, kg_rel_res,
         lin_res, lin_rel_res) = run_experiment(
            wiki_path=args.wiki_path, wiki_lines=args.wiki_lines,
            vocab_size=args.vocab_size, seed=seed, models_to_run=models,
            checkpoint_dir=args.checkpoint_dir,
            load_checkpoints=args.load_checkpoints,
            resume_training=args.resume_training,
            kg_only=args.kg_only, causal_kg=args.causal_kg,
            inverse_kg=args.inverse_kg, kg_as_text=args.kg_as_text,
            wordnet_path=args.wordnet_path, framenet_path=args.framenet_path,
            bats_dir=args.bats_dir, google_analogies_path=args.google_analogies,
            word_analogies_path=args.word_analogies)
        seed_results_list.append(res)
        seed_kg_results_list.append(kg_res)
        seed_lin_results_list.append(lin_res)

    # Average across seeds
    avg_results = {}
    for model_name in models:
        avg_results[model_name] = {}
        for tier in ALL_TIERS:
            metrics = {}
            for metric in ["hit1", "hit5", "full_correct", "ppl"]:
                vals = [sr[model_name][tier][metric]
                        for sr in seed_results_list
                        if model_name in sr and tier in sr[model_name]
                           and metric in sr[model_name][tier]]
                if vals:
                    metrics[metric] = np.mean(vals)
                    metrics[f"{metric}_std"] = np.std(vals)
            avg_results[model_name][tier] = metrics

    avg_kg_results = {}
    kg_models = [m for m in models if m.replace("'", "") in ("A", "D", "E", "F", "G", "H", "I")]
    for model_name in kg_models:
        avg_kg_results[model_name] = {}
        for tier in ALL_TIERS:
            metrics = {}
            for metric in ["hit1", "hit5", "ppl"]:
                vals = [skr[model_name][tier][metric]
                        for skr in seed_kg_results_list
                        if model_name in skr and tier in skr[model_name]
                           and metric in skr[model_name][tier]]
                if vals:
                    metrics[metric] = np.mean(vals)
                    metrics[f"{metric}_std"] = np.std(vals)
            avg_kg_results[model_name][tier] = metrics

    exp_label = "Exp 8 (Word-Level)"
    print_comparison_table(avg_results, exp_label, models)
    print_kg_comparison_table(avg_kg_results, exp_label, models)

    # Save results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"exp8_results_{timestamp}.json")

    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        return obj

    save_data = {
        "config": {
            "n_embed": cfg.n_embed, "n_layers": cfg.n_layers,
            "max_iters": cfg.max_iters, "batch_size": cfg.batch_size,
            "lr": cfg.lr, "n_seeds": cfg.n_seeds, "models": models,
        },
        "results": to_serializable({"text": avg_results, "kg": avg_kg_results}),
        "timestamp": timestamp,
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    latest_file = os.path.join(results_dir, "exp8_results_latest.json")
    with open(latest_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Latest results: {latest_file}")

    print("\n\nDone.")
    return all_results


if __name__ == "__main__":
    results = main()
