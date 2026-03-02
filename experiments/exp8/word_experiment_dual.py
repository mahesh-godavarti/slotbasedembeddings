# -----------------------------------------------------------------------------
# Exp 8: Word-Level KG+Text on Real Data (Dual-Objective + Model J/J')
#
# Same architectural comparisons as Exp 7 (Models A-J with prime variants),
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
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
    use_softmax = False   # Use softmax attention instead of log(exp(x)+1)
    dual_objective = False # Enable dual-objective: MLM+causal on both text and KG

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

    def save_tokenizer(self, path):
        """Save trained BPE tokenizer to disk."""
        self.tokenizer.save(path)
        print(f"  Tokenizer saved to {path}")

    def load_tokenizer(self, path):
        """Load a pre-trained BPE tokenizer from disk."""
        self.tokenizer = Tokenizer.from_file(path)
        self.size = self.tokenizer.get_vocab_size()

        vocab_map = self.tokenizer.get_vocab()
        self.PAD = vocab_map["<PAD>"]
        self.BOS = vocab_map["<BOS>"]
        self.EOS = vocab_map["<EOS>"]
        self.MASK = vocab_map["<MASK>"]
        self.UNK = vocab_map["<UNK>"]

        for token_str in self._relation_tokens:
            rel_name = token_str[1:-1]
            if token_str in vocab_map:
                self.kg_relations[rel_name] = vocab_map[token_str]
                self.word2idx[token_str] = vocab_map[token_str]

        print(f"  Tokenizer loaded from {path} (vocab size: {self.size})")

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

    def get_mlm_batch(self, batch_size, device, mask_prob=0.15):
        """Get a bidirectional MLM batch from text data.
        Randomly masks tokens and predicts them with bidirectional attention.

        Returns:
            tokens: (B, T) token ids with some positions replaced by MASK
            targets: (B, T) original token ids at masked positions, -100 elsewhere
        """
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        max_len = max(len(s) for s in batch)
        tokens = torch.full((batch_size, max_len), self.vocab.PAD, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, seq in enumerate(batch):
            seq_len = len(seq)
            tokens[i, :seq_len] = seq

            mask = torch.rand(seq_len) < mask_prob
            # Don't mask BOS (position 0)
            mask[0] = False
            if mask.sum() == 0 and seq_len > 1:
                mask[torch.randint(1, seq_len, (1,))] = True

            for j in range(seq_len):
                if mask[j]:
                    targets[i, j] = seq[j]
                    tokens[i, j] = self.vocab.MASK

        return tokens.to(device), targets.to(device)


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

    def get_slot_causal_batch_slotted(self, batch_size, device):
        """Slot-causal batch for A/G: mask one slot, predict it causally.

        Rearranges sequence: [unmasked_slots...] [masked_slot]
        so causal prediction within the masked slot works naturally.

        Returns:
            tokens, targets, slot_assignments, slot_positions, rel_names, context_lens
        """
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        seqs = []
        all_slot_assignments = []
        all_slot_positions = []
        context_lens = []
        for b in batch:
            head = b["head"]
            rel = [b["rel_token"]]
            tail = b["tail"]
            slots = [(head, 0), (rel, 1), (tail, 2)]
            masked_slot_idx = random.randint(0, 2)
            unmasked = [s for i, s in enumerate(slots) if i != masked_slot_idx]
            masked = slots[masked_slot_idx]
            seq = []
            sa = []
            sp = []
            for toks, sid in unmasked:
                for j, t in enumerate(toks):
                    seq.append(t)
                    sa.append(sid)
                    sp.append(j)
            ctx_len = len(seq)
            context_lens.append(ctx_len)
            toks, sid = masked
            for j, t in enumerate(toks):
                seq.append(t)
                sa.append(sid)
                sp.append(j)
            seqs.append(seq)
            all_slot_assignments.append(sa)
            all_slot_positions.append(sp)

        max_len = max(len(s) for s in seqs)

        tokens = torch.full((batch_size, max_len), self.vocab.PAD, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)
        slot_assignments = torch.zeros(batch_size, max_len, dtype=torch.long)
        slot_positions = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            tokens[i, :len(seq)] = seq_t
            slot_assignments[i, :len(seq)] = torch.tensor(all_slot_assignments[i], dtype=torch.long)
            slot_positions[i, :len(seq)] = torch.tensor(all_slot_positions[i], dtype=torch.long)
            C = context_lens[i]
            if C > 0 and C < len(seq):
                targets[i, C - 1:len(seq) - 1] = seq_t[C:]

        rel_names = [batch[i]["rel"] for i in range(batch_size)]
        return (tokens.to(device), targets.to(device),
                slot_assignments.to(device), slot_positions.to(device),
                rel_names, context_lens)

    def get_slot_causal_batch_flat(self, batch_size, device):
        """Slot-causal batch for D/F: mask one slot, predict it causally.

        Returns:
            tokens, targets, rel_names, context_lens
        """
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        seqs = []
        context_lens = []
        for b in batch:
            head = b["head"]
            rel = [b["rel_token"]]
            tail = b["tail"]
            slots = [head, rel, tail]
            masked_slot_idx = random.randint(0, 2)
            unmasked = [s for i, s in enumerate(slots) if i != masked_slot_idx]
            masked = slots[masked_slot_idx]
            seq = []
            for toks in unmasked:
                seq.extend(toks)
            ctx_len = len(seq)
            context_lens.append(ctx_len)
            seq.extend(masked)
            seqs.append(seq)

        max_len = max(len(s) for s in seqs)

        tokens = torch.full((batch_size, max_len), self.vocab.PAD, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            tokens[i, :len(seq)] = seq_t
            C = context_lens[i]
            if C > 0 and C < len(seq):
                targets[i, C - 1:len(seq) - 1] = seq_t[C:]

        rel_names = [batch[i]["rel"] for i in range(batch_size)]
        return tokens.to(device), targets.to(device), rel_names, context_lens

    def get_slot_causal_batch_native(self, batch_size, device):
        """Slot-causal batch for E/H/I: mask HEAD or TAIL, predict causally.

        Returns:
            char_tokens, targets, head_lens, rel_names, negate_angles, context_lens
        """
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        seqs = []
        head_lens = []
        negate_angles = []
        context_lens = []
        for b in batch:
            mask_head = random.random() < 0.5
            if mask_head:
                seq = b["tail"] + b["head"]
                head_lens.append(len(b["tail"]))
                negate_angles.append(True)
                context_lens.append(len(b["tail"]))
            else:
                seq = b["head"] + b["tail"]
                head_lens.append(len(b["head"]))
                negate_angles.append(False)
                context_lens.append(len(b["head"]))
            seqs.append(seq)

        max_len = max(len(s) for s in seqs)

        char_tokens = torch.full((batch_size, max_len), self.vocab.PAD, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            char_tokens[i, :len(seq)] = seq_t
            C = context_lens[i]
            if C > 0 and C < len(seq):
                targets[i, C - 1:len(seq) - 1] = seq_t[C:]

        rel_names = [batch[i]["rel"] for i in range(batch_size)]
        return (char_tokens.to(device), targets.to(device),
                head_lens, rel_names, negate_angles, context_lens)

    def get_mlm_batch_native_slots(self, batch_size, device, mask_prob=0.15):
        """Get a batch for native-slot KG models (J/J'): no relation token.

        Returns:
            char_tokens, targets, head_lens, rel_names
        """
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
        return char_tokens.to(device), targets.to(device), head_lens, rel_names

    def get_slot_causal_batch_native_slots(self, batch_size, device):
        """Slot-causal batch for J/J': mask HEAD or TAIL, predict causally.

        Returns:
            tokens, targets, slot_assignments, slot_positions, rel_names, context_lens
        """
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        seqs = []
        all_slot_assignments = []
        all_slot_positions = []
        context_lens = []
        for b in batch:
            head = b["head"]
            tail = b["tail"]
            slots = [(head, 0), (tail, 1)]
            masked_slot_idx = random.randint(0, 1)
            unmasked = [s for i, s in enumerate(slots) if i != masked_slot_idx]
            masked = slots[masked_slot_idx]
            seq = []
            sa = []
            sp = []
            for toks, sid in unmasked:
                for j, t in enumerate(toks):
                    seq.append(t)
                    sa.append(sid)
                    sp.append(j)
            ctx_len = len(seq)
            context_lens.append(ctx_len)
            toks, sid = masked
            for j, t in enumerate(toks):
                seq.append(t)
                sa.append(sid)
                sp.append(j)
            seqs.append(seq)
            all_slot_assignments.append(sa)
            all_slot_positions.append(sp)

        max_len = max(len(s) for s in seqs)

        tokens = torch.full((batch_size, max_len), self.vocab.PAD, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)
        slot_assignments = torch.zeros(batch_size, max_len, dtype=torch.long)
        slot_positions = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq_t = torch.tensor(seq, dtype=torch.long)
            tokens[i, :len(seq)] = seq_t
            slot_assignments[i, :len(seq)] = torch.tensor(all_slot_assignments[i], dtype=torch.long)
            slot_positions[i, :len(seq)] = torch.tensor(all_slot_positions[i], dtype=torch.long)
            C = context_lens[i]
            if C > 0 and C < len(seq):
                targets[i, C - 1:len(seq) - 1] = seq_t[C:]

        rel_names = [batch[i]["rel"] for i in range(batch_size)]
        return (tokens.to(device), targets.to(device),
                slot_assignments.to(device), slot_positions.to(device),
                rel_names, context_lens)


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

    def __init__(self, n_embed, block_size, dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.keys = nn.Linear(n_embed, n_embed)
        self.queries = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.rotate_v = rotate_v
        self.use_softmax = use_softmax
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, angles, causal=True, pad_mask=None, attn_mask=None):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        k = apply_rotation(k, angles)
        q = apply_rotation(q, angles)

        if self.rotate_v:
            v = apply_rotation(v, angles)

        wei = k @ q.transpose(-1, -2) * C ** (-0.5)

        if attn_mask is not None:
            # attn_mask: (B, T, T) bool, True = can attend
            wei = wei.masked_fill(~attn_mask, float('-inf'))
        elif causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        if pad_mask is not None:
            pad_mask_k = pad_mask.unsqueeze(1)
            wei = wei.masked_fill(~pad_mask_k, float('-inf'))

        if self.use_softmax:
            wei = F.softmax(wei, dim=-1)
        else:
            wei = torch.log(torch.exp(wei) + 1)
            wei = wei / (wei.sum(dim=-1, keepdim=True) + 1e-6)
        wei = self.dropout(wei)
        out = wei @ v

        if self.rotate_v:
            out = apply_inverse_rotation(out, angles)

        out = self.proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, n_embed, block_size, dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.sa_head = RotaryAttention(n_embed, block_size, dropout, rotate_v, use_softmax)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, angles, causal=True, pad_mask=None, attn_mask=None):
        x = x + self.sa_head(self.ln1(x), angles, causal, pad_mask, attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


def build_slot_causal_mask(batch_size, seq_len, context_lens, device):
    """Build attention mask for slot-causal training.

    For each sample:
      - Context positions (0..context_len-1): bidirectional among themselves,
        CANNOT see masked slot positions.
      - Masked positions (context_len..seq_len-1): can see all context positions
        + causal within the masked slot.
    """
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        C = context_lens[i]
        # Context rows: bidirectional within context only
        mask[i, :C, :C] = True
        # Masked rows: see all context + causal within masked slot
        for j in range(C, seq_len):
            mask[i, j, :C] = True       # all context
            mask[i, j, C:j + 1] = True  # causal within masked slot
    return mask


# ============================================================================
# Model A/A': RoPE + Slot Angles, Native KG
# ============================================================================

class ModelA(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.slot_angles = nn.Parameter(torch.randn(3, n_embed // 2) * 0.1)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
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

    def _kg_angles_slotcausal(self, slot_assignments, slot_positions, rel_names, device):
        """KG angles for rearranged slot-causal sequences (shared slot angles)."""
        B, T = slot_assignments.shape
        angles = torch.zeros(B, T, self.n_embed // 2, device=device)
        for i in range(B):
            for j in range(T):
                sid = slot_assignments[i, j].item()
                pos = slot_positions[i, j].item()
                angles[i, j] = pos * self.base_freq + self.slot_angles[sid]
        return angles

    def forward_text_mlm(self, tokens, targets):
        """Text mode: bidirectional attention, MLM on randomly masked tokens."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._rope_angles(T, tokens.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, tokens, targets, slot_assignments, slot_positions,
                          rel_names, context_lens):
        """KG slot-causal: mask one slot, predict it with causal attention."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._kg_angles_slotcausal(slot_assignments, slot_positions,
                                            rel_names, tokens.device)
        attn_mask = build_slot_causal_mask(B, T, context_lens, tokens.device)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask, attn_mask=attn_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model B/B': Standard RoPE, Linearized KG-as-Text
# ============================================================================

class ModelB(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
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

    def forward_text_mlm(self, tokens, targets):
        """Text mode: bidirectional attention, MLM on randomly masked tokens."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._rope_angles(T, tokens.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits


# ============================================================================
# Model C/C': Per-Token Cumsum (Journey), Linearized KG-as-Text
# ============================================================================

class ModelC(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
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

    def forward_text_mlm(self, tokens, targets):
        """Text mode: bidirectional attention, MLM on randomly masked tokens."""
        B, T = tokens.shape
        x = self.expander(self.token_embedding(tokens))
        pad_mask = (tokens != 0)
        raw_angles = self.angle_embedding(tokens)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward(idx)
        return logits


# ============================================================================
# Model D/D': Per-Token Cumsum, Flat KG (Relation as Token)
# ============================================================================

class ModelD(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
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

    def forward_text_mlm(self, tokens, targets):
        """Text mode: bidirectional attention, MLM on randomly masked tokens."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.expander(self.token_embedding(tokens))
        angles = self._cumsum_angles(tokens)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, tokens, targets, context_lens):
        """KG slot-causal: mask one slot, predict it with causal attention."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.expander(self.token_embedding(tokens))
        angles = self._cumsum_angles(tokens)
        attn_mask = build_slot_causal_mask(B, T, context_lens, tokens.device)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask, attn_mask=attn_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model E/E': Per-Token Cumsum + Relation Operator, Native KG
# ============================================================================

class ModelE(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)
        self.relation_angles = nn.Parameter(torch.randn(n_relations, n_embed // 2) * 0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
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

    def forward_kg_causal(self, char_tokens, targets, head_lens, rel_names, negate_angles,
                          context_lens=None):
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)
        x = self.expander(self.token_embedding(char_tokens))
        angles = self._cumsum_angles_kg(char_tokens, head_lens, rel_names,
                                        char_tokens.device, negate_angles=negate_angles)
        if context_lens is not None:
            attn_mask = build_slot_causal_mask(B, T, context_lens, char_tokens.device)
            for block in self.blocks:
                x = block(x, angles, causal=False, pad_mask=pad_mask, attn_mask=attn_mask)
        else:
            for block in self.blocks:
                x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_text_mlm(self, tokens, targets):
        """Text mode: bidirectional attention, MLM on randomly masked tokens."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.expander(self.token_embedding(tokens))
        angles = self._cumsum_angles_text(tokens)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model F/F': Fixed-Angle RoPE, Flat KG (Relation as Token)
# ============================================================================

class ModelF(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
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

    def forward_text_mlm(self, tokens, targets):
        """Text mode: bidirectional attention, MLM on randomly masked tokens."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._rope_angles(T, tokens.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, tokens, targets, context_lens):
        """KG slot-causal: mask one slot, predict it with causal attention."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._rope_angles(T, tokens.device).expand(B, -1, -1)
        attn_mask = build_slot_causal_mask(B, T, context_lens, tokens.device)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask, attn_mask=attn_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model G/G': Relation-Dependent Slot Angles, Native KG
# ============================================================================

class ModelG(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.slot_angles = nn.Parameter(torch.randn(n_relations, 3, n_embed // 2) * 0.1)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
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

    def _kg_angles_slotcausal(self, slot_assignments, slot_positions, rel_names, device):
        """KG angles for rearranged slot-causal sequences (per-relation)."""
        B, T = slot_assignments.shape
        angles = torch.zeros(B, T, self.n_embed // 2, device=device)
        for i in range(B):
            rel_idx = self.rel_to_idx[rel_names[i]]
            for j in range(T):
                sid = slot_assignments[i, j].item()
                pos = slot_positions[i, j].item()
                angles[i, j] = pos * self.base_freq + self.slot_angles[rel_idx, sid]
        return angles

    def forward_text_mlm(self, tokens, targets):
        """Text mode: bidirectional attention, MLM on randomly masked tokens."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._rope_angles(T, tokens.device).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, tokens, targets, slot_assignments, slot_positions,
                          rel_names, context_lens):
        """KG slot-causal: mask one slot, predict it with causal attention."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._kg_angles_slotcausal(slot_assignments, slot_positions,
                                            rel_names, tokens.device)
        attn_mask = build_slot_causal_mask(B, T, context_lens, tokens.device)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask, attn_mask=attn_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model H/H': Fixed Angles + Relation Operator, Native KG
# ============================================================================

class ModelH(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))
        self.relation_angles = nn.Parameter(torch.randn(n_relations, n_embed // 2) * 0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
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

    def forward_kg_causal(self, char_tokens, targets, head_lens, rel_names, negate_angles,
                          context_lens=None):
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)
        x = self.token_embedding(char_tokens)
        angles = self._cumsum_angles_kg(T, head_lens, rel_names, char_tokens.device,
                                        negate_angles=negate_angles)
        if context_lens is not None:
            attn_mask = build_slot_causal_mask(B, T, context_lens, char_tokens.device)
            for block in self.blocks:
                x = block(x, angles, causal=False, pad_mask=pad_mask, attn_mask=attn_mask)
        else:
            for block in self.blocks:
                x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_text_mlm(self, tokens, targets):
        """Text mode: bidirectional attention, MLM on randomly masked tokens."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._cumsum_angles_text(T, B, tokens.device)
        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model I/I': Per-Layer Angle Computation, Native KG
# ============================================================================

class ModelI(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False, use_softmax=False):
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
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
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

    def forward_kg_causal(self, char_tokens, targets, head_lens, rel_names, negate_angles,
                          context_lens=None):
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)
        x = self.token_embedding(char_tokens)
        if context_lens is not None:
            attn_mask = build_slot_causal_mask(B, T, context_lens, char_tokens.device)
            for l, block in enumerate(self.blocks):
                raw_angles = self.angle_projectors[l](x)
                angles = self._cumsum_angles_kg(raw_angles, head_lens, rel_names,
                                                char_tokens.device, negate_angles=negate_angles)
                x = block(x, angles, causal=False, pad_mask=pad_mask, attn_mask=attn_mask)
        else:
            for l, block in enumerate(self.blocks):
                raw_angles = self.angle_projectors[l](x)
                angles = self._cumsum_angles_kg(raw_angles, head_lens, rel_names,
                                                char_tokens.device, negate_angles=negate_angles)
                x = block(x, angles, causal=True, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_text_mlm(self, tokens, targets):
        """Text mode: bidirectional attention, MLM on randomly masked tokens."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        for l, block in enumerate(self.blocks):
            raw_angles = self.angle_projectors[l](x)
            angles = self._cumsum_angles_text(raw_angles)
            x = block(x, angles, causal=False, pad_mask=pad_mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model J/J': Per-Relation Slot Angles, Native KG (no REL token, 2 slots)
# ============================================================================

class ModelJ(nn.Module):
    """RoPE + per-relation named slot angles, native KG (no REL token).

    Like Model G (per-relation slot angles) but with native KG format:
    sequences are [head_tokens][tail_tokens] with NO relation token.
    Each relation gets 2 learned slot angle vectors (HEAD and TAIL).

    Text: slot angles = 0 -> pure RoPE. Causal mask, NTP.
    KG: 2 slots with per-relation learned slot angles. Positions reset per slot.
        Bidirectional attention, MLM on token positions only.
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False, use_softmax=False):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, n_embed)

        # Per-relation slot angle vectors: [n_relations, 2, n_embed//2]
        # Slot 0 = HEAD, Slot 1 = TAIL (no REL slot)
        self.slot_angles = nn.Parameter(torch.randn(n_relations, 2, n_embed // 2) * 0.1)

        # RoPE base frequencies
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v, use_softmax)
            for _ in range(n_layers)
        ])

        self.lm_head = nn.Linear(n_embed, vocab_size)

        # Mapping from relation name to index
        self.rel_to_idx = {}  # populated at runtime

    def _rope_angles(self, T, device):
        """Standard RoPE angles: position * base_freq. Shape: (1, T, C//2)."""
        positions = torch.arange(T, device=device, dtype=torch.float)
        angles = torch.outer(positions, self.base_freq)
        return angles.unsqueeze(0)

    def _kg_angles(self, head_lens, seq_len, batch_size, rel_names, device):
        """KG angles: position_in_slot * base_freq + slot_angle[rel].

        Native sequence layout: [head_tokens] [tail_tokens] (no relation token)
        """
        angles = torch.zeros(batch_size, seq_len, self.n_embed // 2, device=device)

        for i in range(batch_size):
            h_len = head_lens[i]
            rel_idx = self.rel_to_idx[rel_names[i]]

            # HEAD slot: positions 0..h_len-1
            for j in range(h_len):
                angles[i, j] = j * self.base_freq + self.slot_angles[rel_idx, 0]

            # TAIL slot: positions 0..t_len-1
            tail_start = h_len
            for j in range(seq_len - tail_start):
                if tail_start + j < seq_len:
                    angles[i, tail_start + j] = j * self.base_freq + self.slot_angles[rel_idx, 1]

        return angles

    def _kg_angles_slotcausal(self, slot_assignments, slot_positions, rel_names, device):
        """KG angles for rearranged slot-causal sequences (per-relation, 2 slots)."""
        B, T = slot_assignments.shape
        angles = torch.zeros(B, T, self.n_embed // 2, device=device)

        for i in range(B):
            rel_idx = self.rel_to_idx[rel_names[i]]
            for j in range(T):
                sid = slot_assignments[i, j].item()
                pos = slot_positions[i, j].item()
                angles[i, j] = pos * self.base_freq + self.slot_angles[rel_idx, sid]

        return angles

    def forward_text(self, idx, targets=None):
        """Text mode: pure RoPE (slot angles = 0), causal, next-token prediction."""
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._rope_angles(T, idx.device).expand(B, -1, -1)

        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)

        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg(self, char_tokens, targets, head_lens, rel_names, negate_angles=None):
        """KG mode: slot-based RoPE with per-relation angles, bidirectional, MLM.

        Native format: [head_tokens][tail_tokens], no relation token in sequence.
        negate_angles is accepted for API compatibility but ignored.
        """
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)

        x = self.token_embedding(char_tokens)
        angles = self._kg_angles(head_lens, T, B, rel_names, char_tokens.device)

        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)

        logits = self.lm_head(x)

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_text_mlm(self, tokens, targets):
        """Text mode: bidirectional attention, MLM on randomly masked tokens."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)
        x = self.token_embedding(tokens)
        angles = self._rope_angles(T, tokens.device).expand(B, -1, -1)

        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)

        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg_causal(self, tokens, targets, slot_assignments, slot_positions,
                          rel_names, context_lens):
        """KG slot-causal: mask one slot (HEAD or TAIL), predict it with causal attention."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)

        x = self.token_embedding(tokens)
        angles = self._kg_angles_slotcausal(slot_assignments, slot_positions,
                                            rel_names, tokens.device)
        attn_mask = build_slot_causal_mask(B, T, context_lens, tokens.device)

        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask, attn_mask=attn_mask)

        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def predict_text(self, idx):
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Chain-Based Tier System
# ============================================================================

ALL_TIERS = [
    "memorization", "transfer", "generalization",
    "kg_exclusive_memorization", "kg_exclusive_generalization",
    "text_exclusive_memorization", "text_exclusive_generalization",
]

# Chain relation tokens used in KG triples and templates
CHAIN_RELATION_TOKENS = [
    "<synonym_of>", "<antonym_of>",
    "<father_of>", "<son_of>", "<grandfather_of>", "<grandson_of>",
    "<capital_of>", "<language_of>", "<spoken_in>",
    "<is_a>",
]

CHAIN_RELATIONS = [
    "synonym_of", "antonym_of",
    "father_of", "son_of", "grandfather_of", "grandson_of",
    "capital_of", "language_of", "spoken_in",
    "is_a",
]


@dataclass
class Chain:
    chain_type: str            # "family", "synonym", "antonym_synonym", "capital_language", "hypernym"
    entities: tuple            # (A, B, C) — the 3 entities
    base_facts: list           # [(head, rel, tail), ...] — base KG triples
    derived_facts: list        # [(head, rel, tail), ...] — derived KG triples
    base_text_relations: list  # [(A, B, rel_name), ...] — for template generation
    derived_text_relations: list  # [(A, B, rel_name), ...] — for template generation


TEXT_TEMPLATES = {
    # --- Family ---
    "father_of": [
        "{A} is the father of {B}.",
        "{B} is the son of {A}.",
        "{A} has a son named {B}.",
        "{B}'s father is {A}.",
        "{A} is {B}'s father.",
        "{B} is {A}'s son.",
        "{A} is the parent of {B}.",
        "{B} was fathered by {A}.",
        "The father of {B} is {A}.",
        "{A} raised his son {B}.",
    ],
    "grandfather_of": [
        "{A} is the grandfather of {B}.",
        "{B} is the grandson of {A}.",
        "{A} is {B}'s grandfather.",
        "{B}'s grandfather is {A}.",
        "{A} has a grandson named {B}.",
        "{B} is the grandchild of {A}.",
        "{A} is the grandparent of {B}.",
        "The grandfather of {B} is {A}.",
        "{B} is {A}'s grandson.",
        "{A} is a grandparent to {B}.",
    ],
    # --- Synonyms ---
    "synonym_of": [
        "{A} is a synonym of {B}.",
        "{A} and {B} are synonyms.",
        "{A} and {B} mean the same thing.",
        "{A} means the same as {B}.",
        "{A} has the same meaning as {B}.",
        "{B} is another word for {A}.",
        "{A} can be used interchangeably with {B}.",
        "The word {A} is synonymous with {B}.",
        "{A} is similar in meaning to {B}.",
        "{B} and {A} share the same meaning.",
    ],
    # --- Antonyms ---
    "antonym_of": [
        "{A} is the opposite of {B}.",
        "{A} and {B} are antonyms.",
        "{A} and {B} are opposites.",
        "{A} means the opposite of {B}.",
        "{B} is the antonym of {A}.",
        "{A} is contrary to {B}.",
        "{A} and {B} have opposite meanings.",
        "The opposite of {A} is {B}.",
        "{A} contrasts with {B} in meaning.",
        "{B} is the reverse of {A} in meaning.",
    ],
    # --- Capitals ---
    "capital_of": [
        "{A} is the capital of {B}.",
        "The capital of {B} is {A}.",
        "{A} serves as the capital city of {B}.",
        "{B} has {A} as its capital.",
        "{A} is the capital city of {B}.",
        "The capital city of {B} is {A}.",
        "{A} is where the government of {B} is located.",
        "The main city of {B} is {A}.",
        "{A} functions as the capital of {B}.",
        "{B} is governed from {A}.",
    ],
    "language_of": [
        "{B} is spoken in {A}.",
        "People in {A} speak {B}.",
        "The language of {A} is {B}.",
        "{B} is the language of {A}.",
        "In {A}, the primary language is {B}.",
        "{A} is a {B}-speaking country.",
        "The official language of {A} is {B}.",
        "{B} is widely spoken in {A}.",
        "The people of {A} speak {B}.",
        "{A} uses {B} as its language.",
    ],
    "spoken_in": [
        "{B} is spoken in {A}.",
        "People in {A} speak {B}.",
        "The language spoken in {A} is {B}.",
        "{A} is a city where {B} is spoken.",
        "In {A}, people communicate in {B}.",
        "{B} is the language of {A}.",
        "Residents of {A} speak {B}.",
        "{A} is a {B}-speaking city.",
        "The main language in {A} is {B}.",
        "You can hear {B} spoken in {A}.",
    ],
    # --- Hypernyms ---
    "is_a": [
        "{A} is a type of {B}.",
        "{A} is a kind of {B}.",
        "A {A} is a {B}.",
        "{A} belongs to the category of {B}.",
        "{A} is classified as a {B}.",
        "{A} falls under {B}.",
        "{B} includes {A} as an example.",
        "{A} is an example of a {B}.",
        "The word {A} refers to a type of {B}.",
        "{A} can be described as a {B}.",
    ],
}

# ~600 common male first names (lowercase)
MALE_NAMES = [
    "james", "john", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "charles", "christopher", "daniel", "matthew",
    "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
    "kenneth", "kevin", "brian", "george", "timothy", "ronald", "edward",
    "jason", "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric",
    "jonathan", "stephen", "larry", "justin", "scott", "brandon", "benjamin",
    "samuel", "raymond", "gregory", "frank", "alexander", "patrick",
    "jack", "dennis", "jerry", "tyler", "aaron", "jose", "nathan",
    "henry", "peter", "douglas", "zachary", "adam", "kyle", "noah",
    "ethan", "jeremy", "walter", "christian", "keith", "roger", "terry",
    "austin", "sean", "gerald", "carl", "harold", "dylan", "arthur",
    "lawrence", "jordan", "jesse", "bryan", "billy", "bruce", "gabriel",
    "joe", "logan", "albert", "willie", "alan", "eugene", "elijah",
    "alfred", "russell", "wayne", "roy", "vincent", "philip", "bobby",
    "johnny", "bradley", "ralph", "eugene", "randy", "howard", "carlos",
    "russell", "louis", "harry", "glenn", "ernest", "todd", "craig",
    "steve", "alan", "shawn", "clarence", "travis", "lance", "darren",
    "ross", "marshall", "mario", "dale", "leon", "curtis", "rafael",
    "edgar", "floyd", "lloyd", "barry", "herbert", "fred", "lester",
    "clifford", "nelson", "cecil", "clifton", "daryl", "gordon", "harvey",
    "perry", "brent", "vernon", "ivan", "oscar", "dewey", "luther",
    "rodney", "duane", "kurt", "roland", "rex", "clyde", "glen",
    "hector", "karl", "marcus", "sergio", "ted", "marvin", "wade",
    "lyle", "kirk", "andy", "neil", "clarence", "darrell", "leslie",
    "fredrick", "bert", "norman", "gene", "otis", "omar", "grady",
    "doug", "benny", "irving", "homer", "roscoe", "edgar", "hubert",
    "loren", "stan", "archie", "aldo", "bud", "gus", "ernie",
    "felix", "amos", "boyd", "clem", "mack", "otis", "ward",
    "rufus", "virgil", "silas", "jasper", "calvin", "alvin", "percy",
    "angus", "boris", "cedric", "derek", "emery", "floyd", "grant",
    "hugo", "irvin", "jules", "kermit", "leland", "merle", "noel",
    "orville", "porter", "quincy", "ruben", "seldon", "trent", "vance",
    "wendell", "xavier", "yves", "abner", "barney", "cliff", "dirk",
    "elmer", "forrest", "giles", "horace", "isaiah", "jarvis", "kelvin",
    "lamar", "miles", "norbert", "olaf", "prescott", "quentin", "reginald",
    "sanford", "thurman", "ulysses", "vaughn", "wilbert", "yancy", "zane",
    "abel", "benedict", "clive", "duncan", "elton", "fletcher", "garth",
    "heath", "irwin", "jerald", "kingsley", "lionel", "merlin", "neville",
    "oswald", "philip", "rafael", "stefan", "trevor", "uriah", "victor",
    "woodrow", "aldous", "baxter", "cornelius", "dexter", "elias", "fabian",
    "griffin", "hannibal", "ignatius", "jarrett", "kendrick", "luther", "montgomery",
    "nigel", "orson", "percival", "rayburn", "sterling", "thaddeus", "ulf",
    "vincent", "waylon", "alden", "barnaby", "colton", "dalton", "emmett",
    "fergus", "graham", "holden", "isidore", "jude", "keegan", "lachlan",
    "magnus", "nolan", "otto", "pierce", "reuben", "solomon", "tobias",
    "ulric", "vernon", "wallace", "ansel", "blaine", "casper", "donovan",
    "elwood", "fritz", "gilmore", "hank", "ira", "jefferson", "knox",
    "leopold", "mitchel", "niles", "odin", "prescott", "randal", "shepard",
    "terrance", "vernon", "arlo", "bryce", "colby", "dorian", "ellis",
    "finnegan", "gideon", "hadley", "ivan", "jasper", "keaton", "landon",
    "maverick", "nico", "orion", "phoenix", "rowan", "silas", "thatcher",
    "vaughn", "wilder", "wyatt", "zeke", "alec", "brooks", "callum",
    "dashiell", "evander", "fox", "griffith", "hayes", "ike", "jensen",
    "kit", "leander", "milo", "nash", "oakley", "pascal", "rhys",
    "sawyer", "thane", "urban", "viggo", "wells", "yuri", "zander",
    "atlas", "beckett", "cruz", "dane", "eamon", "flynn", "grey",
    "harrison", "ingram", "joel", "kane", "lincoln", "marshall", "navarro",
    "ozzy", "paxton", "reign", "slate", "tatum", "vance", "weston",
    "york", "zion", "archer", "bodhi", "chase", "damon", "ezra",
    "frank", "gunner", "hendrix", "ian", "judah", "kingston", "lennox",
    "maddox", "noel", "omar", "presley", "reed", "stone", "tristan",
    "wade", "xander", "yosef", "zavier", "ashton", "blake", "cohen",
    "drew", "easton", "ford", "gage", "hugo", "ivan", "jett",
    "kade", "lars", "maxim", "neil", "orion", "phineas", "rocco",
    "santiago", "tanner", "upton", "vivian", "warren", "yael", "zain",
    "axel", "brock", "cody", "darius", "eli", "fenton", "gentry",
    "hunter", "ishmael", "jacoby", "kian", "layton", "monroe", "nigel",
    "otto", "porter", "reid", "soren", "ty", "uriel", "vito",
    "wynn", "yohan", "zander", "anton", "barrett", "cedric", "devlin",
    "enoch", "floyd", "garrett", "hamish", "inigo", "joaquin", "kellan",
    "levi", "mack", "nestor", "oscar", "padraig", "ramon", "stellan",
    "tobin", "ugo", "viggo", "winston", "yardley", "zeno",
]


def build_family_chains(n_chains=500, seed=42):
    """Generate synthetic father/grandfather chains from common names."""
    rng = random.Random(seed)
    # Deduplicate names preserving order
    seen = set()
    names = []
    for n in MALE_NAMES:
        if n not in seen:
            seen.add(n)
            names.append(n)
    rng.shuffle(names)
    chains = []
    for i in range(0, min(n_chains * 3, len(names)) - 2, 3):
        g, f, c = names[i], names[i+1], names[i+2]
        chains.append(Chain(
            chain_type="family",
            entities=(g, f, c),
            base_facts=[
                (g, "father_of", f), (f, "son_of", g),
                (f, "father_of", c), (c, "son_of", f),
            ],
            derived_facts=[
                (g, "grandfather_of", c), (c, "grandson_of", g),
            ],
            base_text_relations=[
                (g, f, "father_of"), (f, g, "father_of"),
                (f, c, "father_of"), (c, f, "father_of"),
            ],
            derived_text_relations=[
                (g, c, "grandfather_of"), (c, g, "grandfather_of"),
            ],
        ))
    return chains


def build_synonym_chains(wordnet_path, max_chains=400, seed=42):
    """Find (A, B, C) chains where A-B and B-C are synonyms but A-C are not."""
    rng = random.Random(seed)
    # Parse WordNet into adjacency map
    adj = defaultdict(set)
    with open(wordnet_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if len(words) < 2:
                continue
            # Only use single-word entries (no underscores)
            singles = [w.lower() for w in words if '_' not in w and w.isalpha()]
            for i in range(len(singles)):
                for j in range(i + 1, len(singles)):
                    adj[singles[i]].add(singles[j])
                    adj[singles[j]].add(singles[i])

    # For each word B with ≥2 synonyms, find pairs (A, C) not directly connected
    chains = []
    used_entities = set()
    candidates = [(b, syns) for b, syns in adj.items() if len(syns) >= 2]
    rng.shuffle(candidates)

    for b, syns in candidates:
        if len(chains) >= max_chains:
            break
        syn_list = sorted(syns)
        rng.shuffle(syn_list)
        found = False
        for i in range(len(syn_list)):
            if found:
                break
            a = syn_list[i]
            if a in used_entities:
                continue
            for j in range(i + 1, len(syn_list)):
                c = syn_list[j]
                if c in used_entities:
                    continue
                # A-C should NOT be direct synonyms
                if c not in adj[a]:
                    if a != b and b != c and a != c:
                        if b not in used_entities:
                            chains.append(Chain(
                                chain_type="synonym",
                                entities=(a, b, c),
                                base_facts=[
                                    (a, "synonym_of", b), (b, "synonym_of", a),
                                    (b, "synonym_of", c), (c, "synonym_of", b),
                                ],
                                derived_facts=[
                                    (a, "synonym_of", c), (c, "synonym_of", a),
                                ],
                                base_text_relations=[
                                    (a, b, "synonym_of"), (b, a, "synonym_of"),
                                    (b, c, "synonym_of"), (c, b, "synonym_of"),
                                ],
                                derived_text_relations=[
                                    (a, c, "synonym_of"), (c, a, "synonym_of"),
                                ],
                            ))
                            used_entities.update([a, b, c])
                            found = True
                            break
    return chains


def build_antonym_synonym_chains(antonym_path, wordnet_path, max_chains=150, seed=42):
    """Find (A, B, C) where antonym(A,B) and synonym(B,C) but NOT antonym(A,C)."""
    rng = random.Random(seed)

    # Parse antonym pairs from L09_comprehensive
    antonym_pairs = set()
    antonym_adj = defaultdict(set)
    with open(antonym_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                a, b = parts[0].strip().lower(), parts[1].strip().lower()
                if a.isalpha() and b.isalpha():
                    antonym_pairs.add((a, b))
                    antonym_adj[a].add(b)
                    antonym_adj[b].add(a)

    # Parse WordNet synonyms
    syn_adj = defaultdict(set)
    with open(wordnet_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if len(words) < 2:
                continue
            singles = [w.lower() for w in words if '_' not in w and w.isalpha()]
            for i in range(len(singles)):
                for j in range(i + 1, len(singles)):
                    syn_adj[singles[i]].add(singles[j])
                    syn_adj[singles[j]].add(singles[i])

    chains = []
    used_entities = set()
    # For each antonym pair (A, B), find C as synonym of B where C is NOT antonym of A
    ant_list = sorted(antonym_pairs)
    rng.shuffle(ant_list)

    for a, b in ant_list:
        if len(chains) >= max_chains:
            break
        if a in used_entities or b in used_entities:
            continue
        synonyms_of_b = sorted(syn_adj.get(b, set()))
        rng.shuffle(synonyms_of_b)
        for c in synonyms_of_b:
            if c in used_entities or c == a or c == b:
                continue
            # C should NOT be an antonym of A
            if c not in antonym_adj.get(a, set()):
                chains.append(Chain(
                    chain_type="antonym_synonym",
                    entities=(a, b, c),
                    base_facts=[
                        (a, "antonym_of", b), (b, "antonym_of", a),
                        (b, "synonym_of", c), (c, "synonym_of", b),
                    ],
                    derived_facts=[
                        (a, "antonym_of", c), (c, "antonym_of", a),
                    ],
                    base_text_relations=[
                        (a, b, "antonym_of"), (b, a, "antonym_of"),
                        (b, c, "synonym_of"), (c, b, "synonym_of"),
                    ],
                    derived_text_relations=[
                        (a, c, "antonym_of"), (c, a, "antonym_of"),
                    ],
                ))
                used_entities.update([a, b, c])
                break
    return chains


def build_capital_language_chains(e01_path, e02_path, google_path, max_chains=50, seed=42):
    """Find (city, country, language) chains from capital + language data."""
    rng = random.Random(seed)

    # Parse E01 for capital→country (file format: capital\tcountry)
    capital_to_country = {}
    with open(e01_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                capital, country = parts[0].strip().lower(), parts[1].strip().lower()
                capital_to_country[capital] = country

    # Parse Google analogies for more capital-country pairs
    if os.path.exists(google_path):
        in_capital_section = False
        with open(google_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith(':'):
                    in_capital_section = 'capital' in line.lower()
                    continue
                if in_capital_section:
                    parts = line.split()
                    if len(parts) == 4:
                        # A B C D: A is to B as C is to D (capital-country)
                        capital_to_country.setdefault(parts[0].lower(), parts[1].lower())
                        capital_to_country.setdefault(parts[2].lower(), parts[3].lower())

    # Parse E02 for country→language (file format: country\tlanguage1/language2)
    country_to_language = {}
    with open(e02_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                country = parts[0].strip().lower()
                # Take first language only
                lang = parts[1].strip().split('/')[0].strip().lower()
                country_to_language[country] = lang

    # Match on country name
    chains = []
    items = sorted(capital_to_country.items())
    rng.shuffle(items)
    used = set()
    for capital, country in items:
        if len(chains) >= max_chains:
            break
        if country in country_to_language and capital not in used:
            language = country_to_language[country]
            if capital != country and capital != language and country != language:
                chains.append(Chain(
                    chain_type="capital_language",
                    entities=(capital, country, language),
                    base_facts=[
                        (capital, "capital_of", country),
                        (country, "language_of", language),
                    ],
                    derived_facts=[
                        (capital, "spoken_in", language),
                    ],
                    base_text_relations=[
                        (capital, country, "capital_of"),
                        (country, language, "language_of"),
                    ],
                    derived_text_relations=[
                        (capital, language, "spoken_in"),
                    ],
                ))
                used.add(capital)
    return chains


def build_hypernym_chains(l01_path, l02_path, max_chains=100, seed=42):
    """Build is_a chains from BATS hypernym data."""
    rng = random.Random(seed)

    entries = []  # list of (word, [hyp1, hyp2, ...])
    for path in [l01_path, l02_path]:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word = parts[0].strip().lower()
                    hyps = [h.strip().lower() for h in parts[1].split('/')
                            if h.strip() and h.strip().lower() != word]
                    if len(hyps) >= 2:
                        entries.append((word, hyps))

    chains = []
    rng.shuffle(entries)
    used = set()
    for word, hyps in entries:
        if len(chains) >= max_chains:
            break
        if word in used:
            continue
        # Take first hypernym as intermediate, second as general
        hyp1, hyp2 = hyps[0], hyps[1]
        if hyp1 == hyp2 or word == hyp1 or word == hyp2:
            continue
        if hyp1 in used or hyp2 in used:
            continue
        chains.append(Chain(
            chain_type="hypernym",
            entities=(word, hyp1, hyp2),
            base_facts=[
                (word, "is_a", hyp1),
                (hyp1, "is_a", hyp2),
            ],
            derived_facts=[
                (word, "is_a", hyp2),
            ],
            base_text_relations=[
                (word, hyp1, "is_a"),
                (hyp1, hyp2, "is_a"),
            ],
            derived_text_relations=[
                (word, hyp2, "is_a"),
            ],
        ))
        used.update([word, hyp1, hyp2])
    return chains


# ============================================================================
# Chain Assembly and Tier Splitting
# ============================================================================

def generate_text_for_chain(chain, include_derived):
    """Generate template sentences for a chain's facts."""
    sentences = []
    for a, b, rel_name in chain.base_text_relations:
        for tmpl in TEXT_TEMPLATES[rel_name]:
            sentences.append(tmpl.format(A=a, B=b))
    if include_derived:
        for a, b, rel_name in chain.derived_text_relations:
            for tmpl in TEXT_TEMPLATES[rel_name]:
                sentences.append(tmpl.format(A=a, B=b))
    return sentences


def generate_kg_for_chain(chain, include_derived):
    """Generate KG triples for a chain."""
    triples = list(chain.base_facts)
    if include_derived:
        triples.extend(chain.derived_facts)
    return triples


def build_all_chains(args, seed=42):
    """Build and pool all chain types."""
    chains = []
    chains += build_family_chains(n_chains=500, seed=seed)
    print(f"  Family chains: {len(chains)}")

    if hasattr(args, 'wordnet_path') and args.wordnet_path and os.path.exists(args.wordnet_path):
        syn_chains = build_synonym_chains(args.wordnet_path, max_chains=400, seed=seed)
        print(f"  Synonym chains: {len(syn_chains)}")
        chains += syn_chains

        if hasattr(args, 'antonym_path') and args.antonym_path and os.path.exists(args.antonym_path):
            ant_chains = build_antonym_synonym_chains(
                args.antonym_path, args.wordnet_path, max_chains=150, seed=seed)
            print(f"  Antonym-synonym chains: {len(ant_chains)}")
            chains += ant_chains

    if (hasattr(args, 'e01_path') and args.e01_path and os.path.exists(args.e01_path) and
            hasattr(args, 'e02_path') and args.e02_path and os.path.exists(args.e02_path)):
        google_path = getattr(args, 'google_analogies', '') or ''
        cap_chains = build_capital_language_chains(
            args.e01_path, args.e02_path, google_path, max_chains=50, seed=seed)
        print(f"  Capital-language chains: {len(cap_chains)}")
        chains += cap_chains

    if hasattr(args, 'l01_path') and args.l01_path and os.path.exists(args.l01_path):
        l02 = getattr(args, 'l02_path', '') or ''
        hyp_chains = build_hypernym_chains(args.l01_path, l02, max_chains=100, seed=seed)
        print(f"  Hypernym chains: {len(hyp_chains)}")
        chains += hyp_chains

    rng = random.Random(seed)
    rng.shuffle(chains)
    print(f"  Total chains: {len(chains)}")
    return chains


def split_chains_into_tiers(chains):
    """Split chains into 7 tiers using proportional splits."""
    n = len(chains)
    if n == 0:
        return {tier: [] for tier in ALL_TIERS}
    return {
        "memorization":                  chains[:int(n * 0.58)],
        "transfer":                      chains[int(n * 0.58):int(n * 0.71)],
        "generalization":                chains[int(n * 0.71):int(n * 0.79)],
        "kg_exclusive_memorization":     chains[int(n * 0.79):int(n * 0.85)],
        "kg_exclusive_generalization":   chains[int(n * 0.85):int(n * 0.89)],
        "text_exclusive_memorization":   chains[int(n * 0.89):int(n * 0.95)],
        "text_exclusive_generalization": chains[int(n * 0.95):],
    }


def assemble_training_data(tiers, wiki_sentences, non_chain_triples):
    """Assemble text sentences and KG triples per tier rules.

    Returns (text_sentences, kg_triples) where:
    - text_sentences: list of word lists (wiki) + list of template strings
    - kg_triples: list of (head, rel, tail) tuples
    """
    # Template sentences (strings to be BPE-encoded later)
    template_sentences = []
    for chain in tiers["memorization"]:
        template_sentences += generate_text_for_chain(chain, include_derived=True)
    for chain in tiers["transfer"]:
        template_sentences += generate_text_for_chain(chain, include_derived=False)
    for chain in tiers["generalization"]:
        template_sentences += generate_text_for_chain(chain, include_derived=False)
    # kg_excl: NO template text
    for chain in tiers["text_exclusive_memorization"]:
        template_sentences += generate_text_for_chain(chain, include_derived=True)
    for chain in tiers["text_exclusive_generalization"]:
        template_sentences += generate_text_for_chain(chain, include_derived=False)

    # Convert template sentences to word lists for TextDataset
    text_sentences = list(wiki_sentences)
    for s in template_sentences:
        text_sentences.append(s.split())

    # KG triples from chains
    kg_triples = []
    for chain in tiers["memorization"]:
        kg_triples += generate_kg_for_chain(chain, include_derived=True)
    for chain in tiers["transfer"]:
        kg_triples += generate_kg_for_chain(chain, include_derived=True)
    for chain in tiers["generalization"]:
        kg_triples += generate_kg_for_chain(chain, include_derived=False)
    for chain in tiers["kg_exclusive_memorization"]:
        kg_triples += generate_kg_for_chain(chain, include_derived=True)
    for chain in tiers["kg_exclusive_generalization"]:
        kg_triples += generate_kg_for_chain(chain, include_derived=False)
    # text_excl: NO chain KG triples

    # Add non-chain background KG triples
    kg_triples += non_chain_triples

    return text_sentences, kg_triples


def build_chain_eval_prompts(tiers, vocab):
    """Build eval prompts from chain derived facts for all tiers.

    Returns (text_eval_prompts, kg_eval_prompts).
    """
    text_prompts = []
    kg_prompts = []

    for tier_name, chains in tiers.items():
        for chain in chains:
            for head, rel, tail in chain.derived_facts:
                # Text eval: use first template as prompt, truncated before target
                if rel in TEXT_TEMPLATES:
                    tmpl = TEXT_TEMPLATES[rel][0]
                    # Generate prompt by replacing {A} with head, {B} with empty
                    # Find where {B} starts and use text up to that point
                    prompt_text = tmpl.format(A=head, B="").rstrip()
                    # Remove trailing period/punctuation from truncated prompt
                    prompt_text = prompt_text.rstrip('.')
                    text_prompts.append({
                        "tier": tier_name,
                        "prompt": prompt_text,
                        "target": tail,
                        "relation": rel,
                        "chain_type": chain.chain_type,
                        "prompt_tokens": vocab.encode_sentence(prompt_text.split()),
                        "target_tokens": vocab.encode_entity(tail),
                    })

                # KG eval: head + rel → predict tail
                kg_prompts.append({
                    "tier": tier_name,
                    "head": head,
                    "rel": rel,
                    "tail": tail,
                    "relation": rel,
                    "chain_type": chain.chain_type,
                })

    return text_prompts, kg_prompts


def prepare_data(wiki_path, wiki_lines, vocab_size,
                 wordnet_path=None, framenet_path=None,
                 bats_dir=None, google_analogies_path=None,
                 word_analogies_path=None,
                 inverse_kg=False, kg_as_text=False, seed=42,
                 tokenizer_path=None, args=None):
    """Prepare all data for Exp 8 using chain-based tier system."""
    print("Loading data...")

    sentences = load_wiki_text(wiki_path, wiki_lines)
    print(f"  Wiki text: {len(sentences)} sentences")

    # Load non-chain KG triples (background data)
    non_chain_triples = []
    if wordnet_path and os.path.exists(wordnet_path):
        wn = load_wordnet_synonyms(wordnet_path)
        print(f"  WordNet: {len(wn)} triples")
        non_chain_triples.extend(wn)
    if framenet_path and os.path.exists(framenet_path):
        fn = load_framenet(framenet_path)
        print(f"  FrameNet: {len(fn)} triples")
        non_chain_triples.extend(fn)
    if bats_dir and os.path.exists(bats_dir):
        bt = load_bats_analogies(bats_dir)
        print(f"  BATS: {len(bt)} triples")
        non_chain_triples.extend(bt)
    if google_analogies_path and os.path.exists(google_analogies_path):
        ga = load_google_analogies(google_analogies_path)
        print(f"  Google analogies: {len(ga)} triples")
        non_chain_triples.extend(ga)
    if word_analogies_path and os.path.exists(word_analogies_path):
        wa = load_word_analogies(word_analogies_path)
        print(f"  Word analogies: {len(wa)} triples")
        non_chain_triples.extend(wa)

    print(f"  Total non-chain KG triples: {len(non_chain_triples)}")

    # Build chains and split into tiers
    print("Building chains...")
    chains = build_all_chains(args, seed=seed)
    tiers = split_chains_into_tiers(chains)

    for tier_name in ALL_TIERS:
        print(f"  {tier_name}: {len(tiers[tier_name])} chains")

    # Assemble training data per tier rules
    train_sentences, train_triples = assemble_training_data(
        tiers, sentences, non_chain_triples)
    print(f"  Training text sentences: {len(train_sentences)}")
    print(f"  Training KG triples: {len(train_triples)}")

    # Collect all KG relations (chain + non-chain)
    all_triples_for_rels = train_triples + non_chain_triples
    # Add chain relations explicitly
    kg_relations_list = sorted(set(
        _collect_kg_relations(all_triples_for_rels) + CHAIN_RELATIONS
    ))
    kg_relations_inverse = {r: f"inverse_{r}" for r in kg_relations_list}

    vocab = Vocabulary()
    for rel in kg_relations_list:
        vocab.add_relation(rel)

    if tokenizer_path and os.path.exists(tokenizer_path):
        vocab.load_tokenizer(tokenizer_path)
    else:
        # Collect all unique KG entity words for BPE training
        kg_entity_words = set()
        for head, rel, tail in train_triples:
            kg_entity_words.add(head)
            kg_entity_words.add(tail)
        # Also add chain entity words
        for chain in chains:
            for e in chain.entities:
                kg_entity_words.add(e)

        vocab.build_bpe(train_sentences, sorted(kg_entity_words), vocab_size=vocab_size)
        if tokenizer_path:
            vocab.save_tokenizer(tokenizer_path)

    print(f"  Vocabulary size: {vocab.size}")

    # Build eval prompts from chain derived facts
    eval_prompts, kg_eval_prompts = build_chain_eval_prompts(tiers, vocab)
    print(f"  Text eval prompts: {len(eval_prompts)}")
    print(f"  KG eval prompts: {len(kg_eval_prompts)}")

    text_dataset_base = TextDataset(train_sentences, vocab, cfg.block_size)

    linearized_kg_token_lists = []
    if kg_as_text:
        lin_strs = linearize_kg_triples(train_triples, kg_relations_inverse)
        for s in lin_strs:
            token_ids = _encode_linearized_string(s, vocab)
            linearized_kg_token_lists.append(token_ids)

    lin_sents = train_sentences + linearized_kg_token_lists if kg_as_text else train_sentences
    text_dataset_linearized = TextDataset(lin_sents, vocab, cfg.block_size)
    kg_dataset = KGDataset(train_triples, vocab, cfg.device, inverse_kg=inverse_kg)

    linearized_eval_prompts = None
    if kg_as_text:
        linearized_eval_prompts = _build_linearized_eval_prompts_from_chains(
            tiers, vocab, kg_relations_inverse)

    return (vocab, text_dataset_base, text_dataset_linearized, kg_dataset,
            eval_prompts, kg_eval_prompts, linearized_eval_prompts, kg_relations_list)


def _build_linearized_eval_prompts_from_chains(tiers, vocab, kg_relations_inverse):
    """Build linearized KG eval prompts from chain derived facts."""
    prompts = []
    for tier_name, chains in tiers.items():
        for chain in chains:
            for head, rel, tail in chain.derived_facts:
                inv_r = kg_relations_inverse.get(rel, f"inverse_{rel}")
                h_tokens = vocab.encode_entity(head)
                t_tokens = vocab.encode_entity(tail)
                rel_id = vocab.kg_relations.get(rel, vocab.UNK)
                inv_rel_id = vocab.kg_relations.get(inv_r, vocab.UNK)
                prompts.append({
                    "tier": tier_name, "prompt": f"{head} <{rel}> ",
                    "target": tail, "relation": rel,
                    "prompt_tokens": h_tokens + [rel_id],
                    "target_tokens": t_tokens,
                })
                prompts.append({
                    "tier": tier_name, "prompt": f"{tail} <{inv_r}> ",
                    "target": head, "relation": inv_r,
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
    elif model_type in ("E", "H", "I", "J"):
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
            if model_type in ("E", "H", "I", "J"):
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
                        if model_type in ("E", "H", "I", "J"):
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
                    mp = head_len + t_idx if model_type in ("E", "H", "I", "J") else head_len + 1 + t_idx
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
                    if model_type in ("E", "H", "I", "J"):
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
                        mp = head_len + ti if model_type in ("E", "H", "I", "J") else head_len + 1 + ti
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

@torch.no_grad()
def _eval_text_ppl(model, text_dataset, config, n_batches=10):
    """Estimate text PPL (causal) by averaging over n_batches."""
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = text_dataset.get_batch(config.batch_size, config.device)
        if hasattr(model, 'forward_text'):
            _, loss = model.forward_text(x, y)
        else:
            _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return math.exp(total_loss / n_batches)


@torch.no_grad()
def _eval_kg_ppl(model, kg_dataset, config, kg_batch_fn, n_batches=10):
    """Estimate KG PPL (MLM) by averaging over n_batches."""
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        if kg_batch_fn == "slotted":
            tokens, targets, head_lens, rel_names = kg_dataset.get_mlm_batch_slotted(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, loss = model.forward_kg(tokens, targets, head_lens, rel_names)
        elif kg_batch_fn == "native":
            ct, tgt, hl, rn, neg = kg_dataset.get_mlm_batch_native(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, loss = model.forward_kg(ct, tgt, hl, rn, neg)
        elif kg_batch_fn == "native_slots":
            ct, tgt, hl, rn = kg_dataset.get_mlm_batch_native_slots(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, loss = model.forward_kg(ct, tgt, hl, rn)
        elif kg_batch_fn == "flat":
            tokens, targets, rel_names = kg_dataset.get_mlm_batch_flat(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, loss = model.forward_kg(tokens, targets)
        total_loss += loss.item()
    model.train()
    return math.exp(total_loss / n_batches)


def train_model_text_only(model, text_dataset, config, name="?",
                          resume_optimizer_state=None):
    """Train text-only model (B/C) with next-token prediction.

    When config.dual_objective is True, randomly picks one objective per
    iteration: text_causal or text_mlm.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
    model.to(config.device)
    model.train()

    dual = config.dual_objective
    losses_log = {"text": [], "iter": [], "eval_text_ppl": []}
    if dual:
        losses_log["text_mlm"] = []

    for it in tqdm(range(config.max_iters), desc=f"Model {name}"):
        if dual:
            # Randomly pick one objective per iteration
            obj = random.choice(["text_causal", "text_mlm"])
        else:
            obj = "text_causal"

        causal_loss = torch.tensor(0.0)
        mlm_loss = torch.tensor(0.0)

        if obj == "text_causal":
            x, y = text_dataset.get_batch(config.batch_size, config.device)
            _, causal_loss = model(x, y)
            loss = causal_loss
        else:  # text_mlm
            mlm_tokens, mlm_targets = text_dataset.get_mlm_batch(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, mlm_loss = model.forward_text_mlm(mlm_tokens, mlm_targets)
            loss = mlm_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % config.eval_interval == 0:
            losses_log["iter"].append(it)
            losses_log["text"].append(causal_loss.item())
            train_ppl = math.exp(loss.item())
            eval_text_ppl = _eval_text_ppl(model, text_dataset, config)
            losses_log["eval_text_ppl"].append(round(eval_text_ppl, 2))
            if dual:
                losses_log["text_mlm"].append(mlm_loss.item())
                print(f"  [{name}] iter {it} [{obj}], train_ppl: {train_ppl:.2f}, "
                      f"eval_text_ppl: {eval_text_ppl:.2f}")
            else:
                print(f"  [{name}] iter {it}, train_ppl: {train_ppl:.2f}, "
                      f"eval_text_ppl: {eval_text_ppl:.2f}")

    return losses_log, optimizer.state_dict()


def train_model_mixed(model, text_dataset, kg_dataset, config, name="?",
                      kg_batch_fn="native", resume_optimizer_state=None,
                      kg_only=False, causal_kg=False):
    """Train mixed text+KG model.

    Each iteration: text batch (causal, NTP) + KG batch (bidir, MLM).
    When config.dual_objective is True, randomly picks one objective per
    iteration from {text_causal, text_mlm, kg_mlm, kg_causal}.

    Args:
        kg_batch_fn: "slotted" for A/A',G/G', "native" for E/E',H/H',I/I',
                     "native_slots" for J/J', "flat" for D/D',F/F'
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
    model.to(config.device)
    model.train()

    dual = config.dual_objective

    losses_log = {"text": [], "kg": [], "iter": [],
                  "eval_text_ppl": [], "eval_kg_ppl": []}
    if dual:
        losses_log["text_mlm"] = []
        losses_log["kg_causal"] = []

    # Build the pool of objectives for random selection
    if dual:
        if kg_only:
            objectives = ["kg_mlm", "kg_causal"]
        else:
            objectives = ["text_causal", "text_mlm", "kg_mlm", "kg_causal"]

    for it in tqdm(range(config.max_iters), desc=f"Model {name}"):
        text_loss = torch.tensor(0.0)
        text_mlm_loss = torch.tensor(0.0)
        kg_loss = torch.tensor(0.0)
        kg_causal_loss = torch.tensor(0.0)

        if dual:
            obj = random.choice(objectives)
        else:
            obj = None  # non-dual: always do text_causal + kg_mlm

        if not dual:
            # --- Original behavior: text causal + KG MLM ---
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
            elif kg_batch_fn == "native_slots":
                ct, tgt, hl, rn = kg_dataset.get_mlm_batch_native_slots(
                    config.batch_size, config.device, config.mlm_mask_prob)
                _, kg_loss = model.forward_kg(ct, tgt, hl, rn)
            elif kg_batch_fn == "flat":
                tokens, targets, rel_names = kg_dataset.get_mlm_batch_flat(
                    config.batch_size, config.device, config.mlm_mask_prob)
                _, kg_loss = model.forward_kg(tokens, targets)

            if kg_only:
                loss = kg_loss
            else:
                loss = text_loss + kg_loss

        else:
            # --- Dual: randomly picked single objective ---
            if obj == "text_causal":
                x, y = text_dataset.get_batch(config.batch_size, config.device)
                if hasattr(model, 'forward_text'):
                    _, text_loss = model.forward_text(x, y)
                else:
                    _, text_loss = model(x, y)
                loss = text_loss

            elif obj == "text_mlm":
                mlm_tokens, mlm_targets = text_dataset.get_mlm_batch(
                    config.batch_size, config.device, config.mlm_mask_prob)
                _, text_mlm_loss = model.forward_text_mlm(mlm_tokens, mlm_targets)
                loss = text_mlm_loss

            elif obj == "kg_mlm":
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
                elif kg_batch_fn == "native_slots":
                    ct, tgt, hl, rn = kg_dataset.get_mlm_batch_native_slots(
                        config.batch_size, config.device, config.mlm_mask_prob)
                    _, kg_loss = model.forward_kg(ct, tgt, hl, rn)
                elif kg_batch_fn == "flat":
                    tokens, targets, rel_names = kg_dataset.get_mlm_batch_flat(
                        config.batch_size, config.device, config.mlm_mask_prob)
                    _, kg_loss = model.forward_kg(tokens, targets)
                loss = kg_loss

            elif obj == "kg_causal":
                if kg_batch_fn == "slotted":
                    c_tokens, c_targets, c_sa, c_sp, c_rels, c_ctx = \
                        kg_dataset.get_slot_causal_batch_slotted(config.batch_size, config.device)
                    _, kg_causal_loss = model.forward_kg_causal(
                        c_tokens, c_targets, c_sa, c_sp, c_rels, c_ctx)
                elif kg_batch_fn == "native":
                    c_tokens, c_targets, c_hlens, c_rels, c_neg, c_ctx = \
                        kg_dataset.get_slot_causal_batch_native(config.batch_size, config.device)
                    _, kg_causal_loss = model.forward_kg_causal(
                        c_tokens, c_targets, c_hlens, c_rels, c_neg, c_ctx)
                elif kg_batch_fn == "flat":
                    c_tokens, c_targets, c_rels, c_ctx = \
                        kg_dataset.get_slot_causal_batch_flat(config.batch_size, config.device)
                    _, kg_causal_loss = model.forward_kg_causal(
                        c_tokens, c_targets, c_ctx)
                elif kg_batch_fn == "native_slots":
                    c_tokens, c_targets, c_sa, c_sp, c_rels, c_ctx = \
                        kg_dataset.get_slot_causal_batch_native_slots(config.batch_size, config.device)
                    _, kg_causal_loss = model.forward_kg_causal(
                        c_tokens, c_targets, c_sa, c_sp, c_rels, c_ctx)
                loss = kg_causal_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % config.eval_interval == 0:
            losses_log["iter"].append(it)
            losses_log["text"].append(text_loss.item())
            losses_log["kg"].append(kg_loss.item())
            train_ppl = math.exp(loss.item())
            eval_text_ppl = _eval_text_ppl(model, text_dataset, config) if not kg_only else 0.0
            eval_kg_ppl = _eval_kg_ppl(model, kg_dataset, config, kg_batch_fn)
            losses_log["eval_text_ppl"].append(round(eval_text_ppl, 2))
            losses_log["eval_kg_ppl"].append(round(eval_kg_ppl, 2))
            if dual:
                losses_log["text_mlm"].append(text_mlm_loss.item())
                losses_log["kg_causal"].append(kg_causal_loss.item())
                print(f"  [{name}] iter {it} [{obj}], train_ppl: {train_ppl:.2f}, "
                      f"eval_text_ppl: {eval_text_ppl:.2f}, eval_kg_ppl: {eval_kg_ppl:.2f}")
            else:
                print(f"  [{name}] iter {it}, train_ppl: {train_ppl:.2f}, "
                      f"eval_text_ppl: {eval_text_ppl:.2f}, eval_kg_ppl: {eval_kg_ppl:.2f}")

    return losses_log, optimizer.state_dict()


# ============================================================================
# Model Factory
# ============================================================================

MODEL_NAMES = ["A", "A'", "B", "B'", "C", "C'", "D", "D'", "E", "E'",
               "F", "F'", "G", "G'", "H", "H'", "I", "I'", "J", "J'"]

LINEARIZED_MODELS = {"B", "B'", "C", "C'"}
SLOTTED_KG_MODELS = {"A", "A'", "G", "G'"}
NATIVE_KG_MODELS = {"E", "E'", "H", "H'", "I", "I'"}
NATIVE_SLOTS_KG_MODELS = {"J", "J'"}           # use get_mlm_batch_native_slots (2 slots: HEAD/TAIL, no rel token)
FLAT_KG_MODELS = {"D", "D'", "F", "F'"}


def create_model(name, vocab_size, config, n_relations=8):
    """Create a model by name."""
    n_e = config.n_embed
    n_l = config.n_layers
    bs = config.block_size
    d = config.dropout
    sm = config.use_softmax

    if name == "A":    return ModelA(vocab_size, n_e, n_l, bs, d, rotate_v=False, use_softmax=sm)
    elif name == "A'": return ModelA(vocab_size, n_e, n_l, bs, d, rotate_v=True, use_softmax=sm)
    elif name == "B":  return ModelB(vocab_size, n_e, n_l, bs, d, rotate_v=False, use_softmax=sm)
    elif name == "B'": return ModelB(vocab_size, n_e, n_l, bs, d, rotate_v=True, use_softmax=sm)
    elif name == "C":  return ModelC(vocab_size, n_e, n_l, bs, d, rotate_v=False, use_softmax=sm)
    elif name == "C'": return ModelC(vocab_size, n_e, n_l, bs, d, rotate_v=True, use_softmax=sm)
    elif name == "D":  return ModelD(vocab_size, n_e, n_l, bs, d, rotate_v=False, use_softmax=sm)
    elif name == "D'": return ModelD(vocab_size, n_e, n_l, bs, d, rotate_v=True, use_softmax=sm)
    elif name == "E":  return ModelE(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=False, use_softmax=sm)
    elif name == "E'": return ModelE(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=True, use_softmax=sm)
    elif name == "F":  return ModelF(vocab_size, n_e, n_l, bs, d, rotate_v=False, use_softmax=sm)
    elif name == "F'": return ModelF(vocab_size, n_e, n_l, bs, d, rotate_v=True, use_softmax=sm)
    elif name == "G":  return ModelG(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=False, use_softmax=sm)
    elif name == "G'": return ModelG(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=True, use_softmax=sm)
    elif name == "H":  return ModelH(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=False, use_softmax=sm)
    elif name == "H'": return ModelH(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=True, use_softmax=sm)
    elif name == "I":  return ModelI(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=False, use_softmax=sm)
    elif name == "I'": return ModelI(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=True, use_softmax=sm)
    elif name == "J":  return ModelJ(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=False, use_softmax=sm)
    elif name == "J'": return ModelJ(vocab_size, n_e, n_l, bs, n_relations=n_relations, dropout=d, rotate_v=True, use_softmax=sm)
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
                   google_analogies_path=None, word_analogies_path=None,
                   tokenizer_path=None, args=None):
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
        inverse_kg=inverse_kg, kg_as_text=kg_as_text, seed=seed,
        tokenizer_path=tokenizer_path, args=args)

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

    training_curves = {}
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
            curves = {}
            if name in LINEARIZED_MODELS:
                curves, opt_state = train_model_text_only(model, text_linearized, cfg, name=name,
                                                          resume_optimizer_state=resume_opt)
            elif name in SLOTTED_KG_MODELS:
                curves, opt_state = train_model_mixed(model, text_base, kg_dataset, cfg,
                                                       name=name, kg_batch_fn="slotted",
                                                       resume_optimizer_state=resume_opt, kg_only=kg_only)
            elif name in NATIVE_KG_MODELS:
                curves, opt_state = train_model_mixed(model, text_base, kg_dataset, cfg,
                                                       name=name, kg_batch_fn="native",
                                                       resume_optimizer_state=resume_opt,
                                                       kg_only=kg_only, causal_kg=causal_kg)
            elif name in NATIVE_SLOTS_KG_MODELS:
                curves, opt_state = train_model_mixed(model, text_base, kg_dataset, cfg,
                                                       name=name, kg_batch_fn="native_slots",
                                                       resume_optimizer_state=resume_opt, kg_only=kg_only)
            elif name in FLAT_KG_MODELS:
                curves, opt_state = train_model_mixed(model, text_base, kg_dataset, cfg,
                                                       name=name, kg_batch_fn="flat",
                                                       resume_optimizer_state=resume_opt, kg_only=kg_only)
            training_curves[name] = curves
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
        if not kg_as_text and base_name in ("A", "D", "E", "F", "G", "H", "I", "J"):
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
            linearized_results, linearized_relation_results, training_curves)


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
    kg_models = [m for m in models if m.replace("'", "") in ("A", "D", "E", "F", "G", "H", "I", "J")]
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
    parser.add_argument("--softmax", action="store_true",
                        help="Use softmax attention instead of log(exp(x)+1)")
    parser.add_argument("--dual_objective", action="store_true",
                        help="Enable dual-objective training: MLM+causal on text, MLM+causal on KG")
    # Data paths (relative to script directory)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _data_dir = os.path.join(_script_dir, "data")
    _bats_dir = os.path.join(_data_dir, "BATS_3.0")
    parser.add_argument("--wiki_path", default=os.path.join(_data_dir, "wiki.en.txt"))
    parser.add_argument("--wiki_lines", type=int, default=1000000)
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--wordnet_path", default=os.path.join(_data_dir, "wordnet-synonyms.txt"))
    parser.add_argument("--framenet_path", default=os.path.join(_data_dir, "framenet.txt"))
    parser.add_argument("--bats_dir", default=_bats_dir)
    parser.add_argument("--google_analogies", default=os.path.join(_data_dir, "questions-words_for_training.txt"))
    parser.add_argument("--word_analogies", default=os.path.join(_data_dir, "wordanalogies.txt"))
    parser.add_argument("--tokenizer_path", default=os.path.join(_data_dir, "tokenizer.json"))
    parser.add_argument("--train_tokenizer", action="store_true",
                        help="Train BPE tokenizer on full dataset and save, then exit")
    parser.add_argument("--exp", default="8a")
    # Chain-specific data paths
    parser.add_argument("--antonym_path", default=os.path.join(
        _bats_dir, "4_Lexicographic_semantics", "L09_comprehensive.txt"))
    parser.add_argument("--e01_path", default=os.path.join(
        _bats_dir, "3_Encyclopedic_semantics", "E01 [country - capital].txt"))
    parser.add_argument("--e02_path", default=os.path.join(
        _bats_dir, "3_Encyclopedic_semantics", "E02 [country - language].txt"))
    parser.add_argument("--l01_path", default=os.path.join(
        _bats_dir, "4_Lexicographic_semantics", "L01 [hypernyms - animals].txt"))
    parser.add_argument("--l02_path", default=os.path.join(
        _bats_dir, "4_Lexicographic_semantics", "L02 [hypernyms - misc].txt"))
    parser.add_argument("--n_chains", type=int, default=1200,
                        help="Target total chain count (actual may be less)")
    args = parser.parse_args()

    # Train-tokenizer-only mode: train BPE on full dataset and exit
    if args.train_tokenizer:
        print("Training BPE tokenizer on full dataset...")
        sentences = load_wiki_text(args.wiki_path, args.wiki_lines)
        print(f"  Wiki text: {len(sentences)} sentences")
        all_triples = []
        if args.wordnet_path and os.path.exists(args.wordnet_path):
            wn = load_wordnet_synonyms(args.wordnet_path)
            print(f"  WordNet: {len(wn)} triples")
            all_triples.extend(wn)
        if args.framenet_path and os.path.exists(args.framenet_path):
            fn = load_framenet(args.framenet_path)
            print(f"  FrameNet: {len(fn)} triples")
            all_triples.extend(fn)
        if args.bats_dir and os.path.exists(args.bats_dir):
            bt = load_bats_analogies(args.bats_dir)
            print(f"  BATS: {len(bt)} triples")
            all_triples.extend(bt)
        if args.google_analogies and os.path.exists(args.google_analogies):
            ga = load_google_analogies(args.google_analogies)
            print(f"  Google analogies: {len(ga)} triples")
            all_triples.extend(ga)
        if args.word_analogies and os.path.exists(args.word_analogies):
            wa = load_word_analogies(args.word_analogies)
            print(f"  Word analogies: {len(wa)} triples")
            all_triples.extend(wa)
        kg_relations_list = sorted(set(
            _collect_kg_relations(all_triples) + CHAIN_RELATIONS
        ))
        kg_entity_words = set()
        for h, r, t in all_triples:
            kg_entity_words.add(h)
            kg_entity_words.add(t)
        # Also build chains so their entities (esp. family names) are in BPE training
        print("  Building chains for tokenizer training...")
        chains = build_all_chains(args, seed=42)
        for chain in chains:
            for e in chain.entities:
                kg_entity_words.add(e)
        # Add template sentences to training corpus
        for chain in chains:
            for s in generate_text_for_chain(chain, include_derived=True):
                sentences.append(s.split())
        print(f"  Chain entity words added: {sum(1 for c in chains for _ in c.entities)}")
        vocab = Vocabulary()
        for rel in kg_relations_list:
            vocab.add_relation(rel)
        vocab.build_bpe(sentences, sorted(kg_entity_words), vocab_size=args.vocab_size)
        vocab.save_tokenizer(args.tokenizer_path)
        print(f"Done. Vocab size: {vocab.size}")
        return

    if args.n_embed is not None:
        cfg.n_embed = args.n_embed
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.softmax:
        cfg.use_softmax = True
    if args.dual_objective:
        cfg.dual_objective = True

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
        if m.endswith("p") and len(m) == 2 and m[0] in "ABCDEFGHIJ":
            normalized.append(m[0] + "'")
        else:
            normalized.append(m)
    models = normalized

    print("=" * 70)
    print("  Exp 8: Word-Level KG+Text on Real Data (Dual)")
    print("=" * 70)
    print(f"\nConfig: n_embed={cfg.n_embed}, n_layers={cfg.n_layers}, "
          f"max_iters={cfg.max_iters}, batch_size={cfg.batch_size}, "
          f"lr={cfg.lr}, device={cfg.device}")
    print(f"Wiki lines: {args.wiki_lines}, Vocab size: {args.vocab_size}")
    if cfg.dual_objective:
        print("Dual objective: ENABLED (text causal+MLM, KG MLM+causal)")
    if cfg.use_softmax:
        print("Attention: softmax")
    print(f"Models: {models}")

    all_results = {}
    seed_results_list = []
    seed_kg_results_list = []
    seed_lin_results_list = []
    all_training_curves = {}  # model_name -> list of per-seed curves

    for seed in range(cfg.n_seeds):
        (res, rel_res, kg_res, kg_rel_res,
         lin_res, lin_rel_res, curves) = run_experiment(
            wiki_path=args.wiki_path, wiki_lines=args.wiki_lines,
            vocab_size=args.vocab_size, seed=seed, models_to_run=models,
            checkpoint_dir=args.checkpoint_dir,
            load_checkpoints=args.load_checkpoints,
            resume_training=args.resume_training,
            kg_only=args.kg_only, causal_kg=args.causal_kg,
            inverse_kg=args.inverse_kg, kg_as_text=args.kg_as_text,
            wordnet_path=args.wordnet_path, framenet_path=args.framenet_path,
            bats_dir=args.bats_dir, google_analogies_path=args.google_analogies,
            word_analogies_path=args.word_analogies,
            tokenizer_path=args.tokenizer_path, args=args)
        seed_results_list.append(res)
        seed_kg_results_list.append(kg_res)
        seed_lin_results_list.append(lin_res)
        for m, c in curves.items():
            all_training_curves.setdefault(m, []).append(c)

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
    kg_models = [m for m in models if m.replace("'", "") in ("A", "D", "E", "F", "G", "H", "I", "J")]
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
        "training_curves": to_serializable(all_training_curves),
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
