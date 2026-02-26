# -----------------------------------------------------------------------------
# Exp 7: Native KG+Text vs Linearized KG-as-Text
#
# Tests the core claim: journey attention with native KG role operators
# outperforms linearizing KG triples into text.
#
# Ten models organized by two axes:
#   Angle computation / KG handling:
#     A: RoPE + learned slot angles, native KG (HEAD/REL/TAIL slots, bidir MLM)
#     B: RoPE positional, linearized KG-as-text (causal LM)
#     C: Per-token cumsum (journey), linearized KG-as-text (causal LM)
#     D: Per-token cumsum (journey), flat KG sequence, rel is a token (bidir MLM)
#     E: Per-token cumsum + relation operator, native KG, rel is operator-only (bidir MLM)
#
#   Value aggregation (primed = operator-based):
#     Unprimed (A,B,C,D,E): rotate Q,K only (commutative)
#     Primed (A',B',C',D',E'): rotate Q,K,V + inverse on output (operator-based)
#
# Two sub-experiments:
#   7a: Generous linearization (all text variations for KG-as-text)
#   7b: Realistic linearization (one text variation per KG triple)
#
# Seven evaluation tiers (text + KG evaluation):
#   Tier 1: Memorization (teaching chains -- all models saw all facts)
#   Tier 2: Transfer (derived facts -- all models saw them, format differs)
#   Tier 3: Generalization (NO model saw derived facts)
#   Tier 4: KG-exclusive memorization (ALL facts in KG only, zero text)
#   Tier 5: KG-exclusive generalization (base in KG only, derived nowhere)
#   Tier 6: Text-exclusive memorization (ALL facts in text only, zero KG)
#   Tier 7: Text-exclusive generalization (base in text only, derived nowhere)
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

class Config:
    n_embed = 24
    n_layers = 1
    dropout = 0.2
    block_size = 48       # max text sequence length
    kg_block_size = 32    # max KG sequence length (for flat KG models)
    batch_size = 32
    max_iters = 5000
    lr = 5e-4
    eval_interval = 500
    eval_iters = 20
    n_seeds = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlm_mask_prob = 0.15  # MLM masking probability

cfg = Config()

# ============================================================================
# Data Generation: Synthetic Family + Geography World
# ============================================================================

MALE_NAMES = [
    # 390 distinct real male names (need 390 for 130 chains of 3)
    "Adam", "Brian", "Carl", "Dave", "Eric", "Frank", "George", "Henry",
    "Ivan", "Jack", "Kevin", "Leo", "Mike", "Nick", "Oscar", "Paul",
    "Quinn", "Ryan", "Sam", "Tom", "Umar", "Vince", "Will", "Xander",
    "Yuri", "Zack", "Aaron", "Blake", "Chase", "Dean", "Ethan", "Felix",
    "Grant", "Hugo", "Ian", "Joel", "Kurt", "Liam", "Marco", "Noel",
    "Owen", "Peter", "Reed", "Scott", "Troy", "Uri", "Vlad", "Wade",
    "Yale", "Zane", "Abel", "Beau", "Clay", "Drew", "Earl", "Finn",
    "Glen", "Hank", "Ira", "Jeff", "Kane", "Luke", "Mark", "Ned",
    "Otis", "Phil", "Russ", "Seth", "Tate", "Usher", "Val", "Wes",
    "Axel", "Zeke", "Alec", "Brad", "Cole", "Doug", "Eli", "Fred",
    "Gus", "Herb", "Ike", "Jay", "Kent", "Lars", "Milo", "Nate",
    "Olaf", "Pat", "Ray", "Stan", "Ted", "Vern", "Walt", "Xeno",
    "York", "Zion", "Amos", "Bart", "Cody", "Dale", "Emil", "Ford",
    "Gage", "Hart", "Igor", "Jude", "Kirk", "Lane", "Mort", "Nash",
    "Odin", "Penn", "Rolf", "Saul", "Thad", "Ulric", "Vic", "Wyatt",
    "Yves", "Zach", "Alvin", "Basil", "Clyde", "Damon", "Edgar", "Floyd",
    "Gavin", "Homer", "Irwin", "Jonas", "Keith", "Lloyd", "Miles", "Nigel",
    "Orion", "Percy", "Ralph", "Simon", "Tobin", "Uriah", "Virgil", "Wayne",
    "Xerxes", "Yosef", "Zahir", "Andre", "Boris", "Cyrus", "Derek", "Edwin",
    "Farid", "Glenn", "Hiram", "Isaac", "Jamal", "Karim", "Louis", "Mason",
    "Nigel", "Omar", "Pryce", "Roger", "Silas", "Tyson", "Usama", "Vidal",
    "Waldo", "Yakov", "Zubin", "Atlas", "Brent", "Craig", "Daryl", "Elton",
    "Flynn", "Gregg", "Heath", "Idris", "Jarvis", "Knox", "Leroy", "Myron",
    "Noble", "Orson", "Piers", "Rufus", "Stone", "Trent", "Urban", "Viggo",
    "Wyman", "Yates", "Zared", "Angus", "Byron", "Corey", "Devin", "Emery",
    "Fritz", "Galen", "Haven", "Inigo", "James", "Kaleb", "Linus", "Myles",
    "Nolan", "Ozias", "Pryor", "Rowan", "Soren", "Tarik", "Umbra", "Valor",
    "Wolfe", "Yancy", "Zelig", "Ariel", "Burke", "Cliff", "Doran", "Eamon",
    "Fabio", "Garth", "Hogan", "Izzy", "Jovan", "Klaus", "Laird", "Macon",
    "Norris", "Orval", "Price", "Reese", "Shawn", "Thane", "Usain", "Vance",
    "Walsh", "Yehuda", "Zoltan", "Abram", "Banks", "Colby", "Dante", "Einar",
    "Faron", "Grady", "Hamza", "Ismet", "Jasper", "Kenji", "Lonny", "Magnus",
    "Nevin", "Odell", "Paulo", "Renzo", "Slade", "Tucker", "Upton", "Varun",
    "Wiley", "Yannis", "Zander", "Anton", "Bryce", "Caleb", "Dixon", "Elias",
    "Fitch", "Giles", "Hadley", "Ivor", "Jethro", "Kobe", "Lennox", "Morris",
    "Niles", "Oakley", "Pierce", "Ramon", "Spence", "Travis", "Usher", "Vernon",
    "Wilson", "Yasser", "Zenon", "Ashby", "Bowen", "Clint", "Doyle", "Enoch",
    "Frost", "Grover", "Holmes", "Irving", "Josiah", "Kelvin", "Lester", "Murray",
    "Newman", "Osborn", "Porter", "Rhodes", "Stein", "Tobias", "Ulysses", "Viktor",
    "Warren", "Yusuf", "Zephyr", "Aldric", "Blaine", "Corbin", "Desmond",
    "Elmer", "Fenton", "Gilroy", "Harlan", "Ingram", "Jerome", "Kelsey",
    "Layton", "Marcel", "Nestor", "Osmund", "Palmer", "Quentin",
    # Extra 63 names for text-exclusive chains (130 chains = 390 names)
    "Alaric", "Benito", "Cedric", "Dmitri", "Emilio", "Fergus",
    "Gunnar", "Hamish", "Isaias", "Jorgen", "Kaspar", "Lachlan",
    "Matteo", "Nikita", "Osman", "Petros", "Rashid", "Sergei",
    "Thorin", "Ulrich", "Vasily", "Willem", "Yoshi", "Kenichi",
    "Zoltar", "Aldous", "Bennet", "Crispin", "Dalton", "Egbert",
    "Felipe", "Gustaf", "Horace", "Ignacio", "Jacoby", "Keenan",
    "Leopold", "Merritt", "Nikolai", "Octavio", "Preston", "Roderic",
    "Sheldon", "Theron", "Lucian", "Vaughan", "Oswald", "Braden",
    "Yardley", "Zebulon", "Alistair", "Barrett", "Carlton", "Dominik",
    "Ephraim", "Florian", "Godwin", "Hadrian", "Isidore", "Jericho",
    "Kendall", "Lysander", "Montague",
]
# Deduplicate while preserving order
_seen = set()
_unique = []
for _n in MALE_NAMES:
    if _n not in _seen:
        _seen.add(_n)
        _unique.append(_n)
MALE_NAMES = _unique
del _seen, _unique, _n


def _generate_extra_names(needed, existing):
    """Generate deterministic CVCVC name-like strings not in existing."""
    consonants = "bcdfghjklmnprstvwz"
    vowels = "aeiou"
    existing_set = set(existing)
    extras = []
    for c1 in consonants:
        for v1 in vowels:
            for c2 in consonants:
                for v2 in vowels:
                    for c3 in consonants:
                        name = (c1 + v1 + c2 + v2 + c3).capitalize()
                        if name not in existing_set:
                            existing_set.add(name)
                            extras.append(name)
                            if len(extras) >= needed:
                                return extras
    return extras


def get_names(n_needed, expanded=False):
    """Return at least n_needed unique names, generating extras if necessary."""
    if expanded:
        return generate_expanded_names(n_needed)
    if len(MALE_NAMES) >= n_needed:
        return MALE_NAMES[:n_needed]
    extras = _generate_extra_names(n_needed - len(MALE_NAMES), MALE_NAMES)
    return MALE_NAMES + extras

EXPANDED_NAME_CHARS = sorted(set(
    [chr(i) for i in range(0x21, 0x7F) if chr(i) not in ".'{}<>"]
    + [chr(i) for i in range(0x00C0, 0x0100)]
    + [chr(i) for i in range(0x0391, 0x03CA) if chr(i).isalpha()]
    + [chr(i) for i in range(0x0410, 0x0450)]
))

def generate_expanded_names(n_needed, name_length=4, seed=42):
    """Generate n_needed unique random names from expanded ~240-char alphabet."""
    rng = random.Random(seed)
    chars = EXPANDED_NAME_CHARS
    names_set = set()
    names = []
    while len(names) < n_needed:
        name = ''.join(rng.choice(chars) for _ in range(name_length))
        if name not in names_set:
            names_set.add(name)
            names.append(name)
    return names

CAPITALS = {
    "London": "England", "Paris": "France", "Berlin": "Germany",
    "Rome": "Italy", "Madrid": "Spain", "Tokyo": "Japan",
    "Delhi": "India", "Cairo": "Egypt", "Lima": "Peru",
    "Oslo": "Norway", "Athens": "Greece", "Lisbon": "Portugal",
    "Vienna": "Austria", "Prague": "Czechia", "Dublin": "Ireland",
    "Seoul": "Korea", "Ankara": "Turkey", "Nairobi": "Kenya",
    "Havana": "Cuba", "Bogota": "Colombia",
}


def generate_world(seed=42, expanded_names=False):
    """Generate a synthetic world with family trees and geography.

    1070 chains (3210 names) split into 7 tiers:
      - Teaching (1000): all facts in text + KG  (6000 shared facts)
      - Transfer (15): base in text, derived in KG/extra-text
      - Generalization (15): base in text/KG, derived nowhere
      - KG-exclusive memorization (10): ALL facts in KG only, zero text
      - KG-exclusive generalization (10): base facts in KG only, derived nowhere, zero text
      - Text-exclusive memorization (10): ALL facts in text only, zero KG
      - Text-exclusive generalization (10): base facts in text only, derived nowhere, zero KG
    """
    rng = random.Random(seed)

    n_chains = 1070
    names = get_names(n_chains * 3, expanded=expanded_names)
    names = names.copy()
    rng.shuffle(names)

    assert len(names) >= n_chains * 3, (
        f"Need {n_chains * 3} names, have {len(names)}")

    chains = []
    idx = 0
    for i in range(n_chains):
        grandfather = names[idx]
        father = names[idx + 1]
        son = names[idx + 2]
        chains.append((son, father, grandfather))
        idx += 3

    teaching_chains = chains[:1000]
    transfer_chains = chains[1000:1015]
    generalization_chains = chains[1015:1030]
    kg_excl_mem_chains = chains[1030:1040]
    kg_excl_gen_chains = chains[1040:1050]
    text_excl_mem_chains = chains[1050:1060]
    text_excl_gen_chains = chains[1060:1070]

    # Geography: assign cities to all teaching/transfer/generalization people
    non_exclusive_names = []
    for chain in teaching_chains + transfer_chains + generalization_chains:
        non_exclusive_names.extend(chain)

    city_list = list(CAPITALS.keys())
    geo_facts = []
    for i, person in enumerate(non_exclusive_names):
        city = city_list[i % len(city_list)]
        geo_facts.append((person, city))

    return (teaching_chains, transfer_chains, generalization_chains,
            kg_excl_mem_chains, kg_excl_gen_chains,
            text_excl_mem_chains, text_excl_gen_chains, geo_facts)


ALL_TIERS = [
    "memorization", "transfer", "generalization",
    "kg_exclusive_memorization", "kg_exclusive_generalization",
    "text_exclusive_memorization", "text_exclusive_generalization",
]

TEXT_TEMPLATES = {
    "son_of": [
        "{A} is the son of {B}.",
        "{A} is {B}'s son.",
        "{B} has a son named {A}.",
    ],
    "father_of": [
        "{B} is the father of {A}.",
        "{B} is {A}'s father.",
        "{A}'s father is {B}.",
    ],
    "grandson_of": [
        "{A} is the grandson of {C}.",
        "{A} is {C}'s grandson.",
        "{C} has a grandson named {A}.",
    ],
    "grandfather_of": [
        "{C} is the grandfather of {A}.",
        "{C} is {A}'s grandfather.",
        "{A}'s grandfather is {C}.",
    ],
    "brother_of": [
        "{A} is the brother of {B}.",
        "{A} and {B} are brothers.",
    ],
    "capital_of": [
        "{X} is the capital of {Y}.",
        "The capital of {Y} is {X}.",
        "{Y}'s capital is {X}.",
    ],
    "lives_in": [
        "{A} lives in {X}.",
        "{A} resides in {X}.",
        "{A} has been living in {X}.",
    ],
    "lives_in_country": [
        "{A} lives in {Y}.",
        "{A} resides in {Y}.",
        "{A} has been living in {Y}.",
    ],
}


def generate_text_sentences(chains, include_derived, all_variations=True):
    sentences = []
    for son, father, grandfather in chains:
        base_relations = [
            ("son_of", {"A": son, "B": father}),
            ("father_of", {"B": father, "A": son}),
            ("son_of", {"A": father, "B": grandfather}),
            ("father_of", {"B": grandfather, "A": father}),
        ]
        for rel, params in base_relations:
            templates = TEXT_TEMPLATES[rel]
            if all_variations:
                for t in templates:
                    sentences.append(t.format(**params))
            else:
                sentences.append(templates[0].format(**params))

        if include_derived:
            derived_relations = [
                ("grandson_of", {"A": son, "C": grandfather}),
                ("grandfather_of", {"C": grandfather, "A": son}),
            ]
            for rel, params in derived_relations:
                templates = TEXT_TEMPLATES[rel]
                if all_variations:
                    for t in templates:
                        sentences.append(t.format(**params))
                else:
                    sentences.append(templates[0].format(**params))

    return sentences


def generate_geo_sentences(geo_facts, all_variations=True):
    sentences = []
    for person, city in geo_facts:
        templates = TEXT_TEMPLATES["lives_in"]
        if all_variations:
            for t in templates:
                sentences.append(t.format(A=person, X=city))
        else:
            sentences.append(templates[0].format(A=person, X=city))

    for city, country in CAPITALS.items():
        templates = TEXT_TEMPLATES["capital_of"]
        if all_variations:
            for t in templates:
                sentences.append(t.format(X=city, Y=country))
        else:
            sentences.append(templates[0].format(X=city, Y=country))

    for person, city in geo_facts:
        if city in CAPITALS:
            country = CAPITALS[city]
            templates = TEXT_TEMPLATES["lives_in_country"]
            if all_variations:
                for t in templates:
                    sentences.append(t.format(A=person, Y=country))
            else:
                sentences.append(templates[0].format(A=person, Y=country))

    return sentences


def generate_kg_triples(chains, include_derived):
    triples = []
    for son, father, grandfather in chains:
        triples.append((son, "son_of", father))
        triples.append((father, "father_of", son))
        triples.append((father, "son_of", grandfather))
        triples.append((grandfather, "father_of", father))

        if include_derived:
            triples.append((son, "grandson_of", grandfather))
            triples.append((grandfather, "grandfather_of", son))

    return triples


def generate_geo_kg_triples(geo_facts):
    triples = []
    for person, city in geo_facts:
        triples.append((person, "lives_in", city))

    for city, country in CAPITALS.items():
        triples.append((city, "capital_of", country))

    for person, city in geo_facts:
        if city in CAPITALS:
            country = CAPITALS[city]
            triples.append((person, "lives_in_country", country))

    return triples


# ============================================================================
# Vocabulary and Tokenization
# ============================================================================

# Relation names used across all models
KG_RELATIONS = [
    "son_of", "father_of", "grandson_of", "grandfather_of",
    "brother_of", "capital_of", "lives_in", "lives_in_country"
]


class Vocabulary:
    """Character-level vocabulary with special tokens."""

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.size = 0

        self.PAD = self._add("<PAD>")
        self.BOS = self._add("<BOS>")
        self.EOS = self._add("<EOS>")
        self.MASK = self._add("<MASK>")  # for MLM

        # KG relation tokens (used by Model D as actual tokens in the sequence)
        self.kg_relations = {}
        for rel in KG_RELATIONS:
            self.kg_relations[rel] = self._add(f"<{rel}>")

    def _add(self, token):
        if token not in self.char2idx:
            idx = self.size
            self.char2idx[token] = idx
            self.idx2char[idx] = token
            self.size += 1
        return self.char2idx[token]

    def build_from_sentences(self, sentences):
        for s in sentences:
            for ch in s:
                self._add(ch)

    def encode_sentence(self, sentence):
        return [self.char2idx[ch] for ch in sentence]

    def encode_kg_triple(self, head, rel, tail):
        head_tokens = [self.char2idx[ch] for ch in head]
        rel_token = self.kg_relations[rel]
        tail_tokens = [self.char2idx[ch] for ch in tail]
        return {
            "head": head_tokens,
            "rel": rel,
            "rel_token": rel_token,
            "tail": tail_tokens,
        }

    def decode(self, indices):
        return "".join(self.idx2char.get(i, "?") for i in indices)

    def encode_entity(self, entity):
        return [self.char2idx[ch] for ch in entity]


# ============================================================================
# Dataset Classes
# ============================================================================

class TextDataset:
    """Dataset for text next-token prediction. Samples individual sentences."""

    def __init__(self, sentences, vocab, block_size):
        self.vocab = vocab
        self.block_size = block_size

        # Store each sentence as a separate encoded sequence
        self.encoded = []
        for s in sentences:
            tokens = vocab.encode_sentence(s)  # [BOS, ...chars..., EOS]
            self.encoded.append(torch.tensor(tokens, dtype=torch.long))

        self.data = torch.cat(self.encoded)  # for backward compat (len reporting)

    def get_batch(self, batch_size, device):
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        max_len = max(len(s) for s in batch)
        # x is input (all tokens except last), y is target (all tokens except first)
        # Pad with PAD token, target padded with -100 (ignored in loss)
        x = torch.full((batch_size, max_len - 1), self.vocab.PAD, dtype=torch.long)
        y = torch.full((batch_size, max_len - 1), -100, dtype=torch.long)

        for i, seq in enumerate(batch):
            seq_len = len(seq) - 1  # input length = full length - 1
            x[i, :seq_len] = seq[:-1]
            y[i, :seq_len] = seq[1:]

        return x.to(device), y.to(device)


class KGDataset:
    """Dataset for KG training with MLM or tail-prediction."""

    def __init__(self, triples, vocab, device):
        self.triples = triples
        self.vocab = vocab
        self.device = device

        self.encoded = []
        for head, rel, tail in triples:
            enc = vocab.encode_kg_triple(head, rel, tail)
            self.encoded.append(enc)

    def get_mlm_batch_flat(self, batch_size, device, mask_prob=0.15):
        """Get a batch for flat KG models (D/D'): relation is a token in the sequence.

        Returns:
            tokens: (B, T) token ids including relation token
            targets: (B, T) original token ids (-100 for non-masked)
            rel_names: list of relation name strings
        """
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        # Build flat sequences: head_chars + [rel_token] + tail_chars
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

            # MLM: mask random positions (not PAD)
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
        """Get a batch for Model A/A': HEAD + REL + TAIL slots, all as tokens.

        Sequence: head_chars + [rel_token] + tail_chars
        Each token gets a slot assignment for angle computation.

        Returns:
            tokens: (B, T) token ids including relation token
            targets: (B, T) original token ids (-100 for non-masked)
            head_lens: list of int, length of head part for each sample
            rel_names: list of relation name strings
        """
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
        """Get a batch for native KG models (E/E'): only character tokens, no relation token.

        Returns:
            char_tokens: (B, T) character token ids (head + tail, no relation token)
            targets: (B, T) original token ids (-100 for non-masked)
            head_lens: list of int, length of head part for each sample
            rel_names: list of relation name strings
        """
        indices = torch.randint(0, len(self.encoded), (batch_size,))
        batch = [self.encoded[i] for i in indices]

        # Build sequences: head_chars + tail_chars (no relation token)
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

            # MLM masking
            mask = torch.rand(len(seq)) < mask_prob
            if mask.sum() == 0:
                mask[torch.randint(0, len(seq), (1,))] = True

            for j in range(len(seq)):
                if mask[j]:
                    targets[i, j] = seq_t[j]
                    char_tokens[i, j] = self.vocab.MASK

        rel_names = [batch[i]["rel"] for i in range(batch_size)]
        return char_tokens.to(device), targets.to(device), head_lens, rel_names


# ============================================================================
# Rotation Utilities
# ============================================================================

def apply_rotation(x, angles):
    """Apply 2D rotation matrices to pairs of dimensions.

    Args:
        x: (B, T, C) tensor
        angles: (B, T, C//2) tensor of angles

    Returns:
        rotated x: (B, T, C) tensor
    """
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
        """
        Args:
            x: (B, T, C) input embeddings
            angles: (B, T, C//2) cumulative angles for rotation
            causal: whether to apply causal mask
            pad_mask: (B, T) boolean mask, True for valid positions
        """
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
            # pad_mask: (B, T), True=valid, False=pad
            # Mask out attention TO pad positions
            pad_mask_k = pad_mask.unsqueeze(1)  # (B, 1, T)
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
    """RoPE + learned slot angles. Same transformer for text and KG.

    Text: slot angles = 0 -> pure RoPE. Causal mask, next-token prediction.
    KG: HEAD/REL/TAIL slots with learned slot angles. Positions reset per slot.
        Bidirectional attention, MLM on all character tokens.

    rotate_v=False -> Model A (commutative)
    rotate_v=True  -> Model A' (operator-based)
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, n_embed)

        # Learned slot angle vectors: HEAD=0, REL=1, TAIL=2
        self.slot_angles = nn.Parameter(torch.randn(3, n_embed // 2) * 0.1)

        # RoPE base frequencies
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def _rope_angles(self, T, device):
        """Standard RoPE angles: position * base_freq. Shape: (1, T, C//2)."""
        positions = torch.arange(T, device=device, dtype=torch.float)
        angles = torch.outer(positions, self.base_freq)  # (T, C//2)
        return angles.unsqueeze(0)  # (1, T, C//2)

    def _kg_angles(self, head_lens, seq_len, batch_size, rel_names, device):
        """KG angles: position_in_slot * base_freq + slot_angle.

        Sequence layout: [head_chars] [rel_token] [tail_chars]
        - HEAD slot: positions 0..h_len-1, angle = pos * base_freq + slot_angles[0]
        - REL slot: position 0 (single token), angle = 0 * base_freq + slot_angles[1]
        - TAIL slot: positions 0..t_len-1, angle = pos * base_freq + slot_angles[2]

        Args:
            head_lens: list of int, head length per sample
            seq_len: total sequence length (head + 1 rel + tail, padded)
            batch_size: batch size
            rel_names: list of relation name strings
        """
        angles = torch.zeros(batch_size, seq_len, self.n_embed // 2, device=device)

        for i in range(batch_size):
            h_len = head_lens[i]
            rel_pos = h_len        # position of the relation token
            tail_start = h_len + 1  # first tail character position

            # HEAD slot: positions 0..h_len-1
            for j in range(h_len):
                angles[i, j] = j * self.base_freq + self.slot_angles[0]

            # REL slot: single token at position 0 in its slot
            angles[i, rel_pos] = 0 * self.base_freq + self.slot_angles[1]

            # TAIL slot: positions 0..t_len-1
            for j in range(seq_len - tail_start):
                if tail_start + j < seq_len:
                    angles[i, tail_start + j] = j * self.base_freq + self.slot_angles[2]

        return angles

    def forward_text(self, idx, targets=None):
        """Text mode: pure RoPE (slot angles = 0), causal, next-token prediction."""
        B, T = idx.shape
        pad_mask = (idx != 0)  # PAD = 0
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

    def forward_kg(self, tokens, targets, head_lens, rel_names):
        """KG mode: slot-based RoPE, bidirectional, MLM.

        Sequence: head_chars + [rel_token] + tail_chars (all are real tokens).

        Args:
            tokens: (B, T) token ids (head + rel + tail, masked for MLM)
            targets: (B, T) original tokens, -100 for non-masked
            head_lens: list of head lengths (not counting rel token)
            rel_names: list of relation strings
        """
        B, T = tokens.shape
        pad_mask = (tokens != 0)  # PAD = 0

        x = self.token_embedding(tokens)
        angles = self._kg_angles(head_lens, T, B, rel_names, tokens.device)

        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)

        logits = self.lm_head(x)

        # MLM loss on masked positions only
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
    """Standard RoPE transformer. KG facts converted to text. Causal, next-token prediction.

    rotate_v=False -> Model B (commutative, standard RoPE on Q,K only)
    rotate_v=True  -> Model B' (operator-based, rotate Q,K,V + inverse on output)
    """

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
    """Journey transformer with per-token learned angle embeddings and cumsum.
    KG facts converted to text. Causal, next-token prediction.

    rotate_v=False -> Model C (commutative, rotate Q,K only)
    rotate_v=True  -> Model C' (operator-based, rotate Q,K,V + inverse on output)
    """

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

        # Right-cumsum: flip -> cumsum -> flip
        # KEY token's angle IS included, QUERY token's angle is NOT
        raw_angles = self.angle_embedding(idx)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))

        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)

        logits = self.lm_head(x)

        if targets is None:
            return logits, None

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
    """Journey transformer with per-token cumsum. KG triple laid out as flat sequence:
    head_chars + [rel_token] + tail_chars

    The relation (e.g. son_of) is a TOKEN with its own token embedding AND angle embedding.
    It produces Q/K/V, receives attention, IS predicted by MLM.

    Text: causal mask, next-token prediction (same as Model C).
    KG: bidirectional attention, MLM on all tokens (including relation token).

    rotate_v=False -> Model D (commutative)
    rotate_v=True  -> Model D' (operator-based)
    """

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
        """Right-cumsum angles from token angle embeddings."""
        raw_angles = self.angle_embedding(idx)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        return angles

    def forward_text(self, idx, targets=None):
        """Text mode: causal, next-token prediction."""
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.expander(self.token_embedding(idx))
        angles = self._cumsum_angles(idx)

        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)

        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg(self, tokens, targets):
        """KG mode: bidirectional, MLM.

        Args:
            tokens: (B, T) flat KG tokens (head_chars + rel_token + tail_chars), with MLM masking
            targets: (B, T) original tokens, -100 for non-masked
        """
        B, T = tokens.shape
        pad_mask = (tokens != 0)  # PAD = 0

        x = self.expander(self.token_embedding(tokens))
        angles = self._cumsum_angles(tokens)

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
    """Fixed-angle RoPE transformer with flat KG path. Same dual text/KG paths as
    Model D, but uses fixed RoPE angles instead of learned cumsum angles.

    This isolates the effect of learned vs fixed positional encoding while keeping
    the native KG path (relation as token, bidirectional MLM).

    Text: causal mask, next-token prediction (same as Model B).
    KG: bidirectional attention, MLM on all tokens (same format as Model D).

    rotate_v=False -> Model F (commutative, standard RoPE on Q,K only)
    rotate_v=True  -> Model F' (operator-based, rotate Q,K,V + inverse on output)
    """

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
        """Text mode: causal, next-token prediction."""
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

    def forward_kg(self, tokens, targets):
        """KG mode: bidirectional, MLM.

        Args:
            tokens: (B, T) flat KG tokens (head_chars + rel_token + tail_chars), with MLM masking
            targets: (B, T) original tokens, -100 for non-masked
        """
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
        logits, _ = self.forward_text(idx)
        return logits


# ============================================================================
# Model E/E': Per-Token Cumsum + Relation Operator, Native KG
# ============================================================================

class ModelE(nn.Module):
    """Journey transformer with per-token cumsum. The relation is NOT a token --
    it has a learned angle vector but NO token embedding. It does NOT produce Q/K/V,
    does NOT receive attention, is NOT predicted by MLM.

    The relation angle is inserted into the cumsum sequence between HEAD and TAIL chars,
    acting as a "gap" that separates HEAD from TAIL in angle space.

    Text: causal mask, next-token prediction (same as Model C).
    KG: bidirectional attention on character tokens only, MLM.
        Cumsum includes relation angle between head and tail chars.

    rotate_v=False -> Model E (commutative)
    rotate_v=True  -> Model E' (operator-based)
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.angle_embedding = nn.Embedding(vocab_size, n_embed // 2)
        self.expander = nn.Linear(n_embed // 2, n_embed)

        # Learned angle vectors for each relation type (shared across layers)
        self.relation_angles = nn.Parameter(torch.randn(n_relations, n_embed // 2) * 0.1)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # Mapping from relation name to index
        self.rel_to_idx = {rel: i for i, rel in enumerate(KG_RELATIONS)}

    def _cumsum_angles_text(self, idx):
        """Right-cumsum angles for text (same as Model C)."""
        raw_angles = self.angle_embedding(idx)
        angles = torch.flip(raw_angles, dims=(1,))
        angles = torch.cumsum(angles, dim=1)
        angles = torch.flip(angles, dims=(1,))
        return angles

    def _cumsum_angles_kg(self, char_tokens, head_lens, rel_names, device):
        """Cumsum angles for KG with relation angle inserted between head and tail.

        The cumsum sequence has len(chars)+1 entries (one extra for the relation angle),
        but attention operates on len(chars) tokens only.

        Cumsum entries: [theta_h0, theta_h1, ..., theta_{h_last}, theta_rel, theta_t0, theta_t1, ...]
        The right-cumsum produces len(chars)+1 cumulative angle entries.
        We then drop the relation entry to get len(chars) angles for the actual tokens.

        Specifically:
        - Positions 0..h_len-1: head char angles
        - Position h_len: relation angle (this is the "gap")
        - Positions h_len+1..h_len+t_len: tail char angles
        After cumsum, we extract indices [0..h_len-1] + [h_len+1..h_len+t_len] for the char tokens.
        """
        B, T = char_tokens.shape

        # Get per-token raw angles
        raw_char_angles = self.angle_embedding(char_tokens)  # (B, T, C//2)

        # Build extended angle sequence with relation angle inserted
        # Extended length = T + 1 (one extra for relation)
        ext_angles = torch.zeros(B, T + 1, self.n_embed // 2, device=device)

        for i in range(B):
            h_len = head_lens[i]
            rel_idx = self.rel_to_idx[rel_names[i]]

            # Head char angles: positions 0..h_len-1
            ext_angles[i, :h_len] = raw_char_angles[i, :h_len]

            # Relation angle: position h_len
            ext_angles[i, h_len] = self.relation_angles[rel_idx]

            # Tail char angles: positions h_len+1..
            t_len = T - h_len
            ext_angles[i, h_len + 1:h_len + 1 + t_len] = raw_char_angles[i, h_len:h_len + t_len]

        # Right-cumsum on extended sequence
        ext_cumsum = torch.flip(ext_angles, dims=(1,))
        ext_cumsum = torch.cumsum(ext_cumsum, dim=1)
        ext_cumsum = torch.flip(ext_cumsum, dims=(1,))

        # Extract angles for actual char positions (skip the relation position)
        angles = torch.zeros(B, T, self.n_embed // 2, device=device)
        for i in range(B):
            h_len = head_lens[i]
            t_len = T - h_len
            # Head positions
            angles[i, :h_len] = ext_cumsum[i, :h_len]
            # Tail positions (skip the relation entry at index h_len)
            angles[i, h_len:h_len + t_len] = ext_cumsum[i, h_len + 1:h_len + 1 + t_len]

        return angles

    def forward_text(self, idx, targets=None):
        """Text mode: causal, next-token prediction."""
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.expander(self.token_embedding(idx))
        angles = self._cumsum_angles_text(idx)

        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)

        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg(self, char_tokens, targets, head_lens, rel_names):
        """KG mode: bidirectional on char tokens only, MLM.
        Relation angle is only in the cumsum, not in the attention sequence.

        Args:
            char_tokens: (B, T) character tokens (head + tail, no relation token), MLM masked
            targets: (B, T) original tokens, -100 for non-masked
            head_lens: list of head lengths
            rel_names: list of relation strings
        """
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)  # PAD = 0

        x = self.expander(self.token_embedding(char_tokens))
        angles = self._cumsum_angles_kg(char_tokens, head_lens, rel_names, char_tokens.device)

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
# Model G/G': Relation-Dependent Slot Angles, Native KG
# ============================================================================

class ModelG(nn.Module):
    """RoPE + relation-dependent learned slot angles. Same transformer for text and KG.

    Like Model A, but each relation gets its own set of slot angle vectors
    (theta_h[rel], theta_rel[rel], theta_t[rel]), allowing the model to learn
    different geometric structures per relation (e.g. son_of vs grandfather_of).

    Text: slot angles = 0 -> pure RoPE. Causal mask, next-token prediction.
    KG: HEAD/REL/TAIL slots with relation-specific learned slot angles.
        Positions reset per slot. Bidirectional attention, MLM on all character tokens.

    rotate_v=False -> Model G (commutative)
    rotate_v=True  -> Model G' (operator-based)
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, n_embed)

        # Relation-dependent slot angle vectors: [n_relations, 3, n_embed//2]
        # Slot 0 = HEAD, 1 = REL, 2 = TAIL
        self.slot_angles = nn.Parameter(torch.randn(n_relations, 3, n_embed // 2) * 0.1)

        # RoPE base frequencies
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])

        self.lm_head = nn.Linear(n_embed, vocab_size)

        # Mapping from relation name to index
        self.rel_to_idx = {rel: i for i, rel in enumerate(KG_RELATIONS)}

    def _rope_angles(self, T, device):
        """Standard RoPE angles: position * base_freq. Shape: (1, T, C//2)."""
        positions = torch.arange(T, device=device, dtype=torch.float)
        angles = torch.outer(positions, self.base_freq)  # (T, C//2)
        return angles.unsqueeze(0)  # (1, T, C//2)

    def _kg_angles(self, head_lens, seq_len, batch_size, rel_names, device):
        """KG angles: position_in_slot * base_freq + slot_angle[rel].

        Like ModelA._kg_angles but slot angles are relation-dependent.

        Sequence layout: [head_chars] [rel_token] [tail_chars]
        - HEAD slot: positions 0..h_len-1, angle = pos * base_freq + slot_angles[rel, 0]
        - REL slot: position 0 (single token), angle = 0 * base_freq + slot_angles[rel, 1]
        - TAIL slot: positions 0..t_len-1, angle = pos * base_freq + slot_angles[rel, 2]
        """
        angles = torch.zeros(batch_size, seq_len, self.n_embed // 2, device=device)

        for i in range(batch_size):
            h_len = head_lens[i]
            rel_pos = h_len        # position of the relation token
            tail_start = h_len + 1  # first tail character position
            rel_idx = self.rel_to_idx[rel_names[i]]

            # HEAD slot: positions 0..h_len-1
            for j in range(h_len):
                angles[i, j] = j * self.base_freq + self.slot_angles[rel_idx, 0]

            # REL slot: single token at position 0 in its slot
            angles[i, rel_pos] = 0 * self.base_freq + self.slot_angles[rel_idx, 1]

            # TAIL slot: positions 0..t_len-1
            for j in range(seq_len - tail_start):
                if tail_start + j < seq_len:
                    angles[i, tail_start + j] = j * self.base_freq + self.slot_angles[rel_idx, 2]

        return angles

    def forward_text(self, idx, targets=None):
        """Text mode: pure RoPE (slot angles = 0), causal, next-token prediction."""
        B, T = idx.shape
        pad_mask = (idx != 0)  # PAD = 0
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

    def forward_kg(self, tokens, targets, head_lens, rel_names):
        """KG mode: slot-based RoPE with relation-dependent angles, bidirectional, MLM."""
        B, T = tokens.shape
        pad_mask = (tokens != 0)  # PAD = 0

        x = self.token_embedding(tokens)
        angles = self._kg_angles(head_lens, T, B, rel_names, tokens.device)

        for block in self.blocks:
            x = block(x, angles, causal=False, pad_mask=pad_mask)

        logits = self.lm_head(x)

        # MLM loss on masked positions only
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
    """Fixed-angle cumsum transformer with relation angle operator. Same native KG
    path as Model E, but uses fixed base frequencies instead of learned per-token
    angle embeddings.

    This is to E what F is to D: replace learned cumsum angles with fixed ones,
    while keeping the relation angle operator mechanism.

    The relation angle is still learned and inserted into the cumsum between HEAD
    and TAIL chars, acting as a "gap" that separates HEAD from TAIL in angle space.

    Text: causal mask, next-token prediction.
        Cumsum of fixed base_freq angles (no learned per-token angles).
    KG: bidirectional attention on character tokens only, MLM.
        Cumsum includes learned relation angle between head and tail chars.

    rotate_v=False -> Model H (commutative)
    rotate_v=True  -> Model H' (operator-based)
    """

    def __init__(self, vocab_size, n_embed, n_layers, block_size, n_relations=8,
                 dropout=0.2, rotate_v=False):
        super().__init__()
        self.n_embed = n_embed
        self.token_embedding = nn.Embedding(vocab_size, n_embed)

        # Fixed base frequencies (no learned per-token angles)
        self.register_buffer('base_freq',
            1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2))))

        # Learned angle vectors for each relation type (shared across layers)
        self.relation_angles = nn.Parameter(torch.randn(n_relations, n_embed // 2) * 0.1)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, block_size, dropout, rotate_v)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # Mapping from relation name to index
        self.rel_to_idx = {rel: i for i, rel in enumerate(KG_RELATIONS)}

    def _cumsum_angles_text(self, T, B, device):
        """Right-cumsum of fixed base_freq angles for text."""
        raw_angles = self.base_freq.unsqueeze(0).expand(T, -1)  # (T, C//2)
        angles = torch.flip(raw_angles, dims=(0,))
        angles = torch.cumsum(angles, dim=0)
        angles = torch.flip(angles, dims=(0,))
        return angles.unsqueeze(0).expand(B, -1, -1)

    def _cumsum_angles_kg(self, T, head_lens, rel_names, device):
        """Cumsum angles for KG with relation angle inserted between head and tail.

        Same structure as ModelE._cumsum_angles_kg but uses fixed base_freq instead
        of learned angle_embedding for per-token angles.

        Cumsum entries: [freq, freq, ..., rel_angle, freq, freq, ...]
        The relation angle is inserted at position h_len between HEAD and TAIL.
        After cumsum, we drop the relation entry to get T angles for the char tokens.
        """
        B = len(head_lens)

        # Build extended angle sequence with relation angle inserted
        # Extended length = T + 1 (one extra for relation)
        ext_angles = torch.zeros(B, T + 1, self.n_embed // 2, device=device)

        for i in range(B):
            h_len = head_lens[i]

            # Head positions: base_freq repeated h_len times
            ext_angles[i, :h_len] = self.base_freq.unsqueeze(0).expand(h_len, -1)

            # Relation angle: position h_len
            rel_idx = self.rel_to_idx[rel_names[i]]
            ext_angles[i, h_len] = self.relation_angles[rel_idx]

            # Tail positions: base_freq repeated t_len times
            t_len = T - h_len
            ext_angles[i, h_len + 1:h_len + 1 + t_len] = self.base_freq.unsqueeze(0).expand(t_len, -1)

        # Right-cumsum on extended sequence
        ext_cumsum = torch.flip(ext_angles, dims=(1,))
        ext_cumsum = torch.cumsum(ext_cumsum, dim=1)
        ext_cumsum = torch.flip(ext_cumsum, dims=(1,))

        # Extract angles for actual char positions (skip the relation position)
        angles = torch.zeros(B, T, self.n_embed // 2, device=device)
        for i in range(B):
            h_len = head_lens[i]
            t_len = T - h_len
            # Head positions
            angles[i, :h_len] = ext_cumsum[i, :h_len]
            # Tail positions (skip the relation entry at index h_len)
            angles[i, h_len:h_len + t_len] = ext_cumsum[i, h_len + 1:h_len + 1 + t_len]

        return angles

    def forward_text(self, idx, targets=None):
        """Text mode: causal, next-token prediction."""
        B, T = idx.shape
        pad_mask = (idx != 0)
        x = self.token_embedding(idx)
        angles = self._cumsum_angles_text(T, B, idx.device)

        for block in self.blocks:
            x = block(x, angles, causal=True, pad_mask=pad_mask)

        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                               ignore_index=-100)
        return logits, loss

    def forward_kg(self, char_tokens, targets, head_lens, rel_names):
        """KG mode: bidirectional on char tokens only, MLM.
        Relation angle is only in the cumsum, not in the attention sequence.

        Args:
            char_tokens: (B, T) character tokens (head + tail, no relation token), MLM masked
            targets: (B, T) original tokens, -100 for non-masked
            head_lens: list of head lengths
            rel_names: list of relation strings
        """
        B, T = char_tokens.shape
        pad_mask = (char_tokens != 0)  # PAD = 0

        x = self.token_embedding(char_tokens)
        angles = self._cumsum_angles_kg(T, head_lens, rel_names, char_tokens.device)

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
# Data Preparation
# ============================================================================

def prepare_data(generous_linearization=True, expanded_names=False):
    """Prepare train data for all models.

    Returns:
        vocab, text_dataset_base (for A/D/E), text_dataset_linearized (for B/C),
        kg_dataset, eval_prompts
    """
    (teaching, transfer, generalization,
     kg_excl_mem, kg_excl_gen,
     text_excl_mem, text_excl_gen, geo_facts) = generate_world(expanded_names=expanded_names)

    # ---- Base text (all models see this) ----
    # KG-exclusive chains get NO text at all
    # Text-exclusive chains get text but NO KG
    base_teaching = generate_text_sentences(teaching, include_derived=True, all_variations=True)
    base_transfer = generate_text_sentences(transfer, include_derived=False, all_variations=True)
    base_generalization = generate_text_sentences(generalization, include_derived=False, all_variations=True)
    base_geo = generate_geo_sentences(geo_facts, all_variations=True)
    # Text-exclusive: text only, never KG
    text_excl_mem_text = generate_text_sentences(text_excl_mem, include_derived=True, all_variations=True)
    text_excl_gen_text = generate_text_sentences(text_excl_gen, include_derived=False, all_variations=True)
    base_text = (base_teaching + base_transfer + base_generalization + base_geo
                 + text_excl_mem_text + text_excl_gen_text)

    # ---- KG triples (for A/D/E) ----
    # Text-exclusive chains get NO KG triples
    kg_teaching = generate_kg_triples(teaching, include_derived=True)
    kg_transfer = generate_kg_triples(transfer, include_derived=True)
    kg_generalization = generate_kg_triples(generalization, include_derived=False)
    kg_geo = generate_geo_kg_triples(geo_facts)
    # KG-exclusive chains: only in KG, never in text
    kg_excl_mem_triples = generate_kg_triples(kg_excl_mem, include_derived=True)
    kg_excl_gen_triples = generate_kg_triples(kg_excl_gen, include_derived=False)
    all_kg_triples = (kg_teaching + kg_transfer + kg_generalization + kg_geo
                      + kg_excl_mem_triples + kg_excl_gen_triples)

    # ---- Extra linearized text for B/C (derived facts from transfer chains) ----
    extra_transfer_text = generate_text_sentences(
        transfer, include_derived=True, all_variations=generous_linearization
    )
    extra_derived_only = [s for s in extra_transfer_text if s not in base_transfer]
    linearized_extra_text = extra_derived_only

    # ---- Build vocabulary ----
    vocab = Vocabulary()
    all_sentences = base_text + linearized_extra_text
    vocab.build_from_sentences(all_sentences)

    # Add chars from KG entities (including KG-exclusive names)
    for head, rel, tail in all_kg_triples:
        vocab.build_from_sentences([head, tail])

    # ---- Build datasets ----
    text_dataset_base = TextDataset(base_text, vocab, cfg.block_size)
    text_linearized = base_text + linearized_extra_text
    text_dataset_linearized = TextDataset(text_linearized, vocab, cfg.block_size)
    kg_dataset = KGDataset(all_kg_triples, vocab, cfg.device)

    # ---- Build evaluation prompts ----
    eval_prompts = build_eval_prompts(
        teaching, transfer, generalization,
        kg_excl_mem, kg_excl_gen,
        text_excl_mem, text_excl_gen, vocab)

    # ---- Build KG evaluation prompts ----
    kg_eval_prompts = build_kg_eval_prompts(
        teaching, transfer, generalization,
        kg_excl_mem, kg_excl_gen,
        text_excl_mem, text_excl_gen)

    return (vocab, text_dataset_base, text_dataset_linearized, kg_dataset,
            eval_prompts, kg_eval_prompts)


def build_eval_prompts(teaching, transfer, generalization,
                       kg_excl_mem, kg_excl_gen,
                       text_excl_mem, text_excl_gen, vocab):
    prompts = []

    def add_prompts(chains, tier):
        for son, father, grandfather in chains:
            prompts.append({
                "tier": tier,
                "prompt": f"{son} is the son of ",
                "target": father,
                "relation": "son_of",
            })
            prompts.append({
                "tier": tier,
                "prompt": f"{father} is the father of ",
                "target": son,
                "relation": "father_of",
            })
            prompts.append({
                "tier": tier,
                "prompt": f"{father} is the son of ",
                "target": grandfather,
                "relation": "son_of",
            })
            prompts.append({
                "tier": tier,
                "prompt": f"{grandfather} is the father of ",
                "target": father,
                "relation": "father_of",
            })
            prompts.append({
                "tier": tier,
                "prompt": f"{son} is the grandson of ",
                "target": grandfather,
                "relation": "grandson_of",
            })
            prompts.append({
                "tier": tier,
                "prompt": f"{grandfather} is the grandfather of ",
                "target": son,
                "relation": "grandfather_of",
            })

    add_prompts(teaching, "memorization")
    add_prompts(transfer, "transfer")
    add_prompts(generalization, "generalization")
    add_prompts(kg_excl_mem, "kg_exclusive_memorization")
    add_prompts(kg_excl_gen, "kg_exclusive_generalization")
    add_prompts(text_excl_mem, "text_exclusive_memorization")
    add_prompts(text_excl_gen, "text_exclusive_generalization")

    for p in prompts:
        p["prompt_tokens"] = vocab.encode_sentence(p["prompt"][:-1])
        p["prompt_tokens"].append(vocab.char2idx.get(" ", vocab.PAD))
        p["target_tokens"] = [vocab.char2idx.get(ch, vocab.PAD) for ch in p["target"]]

    return prompts


def build_kg_eval_prompts(teaching, transfer, generalization,
                          kg_excl_mem, kg_excl_gen,
                          text_excl_mem, text_excl_gen):
    """Build KG evaluation prompts: (head, relation, tail) for each tier.

    Returns dict: tier -> list of {head, rel, tail, relation} dicts.
    """
    prompts = []

    def add_kg_prompts(chains, tier):
        for son, father, grandfather in chains:
            # Base relations
            prompts.append({"tier": tier, "head": son, "rel": "son_of", "tail": father, "relation": "son_of"})
            prompts.append({"tier": tier, "head": father, "rel": "father_of", "tail": son, "relation": "father_of"})
            prompts.append({"tier": tier, "head": father, "rel": "son_of", "tail": grandfather, "relation": "son_of"})
            prompts.append({"tier": tier, "head": grandfather, "rel": "father_of", "tail": father, "relation": "father_of"})
            # Derived relations
            prompts.append({"tier": tier, "head": son, "rel": "grandson_of", "tail": grandfather, "relation": "grandson_of"})
            prompts.append({"tier": tier, "head": grandfather, "rel": "grandfather_of", "tail": son, "relation": "grandfather_of"})

    add_kg_prompts(teaching, "memorization")
    add_kg_prompts(transfer, "transfer")
    add_kg_prompts(generalization, "generalization")
    add_kg_prompts(kg_excl_mem, "kg_exclusive_memorization")
    add_kg_prompts(kg_excl_gen, "kg_exclusive_generalization")
    add_kg_prompts(text_excl_mem, "text_exclusive_memorization")
    add_kg_prompts(text_excl_gen, "text_exclusive_generalization")

    return prompts


def evaluate_model_kg(model, kg_eval_prompts, vocab, config, model_name="?", model_type="A"):
    """Evaluate a KG model (A/D/E) on KG completion: given head+relation, predict tail.

    Uses pseudo-perplexity (mask one tail position at a time) and hit@1/hit@5.
    Also reports simultaneous-mask hit@1 (mask all tail positions at once).

    Args:
        model: the trained model
        kg_eval_prompts: list of dicts with head, rel, tail, tier, relation
        vocab: Vocabulary
        config: Config
        model_name: string for display
        model_type: "A", "D", "E", or "F"
    """
    model.eval()
    model.to(config.device)

    results = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for p in kg_eval_prompts:
            tier = p["tier"]
            relation = p["relation"]
            head_name = p["head"]
            rel_name = p["rel"]
            tail_name = p["tail"]

            head_tokens = vocab.encode_entity(head_name)
            tail_tokens = vocab.encode_entity(tail_name)
            head_len = len(head_tokens)
            tail_len = len(tail_tokens)

            # --- Pseudo-perplexity: mask one tail position at a time ---
            total_log_prob = 0.0
            log_prob_first = None
            log_prob_last = None
            all_in_top5 = True

            for t_idx in range(tail_len):
                tokens, targets, head_lens, rel_names = _build_kg_eval_batch(
                    head_tokens, rel_name, tail_tokens, [head_len + t_idx],
                    vocab, model_type)
                tokens = tokens.to(config.device)
                targets = targets.to(config.device)

                logits = _forward_kg_eval(model, tokens, targets, head_lens, rel_names, model_type)

                # Find the mask position in the output
                mask_pos = head_len + t_idx if model_type == "E" else head_len + 1 + t_idx
                step_logits = logits[0, mask_pos, :]
                log_probs = F.log_softmax(step_logits, dim=0)
                true_token = tail_tokens[t_idx]
                total_log_prob += log_probs[true_token].item()
                if t_idx == 0:
                    log_prob_first = log_probs[true_token].item()
                log_prob_last = log_probs[true_token].item()

                top5 = torch.topk(step_logits, k=min(5, step_logits.shape[0])).indices.tolist()
                if true_token not in top5:
                    all_in_top5 = False

            ppl = np.exp(-total_log_prob / max(tail_len, 1))
            first_char_ppl = np.exp(-log_prob_first) if log_prob_first is not None else ppl
            last_char_ppl = np.exp(-log_prob_last) if log_prob_last is not None else ppl
            hit5 = 1 if all_in_top5 else 0

            # --- Simultaneous mask: mask ALL tail positions at once ---
            all_mask_positions = list(range(head_len, head_len + tail_len))
            tokens, targets, head_lens, rel_names = _build_kg_eval_batch(
                head_tokens, rel_name, tail_tokens, all_mask_positions,
                vocab, model_type)
            tokens = tokens.to(config.device)
            targets = targets.to(config.device)

            logits = _forward_kg_eval(model, tokens, targets, head_lens, rel_names, model_type)

            hit1 = 1
            for t_idx in range(tail_len):
                mask_pos = head_len + t_idx if model_type == "E" else head_len + 1 + t_idx
                pred = torch.argmax(logits[0, mask_pos, :]).item()
                if pred != tail_tokens[t_idx]:
                    hit1 = 0
                    break

            results[tier][relation].append({
                "hit1": hit1,
                "hit5": hit5,
                "ppl": ppl,
                "first_char_ppl": first_char_ppl,
                "last_char_ppl": last_char_ppl,
                "head": head_name,
                "rel": rel_name,
                "tail": tail_name,
            })

    # --- Tier-level summary ---
    summary = {}
    for tier in results:
        tier_results = {"hit1": [], "hit5": [], "ppl": [], "first_char_ppl": [], "last_char_ppl": []}
        for rel in results[tier]:
            for r in results[tier][rel]:
                tier_results["hit1"].append(r["hit1"])
                tier_results["hit5"].append(r["hit5"])
                tier_results["ppl"].append(r["ppl"])
                tier_results["first_char_ppl"].append(r["first_char_ppl"])
                tier_results["last_char_ppl"].append(r["last_char_ppl"])

        summary[tier] = {
            "hit1": np.mean(tier_results["hit1"]),
            "hit5": np.mean(tier_results["hit5"]),
            "ppl": np.exp(np.mean(np.log(tier_results["ppl"]))),
            "first_char_ppl": np.exp(np.mean(np.log(tier_results["first_char_ppl"]))),
            "last_char_ppl": np.exp(np.mean(np.log(tier_results["last_char_ppl"]))),
            "n": len(tier_results["hit1"]),
        }

    # --- Per-relation summary ---
    relation_summary = {}
    for tier in results:
        relation_summary[tier] = {}
        for rel in results[tier]:
            rel_data = results[tier][rel]
            if len(rel_data) == 0:
                continue
            relation_summary[tier][rel] = {
                "hit1": np.mean([r["hit1"] for r in rel_data]),
                "hit5": np.mean([r["hit5"] for r in rel_data]),
                "ppl": np.exp(np.mean([np.log(r["ppl"]) for r in rel_data])),
                "n": len(rel_data),
            }

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"  KG Evaluation: {model_name}")
    print(f"{'='*60}")
    for tier in ALL_TIERS:
        if tier in summary:
            s = summary[tier]
            print(f"  {tier:>35s}: hit@1={s['hit1']:.3f}  hit@5={s['hit5']:.3f}  "
                  f"ppl={s['ppl']:.2f}  fc_ppl={s['first_char_ppl']:.2f}  lc_ppl={s['last_char_ppl']:.2f}  (n={s['n']})")

    print(f"\n  Per-relation breakout (KG):")
    print(f"  {'Tier':<35s} {'Relation':<20s} {'hit@1':>6s} {'hit@5':>6s} {'ppl':>8s} {'n':>4s}")
    print(f"  {'-'*80}")
    for tier in ALL_TIERS:
        if tier not in relation_summary:
            continue
        for rel in sorted(relation_summary[tier].keys()):
            rs = relation_summary[tier][rel]
            print(f"  {tier:<35s} {rel:<20s} {rs['hit1']:>6.3f} {rs['hit5']:>6.3f} "
                  f"{rs['ppl']:>8.2f} {rs['n']:>4d}")

    model.train()
    return summary, relation_summary, results


def _build_kg_eval_batch(head_tokens, rel_name, tail_tokens, mask_positions,
                         vocab, model_type):
    """Build a single-example KG input batch with specified mask positions.

    mask_positions: list of indices into the TAIL part (0-based from tail start).
                    These are offsets relative to the start of the full sequence's
                    tail section (i.e., position head_len+i means mask tail[i]).

    Returns (tokens, targets, head_lens, rel_names) ready for forward_kg.
    """
    head_len = len(head_tokens)
    tail_len = len(tail_tokens)

    if model_type == "E":
        # Model E: char_tokens = head + tail (no relation token in sequence)
        seq = list(head_tokens) + list(tail_tokens)
        seq_len = len(seq)
        tokens = torch.full((1, seq_len), vocab.PAD, dtype=torch.long)
        targets = torch.full((1, seq_len), -100, dtype=torch.long)
        tokens[0, :seq_len] = torch.tensor(seq, dtype=torch.long)

        for mp in mask_positions:
            # mp is index in tail (0-based), actual position is head_len + mp_offset
            pos = mp  # already absolute from caller
            if pos < seq_len:
                targets[0, pos] = tokens[0, pos].item()
                tokens[0, pos] = vocab.MASK

        return tokens, targets, [head_len], [rel_name]

    else:
        # Model A and D: tokens = head + [rel_token] + tail
        rel_token = vocab.kg_relations[rel_name]
        seq = list(head_tokens) + [rel_token] + list(tail_tokens)
        seq_len = len(seq)
        tokens = torch.full((1, seq_len), vocab.PAD, dtype=torch.long)
        targets = torch.full((1, seq_len), -100, dtype=torch.long)
        tokens[0, :seq_len] = torch.tensor(seq, dtype=torch.long)

        for mp in mask_positions:
            # mp is tail-relative (head_len + i), but in A/D the rel_token shifts tail by 1
            pos = mp + 1  # shift by 1 for the relation token
            if pos < seq_len:
                targets[0, pos] = tokens[0, pos].item()
                tokens[0, pos] = vocab.MASK

        return tokens, targets, [head_len], [rel_name]


def _forward_kg_eval(model, tokens, targets, head_lens, rel_names, model_type):
    """Run forward_kg for the appropriate model type and return logits."""
    if model_type in ("A", "G"):
        logits, _ = model.forward_kg(tokens, targets, head_lens, rel_names)
    elif model_type in ("D", "F"):
        logits, _ = model.forward_kg(tokens, targets)
    elif model_type in ("E", "H"):
        logits, _ = model.forward_kg(tokens, targets, head_lens, rel_names)
    return logits


# ============================================================================
# Training
# ============================================================================

def train_model_text_only(model, text_dataset, config, name="?",
                          resume_optimizer_state=None):
    """Train text-only model (B/B', C/C') with next-token prediction."""
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
                      kg_batch_fn="native", resume_optimizer_state=None):
    """Train mixed text+KG model (A/A', D/D', E/E').

    Each iteration: text batch (causal, NTP) + KG batch (bidir, MLM).

    Args:
        kg_batch_fn: "slotted" for A/A' (get_mlm_batch_slotted),
                     "native" for E/E' (get_mlm_batch_native),
                     "flat" for D/D' (get_mlm_batch_flat)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
    model.to(config.device)
    model.train()

    losses_log = {"text": [], "kg": [], "iter": []}

    for it in tqdm(range(config.max_iters), desc=f"Model {name}"):
        # --- Text batch ---
        x, y = text_dataset.get_batch(config.batch_size, config.device)
        if hasattr(model, 'forward_text'):
            _, text_loss = model.forward_text(x, y)
        else:
            _, text_loss = model(x, y)

        # --- KG batch ---
        if kg_batch_fn == "slotted":
            tokens, targets, head_lens, rel_names = kg_dataset.get_mlm_batch_slotted(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, kg_loss = model.forward_kg(tokens, targets, head_lens, rel_names)
        elif kg_batch_fn == "native":
            char_tokens, targets, head_lens, rel_names = kg_dataset.get_mlm_batch_native(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, kg_loss = model.forward_kg(char_tokens, targets, head_lens, rel_names)
        elif kg_batch_fn == "flat":
            tokens, targets, rel_names = kg_dataset.get_mlm_batch_flat(
                config.batch_size, config.device, config.mlm_mask_prob)
            _, kg_loss = model.forward_kg(tokens, targets)

        # Combined loss
        loss = text_loss + kg_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % config.eval_interval == 0:
            losses_log["iter"].append(it)
            losses_log["text"].append(text_loss.item())
            losses_log["kg"].append(kg_loss.item())
            print(f"  [{name}] iter {it}, text_loss: {text_loss.item():.4f}, "
                  f"kg_loss: {kg_loss.item():.4f}")

    return losses_log, optimizer.state_dict()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, eval_prompts, vocab, config, model_name="?"):
    """Evaluate a model on cloze-style text prompts.

    All models evaluated as text models -- given a text prompt, predict next chars.
    """
    model.eval()
    model.to(config.device)

    results = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for p in eval_prompts:
            tier = p["tier"]
            relation = p["relation"]
            prompt_tokens = p["prompt_tokens"]
            target_tokens = p["target_tokens"]

            if len(prompt_tokens) > config.block_size:
                prompt_tokens = prompt_tokens[-config.block_size:]

            # Autoregressive generation of full target name
            full_correct = True
            total_log_prob = 0.0
            generated_name = []
            current_tokens = prompt_tokens.copy()
            log_probs_first = None
            log_probs_last = None
            for t_idx, t_tok in enumerate(target_tokens):
                x = torch.tensor([current_tokens[-config.block_size:]],
                                 dtype=torch.long, device=config.device)
                logits = model.predict_text(x)
                step_logits = logits[0, -1, :]
                log_probs = F.log_softmax(step_logits, dim=0)
                total_log_prob += log_probs[t_tok].item()
                if t_idx == 0:
                    log_probs_first = log_probs[t_tok]
                log_probs_last = log_probs[t_tok]
                pred = torch.argmax(step_logits).item()
                generated_name.append(pred)
                if pred != t_tok:
                    full_correct = False
                # Teacher forcing: always append true token
                current_tokens.append(t_tok)

            ppl = np.exp(-total_log_prob / len(target_tokens))
            first_char_ppl = np.exp(-log_probs_first.item()) if target_tokens else ppl
            last_char_ppl = np.exp(-log_probs_last.item()) if target_tokens else ppl

            # hit@1: greedy-decoded name matches target exactly
            hit1 = 1 if generated_name == target_tokens else 0

            # hit@5: target name's first char is in top-5 at first step,
            # AND all subsequent chars are correct under teacher forcing
            # (approximation: full name would require beam search;
            #  instead, check if all chars are in top-5 at each step)
            all_in_top5 = True
            current_tokens_t5 = prompt_tokens.copy()
            for t_idx, t_tok in enumerate(target_tokens):
                x = torch.tensor([current_tokens_t5[-config.block_size:]],
                                 dtype=torch.long, device=config.device)
                logits = model.predict_text(x)
                step_logits = logits[0, -1, :]
                top5 = torch.topk(step_logits, k=min(5, step_logits.shape[0])).indices.tolist()
                if t_tok not in top5:
                    all_in_top5 = False
                    break
                current_tokens_t5.append(t_tok)
            hit5 = 1 if all_in_top5 else 0

            results[tier][relation].append({
                "hit1": hit1,
                "hit5": hit5,
                "ppl": ppl,
                "first_char_ppl": first_char_ppl,
                "last_char_ppl": last_char_ppl,
                "full_correct": 1 if full_correct else 0,
                "prompt": p["prompt"],
                "target": p["target"],
            })

    # --- Tier-level summary ---
    summary = {}
    for tier in results:
        tier_results = {"hit1": [], "hit5": [], "ppl": [], "first_char_ppl": [], "last_char_ppl": [], "full_correct": []}
        for rel in results[tier]:
            for r in results[tier][rel]:
                tier_results["hit1"].append(r["hit1"])
                tier_results["hit5"].append(r["hit5"])
                tier_results["ppl"].append(r["ppl"])
                tier_results["first_char_ppl"].append(r["first_char_ppl"])
                tier_results["last_char_ppl"].append(r["last_char_ppl"])
                tier_results["full_correct"].append(r["full_correct"])

        summary[tier] = {
            "hit1": np.mean(tier_results["hit1"]),
            "hit5": np.mean(tier_results["hit5"]),
            "ppl": np.exp(np.mean(np.log(tier_results["ppl"]))),
            "first_char_ppl": np.exp(np.mean(np.log(tier_results["first_char_ppl"]))),
            "last_char_ppl": np.exp(np.mean(np.log(tier_results["last_char_ppl"]))),
            "full_correct": np.mean(tier_results["full_correct"]),
            "n": len(tier_results["hit1"]),
        }

    # --- Per-relation summary within each tier ---
    relation_summary = {}
    for tier in results:
        relation_summary[tier] = {}
        for rel in results[tier]:
            rel_data = results[tier][rel]
            if len(rel_data) == 0:
                continue
            relation_summary[tier][rel] = {
                "hit1": np.mean([r["hit1"] for r in rel_data]),
                "hit5": np.mean([r["hit5"] for r in rel_data]),
                "ppl": np.exp(np.mean([np.log(r["ppl"]) for r in rel_data])),
                "full_correct": np.mean([r["full_correct"] for r in rel_data]),
                "n": len(rel_data),
            }

    # --- Print tier-level summary ---
    print(f"\n{'='*60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*60}")
    for tier in ALL_TIERS:
        if tier in summary:
            s = summary[tier]
            print(f"  {tier:>30s}: hit@1={s['hit1']:.3f}  hit@5={s['hit5']:.3f}  "
                  f"ppl={s['ppl']:.2f}  fc_ppl={s['first_char_ppl']:.2f}  lc_ppl={s['last_char_ppl']:.2f}  full_acc={s['full_correct']:.3f}  (n={s['n']})")

    # --- Print per-relation breakout ---
    print(f"\n  Per-relation breakout:")
    print(f"  {'Tier':<30s} {'Relation':<25s} {'hit@1':>6s} {'hit@5':>6s} {'ppl':>8s} {'full':>6s} {'n':>4s}")
    print(f"  {'-'*85}")
    for tier in ALL_TIERS:
        if tier not in relation_summary:
            continue
        for rel in sorted(relation_summary[tier].keys()):
            rs = relation_summary[tier][rel]
            print(f"  {tier:<30s} {rel:<25s} {rs['hit1']:>6.3f} {rs['hit5']:>6.3f} "
                  f"{rs['ppl']:>8.2f} {rs['full_correct']:>6.3f} {rs['n']:>4d}")

    model.train()
    return summary, relation_summary, results


# ============================================================================
# Model Factory
# ============================================================================

MODEL_NAMES = ["A", "A'", "B", "B'", "C", "C'", "D", "D'", "E", "E'", "F", "F'", "G", "G'", "H", "H'"]

# Which models use linearized text (B/C family) vs structured KG (A/D/E family)
LINEARIZED_MODELS = {"B", "B'", "C", "C'"}
SLOTTED_KG_MODELS = {"A", "A'", "G", "G'"}    # use get_mlm_batch_slotted (3 slots: HEAD/REL/TAIL)
NATIVE_KG_MODELS = {"E", "E'", "H", "H'"}     # use get_mlm_batch_native (rel as angle operator only)
FLAT_KG_MODELS = {"D", "D'", "F", "F'"}       # use get_mlm_batch_flat (rel as token)


def create_model(name, vocab_size, config):
    """Create a model by name."""
    n_e = config.n_embed
    n_l = config.n_layers
    bs = config.block_size
    d = config.dropout

    if name == "A":
        return ModelA(vocab_size, n_e, n_l, bs, d, rotate_v=False)
    elif name == "A'":
        return ModelA(vocab_size, n_e, n_l, bs, d, rotate_v=True)
    elif name == "B":
        return ModelB(vocab_size, n_e, n_l, bs, d, rotate_v=False)
    elif name == "B'":
        return ModelB(vocab_size, n_e, n_l, bs, d, rotate_v=True)
    elif name == "C":
        return ModelC(vocab_size, n_e, n_l, bs, d, rotate_v=False)
    elif name == "C'":
        return ModelC(vocab_size, n_e, n_l, bs, d, rotate_v=True)
    elif name == "D":
        return ModelD(vocab_size, n_e, n_l, bs, d, rotate_v=False)
    elif name == "D'":
        return ModelD(vocab_size, n_e, n_l, bs, d, rotate_v=True)
    elif name == "E":
        return ModelE(vocab_size, n_e, n_l, bs, n_relations=len(KG_RELATIONS), dropout=d, rotate_v=False)
    elif name == "E'":
        return ModelE(vocab_size, n_e, n_l, bs, n_relations=len(KG_RELATIONS), dropout=d, rotate_v=True)
    elif name == "F":
        return ModelF(vocab_size, n_e, n_l, bs, d, rotate_v=False)
    elif name == "F'":
        return ModelF(vocab_size, n_e, n_l, bs, d, rotate_v=True)
    elif name == "G":
        return ModelG(vocab_size, n_e, n_l, bs, n_relations=len(KG_RELATIONS), dropout=d, rotate_v=False)
    elif name == "G'":
        return ModelG(vocab_size, n_e, n_l, bs, n_relations=len(KG_RELATIONS), dropout=d, rotate_v=True)
    elif name == "H":
        return ModelH(vocab_size, n_e, n_l, bs, n_relations=len(KG_RELATIONS), dropout=d, rotate_v=False)
    elif name == "H'":
        return ModelH(vocab_size, n_e, n_l, bs, n_relations=len(KG_RELATIONS), dropout=d, rotate_v=True)
    else:
        raise ValueError(f"Unknown model name: {name}")


# ============================================================================
# Main Experiment Runner
# ============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(generous_linearization=True, seed=42, models_to_run=None,
                   checkpoint_dir=None, load_checkpoints=False,
                   resume_training=False, expanded_names=False):
    """Run one sub-experiment (7a or 7b) with a given seed.

    Args:
        generous_linearization: True for 7a, False for 7b
        seed: random seed
        models_to_run: list of model names to run, or None for all
        checkpoint_dir: directory to save/load model checkpoints
        load_checkpoints: if True, load saved models instead of training
    """
    if models_to_run is None:
        models_to_run = MODEL_NAMES

    exp = "7a" if generous_linearization else "7b"
    exp_name = f"{exp} (generous)" if generous_linearization else f"{exp} (realistic)"
    print(f"\n{'#'*70}")
    print(f"# Experiment {exp_name}, seed={seed}")
    print(f"{'#'*70}\n")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    (vocab, text_base, text_linearized, kg_dataset,
     eval_prompts, kg_eval_prompts) = prepare_data(generous_linearization, expanded_names=expanded_names)
    print(f"Vocabulary size: {vocab.size}")
    print(f"Text dataset (base): {len(text_base.data)} tokens")
    print(f"Text dataset (linearized): {len(text_linearized.data)} tokens")
    print(f"KG triples: {len(kg_dataset.triples)}")
    print(f"Eval prompts (text): {len(eval_prompts)}")
    print(f"Eval prompts (KG): {len(kg_eval_prompts)}")

    results = {}
    relation_results = {}
    kg_results = {}
    kg_relation_results = {}

    # Create checkpoint dir if needed
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for name in models_to_run:
        model = create_model(name, vocab.size, cfg)
        # Store vocab reference on model for pad_mask in forward_kg
        model.vocab = vocab
        print(f"\nModel {name}: {count_parameters(model):,} params")

        # Sanitize model name for filename (A' -> Ap)
        safe_name = name.replace("'", "p")
        ckpt_path = os.path.join(checkpoint_dir, f"{exp}_{safe_name}_seed{seed}.pt") if checkpoint_dir else None

        loaded = False
        resume_opt = None
        iters_done = 0
        if (load_checkpoints or resume_training) and ckpt_path and os.path.exists(ckpt_path):
            print(f"--- Loading Model {name} from {ckpt_path} ---")
            ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(cfg.device)
            iters_done = ckpt.get("iters_done",
                                  ckpt.get("config", {}).get("max_iters", 0))
            if resume_training:
                resume_opt = ckpt.get("optimizer_state_dict", None)
                remaining = cfg.max_iters - iters_done
                if remaining <= 0:
                    print(f"  Already trained {iters_done} iters (target {cfg.max_iters}), skipping")
                    loaded = True
                else:
                    if resume_opt:
                        print(f"  Resuming from iter {iters_done}, training {remaining} more (optimizer state loaded)")
                    else:
                        print(f"  Resuming from iter {iters_done}, training {remaining} more (fresh optimizer)")
            else:
                loaded = True

        if not loaded:
            remaining = cfg.max_iters - iters_done
            # Temporarily override max_iters for this training run
            orig_max_iters = cfg.max_iters
            cfg.max_iters = remaining
            print(f"--- Training Model {name} ---")
            opt_state = None
            if name in LINEARIZED_MODELS:
                text_ds = text_linearized
                _, opt_state = train_model_text_only(
                    model, text_ds, cfg, name=name,
                    resume_optimizer_state=resume_opt)
            elif name in SLOTTED_KG_MODELS:
                _, opt_state = train_model_mixed(
                    model, text_base, kg_dataset, cfg,
                    name=name, kg_batch_fn="slotted",
                    resume_optimizer_state=resume_opt)
            elif name in NATIVE_KG_MODELS:
                _, opt_state = train_model_mixed(
                    model, text_base, kg_dataset, cfg,
                    name=name, kg_batch_fn="native",
                    resume_optimizer_state=resume_opt)
            elif name in FLAT_KG_MODELS:
                _, opt_state = train_model_mixed(
                    model, text_base, kg_dataset, cfg,
                    name=name, kg_batch_fn="flat",
                    resume_optimizer_state=resume_opt)
            cfg.max_iters = orig_max_iters
            total_iters = iters_done + remaining

            # Save checkpoint
            if ckpt_path:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt_state,
                    "iters_done": total_iters,
                    "model_name": name,
                    "seed": seed,
                    "config": {attr: getattr(cfg, attr) for attr in vars(cfg)},
                    "vocab_data": {
                        "char2idx": vocab.char2idx,
                        "idx2char": vocab.idx2char,
                    },
                }
                torch.save(checkpoint, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path} (total iters: {total_iters})")

        summary, rel_summary, details = evaluate_model(model, eval_prompts, vocab, cfg, name)
        results[name] = summary
        relation_results[name] = rel_summary

        # KG evaluation for A/D/E models
        base_name = name.replace("'", "")
        if base_name in ("A", "D", "E", "F", "G", "H"):
            kg_summary, kg_rel_summary, _ = evaluate_model_kg(
                model, kg_eval_prompts, vocab, cfg, model_name=name, model_type=base_name)
            kg_results[name] = kg_summary
            kg_relation_results[name] = kg_rel_summary

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, relation_results, kg_results, kg_relation_results


def print_comparison_table(results, exp_name, models):
    """Print a formatted comparison table."""
    print(f"\n{'='*140}")
    print(f"  COMPARISON TABLE -- {exp_name}")
    print(f"{'='*140}")

    header = f"{'Tier':<32} {'Metric':<12} "
    for m in models:
        header += f"{m:<12}"
    print(header)
    print(f"{'-'*140}")

    for tier in ALL_TIERS:
        for metric in ["hit1", "hit5", "full_correct", "ppl"]:
            line = f"{tier:<32} {metric:<12} "
            for m in models:
                if m in results and tier in results[m]:
                    v = results[m][tier].get(metric, float('nan'))
                else:
                    v = float('nan')
                if metric == "ppl":
                    line += f"{v:<12.2f}"
                else:
                    line += f"{v:<12.3f}"
            print(line)
        print()


def print_relation_table(relation_results, exp_name, models):
    """Print a per-relation × per-model comparison table."""
    print(f"\n{'='*140}")
    print(f"  PER-RELATION TABLE -- {exp_name}")
    print(f"{'='*140}")

    header = f"{'Tier':<30s} {'Relation':<25s} "
    for m in models:
        header += f"{m:<12}"
    print(header)
    print(f"{'-'*140}")

    for tier in ALL_TIERS:
        # Collect all relations across models for this tier
        all_rels = set()
        for m in models:
            if m in relation_results and tier in relation_results[m]:
                all_rels.update(relation_results[m][tier].keys())
        for rel in sorted(all_rels):
            line = f"{tier:<30s} {rel:<25s} "
            for m in models:
                if (m in relation_results and tier in relation_results[m]
                        and rel in relation_results[m][tier]):
                    v = relation_results[m][tier][rel]["hit1"]
                    line += f"{v:<12.3f}"
                else:
                    line += f"{'n/a':<12}"
            print(line)
        if all_rels:
            print()


def print_kg_comparison_table(kg_results, exp_name, models):
    """Print KG evaluation comparison table (A/D/E models only)."""
    kg_models = [m for m in models if m.replace("'", "") in ("A", "D", "E", "F", "G", "H")]
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
                if metric == "ppl":
                    line += f"{v:<12.2f}"
                else:
                    line += f"{v:<12.3f}"
            print(line)
        print()


def print_kg_relation_table(kg_relation_results, exp_name, models):
    """Print per-relation KG evaluation table (A/D/E models only)."""
    kg_models = [m for m in models if m.replace("'", "") in ("A", "D", "E", "F", "G", "H")]
    if not kg_models:
        return

    print(f"\n{'='*120}")
    print(f"  KG PER-RELATION TABLE -- {exp_name}")
    print(f"{'='*120}")

    header = f"{'Tier':<35s} {'Relation':<20s} "
    for m in kg_models:
        header += f"{m:<12}"
    print(header)
    print(f"{'-'*120}")

    for tier in ALL_TIERS:
        all_rels = set()
        for m in kg_models:
            if m in kg_relation_results and tier in kg_relation_results[m]:
                all_rels.update(kg_relation_results[m][tier].keys())
        for rel in sorted(all_rels):
            line = f"{tier:<35s} {rel:<20s} "
            for m in kg_models:
                if (m in kg_relation_results and tier in kg_relation_results[m]
                        and rel in kg_relation_results[m][tier]):
                    v = kg_relation_results[m][tier][rel]["hit1"]
                    line += f"{v:<12.3f}"
                else:
                    line += f"{'n/a':<12}"
            print(line)
        if all_rels:
            print()


def main():
    parser = argparse.ArgumentParser(description="Exp 7: KG+Text vs Linearized")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to run (e.g., A B C'). Default: all.")
    parser.add_argument("--iters", type=int, default=None,
                        help="Override max_iters")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override n_seeds")
    parser.add_argument("--exp", choices=["7a", "7b", "both"], default="both",
                        help="Which sub-experiment to run")
    parser.add_argument("--n_embed", type=int, default=None,
                        help="Override n_embed")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Override n_layers (default 1)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 50 iters, 1 seed, all models")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="Dir to save/load model checkpoints")
    parser.add_argument("--load_checkpoints", action="store_true",
                        help="Load saved models, skip training")
    parser.add_argument("--resume_training", action="store_true",
                        help="Load checkpoint and continue training")
    parser.add_argument("--expanded_names", action="store_true",
                        help="Use expanded 240-char alphabet for name generation")
    args = parser.parse_args()

    if args.n_embed is not None:
        cfg.n_embed = args.n_embed
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers

    if args.smoke:
        cfg.max_iters = 50
        cfg.n_seeds = 1
        cfg.eval_interval = 25
        models = args.models or MODEL_NAMES
    else:
        if args.iters is not None:
            cfg.max_iters = args.iters
        if args.seeds is not None:
            cfg.n_seeds = args.seeds
        models = args.models or MODEL_NAMES

    # Normalize model names: allow both "A'" and "Ap" etc.
    normalized = []
    for m in models:
        if m.endswith("p") and len(m) == 2 and m[0] in "ABCDE":
            normalized.append(m[0] + "'")
        else:
            normalized.append(m)
    models = normalized

    print("=" * 70)
    print("  Exp 7: Native KG+Text vs Linearized KG-as-Text")
    print("=" * 70)
    print(f"\nConfig: n_embed={cfg.n_embed}, n_layers={cfg.n_layers}, "
          f"max_iters={cfg.max_iters}, batch_size={cfg.batch_size}, "
          f"lr={cfg.lr}, device={cfg.device}")
    print(f"Models: {models}")

    experiments = []
    if args.exp in ("7a", "both"):
        experiments.append(True)
    if args.exp in ("7b", "both"):
        experiments.append(False)

    all_results = {}

    for generous in experiments:
        exp_name = "7a" if generous else "7b"
        seed_results = []
        seed_relation_results = []
        seed_kg_results = []
        seed_kg_relation_results = []

        for seed in range(cfg.n_seeds):
            results, rel_results, kg_res, kg_rel_res = run_experiment(
                generous_linearization=generous, seed=seed,
                models_to_run=models,
                checkpoint_dir=args.checkpoint_dir,
                load_checkpoints=args.load_checkpoints,
                resume_training=args.resume_training,
                expanded_names=args.expanded_names)
            seed_results.append(results)
            seed_relation_results.append(rel_results)
            seed_kg_results.append(kg_res)
            seed_kg_relation_results.append(kg_rel_res)

        # Average tier-level text results across seeds
        avg_results = {}
        for model_name in models:
            avg_results[model_name] = {}
            for tier in ALL_TIERS:
                metrics = {}
                for metric in ["hit1", "hit5", "full_correct", "ppl"]:
                    values = [
                        sr[model_name][tier][metric]
                        for sr in seed_results
                        if model_name in sr and tier in sr[model_name]
                    ]
                    if values:
                        metrics[metric] = np.mean(values)
                        metrics[f"{metric}_std"] = np.std(values)
                avg_results[model_name][tier] = metrics

        # Average per-relation text results across seeds
        avg_relation_results = {}
        for model_name in models:
            avg_relation_results[model_name] = {}
            for tier in ALL_TIERS:
                avg_relation_results[model_name][tier] = {}
                all_rels = set()
                for srr in seed_relation_results:
                    if model_name in srr and tier in srr[model_name]:
                        all_rels.update(srr[model_name][tier].keys())
                for rel in all_rels:
                    rel_metrics = {}
                    for metric in ["hit1", "hit5", "full_correct", "ppl"]:
                        values = [
                            srr[model_name][tier][rel][metric]
                            for srr in seed_relation_results
                            if (model_name in srr and tier in srr[model_name]
                                and rel in srr[model_name][tier])
                        ]
                        if values:
                            rel_metrics[metric] = np.mean(values)
                    avg_relation_results[model_name][tier][rel] = rel_metrics

        # Average KG results across seeds (A/D/E models only)
        avg_kg_results = {}
        kg_models = [m for m in models if m.replace("'", "") in ("A", "D", "E", "F", "G", "H")]
        for model_name in kg_models:
            avg_kg_results[model_name] = {}
            for tier in ALL_TIERS:
                metrics = {}
                for metric in ["hit1", "hit5", "ppl"]:
                    values = [
                        skr[model_name][tier][metric]
                        for skr in seed_kg_results
                        if model_name in skr and tier in skr[model_name]
                    ]
                    if values:
                        metrics[metric] = np.mean(values)
                        metrics[f"{metric}_std"] = np.std(values)
                avg_kg_results[model_name][tier] = metrics

        # Average per-relation KG results across seeds
        avg_kg_relation_results = {}
        for model_name in kg_models:
            avg_kg_relation_results[model_name] = {}
            for tier in ALL_TIERS:
                avg_kg_relation_results[model_name][tier] = {}
                all_rels = set()
                for skrr in seed_kg_relation_results:
                    if model_name in skrr and tier in skrr[model_name]:
                        all_rels.update(skrr[model_name][tier].keys())
                for rel in all_rels:
                    rel_metrics = {}
                    for metric in ["hit1", "hit5", "ppl"]:
                        values = [
                            skrr[model_name][tier][rel][metric]
                            for skrr in seed_kg_relation_results
                            if (model_name in skrr and tier in skrr[model_name]
                                and rel in skrr[model_name][tier])
                        ]
                        if values:
                            rel_metrics[metric] = np.mean(values)
                    avg_kg_relation_results[model_name][tier][rel] = rel_metrics

        exp_label = f"Exp {exp_name} ({'Generous' if generous else 'Realistic'} Linearization)"
        print_comparison_table(avg_results, exp_label, models)
        print_relation_table(avg_relation_results, exp_label, models)
        print_kg_comparison_table(avg_kg_results, exp_label, models)
        print_kg_relation_table(avg_kg_relation_results, exp_label, models)

        print(f"\n  Mean +/- Std across {cfg.n_seeds} seeds (TEXT):")
        for tier in ALL_TIERS:
            print(f"\n  {tier}:")
            for metric in ["hit1", "hit5", "full_correct"]:
                line = f"    {metric:<15}"
                for model_name in models:
                    m = avg_results[model_name][tier].get(metric, 0)
                    s = avg_results[model_name][tier].get(f"{metric}_std", 0)
                    line += f"  {m:.3f}+/-{s:.3f}"
                print(line)

        if kg_models:
            print(f"\n  Mean +/- Std across {cfg.n_seeds} seeds (KG):")
            for tier in ALL_TIERS:
                print(f"\n  {tier}:")
                for metric in ["hit1", "hit5"]:
                    line = f"    {metric:<15}"
                    for model_name in kg_models:
                        m = avg_kg_results[model_name][tier].get(metric, 0)
                        s = avg_kg_results[model_name][tier].get(f"{metric}_std", 0)
                        line += f"  {m:.3f}+/-{s:.3f}"
                    print(line)

        all_results[exp_name] = avg_results
        all_results[f"{exp_name}_per_seed"] = seed_results
        all_results[f"{exp_name}_per_relation"] = avg_relation_results
        all_results[f"{exp_name}_kg"] = avg_kg_results
        all_results[f"{exp_name}_kg_per_relation"] = avg_kg_relation_results

    # Save results to JSON
    results_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"exp7_results_{timestamp}.json")

    # Convert numpy floats to Python floats for JSON serialization
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
            "n_embed": cfg.n_embed,
            "n_layers": cfg.n_layers,
            "max_iters": cfg.max_iters,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "n_seeds": cfg.n_seeds,
            "models": models,
        },
        "results": to_serializable(all_results),
        "timestamp": timestamp,
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Also save a latest symlink/copy for easy access
    latest_file = os.path.join(results_dir, "exp7_results_latest.json")
    with open(latest_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Latest results: {latest_file}")

    print("\n\nDone.")
    return all_results


if __name__ == "__main__":
    results = main()
