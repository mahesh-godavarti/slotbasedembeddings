#!/usr/bin/env python3
"""Build cross-pollination evaluation sets.

1. KG→Text test: Wiki sentences containing words that are in KG but rare in wiki.
   If KG training helps, the model should predict these rare-KG-word sentences better.

2. Text→KG test: Novel KG triples using wiki words that follow morphological
   relations but aren't in the KG training data. If text training helps,
   the model should predict these novel triples better.

Output files in data_v8k/:
  - cross_text_eval.bin + .meta — tokenized wiki sentences with rare KG words
  - cross_kg_eval.pkl + .meta  — novel morphological triples for KG eval
"""

import json
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_wiki_kg_streaming import load_all_kg_triples, encode_kg_triples


def load_wiki_word_freq(wiki_path, max_lines=None):
    """Load wiki text and return word frequency counter."""
    word_counts = Counter()
    with open(wiki_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if line:
                words = re.findall(r'[a-zA-Z]+', line.lower())
                word_counts.update(words)
    return word_counts


def find_rare_kg_words(kg_triples, wiki_word_counts, max_freq=5):
    """Find words that are in KG entities but rare (<=max_freq) in wiki."""
    kg_words = set()
    for h, r, t in kg_triples:
        kg_words.update(re.findall(r'[a-zA-Z]+', h.lower()))
        kg_words.update(re.findall(r'[a-zA-Z]+', t.lower()))

    rare = set()
    for w in kg_words:
        if wiki_word_counts.get(w, 0) <= max_freq:
            rare.add(w)
    return rare, kg_words


def extract_rare_kg_sentences(wiki_path, rare_words, max_lines=None,
                               max_sentences=50000):
    """Extract wiki lines containing at least one rare KG word."""
    sentences = []
    for w in rare_words:
        if len(w) < 3:
            rare_words.discard(w)

    # Build a set for O(1) lookup
    rare_set = set(rare_words)

    with open(wiki_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            if len(sentences) >= max_sentences:
                break
            stripped = line.strip()
            if not stripped or len(stripped) < 20:
                continue
            line_words = set(re.findall(r'[a-zA-Z]+', stripped.lower()))
            overlap = line_words & rare_set
            if overlap:
                sentences.append(stripped)

    return sentences


def generate_morphological_triples(wiki_word_counts, existing_triples_set,
                                    min_wiki_freq=10):
    """Generate novel morphological triples from wiki words not in KG.

    Only generates triples where BOTH forms appear in wiki with sufficient
    frequency, and the triple is NOT in the existing KG.
    """
    wiki_words = set(wiki_word_counts.keys())
    novel_triples = []

    # Rules: (relation_name, suffix_to_remove, suffix_to_add)
    # We try to find base→derived pairs in wiki vocabulary
    rules = [
        # Plural
        ('plural_of', '', 's', lambda w: len(w) > 3),
        ('plural_of', '', 'es', lambda w: len(w) > 3 and w[-1] in 'sxz'),
        ('plural_of', 'y', 'ies', lambda w: len(w) > 3),
        # Verb 3rd person singular
        ('verb_inf___3psg', '', 's', lambda w: len(w) > 3),
        ('verb_inf___3psg', '', 'es', lambda w: len(w) > 3),
        # Verb past tense
        ('verb_inf___ved', '', 'ed', lambda w: len(w) > 3),
        ('verb_inf___ved', '', 'd', lambda w: len(w) > 3 and w[-1] == 'e'),
        ('verb_inf___ved', 'y', 'ied', lambda w: len(w) > 3),
        # Verb progressive
        ('verb_inf___ving', '', 'ing', lambda w: len(w) > 3),
        ('verb_inf___ving', 'e', 'ing', lambda w: len(w) > 3),
        # Comparative
        ('adj___comparative', '', 'er', lambda w: len(w) > 3),
        # Superlative
        ('adj___superlative', '', 'est', lambda w: len(w) > 3),
        # Adjective to adverb
        ('adjective_to_adverb_operator', '', 'ly', lambda w: len(w) > 3),
        # un+adjective
        ('un+adj_reg', '', '', lambda w: False),  # placeholder, handled below
    ]

    # For each rule, find candidate pairs
    for rel, remove_suffix, add_suffix, filter_fn in rules:
        count = 0
        for base_word in wiki_words:
            if not filter_fn(base_word):
                continue
            if not base_word.isalpha():
                continue

            # Construct derived form
            if remove_suffix:
                if not base_word.endswith(remove_suffix):
                    continue
                stem = base_word[:-len(remove_suffix)]
            else:
                stem = base_word
            derived = stem + add_suffix

            # Check both forms exist in wiki with sufficient frequency
            if derived not in wiki_words:
                continue
            if wiki_word_counts[base_word] < min_wiki_freq:
                continue
            if wiki_word_counts[derived] < min_wiki_freq:
                continue

            # Check triple not in existing KG
            triple = (base_word, rel, derived)
            if triple in existing_triples_set:
                continue

            novel_triples.append(triple)
            count += 1
            if count >= 5000:  # cap per relation
                break

    # un+adj: find words where "un"+word also exists
    count = 0
    for word in wiki_words:
        if not word.isalpha() or len(word) < 4:
            continue
        un_word = 'un' + word
        if un_word in wiki_words:
            if wiki_word_counts[word] < min_wiki_freq:
                continue
            if wiki_word_counts[un_word] < min_wiki_freq:
                continue
            triple = (word, 'un+adj_reg', un_word)
            if triple not in existing_triples_set:
                novel_triples.append(triple)
                count += 1
                if count >= 5000:
                    break

    return novel_triples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='joformer/data_v8k')
    parser.add_argument('--max_freq', type=int, default=5,
                        help='Max wiki frequency for a word to be "rare"')
    parser.add_argument('--min_wiki_freq', type=int, default=10,
                        help='Min wiki frequency for morphological triple words')
    parser.add_argument('--max_sentences', type=int, default=50000,
                        help='Max wiki sentences to extract')
    # KG data paths
    parser.add_argument('--wordnet_path', type=str, default=None)
    parser.add_argument('--framenet_path', type=str, default=None)
    parser.add_argument('--bats_dir', type=str, default=None)
    parser.add_argument('--google_path', type=str, default=None)
    parser.add_argument('--analogies_path', type=str, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.wiki_path is None:
        args.wiki_path = os.path.join(script_dir, '..', 'exp8', 'data', 'wiki.en.txt')
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(script_dir, os.path.basename(args.data_dir))

    # Load tokenizer
    from tokenizers import Tokenizer
    tok_path = os.path.join(args.data_dir, 'wiki_tokenizer.json')
    tokenizer = Tokenizer.from_file(tok_path)

    # Load KG triples
    print("Loading KG triples...")
    kg_triples = load_all_kg_triples(args)
    existing_set = set((h.lower(), r, t.lower()) for h, r, t in kg_triples)

    # Load wiki word frequencies
    print("Loading wiki word frequencies (full corpus)...")
    wiki_word_counts = load_wiki_word_freq(args.wiki_path)
    print(f"  {len(wiki_word_counts):,} unique words, "
          f"{sum(wiki_word_counts.values()):,} total tokens")

    # =====================================================================
    # 1. KG→Text test: rare KG word sentences
    # =====================================================================
    print(f"\n{'='*60}")
    print("BUILDING KG→TEXT EVAL SET")
    print(f"{'='*60}")

    rare_words, all_kg_words = find_rare_kg_words(
        kg_triples, wiki_word_counts, max_freq=args.max_freq)
    print(f"  KG words with wiki freq <= {args.max_freq}: {len(rare_words):,} / {len(all_kg_words):,}")

    # Also include KG words with moderate frequency (6-50) — still "rare-ish"
    moderate_words = set()
    for w in all_kg_words:
        freq = wiki_word_counts.get(w, 0)
        if freq > args.max_freq and freq <= 50 and len(w) >= 3:
            moderate_words.add(w)
    target_words = rare_words | moderate_words
    print(f"  Including moderate freq (6-50): +{len(moderate_words):,} = {len(target_words):,} target words")

    print(f"  Extracting wiki sentences with target words...")
    sentences = extract_rare_kg_sentences(
        args.wiki_path, target_words,
        max_sentences=args.max_sentences)
    print(f"  Extracted {len(sentences):,} sentences")

    # Tokenize and save
    text_eval_bin = os.path.join(args.data_dir, 'cross_text_eval.bin')
    total_tokens = 0
    with open(text_eval_bin, 'wb') as dst:
        for sent in sentences:
            enc = tokenizer.encode(sent)
            ids = enc.ids
            if ids:
                chunk = np.array(ids, dtype=np.int32)
                dst.write(chunk.tobytes())
                total_tokens += len(ids)

    text_eval_meta = {
        'total_tokens': total_tokens,
        'n_sentences': len(sentences),
        'max_freq_threshold': args.max_freq,
        'n_rare_kg_words': len(rare_words),
        'n_moderate_kg_words': len(moderate_words),
        'n_target_words': len(target_words),
        'dtype': 'int32',
    }
    meta_path = os.path.join(args.data_dir, 'cross_text_eval.meta')
    with open(meta_path, 'w') as f:
        json.dump(text_eval_meta, f, indent=2)

    print(f"  Saved: {text_eval_bin} ({total_tokens:,} tokens)")
    print(f"  Saved: {meta_path}")

    # Show some example sentences
    print(f"\n  Example sentences (first 5):")
    for s in sentences[:5]:
        print(f"    {s[:120]}...")

    # =====================================================================
    # 2. Text→KG test: novel morphological triples
    # =====================================================================
    print(f"\n{'='*60}")
    print("BUILDING TEXT→KG EVAL SET")
    print(f"{'='*60}")

    novel_triples = generate_morphological_triples(
        wiki_word_counts, existing_set, min_wiki_freq=args.min_wiki_freq)

    # Count by relation
    rel_counts = Counter(r for _, r, _ in novel_triples)
    print(f"  Generated {len(novel_triples):,} novel triples")
    print(f"  By relation:")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"    {rel}: {cnt:,}")

    # Encode with tokenizer
    encoded_triples, relations = encode_kg_triples(tokenizer, novel_triples)
    print(f"  Encoded: {len(encoded_triples):,} triples, {len(relations)} relations")

    # Save
    kg_eval_pkl = os.path.join(args.data_dir, 'cross_kg_eval.pkl')
    with open(kg_eval_pkl, 'wb') as f:
        pickle.dump(encoded_triples, f)

    kg_eval_meta = {
        'n_triples': len(encoded_triples),
        'n_relations': len(relations),
        'relations': relations,
        'n_source_novel': len(novel_triples),
        'min_wiki_freq': args.min_wiki_freq,
        'relation_counts': dict(rel_counts),
    }
    kg_eval_meta_path = os.path.join(args.data_dir, 'cross_kg_eval.meta')
    with open(kg_eval_meta_path, 'w') as f:
        json.dump(kg_eval_meta, f, indent=2)

    print(f"  Saved: {kg_eval_pkl}")
    print(f"  Saved: {kg_eval_meta_path}")

    # Show examples
    print(f"\n  Example novel triples (first 20):")
    for h, r, t in novel_triples[:20]:
        wiki_h = wiki_word_counts.get(h, 0)
        wiki_t = wiki_word_counts.get(t, 0)
        print(f"    ({h}, {r}, {t})  [wiki freq: {wiki_h}, {wiki_t}]")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  KG→Text eval: {total_tokens:,} tokens from {len(sentences):,} wiki sentences")
    print(f"  Text→KG eval: {len(encoded_triples):,} novel morphological triples")


if __name__ == '__main__':
    main()
