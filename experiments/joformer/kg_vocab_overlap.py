#!/usr/bin/env python3
"""Analyze word-level vocabulary overlap between KG entities and wiki text."""

import os
import sys
import re
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_wiki_kg_streaming import load_all_kg_triples


def load_wiki_words(wiki_path, max_lines=None):
    """Load wiki text and extract word-level vocabulary with counts."""
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki_path', type=str, default=None)
    parser.add_argument('--wiki_lines', type=int, default=None,
                        help='Max wiki lines (default: all)')
    # KG data paths
    parser.add_argument('--wordnet_path', type=str, default=None)
    parser.add_argument('--framenet_path', type=str, default=None)
    parser.add_argument('--bats_dir', type=str, default=None)
    parser.add_argument('--google_path', type=str, default=None)
    parser.add_argument('--analogies_path', type=str, default=None)
    args = parser.parse_args()

    if args.wiki_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.wiki_path = os.path.join(script_dir, '..', 'exp8', 'data', 'wiki.en.txt')

    # Load KG triples
    print("Loading KG triples...")
    triples = load_all_kg_triples(args)

    # Extract KG words (lowercased)
    kg_heads = set()
    kg_tails = set()
    kg_relations = set()
    for h, r, t in triples:
        kg_heads.add(h.lower())
        kg_tails.add(t.lower())
        kg_relations.add(r.lower())
    kg_entities = kg_heads | kg_tails
    kg_all_words = set()
    for entity in kg_entities:
        kg_all_words.update(re.findall(r'[a-zA-Z]+', entity.lower()))

    print(f"\nKG stats:")
    print(f"  Triples: {len(triples):,}")
    print(f"  Unique heads: {len(kg_heads):,}")
    print(f"  Unique tails: {len(kg_tails):,}")
    print(f"  Unique entities (heads+tails): {len(kg_entities):,}")
    print(f"  Unique words in entities: {len(kg_all_words):,}")
    print(f"  Unique relations: {len(kg_relations):,}")
    print(f"  Relations: {sorted(kg_relations)}")

    # Load wiki words
    print(f"\nLoading wiki text from {args.wiki_path}...")
    wiki_counts = load_wiki_words(args.wiki_path, args.wiki_lines)
    wiki_words = set(wiki_counts.keys())
    print(f"  Unique wiki words: {len(wiki_words):,}")
    print(f"  Total wiki word tokens: {sum(wiki_counts.values()):,}")

    # Overlap analysis
    kg_in_wiki = kg_all_words & wiki_words
    kg_not_in_wiki = kg_all_words - wiki_words
    wiki_not_in_kg = wiki_words - kg_all_words

    print(f"\n{'='*60}")
    print("OVERLAP ANALYSIS (word level)")
    print(f"{'='*60}")
    print(f"  KG words in wiki:      {len(kg_in_wiki):,} / {len(kg_all_words):,} "
          f"({100*len(kg_in_wiki)/len(kg_all_words):.1f}%)")
    print(f"  KG words NOT in wiki:  {len(kg_not_in_wiki):,} / {len(kg_all_words):,} "
          f"({100*len(kg_not_in_wiki)/len(kg_all_words):.1f}%)")
    print(f"  Wiki words NOT in KG:  {len(wiki_not_in_kg):,} / {len(wiki_words):,} "
          f"({100*len(wiki_not_in_kg)/len(wiki_words):.1f}%)")

    # Coverage: what fraction of wiki tokens are KG words?
    kg_token_count = sum(wiki_counts[w] for w in kg_in_wiki)
    total_tokens = sum(wiki_counts.values())
    print(f"\n  Wiki token coverage by KG words: {kg_token_count:,} / {total_tokens:,} "
          f"({100*kg_token_count/total_tokens:.1f}%)")

    # Sample KG words not in wiki
    print(f"\n  Sample KG words NOT in wiki (up to 50):")
    for w in sorted(kg_not_in_wiki)[:50]:
        print(f"    {w}")

    # Sample KG words in wiki with their wiki frequency
    print(f"\n  Sample KG words in wiki — lowest frequency (up to 50):")
    rare_kg_in_wiki = sorted(kg_in_wiki, key=lambda w: wiki_counts[w])[:50]
    for w in rare_kg_in_wiki:
        print(f"    {w}: {wiki_counts[w]:,} occurrences")

    print(f"\n  Sample KG words in wiki — highest frequency (up to 30):")
    common_kg_in_wiki = sorted(kg_in_wiki, key=lambda w: wiki_counts[w], reverse=True)[:30]
    for w in common_kg_in_wiki:
        print(f"    {w}: {wiki_counts[w]:,} occurrences")

    # Save full lists to files
    output_dir = os.path.dirname(os.path.abspath(__file__))

    path = os.path.join(output_dir, 'kg_words_not_in_wiki.txt')
    with open(path, 'w') as f:
        for w in sorted(kg_not_in_wiki):
            f.write(w + '\n')
    print(f"\nSaved: {path} ({len(kg_not_in_wiki)} words)")

    path = os.path.join(output_dir, 'kg_words_in_wiki.txt')
    with open(path, 'w') as f:
        for w in sorted(kg_in_wiki, key=lambda w: wiki_counts[w], reverse=True):
            f.write(f"{w}\t{wiki_counts[w]}\n")
    print(f"Saved: {path} ({len(kg_in_wiki)} words)")

    path = os.path.join(output_dir, 'kg_entity_list.txt')
    with open(path, 'w') as f:
        for e in sorted(kg_entities):
            in_wiki = "YES" if e in wiki_words or all(
                w in wiki_words for w in re.findall(r'[a-zA-Z]+', e)) else "NO"
            f.write(f"{e}\t{in_wiki}\n")
    print(f"Saved: {path} ({len(kg_entities)} entities)")


if __name__ == '__main__':
    main()
