#!/usr/bin/env python3
"""Download and preprocess OpenWebText for training.

Produces the same format as train_wiki_streaming.py preprocess:
  - wiki_tokens.bin (flat int32 memmap)
  - wiki_tokens.meta (JSON metadata)
  - wiki_tokenizer.json (BPE tokenizer)

Usage:
  python preprocess_openwebtext.py --output_dir /path/to/output --vocab_size 32000
"""

import argparse
import json
import os
import time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--max_docs', type=int, default=None,
                        help='Limit number of documents (for testing)')
    parser.add_argument('--tokenizer_train_docs', type=int, default=100000,
                        help='Number of documents to train tokenizer on')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bin_path = os.path.join(args.output_dir, 'wiki_tokens.bin')
    meta_path = os.path.join(args.output_dir, 'wiki_tokens.meta')
    tok_path = os.path.join(args.output_dir, 'wiki_tokenizer.json')

    # Step 1: Download OpenWebText
    print("Loading OpenWebText from HuggingFace...")
    t0 = time.time()
    dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    print(f"  Loaded {len(dataset)} documents in {time.time()-t0:.1f}s")

    if args.max_docs:
        dataset = dataset.select(range(min(args.max_docs, len(dataset))))
        print(f"  Limited to {len(dataset)} documents")

    # Step 2: Train BPE tokenizer
    print(f"\n[1/2] Training BPE tokenizer (vocab_size={args.vocab_size})...")
    t0 = time.time()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<unk>"],
        show_progress=True,
    )

    # Train on a subset of documents
    n_train = min(args.tokenizer_train_docs, len(dataset))
    print(f"  Training on {n_train} documents...")

    def doc_iterator():
        for i in range(n_train):
            yield dataset[i]['text']

    tokenizer.train_from_iterator(doc_iterator(), trainer=trainer)
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Vocab size: {actual_vocab}, trained in {time.time()-t0:.1f}s")
    tokenizer.save(tok_path)

    # Step 3: Tokenize all documents to binary
    print(f"\n[2/2] Tokenizing {len(dataset)} documents to binary...")
    t0 = time.time()
    total_tokens = 0

    with open(bin_path, 'wb') as f:
        for i in range(len(dataset)):
            text = dataset[i]['text']
            if text.strip():
                enc = tokenizer.encode(text)
                ids = enc.ids
                if ids:
                    chunk = np.array(ids, dtype=np.int32)
                    f.write(chunk.tobytes())
                    total_tokens += len(ids)
            if (i + 1) % 100000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(dataset) - i - 1) / rate
                print(f"  {i+1}/{len(dataset)} docs, {total_tokens:,} tokens, "
                      f"{rate:.0f} docs/s, ETA {eta/60:.0f}min")

    dt = time.time() - t0
    file_size_gb = os.path.getsize(bin_path) / (1024**3)
    print(f"  {total_tokens:,} tokens written in {dt:.1f}s")
    print(f"  Binary file: {bin_path} ({file_size_gb:.2f} GB)")

    # Step 4: Save metadata
    meta = {
        'total_tokens': total_tokens,
        'vocab_size': actual_vocab,
        'source': 'openwebtext (HuggingFace)',
        'max_docs': args.max_docs,
        'doc_count': len(dataset),
        'dtype': 'int32',
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}")
    print(f"\nDone. Data ready at {args.output_dir}")


if __name__ == '__main__':
    main()
