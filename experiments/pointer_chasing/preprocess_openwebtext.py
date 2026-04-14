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
import shutil
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

    # Step 1: Train BPE tokenizer using streaming (no full download)
    print(f"[1/3] Training BPE tokenizer (vocab_size={args.vocab_size})...")
    t0 = time.time()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<unk>"],
        show_progress=True,
    )

    # Stream documents for tokenizer training
    n_train = args.tokenizer_train_docs
    print(f"  Streaming {n_train} documents for tokenizer training...")

    def doc_iterator():
        ds_stream = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
        for i, example in enumerate(ds_stream):
            if i >= n_train:
                break
            yield example['text']

    tokenizer.train_from_iterator(doc_iterator(), trainer=trainer)
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Vocab size: {actual_vocab}, trained in {time.time()-t0:.1f}s")
    tokenizer.save(tok_path)

    # Step 2: Tokenize all documents using streaming
    print(f"\n[2/3] Tokenizing documents (streaming mode)...")
    t0 = time.time()
    total_tokens = 0
    doc_count = 0

    ds_stream = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

    with open(bin_path, 'wb') as f:
        for example in ds_stream:
            if args.max_docs and doc_count >= args.max_docs:
                break
            text = example['text']
            if text.strip():
                enc = tokenizer.encode(text)
                ids = enc.ids
                if ids:
                    chunk = np.array(ids, dtype=np.int32)
                    f.write(chunk.tobytes())
                    total_tokens += len(ids)
            doc_count += 1
            if doc_count % 100000 == 0:
                elapsed = time.time() - t0
                rate = doc_count / elapsed
                print(f"  {doc_count} docs, {total_tokens:,} tokens, "
                      f"{rate:.0f} docs/s, {elapsed/60:.0f}min elapsed")

    dt = time.time() - t0
    file_size_gb = os.path.getsize(bin_path) / (1024**3)
    print(f"  {total_tokens:,} tokens from {doc_count} docs in {dt:.1f}s")
    print(f"  Binary file: {bin_path} ({file_size_gb:.2f} GB)")

    # Step 3: Save metadata and clean up HF cache
    meta = {
        'total_tokens': total_tokens,
        'vocab_size': actual_vocab,
        'source': 'openwebtext (HuggingFace)',
        'max_docs': args.max_docs,
        'doc_count': doc_count,
        'dtype': 'int32',
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    # Clean up HF cache
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(hf_cache):
        print(f"  Cleaning up HF cache ({hf_cache})...")
        shutil.rmtree(hf_cache, ignore_errors=True)

    print(f"\nDone. Data ready at {args.output_dir}")


if __name__ == '__main__':
    main()
