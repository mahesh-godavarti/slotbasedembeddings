#!/usr/bin/env python3
"""Run all PE variants on the same preloaded data.
Loads data once, trains each model sequentially."""

import functools
import builtins
builtins.print = functools.partial(builtins.print, flush=True)

import argparse
import torch
import torch.nn as nn
from torchvision import datasets

from vit_cifar10 import (ViT, load_cifar_to_tensors, get_batch, evaluate,
                          augment_batch, CIFAR_MEAN, CIFAR_STD)


def train_one(pe_type, train_x, train_y, test_x, test_y, args):
    """Train a single model and return results."""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = train_x.device
    n_train = train_x.shape[0]
    iters_per_epoch = n_train // args.batch_size

    model = ViT(
        img_size=32, patch_size=args.patch_size, in_channels=3,
        embed_dim=args.embed_dim, n_layers=args.n_layers,
        n_heads=args.n_heads, pe_type=pe_type, n_classes=args.n_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    pe_params = sum(p.numel() for p in model.pe.parameters())
    print(f"\n{'='*60}")
    print(f"ViT with {pe_type} PE  ({n_params:,} params, {pe_params} PE params)")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_iters = args.epochs * iters_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)
    criterion = nn.CrossEntropyLoss()

    results = []
    best_acc = 0
    it = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for _ in range(iters_per_epoch):
            x, y = get_batch(train_x, train_y, args.batch_size)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * x.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += x.size(0)
            it += 1

        train_loss = total_loss / total
        train_acc = 100. * correct / total
        test_acc = evaluate(model, test_x, test_y)
        if test_acc > best_acc:
            best_acc = test_acc

        results.append({
            'epoch': epoch, 'train_loss': train_loss,
            'train_acc': train_acc, 'test_acc': test_acc, 'best': best_acc
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%, "
                  f"test_acc={test_acc:.2f}%, best={best_acc:.2f}%")

    print(f"Final: {pe_type} PE, Best Test Accuracy: {best_acc:.2f}%")
    return best_acc, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--models", nargs='+',
                        default=["learned", "rope2d", "joformer_old",
                                 "monoidal_axial", "joformer_axial",
                                 "rope2dv2", "monoidal", "joformer", "joformer_fixed"])
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data ONCE
    dataset_class = datasets.CIFAR100 if args.dataset == "cifar100" else datasets.CIFAR10
    args.n_classes = 100 if args.dataset == "cifar100" else 10
    train_x, train_y, test_x, test_y = load_cifar_to_tensors(dataset_class)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    print(f"Loaded {train_x.shape[0]} train, {test_x.shape[0]} test images on {device}")

    # Train each model
    all_results = {}
    for pe_type in args.models:
        best_acc, results = train_one(pe_type, train_x, train_y, test_x, test_y, args)
        all_results[pe_type] = {'best': best_acc, 'curve': results}

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.epochs} epochs, seed={args.seed})")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Best Acc':>10}")
    print("-" * 32)
    for pe_type in args.models:
        print(f"{pe_type:<20} {all_results[pe_type]['best']:>10.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
