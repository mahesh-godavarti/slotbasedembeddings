#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Mahesh Godavarti. All Rights Reserved.
#
# License: This software is provided for non-commercial research purposes only.
# Patent Pending.
# Contact: m@qalaxia.com
# -----------------------------------------------------------------------------
#
# vit_imagenet.py — ViT for ImageNet-1K with DeiT-III training recipe.
# Compares positional encoding variants: learned, rope2d, joformer_old, etc.
#
# DeiT-III augmentation: RandAugment, Mixup, CutMix, Random Erasing,
# label smoothing, stochastic depth.

import argparse
import functools
import builtins
builtins.print = functools.partial(builtins.print, flush=True)
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment, RandomErasing
from torch.utils.data import DataLoader
from timm.data.mixup import Mixup

# Import all PE modules from existing code
from vit_cifar10 import (
    LearnedPE, RoPE2D, JoFormerOldPE, MonoidalAxialPE, JoFormerAxialPE,
    RoPE2Dv2, MonoidalPE, JoFormerPE, JoFormerFixedPE,
    Attention, TransformerBlock,
)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =====================================================================
# Stochastic Depth (Drop Path)
# =====================================================================

class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class TransformerBlockSD(nn.Module):
    """TransformerBlock with stochastic depth."""
    def __init__(self, embed_dim, n_heads, pe_module, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, n_heads, pe_module)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =====================================================================
# ViT with stochastic depth
# =====================================================================

class ViTImageNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=384, n_layers=6, n_heads=6,
                 pe_type="learned", n_classes=1000, drop_path_rate=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        n_patches = self.grid_h * self.grid_w

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                      kernel_size=patch_size, stride=patch_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding
        self.pe_type = pe_type
        _learnable_pe = {"monoidal_axial", "joformer_axial", "monoidal", "joformer"}
        PE_CLASSES = {
            "learned": LearnedPE,
            "rope2d": RoPE2D,
            "joformer_old": JoFormerOldPE,
            "monoidal_axial": MonoidalAxialPE,
            "joformer_axial": JoFormerAxialPE,
            "rope2dv2": RoPE2Dv2,
            "monoidal": MonoidalPE,
            "joformer": JoFormerPE,
            "joformer_fixed": JoFormerFixedPE,
        }
        if pe_type not in PE_CLASSES:
            raise ValueError(f"Unknown pe_type: {pe_type}")

        pe_cls = PE_CLASSES[pe_type]
        if pe_type == "learned":
            self.pe = pe_cls(n_patches, embed_dim)
        else:
            self.pe = pe_cls(embed_dim, self.grid_h, self.grid_w, n_heads)

        # Stochastic depth: linearly increasing drop rate per layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        if pe_type in _learnable_pe:
            self.blocks = nn.ModuleList([
                TransformerBlockSD(embed_dim, n_heads,
                                   pe_cls(embed_dim, self.grid_h, self.grid_w, n_heads),
                                   drop_path=dpr[i])
                for i in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlockSD(embed_dim, n_heads, self.pe, drop_path=dpr[i])
                for i in range(n_layers)
            ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# =====================================================================
# Data loading with DeiT-III augmentation
# =====================================================================

def get_imagenet_loaders(data_dir, batch_size, num_workers=8, seed=42):
    """Create ImageNet train/val DataLoaders with DeiT-III augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        RandomErasing(p=0.25),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        generator=g, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True,
    )

    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")

    return train_loader, val_loader


# =====================================================================
# Training
# =====================================================================

@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate on validation set."""
    model.eval()
    correct = 0
    correct_5 = 0
    total = 0
    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        correct += logits.argmax(1).eq(y).sum().item()
        _, pred5 = logits.topk(5, dim=1)
        correct_5 += pred5.eq(y.unsqueeze(1)).any(1).sum().item()
        total += y.size(0)
    return 100. * correct / total, 100. * correct_5 / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/ubuntu/cifar10_composition/data/imagenet")
    parser.add_argument("--pe_type", choices=[
        "learned", "rope2d", "joformer_old", "monoidal_axial", "joformer_axial",
        "rope2dv2", "monoidal", "joformer", "joformer_fixed"
    ], default="learned")
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.8)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (0 or 1)")
    parser.add_argument("--save_dir", type=str, default="/home/ubuntu/cifar10_composition/checkpoints")
    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluate every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    train_loader, val_loader = get_imagenet_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.seed
    )

    # Model
    model = ViTImageNet(
        img_size=args.img_size, patch_size=args.patch_size, in_channels=3,
        embed_dim=args.embed_dim, n_layers=args.n_layers,
        n_heads=args.n_heads, pe_type=args.pe_type, n_classes=1000,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    pe_params = sum(p.numel() for p in model.pe.parameters())
    grid_size = args.img_size // args.patch_size
    print(f"ViT-S with {args.pe_type} PE on ImageNet-1K (DeiT-III recipe)")
    print(f"  embed_dim={args.embed_dim}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"  patch_size={args.patch_size}, grid={grid_size}x{grid_size} = {grid_size**2} patches")
    print(f"  Total params: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  PE params: {pe_params:,}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}, Warmup: {args.warmup_epochs}")
    print(f"  LR: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"  Drop path: {args.drop_path_rate}, Label smoothing: {args.label_smoothing}")
    print(f"  Mixup: {args.mixup_alpha}, CutMix: {args.cutmix_alpha}")

    # Mixup / CutMix
    mixup_fn = Mixup(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        prob=args.mixup_prob,
        switch_prob=args.mixup_switch_prob,
        label_smoothing=args.label_smoothing,
        num_classes=1000,
    )

    # Optimizer
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or 'bias' in name or 'pos_embed' in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=args.lr, betas=(0.9, 0.999))

    # Cosine schedule with linear warmup
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss: soft cross-entropy for mixup (labels are soft)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # AMP
    scaler = torch.amp.GradScaler('cuda')

    best_acc1 = 0
    best_acc5 = 0
    start_epoch = 1

    # Resume from latest checkpoint
    latest_ckpt = os.path.join(args.save_dir, f'latest_{args.pe_type}.pt')
    if args.resume and os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc1 = ckpt['best_acc1']
        best_acc5 = ckpt['best_acc5']
        print(f"  Resumed at epoch {start_epoch}, best_top1={best_acc1:.2f}%")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0
        total = 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Apply Mixup/CutMix (produces soft labels)
            x, y_mixed = mixup_fn(x, y)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits, y_mixed)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            if batch_idx % 200 == 0 and batch_idx > 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                      f"loss={total_loss/total:.4f} lr={lr_now:.6f}")

        train_loss = total_loss / total
        elapsed = time.time() - t0

        # Evaluate (no mixup during eval)
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            acc1, acc5 = evaluate(model, val_loader, device)
            if acc1 > best_acc1:
                best_acc1 = acc1
                best_acc5 = acc5
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc1': acc1,
                    'acc5': acc5,
                    'pe_type': args.pe_type,
                }, os.path.join(args.save_dir, f'best_{args.pe_type}.pt'))
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} "
                  f"val_top1={acc1:.2f}% val_top5={acc5:.2f}% "
                  f"best_top1={best_acc1:.2f}% [{elapsed:.0f}s]")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} [{elapsed:.0f}s]")

        # Save latest checkpoint for resume (overwritten each epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc1': best_acc1,
            'best_acc5': best_acc5,
            'pe_type': args.pe_type,
            'args': vars(args),
        }, latest_ckpt)

    print(f"\nFinal: {args.pe_type} PE, Best Top-1: {best_acc1:.2f}%, Best Top-5: {best_acc5:.2f}%")


if __name__ == "__main__":
    main()
