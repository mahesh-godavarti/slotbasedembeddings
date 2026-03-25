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
# directional_composition_cifar10.py — Directional monoidal embeddings for CIFAR-10
#
# Extends the MNIST composition approach to CIFAR-10 (3-channel, 32x32).
# Each channel is composed independently, then concatenated.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math


class CompositionLayer(nn.Module):
    """Directional monoidal embedding for 2D images.

    Each pixel's value is placed into a (v, 0) vector pair, rotated by
    learnable per-axis angles theta_x[d]*i + theta_y[d]*j, and summed
    over all spatial positions. This generalizes the 2D DFT — the angles
    are initialized to DFT frequencies but learned during training.

    For multi-channel images, each channel gets its own set of angles.
    """

    def __init__(self, embedding_dim, n_channels=3, height=32, width=32):
        super().__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even"
        self.embedding_dim = embedding_dim
        self.n_channels = n_channels
        self.H = height
        self.W = width
        self.D2 = embedding_dim // 2

        # Fixed vector template: [1, 0, 1, 0, ...] for pairing into (x, y) rotation blocks
        self.register_buffer('vector', torch.tensor([1.0, 0.0] * self.D2))

        # Learnable angles per channel, initialized to DFT frequencies
        self.theta_x = nn.Parameter(
            2 * torch.pi * torch.arange(1, self.D2 + 1).float().unsqueeze(0).expand(n_channels, -1) / self.W
        )  # (C, D2)
        self.theta_y = nn.Parameter(
            2 * torch.pi * torch.arange(1, self.D2 + 1).float().unsqueeze(0).expand(n_channels, -1) / self.H
        )  # (C, D2)

        # Precompute spatial grid
        ii, jj = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        self.register_buffer('i_grid', ii.reshape(-1).float())  # (H*W,)
        self.register_buffer('j_grid', jj.reshape(-1).float())  # (H*W,)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image

        Returns:
            (B, C * embedding_dim) composed embedding
        """
        B, C, H, W = x.shape
        assert H == self.H and W == self.W and C == self.n_channels
        D2 = self.D2

        channel_outputs = []

        for c in range(C):
            # Get pixel values for this channel: (B, H*W)
            pixels = x[:, c].reshape(B, H * W)

            # Scale vector template by pixel values: (B, H*W, D2, 2)
            v = pixels.unsqueeze(-1) * self.vector  # (B, H*W, embedding_dim)
            v = v.view(B, H * W, D2, 2)

            # Compute rotation angles: theta_x[c] * i + theta_y[c] * j
            # i_grid: (H*W,), theta_x[c]: (D2,) -> (H*W, D2)
            theta_total = (
                self.i_grid[:, None] * self.theta_x[c][None, :] +
                self.j_grid[:, None] * self.theta_y[c][None, :]
            )  # (H*W, D2)

            cos_theta = torch.cos(theta_total)  # (H*W, D2)
            sin_theta = torch.sin(theta_total)

            # Apply rotation to each pixel's vector
            x_comp = v[..., 0]  # (B, H*W, D2)
            y_comp = v[..., 1]  # (B, H*W, D2)

            x_rot = x_comp * cos_theta.unsqueeze(0) + y_comp * sin_theta.unsqueeze(0)
            y_rot = -x_comp * sin_theta.unsqueeze(0) + y_comp * cos_theta.unsqueeze(0)

            # Sum over all spatial positions
            v_rot = torch.stack([x_rot, y_rot], dim=-1)  # (B, H*W, D2, 2)
            v_out = v_rot.sum(dim=1).reshape(B, self.embedding_dim)  # (B, embedding_dim)

            channel_outputs.append(v_out)

        # Concatenate all channels
        return torch.cat(channel_outputs, dim=1)  # (B, C * embedding_dim)


class CompositionClassifier(nn.Module):
    """CIFAR-10 classifier using directional monoidal embeddings."""

    def __init__(self, embedding_dim, n_channels=3):
        super().__init__()
        self.comp = CompositionLayer(embedding_dim, n_channels=n_channels)
        total_dim = n_channels * embedding_dim
        self.fc1 = nn.Linear(total_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        v = self.comp(x)
        x = F.relu(self.fc1(v))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += x.size(0)
    avg_loss = total_loss / total
    acc = 100. * correct / total
    print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, train_acc={acc:.2f}%")


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += x.size(0)
    acc = 100. * correct / total
    print(f"  Test Accuracy: {acc:.2f}%")
    return acc


def main():
    embedding_dim = 32
    batch_size = 128
    epochs = 50
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standard CIFAR-10 transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=256)

    model = CompositionClassifier(embedding_dim=embedding_dim, n_channels=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: CompositionClassifier, embedding_dim={embedding_dim}")
    print(f"Parameters: {n_params:,}")
    print(f"Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        acc = test(model, test_loader, device)
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_composition.pt')

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
