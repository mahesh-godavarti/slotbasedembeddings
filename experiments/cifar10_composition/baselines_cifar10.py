#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Mahesh Godavarti. All Rights Reserved.
#
# License: This software is provided for non-commercial research purposes only.
# Contact: m@qalaxia.com
# -----------------------------------------------------------------------------
#
# baselines_cifar10.py — Baseline models for CIFAR-10 comparison
#
# DFT features + logistic regression, MLP, and CNN baselines.

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math


class DFTClassifier(nn.Module):
    """2D DFT features (fixed, not learned) + linear classifier.

    Takes the top-k low-frequency DFT coefficients per channel,
    concatenates magnitudes, and classifies with a linear layer.
    """

    def __init__(self, n_features=32, n_channels=3):
        super().__init__()
        self.n_features = n_features
        self.n_channels = n_channels
        self.fc = nn.Linear(n_channels * n_features, 10)

    def forward(self, x):
        B, C, H, W = x.shape
        features = []
        for c in range(C):
            # 2D FFT
            fft = torch.fft.fft2(x[:, c])
            # Take low-frequency coefficients (top-left corner)
            n_side = int(math.ceil(math.sqrt(self.n_features)))
            low_freq = fft[:, :n_side, :n_side]
            magnitudes = torch.abs(low_freq).reshape(B, -1)[:, :self.n_features]
            features.append(magnitudes)
        features = torch.cat(features, dim=1)
        return self.fc(features)


class MLPClassifier(nn.Module):
    """Standard MLP on flattened pixels."""

    def __init__(self, hidden=256, n_channels=3, height=32, width=32):
        super().__init__()
        input_dim = n_channels * height * width
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNNClassifier(nn.Module):
    """Simple CNN baseline."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += x.size(0)
    print(f"Epoch {epoch}: train_loss={total_loss/total:.4f}, train_acc={100.*correct/total:.2f}%")


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
            total += x.size(0)
    acc = 100. * correct / total
    print(f"  Test Accuracy: {acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dft", "mlp", "cnn"], required=True)
    parser.add_argument("--n_features", type=int, default=32,
                        help="DFT feature count per channel")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=256)

    if args.model == "dft":
        model = DFTClassifier(n_features=args.n_features).to(device)
    elif args.model == "mlp":
        model = MLPClassifier().to(device)
    elif args.model == "cnn":
        model = CNNClassifier().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Parameters: {n_params:,}, Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        acc = test(model, test_loader, device)
        scheduler.step()
        if acc > best_acc:
            best_acc = acc

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
