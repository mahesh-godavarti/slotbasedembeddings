#!/usr/bin/env python3
"""Speech Commands v2 keyword spotting: CNN baselines + rotation-based SSM models."""

import argparse
import os
import random
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

# ─── Constants ───────────────────────────────────────────────────────────────

TARGET_COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = TARGET_COMMANDS + ["unknown", "silence"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)
SAMPLE_RATE = 16000
NUM_SAMPLES = 16000  # 1 second


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SpeechCommandsDataset(Dataset):
    """Wraps torchaudio SPEECHCOMMANDS with unknown subsampling and silence generation."""

    def __init__(self, root, subset, augment=False, spec_augment=False,
                 unknown_cap=None, noise_wavs=None, time_shift=0,
                 time_stretch=False, split_stretch=False, distortion=False,
                 stretch_range=(0.8, 1.2)):
        self.subset = subset
        self.augment = augment and (subset == "training")
        self.spec_augment = spec_augment and (subset == "training")
        self.time_shift = time_shift if subset != "validation" else 0  # shift train+test, not val
        self.time_stretch = time_stretch and (subset == "training")
        self.split_stretch = split_stretch and (subset == "training")
        self.distortion = distortion and (subset == "training")
        self.stretch_range = stretch_range
        self.samples = []  # list of (waveform_getter, label_idx)

        # Load the torchaudio dataset
        ds = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset=subset)

        # Collect by label
        target_samples = []
        unknown_samples = []
        for i in range(len(ds)):
            waveform, sr, label, *_ = ds[i]
            if label in TARGET_COMMANDS:
                target_samples.append((waveform, LABEL2IDX[label]))
            else:
                unknown_samples.append((waveform, LABEL2IDX["unknown"]))

        # Subsample unknowns
        if unknown_cap is None:
            # Match average count per target command
            avg_per_cmd = len(target_samples) // len(TARGET_COMMANDS)
            unknown_cap = avg_per_cmd
        if len(unknown_samples) > unknown_cap:
            random.shuffle(unknown_samples)
            unknown_samples = unknown_samples[:unknown_cap]

        self.samples = target_samples + unknown_samples

        # Generate silence samples from background noise
        if noise_wavs is not None and len(noise_wavs) > 0:
            n_silence = unknown_cap
            silence_samples = []
            for _ in range(n_silence):
                noise = random.choice(noise_wavs)
                start = random.randint(0, noise.shape[-1] - NUM_SAMPLES)
                clip = noise[:, start:start + NUM_SAMPLES]
                # Scale down
                clip = clip * random.uniform(0.0, 0.5)
                silence_samples.append((clip, LABEL2IDX["silence"]))
            self.samples += silence_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, label = self.samples[idx]
        waveform = self._pad_or_truncate(waveform)

        if self.augment:
            waveform = self._augment(waveform)

        if self.time_shift > 0:
            shift = random.randint(-self.time_shift, self.time_shift)
            if shift > 0:
                waveform = F.pad(waveform[:, shift:], (0, shift))
            elif shift < 0:
                waveform = F.pad(waveform[:, :shift], (-shift, 0))

        return waveform.squeeze(0), label  # (16000,), int

    def _pad_or_truncate(self, waveform):
        if waveform.shape[-1] > NUM_SAMPLES:
            waveform = waveform[:, :NUM_SAMPLES]
        elif waveform.shape[-1] < NUM_SAMPLES:
            pad = NUM_SAMPLES - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad))
        return waveform

    def _augment(self, waveform):
        # Time shift ±100ms (±1600 samples)
        shift = random.randint(-1600, 1600)
        if shift > 0:
            waveform = F.pad(waveform[:, shift:], (0, shift))
        elif shift < 0:
            waveform = F.pad(waveform[:, :shift], (-shift, 0))

        # Random gain ±2dB
        gain_db = random.uniform(-2.0, 2.0)
        waveform = waveform * (10 ** (gain_db / 20.0))

        # Nonlinear distortion: normalize, apply x + alpha*x^3, clip
        if self.distortion:
            alpha = random.uniform(0.0, 5.0)
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak
            waveform = waveform + alpha * waveform.pow(3)
            waveform = waveform.clamp(-1.0, 1.0)

        # Time-stretch via resampling
        if self.time_stretch:
            stretch = random.uniform(self.stretch_range[0], self.stretch_range[1])
            new_len = int(waveform.shape[-1] * stretch)
            waveform = F.interpolate(waveform.unsqueeze(0), size=new_len, mode='linear', align_corners=False).squeeze(0)
            if waveform.shape[-1] > NUM_SAMPLES:
                waveform = waveform[:, :NUM_SAMPLES]
            else:
                waveform = F.pad(waveform, (0, NUM_SAMPLES - waveform.shape[-1]))

        # Split-stretch: split in half, independent stretch each half, reassemble
        if self.split_stretch:
            half = waveform.shape[-1] // 2
            w1 = waveform[:, :half]
            w2 = waveform[:, half:]
            s1 = random.uniform(0.8, 1.2)
            s2 = random.uniform(0.8, 1.2)
            w1 = F.interpolate(w1.unsqueeze(0), size=int(half * s1), mode='linear', align_corners=False).squeeze(0)
            w2 = F.interpolate(w2.unsqueeze(0), size=int(half * s2), mode='linear', align_corners=False).squeeze(0)
            waveform = torch.cat([w1, w2], dim=-1)
            if waveform.shape[-1] > NUM_SAMPLES:
                waveform = waveform[:, :NUM_SAMPLES]
            else:
                waveform = F.pad(waveform, (0, NUM_SAMPLES - waveform.shape[-1]))

        return waveform


def load_noise_wavs(root):
    """Load background noise files for silence generation."""
    noise_dir = Path(root) / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"
    if not noise_dir.exists():
        return []
    wavs = []
    for f in noise_dir.glob("*.wav"):
        waveform, sr = torchaudio.load(str(f))
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        wavs.append(waveform)
    return wavs


# ─── Models ──────────────────────────────────────────────────────────────────

class ResBlock1d(nn.Module):
    """Residual block with 1D convolutions."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class LearnedSpecCNN(nn.Module):
    """CNN with learned frequency decomposition replacing FFT + mel filterbank.
    Uses windowed cumsum with learnable frequencies instead of fixed FFT.
    FFT is a special case (equally spaced frequencies). We learn the frequencies,
    so they can concentrate on perceptually important ranges.
    Init: mel-scale center frequencies (matching standard mel filterbank)."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, freeze_freqs=False, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.window = window
        self.hop = hop_length
        # Initialize frequencies at mel-scale center frequencies
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]  # n_freqs center frequencies in Hz
        # Angular frequencies: ω = 2π·f/fs
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        # Learn in log space (frequencies span wide range)
        log_freqs = torch.log(angular_freqs)
        if freeze_freqs:
            self.register_buffer('log_freqs', log_freqs)
        else:
            self.log_freqs = nn.Parameter(log_freqs)

        # Same CNN architecture as MelCNN
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        B_batch, T = x.shape
        freqs = self.log_freqs.exp()  # (n_freqs,)

        # Learned STFT via windowed cumsum
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)  # (T, n_freqs)

        x_complex = x.to(torch.complex64).unsqueeze(-1)  # (B, T, 1)
        rotated = x_complex * phases.conj().unsqueeze(0)  # (B, T, n_freqs)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted  # (B, T, n_freqs) — windowed frequency response

        # Power spectrum + subsample by hop
        power = d.real ** 2 + d.imag ** 2  # (B, T, n_freqs)
        power = power[:, ::self.hop]  # (B, ~100, n_freqs)

        # Log power spectrogram
        spec = torch.log(power + 1e-8)

        # (B, n_freqs, T_frames) for CNN
        spec = spec.transpose(1, 2)

        # CNN (same as MelCNN)
        x = self.conv_in(spec)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LearnedSpecMagReImCNN(nn.Module):
    """LearnedSpecCNN but feeds [log_mag, re, im] → Linear → n_freqs into CNN."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, freeze_freqs=False, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.window = window
        self.hop = hop_length
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        log_freqs = torch.log(angular_freqs)
        if freeze_freqs:
            self.register_buffer('log_freqs', log_freqs)
        else:
            self.log_freqs = nn.Parameter(log_freqs)
        self.embed = nn.Linear(3 * n_freqs, n_freqs)

        # Same CNN architecture as MelCNN
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        B_batch, T = x.shape
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj().unsqueeze(0)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted

        # [log_mag, re, im] → linear → n_freqs
        mag = d.real ** 2 + d.imag ** 2
        log_mag = (mag + 1e-8).log()
        features = torch.cat([log_mag, d.real, d.imag], dim=-1)  # (B, T, 3*n_freqs)
        features = features[:, ::self.hop]  # (B, ~100, 3*n_freqs)
        spec = self.embed(features)  # (B, ~100, n_freqs)
        spec = spec.transpose(1, 2)  # (B, n_freqs, T_frames)

        x = self.conv_in(spec)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LearnedSpecMultiCNN(nn.Module):
    """LearnedSpecCNN with multiple frequencies per bin.
    n_per_bin frequencies per mel bin, all initialized to the same center freq.
    Power of each group is summed → n_bins output channels (like mel averaging FFT bins).
    With n_per_bin=2: 80 learned frequencies → 40 power channels."""
    def __init__(self, n_bins=40, n_per_bin=2, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_bins = n_bins
        self.n_per_bin = n_per_bin
        self.n_freqs = n_bins * n_per_bin
        self.window = window
        self.hop = hop_length
        # Initialize: n_per_bin frequencies per mel center, all starting at same value
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_bins + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]  # (n_bins,)
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        # Repeat each center freq n_per_bin times
        log_freqs = torch.log(angular_freqs).repeat_interleave(n_per_bin)
        self.log_freqs = nn.Parameter(log_freqs)

        # Same CNN as MelCNN (n_bins input channels)
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_bins, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        B_batch, T = x.shape
        freqs = self.log_freqs.exp()  # (n_freqs,)

        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)  # (T, n_freqs)

        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj().unsqueeze(0)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted

        # Power per frequency
        power = d.real ** 2 + d.imag ** 2  # (B, T, n_freqs)
        power = power[:, ::self.hop]  # (B, T_frames, n_freqs)

        # Sum power within each bin: (B, T_frames, n_bins)
        power = power.view(power.shape[0], power.shape[1], self.n_bins, self.n_per_bin)
        power = power.sum(dim=-1)

        spec = torch.log(power + 1e-8)
        spec = spec.transpose(1, 2)  # (B, n_bins, T_frames)

        x = self.conv_in(spec)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LearnedSpecLinearCNN(nn.Module):
    """LearnedSpecCNN without d² — stays linear.
    Windowed cumsum → unrotate → cat(real, imag) → stride → CNN.
    80 channels (2×40) instead of 40. No squaring, no log."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.window = window
        self.hop = hop_length
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))

        # 2*n_freqs input channels (real + imag)
        self.conv_in = nn.Sequential(
            nn.Conv1d(2 * n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        B_batch, T = x.shape
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)

        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj().unsqueeze(0)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)  # unrotate

        h = torch.cat([d.real, d.imag], dim=-1)  # (B, T, 2*n_freqs)
        h = h[:, ::self.hop]  # (B, ~100, 2*n_freqs)
        h = h.transpose(1, 2)  # (B, 2*n_freqs, T_frames)

        x = self.conv_in(h)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LearnedSpecCNNMod(nn.Module):
    """LearnedSpecCNN with data-dependent frequencies.
    Each sample x[t] modulates the base frequencies via Linear(1 → n_freqs).
    Phase = cumsum(base_freq + offset(x[t])), rest identical to LearnedSpecCNN."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.window = window
        self.hop = hop_length
        # Data-dependent frequencies: scalar → n_freqs → LayerNorm
        self.freq_proj = nn.Linear(1, n_freqs)
        self.freq_ln = nn.LayerNorm(n_freqs)

        # Same CNN architecture as MelCNN
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        B_batch, T = x.shape

        # Data-dependent frequencies
        inst_freqs = self.freq_ln(self.freq_proj(x.unsqueeze(-1)))  # (B, T, n_freqs)
        # Cumulative phase
        cum_phase = inst_freqs.cumsum(dim=1)  # (B, T, n_freqs)
        phases = torch.exp(1j * cum_phase)  # (B, T, n_freqs)

        x_complex = x.to(torch.complex64).unsqueeze(-1)  # (B, T, 1)
        rotated = x_complex * phases.conj()  # (B, T, n_freqs)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted

        # Power spectrum + subsample by hop
        power = d.real ** 2 + d.imag ** 2  # (B, T, n_freqs)
        power = power[:, ::self.hop]

        # Log power spectrogram → CNN
        spec = torch.log(power + 1e-8)
        spec = spec.transpose(1, 2)  # (B, n_freqs, T_frames)

        x = self.conv_in(spec)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LearnedSpecCNNMod2(nn.Module):
    """LearnedSpecCNNMod with deeper frequency network:
    LayerNorm(Linear(ReLU(Linear(x[t]))))."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.window = window
        self.hop = hop_length
        # Data-dependent frequencies: scalar → n_freqs → ReLU → n_freqs → LayerNorm
        self.freq_proj = nn.Sequential(
            nn.Linear(1, n_freqs),
            nn.ReLU(),
            nn.Linear(n_freqs, n_freqs),
            nn.LayerNorm(n_freqs),
        )

        # Same CNN architecture as MelCNN
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        B_batch, T = x.shape
        inst_freqs = self.freq_proj(x.unsqueeze(-1))  # (B, T, n_freqs)
        cum_phase = inst_freqs.cumsum(dim=1)
        phases = torch.exp(1j * cum_phase)

        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj()
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted

        power = d.real ** 2 + d.imag ** 2
        power = power[:, ::self.hop]
        spec = torch.log(power + 1e-8)
        spec = spec.transpose(1, 2)

        x = self.conv_in(spec)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LearnedSpecCNNConv(nn.Module):
    """LearnedSpecCNN with data-dependent frequencies from local context.
    Conv1d(1, n_freqs, kernel_size=80) extracts per-timestep frequencies
    from ~5ms of local waveform context."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, conv_k=80,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.window = window
        self.hop = hop_length
        # Data-dependent frequencies from local context
        self.freq_conv = nn.Conv1d(1, n_freqs, kernel_size=conv_k,
                                   padding=conv_k // 2, bias=True)
        self.conv_k = conv_k
        self.freq_ln = nn.LayerNorm(n_freqs)

        # Same CNN architecture as MelCNN
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        B_batch, T = x.shape
        # Conv1d: (B, 1, T) → (B, n_freqs, T+1) → trim → transpose → (B, T, n_freqs)
        inst_freqs = self.freq_conv(x.unsqueeze(1))[:, :, :T].transpose(1, 2)
        inst_freqs = self.freq_ln(inst_freqs)  # (B, T, n_freqs)
        # Cumulative phase
        cum_phase = inst_freqs.cumsum(dim=1)
        phases = torch.exp(1j * cum_phase)  # (B, T, n_freqs)

        x_complex = x.to(torch.complex64).unsqueeze(-1)  # (B, T, 1)
        rotated = x_complex * phases.conj()
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted

        # Power spectrum + subsample by hop
        power = d.real ** 2 + d.imag ** 2
        power = power[:, ::self.hop]
        spec = torch.log(power + 1e-8)
        spec = spec.transpose(1, 2)  # (B, n_freqs, T_frames)

        x = self.conv_in(spec)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FilterbankCNN(nn.Module):
    """Learned filterbank: Conv1d(1, n_filters, k=window, stride=hop) → x² → log → CNN.
    Like LearnedSpecCNN but learns arbitrary filters instead of sinusoidal basis."""
    def __init__(self, n_filters=40, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.filterbank = nn.Conv1d(1, n_filters, kernel_size=window,
                                     stride=hop_length, bias=False)
        # Same CNN backend as MelCNN
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_filters, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        x = x.unsqueeze(1)  # (B, 1, 16000)
        x = self.filterbank(x)  # (B, n_filters, ~100)
        x = x ** 2  # power
        x = torch.log(x + 1e-8)  # log power
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FilterbankMelInitCNN(nn.Module):
    """FilterbankCNN with Hann-tapered mel-scale sinusoid initialization.
    Conv filters start as windowed sinusoids at mel frequencies, then learn
    arbitrary shapes via backprop."""
    def __init__(self, n_filters=40, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.filterbank = nn.Conv1d(1, n_filters, kernel_size=window,
                                     stride=hop_length, bias=False)
        # Initialize to Hann-tapered mel-scale sinusoids
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_filters + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        t = torch.arange(window, dtype=torch.float32) / SAMPLE_RATE
        hann = torch.hann_window(window)
        with torch.no_grad():
            for i in range(n_filters):
                self.filterbank.weight.data[i, 0] = hann * torch.sin(
                    2 * math.pi * center_freqs[i] * t)

        # Same CNN backend
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_filters, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.filterbank(x)
        x = x ** 2
        x = torch.log(x + 1e-8)
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FilterbankSinCosCNN(nn.Module):
    """Filterbank with sin+cos pairs at each mel frequency.
    80 conv filters (40 sin + 40 cos), then sin² + cos² = magnitude² per freq → 40 channels.
    Phase-invariant power, like a proper DFT magnitude."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, freeze_filters=False, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.filterbank = nn.Conv1d(1, 2 * n_freqs, kernel_size=window,
                                     stride=hop_length, bias=False)
        # Initialize: filter[2i] = hann*sin, filter[2i+1] = hann*cos
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        t = torch.arange(window, dtype=torch.float32) / SAMPLE_RATE
        hann = torch.hann_window(window)
        with torch.no_grad():
            for i in range(n_freqs):
                self.filterbank.weight.data[2*i, 0] = hann * torch.sin(
                    2 * math.pi * center_freqs[i] * t)
                self.filterbank.weight.data[2*i+1, 0] = hann * torch.cos(
                    2 * math.pi * center_freqs[i] * t)
        if freeze_filters:
            self.filterbank.weight.requires_grad = False

        self.conv_in = nn.Sequential(
            nn.Conv1d(n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 16000)
        x = self.filterbank(x)  # (B, 2*n_freqs, T)
        # sin² + cos² per frequency pair → magnitude²
        x = x.view(x.shape[0], self.n_freqs, 2, x.shape[2])  # (B, n_freqs, 2, T)
        x = (x ** 2).sum(dim=2)  # (B, n_freqs, T)
        x = torch.log(x + 1e-8)
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FilterbankSinCosMultiCNN(nn.Module):
    """Sin+cos filterbank with multiple frequencies per bin.
    n_per_bin sin+cos pairs per mel bin → d²+log per pair → sum within bin → n_bins channels.
    With n_per_bin=2: 160 conv filters → 80 magnitudes → 40 channels."""
    def __init__(self, n_bins=40, n_per_bin=2, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_bins = n_bins
        self.n_per_bin = n_per_bin
        self.n_freqs = n_bins * n_per_bin
        self.filterbank = nn.Conv1d(1, 2 * self.n_freqs, kernel_size=window,
                                     stride=hop_length, bias=False)
        # Initialize: n_per_bin sin+cos pairs per mel center, all starting at same freq
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_bins + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]  # (n_bins,)
        t = torch.arange(window, dtype=torch.float32) / SAMPLE_RATE
        hann = torch.hann_window(window)
        with torch.no_grad():
            for i in range(n_bins):
                for j in range(n_per_bin):
                    idx = i * n_per_bin + j
                    self.filterbank.weight.data[2*idx, 0] = hann * torch.sin(
                        2 * math.pi * center_freqs[i] * t)
                    self.filterbank.weight.data[2*idx+1, 0] = hann * torch.cos(
                        2 * math.pi * center_freqs[i] * t)

        self.conv_in = nn.Sequential(
            nn.Conv1d(n_bins, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 16000)
        x = self.filterbank(x)  # (B, 2*n_freqs, T)
        # Pair up sin+cos: (B, n_freqs, 2, T)
        x = x.view(x.shape[0], self.n_freqs, 2, x.shape[2])
        # d²+log per pair
        mag = (x ** 2).sum(dim=2)  # (B, n_freqs, T)
        mag = torch.log(mag + 1e-8)
        # Sum within each bin: (B, n_bins, T)
        mag = mag.view(mag.shape[0], self.n_bins, self.n_per_bin, mag.shape[2])
        mag = mag.sum(dim=2)
        x = self.conv_in(mag)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FilterbankSinCosLinearCNN(nn.Module):
    """Sin+cos filterbank, no d² — all 80 channels (40 sin + 40 cos) straight to CNN."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.filterbank = nn.Conv1d(1, 2 * n_freqs, kernel_size=window,
                                     stride=hop_length, bias=False)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        t = torch.arange(window, dtype=torch.float32) / SAMPLE_RATE
        hann = torch.hann_window(window)
        with torch.no_grad():
            for i in range(n_freqs):
                self.filterbank.weight.data[2*i, 0] = hann * torch.sin(
                    2 * math.pi * center_freqs[i] * t)
                self.filterbank.weight.data[2*i+1, 0] = hann * torch.cos(
                    2 * math.pi * center_freqs[i] * t)

        self.conv_in = nn.Sequential(
            nn.Conv1d(2 * n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.filterbank(x)  # (B, 80, T)
        # No d², no log — raw sin+cos outputs
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FilterbankSinCosCombinedCNN(nn.Module):
    """Sin+cos filterbank with both magnitude and raw phase channels.
    80 conv filters (40 sin + 40 cos) → 40 log(sin²+cos²) + 80 raw sin+cos = 120 channels to CNN.
    CNN gets magnitude features it likes plus raw phase information."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.filterbank = nn.Conv1d(1, 2 * n_freqs, kernel_size=window,
                                     stride=hop_length, bias=False)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        t = torch.arange(window, dtype=torch.float32) / SAMPLE_RATE
        hann = torch.hann_window(window)
        with torch.no_grad():
            for i in range(n_freqs):
                self.filterbank.weight.data[2*i, 0] = hann * torch.sin(
                    2 * math.pi * center_freqs[i] * t)
                self.filterbank.weight.data[2*i+1, 0] = hann * torch.cos(
                    2 * math.pi * center_freqs[i] * t)

        # 120 input channels: 40 log-magnitude + 80 raw sin+cos
        self.conv_in = nn.Sequential(
            nn.Conv1d(3 * n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 16000)
        x = self.filterbank(x)  # (B, 2*n_freqs, T)
        # Compute magnitude: sin² + cos² per freq pair → log
        x_pairs = x.view(x.shape[0], self.n_freqs, 2, x.shape[2])  # (B, n_freqs, 2, T)
        mag = (x_pairs ** 2).sum(dim=2)  # (B, n_freqs, T)
        mag = torch.log(mag + 1e-8)
        # Concatenate: 40 log-magnitude + 80 raw sin+cos = 120 channels
        x = torch.cat([mag, x], dim=1)  # (B, 3*n_freqs, T)
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FilterbankSinCosMagReImCNN(nn.Module):
    """Sin+cos filterbank → [log_mag, sin, cos] → Linear(3*n_freqs, n_freqs) → CNN.
    Like LearnedSpecMagReImCNN but using conv filterbank instead of cumsum."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, freeze_filters=False, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.filterbank = nn.Conv1d(1, 2 * n_freqs, kernel_size=window,
                                     stride=hop_length, bias=False)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        t = torch.arange(window, dtype=torch.float32) / SAMPLE_RATE
        hann = torch.hann_window(window)
        with torch.no_grad():
            for i in range(n_freqs):
                self.filterbank.weight.data[2*i, 0] = hann * torch.sin(
                    2 * math.pi * center_freqs[i] * t)
                self.filterbank.weight.data[2*i+1, 0] = hann * torch.cos(
                    2 * math.pi * center_freqs[i] * t)
        if freeze_filters:
            self.filterbank.weight.requires_grad = False

        # Linear bottleneck: [log_mag, sin, cos] → n_freqs
        self.embed = nn.Linear(3 * n_freqs, n_freqs)

        self.conv_in = nn.Sequential(
            nn.Conv1d(n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 16000)
        x = self.filterbank(x)  # (B, 2*n_freqs, T)
        # Split into sin/cos pairs
        x_pairs = x.view(x.shape[0], self.n_freqs, 2, x.shape[2])  # (B, n_freqs, 2, T)
        sin_out = x_pairs[:, :, 0]  # (B, n_freqs, T)
        cos_out = x_pairs[:, :, 1]  # (B, n_freqs, T)
        mag = sin_out ** 2 + cos_out ** 2
        log_mag = torch.log(mag + 1e-8)
        # (B, T, 3*n_freqs)
        features = torch.cat([log_mag, sin_out, cos_out], dim=1).transpose(1, 2)
        # Linear bottleneck → (B, T, n_freqs)
        spec = self.embed(features).transpose(1, 2)  # (B, n_freqs, T)
        x = self.conv_in(spec)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FilterbankLinearCNN(nn.Module):
    """Learned filterbank without d² or log: Conv1d(1, n_filters, k, s) → CNN directly.
    Tests whether the CNN can learn from raw filterbank output."""
    def __init__(self, n_filters=40, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.filterbank = nn.Conv1d(1, n_filters, kernel_size=window,
                                     stride=hop_length, bias=False)
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_filters, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 16000)
        x = self.filterbank(x)  # (B, n_filters, ~100)
        # No d², no log — raw filterbank output
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FilterbankMelInitLinearCNN(nn.Module):
    """FilterbankLinearCNN with Hann-tapered mel-scale sinusoid initialization.
    No d², no log — raw filterbank output straight to CNN."""
    def __init__(self, n_filters=40, window=400, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.filterbank = nn.Conv1d(1, n_filters, kernel_size=window,
                                     stride=hop_length, bias=False)
        # Initialize to Hann-tapered mel-scale sinusoids
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_filters + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        t = torch.arange(window, dtype=torch.float32) / SAMPLE_RATE
        hann = torch.hann_window(window)
        with torch.no_grad():
            for i in range(n_filters):
                self.filterbank.weight.data[i, 0] = hann * torch.sin(
                    2 * math.pi * center_freqs[i] * t)

        self.conv_in = nn.Sequential(
            nn.Conv1d(n_filters, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.filterbank(x)
        # No d², no log — raw filterbank output
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LearnedSpecCNNConv2(nn.Module):
    """LearnedSpecCNNConv with deeper frequency network:
    Conv1d(1→40, k=80) → Linear → ReLU → Linear → LayerNorm."""
    def __init__(self, n_freqs=40, window=400, hop_length=160, conv_k=80,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.window = window
        self.hop = hop_length
        # Conv filterbank → nonlinear transform → LayerNorm
        self.freq_conv = nn.Conv1d(1, n_freqs, kernel_size=conv_k,
                                   padding=conv_k // 2, bias=True)
        self.conv_k = conv_k
        self.freq_mlp = nn.Sequential(
            nn.Linear(n_freqs, n_freqs),
            nn.ReLU(),
            nn.Linear(n_freqs, n_freqs),
            nn.LayerNorm(n_freqs),
        )

        # Same CNN architecture as MelCNN
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_freqs, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        B_batch, T = x.shape
        # Conv1d → MLP → frequencies
        inst_freqs = self.freq_conv(x.unsqueeze(1))[:, :, :T].transpose(1, 2)
        inst_freqs = self.freq_mlp(inst_freqs)  # (B, T, n_freqs)
        # Cumulative phase
        cum_phase = inst_freqs.cumsum(dim=1)
        phases = torch.exp(1j * cum_phase)

        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj()
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted

        power = d.real ** 2 + d.imag ** 2
        power = power[:, ::self.hop]
        spec = torch.log(power + 1e-8)
        spec = spec.transpose(1, 2)

        x = self.conv_in(spec)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelCNN(nn.Module):
    """TC-ResNet style: mel bins as channels, 1D temporal convolutions."""
    def __init__(self, n_mels=40, hop_length=160, num_classes=NUM_CLASSES):
        super().__init__()
        self.hop_length = hop_length
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=hop_length,
            n_mels=n_mels, power=2.0,
        )
        # Input: (B, n_mels, T=101) — treat mel bins as channels
        self.conv_in = nn.Sequential(
            nn.Conv1d(n_mels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(16, 24),
            nn.MaxPool1d(2),
            ResBlock1d(24, 32),
            nn.MaxPool1d(2),
            ResBlock1d(32, 48),
            nn.MaxPool1d(2),
        )
        # Baseline (hop=160) gives ~12 frames after ResBlocks.
        # With finer hop, avg pool in groups of ~12, then max over groups.
        # N=1 at hop=160 matches standard AdaptiveAvgPool1d(1).
        baseline_frames = (SAMPLE_RATE // 160 + 1) // 8  # ~12
        self.avg_kernel = baseline_frames
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        x = x.unsqueeze(1)  # (B, 1, 16000)
        x = self.mel_spec(x)  # (B, 1, n_mels, T)
        x = x.squeeze(1)  # (B, n_mels, T)
        x = (x + 1e-8).log()  # log mel
        x = self.conv_in(x)
        x = self.blocks(x)
        # Avg pool in groups, then max over groups
        T = x.shape[2]
        if T <= self.avg_kernel:
            # N=1: global avg pool (matches standard behavior)
            x = x.mean(dim=2)
        else:
            x = F.avg_pool1d(x, kernel_size=self.avg_kernel, stride=self.avg_kernel)
            x = x.max(dim=2).values
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RawCNN(nn.Module):
    """M5-style raw waveform CNN."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=80, stride=16, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        x = x.unsqueeze(1)  # (B, 1, 16000)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── SpecAugment ─────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    """Frequency and time masking for mel spectrograms."""
    def __init__(self, freq_mask=7, time_mask=25):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask)

    def forward(self, x):
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x


class MelCNNWithSpecAugment(MelCNN):
    """MelCNN with SpecAugment applied during training."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec_aug = SpecAugment()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = x.squeeze(1)
        x = (x + 1e-8).log()
        if self.training:
            x = self.spec_aug(x)
        x = self.conv_in(x)
        x = self.blocks(x)
        T = x.shape[2]
        if T <= self.avg_kernel:
            x = x.mean(dim=2)
        else:
            x = F.avg_pool1d(x, kernel_size=self.avg_kernel, stride=self.avg_kernel)
            x = x.max(dim=2).values
        x = self.fc(x)
        return x


class MelCNNMaxPool(MelCNN):
    """MelCNN with SpecAugment and global MaxPool instead of AvgPool."""
    def __init__(self, *args, no_spec_aug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec_aug = None if no_spec_aug else SpecAugment()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = x.squeeze(1)
        x = (x + 1e-8).log()
        if self.training and self.spec_aug is not None:
            x = self.spec_aug(x)
        x = self.conv_in(x)
        x = self.blocks(x)
        x = x.max(dim=2).values  # global max pool
        x = self.fc(x)
        return x


class MelCNNMultiPhase(MelCNN):
    """MelCNN with finer hop, interleaved subsequences, and max over outputs.
    hop=80: 2 phase-shifted views (even/odd frames), same CNN, max over outputs.
    hop=40: 4 phase-shifted views, same CNN, max over outputs.
    Each view has ~101 frames, preserving CNN receptive field."""
    def __init__(self, hop_length=80, **kwargs):
        base_hop = 160
        assert base_hop % hop_length == 0, f"base hop {base_hop} must be divisible by {hop_length}"
        self.n_phases = base_hop // hop_length
        super().__init__(hop_length=hop_length, **kwargs)
        self.spec_aug = SpecAugment()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = x.squeeze(1)          # (B, n_mels, T_fine)
        x = (x + 1e-8).log()
        if self.training:
            x = self.spec_aug(x)

        # Split into n_phases interleaved subsequences
        outputs = []
        for p in range(self.n_phases):
            x_phase = x[:, :, p::self.n_phases]  # (B, n_mels, ~101)
            h = self.conv_in(x_phase)
            h = self.blocks(h)
            h = h.max(dim=2).values               # global max pool
            outputs.append(self.fc(h))             # (B, 12)

        # Max over phase-shifted outputs
        return torch.stack(outputs, dim=0).max(dim=0).values


# ─── Rotation primitives (from experiments/kg_text_experiment.py) ────────────

def apply_rotation(x, angles):
    """Apply 2D rotation matrices to pairs of dimensions.
    x: (B, T, C), angles: (B, T, C//2) → rotated (B, T, C)
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


# ─── Shared building blocks (paper architecture) ────────────────────────────
# Matches S4/S5 paper architecture: BatchNorm, GLU, bidirectional.
# Refs: S4 (arXiv:2111.00396), S5 (arXiv:2208.04933)

class TransposedBN(nn.Module):
    """BatchNorm for (B, T, C) sequences — normalizes across batch+time per feature."""
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)
    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class GLU(nn.Module):
    """Gated Linear Unit: Linear(d→2d), sigmoid gate. Replaces FFN in S4/S5."""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(d_model, 2 * d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x, gate = self.linear(x).chunk(2, dim=-1)
        return self.dropout(x * torch.sigmoid(gate))


def rotation_scan(v, cum_angles, reverse=False):
    """Rotation-based linear recurrence via cumsum trick.
    Forward: h_t = R(θ_t) h_{t-1} + v_t  →  R(-Θ) → cumsum → R(Θ)
    Backward: h_t = R(θ_t) h_{t+1} + v_t  →  R(Θ) → rev_cumsum → R(-Θ)
    """
    if reverse:
        v_rot = apply_rotation(v, cum_angles)
        v_cumsum = v_rot.flip(1).cumsum(1).flip(1)
        return apply_rotation(v_cumsum, -cum_angles)
    else:
        v_rot = apply_rotation(v, -cum_angles)
        return apply_rotation(v_rot.cumsum(1), cum_angles)


class RotationLayerBidir(nn.Module):
    """Bidirectional rotation: forward + backward cumsum scans."""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.values_f = nn.Linear(d_model, d_model)
        self.values_b = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cum_angles_f, cum_angles_b):
        out_f = rotation_scan(self.values_f(x), cum_angles_f, reverse=False)
        out_b = rotation_scan(self.values_b(x), cum_angles_b, reverse=True)
        return self.dropout(self.proj(out_f + out_b))


class RotBlock(nn.Module):
    """Pre-norm rotation block: BN → Rotation → residual → BN → GLU → residual."""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.rot = RotationLayerBidir(d_model, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x, cum_angles_f, cum_angles_b):
        x = x + self.rot(self.bn1(x), cum_angles_f, cum_angles_b)
        x = x + self.glu(self.bn2(x))
        return x


class RotFixed(nn.Module):
    """Fixed learned angles, paper architecture: Linear front-end, bidirectional,
    BatchNorm, GLU. Separate forward/backward angles. Pool reduces T for speed."""
    def __init__(self, d_model=64, n_layers=6, pool=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        base_freq = 1.0 / (10000 ** (torch.arange(0, d_model // 2).float() / (d_model // 2)))
        self.angles_f = nn.Parameter(base_freq.clone())
        self.angles_b = nn.Parameter(base_freq.clone())
        self.blocks = nn.ModuleList([RotBlock(d_model, dropout) for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))  # (B, 16000, d_model)
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)  # (B, T/pool, d_model)
        B, T, C = x.shape
        t_idx = torch.arange(T, device=x.device, dtype=x.dtype)
        cum_f = (t_idx.unsqueeze(1) * self.angles_f).unsqueeze(0).expand(B, -1, -1)
        cum_b = (t_idx.unsqueeze(1) * self.angles_b).unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, cum_f, cum_b)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RotInput(nn.Module):
    """Input-dependent angles, paper architecture: Linear front-end, bidirectional,
    BatchNorm, GLU. Shared angle projector recomputed per layer."""
    def __init__(self, d_model=64, n_layers=6, pool=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.angle_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )
        self.angle_ln = nn.LayerNorm(d_model // 2)
        with torch.no_grad():
            base_freq = 1.0 / (10000 ** (torch.arange(0, d_model // 2).float() / (d_model // 2)))
            self.angle_projector[2].bias.copy_(base_freq)
        self.blocks = nn.ModuleList([RotBlock(d_model, dropout) for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))  # (B, 16000, d_model)
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            raw_angles = self.angle_ln(self.angle_projector(x))
            cum_f = torch.cumsum(raw_angles, dim=1)
            cum_b = torch.cumsum(raw_angles.flip(1), dim=1).flip(1)
            x = block(x, cum_f, cum_b)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Finite-window rotation ────────────────────────────────────────────────
# h_t^W = R(θ)·h_{t-1}^W + v_t - R(Wθ)·v_{t-W}
# Only the last W values contribute. Implemented as shifted cumsum difference.

def rotation_scan_windowed(v, cum_angles, window, reverse=False):
    """Finite-window rotation scan via cumsum difference.
    h_t = R(Θ_t) · [cumsum(u)_t - cumsum(u)_{t-W}]  where u_s = R(-Θ_s)·v_s
    """
    if reverse:
        v_rot = apply_rotation(v, cum_angles)
        cs = v_rot.flip(1).cumsum(1).flip(1)
        # Shift: cumsum_{t+W} in the reversed sense = shift left by W, pad zeros on right
        cs_shifted = F.pad(cs[:, window:], (0, 0, 0, window))
        return apply_rotation(cs - cs_shifted, -cum_angles)
    else:
        v_rot = apply_rotation(v, -cum_angles)
        cs = v_rot.cumsum(1)
        # Shift: cumsum_{t-W} = shift right by W, pad zeros on left
        cs_shifted = F.pad(cs[:, :-window], (0, 0, window, 0))
        return apply_rotation(cs - cs_shifted, cum_angles)


class RotationLayerWindowed(nn.Module):
    """Bidirectional finite-window rotation layer."""
    def __init__(self, d_model, window, dropout=0.0):
        super().__init__()
        self.values_f = nn.Linear(d_model, d_model)
        self.values_b = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.window = window

    def forward(self, x, cum_angles_f, cum_angles_b):
        out_f = rotation_scan_windowed(self.values_f(x), cum_angles_f, self.window, reverse=False)
        out_b = rotation_scan_windowed(self.values_b(x), cum_angles_b, self.window, reverse=True)
        return self.dropout(self.proj(out_f + out_b))


class RotBlockWindowed(nn.Module):
    """Pre-norm windowed rotation block: BN → Rotation → residual → BN → GLU → residual."""
    def __init__(self, d_model, window, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.rot = RotationLayerWindowed(d_model, window, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x, cum_angles_f, cum_angles_b):
        x = x + self.rot(self.bn1(x), cum_angles_f, cum_angles_b)
        x = x + self.glu(self.bn2(x))
        return x


class RotWindow(nn.Module):
    """Finite-window rotation model. Fixed learned angles, paper architecture.
    Each position only accumulates the last W values (like a learned filterbank).
    W is per-layer learnable (softplus-parameterized)."""
    def __init__(self, d_model=64, n_layers=6, pool=4, window=80,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.window = window
        base_freq = 1.0 / (10000 ** (torch.arange(0, d_model // 2).float() / (d_model // 2)))
        self.angles_f = nn.Parameter(base_freq.clone())
        self.angles_b = nn.Parameter(base_freq.clone())
        self.blocks = nn.ModuleList([
            RotBlockWindowed(d_model, window, dropout) for _ in range(n_layers)
        ])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        B, T, C = x.shape
        t_idx = torch.arange(T, device=x.device, dtype=x.dtype)
        cum_f = (t_idx.unsqueeze(1) * self.angles_f).unsqueeze(0).expand(B, -1, -1)
        cum_b = (t_idx.unsqueeze(1) * self.angles_b).unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, cum_f, cum_b)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RotWindowInput(nn.Module):
    """Finite-window rotation with input-dependent angles from shared MLP projector.
    Combines RotWindow's finite window with RotInput's learned angle projector."""
    def __init__(self, d_model=64, n_layers=6, pool=4, window=80,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.window = window
        self.angle_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )
        self.angle_ln = nn.LayerNorm(d_model // 2)
        with torch.no_grad():
            base_freq = 1.0 / (10000 ** (torch.arange(0, d_model // 2).float() / (d_model // 2)))
            self.angle_projector[2].bias.copy_(base_freq)
        self.blocks = nn.ModuleList([
            RotBlockWindowed(d_model, window, dropout) for _ in range(n_layers)
        ])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            raw_angles = self.angle_ln(self.angle_projector(x))
            cum_f = torch.cumsum(raw_angles, dim=1)
            cum_b = torch.cumsum(raw_angles.flip(1), dim=1).flip(1)
            x = block(x, cum_f, cum_b)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Rotation + decay (λ < 1) ──────────────────────────────────────────────
# Gate = λ·e^{iθ} — rotation on the unit circle with exponential decay.
# Equivalent to S5's complex eigenvalues but parameterized directly as (λ, θ).
# Uses parallel scan on complex numbers (no cumsum trick needed).

def complex_rotation_scan(v, gates, reverse=False):
    """Rotation+decay via parallel scan on complex numbers.
    v: (B, T, C) real → reshape to (B, T, C//2) complex → scan → real.
    gates: (B, T, C//2) complex gates (λ·e^{iθ}).
    """
    B, T, C = v.shape
    v_c = torch.view_as_complex(v.reshape(B, T, C // 2, 2).contiguous())
    if reverse:
        h = parallel_scan(gates.flip(1), v_c.flip(1)).flip(1)
    else:
        h = parallel_scan(gates, v_c)
    return torch.view_as_real(h).reshape(B, T, C)


class RotLayerDecayBidir(nn.Module):
    """Bidirectional rotation+decay layer using complex parallel scan."""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.values_f = nn.Linear(d_model, d_model)
        self.values_b = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gates_f, gates_b):
        out_f = complex_rotation_scan(self.values_f(x), gates_f, reverse=False)
        out_b = complex_rotation_scan(self.values_b(x), gates_b, reverse=True)
        return self.dropout(self.proj(out_f + out_b))


class RotBlockDecay(nn.Module):
    """Pre-norm rotation+decay block: BN → RotDecay → residual → BN → GLU → residual."""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.rot = RotLayerDecayBidir(d_model, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x, gates_f, gates_b):
        x = x + self.rot(self.bn1(x), gates_f, gates_b)
        x = x + self.glu(self.bn2(x))
        return x


class RotDecayFixed(nn.Module):
    """Rotation + decay with fixed learned angles and decay rates.
    Gate = λ·e^{iθ}, constant across positions, learned per dimension pair.
    Uses S5 lr/wd settings."""
    def __init__(self, d_model=64, n_layers=6, pool=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        n = d_model // 2
        # S4D-Lin init: Λ = -½ + πi·k, gate = exp(Λ·dt), dt=0.01
        dt = 0.01
        s4d_angles = math.pi * torch.arange(n).float() * dt
        s4d_log_lambda = torch.full((n,), -0.5 * dt)  # log(exp(-0.5*dt)) = -0.005
        self.angles_f = nn.Parameter(s4d_angles.clone())
        self.angles_b = nn.Parameter(s4d_angles.clone())
        self.log_lambda_f = nn.Parameter(s4d_log_lambda.clone())
        self.log_lambda_b = nn.Parameter(s4d_log_lambda.clone())
        self.blocks = nn.ModuleList([RotBlockDecay(d_model, dropout) for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        B, T, C = x.shape
        lam_f = self.log_lambda_f.clamp(max=-1e-4).exp()
        gates_f = (lam_f * torch.exp(1j * self.angles_f)).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        lam_b = self.log_lambda_b.clamp(max=-1e-4).exp()
        gates_b = (lam_b * torch.exp(1j * self.angles_b)).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        for block in self.blocks:
            x = block(x, gates_f, gates_b)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RotDecayInput(nn.Module):
    """Rotation + decay with input-dependent angles AND lambda from one shared projector.
    Gate = λ_t·e^{iθ_t}, projector outputs 2n values: [angles | log_lambda].
    Uses S5 lr/wd settings."""
    def __init__(self, d_model=64, n_layers=6, pool=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        n = d_model // 2
        self.n = n
        # Single projector: first n outputs = angles, second n = log_lambda
        self.gate_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * n),
        )
        self.angle_ln = nn.LayerNorm(n)
        with torch.no_grad():
            # S4D-Lin init: Λ = -½ + πi·k, gate = exp(Λ·dt), dt=0.01
            dt = 0.01
            bias = self.gate_projector[2].bias
            bias[:n].copy_(math.pi * torch.arange(n).float() * dt)  # angles
            bias[n:].fill_(-0.5 * dt)                                # log_λ ≈ -0.005
        self.blocks = nn.ModuleList([RotBlockDecay(d_model, dropout) for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            out = self.gate_projector(x)
            raw_angles = self.angle_ln(out[..., :self.n])
            log_lam = out[..., self.n:].clamp(max=-1e-4)
            lam = log_lam.exp()  # (B, T, n), in (0, ~1)
            gates_f = lam * torch.exp(1j * raw_angles)
            gates_b = lam * torch.exp(1j * raw_angles)
            x = block(x, gates_f, gates_b)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Rotation + decay + complex B/C/D (S5-style projections) ────────────────
# Same gates as RotDecay but with complex B (input→state), C (state→output),
# and D feedthrough skip — matching S5's full parameterization.

class RotS5LayerBidir(nn.Module):
    """Bidirectional rotation+decay with complex B/C projections and D skip."""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        n = d_model // 2
        H = d_model
        # Forward: complex B (n, H), C (H, n)
        self.B_re_f = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_f = nn.Parameter(torch.zeros(n, H))
        self.C_re_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # Backward: complex B (n, H), C (H, n)
        self.B_re_b = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_b = nn.Parameter(torch.zeros(n, H))
        self.C_re_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # D feedthrough
        self.D = nn.Parameter(torch.ones(H))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gates_f, gates_b, Lambda_f=None, Lambda_b=None):
        B_f = torch.complex(self.B_re_f, self.B_im_f)
        C_f = torch.complex(self.C_re_f, self.C_im_f)
        # ZOH discretization: B_bar = ((gate - 1) / Λ) * B
        if Lambda_f is not None:
            B_bar_f = ((gates_f[0, 0] - 1.0) / Lambda_f).unsqueeze(-1) * B_f
        else:
            B_bar_f = B_f
        Bu_f = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_f)
        h_f = parallel_scan(gates_f, Bu_f)
        y_f = 2.0 * torch.einsum('btn,hn->bth', h_f, C_f.conj()).real

        B_b = torch.complex(self.B_re_b, self.B_im_b)
        C_b = torch.complex(self.C_re_b, self.C_im_b)
        if Lambda_b is not None:
            B_bar_b = ((gates_b[0, 0] - 1.0) / Lambda_b).unsqueeze(-1) * B_b
        else:
            B_bar_b = B_b
        Bu_b = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_b)
        h_b = parallel_scan(gates_b, Bu_b.flip(1)).flip(1)
        y_b = 2.0 * torch.einsum('btn,hn->bth', h_b, C_b.conj()).real

        return self.dropout(y_f + y_b + x * self.D)


class RotS5Block(nn.Module):
    """Pre-norm block with per-layer continuous eigenvalues + shared dt, like S5.
    gate_k = exp((Λ_re_k + i·Λ_im_k) · dt), one dt per direction."""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        n = d_model // 2
        # Continuous-time eigenvalues: Λ = Λ_re + i·Λ_im
        # S4D-Lin init: Λ_re = -0.5, Λ_im = π·k
        self.Lambda_re_f = nn.Parameter(-0.5 * torch.ones(n))
        self.Lambda_im_f = nn.Parameter(math.pi * torch.arange(n).float())
        self.Lambda_re_b = nn.Parameter(-0.5 * torch.ones(n))
        self.Lambda_im_b = nn.Parameter(math.pi * torch.arange(n).float())
        # Shared dt (one scalar per direction)
        self.log_dt_f = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt_b = nn.Parameter(torch.tensor(math.log(0.01)))
        self.bn1 = TransposedBN(d_model)
        self.ssm = RotS5LayerBidir(d_model, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        B, T, _ = x.shape
        dt_f = self.log_dt_f.exp()
        Lambda_f = torch.complex(self.Lambda_re_f.clamp(max=-1e-4), self.Lambda_im_f)
        gates_f = torch.exp(Lambda_f * dt_f).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        dt_b = self.log_dt_b.exp()
        Lambda_b = torch.complex(self.Lambda_re_b.clamp(max=-1e-4), self.Lambda_im_b)
        gates_b = torch.exp(Lambda_b * dt_b).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        x = x + self.ssm(self.bn1(x), gates_f, gates_b, Lambda_f, Lambda_b)
        x = x + self.glu(self.bn2(x))
        return x


class RotS5Fixed(nn.Module):
    """Per-layer fixed angles + decay with complex B/C/D. S4D-Lin init."""
    def __init__(self, d_model=64, n_layers=6, pool=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([RotS5Block(d_model, dropout) for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RotS5BlockInput(nn.Module):
    """Pre-norm block with per-layer gate projector + RotS5 layer."""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        n = d_model // 2
        self.n = n
        self.gate_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * n),
        )
        self.angle_ln = nn.LayerNorm(n)
        with torch.no_grad():
            dt = 0.01
            bias = self.gate_projector[2].bias
            bias[:n].copy_(math.pi * torch.arange(n).float() * dt)
            bias[n:].fill_(-0.5 * dt)
        self.bn1 = TransposedBN(d_model)
        self.ssm = RotS5LayerBidir(d_model, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        out = self.gate_projector(x)
        raw_angles = self.angle_ln(out[..., :self.n])
        log_lam = out[..., self.n:].clamp(max=-1e-4)
        lam = log_lam.exp()
        gates_f = lam * torch.exp(1j * raw_angles)
        gates_b = lam * torch.exp(1j * raw_angles)
        x = x + self.ssm(self.bn1(x), gates_f, gates_b)
        x = x + self.glu(self.bn2(x))
        return x


class RotS5Input(nn.Module):
    """Per-layer input-dependent angles + decay with complex B/C/D. S4D-Lin init."""
    def __init__(self, d_model=64, n_layers=6, pool=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([RotS5BlockInput(d_model, dropout) for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Cumsum-based S5 variants (fast, no parallel scan) ──────────────────────
# Pure rotation (Λ_re=0, |gate|=1) with complex B/C/D and B_bar.
# Uses torch.cumsum — single CUDA kernel, ~10x faster than parallel scan.

def complex_cumsum_scan(Bu, angles, reverse=False, window=None, phases=None):
    """Pure rotation scan via complex cumsum. Bu: (B, T, n) complex.
    angles: (n,) fixed per-dim angles, OR phases: (B, T, n) complex precomputed.
    Optional finite window via cumsum difference."""
    if phases is None:
        B_batch, T, n = Bu.shape
        t_idx = torch.arange(T, device=Bu.device, dtype=angles.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * angles)  # (T, n)
    if reverse:
        rotated = Bu * phases
        cs = rotated.flip(1).cumsum(1).flip(1)
        if window is not None:
            cs_shifted = F.pad(cs[:, window:], (0, 0, 0, window))
            cs = cs - cs_shifted
        return cs * phases.conj()
    else:
        rotated = Bu * phases.conj()
        cs = rotated.cumsum(1)
        if window is not None:
            cs_shifted = F.pad(cs[:, :-window], (0, 0, window, 0))
            cs = cs - cs_shifted
        return cs * phases


def block_decay_cumsum_scan(Bu, angles, lam, window, reverse=False, phases=None):
    """Cumsum scan with block-wise decay. Combines cumsum speed with decay.
    Bu: (B, T, n) complex. angles: (n,) fixed angles OR None if phases provided.
    lam: (n,) decay per block. window: int block size W.
    phases: (B, T, n) complex precomputed phases (for input-dependent angles).
    h_t = Σ_k λ^(k+1) · [cs(t-kW) - cs(t-(k+1)W)]  for k=0,1,...,K-1
    where cs is the rotated cumsum."""
    B_batch, T, n = Bu.shape
    K = (T + window - 1) // window  # number of blocks
    if phases is None:
        t_idx = torch.arange(T, device=Bu.device, dtype=angles.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * angles)  # (T, n)

    if reverse:
        rotated = Bu * phases
        cs = rotated.flip(1).cumsum(1).flip(1)  # (B, T, n)
        # Build block decay sum: start from farthest block, accumulate with λ
        # block_0 gets weight 1, block_1 gets λ, block_2 gets λ², ...
        h = torch.zeros_like(cs)
        for k in range(K - 1, -1, -1):
            shift = k * window
            if shift == 0:
                block_k = cs
            else:
                block_k = F.pad(cs[:, shift:], (0, 0, 0, shift))
            if shift + window <= T:
                block_k_next = F.pad(cs[:, shift + window:], (0, 0, 0, shift + window))
            else:
                block_k_next = torch.zeros_like(cs)
            h = block_k - block_k_next + lam * h
        return h * phases.conj()
    else:
        rotated = Bu * phases.conj()
        cs = rotated.cumsum(1)  # (B, T, n)
        # Build block decay sum: start from farthest block, accumulate with λ
        # block_0 gets weight 1, block_1 gets λ, block_2 gets λ², ...
        h = torch.zeros_like(cs)
        for k in range(K - 1, -1, -1):
            shift = k * window
            if shift == 0:
                block_k = cs
            else:
                block_k = F.pad(cs[:, :-shift], (0, 0, shift, 0))
            if shift + window <= T:
                block_k_next = F.pad(cs[:, :-(shift + window)], (0, 0, shift + window, 0))
            else:
                block_k_next = torch.zeros_like(cs)
            h = block_k - block_k_next + lam * h
        return h * phases


def block_decay_cumsum_scan_v2(Bu, angles, lam, window, reverse=False, phases=None):
    """Fast block-decay cumsum scan. Reshapes into K blocks of W, scans over blocks.
    Each loop iteration operates on (B, W, n) instead of (B, T, n).
    Bu: (B, T, n) complex. angles: (n,) or None. lam: (n,). window: int W.
    phases: (B, T, n) or None."""
    B_batch, T, n = Bu.shape
    K = T // window  # T must be divisible by W
    assert T == K * window, f"T={T} not divisible by window={window}"
    if phases is None:
        t_idx = torch.arange(T, device=Bu.device, dtype=angles.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * angles)  # (T, n)

    if reverse:
        rotated = Bu * phases
        cs = rotated.flip(1).cumsum(1).flip(1)  # (B, T, n)
        # d[t] = cs[t] - cs[t+W] (reverse window diff)
        d = cs - F.pad(cs[:, window:], (0, 0, 0, window))
        # Reshape into blocks: (B, K, W, n)
        d_blocks = d.view(B_batch, K, window, n)
        # Scan over blocks from last to first (reverse)
        h_list = [None] * K
        h_list[K - 1] = d_blocks[:, K - 1]
        for k in range(K - 2, -1, -1):
            h_list[k] = d_blocks[:, k] + lam * h_list[k + 1]
        return torch.stack(h_list, dim=1).view(B_batch, T, n) * phases.conj()
    else:
        rotated = Bu * phases.conj()
        cs = rotated.cumsum(1)  # (B, T, n)
        # d[t] = cs[t] - cs[t-W] (forward window diff)
        d = cs - F.pad(cs[:, :-window], (0, 0, window, 0))
        # Reshape into blocks: (B, K, W, n)
        d_blocks = d.view(B_batch, K, window, n)
        # Scan over blocks from first to last (forward)
        h_list = [None] * K
        h_list[0] = d_blocks[:, 0]
        for k in range(1, K):
            h_list[k] = d_blocks[:, k] + lam * h_list[k - 1]
        return torch.stack(h_list, dim=1).view(B_batch, T, n) * phases


def block_decay_cumsum_scan_overlap(Bu, angles, lam, window, reverse=False, phases=None):
    """Block-decay cumsum scan with 50% overlapping windows.
    Blocks of size W with stride W/2. Smoother decay transition.
    Bu: (B, T, n) complex. angles: (n,) or None. lam: (n,). window: int W.
    phases: (B, T, n) or None."""
    B_batch, T, n = Bu.shape
    stride = window // 2
    K = (T - window) // stride + 1  # number of overlapping blocks
    if phases is None:
        t_idx = torch.arange(T, device=Bu.device, dtype=angles.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * angles)

    if reverse:
        rotated = Bu * phases
        cs = rotated.flip(1).cumsum(1).flip(1)  # (B, T, n)
        # Compute block diffs: block_k covers [t + k*stride, t + k*stride + W)
        # Use cs.unfold to get sliding windows
        # cs_windows[b, k, w, n] = cs[b, k*stride + w, n]
        cs_windows = cs.unfold(1, window, stride)  # (B, K, n, W)
        cs_windows = cs_windows.permute(0, 1, 3, 2)  # (B, K, W, n)
        # Block diff: for reverse, block_k = cs at start of window - cs at end
        # block_k[w] = cs[k*stride + w] - cs[k*stride + W] (if exists, else 0)
        # Simpler: compute d = cs - shifted_cs for each block
        # Actually, each block is just the windowed diff at offset k*stride
        # d_k[w] = cs[k*stride + w] - cs[k*stride + w + W] for reverse
        # But we can just compute the full d once and slice
        d_full = cs - F.pad(cs[:, window:], (0, 0, 0, window))
        d_windows = d_full.unfold(1, window, stride)  # (B, K, n, W)
        d_windows = d_windows.permute(0, 1, 3, 2)  # (B, K, W, n)
        # Scan over blocks from last to first
        h_list = [None] * K
        h_list[K - 1] = d_windows[:, K - 1]
        for k in range(K - 2, -1, -1):
            h_list[k] = d_windows[:, k] + lam * h_list[k + 1]
        # Overlap-add: each position gets contribution from up to 2 blocks
        h = torch.zeros(B_batch, T, n, device=Bu.device, dtype=Bu.dtype)
        counts = torch.zeros(T, device=Bu.device)
        for k in range(K):
            start = k * stride
            h[:, start:start + window] += h_list[k]
            counts[start:start + window] += 1
        h = h / counts.unsqueeze(0).unsqueeze(-1).clamp(min=1)
        return h * phases.conj()
    else:
        rotated = Bu * phases.conj()
        cs = rotated.cumsum(1)  # (B, T, n)
        d_full = cs - F.pad(cs[:, :-window], (0, 0, window, 0))
        d_windows = d_full.unfold(1, window, stride)  # (B, K, n, W)
        d_windows = d_windows.permute(0, 1, 3, 2)  # (B, K, W, n)
        # Scan over blocks from first to last
        h_list = [None] * K
        h_list[0] = d_windows[:, 0]
        for k in range(1, K):
            h_list[k] = d_windows[:, k] + lam * h_list[k - 1]
        # Overlap-add: average overlapping contributions
        h = torch.zeros(B_batch, T, n, device=Bu.device, dtype=Bu.dtype)
        counts = torch.zeros(T, device=Bu.device)
        for k in range(K):
            start = k * stride
            h[:, start:start + window] += h_list[k]
            counts[start:start + window] += 1
        h = h / counts.unsqueeze(0).unsqueeze(-1).clamp(min=1)
        return h * phases


class BlockDecayS5OverlapLayerBidir(nn.Module):
    """Bidirectional S5 layer with overlapping block-wise decay.
    50% overlap for smoother decay transition."""
    def __init__(self, d_model, window=80, dropout=0.0):
        super().__init__()
        n = d_model // 2
        H = d_model
        self.window = window
        self.Lambda_im_f = nn.Parameter(math.pi * torch.arange(n).float())
        self.Lambda_im_b = nn.Parameter(math.pi * torch.arange(n).float())
        self.log_dt_f = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt_b = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_lambda_f = nn.Parameter(-0.1 * torch.ones(n))
        self.log_lambda_b = nn.Parameter(-0.1 * torch.ones(n))
        self.B_re_f = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_f = nn.Parameter(torch.zeros(n, H))
        self.C_re_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.B_re_b = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_b = nn.Parameter(torch.zeros(n, H))
        self.C_re_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.D = nn.Parameter(torch.ones(H))
        self.dropout = nn.Dropout(dropout)

    def _b_bar(self, Lambda_im, log_dt, B_re, B_im):
        dt = log_dt.exp()
        Lambda = 1j * Lambda_im
        B = torch.complex(B_re, B_im)
        Ldt = Lambda * dt
        safe_L = torch.where(Lambda.abs() < 1e-6, torch.ones_like(Lambda), Lambda)
        scale = torch.where(Lambda.abs() < 1e-6,
                            dt * torch.ones_like(Lambda),
                            (torch.exp(Ldt) - 1.0) / safe_L)
        return scale.unsqueeze(-1) * B

    def forward(self, x):
        B_batch, T, H = x.shape
        dt_f = self.log_dt_f.exp()
        angles_f = self.Lambda_im_f * dt_f
        lam_f = self.log_lambda_f.clamp(max=-1e-4).exp()
        B_bar_f = self._b_bar(self.Lambda_im_f, self.log_dt_f, self.B_re_f, self.B_im_f)
        C_f = torch.complex(self.C_re_f, self.C_im_f)
        Bu_f = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_f)
        h_f = block_decay_cumsum_scan_overlap(Bu_f, angles_f, lam_f, self.window, reverse=False)
        y_f = 2.0 * torch.einsum('btn,hn->bth', h_f, C_f.conj()).real
        dt_b = self.log_dt_b.exp()
        angles_b = self.Lambda_im_b * dt_b
        lam_b = self.log_lambda_b.clamp(max=-1e-4).exp()
        B_bar_b = self._b_bar(self.Lambda_im_b, self.log_dt_b, self.B_re_b, self.B_im_b)
        C_b = torch.complex(self.C_re_b, self.C_im_b)
        Bu_b = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_b)
        h_b = block_decay_cumsum_scan_overlap(Bu_b, angles_b, lam_b, self.window, reverse=True)
        y_b = 2.0 * torch.einsum('btn,hn->bth', h_b, C_b.conj()).real
        return self.dropout(y_f + y_b + x * self.D)


class BlockDecayS5OverlapBlock(nn.Module):
    def __init__(self, d_model, window=80, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.ssm = BlockDecayS5OverlapLayerBidir(d_model, window, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        x = x + self.ssm(self.bn1(x))
        x = x + self.glu(self.bn2(x))
        return x


class BlockDecayS5Overlap(nn.Module):
    """S5 with overlapping block-wise decay — smoother forgetting. B/C/D, B_bar."""
    def __init__(self, d_model=64, n_layers=6, pool=4, window=80,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([BlockDecayS5OverlapBlock(d_model, window, dropout)
                                     for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BlockDecayS5LayerBidir(nn.Module):
    """Bidirectional S5 layer with block-wise decay. Cumsum speed + decay.
    Per-layer: angles, dt, lambda, B, C, D. W=800, K=T/W blocks."""
    def __init__(self, d_model, window=800, dropout=0.0):
        super().__init__()
        n = d_model // 2
        H = d_model
        self.window = window
        # Per-layer angles: S4D-Lin init
        self.Lambda_im_f = nn.Parameter(math.pi * torch.arange(n).float())
        self.Lambda_im_b = nn.Parameter(math.pi * torch.arange(n).float())
        self.log_dt_f = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt_b = nn.Parameter(torch.tensor(math.log(0.01)))
        # Per-dimension block decay λ ∈ (0, 1)
        self.log_lambda_f = nn.Parameter(-0.1 * torch.ones(n))
        self.log_lambda_b = nn.Parameter(-0.1 * torch.ones(n))
        # Complex B, C — forward
        self.B_re_f = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_f = nn.Parameter(torch.zeros(n, H))
        self.C_re_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # Complex B, C — backward
        self.B_re_b = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_b = nn.Parameter(torch.zeros(n, H))
        self.C_re_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # D feedthrough
        self.D = nn.Parameter(torch.ones(H))
        self.dropout = nn.Dropout(dropout)

    def _b_bar(self, Lambda_im, log_dt, B_re, B_im):
        dt = log_dt.exp()
        Lambda = 1j * Lambda_im
        B = torch.complex(B_re, B_im)
        Ldt = Lambda * dt
        safe_L = torch.where(Lambda.abs() < 1e-6, torch.ones_like(Lambda), Lambda)
        scale = torch.where(Lambda.abs() < 1e-6,
                            dt * torch.ones_like(Lambda),
                            (torch.exp(Ldt) - 1.0) / safe_L)
        return scale.unsqueeze(-1) * B

    def forward(self, x):
        B_batch, T, H = x.shape
        # Forward
        dt_f = self.log_dt_f.exp()
        angles_f = self.Lambda_im_f * dt_f
        lam_f = self.log_lambda_f.clamp(max=-1e-4).exp()  # (n,) ∈ (0, 1)
        B_bar_f = self._b_bar(self.Lambda_im_f, self.log_dt_f, self.B_re_f, self.B_im_f)
        C_f = torch.complex(self.C_re_f, self.C_im_f)
        Bu_f = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_f)
        h_f = block_decay_cumsum_scan(Bu_f, angles_f, lam_f, self.window, reverse=False)
        y_f = 2.0 * torch.einsum('btn,hn->bth', h_f, C_f.conj()).real
        # Backward
        dt_b = self.log_dt_b.exp()
        angles_b = self.Lambda_im_b * dt_b
        lam_b = self.log_lambda_b.clamp(max=-1e-4).exp()
        B_bar_b = self._b_bar(self.Lambda_im_b, self.log_dt_b, self.B_re_b, self.B_im_b)
        C_b = torch.complex(self.C_re_b, self.C_im_b)
        Bu_b = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_b)
        h_b = block_decay_cumsum_scan(Bu_b, angles_b, lam_b, self.window, reverse=True)
        y_b = 2.0 * torch.einsum('btn,hn->bth', h_b, C_b.conj()).real

        return self.dropout(y_f + y_b + x * self.D)


class BlockDecayS5Block(nn.Module):
    """Pre-norm block with block-decay S5 layer."""
    def __init__(self, d_model, window=800, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.ssm = BlockDecayS5LayerBidir(d_model, window, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        x = x + self.ssm(self.bn1(x))
        x = x + self.glu(self.bn2(x))
        return x


class BlockDecayS5(nn.Module):
    """S5 with block-wise decay — cumsum speed + learnable decay. B/C/D, B_bar."""
    def __init__(self, d_model=64, n_layers=6, pool=4, window=800,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([BlockDecayS5Block(d_model, window, dropout)
                                     for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BlockDecayS5V2LayerBidir(nn.Module):
    """Bidirectional S5 layer with block-wise decay — fast v2 (reshape + block scan).
    Same math as BlockDecayS5LayerBidir but operates on (B,W,n) tensors."""
    def __init__(self, d_model, window=800, dropout=0.0):
        super().__init__()
        n = d_model // 2
        H = d_model
        self.window = window
        self.Lambda_im_f = nn.Parameter(math.pi * torch.arange(n).float())
        self.Lambda_im_b = nn.Parameter(math.pi * torch.arange(n).float())
        self.log_dt_f = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt_b = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_lambda_f = nn.Parameter(-0.1 * torch.ones(n))
        self.log_lambda_b = nn.Parameter(-0.1 * torch.ones(n))
        self.B_re_f = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_f = nn.Parameter(torch.zeros(n, H))
        self.C_re_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.B_re_b = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_b = nn.Parameter(torch.zeros(n, H))
        self.C_re_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.D = nn.Parameter(torch.ones(H))
        self.dropout = nn.Dropout(dropout)

    def _b_bar(self, Lambda_im, log_dt, B_re, B_im):
        dt = log_dt.exp()
        Lambda = 1j * Lambda_im
        B = torch.complex(B_re, B_im)
        Ldt = Lambda * dt
        safe_L = torch.where(Lambda.abs() < 1e-6, torch.ones_like(Lambda), Lambda)
        scale = torch.where(Lambda.abs() < 1e-6,
                            dt * torch.ones_like(Lambda),
                            (torch.exp(Ldt) - 1.0) / safe_L)
        return scale.unsqueeze(-1) * B

    def forward(self, x):
        B_batch, T, H = x.shape
        dt_f = self.log_dt_f.exp()
        angles_f = self.Lambda_im_f * dt_f
        lam_f = self.log_lambda_f.clamp(max=-1e-4).exp()
        B_bar_f = self._b_bar(self.Lambda_im_f, self.log_dt_f, self.B_re_f, self.B_im_f)
        C_f = torch.complex(self.C_re_f, self.C_im_f)
        Bu_f = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_f)
        h_f = block_decay_cumsum_scan_v2(Bu_f, angles_f, lam_f, self.window, reverse=False)
        y_f = 2.0 * torch.einsum('btn,hn->bth', h_f, C_f.conj()).real
        dt_b = self.log_dt_b.exp()
        angles_b = self.Lambda_im_b * dt_b
        lam_b = self.log_lambda_b.clamp(max=-1e-4).exp()
        B_bar_b = self._b_bar(self.Lambda_im_b, self.log_dt_b, self.B_re_b, self.B_im_b)
        C_b = torch.complex(self.C_re_b, self.C_im_b)
        Bu_b = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_b)
        h_b = block_decay_cumsum_scan_v2(Bu_b, angles_b, lam_b, self.window, reverse=True)
        y_b = 2.0 * torch.einsum('btn,hn->bth', h_b, C_b.conj()).real
        return self.dropout(y_f + y_b + x * self.D)


class BlockDecayS5V2Block(nn.Module):
    def __init__(self, d_model, window=800, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.ssm = BlockDecayS5V2LayerBidir(d_model, window, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        x = x + self.ssm(self.bn1(x))
        x = x + self.glu(self.bn2(x))
        return x


class BlockDecayS5V2(nn.Module):
    """S5 with block-wise decay — fast v2 scan. B/C/D, B_bar."""
    def __init__(self, d_model=64, n_layers=6, pool=4, window=800,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([BlockDecayS5V2Block(d_model, window, dropout)
                                     for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BlockDecayS5LayerBidirMod(nn.Module):
    """Bidirectional S5 layer with block-wise decay + modulated angles.
    angles = base_angles * (1 + proj(x)). No LayerNorm."""
    def __init__(self, d_model, window=800, dropout=0.0):
        super().__init__()
        n = d_model // 2
        H = d_model
        self.window = window
        # Base eigenvalues: S4D-Lin init
        self.Lambda_im_f = nn.Parameter(math.pi * torch.arange(n).float())
        self.Lambda_im_b = nn.Parameter(math.pi * torch.arange(n).float())
        self.log_dt_f = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt_b = nn.Parameter(torch.tensor(math.log(0.01)))
        # Per-dimension block decay λ ∈ (0, 1)
        self.log_lambda_f = nn.Parameter(-0.1 * torch.ones(n))
        self.log_lambda_b = nn.Parameter(-0.1 * torch.ones(n))
        # Modulation projector
        self.mod_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n),
        )
        with torch.no_grad():
            self.mod_projector[2].bias.zero_()
            self.mod_projector[2].weight.zero_()
        # Complex B, C — forward
        self.B_re_f = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_f = nn.Parameter(torch.zeros(n, H))
        self.C_re_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # Complex B, C — backward
        self.B_re_b = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_b = nn.Parameter(torch.zeros(n, H))
        self.C_re_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # D feedthrough
        self.D = nn.Parameter(torch.ones(H))
        self.dropout = nn.Dropout(dropout)

    def _b_bar(self, Lambda_im, log_dt, B_re, B_im):
        dt = log_dt.exp()
        Lambda = 1j * Lambda_im
        B = torch.complex(B_re, B_im)
        Ldt = Lambda * dt
        safe_L = torch.where(Lambda.abs() < 1e-6, torch.ones_like(Lambda), Lambda)
        scale = torch.where(Lambda.abs() < 1e-6,
                            dt * torch.ones_like(Lambda),
                            (torch.exp(Ldt) - 1.0) / safe_L)
        return scale.unsqueeze(-1) * B

    def forward(self, x):
        B_batch, T, H = x.shape
        dt_f = self.log_dt_f.exp()
        dt_b = self.log_dt_b.exp()
        # Base angles
        base_angles_f = self.Lambda_im_f * dt_f
        base_angles_b = self.Lambda_im_b * dt_b
        # Input-dependent modulation
        mod = self.mod_projector(x)  # (B, T, n)
        angles_f = base_angles_f * (1.0 + mod)
        angles_b = base_angles_b * (1.0 + mod)
        # Cumulative phases
        cum_f = torch.cumsum(angles_f, dim=1)
        phases_f = torch.exp(1j * cum_f.to(torch.float32))
        cum_b = torch.cumsum(angles_b.flip(1), dim=1).flip(1)
        phases_b = torch.exp(1j * cum_b.to(torch.float32))
        # Forward
        lam_f = self.log_lambda_f.clamp(max=-1e-4).exp()
        B_bar_f = self._b_bar(self.Lambda_im_f, self.log_dt_f, self.B_re_f, self.B_im_f)
        C_f = torch.complex(self.C_re_f, self.C_im_f)
        Bu_f = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_f)
        h_f = block_decay_cumsum_scan(Bu_f, None, lam_f, self.window, reverse=False, phases=phases_f)
        y_f = 2.0 * torch.einsum('btn,hn->bth', h_f, C_f.conj()).real
        # Backward
        lam_b = self.log_lambda_b.clamp(max=-1e-4).exp()
        B_bar_b = self._b_bar(self.Lambda_im_b, self.log_dt_b, self.B_re_b, self.B_im_b)
        C_b = torch.complex(self.C_re_b, self.C_im_b)
        Bu_b = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_b)
        h_b = block_decay_cumsum_scan(Bu_b, None, lam_b, self.window, reverse=True, phases=phases_b)
        y_b = 2.0 * torch.einsum('btn,hn->bth', h_b, C_b.conj()).real

        return self.dropout(y_f + y_b + x * self.D)


class BlockDecayS5BlockMod(nn.Module):
    """Pre-norm block with modulated block-decay S5 layer."""
    def __init__(self, d_model, window=800, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.ssm = BlockDecayS5LayerBidirMod(d_model, window, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        x = x + self.ssm(self.bn1(x))
        x = x + self.glu(self.bn2(x))
        return x


class BlockDecayS5Mod(nn.Module):
    """S5 with block-wise decay + modulated angles — cumsum speed. B/C/D, B_bar."""
    def __init__(self, d_model=64, n_layers=6, pool=4, window=800,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([BlockDecayS5BlockMod(d_model, window, dropout)
                                     for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumS5LayerBidir(nn.Module):
    """Bidirectional pure-rotation S5 layer. Cumsum-based (fast).
    Per-layer angles, complex B/C, B_bar, D skip. Λ_re=0 (no decay)."""
    def __init__(self, d_model, window=None, dropout=0.0):
        super().__init__()
        n = d_model // 2
        H = d_model
        self.window = window
        # Per-layer angles: Λ = i·Λ_im, S4D-Lin init
        self.Lambda_im_f = nn.Parameter(math.pi * torch.arange(n).float())
        self.Lambda_im_b = nn.Parameter(math.pi * torch.arange(n).float())
        self.log_dt_f = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt_b = nn.Parameter(torch.tensor(math.log(0.01)))
        # Complex B (n, H), C (H, n)
        self.B_re_f = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_f = nn.Parameter(torch.zeros(n, H))
        self.C_re_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.B_re_b = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_b = nn.Parameter(torch.zeros(n, H))
        self.C_re_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # D feedthrough
        self.D = nn.Parameter(torch.ones(H))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B_batch, T, H = x.shape
        # Forward
        dt_f = self.log_dt_f.exp()
        Lambda_f = 1j * self.Lambda_im_f  # pure imaginary, no decay
        angles_f = self.Lambda_im_f * dt_f  # discrete angles
        B_f = torch.complex(self.B_re_f, self.B_im_f)
        C_f = torch.complex(self.C_re_f, self.C_im_f)
        # B_bar = (exp(Λdt)-1)/Λ · B; use dt for Λ≈0 (L'Hopital)
        Ldt_f = Lambda_f * dt_f
        safe_L_f = torch.where(Lambda_f.abs() < 1e-6, torch.ones_like(Lambda_f), Lambda_f)
        B_bar_scale_f = torch.where(Lambda_f.abs() < 1e-6,
                                     dt_f * torch.ones_like(Lambda_f),
                                     (torch.exp(Ldt_f) - 1.0) / safe_L_f)
        B_bar_f = B_bar_scale_f.unsqueeze(-1) * B_f
        Bu_f = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_f)
        h_f = complex_cumsum_scan(Bu_f, angles_f, reverse=False, window=self.window)
        y_f = 2.0 * torch.einsum('btn,hn->bth', h_f, C_f.conj()).real
        # Backward
        dt_b = self.log_dt_b.exp()
        Lambda_b = 1j * self.Lambda_im_b
        angles_b = self.Lambda_im_b * dt_b
        B_b = torch.complex(self.B_re_b, self.B_im_b)
        C_b = torch.complex(self.C_re_b, self.C_im_b)
        Ldt_b = Lambda_b * dt_b
        safe_L_b = torch.where(Lambda_b.abs() < 1e-6, torch.ones_like(Lambda_b), Lambda_b)
        B_bar_scale_b = torch.where(Lambda_b.abs() < 1e-6,
                                     dt_b * torch.ones_like(Lambda_b),
                                     (torch.exp(Ldt_b) - 1.0) / safe_L_b)
        B_bar_b = B_bar_scale_b.unsqueeze(-1) * B_b
        Bu_b = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_b)
        h_b = complex_cumsum_scan(Bu_b, angles_b, reverse=True, window=self.window)
        y_b = 2.0 * torch.einsum('btn,hn->bth', h_b, C_b.conj()).real

        return self.dropout(y_f + y_b + x * self.D)


class MinimalStridedWindow(nn.Module):
    """LearnedSpecCNN stripped to minimal: no CNN, no log.
    Scalar input → windowed cumsum → power → stride → BN + GLU + residual → avg pool → classify.
    Adapted directly from LearnedSpecCNN for speed parity."""
    def __init__(self, n_freqs=64, window=400, hop_length=160,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.window = window
        self.hop = hop_length
        # Learned frequencies — mel-scale init
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))
        # BN + GLU on strided output (T=100, cheap)
        # 2*n_freqs because we concat real + imag
        self.bn = TransposedBN(2 * n_freqs)
        self.glu = GLU(2 * n_freqs, dropout)
        self.fc = nn.Linear(2 * n_freqs, num_classes)

    def forward(self, x):
        # x: (B, 16000) — same front-end as LearnedSpecCNN
        B_batch, T = x.shape
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)  # (T, n_freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)  # (B, T, 1)
        rotated = x_complex * phases.conj().unsqueeze(0)  # (B, T, n_freqs)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted
        # Unrotate to stabilize phase, then real + imag output → stride
        d = d * phases.unsqueeze(0)  # (B, T, n_freqs)
        h = torch.cat([d.real, d.imag], dim=-1)  # (B, T, 2*n_freqs)
        h = h[:, ::self.hop]  # (B, ~100, 2*n_freqs)
        # BN + GLU + residual (on T=100, cheap)
        h = h + self.glu(self.bn(h))
        # Avg pool over time → classify
        return self.fc(h.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiLayerMinimalStrided(nn.Module):
    """Multi-layer windowed cumsum with configurable downsampling.
    Layer 1: scalar input → windowed cumsum (W=window) → unrotate → real+imag → stride by ds_factor
    Layers 2+: real+imag input → project to complex → windowed cumsum (W=window/ds_factor) → unrotate → real+imag
    Each layer has BN + GLU + residual."""
    def __init__(self, n_freqs=64, window=400, ds_factor=10, n_layers=3,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        assert window % ds_factor == 0, f"window ({window}) must be divisible by ds_factor ({ds_factor})"
        self.n_freqs = n_freqs
        self.window = window
        self.ds_factor = ds_factor
        self.n_layers = n_layers
        self.inner_window = window // ds_factor  # e.g. 400/10 = 40

        # Layer 1: learned frequencies (mel-scale init), scalar input
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))
        dim = 2 * n_freqs
        self.bn1 = TransposedBN(dim)
        self.glu1 = GLU(dim, dropout)

        # Layers 2+: each projects real input to complex, runs windowed cumsum
        self.proj_layers = nn.ModuleList()
        self.freq_params = nn.ParameterList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            # Project 2*n_freqs real → n_freqs complex (real + imag parts)
            self.proj_layers.append(nn.Linear(dim, dim))
            # Learned frequencies for this layer — uniform init over [0, pi]
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            self.bn_layers.append(TransposedBN(dim))
            self.glu_layers.append(GLU(dim, dropout))

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        B_batch, T = x.shape

        # === Layer 1: scalar → complex windowed cumsum → stride ===
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)  # (T, n_freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)  # (B, T, 1)
        rotated = x_complex * phases.conj().unsqueeze(0)  # (B, T, n_freqs)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)  # unrotate
        h = torch.cat([d.real, d.imag], dim=-1)  # (B, T, 2*n_freqs)
        h = h[:, ::self.ds_factor]  # (B, T/ds_factor, 2*n_freqs)
        h = h + self.glu1(self.bn1(h))

        # === Layers 2+: real → project → complex windowed cumsum ===
        T2 = h.shape[1]
        for i in range(self.n_layers - 1):
            proj = self.proj_layers[i](h)  # (B, T2, 2*n_freqs)
            # Split into real and imag to form complex
            z_re, z_im = proj.chunk(2, dim=-1)  # each (B, T2, n_freqs)
            z = torch.complex(z_re, z_im)  # (B, T2, n_freqs)

            layer_freqs = self.freq_params[i]  # (n_freqs,)
            t2_idx = torch.arange(T2, device=x.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t2_idx.unsqueeze(1) * layer_freqs)  # (T2, n_freqs)

            rotated2 = z * layer_phases.conj().unsqueeze(0)  # (B, T2, n_freqs)
            cs2 = rotated2.cumsum(dim=1)
            W = self.inner_window
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases.unsqueeze(0)  # unrotate
            out = torch.cat([d2.real, d2.imag], dim=-1)  # (B, T2, 2*n_freqs)
            h = h + self.glu_layers[i](self.bn_layers[i](out))

        return self.fc(h.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiLayerMinimalMod(nn.Module):
    """MultiLayerMinimalStrided with data-dependent frequencies in layers 2+.
    Layer 1: fixed learned frequencies (mel-scale init), scalar input, stride by ds_factor.
    Layers 2+: data-dependent frequencies via Linear → ReLU → Linear → LayerNorm
    on the feature vector from previous layer."""
    def __init__(self, n_freqs=64, window=400, ds_factor=10, n_layers=3,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        assert window % ds_factor == 0
        self.n_freqs = n_freqs
        self.window = window
        self.ds_factor = ds_factor
        self.n_layers = n_layers
        self.inner_window = window // ds_factor

        # Layer 1: fixed learned frequencies (mel-scale init)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))
        dim = 2 * n_freqs
        self.bn1 = TransposedBN(dim)
        self.glu1 = GLU(dim, dropout)

        # Layers 2+: data-dependent frequencies + projection
        self.proj_layers = nn.ModuleList()
        self.freq_nets = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.proj_layers.append(nn.Linear(dim, dim))
            # Data-dependent frequencies: features → Linear → ReLU → Linear → LN
            self.freq_nets.append(nn.Sequential(
                nn.Linear(dim, n_freqs),
                nn.ReLU(),
                nn.Linear(n_freqs, n_freqs),
                nn.LayerNorm(n_freqs),
            ))
            self.bn_layers.append(TransposedBN(dim))
            self.glu_layers.append(GLU(dim, dropout))

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B_batch, T = x.shape

        # === Layer 1: fixed frequencies, scalar input → stride ===
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj().unsqueeze(0)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)  # unrotate
        h = torch.cat([d.real, d.imag], dim=-1)
        h = h[:, ::self.ds_factor]
        h = h + self.glu1(self.bn1(h))

        # === Layers 2+: data-dependent frequencies ===
        T2 = h.shape[1]
        for i in range(self.n_layers - 1):
            proj = self.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            # Per-timestep frequencies from features
            inst_freqs = self.freq_nets[i](h)  # (B, T2, n_freqs)
            cum_phase = inst_freqs.cumsum(dim=1)  # (B, T2, n_freqs)
            layer_phases = torch.exp(1j * cum_phase)  # (B, T2, n_freqs)

            rotated2 = z * layer_phases.conj()
            cs2 = rotated2.cumsum(dim=1)
            W = self.inner_window
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases  # unrotate
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.glu_layers[i](self.bn_layers[i](out))

        return self.fc(h.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiLayerMinimalV2(nn.Module):
    """Multi-layer windowed cumsum mimicking MelCNN's downsampling hierarchy.
    Layer 1: raw audio → windowed cumsum (W=window, stride=160) → T=100, like STFT
    Layers 2+: windowed cumsum + MaxPool(2) → T halves each layer
    All layers use fixed learned frequencies. Same temporal hierarchy as MelCNN."""
    def __init__(self, n_freqs=40, window=400, n_layers=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.window = window
        self.n_layers = n_layers
        dim = 2 * n_freqs  # real + imag

        # Layer 1: learned frequencies (mel-scale init), scalar input
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))
        self.bn1 = TransposedBN(dim)
        self.glu1 = GLU(dim, dropout)

        # Layers 2+: each has own learned frequencies, projection, BN, GLU
        self.proj_layers = nn.ModuleList()
        self.freq_params = nn.ParameterList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.proj_layers.append(nn.Linear(dim, dim))
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            self.bn_layers.append(TransposedBN(dim))
            self.glu_layers.append(GLU(dim, dropout))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        B_batch, T = x.shape

        # === Layer 1: scalar → windowed cumsum → stride 160 ===
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)  # (T, n_freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)  # (B, T, 1)
        rotated = x_complex * phases.conj().unsqueeze(0)  # (B, T, n_freqs)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)  # unrotate
        h = torch.cat([d.real, d.imag], dim=-1)  # (B, T, 2*n_freqs)
        h = h[:, ::160]  # stride 160 → T=100
        h = h + self.glu1(self.bn1(h))

        # === Layers 2+: cumsum + GLU + stride 2 downsample ===
        for i in range(self.n_layers - 1):
            T_cur = h.shape[1]
            proj = self.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            layer_freqs = self.freq_params[i]
            t2_idx = torch.arange(T_cur, device=x.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t2_idx.unsqueeze(1) * layer_freqs)

            rotated2 = z * layer_phases.conj().unsqueeze(0)
            cs2 = rotated2.cumsum(dim=1)
            # Window = full layer length (no hard cutoff within layer)
            W = T_cur
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases.unsqueeze(0)
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.glu_layers[i](self.bn_layers[i](out))
            # Downsample 2x — stride preserves real/imag pairing
            h = F.avg_pool1d(h.transpose(1, 2), 2).transpose(1, 2)

        return self.fc(h.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiLayerMinimalModV2(nn.Module):
    """MultiLayerMinimalV2 with data-dependent frequencies in layers 2+.
    Layer 1: fixed learned frequencies (mel-scale init) on raw audio, stride=160.
    Layers 2+: data-dependent frequencies via Linear → ReLU → Linear → LayerNorm.
    MaxPool(2) between layers, mimicking MelCNN's temporal hierarchy."""
    def __init__(self, n_freqs=40, window=400, n_layers=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.window = window
        self.n_layers = n_layers
        dim = 2 * n_freqs

        # Layer 1: fixed learned frequencies (mel-scale init)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))
        self.bn1 = TransposedBN(dim)
        self.glu1 = GLU(dim, dropout)

        # Layers 2+: data-dependent frequencies + projection
        self.proj_layers = nn.ModuleList()
        self.freq_nets = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.proj_layers.append(nn.Linear(dim, dim))
            self.freq_nets.append(nn.Sequential(
                nn.Linear(dim, n_freqs),
                nn.ReLU(),
                nn.Linear(n_freqs, n_freqs),
                nn.LayerNorm(n_freqs),
            ))
            self.bn_layers.append(TransposedBN(dim))
            self.glu_layers.append(GLU(dim, dropout))

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B_batch, T = x.shape

        # === Layer 1: fixed frequencies, scalar input → stride 160 ===
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj().unsqueeze(0)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window], (0, 0, self.window, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)  # unrotate
        h = torch.cat([d.real, d.imag], dim=-1)
        h = h[:, ::160]  # stride 160 → T=100
        h = h + self.glu1(self.bn1(h))

        # === Layers 2+: data-dependent frequencies + stride 2 downsample ===
        for i in range(self.n_layers - 1):
            T_cur = h.shape[1]
            proj = self.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            # Per-timestep frequencies from features
            inst_freqs = self.freq_nets[i](h)  # (B, T_cur, n_freqs)
            cum_phase = inst_freqs.cumsum(dim=1)
            layer_phases = torch.exp(1j * cum_phase)

            rotated2 = z * layer_phases.conj()
            cs2 = rotated2.cumsum(dim=1)
            # Window = full layer length
            W = T_cur
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases  # unrotate
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.glu_layers[i](self.bn_layers[i](out))
            # Downsample 2x — stride preserves real/imag pairing
            h = F.avg_pool1d(h.transpose(1, 2), 2).transpose(1, 2)

        return self.fc(h.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumMagDeep(nn.Module):
    """Every layer: proj to complex → rotate → cumsum → unrotate → mag → log → linear.
    Layer 1: raw audio, W=400, stride=160.
    Layers 2+: W=20, stride=1. Mag+log collapses phase every layer."""
    def __init__(self, n_freqs=40, window_l1=400, window=20, n_layers=4,
                 stride_l1=160, downsample=False, use_proj=False, freeze_proj=False,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.window_l1 = window_l1
        self.window = window
        self.n_layers = n_layers
        self.stride_l1 = stride_l1
        self.downsample = downsample
        self.use_proj = use_proj or freeze_proj  # freeze_proj implies use_proj
        dim = 2 * n_freqs

        # Layer 1: learned frequencies (mel-scale init)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))
        self.embed = nn.Linear(3 * n_freqs, dim)  # [log_mag, re, im] from layer 1

        # Layers 2+: cumsum → [log_mag, re, im] → linear
        self.proj_layers = nn.ModuleList() if self.use_proj else None
        self.freq_params = nn.ParameterList()
        self.embed_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            if self.use_proj:
                self.proj_layers.append(nn.Linear(dim, dim))
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            self.embed_layers.append(nn.Linear(3 * n_freqs, dim))

        # Orthogonal init + freeze for proj and embed layers
        if freeze_proj:
            with torch.no_grad():
                nn.init.orthogonal_(self.embed.weight)
                nn.init.zeros_(self.embed.bias)
                self.embed.weight.requires_grad = False
                self.embed.bias.requires_grad = False
                for proj in self.proj_layers:
                    nn.init.orthogonal_(proj.weight)
                    nn.init.zeros_(proj.bias)
                    proj.weight.requires_grad = False
                    proj.bias.requires_grad = False
                for emb in self.embed_layers:
                    nn.init.orthogonal_(emb.weight)
                    nn.init.zeros_(emb.bias)
                    emb.weight.requires_grad = False
                    emb.bias.requires_grad = False

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B_batch, T = x.shape

        # === Layer 1: raw audio → windowed cumsum → mag → log → embed ===
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj().unsqueeze(0)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window_l1], (0, 0, self.window_l1, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)
        mag = d.real ** 2 + d.imag ** 2
        log_mag = (mag + 1e-8).log()         # (B, T, n_freqs)
        features = torch.cat([log_mag, d.real, d.imag], dim=-1)  # (B, T, 3*n_freqs)
        features = features[:, ::self.stride_l1]  # (B, ~100, 3*n_freqs)
        h = self.embed(features)             # (B, ~100, dim)

        # === Layers 2+: [proj →] complex → cumsum(W=20) → [log_mag, re, im] → embed + residual ===
        for i in range(self.n_layers - 1):
            T_cur = h.shape[1]
            inp = self.proj_layers[i](h) if self.use_proj else h
            z_re, z_im = inp.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)    # (B, T_cur, n_freqs)

            layer_freqs = self.freq_params[i]
            t2_idx = torch.arange(T_cur, device=x.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t2_idx.unsqueeze(1) * layer_freqs)

            rotated2 = z * layer_phases.conj().unsqueeze(0)
            cs2 = rotated2.cumsum(dim=1)
            W = self.window
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases.unsqueeze(0)

            mag2 = d2.real ** 2 + d2.imag ** 2
            log_mag = (mag2 + 1e-8).log()     # (B, T_cur, n_freqs)
            features = torch.cat([log_mag, d2.real, d2.imag], dim=-1)  # (B, T_cur, 3*n_freqs)
            h = h + self.embed_layers[i](features)  # residual
            if self.downsample:
                h = F.avg_pool1d(h.transpose(1, 2), 2).transpose(1, 2)

        # Readout: maxpool → fc
        h = h.max(dim=1).values               # (B, dim)
        return self.fc(h)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumComplex(nn.Module):
    """End-to-end cumsum with complex-valued layers throughout.
    All mixing happens in complex domain via complex linear projections.
    Only the final readout splits to [re, im]."""
    def __init__(self, n_freqs=40, window_l1=400, window=20, n_layers=4,
                 stride_l1=160, readout="mlp_direct", readout_mult=4, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.window_l1 = window_l1
        self.window = window
        self.n_layers = n_layers
        self.stride_l1 = stride_l1
        self.readout_mode = readout

        # Layer 1: learned frequencies (mel-scale init)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))

        # Layers 2+: complex linear projections + frequency params + magnitude gating
        self.proj_re = nn.ModuleList()
        self.proj_im = nn.ModuleList()
        self.freq_params = nn.ParameterList()
        self.gate_linears = nn.ModuleList()
        self.scales = nn.ParameterList()
        for _ in range(n_layers - 1):
            self.proj_re.append(nn.Linear(n_freqs, n_freqs, bias=False))
            self.proj_im.append(nn.Linear(n_freqs, n_freqs, bias=False))
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            self.gate_linears.append(nn.Linear(n_freqs, n_freqs))
            self.scales.append(nn.Parameter(torch.ones(n_freqs)))

        # Readout
        dim = 2 * n_freqs
        rdim = dim * readout_mult
        if readout == "mlp_direct":
            self.readout = nn.Sequential(
                nn.Linear(dim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, num_classes),
            )
        elif readout == "mlp":
            self.readout = nn.Sequential(
                nn.Linear(dim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, n_freqs),
            )
            self.fc = nn.Linear(n_freqs, num_classes)
        else:  # mag
            self.fc = nn.Linear(n_freqs, num_classes)

    def forward(self, x):
        B_batch, T = x.shape

        # === Layer 1: raw audio → windowed cumsum → complex output ===
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj().unsqueeze(0)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window_l1], (0, 0, self.window_l1, 0))
        d = cs - cs_shifted
        h = d * phases.unsqueeze(0)          # (B, T, n_freqs) complex
        h = h[:, ::self.stride_l1]           # (B, ~100, n_freqs) complex

        # === Layers 2+: complex linear → windowed cumsum → normalize → gate → residual ===
        for i in range(self.n_layers - 1):
            T_cur = h.shape[1]

            # Complex linear: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            z_re = self.proj_re[i](h.real) - self.proj_im[i](h.imag)
            z_im = self.proj_re[i](h.imag) + self.proj_im[i](h.real)
            z = torch.complex(z_re, z_im)

            # Windowed cumsum
            layer_freqs = self.freq_params[i]
            t2_idx = torch.arange(T_cur, device=x.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t2_idx.unsqueeze(1) * layer_freqs)
            rotated2 = z * layer_phases.conj().unsqueeze(0)
            cs2 = rotated2.cumsum(dim=1)
            W = self.window
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases.unsqueeze(0)

            # Normalize by RMS magnitude per channel
            mag = torch.abs(d2)
            rms = mag.pow(2).mean(dim=1, keepdim=True).sqrt()
            d2 = d2 / (rms + 1e-8) * self.scales[i]

            # Magnitude gating
            mag_normed = torch.abs(d2)
            gate = torch.sigmoid(self.gate_linears[i](mag_normed))
            d2 = d2 * gate

            h = h + d2  # residual in complex domain

        # === Readout → maxpool ===
        if self.readout_mode == "mlp_direct":
            h_out = torch.cat([h.real, h.imag], dim=-1)
            h_out = self.readout(h_out)
            return h_out.max(dim=1).values
        elif self.readout_mode == "mlp":
            h_out = torch.cat([h.real, h.imag], dim=-1)
            h_out = self.readout(h_out)           # (B, T, n_freqs)
            h_out = h_out.max(dim=1).values       # (B, n_freqs)
            return self.fc(h_out)
        else:  # mag
            mag = torch.abs(h)                    # (B, T, n_freqs)
            mag = mag.max(dim=1).values           # (B, n_freqs)
            return self.fc(mag)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumSingleLayer(nn.Module):
    """Single cumsum layer with fixed frequencies → MLP readout → maxpool.
    No layers 2+. Just front-end + direct classification.
    features='re_im' for [re, im], 'mag_phase' for [log_mag, phase]."""
    def __init__(self, n_freqs=40, window_l1=400, stride_l1=160, readout_mult=4,
                 features='re_im', num_classes=NUM_CLASSES):
        super().__init__()
        self.n_freqs = n_freqs
        self.window_l1 = window_l1
        self.stride_l1 = stride_l1
        self.features = features
        dim = 3 * n_freqs if features == 'mag_phase' else 2 * n_freqs
        rdim = dim * readout_mult

        # Learned frequencies (mel-scale init)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))

        self.mlp = nn.Sequential(
            nn.Linear(dim, rdim),
            nn.ReLU(),
            nn.Linear(rdim, dim),
        )
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B_batch, T = x.shape
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj().unsqueeze(0)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window_l1], (0, 0, self.window_l1, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)
        if self.features == 'mag_phase':
            mag = (d.real ** 2 + d.imag ** 2 + 1e-8).sqrt()
            log_mag = mag.log()
            # Unit vector for phase (avoids atan2 discontinuity)
            cos_phase = d.real / mag
            sin_phase = d.imag / mag
            h = torch.cat([log_mag, cos_phase, sin_phase], dim=-1)
        else:
            h = torch.cat([d.real, d.imag], dim=-1)
        h = h[:, ::self.stride_l1]               # (B, ~100, dim)
        h = self.mlp(h)                           # (B, ~100, dim)
        h = h.max(dim=1).values                   # (B, dim)
        return self.fc(h)                          # (B, num_classes)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumEndToEnd(nn.Module):
    """Full cumsum end-to-end: no FFT/mel anywhere.
    Layer 1: learned frequencies (mel-scale init) on raw audio, windowed cumsum, stride 160 → T=100.
    Layers 2+: windowed cumsum with fixed learned frequencies, no downsampling.
    Classification: magnitude maxpool over all frames → fc."""
    def __init__(self, n_freqs=40, window_l1=400, window=20, n_layers=4, stride_l1=160, readout="mlp", readout_mult=1, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.window_l1 = window_l1
        self.window = window
        self.n_layers = n_layers
        self.stride_l1 = stride_l1
        self.readout_mode = readout
        dim = 2 * n_freqs
        rdim = dim * readout_mult

        # Layer 1: learned frequencies (mel-scale init), scalar input
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))
        self.bn1 = TransposedBN(dim)
        self.glu1 = GLU(dim, dropout)

        # Layers 2+: fixed learned frequencies, no downsampling
        self.proj_layers = nn.ModuleList()
        self.freq_params = nn.ParameterList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.proj_layers.append(nn.Linear(dim, dim))
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            self.bn_layers.append(TransposedBN(dim))
            self.glu_layers.append(GLU(dim, dropout))

        # Readout before maxpool
        if readout == "mlp":
            self.readout = nn.Sequential(
                nn.Linear(dim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, n_freqs),
            )
            self.fc = nn.Linear(n_freqs, num_classes)
        elif readout == "mlp_direct":
            self.readout = nn.Sequential(
                nn.Linear(dim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, num_classes),
            )
        else:  # mag
            self.fc = nn.Linear(n_freqs, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        B_batch, T = x.shape

        # === Layer 1: scalar → windowed cumsum → stride ===
        # Work in (B, n_freqs, T) layout so cumsum runs along contiguous dim
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases_t = torch.exp(1j * t_idx.unsqueeze(0) * freqs.unsqueeze(1))  # (n_freqs, T)
        x_complex = x.to(torch.complex64).unsqueeze(1)  # (B, 1, T)
        rotated = x_complex * phases_t.conj().unsqueeze(0)  # (B, n_freqs, T) — contiguous
        cs = rotated.cumsum(dim=2)  # fast: scan along contiguous dim

        # Window subtraction + unrotation at stride positions only
        out_idx = torch.arange(0, T, self.stride_l1, device=x.device)  # 100 positions
        cs_out = cs[:, :, out_idx]                                       # (B, n_freqs, 100)
        delay_idx = (out_idx - self.window_l1).clamp(min=0)
        cs_delayed = cs[:, :, delay_idx]                                 # (B, n_freqs, 100)
        mask = (out_idx >= self.window_l1).unsqueeze(0).unsqueeze(0)     # zero where t < window
        cs_delayed = cs_delayed * mask
        d = (cs_out - cs_delayed) * phases_t[:, out_idx].unsqueeze(0)    # unrotate
        d = d.transpose(1, 2)                                            # (B, 100, n_freqs)
        h = torch.cat([d.real, d.imag], dim=-1)                          # (B, 100, 2*n_freqs)
        h = h + self.glu1(self.bn1(h))

        # === Layers 2+: windowed cumsum, no downsampling ===
        for i in range(self.n_layers - 1):
            T_cur = h.shape[1]
            proj = self.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            layer_freqs = self.freq_params[i]
            t2_idx = torch.arange(T_cur, device=x.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t2_idx.unsqueeze(1) * layer_freqs)

            rotated2 = z * layer_phases.conj().unsqueeze(0)
            cs2 = rotated2.cumsum(dim=1)
            W = self.window
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases.unsqueeze(0)
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.glu_layers[i](self.bn_layers[i](out))

        # Readout → maxpool → classify
        if self.readout_mode == "mlp":
            h = self.readout(h)                              # (B, T, n_freqs)
            h = h.max(dim=1).values                          # (B, n_freqs)
            return self.fc(h)
        elif self.readout_mode == "mlp_direct":
            h = self.readout(h)                              # (B, T, num_classes)
            return h.max(dim=1).values                       # (B, num_classes)
        else:
            h_re, h_im = h.chunk(2, dim=-1)
            h = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)    # (B, T, n_freqs)
            h = h.max(dim=1).values                          # (B, n_freqs)
            return self.fc(h)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumEndToEndMag(nn.Module):
    """Full cumsum end-to-end with magnitude+log between layer 1 and layers 2+.
    Layer 1: learned frequencies on raw audio → re²+im² → log → Linear(n_freqs, dim) → layers 2+.
    Like a learned mel spectrogram feeding windowed cumsum layers.
    Classification: magnitude maxpool over all frames → fc."""
    def __init__(self, n_freqs=40, window_l1=400, window=20, n_layers=4, stride_l1=160, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.window_l1 = window_l1
        self.window = window
        self.n_layers = n_layers
        self.stride_l1 = stride_l1
        dim = 2 * n_freqs

        # Layer 1: learned frequencies (mel-scale init), scalar input
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))
        self.embed = nn.Linear(n_freqs, dim)

        # Layers 2+: fixed learned frequencies, no downsampling
        self.proj_layers = nn.ModuleList()
        self.freq_params = nn.ParameterList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.proj_layers.append(nn.Linear(dim, dim))
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            self.bn_layers.append(TransposedBN(dim))
            self.glu_layers.append(GLU(dim, dropout))

        self.fc = nn.Linear(n_freqs, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        B_batch, T = x.shape

        # === Layer 1: scalar → windowed cumsum → magnitude → log → embed ===
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)  # (T, n_freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)  # (B, T, 1)
        rotated = x_complex * phases.conj().unsqueeze(0)  # (B, T, n_freqs)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window_l1], (0, 0, self.window_l1, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)  # unrotate
        # Magnitude + log (like mel spectrogram)
        mag = d.real ** 2 + d.imag ** 2              # (B, T, n_freqs)
        h = (mag + 1e-8).log()                       # (B, T, n_freqs)
        h = h[:, ::self.stride_l1]                   # stride → T frames
        h = self.embed(h)                            # (B, T, dim=2*n_freqs)

        # === Layers 2+: windowed cumsum, no downsampling ===
        for i in range(self.n_layers - 1):
            T_cur = h.shape[1]
            proj = self.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            layer_freqs = self.freq_params[i]
            t2_idx = torch.arange(T_cur, device=x.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t2_idx.unsqueeze(1) * layer_freqs)

            rotated2 = z * layer_phases.conj().unsqueeze(0)
            cs2 = rotated2.cumsum(dim=1)
            W = self.window
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases.unsqueeze(0)
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.glu_layers[i](self.bn_layers[i](out))

        # Phase-invariant: magnitude maxpool over all frames
        h_re, h_im = h.chunk(2, dim=-1)              # (B, T, n_freqs) each
        mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)  # (B, T, n_freqs)
        mag = mag.max(dim=1).values                   # (B, n_freqs)
        return self.fc(mag)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumEndToEndMagMod(nn.Module):
    """Full cumsum end-to-end with magnitude+log layer 1 and data-dependent layers 2+.
    Layer 1: learned frequencies on raw audio → re²+im² → log → Linear(n_freqs, dim) → layers 2+.
    Layers 2+: data-dependent frequency cumsum, windowed, no downsampling.
    Classification: magnitude maxpool over all frames → fc."""
    def __init__(self, n_freqs=40, window_l1=400, window=20, n_layers=4, stride_l1=160, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.window_l1 = window_l1
        self.window = window
        self.n_layers = n_layers
        self.stride_l1 = stride_l1
        dim = 2 * n_freqs

        # Layer 1: learned frequencies (mel-scale init), scalar input
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))
        self.embed = nn.Linear(n_freqs, dim)

        # Layers 2+: data-dependent frequencies, no downsampling
        self.proj_layers = nn.ModuleList()
        self.freq_nets = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.proj_layers.append(nn.Linear(dim, dim))
            self.freq_nets.append(nn.Sequential(
                nn.Linear(dim, n_freqs),
                nn.ReLU(),
                nn.Linear(n_freqs, n_freqs),
                nn.LayerNorm(n_freqs),
            ))
            self.bn_layers.append(TransposedBN(dim))
            self.glu_layers.append(GLU(dim, dropout))

        self.fc = nn.Linear(n_freqs, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        B_batch, T = x.shape

        # === Layer 1: scalar → windowed cumsum → magnitude → log → embed ===
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)  # (T, n_freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)  # (B, T, 1)
        rotated = x_complex * phases.conj().unsqueeze(0)  # (B, T, n_freqs)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window_l1], (0, 0, self.window_l1, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)  # unrotate
        mag = d.real ** 2 + d.imag ** 2
        h = (mag + 1e-8).log()
        h = h[:, ::self.stride_l1]                   # stride → T frames
        h = self.embed(h)  # (B, T, dim)

        # === Layers 2+: data-dependent windowed cumsum, no downsampling ===
        for i in range(self.n_layers - 1):
            T_cur = h.shape[1]
            proj = self.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            inst_freqs = self.freq_nets[i](h)
            cum_phase = inst_freqs.cumsum(dim=1)
            layer_phases = torch.exp(1j * cum_phase)

            rotated2 = z * layer_phases.conj()
            cs2 = rotated2.cumsum(dim=1)
            W = self.window
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.glu_layers[i](self.bn_layers[i](out))

        # Phase-invariant: magnitude maxpool over all frames
        h_re, h_im = h.chunk(2, dim=-1)
        mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        mag = mag.max(dim=1).values
        return self.fc(mag)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumEndToEndMod(nn.Module):
    """Full cumsum end-to-end with data-dependent frequencies in layers 2+.
    Layer 1: learned frequencies (mel-scale init) on raw audio, windowed cumsum, stride 160 → T=100.
    Layers 2+: data-dependent frequency cumsum, windowed, no downsampling.
    Classification: magnitude maxpool over all frames → fc."""
    def __init__(self, n_freqs=40, window_l1=400, window=20, n_layers=4, stride_l1=160, readout="mlp", readout_mult=1, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.window_l1 = window_l1
        self.window = window
        self.n_layers = n_layers
        self.stride_l1 = stride_l1
        self.readout_mode = readout
        dim = 2 * n_freqs
        rdim = dim * readout_mult

        # Layer 1: learned frequencies (mel-scale init), scalar input
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.log_freqs = nn.Parameter(torch.log(angular_freqs))
        self.bn1 = TransposedBN(dim)
        self.glu1 = GLU(dim, dropout)

        # Layers 2+: data-dependent frequencies, no downsampling
        self.proj_layers = nn.ModuleList()
        self.freq_nets = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.proj_layers.append(nn.Linear(dim, dim))
            self.freq_nets.append(nn.Sequential(
                nn.Linear(dim, n_freqs),
                nn.ReLU(),
                nn.Linear(n_freqs, n_freqs),
                nn.LayerNorm(n_freqs),
            ))
            self.bn_layers.append(TransposedBN(dim))
            self.glu_layers.append(GLU(dim, dropout))

        # Readout before maxpool
        if readout == "mlp":
            self.readout = nn.Sequential(
                nn.Linear(dim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, n_freqs),
            )
            self.fc = nn.Linear(n_freqs, num_classes)
        elif readout == "mlp_direct":
            self.readout = nn.Sequential(
                nn.Linear(dim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, num_classes),
            )
        else:  # mag
            self.fc = nn.Linear(n_freqs, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        B_batch, T = x.shape

        # === Layer 1: scalar → windowed cumsum → stride 160 ===
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)  # (T, n_freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)  # (B, T, 1)
        rotated = x_complex * phases.conj().unsqueeze(0)  # (B, T, n_freqs)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window_l1], (0, 0, self.window_l1, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)  # unrotate
        h = torch.cat([d.real, d.imag], dim=-1)  # (B, T, 2*n_freqs)
        h = h[:, ::self.stride_l1]  # stride → T frames
        h = h + self.glu1(self.bn1(h))

        # === Layers 2+: data-dependent windowed cumsum, no downsampling ===
        for i in range(self.n_layers - 1):
            T_cur = h.shape[1]
            proj = self.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            inst_freqs = self.freq_nets[i](h)    # (B, T_cur, n_freqs)
            cum_phase = inst_freqs.cumsum(dim=1)
            layer_phases = torch.exp(1j * cum_phase)

            rotated2 = z * layer_phases.conj()
            cs2 = rotated2.cumsum(dim=1)
            W = self.window
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.glu_layers[i](self.bn_layers[i](out))

        # Readout → maxpool → classify
        if self.readout_mode == "mlp":
            h = self.readout(h)                              # (B, T, n_freqs)
            h = h.max(dim=1).values                          # (B, n_freqs)
            return self.fc(h)
        elif self.readout_mode == "mlp_direct":
            h = self.readout(h)                              # (B, T, num_classes)
            return h.max(dim=1).values                       # (B, num_classes)
        else:
            h_re, h_im = h.chunk(2, dim=-1)
            h = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)    # (B, T, n_freqs)
            h = h.max(dim=1).values                          # (B, n_freqs)
            return self.fc(h)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelCumsumFixed(nn.Module):
    """Mel spectrogram front-end → fixed-frequency cumsum layers (no downsampling).
    Uses proven FFT+mel+log front-end from MelCNN, then stacks cumsum layers
    with fixed learned frequencies to process the 101-frame sequence.
    Last-frame classification (causal)."""
    def __init__(self, n_embed=80, n_layers=4, window=None, hop_length=160, n_phases=1, tie_layers=False, num_classes=NUM_CLASSES, dropout=0.1, zero_freqs=False):
        super().__init__()
        self.n_layers = n_layers
        self.window = window  # None = full sequence
        self.n_phases = n_phases
        self.tie_layers = tie_layers
        self.zero_freqs = zero_freqs
        n_freqs = n_embed // 2

        # Mel front-end
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=hop_length,
            n_mels=40, power=2.0,
        )
        self.spec_aug = SpecAugment()
        self.embed = nn.Linear(40, n_embed)

        # Cumsum layers (fixed frequencies)
        if tie_layers:
            # Shared proj and GLU across all layers
            self.shared_proj = nn.Linear(n_embed, n_embed)
            self.shared_glu = GLU(n_embed, dropout)
        else:
            self.proj_layers = nn.ModuleList()
            self.glu_layers = nn.ModuleList()
        self.freq_params = nn.ParameterList()
        self.bn_layers = nn.ModuleList()
        for i in range(n_layers):
            if not tie_layers:
                self.proj_layers.append(nn.Linear(n_embed, n_embed))
                self.glu_layers.append(GLU(n_embed, dropout))
            if zero_freqs:
                self.freq_params.append(nn.Parameter(
                    torch.zeros(n_freqs), requires_grad=False))
            else:
                self.freq_params.append(nn.Parameter(
                    torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            self.bn_layers.append(TransposedBN(n_embed))

        self.fc = nn.Linear(n_freqs, num_classes)

    def _process_seq(self, h):
        """Process a single mel sequence through cumsum layers → magnitude."""
        for i in range(self.n_layers):
            T_cur = h.shape[1]
            proj_layer = self.shared_proj if self.tie_layers else self.proj_layers[i]
            glu_layer = self.shared_glu if self.tie_layers else self.glu_layers[i]

            proj = proj_layer(h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            layer_freqs = self.freq_params[i]
            t_idx = torch.arange(T_cur, device=h.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t_idx.unsqueeze(1) * layer_freqs)

            rotated = z * layer_phases.conj().unsqueeze(0)
            cs = rotated.cumsum(dim=1)
            W = self.window if self.window is not None else T_cur
            cs_shifted = F.pad(cs[:, :-W], (0, 0, W, 0))
            d = cs - cs_shifted
            d = d * layer_phases.unsqueeze(0)
            out = torch.cat([d.real, d.imag], dim=-1)
            h = h + glu_layer(self.bn_layers[i](out))

        # Phase-invariant: collapse re/im pairs to magnitude
        if self.window is not None:
            h_re, h_im = h.chunk(2, dim=-1)
            mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
            mag = mag.max(dim=1).values
        else:
            h_last = h[:, -1, :]
            h_re, h_im = h_last.chunk(2, dim=-1)
            mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        return self.fc(mag)

    def forward(self, x):
        # x: (B, 16000)
        x = x.unsqueeze(1)                    # (B, 1, 16000)
        x = self.mel_spec(x)                  # (B, 1, 40, T)
        x = x.squeeze(1)                      # (B, 40, T)
        x = (x + 1e-8).log()
        if self.training:
            x = self.spec_aug(x)

        if self.n_phases <= 1:
            x = x.transpose(1, 2)             # (B, T, 40)
            h = self.embed(x)                 # (B, T, n_embed)
            return self._process_seq(h)
        else:
            outputs = []
            for p in range(self.n_phases):
                x_phase = x[:, :, p::self.n_phases]  # (B, 40, ~101)
                x_phase = x_phase.transpose(1, 2)    # (B, ~101, 40)
                h = self.embed(x_phase)
                outputs.append(self._process_seq(h))
            return torch.stack(outputs, dim=0).max(dim=0).values

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelScanFixed(nn.Module):
    """Mel spectrogram front-end → scan (exponential decay) layers.
    Identical to MelCumsumFixed but replaces cumsum+window with parallel_scan+lambda.
    Apples-to-apples comparison: same mel front-end, same proj, same BN, same GLU,
    same frequencies. Only the sequence operation differs:
      cumsum: d[t] = sum(rotated[t-W:t])     (hard window, FIR)
      scan:   d[t] = λ·d[t-1] + rotated[t]   (exponential decay, IIR)
    Extra params: one decay scalar per frequency per layer."""
    def __init__(self, n_embed=80, n_layers=4, hop_length=160, tie_layers=False, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.tie_layers = tie_layers
        n_freqs = n_embed // 2

        # Mel front-end (identical to MelCumsumFixed)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=hop_length,
            n_mels=40, power=2.0,
        )
        self.spec_aug = SpecAugment()
        self.embed = nn.Linear(40, n_embed)

        # Scan layers (same structure as cumsum, but with decay instead of window)
        if tie_layers:
            self.shared_proj = nn.Linear(n_embed, n_embed)
            self.shared_glu = GLU(n_embed, dropout)
        else:
            self.proj_layers = nn.ModuleList()
            self.glu_layers = nn.ModuleList()
        self.freq_params = nn.ParameterList()
        self.decay_params = nn.ParameterList()
        self.bn_layers = nn.ModuleList()
        for i in range(n_layers):
            if not tie_layers:
                self.proj_layers.append(nn.Linear(n_embed, n_embed))
                self.glu_layers.append(GLU(n_embed, dropout))
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            # sigmoid(2.2) ≈ 0.9, effective window ≈ 10 (matches cumsum W=10)
            self.decay_params.append(nn.Parameter(
                torch.full((n_freqs,), 2.2)))
            self.bn_layers.append(TransposedBN(n_embed))

        self.fc = nn.Linear(n_freqs, num_classes)

    def _process_seq(self, h):
        """Process mel sequence through scan layers → magnitude."""
        for i in range(self.n_layers):
            B_batch, T_cur, _ = h.shape
            proj_layer = self.shared_proj if self.tie_layers else self.proj_layers[i]
            glu_layer = self.shared_glu if self.tie_layers else self.glu_layers[i]

            proj = proj_layer(h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            # Same rotation as cumsum
            layer_freqs = self.freq_params[i]
            t_idx = torch.arange(T_cur, device=h.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t_idx.unsqueeze(1) * layer_freqs)
            rotated = z * layer_phases.conj().unsqueeze(0)

            # SCAN with exponential decay (replaces cumsum+window)
            decay = torch.sigmoid(self.decay_params[i])  # (n_freqs,) in (0, 1)
            gates = torch.complex(decay, torch.zeros_like(decay))
            gates = gates.unsqueeze(0).unsqueeze(0).expand(B_batch, T_cur, -1)
            d = parallel_scan(gates, rotated)

            d = d * layer_phases.unsqueeze(0)  # unrotate
            out = torch.cat([d.real, d.imag], dim=-1)
            h = h + glu_layer(self.bn_layers[i](out))

        # Magnitude maxpool readout
        h_re, h_im = h.chunk(2, dim=-1)
        mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        mag = mag.max(dim=1).values
        return self.fc(mag)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = x.squeeze(1)
        x = (x + 1e-8).log()
        if self.training:
            x = self.spec_aug(x)
        x = x.transpose(1, 2)
        h = self.embed(x)
        return self._process_seq(h)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelCumsumBidirTied(nn.Module):
    """Mel front-end → bidirectional tied cumsum layers with fixed frequencies.
    Forward and backward cumsums with separate freq params, outputs summed before GLU."""
    def __init__(self, n_embed=80, n_layers=4, window=None, hop_length=160, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.window = window
        n_freqs = n_embed // 2

        # Mel front-end
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=hop_length,
            n_mels=40, power=2.0,
        )
        self.spec_aug = SpecAugment()
        self.embed = nn.Linear(40, n_embed)

        # Shared proj and GLU (tied across layers and directions)
        self.shared_proj = nn.Linear(n_embed, n_embed)
        self.shared_glu = GLU(n_embed, dropout)

        # Per-layer: shared freq params for fwd and bwd, shared BN
        self.freq_params = nn.ParameterList()
        self.bn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            self.bn_layers.append(TransposedBN(n_embed))

        self.fc = nn.Linear(n_freqs, num_classes)

    def _process_seq(self, h):
        for i in range(self.n_layers):
            T_cur = h.shape[1]
            W = self.window if self.window is not None else T_cur

            # Shared projection and frequencies
            proj = self.shared_proj(h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)
            freqs = self.freq_params[i]
            t_idx = torch.arange(T_cur, device=h.device, dtype=freqs.dtype)
            phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)

            # Forward cumsum
            rotated_f = z * phases.conj().unsqueeze(0)
            cs_f = rotated_f.cumsum(dim=1)
            cs_shifted_f = F.pad(cs_f[:, :-W], (0, 0, W, 0))
            d_f = (cs_f - cs_shifted_f) * phases.unsqueeze(0)

            # Backward cumsum
            rotated_b = z * phases.conj().unsqueeze(0)
            cs_b = rotated_b.flip(1).cumsum(dim=1).flip(1)
            cs_shifted_b = F.pad(cs_b[:, W:], (0, 0, 0, W))
            d_b = (cs_b - cs_shifted_b) * phases.unsqueeze(0)

            # Combine forward + backward
            d = d_f + d_b
            out = torch.cat([d.real, d.imag], dim=-1)
            h = h + self.shared_glu(self.bn_layers[i](out))

        # Magnitude + max pool
        h_re, h_im = h.chunk(2, dim=-1)
        mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        mag = mag.max(dim=1).values
        return self.fc(mag)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = x.squeeze(1)
        x = (x + 1e-8).log()
        if self.training:
            x = self.spec_aug(x)
        x = x.transpose(1, 2)
        h = self.embed(x)
        return self._process_seq(h)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelCumsumResidualModTied(nn.Module):
    """Mel front-end → tied cumsum layers with fixed frequencies + data-dependent delta.
    inst_freq = freq_params[i] + delta_scale * delta(h), phase = cumsum(inst_freq).
    Shared proj (n_embed → n_embed) for re/im, shared freq_proj (n_embed → n_freqs) for delta.
    Shared GLU, per-layer BN only."""
    def __init__(self, n_embed=80, n_layers=4, window=None, hop_length=160, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.window = window
        n_freqs = n_embed // 2

        # Mel front-end
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=hop_length,
            n_mels=40, power=2.0,
        )
        self.spec_aug = SpecAugment()
        self.embed = nn.Linear(40, n_embed)

        # Shared proj for re/im (same as FixedTied)
        self.shared_proj = nn.Linear(n_embed, n_embed)
        # Shared proj for frequency delta (no bias to prevent constant output)
        self.shared_freq_proj = nn.Linear(n_embed, n_freqs, bias=False)
        self.freq_ln = nn.LayerNorm(n_freqs, bias=False)
        self.shared_glu = GLU(n_embed, dropout)
        # Per-layer: fixed frequencies, delta scale, BN
        self.freq_params = nn.ParameterList()
        self.bn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            self.register_buffer(f'delta_scale_{i}', torch.tensor(0.1))
            self.bn_layers.append(TransposedBN(n_embed))

        self.fc = nn.Linear(n_freqs, num_classes)

    def _process_seq(self, h):
        n_freqs = h.shape[-1] // 2
        for i in range(self.n_layers):
            T_cur = h.shape[1]
            proj = self.shared_proj(h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            # Fixed freq + scaled delta
            delta = getattr(self, f'delta_scale_{i}') * self.freq_ln(self.shared_freq_proj(h))  # (B, T, n_freqs)
            inst_freq = self.freq_params[i] + delta  # (B, T, n_freqs)
            cum_phase = inst_freq.cumsum(dim=1)
            layer_phases = torch.exp(1j * cum_phase)

            rotated = z * layer_phases.conj()
            cs = rotated.cumsum(dim=1)
            W = self.window if self.window is not None else T_cur
            cs_shifted = F.pad(cs[:, :-W], (0, 0, W, 0))
            d = cs - cs_shifted
            d = d * layer_phases
            out = torch.cat([d.real, d.imag], dim=-1)
            h = h + self.shared_glu(self.bn_layers[i](out))

        if self.window is not None:
            h_re, h_im = h.chunk(2, dim=-1)
            mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
            mag = mag.max(dim=1).values
        else:
            h_last = h[:, -1, :]
            h_re, h_im = h_last.chunk(2, dim=-1)
            mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        return self.fc(mag)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = x.squeeze(1)
        x = (x + 1e-8).log()
        if self.training:
            x = self.spec_aug(x)
        x = x.transpose(1, 2)
        h = self.embed(x)
        return self._process_seq(h)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelCumsumModTied(nn.Module):
    """Mel front-end → tied cumsum layers with data-dependent frequencies.
    Shared proj for signal (n_embed → n_embed), separate shared freq_proj (n_embed → n_freqs).
    Shared GLU, per-layer BN only."""
    def __init__(self, n_embed=80, n_layers=4, window=None, hop_length=160, n_phases=1, num_classes=NUM_CLASSES, dropout=0.1, freq_bias=True, freq_bottleneck=0, future_phase=False, phase_mode="default"):
        super().__init__()
        self.n_layers = n_layers
        self.window = window
        self.n_phases = n_phases
        # future_phase=True maps to "derot_prev" for backwards compat
        if future_phase and phase_mode == "default":
            phase_mode = "derot_prev"
        self.phase_mode = phase_mode
        n_freqs = n_embed // 2

        # Mel front-end
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=hop_length,
            n_mels=40, power=2.0,
        )
        self.spec_aug = SpecAugment()
        self.embed = nn.Linear(40, n_embed)

        # Shared proj for re + im
        self.shared_proj = nn.Linear(n_embed, n_embed)
        # Shared proj for freq
        if freq_bottleneck > 0:
            self.shared_freq_proj = nn.Sequential(
                nn.Linear(n_embed, freq_bottleneck),
                nn.ReLU(),
                nn.Linear(freq_bottleneck, n_freqs, bias=False),
            )
        elif freq_bottleneck < 0:
            expanded = -freq_bottleneck
            self.shared_freq_proj = nn.Sequential(
                nn.Linear(n_embed, expanded),
                nn.ReLU(),
                nn.Linear(expanded, n_freqs, bias=False),
            )
        else:
            self.shared_freq_proj = nn.Linear(n_embed, n_freqs, bias=freq_bias)
        self.freq_ln = nn.LayerNorm(n_freqs)
        self.shared_glu = GLU(n_embed, dropout)
        self.bn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.bn_layers.append(TransposedBN(n_embed))

        self.fc = nn.Linear(n_freqs, num_classes)

    def _process_seq(self, h):
        """Process a single mel sequence through cumsum layers → magnitude."""
        n_freqs = h.shape[-1] // 2
        for i in range(self.n_layers):
            T_cur = h.shape[1]
            proj = self.shared_proj(h)  # (B, T, n_embed)
            z_re, z_im = proj.chunk(2, dim=-1)  # each (B, T, n_freqs)
            inst_freqs = self.freq_ln(self.shared_freq_proj(h))  # (B, T, n_freqs)
            z = torch.complex(z_re, z_im)

            cum_phase = inst_freqs.cumsum(dim=1)
            if self.phase_mode == "derot_prev":
                # Derotate with Φ(t-1), rerotate with Φ(t)
                derot_phase = F.pad(cum_phase[:, :-1], (0, 0, 1, 0))
                layer_phases_derot = torch.exp(1j * derot_phase)
                layer_phases_rerot = torch.exp(1j * cum_phase)
            elif self.phase_mode == "both_prev":
                # Both derot and rerot use Φ(t-1)
                shifted_phase = F.pad(cum_phase[:, :-1], (0, 0, 1, 0))
                layer_phases_derot = torch.exp(1j * shifted_phase)
                layer_phases_rerot = layer_phases_derot
            else:
                # Default: both use Φ(t)
                layer_phases_derot = torch.exp(1j * cum_phase)
                layer_phases_rerot = layer_phases_derot

            rotated = z * layer_phases_derot.conj()
            cs = rotated.cumsum(dim=1)
            W = self.window if self.window is not None else T_cur
            cs_shifted = F.pad(cs[:, :-W], (0, 0, W, 0))
            d = cs - cs_shifted
            d = d * layer_phases_rerot
            out = torch.cat([d.real, d.imag], dim=-1)
            h = h + self.shared_glu(self.bn_layers[i](out))

        # Phase-invariant: collapse re/im pairs to magnitude
        if self.window is not None:
            h_re, h_im = h.chunk(2, dim=-1)
            mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
            mag = mag.max(dim=1).values
        else:
            h_last = h[:, -1, :]
            h_re, h_im = h_last.chunk(2, dim=-1)
            mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        return self.fc(mag)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = x.squeeze(1)
        x = (x + 1e-8).log()
        if self.training:
            x = self.spec_aug(x)

        if self.n_phases <= 1:
            x = x.transpose(1, 2)
            h = self.embed(x)
            return self._process_seq(h)
        else:
            outputs = []
            for p in range(self.n_phases):
                x_phase = x[:, :, p::self.n_phases]
                x_phase = x_phase.transpose(1, 2)
                h = self.embed(x_phase)
                outputs.append(self._process_seq(h))
            return torch.stack(outputs, dim=0).max(dim=0).values

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelCumsumMod(nn.Module):
    """Mel spectrogram front-end → data-dependent frequency cumsum layers (no downsampling).
    Uses proven FFT+mel+log front-end from MelCNN, then stacks cumsum layers
    with per-timestep data-dependent frequencies to process the 101-frame sequence.
    Last-frame classification (causal)."""
    def __init__(self, n_embed=80, n_layers=4, window=None, hop_length=160, n_phases=1, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.window = window  # None = full sequence
        self.n_phases = n_phases
        n_freqs = n_embed // 2

        # Mel front-end
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=hop_length,
            n_mels=40, power=2.0,
        )
        self.spec_aug = SpecAugment()
        self.embed = nn.Linear(40, n_embed)

        # Cumsum layers (data-dependent frequencies)
        self.proj_layers = nn.ModuleList()
        self.freq_nets = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.proj_layers.append(nn.Linear(n_embed, n_embed))
            self.freq_nets.append(nn.Sequential(
                nn.Linear(n_embed, n_freqs),
                nn.ReLU(),
                nn.Linear(n_freqs, n_freqs),
                nn.LayerNorm(n_freqs),
            ))
            self.bn_layers.append(TransposedBN(n_embed))
            self.glu_layers.append(GLU(n_embed, dropout))

        self.fc = nn.Linear(n_freqs, num_classes)

    def _process_seq(self, h):
        """Process a single mel sequence through cumsum layers → magnitude."""
        for i in range(self.n_layers):
            T_cur = h.shape[1]
            proj = self.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            inst_freqs = self.freq_nets[i](h)
            cum_phase = inst_freqs.cumsum(dim=1)
            layer_phases = torch.exp(1j * cum_phase)

            rotated = z * layer_phases.conj()
            cs = rotated.cumsum(dim=1)
            W = self.window if self.window is not None else T_cur
            cs_shifted = F.pad(cs[:, :-W], (0, 0, W, 0))
            d = cs - cs_shifted
            d = d * layer_phases
            out = torch.cat([d.real, d.imag], dim=-1)
            h = h + self.glu_layers[i](self.bn_layers[i](out))

        # Phase-invariant: collapse re/im pairs to magnitude
        if self.window is not None:
            h_re, h_im = h.chunk(2, dim=-1)
            mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
            mag = mag.max(dim=1).values
        else:
            h_last = h[:, -1, :]
            h_re, h_im = h_last.chunk(2, dim=-1)
            mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        return self.fc(mag)

    def forward(self, x):
        # x: (B, 16000)
        x = x.unsqueeze(1)                    # (B, 1, 16000)
        x = self.mel_spec(x)                  # (B, 1, 40, T)
        x = x.squeeze(1)                      # (B, 40, T)
        x = (x + 1e-8).log()
        if self.training:
            x = self.spec_aug(x)

        if self.n_phases <= 1:
            x = x.transpose(1, 2)
            h = self.embed(x)
            return self._process_seq(h)
        else:
            outputs = []
            for p in range(self.n_phases):
                x_phase = x[:, :, p::self.n_phases]
                x_phase = x_phase.transpose(1, 2)
                h = self.embed(x_phase)
                outputs.append(self._process_seq(h))
            return torch.stack(outputs, dim=0).max(dim=0).values

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelCumsumMagDeep(nn.Module):
    """Mel front-end → fixed-frequency cumsum layers with MagDeep-style mixing.
    Each layer: proj → complex → rotate → cumsum → unrotate → [log_mag, re, im] → Linear → residual.
    Like MelCumsumFixed but replaces BN+GLU with mag/re/im mixing (from CumsumMagDeep)."""
    def __init__(self, n_embed=80, n_layers=4, window=None, hop_length=160,
                 tie_layers=False, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_layers = n_layers
        self.window = window
        self.tie_layers = tie_layers
        n_freqs = n_embed // 2

        # Mel front-end
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=hop_length,
            n_mels=40, power=2.0,
        )
        self.spec_aug = SpecAugment()
        self.embed = nn.Linear(40, n_embed)

        # Cumsum layers
        if tie_layers:
            self.shared_proj = nn.Linear(n_embed, n_embed)
            self.shared_mix = nn.Linear(3 * n_freqs, n_embed)
        else:
            self.proj_layers = nn.ModuleList()
            self.mix_layers = nn.ModuleList()
        self.freq_params = nn.ParameterList()
        for _ in range(n_layers):
            if not tie_layers:
                self.proj_layers.append(nn.Linear(n_embed, n_embed))
                self.mix_layers.append(nn.Linear(3 * n_freqs, n_embed))
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))

        self.fc = nn.Linear(n_freqs, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = x.squeeze(1)
        x = (x + 1e-8).log()
        if self.training:
            x = self.spec_aug(x)
        x = x.transpose(1, 2)
        h = self.embed(x)

        n_freqs = h.shape[-1] // 2
        for i in range(self.n_layers):
            T_cur = h.shape[1]
            proj_layer = self.shared_proj if self.tie_layers else self.proj_layers[i]
            mix_layer = self.shared_mix if self.tie_layers else self.mix_layers[i]

            proj = proj_layer(h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            layer_freqs = self.freq_params[i]
            t_idx = torch.arange(T_cur, device=h.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t_idx.unsqueeze(1) * layer_freqs)

            rotated = z * layer_phases.conj().unsqueeze(0)
            cs = rotated.cumsum(dim=1)
            W = self.window if self.window is not None else T_cur
            cs_shifted = F.pad(cs[:, :-W], (0, 0, W, 0))
            d = cs - cs_shifted
            d = d * layer_phases.unsqueeze(0)

            mag = d.real ** 2 + d.imag ** 2
            log_mag = (mag + 1e-8).log()
            features = torch.cat([log_mag, d.real, d.imag], dim=-1)  # (B, T, 3*n_freqs)
            h = h + mix_layer(features)

        # Readout: magnitude → max pool → fc
        h_re, h_im = h.chunk(2, dim=-1)
        mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        mag = mag.max(dim=1).values
        return self.fc(mag)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumSpecCumsumTied(nn.Module):
    """Frozen cumsum spectrogram front-end → [log_mag, re, im] → Linear → tied cumsum backend.
    Replaces mel front-end with a frozen cumsum layer (mel-scale init), keeping the
    proven BN+GLU tied cumsum backend from MelCumsumFixedTied."""
    def __init__(self, n_freqs=40, n_embed=80, n_layers=8, window_l1=400,
                 hop_length=160, window=5, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.window = window
        self.hop = hop_length
        self.window_l1 = window_l1
        n_backend_freqs = n_embed // 2

        # Frozen cumsum front-end (mel-scale init)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.register_buffer('log_freqs', torch.log(angular_freqs))

        # [log_mag, re, im] → n_embed
        self.embed = nn.Linear(3 * n_freqs, n_embed)
        self.spec_aug = SpecAugment()

        # Tied cumsum backend (same as MelCumsumFixedTied)
        self.shared_proj = nn.Linear(n_embed, n_embed)
        self.shared_glu = GLU(n_embed, dropout)
        self.freq_params = nn.ParameterList()
        self.bn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_backend_freqs)))
            self.bn_layers.append(TransposedBN(n_embed))

        self.fc = nn.Linear(n_backend_freqs, num_classes)

    def forward(self, x):
        B_batch, T = x.shape

        # Frozen cumsum spectrogram
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj().unsqueeze(0)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window_l1], (0, 0, self.window_l1, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)

        mag = d.real ** 2 + d.imag ** 2
        log_mag = (mag + 1e-8).log()
        features = torch.cat([log_mag, d.real, d.imag], dim=-1)
        features = features[:, ::self.hop]  # (B, ~100, 3*n_freqs)

        h = self.embed(features)  # (B, ~100, n_embed)
        if self.training:
            # Apply SpecAugment in (B, C, T) format
            h = self.spec_aug(h.transpose(1, 2)).transpose(1, 2)

        # Tied cumsum backend
        n_backend_freqs = h.shape[-1] // 2
        for i in range(self.n_layers):
            T_cur = h.shape[1]
            proj = self.shared_proj(h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            layer_freqs = self.freq_params[i]
            t_idx2 = torch.arange(T_cur, device=h.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t_idx2.unsqueeze(1) * layer_freqs)

            rotated2 = z * layer_phases.conj().unsqueeze(0)
            cs2 = rotated2.cumsum(dim=1)
            W = self.window if self.window is not None else T_cur
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases.unsqueeze(0)
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.shared_glu(self.bn_layers[i](out))

        # Readout: magnitude → max pool → fc
        h_re, h_im = h.chunk(2, dim=-1)
        mag_out = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        mag_out = mag_out.max(dim=1).values
        return self.fc(mag_out)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumSpecCumsumModTied(nn.Module):
    """Frozen cumsum spectrogram front-end → [log_mag, re, im] → Linear → tied mod cumsum backend.
    Same as CumsumSpecCumsumTied but with data-dependent frequencies (ModTied backend)."""
    def __init__(self, n_freqs=40, n_embed=80, n_layers=8, window_l1=400,
                 hop_length=160, window=5, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.window = window
        self.hop = hop_length
        self.window_l1 = window_l1
        n_backend_freqs = n_embed // 2

        # Frozen cumsum front-end (mel-scale init)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        angular_freqs = 2 * math.pi * center_freqs / SAMPLE_RATE
        self.register_buffer('log_freqs', torch.log(angular_freqs))

        # [log_mag, re, im] → n_embed
        self.embed = nn.Linear(3 * n_freqs, n_embed)
        self.spec_aug = SpecAugment()

        # Tied mod cumsum backend (same as MelCumsumModTied)
        self.shared_proj = nn.Linear(n_embed, 3 * n_backend_freqs)
        self.freq_ln = nn.LayerNorm(n_backend_freqs)
        self.shared_glu = GLU(n_embed, dropout)
        self.bn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.bn_layers.append(TransposedBN(n_embed))

        self.fc = nn.Linear(n_backend_freqs, num_classes)

    def forward(self, x):
        B_batch, T = x.shape

        # Frozen cumsum spectrogram
        freqs = self.log_freqs.exp()
        t_idx = torch.arange(T, device=x.device, dtype=freqs.dtype)
        phases = torch.exp(1j * t_idx.unsqueeze(1) * freqs)
        x_complex = x.to(torch.complex64).unsqueeze(-1)
        rotated = x_complex * phases.conj().unsqueeze(0)
        cs = rotated.cumsum(dim=1)
        cs_shifted = F.pad(cs[:, :-self.window_l1], (0, 0, self.window_l1, 0))
        d = cs - cs_shifted
        d = d * phases.unsqueeze(0)

        mag = d.real ** 2 + d.imag ** 2
        log_mag = (mag + 1e-8).log()
        features = torch.cat([log_mag, d.real, d.imag], dim=-1)
        features = features[:, ::self.hop]

        h = self.embed(features)
        if self.training:
            h = self.spec_aug(h.transpose(1, 2)).transpose(1, 2)

        # Tied mod cumsum backend
        n_backend_freqs = h.shape[-1] // 2
        for i in range(self.n_layers):
            T_cur = h.shape[1]
            proj = self.shared_proj(h)
            z_re = proj[..., :n_backend_freqs]
            z_im = proj[..., n_backend_freqs:2*n_backend_freqs]
            inst_freqs = self.freq_ln(proj[..., 2*n_backend_freqs:])
            z = torch.complex(z_re, z_im)

            cum_phase = inst_freqs.cumsum(dim=1)
            layer_phases = torch.exp(1j * cum_phase)

            rotated2 = z * layer_phases.conj()
            cs2 = rotated2.cumsum(dim=1)
            W = self.window if self.window is not None else T_cur
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.shared_glu(self.bn_layers[i](out))

        # Readout: magnitude → max pool → fc
        h_re, h_im = h.chunk(2, dim=-1)
        mag_out = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        mag_out = mag_out.max(dim=1).values
        return self.fc(mag_out)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelCumsumResNet(nn.Module):
    """Mel front-end → multi-stage cumsum with progressive channel expansion + stride-2 downsampling.
    Mirrors MelCNNMaxPool's architecture (16→24→32→48 channels, stride-2 between stages)
    but replaces CNN ResBlocks with windowed cumsum + GLU layers."""
    def __init__(self, channels=None, layers_per_stage=None, window=5,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        if channels is None:
            channels = [16, 24, 32, 48]
        if layers_per_stage is None:
            layers_per_stage = [2, 2, 2, 1]
        assert len(channels) == len(layers_per_stage)
        self.channels = channels
        self.layers_per_stage = layers_per_stage
        self.window = window
        self.n_stages = len(channels)

        # Mel front-end
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160,
            n_mels=40, power=2.0,
        )
        self.spec_aug = SpecAugment()
        self.embed = nn.Linear(40, channels[0])

        # Build stages
        self.stage_projs = nn.ModuleList()
        self.stage_freqs = nn.ParameterList()
        self.stage_bns = nn.ModuleList()
        self.stage_glus = nn.ModuleList()
        self.stage_expands = nn.ModuleList()

        for i, (ch, n_layers) in enumerate(zip(channels, layers_per_stage)):
            projs = nn.ModuleList([nn.Linear(ch, ch) for _ in range(n_layers)])
            freqs = nn.ParameterList([
                nn.Parameter(torch.linspace(0.1, math.pi * 0.9, ch // 2))
                for _ in range(n_layers)
            ])
            bns = nn.ModuleList([TransposedBN(ch) for _ in range(n_layers)])
            glus = nn.ModuleList([GLU(ch, dropout) for _ in range(n_layers)])

            self.stage_projs.append(projs)
            self.stage_freqs.append(freqs)
            self.stage_bns.append(bns)
            self.stage_glus.append(glus)

            # Expand to next channel width (except last stage)
            if i < len(channels) - 1:
                self.stage_expands.append(nn.Linear(ch, channels[i + 1]))
            else:
                self.stage_expands.append(nn.Identity())  # placeholder

        self.fc = nn.Linear(channels[-1] // 2, num_classes)

    def forward(self, x):
        # x: (B, 16000)
        x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = x.squeeze(1)
        x = (x + 1e-8).log()
        if self.training:
            x = self.spec_aug(x)
        x = x.transpose(1, 2)  # (B, T, 40)
        h = self.embed(x)      # (B, T, channels[0])

        for i in range(self.n_stages):
            ch = self.channels[i]
            n_freqs = ch // 2

            # Cumsum layers at current width
            for j in range(self.layers_per_stage[i]):
                T_cur = h.shape[1]
                proj = self.stage_projs[i][j](h)
                z_re, z_im = proj.chunk(2, dim=-1)
                z = torch.complex(z_re, z_im)

                layer_freqs = self.stage_freqs[i][j]
                t_idx = torch.arange(T_cur, device=h.device, dtype=layer_freqs.dtype)
                layer_phases = torch.exp(1j * t_idx.unsqueeze(1) * layer_freqs)

                rotated = z * layer_phases.conj().unsqueeze(0)
                cs = rotated.cumsum(dim=1)
                W = self.window if self.window is not None else T_cur
                cs_shifted = F.pad(cs[:, :-W], (0, 0, W, 0))
                d = cs - cs_shifted
                d = d * layer_phases.unsqueeze(0)
                out = torch.cat([d.real, d.imag], dim=-1)
                h = h + self.stage_glus[i][j](self.stage_bns[i][j](out))

            # Downsample + expand (except last stage)
            if i < self.n_stages - 1:
                h = h[:, ::2, :]  # stride-2 downsample
                h = self.stage_expands[i](h)

        # Readout: magnitude → global max pool → fc
        h_re, h_im = h.chunk(2, dim=-1)
        mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        mag = mag.max(dim=1).values
        return self.fc(mag)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConvCumsumV2(nn.Module):
    """Conv1d sin+cos front-end → cumsum layers with MaxPool(2).
    Layer 1: Conv1d(1→80, k=window, stride=160) with Hann-tapered mel sin+cos init → T=100
    Layers 2+: windowed cumsum + GLU + BN + residual + MaxPool(2), fixed learned frequencies."""
    def __init__(self, n_freqs=40, window=400, n_layers=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_layers = n_layers
        dim = 2 * n_freqs  # 80

        # Layer 1: Conv1d filterbank with sin+cos mel init
        self.filterbank = nn.Conv1d(1, dim, kernel_size=window, stride=160, bias=False)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        t = torch.arange(window, dtype=torch.float32) / SAMPLE_RATE
        hann = torch.hann_window(window)
        with torch.no_grad():
            for i in range(n_freqs):
                self.filterbank.weight.data[2*i, 0] = hann * torch.sin(
                    2 * math.pi * center_freqs[i] * t)
                self.filterbank.weight.data[2*i+1, 0] = hann * torch.cos(
                    2 * math.pi * center_freqs[i] * t)
        self.bn1 = TransposedBN(dim)
        self.glu1 = GLU(dim, dropout)

        # Layers 2+: cumsum with fixed learned frequencies
        self.proj_layers = nn.ModuleList()
        self.freq_params = nn.ParameterList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.proj_layers.append(nn.Linear(dim, dim))
            self.freq_params.append(nn.Parameter(
                torch.linspace(0.1, math.pi * 0.9, n_freqs)))
            self.bn_layers.append(TransposedBN(dim))
            self.glu_layers.append(GLU(dim, dropout))

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B_batch = x.shape[0]

        # === Layer 1: Conv1d filterbank → T=100 ===
        h = self.filterbank(x.unsqueeze(1))  # (B, 80, T=100)
        h = h.transpose(1, 2)  # (B, T=100, 80)
        h = h + self.glu1(self.bn1(h))

        # === Layers 2+: cumsum + GLU + stride 2 downsample ===
        for i in range(self.n_layers - 1):
            T_cur = h.shape[1]
            proj = self.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            layer_freqs = self.freq_params[i]
            t2_idx = torch.arange(T_cur, device=x.device, dtype=layer_freqs.dtype)
            layer_phases = torch.exp(1j * t2_idx.unsqueeze(1) * layer_freqs)

            rotated2 = z * layer_phases.conj().unsqueeze(0)
            cs2 = rotated2.cumsum(dim=1)
            W = T_cur
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases.unsqueeze(0)
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.glu_layers[i](self.bn_layers[i](out))
            h = F.avg_pool1d(h.transpose(1, 2), 2).transpose(1, 2)

        return self.fc(h.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConvCumsumModV2(nn.Module):
    """Conv1d sin+cos front-end → cumsum layers with data-dependent frequencies + MaxPool(2).
    Layer 1: Conv1d(1→80, k=window, stride=160) with Hann-tapered mel sin+cos init → T=100
    Layers 2+: data-dependent frequencies via Linear→ReLU→Linear→LayerNorm."""
    def __init__(self, n_freqs=40, window=400, n_layers=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_layers = n_layers
        dim = 2 * n_freqs

        # Layer 1: Conv1d filterbank with sin+cos mel init
        self.filterbank = nn.Conv1d(1, dim, kernel_size=window, stride=160, bias=False)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + SAMPLE_RATE / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_freqs + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        center_freqs = hz_points[1:-1]
        t = torch.arange(window, dtype=torch.float32) / SAMPLE_RATE
        hann = torch.hann_window(window)
        with torch.no_grad():
            for i in range(n_freqs):
                self.filterbank.weight.data[2*i, 0] = hann * torch.sin(
                    2 * math.pi * center_freqs[i] * t)
                self.filterbank.weight.data[2*i+1, 0] = hann * torch.cos(
                    2 * math.pi * center_freqs[i] * t)
        self.bn1 = TransposedBN(dim)
        self.glu1 = GLU(dim, dropout)

        # Layers 2+: data-dependent frequencies
        self.proj_layers = nn.ModuleList()
        self.freq_nets = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.glu_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.proj_layers.append(nn.Linear(dim, dim))
            self.freq_nets.append(nn.Sequential(
                nn.Linear(dim, n_freqs),
                nn.ReLU(),
                nn.Linear(n_freqs, n_freqs),
                nn.LayerNorm(n_freqs),
            ))
            self.bn_layers.append(TransposedBN(dim))
            self.glu_layers.append(GLU(dim, dropout))

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B_batch = x.shape[0]

        # === Layer 1: Conv1d filterbank → T=100 ===
        h = self.filterbank(x.unsqueeze(1))  # (B, 80, T=100)
        h = h.transpose(1, 2)  # (B, T=100, 80)
        h = h + self.glu1(self.bn1(h))

        # === Layers 2+: data-dependent cumsum + stride 2 downsample ===
        for i in range(self.n_layers - 1):
            T_cur = h.shape[1]
            proj = self.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            inst_freqs = self.freq_nets[i](h)
            cum_phase = inst_freqs.cumsum(dim=1)
            layer_phases = torch.exp(1j * cum_phase)

            rotated2 = z * layer_phases.conj()
            cs2 = rotated2.cumsum(dim=1)
            W = T_cur
            cs2_shifted = F.pad(cs2[:, :-W], (0, 0, W, 0))
            d2 = cs2 - cs2_shifted
            d2 = d2 * layer_phases
            out = torch.cat([d2.real, d2.imag], dim=-1)
            h = h + self.glu_layers[i](self.bn_layers[i](out))
            h = F.avg_pool1d(h.transpose(1, 2), 2).transpose(1, 2)

        return self.fc(h.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StridedWindowS5LayerBidir(nn.Module):
    """Bidirectional windowed cumsum S5 layer with stride.
    First layer: operates on full T=16000, W=400, stride=160 → outputs T=100.
    Like an STFT but learned. Subsequent layers can use stride=1."""
    def __init__(self, d_model, window=400, stride=160, dropout=0.0):
        super().__init__()
        n = d_model // 2
        H = d_model
        self.window = window
        self.stride = stride
        # Same params as CumsumS5LayerBidir
        self.Lambda_im_f = nn.Parameter(math.pi * torch.arange(n).float())
        self.Lambda_im_b = nn.Parameter(math.pi * torch.arange(n).float())
        self.log_dt_f = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt_b = nn.Parameter(torch.tensor(math.log(0.01)))
        self.B_re_f = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_f = nn.Parameter(torch.zeros(n, H))
        self.C_re_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.B_re_b = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_b = nn.Parameter(torch.zeros(n, H))
        self.C_re_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.D = nn.Parameter(torch.ones(H))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B_batch, T, H = x.shape
        # Forward
        dt_f = self.log_dt_f.exp()
        Lambda_f = 1j * self.Lambda_im_f
        angles_f = self.Lambda_im_f * dt_f
        B_f = torch.complex(self.B_re_f, self.B_im_f)
        C_f = torch.complex(self.C_re_f, self.C_im_f)
        Ldt_f = Lambda_f * dt_f
        safe_L_f = torch.where(Lambda_f.abs() < 1e-6, torch.ones_like(Lambda_f), Lambda_f)
        B_bar_scale_f = torch.where(Lambda_f.abs() < 1e-6,
                                     dt_f * torch.ones_like(Lambda_f),
                                     (torch.exp(Ldt_f) - 1.0) / safe_L_f)
        B_bar_f = B_bar_scale_f.unsqueeze(-1) * B_f
        Bu_f = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_f)
        h_f = complex_cumsum_scan(Bu_f, angles_f, reverse=False, window=self.window)
        y_f = 2.0 * torch.einsum('btn,hn->bth', h_f, C_f.conj()).real
        # Backward
        dt_b = self.log_dt_b.exp()
        Lambda_b = 1j * self.Lambda_im_b
        angles_b = self.Lambda_im_b * dt_b
        B_b = torch.complex(self.B_re_b, self.B_im_b)
        C_b = torch.complex(self.C_re_b, self.C_im_b)
        Ldt_b = Lambda_b * dt_b
        safe_L_b = torch.where(Lambda_b.abs() < 1e-6, torch.ones_like(Lambda_b), Lambda_b)
        B_bar_scale_b = torch.where(Lambda_b.abs() < 1e-6,
                                     dt_b * torch.ones_like(Lambda_b),
                                     (torch.exp(Ldt_b) - 1.0) / safe_L_b)
        B_bar_b = B_bar_scale_b.unsqueeze(-1) * B_b
        Bu_b = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_b)
        h_b = complex_cumsum_scan(Bu_b, angles_b, reverse=True, window=self.window)
        y_b = 2.0 * torch.einsum('btn,hn->bth', h_b, C_b.conj()).real

        y = y_f + y_b + x * self.D
        # Stride: subsample to reduce temporal resolution
        if self.stride > 1:
            y = y[:, ::self.stride]
        return self.dropout(y)


class StridedWindowS5Block(nn.Module):
    """Pre-norm block with strided windowed S5 layer."""
    def __init__(self, d_model, window=400, stride=160, dropout=0.0):
        super().__init__()
        self.stride = stride
        self.bn1 = TransposedBN(d_model)
        self.ssm = StridedWindowS5LayerBidir(d_model, window, stride, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        # Residual must account for stride
        res = x[:, ::self.stride] if self.stride > 1 else x
        x = res + self.ssm(self.bn1(x))
        x = x + self.glu(self.bn2(x))
        return x


class StridedWindowS5(nn.Module):
    """WindowS5 with strided first layer — no input pooling.
    First layer: W=400, stride=160 on raw T=16000 → T=100.
    Remaining layers: standard WindowS5 on T=100."""
    def __init__(self, d_model=64, n_layers=6, window=400, stride=160,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        # First block: strided — reduces T=16000 → T=100
        blocks = [StridedWindowS5Block(d_model, window=window, stride=stride, dropout=dropout)]
        # Remaining blocks: no stride, small window on T=100
        for _ in range(n_layers - 1):
            blocks.append(CumsumS5Block(d_model, window=None, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 16000) — NO pooling
        x = self.input_proj(x.unsqueeze(-1))  # (B, 16000, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))  # avg pool over ~100 frames

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumS5Block(nn.Module):
    """Pre-norm block with cumsum S5 layer."""
    def __init__(self, d_model, window=None, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.ssm = CumsumS5LayerBidir(d_model, window, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        x = x + self.ssm(self.bn1(x))
        x = x + self.glu(self.bn2(x))
        return x


class CumsumS5(nn.Module):
    """S5 with pure rotation (no decay) — uses fast cumsum. Full B/C/D/B_bar."""
    def __init__(self, d_model=64, n_layers=6, pool=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([CumsumS5Block(d_model, window=None, dropout=dropout)
                                     for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WindowS5(nn.Module):
    """S5 with pure rotation + finite window — uses fast windowed cumsum. Full B/C/D/B_bar."""
    def __init__(self, d_model=64, n_layers=6, pool=4, window=80,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([CumsumS5Block(d_model, window=window, dropout=dropout)
                                     for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumS5LayerBidirInput(nn.Module):
    """Bidirectional pure-rotation S5 layer with input-dependent angles.
    Complex B/C, D skip. Fixed B_bar from reference frequencies (Λ_im, dt)."""
    def __init__(self, d_model, window=None, dropout=0.0):
        super().__init__()
        n = d_model // 2
        H = d_model
        self.n = n
        self.window = window
        # Reference eigenvalues for B_bar (fixed, not used for scan)
        self.Lambda_im_f = nn.Parameter(math.pi * torch.arange(n).float())
        self.Lambda_im_b = nn.Parameter(math.pi * torch.arange(n).float())
        self.log_dt_f = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt_b = nn.Parameter(torch.tensor(math.log(0.01)))
        # Angle projector (for scan phases, shared for forward/backward)
        self.gate_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n),
        )
        self.angle_ln = nn.LayerNorm(n)
        with torch.no_grad():
            dt = 0.01
            self.gate_projector[2].bias.copy_(math.pi * torch.arange(n).float() * dt)
        # Complex B (n, H), C (H, n) — forward
        self.B_re_f = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_f = nn.Parameter(torch.zeros(n, H))
        self.C_re_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # Complex B (n, H), C (H, n) — backward
        self.B_re_b = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_b = nn.Parameter(torch.zeros(n, H))
        self.C_re_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # D feedthrough
        self.D = nn.Parameter(torch.ones(H))
        self.dropout = nn.Dropout(dropout)

    def _b_bar(self, Lambda_im, log_dt, B_re, B_im):
        """Compute fixed B_bar from reference eigenvalues."""
        dt = log_dt.exp()
        Lambda = 1j * Lambda_im
        B = torch.complex(B_re, B_im)
        Ldt = Lambda * dt
        safe_L = torch.where(Lambda.abs() < 1e-6, torch.ones_like(Lambda), Lambda)
        scale = torch.where(Lambda.abs() < 1e-6,
                            dt * torch.ones_like(Lambda),
                            (torch.exp(Ldt) - 1.0) / safe_L)
        return scale.unsqueeze(-1) * B

    def forward(self, x):
        # Input-dependent angles → cumulative phases
        raw_angles = self.angle_ln(self.gate_projector(x))  # (B, T, n)
        cum_f = torch.cumsum(raw_angles, dim=1)
        phases_f = torch.exp(1j * cum_f.to(torch.float32))  # (B, T, n)
        cum_b = torch.cumsum(raw_angles.flip(1), dim=1).flip(1)
        phases_b = torch.exp(1j * cum_b.to(torch.float32))
        # Forward — fixed B_bar from reference frequencies
        B_bar_f = self._b_bar(self.Lambda_im_f, self.log_dt_f, self.B_re_f, self.B_im_f)
        C_f = torch.complex(self.C_re_f, self.C_im_f)
        Bu_f = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_f)
        h_f = complex_cumsum_scan(Bu_f, None, reverse=False, window=self.window, phases=phases_f)
        y_f = 2.0 * torch.einsum('btn,hn->bth', h_f, C_f.conj()).real
        # Backward — fixed B_bar from reference frequencies
        B_bar_b = self._b_bar(self.Lambda_im_b, self.log_dt_b, self.B_re_b, self.B_im_b)
        C_b = torch.complex(self.C_re_b, self.C_im_b)
        Bu_b = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_b)
        h_b = complex_cumsum_scan(Bu_b, None, reverse=True, window=self.window, phases=phases_b)
        y_b = 2.0 * torch.einsum('btn,hn->bth', h_b, C_b.conj()).real

        return self.dropout(y_f + y_b + x * self.D)


class CumsumS5LayerBidirMod(nn.Module):
    """Bidirectional pure-rotation S5 layer with modulated angles.
    Instead of predicting angles from scratch, modulates fixed S4D-Lin base
    angles: angle = base_angle * (1 + proj(x)). No LayerNorm on angles."""
    def __init__(self, d_model, window=None, dropout=0.0):
        super().__init__()
        n = d_model // 2
        H = d_model
        self.n = n
        self.window = window
        # Base eigenvalues (S4D-Lin init) — used for both B_bar and base angles
        self.Lambda_im_f = nn.Parameter(math.pi * torch.arange(n).float())
        self.Lambda_im_b = nn.Parameter(math.pi * torch.arange(n).float())
        self.log_dt_f = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt_b = nn.Parameter(torch.tensor(math.log(0.01)))
        # Modulation projector: outputs small perturbation per dimension
        self.mod_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n),
        )
        # Init bias to zero so initial modulation is (1+0)=1
        with torch.no_grad():
            self.mod_projector[2].bias.zero_()
            self.mod_projector[2].weight.zero_()
        # Complex B (n, H), C (H, n) — forward
        self.B_re_f = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_f = nn.Parameter(torch.zeros(n, H))
        self.C_re_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_f = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # Complex B (n, H), C (H, n) — backward
        self.B_re_b = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im_b = nn.Parameter(torch.zeros(n, H))
        self.C_re_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im_b = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        # D feedthrough
        self.D = nn.Parameter(torch.ones(H))
        self.dropout = nn.Dropout(dropout)

    def _b_bar(self, Lambda_im, log_dt, B_re, B_im):
        """Compute fixed B_bar from reference eigenvalues."""
        dt = log_dt.exp()
        Lambda = 1j * Lambda_im
        B = torch.complex(B_re, B_im)
        Ldt = Lambda * dt
        safe_L = torch.where(Lambda.abs() < 1e-6, torch.ones_like(Lambda), Lambda)
        scale = torch.where(Lambda.abs() < 1e-6,
                            dt * torch.ones_like(Lambda),
                            (torch.exp(Ldt) - 1.0) / safe_L)
        return scale.unsqueeze(-1) * B

    def forward(self, x):
        dt_f = self.log_dt_f.exp()
        dt_b = self.log_dt_b.exp()
        # Base angles from S4D-Lin eigenvalues
        base_angles_f = self.Lambda_im_f * dt_f  # (n,)
        base_angles_b = self.Lambda_im_b * dt_b  # (n,)
        # Input-dependent modulation: angle = base * (1 + mod(x))
        mod = self.mod_projector(x)  # (B, T, n)
        angles_f = base_angles_f * (1.0 + mod)  # (B, T, n)
        angles_b = base_angles_b * (1.0 + mod)
        # Cumulative phases
        cum_f = torch.cumsum(angles_f, dim=1)
        phases_f = torch.exp(1j * cum_f.to(torch.float32))
        cum_b = torch.cumsum(angles_b.flip(1), dim=1).flip(1)
        phases_b = torch.exp(1j * cum_b.to(torch.float32))
        # Forward — fixed B_bar from reference frequencies
        B_bar_f = self._b_bar(self.Lambda_im_f, self.log_dt_f, self.B_re_f, self.B_im_f)
        C_f = torch.complex(self.C_re_f, self.C_im_f)
        Bu_f = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_f)
        h_f = complex_cumsum_scan(Bu_f, None, reverse=False, window=self.window, phases=phases_f)
        y_f = 2.0 * torch.einsum('btn,hn->bth', h_f, C_f.conj()).real
        # Backward — fixed B_bar from reference frequencies
        B_bar_b = self._b_bar(self.Lambda_im_b, self.log_dt_b, self.B_re_b, self.B_im_b)
        C_b = torch.complex(self.C_re_b, self.C_im_b)
        Bu_b = torch.einsum('bth,nh->btn', x.to(torch.complex64), B_bar_b)
        h_b = complex_cumsum_scan(Bu_b, None, reverse=True, window=self.window, phases=phases_b)
        y_b = 2.0 * torch.einsum('btn,hn->bth', h_b, C_b.conj()).real

        return self.dropout(y_f + y_b + x * self.D)


class CumsumS5BlockMod(nn.Module):
    """Pre-norm block with modulated cumsum S5 layer."""
    def __init__(self, d_model, window=None, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.ssm = CumsumS5LayerBidirMod(d_model, window, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        x = x + self.ssm(self.bn1(x))
        x = x + self.glu(self.bn2(x))
        return x


class CumsumS5Mod(nn.Module):
    """S5 with pure rotation, modulated angles — uses fast cumsum. B/C/D, B_bar."""
    def __init__(self, d_model=64, n_layers=6, pool=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([CumsumS5BlockMod(d_model, window=None, dropout=dropout)
                                     for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WindowS5Mod(nn.Module):
    """S5 with pure rotation + finite window, modulated angles — fast cumsum. B/C/D, B_bar."""
    def __init__(self, d_model=64, n_layers=6, pool=4, window=80,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([CumsumS5BlockMod(d_model, window=window, dropout=dropout)
                                     for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CumsumS5BlockInput(nn.Module):
    """Pre-norm block with input-dependent cumsum S5 layer."""
    def __init__(self, d_model, window=None, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.ssm = CumsumS5LayerBidirInput(d_model, window, dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        x = x + self.ssm(self.bn1(x))
        x = x + self.glu(self.bn2(x))
        return x


class CumsumS5Input(nn.Module):
    """S5 with pure rotation, input-dependent angles — uses fast cumsum. B/C/D, no B_bar."""
    def __init__(self, d_model=64, n_layers=6, pool=4, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([CumsumS5BlockInput(d_model, window=None, dropout=dropout)
                                     for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WindowS5Input(nn.Module):
    """S5 with pure rotation + finite window, input-dependent angles — fast cumsum. B/C/D, no B_bar."""
    def __init__(self, d_model=64, n_layers=6, pool=4, window=80,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([CumsumS5BlockInput(d_model, window=window, dropout=dropout)
                                     for _ in range(n_layers)])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Parallel scan (Blelloch-style) ─────────────────────────────────────────
# For the linear recurrence h[t] = gates[t]*h[t-1] + values[t].
# Uses the doubling technique: O(L log L) work in O(log L) sequential steps,
# each step being a fully parallel tensor operation. No Python for-loops over L.

def _parallel_scan_pytorch(gates, values):
    """Fallback parallel scan in pure PyTorch (log2 L kernel launches)."""
    a = gates
    b = values
    L = a.shape[1]
    for step in range(int(math.ceil(math.log2(max(L, 2))))):
        stride = 1 << step
        if stride >= L:
            break
        padding = (0, 0, stride, 0)
        a_shift = F.pad(a[:, :-stride], padding, value=1.0)
        b_shift = F.pad(b[:, :-stride], padding, value=0.0)
        a, b = a * a_shift, a * b_shift + b
    return b


# ─── Triton-fused parallel scan ────────────────────────────────────────────
# Single kernel launch, fully parallel across (batch, feature) pairs.
# Complex arithmetic done as real/imag pairs.

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _scan_fwd_kernel(
        gates_r_ptr, gates_i_ptr, vals_r_ptr, vals_i_ptr,
        out_r_ptr, out_i_ptr,
        B, L, D,
        stride_b, stride_l, stride_d,
        BLOCK_D: tl.constexpr,
    ):
        """Forward scan: h[t] = gate[t]*h[t-1] + val[t] (complex)."""
        bid = tl.program_id(0)  # batch index
        did = tl.program_id(1)  # feature block index
        d_offs = did * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        h_r = tl.zeros([BLOCK_D], dtype=tl.float32)
        h_i = tl.zeros([BLOCK_D], dtype=tl.float32)

        for t in range(L):
            off = bid * stride_b + t * stride_l + d_offs * stride_d
            a_r = tl.load(gates_r_ptr + off, mask=d_mask, other=1.0)
            a_i = tl.load(gates_i_ptr + off, mask=d_mask, other=0.0)
            b_r = tl.load(vals_r_ptr + off, mask=d_mask, other=0.0)
            b_i = tl.load(vals_i_ptr + off, mask=d_mask, other=0.0)
            # h = a * h + b  (complex multiply)
            new_h_r = a_r * h_r - a_i * h_i + b_r
            new_h_i = a_r * h_i + a_i * h_r + b_i
            h_r = new_h_r
            h_i = new_h_i
            tl.store(out_r_ptr + off, h_r, mask=d_mask)
            tl.store(out_i_ptr + off, h_i, mask=d_mask)

    @triton.jit
    def _scan_bwd_kernel(
        gates_r_ptr, gates_i_ptr, out_r_ptr, out_i_ptr,
        dout_r_ptr, dout_i_ptr,
        dgates_r_ptr, dgates_i_ptr, dvals_r_ptr, dvals_i_ptr,
        B, L, D,
        stride_b, stride_l, stride_d,
        BLOCK_D: tl.constexpr,
    ):
        """Backward scan: propagate gradients in reverse."""
        bid = tl.program_id(0)
        did = tl.program_id(1)
        d_offs = did * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # dh accumulates gradient flowing back through time
        dh_r = tl.zeros([BLOCK_D], dtype=tl.float32)
        dh_i = tl.zeros([BLOCK_D], dtype=tl.float32)

        for t in range(L - 1, -1, -1):
            off = bid * stride_b + t * stride_l + d_offs * stride_d
            # dout for this timestep
            do_r = tl.load(dout_r_ptr + off, mask=d_mask, other=0.0)
            do_i = tl.load(dout_i_ptr + off, mask=d_mask, other=0.0)
            dh_r = dh_r + do_r
            dh_i = dh_i + do_i

            # d_val = dh (complex)
            tl.store(dvals_r_ptr + off, dh_r, mask=d_mask)
            tl.store(dvals_i_ptr + off, dh_i, mask=d_mask)

            # d_gate = dh * conj(h[t-1])... actually d_gate = dh * h[t-1]
            # For complex: d(a*h) w.r.t. a = h (Wirtinger derivative)
            # d_gate_r = dh_r * h_prev_r + dh_i * h_prev_i
            # d_gate_i = -dh_r * h_prev_i + dh_i * h_prev_r
            if t > 0:
                prev_off = bid * stride_b + (t - 1) * stride_l + d_offs * stride_d
                hp_r = tl.load(out_r_ptr + prev_off, mask=d_mask, other=0.0)
                hp_i = tl.load(out_i_ptr + prev_off, mask=d_mask, other=0.0)
            else:
                hp_r = tl.zeros([BLOCK_D], dtype=tl.float32)
                hp_i = tl.zeros([BLOCK_D], dtype=tl.float32)
            dg_r = dh_r * hp_r + dh_i * hp_i
            dg_i = -dh_r * hp_i + dh_i * hp_r
            tl.store(dgates_r_ptr + off, dg_r, mask=d_mask)
            tl.store(dgates_i_ptr + off, dg_i, mask=d_mask)

            # Propagate: dh_prev = conj(gate[t]) * dh
            a_r = tl.load(gates_r_ptr + off, mask=d_mask, other=1.0)
            a_i = tl.load(gates_i_ptr + off, mask=d_mask, other=0.0)
            # conj(a) * dh = (a_r - a_i*j)(dh_r + dh_i*j)
            new_dh_r = a_r * dh_r + a_i * dh_i
            new_dh_i = a_r * dh_i - a_i * dh_r
            dh_r = new_dh_r
            dh_i = new_dh_i

    class _TritonScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gates_r, gates_i, vals_r, vals_i):
            B, L, D = gates_r.shape
            out_r = torch.empty_like(gates_r)
            out_i = torch.empty_like(gates_i)
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _scan_fwd_kernel[grid](
                gates_r, gates_i, vals_r, vals_i, out_r, out_i,
                B, L, D,
                gates_r.stride(0), gates_r.stride(1), gates_r.stride(2),
                BLOCK_D=BLOCK_D,
            )
            ctx.save_for_backward(gates_r, gates_i, out_r, out_i)
            ctx.shape = (B, L, D)
            return out_r, out_i

        @staticmethod
        def backward(ctx, dout_r, dout_i):
            gates_r, gates_i, out_r, out_i = ctx.saved_tensors
            B, L, D = ctx.shape
            dgates_r = torch.empty_like(gates_r)
            dgates_i = torch.empty_like(gates_i)
            dvals_r = torch.empty_like(gates_r)
            dvals_i = torch.empty_like(gates_i)
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _scan_bwd_kernel[grid](
                gates_r, gates_i, out_r, out_i,
                dout_r.contiguous(), dout_i.contiguous(),
                dgates_r, dgates_i, dvals_r, dvals_i,
                B, L, D,
                gates_r.stride(0), gates_r.stride(1), gates_r.stride(2),
                BLOCK_D=BLOCK_D,
            )
            return dgates_r, dgates_i, dvals_r, dvals_i

    def _triton_scan_complex(gates, values):
        """Fused Triton scan for complex gates/values with autograd."""
        g_r, g_i = gates.real.contiguous(), gates.imag.contiguous()
        v_r, v_i = values.real.contiguous(), values.imag.contiguous()
        h_r, h_i = _TritonScanFn.apply(g_r, g_i, v_r, v_i)
        return torch.complex(h_r, h_i)

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def parallel_scan(gates, values):
    """Parallel prefix scan for h[t] = gates[t]*h[t-1] + values[t].

    Uses a fused Triton kernel (single launch, full autograd) when available,
    otherwise falls back to PyTorch implementation (log2 L kernel launches).
    gates, values: (B, L, D) — real or complex. Returns h: (B, L, D).
    """
    if _HAS_TRITON and gates.is_complex() and gates.is_cuda:
        return _triton_scan_complex(gates, values)
    return _parallel_scan_pytorch(gates, values)


# Keep old name as alias
chunked_scan = parallel_scan


# ─── S5: Simplified State Spaces ────────────────────────────────────────────
# Ref: Smith et al., "Simplified State Space Layers for Sequence Modeling"
#      (ICLR 2023, arXiv:2208.04933)
# Init: Gu et al., "On the Parameterization and Initialization of Diagonal
#       State Space Models" (NeurIPS 2022, arXiv:2206.11893) — S4D-Lin

class S5SSM(nn.Module):
    """Single-direction diagonal complex SSM (no skip connection)."""
    def __init__(self, H, N=64, dt_init=0.01):
        super().__init__()
        n = N // 2
        self.Lambda_re = nn.Parameter(-0.5 * torch.ones(n))
        self.Lambda_im = nn.Parameter(math.pi * torch.arange(n).float())
        self.B_re = nn.Parameter(torch.randn(n, H) / math.sqrt(H))
        self.B_im = nn.Parameter(torch.zeros(n, H))
        self.C_re = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.C_im = nn.Parameter(torch.randn(H, n) / math.sqrt(n))
        self.log_dt = nn.Parameter(torch.tensor(math.log(dt_init)))

    def forward(self, u):
        B_batch, L, H = u.shape
        Lambda = torch.complex(self.Lambda_re.clamp(max=-1e-4), self.Lambda_im)
        Bmat = torch.complex(self.B_re, self.B_im)
        Cmat = torch.complex(self.C_re, self.C_im)
        dt = self.log_dt.exp()
        Lambda_bar = torch.exp(Lambda * dt)
        B_bar = ((Lambda_bar - 1.0) / Lambda).unsqueeze(-1) * Bmat
        Bu = torch.einsum('blh,nh->bln', u.to(torch.complex64), B_bar)
        gates = Lambda_bar.unsqueeze(0).unsqueeze(0).expand(B_batch, L, -1)
        x_all = chunked_scan(gates, Bu)
        return 2.0 * torch.einsum('bln,hn->blh', x_all, Cmat.conj()).real


class S5Layer(nn.Module):
    """Bidirectional S5: forward + backward SSMs with separate parameters."""
    def __init__(self, H, N=64, bidirectional=True):
        super().__init__()
        self.ssm_f = S5SSM(H, N)
        self.ssm_b = S5SSM(H, N) if bidirectional else None
        self.D = nn.Parameter(torch.ones(H))

    def forward(self, u):
        y = self.ssm_f(u)
        if self.ssm_b is not None:
            y = y + self.ssm_b(u.flip(1)).flip(1)
        return y + u * self.D


class S5Block(nn.Module):
    """Pre-norm S5 block: BN → S5 → residual → BN → GLU → residual."""
    def __init__(self, d_model, state_dim=64, bidirectional=True, dropout=0.0):
        super().__init__()
        self.bn1 = TransposedBN(d_model)
        self.ssm = S5Layer(d_model, state_dim, bidirectional)
        self.drop1 = nn.Dropout(dropout)
        self.bn2 = TransposedBN(d_model)
        self.glu = GLU(d_model, dropout)

    def forward(self, x):
        x = x + self.drop1(self.ssm(self.bn1(x)))
        x = x + self.glu(self.bn2(x))
        return x


class S5Model(nn.Module):
    """S5 for speech commands, matching paper architecture:
    Linear(1→d) front-end, bidirectional, BatchNorm, GLU."""
    def __init__(self, d_model=96, n_layers=6, state_dim=64, pool=4,
                 bidirectional=True, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.pool_size = pool
        self.blocks = nn.ModuleList([
            S5Block(d_model, state_dim, bidirectional, dropout) for _ in range(n_layers)
        ])
        self.bn_out = TransposedBN(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))  # (B, 16000, d_model)
        if self.pool_size > 1:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)  # (B, T/pool, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.bn_out(x)
        return self.fc(x.mean(dim=1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# ─── Training ───────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(waveforms)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        logits = model(waveforms)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return correct / total, np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels):
    """Compute per-class accuracy, macro F1, and confusion matrix."""
    per_class_acc = {}
    per_class_f1 = {}
    for i, name in enumerate(LABELS):
        mask = labels == i
        if mask.sum() == 0:
            per_class_acc[name] = float('nan')
            per_class_f1[name] = float('nan')
            continue
        tp = ((preds == i) & (labels == i)).sum()
        fp = ((preds == i) & (labels != i)).sum()
        fn = ((preds != i) & (labels == i)).sum()
        per_class_acc[name] = tp / mask.sum() if mask.sum() > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[name] = f1

    macro_f1 = np.nanmean(list(per_class_f1.values()))

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1

    return per_class_acc, per_class_f1, macro_f1, cm


def print_confusion_matrix(cm):
    """Print confusion matrix with labels."""
    short = [l[:4] for l in LABELS]
    header = "     " + " ".join(f"{s:>5s}" for s in short)
    print(header)
    for i, name in enumerate(LABELS):
        row = f"{name[:4]:>4s} " + " ".join(f"{cm[i, j]:5d}" for j in range(NUM_CLASSES))
        print(row)


def train_model(model, train_loader, val_loader, epochs, device, model_name="model", fixed_lr=None):
    if model_name in ('MelCNN', 'RawCNN', 'LearnedSpecCNN', 'LearnedSpecLinear', 'LearnedSpecMulti', 'LearnedSpecMulti4', 'LearnedSpecCNNMod', 'LearnedSpecCNNMod2', 'LearnedSpecCNNConv', 'LearnedSpecCNNConv2', 'FilterbankCNN', 'FilterbankMelInit', 'FilterbankSinCos', 'FilterbankSinCosMulti', 'FilterbankSinCosLinear', 'FilterbankSinCosCombined', 'FilterbankLinear', 'FilterbankMelInitLinear', 'MultiLayerV2', 'MultiLayerModV2', 'ConvCumsumV2', 'ConvCumsumModV2', 'MelCumsumFixed', 'MelCumsumMod', 'MelCumsumFixedW', 'MelCumsumModW', 'CumsumE2E', 'CumsumE2EMag', 'CumsumE2EMagMod', 'CumsumE2EMod', 'CumsumE2EMagS1', 'CumsumE2EMagModS1', 'MelMaxPool160', 'MelMaxPool80', 'MelMaxPool40', 'MelMultiPhase80', 'MelMultiPhase40', 'MelCumsumFixedMP2', 'MelCumsumFixedMP4', 'MelCumsumModMP2', 'MelCumsumModMP4', 'CumsumMagDeep', 'CumsumComplex', 'CumsumSingle', 'CumsumE2E_s320', 'CumsumE2E_s640', 'CumsumE2EMag_s320', 'CumsumE2EMag_s640'):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif model_name == 'S5':
        # S4/S5 convention: higher LR for non-SSM, lower for SSM params
        ssm_keys = ['Lambda', 'log_dt', 'B_re', 'B_im', 'C_re', 'C_im']
        ssm_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and any(k in n for k in ssm_keys)]
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(k in n for k in ssm_keys)]
        optimizer = torch.optim.AdamW([
            {'params': ssm_params, 'lr': 1e-3, 'weight_decay': 0.0},
            {'params': other_params, 'lr': 1e-2, 'weight_decay': 0.05},
        ])
    elif model_name in ('RotDecayFixed', 'RotDecayInput', 'RotS5Fixed', 'RotS5Input',
                        'CumsumS5', 'WindowS5', 'CumsumS5Input', 'WindowS5Input',
                        'CumsumS5Mod', 'WindowS5Mod', 'BlockDecayS5', 'BlockDecayS5V2',
                        'BlockDecayS5Overlap', 'BlockDecayS5Mod',
                        'StridedWindowS5'):
        # Same lr/wd as S5: SSM params get lr=1e-3/wd=0,
        # everything else gets lr=1e-2/wd=0.05
        ssm_keys = ['angles_f', 'angles_b', 'log_lambda_f', 'log_lambda_b',
                     'Lambda_re', 'Lambda_im', 'log_dt',
                     'angle_projector', 'angle_ln', 'gate_projector', 'mod_projector',
                     'B_re', 'B_im', 'C_re', 'C_im', '.D']
        ssm_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and any(k in n for k in ssm_keys)]
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(k in n for k in ssm_keys)]
        optimizer = torch.optim.AdamW([
            {'params': ssm_params, 'lr': 1e-3, 'weight_decay': 0.0},
            {'params': other_params, 'lr': 1e-2, 'weight_decay': 0.05},
        ])
    else:
        # Rotation models (no decay): Adam lr=1e-3, light weight decay
        lr = fixed_lr if fixed_lr is not None else 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    print(f"\n{'='*60}")
    print(f"Training {model_name} ({model.param_count():,} params)")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  lr={lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    return model


# ─── Perturbation evaluation ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_perturbation(model, dataset, device, stretch=1.0, distortion_alpha=0.0, batch_size=128):
    """Evaluate with a fixed time-stretch or distortion applied to all samples."""
    model.eval()
    correct = total = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    for waveforms, labels in loader:
        if stretch != 1.0:
            new_len = int(waveforms.shape[-1] * stretch)
            waveforms = F.interpolate(waveforms.unsqueeze(1), size=new_len, mode='linear', align_corners=False).squeeze(1)
            if waveforms.shape[-1] > NUM_SAMPLES:
                waveforms = waveforms[:, :NUM_SAMPLES]
            else:
                waveforms = F.pad(waveforms, (0, NUM_SAMPLES - waveforms.shape[-1]))
        if distortion_alpha > 0.0:
            peak = waveforms.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            waveforms = waveforms / peak
            waveforms = waveforms + distortion_alpha * waveforms.pow(3)
            waveforms = waveforms.clamp(-1.0, 1.0)
        waveforms, labels = waveforms.to(device), labels.to(device)
        preds = model(waveforms).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def evaluate_perturbation_grid(model, test_ds, device, model_name="model"):
    """Evaluate model across grids of time-stretch and distortion."""
    stretch_factors = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
    distortion_alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

    print(f"\n{'='*60}")
    print(f"Perturbation Grid: {model_name}")
    print(f"{'='*60}")

    print("\nTime-stretch:")
    for s in stretch_factors:
        acc = evaluate_with_perturbation(model, test_ds, device, stretch=s)
        print(f"  stretch={s:.2f}: {acc:.4f}")

    print("\nDistortion (x + alpha*x^3, clipped):")
    for a in distortion_alphas:
        acc = evaluate_with_perturbation(model, test_ds, device, distortion_alpha=a)
        print(f"  alpha={a:.1f}: {acc:.4f}")


@torch.no_grad()
def evaluate_split_stretch(model, dataset, device, max_stretches=None, batch_size=128):
    """Evaluate with split-stretch: each waveform is split in half, each half gets
    an independent random time-stretch, then reassembled to 16000 samples.
    This tests whether the model can handle mid-sequence tempo changes."""
    if max_stretches is None:
        max_stretches = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    for max_s in max_stretches:
        correct = total = 0
        rng = np.random.RandomState(123)
        for waveforms, labels in loader:
            B = waveforms.shape[0]
            half = NUM_SAMPLES // 2  # 8000
            out_waveforms = []
            for b in range(B):
                w = waveforms[b]  # (16000,)
                w1 = w[:half]
                w2 = w[half:]
                if max_s > 0:
                    s1 = rng.uniform(1.0 - max_s, 1.0 + max_s)
                    s2 = rng.uniform(1.0 - max_s, 1.0 + max_s)
                    new_len1 = int(half * s1)
                    w1 = F.interpolate(w1.view(1, 1, -1), size=new_len1, mode='linear', align_corners=False).view(-1)
                    new_len2 = int(half * s2)
                    w2 = F.interpolate(w2.view(1, 1, -1), size=new_len2, mode='linear', align_corners=False).view(-1)
                combined = torch.cat([w1, w2])
                # Pad or truncate to NUM_SAMPLES
                if combined.shape[-1] > NUM_SAMPLES:
                    combined = combined[:NUM_SAMPLES]
                else:
                    combined = F.pad(combined, (0, NUM_SAMPLES - combined.shape[-1]))
                out_waveforms.append(combined)
            waveforms = torch.stack(out_waveforms).to(device)
            labels = labels.to(device)
            preds = model(waveforms).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        print(f"  max_stretch={max_s:.2f}: {acc:.4f}")


@torch.no_grad()
def evaluate_dual_command(model, dataset, device, max_stretches=None, batch_size=128):
    """Evaluate on concatenated dual-command samples.
    Each sample is two waveforms (16000 each) with independent random time-stretch,
    concatenated to 32000 samples. Model backbone runs on full sequence,
    then maxpool over each half → two predictions.
    A sample is correct only if both halves are classified correctly.
    """
    if max_stretches is None:
        max_stretches = [0.0, 0.05, 0.1, 0.15, 0.2]

    # Collect all waveforms and labels from dataset
    all_waveforms = []
    all_labels = []
    for i in range(len(dataset)):
        w, l = dataset[i]
        all_waveforms.append(w)
        all_labels.append(l)

    n = len(all_waveforms)
    # Create fixed random pairing
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)

    for max_s in max_stretches:
        correct = total = 0
        rng_stretch = np.random.RandomState(123)
        # Process in batches
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            waveforms_cat = []
            labels1 = []
            labels2 = []
            for idx in range(batch_start, batch_end):
                w1 = all_waveforms[idx]
                w2 = all_waveforms[perm[idx]]
                # Each is full 16000 samples
                if w1.shape[-1] < NUM_SAMPLES:
                    w1 = F.pad(w1, (0, NUM_SAMPLES - w1.shape[-1]))
                else:
                    w1 = w1[:NUM_SAMPLES]
                if w2.shape[-1] < NUM_SAMPLES:
                    w2 = F.pad(w2, (0, NUM_SAMPLES - w2.shape[-1]))
                else:
                    w2 = w2[:NUM_SAMPLES]
                # Apply independent random time-stretch to each half
                if max_s > 0:
                    s1 = rng_stretch.uniform(1.0 - max_s, 1.0 + max_s)
                    s2 = rng_stretch.uniform(1.0 - max_s, 1.0 + max_s)
                    new_len1 = int(NUM_SAMPLES * s1)
                    w1 = F.interpolate(w1.view(1, 1, -1), size=new_len1, mode='linear', align_corners=False).view(-1)
                    if w1.shape[-1] > NUM_SAMPLES:
                        w1 = w1[:NUM_SAMPLES]
                    else:
                        w1 = F.pad(w1, (0, NUM_SAMPLES - w1.shape[-1]))
                    new_len2 = int(NUM_SAMPLES * s2)
                    w2 = F.interpolate(w2.view(1, 1, -1), size=new_len2, mode='linear', align_corners=False).view(-1)
                    if w2.shape[-1] > NUM_SAMPLES:
                        w2 = w2[:NUM_SAMPLES]
                    else:
                        w2 = F.pad(w2, (0, NUM_SAMPLES - w2.shape[-1]))
                waveforms_cat.append(torch.cat([w1, w2]))  # (32000,)
                labels1.append(all_labels[idx])
                labels2.append(all_labels[perm[idx]])

            waveforms_batch = torch.stack(waveforms_cat).to(device)  # (B, 16000)
            l1 = torch.tensor(labels1, device=device)
            l2 = torch.tensor(labels2, device=device)

            # Run model backbone up to sequence features
            x = waveforms_batch.unsqueeze(1)
            x = model.mel_spec(x)
            x = x.squeeze(1)
            x = (x + 1e-8).log()
            x = x.transpose(1, 2)
            h = model.embed(x)  # (B, T, n_embed)

            # Run cumsum layers
            if hasattr(model, 'shared_freq_proj'):
                # ModTied
                n_freqs = h.shape[-1] // 2
                for i in range(model.n_layers):
                    proj = model.shared_proj(h)
                    z_re, z_im = proj.chunk(2, dim=-1)
                    inst_freqs = model.freq_ln(model.shared_freq_proj(h))
                    z = torch.complex(z_re, z_im)
                    cum_phase = inst_freqs.cumsum(dim=1)
                    layer_phases = torch.exp(1j * cum_phase)
                    rotated = z * layer_phases.conj()
                    cs = rotated.cumsum(dim=1)
                    W = model.window if model.window is not None else h.shape[1]
                    cs_shifted = F.pad(cs[:, :-W], (0, 0, W, 0))
                    d = cs - cs_shifted
                    d = d * layer_phases
                    out = torch.cat([d.real, d.imag], dim=-1)
                    h = h + model.shared_glu(model.bn_layers[i](out))
            else:
                # FixedTied
                for i in range(model.n_layers):
                    T_cur = h.shape[1]
                    proj_layer = model.shared_proj if model.tie_layers else model.proj_layers[i]
                    glu_layer = model.shared_glu if model.tie_layers else model.glu_layers[i]
                    proj = proj_layer(h)
                    z_re, z_im = proj.chunk(2, dim=-1)
                    z = torch.complex(z_re, z_im)
                    layer_freqs = model.freq_params[i]
                    t_idx = torch.arange(T_cur, device=h.device, dtype=layer_freqs.dtype)
                    layer_phases = torch.exp(1j * t_idx.unsqueeze(1) * layer_freqs)
                    rotated = z * layer_phases.conj().unsqueeze(0)
                    cs = rotated.cumsum(dim=1)
                    W = model.window if model.window is not None else T_cur
                    cs_shifted = F.pad(cs[:, :-W], (0, 0, W, 0))
                    d = cs - cs_shifted
                    d = d * layer_phases.unsqueeze(0)
                    out = torch.cat([d.real, d.imag], dim=-1)
                    h = h + glu_layer(model.bn_layers[i](out))

            # Split at midpoint, maxpool each half, classify
            T = h.shape[1]
            mid = T // 2
            h_re, h_im = h.chunk(2, dim=-1)
            mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)

            mag1 = mag[:, :mid].max(dim=1).values
            mag2 = mag[:, mid:].max(dim=1).values

            pred1 = model.fc(mag1).argmax(dim=1)
            pred2 = model.fc(mag2).argmax(dim=1)

            both_correct = ((pred1 == l1) & (pred2 == l2)).sum().item()
            correct += both_correct
            total += len(labels1)

        acc = correct / total
        print(f"  max_stretch={max_s:.2f}: {acc:.4f} (both correct)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Speech Commands v2 baselines")
    model_choices = ["mel", "raw", "learned_spec", "rot_fixed", "rot_input", "rot_window",
                     "rot_window_input", "rot_decay_fixed", "rot_decay_input",
                     "rot_s5_fixed", "rot_s5_input",
                     "cumsum_s5", "window_s5",
                     "cumsum_s5_input", "window_s5_input",
                     "cumsum_s5_mod", "window_s5_mod",
                     "learned_spec_linear", "learned_spec_magreim", "learned_spec_multi", "learned_spec_multi4",
                     "learned_spec_mod", "learned_spec_mod2", "learned_spec_conv", "learned_spec_conv2",
                     "filterbank", "filterbank_mel", "filterbank_sincos", "filterbank_sincos_multi", "filterbank_sincos_linear", "filterbank_sincos_combined",
                     "filterbank_linear", "filterbank_mel_linear",
                     "minimal_strided", "multi_layer_minimal", "multi_layer_mod",
                     "multi_layer_v2", "multi_layer_mod_v2", "conv_cumsum_v2", "conv_cumsum_mod_v2",
                     "mel_cumsum_fixed", "mel_cumsum_mod",
                     "mel_cumsum_fixed_w", "mel_cumsum_mod_w",
                     "cumsum_mag_deep", "cumsum_mag_deep_proj", "cumsum_mag_deep_ds", "cumsum_complex", "cumsum_single", "cumsum_single_mp",
                     "cumsum_e2e", "cumsum_e2e_s320", "cumsum_e2e_s640",
                     "cumsum_e2e_mag", "cumsum_e2e_mag_s320", "cumsum_e2e_mag_s640",
                     "cumsum_e2e_mag_mod", "cumsum_e2e_mod",
                     "cumsum_e2e_mag_s1", "cumsum_e2e_mag_mod_s1",
                     "mel_maxpool", "mel_maxpool_nosa", "mel_maxpool_160", "mel_maxpool_80", "mel_maxpool_40",
                     "learned_spec_frozen", "learned_spec_magreim_frozen",
                     "filterbank_sincos_frozen", "filterbank_sincos_magreim_frozen",
                     "cumsum_mag_deep_freeze_proj",
                     "mel_multiphase_80", "mel_multiphase_40",
                     "mel_cumsum_fixed_tied", "mel_cumsum_mod_tied", "mel_cumsum_residual_mod_tied",
                     "mel_cumsum_fixed_mp2", "mel_cumsum_fixed_mp4",
                     "mel_cumsum_mod_mp2", "mel_cumsum_mod_mp4",
                     "mel_cumsum_fixed_tied_mp2", "mel_cumsum_mod_tied_mp2",
                     "mel_cumsum_bidir_tied",
                     "mel_cumsum_magdeep", "mel_cumsum_magdeep_tied",
                     "cumsum_spec_cumsum_tied", "cumsum_spec_cumsum_mod_tied",
                     "mel_cumsum_resnet",
                     "simple_strided", "strided_window_s5",
                     "block_decay_s5", "block_decay_s5_v2", "block_decay_s5_overlap", "block_decay_s5_mod", "block_decay",
                     "s5", "baselines", "rotation", "decay", "rots5", "cumsum", "cumsum_input", "cumsum_mod", "all"]
    parser.add_argument("--model", choices=model_choices, default="baselines",
                        help="Which model(s) to train")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: 128 for CNN, 16 for sequence models)")
    parser.add_argument("--n_embed", type=int, default=64, help="Embedding dim for sequence models")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers for sequence models")
    parser.add_argument("--pool", type=int, default=4,
                        help="Pool factor after Linear front-end (1=no pool=16K seq, 4=4K, 16=1K)")
    parser.add_argument("--window", type=int, default=80,
                        help="Window size for rot_window model (in pooled timesteps)")
    parser.add_argument("--window_l1", type=int, default=400, help="Window size for layer 1 of E2E cumsum (raw audio, default 400=25ms)")
    parser.add_argument("--stride_l1", type=int, default=160, help="Stride for layer 1 of E2E cumsum (default 160=10ms)")
    parser.add_argument("--readout", choices=["mag", "mlp", "mlp_direct"], default="mlp", help="Readout before maxpool: mag (magnitude), mlp (MLP→maxpool→fc), mlp_direct (MLP→12→maxpool)")
    parser.add_argument("--readout_mult", type=int, default=1, help="Multiplier for readout MLP hidden dim (e.g. 4 → Linear(dim, 4*dim) → ReLU → Linear(4*dim, ...))")
    parser.add_argument("--hop", type=int, default=160, help="Hop length for MelCNN/LearnedSpecCNN (default 160=10ms)")
    parser.add_argument("--ds_factor", type=int, default=10, help="Downsample factor for MultiLayerMinimalStrided (default 10)")
    parser.add_argument("--time_shift", type=int, default=0, help="Random time shift ±N samples on train+test (not val)")
    parser.add_argument("--lr", type=float, default=None, help="Fixed learning rate (disables cosine schedule)")
    parser.add_argument("--no_freq_bias", action="store_true", help="Remove bias from freq projection and LayerNorm in ModTied")
    parser.add_argument("--future_phase", action="store_true", help="(deprecated, use --phase_mode)")
    parser.add_argument("--phase_mode", type=str, default="default", choices=["default", "derot_prev", "both_prev"], help="Phase mode: default=both Φ(t), derot_prev=derot Φ(t-1) rerot Φ(t), both_prev=both Φ(t-1)")
    parser.add_argument("--zero_freqs", action="store_true", help="Zero and freeze all frequencies in Fixed model")
    parser.add_argument("--freq_bottleneck", type=int, default=0, help="Bottleneck dim for freq prediction MLP in ModTied (0=disabled)")
    parser.add_argument("--time_stretch", action="store_true", help="Time-stretch augmentation during training")
    parser.add_argument("--stretch_range", type=float, nargs=2, default=[0.8, 1.2], help="Min/max stretch range (default: 0.8 1.2)")
    parser.add_argument("--split_stretch", action="store_true", help="Split-stretch augmentation (each half gets independent 0.8x-1.2x stretch)")
    parser.add_argument("--distortion", action="store_true", help="Nonlinear distortion augmentation (x + alpha*x^3) during training")
    parser.add_argument("--eval_grid", action="store_true", help="Evaluate on perturbation grid after training")
    parser.add_argument("--eval_split", action="store_true", help="Evaluate split-stretch (each half of waveform gets independent time-stretch)")
    parser.add_argument("--eval_dual", action="store_true", help="Evaluate dual-command (two commands concatenated, independent distortion)")
    parser.add_argument("--smoke", action="store_true", help="Quick test (2 epochs, small subset)")
    parser.add_argument("--data_dir", default="./data", help="Data download directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.smoke:
        args.epochs = 2
        print("SMOKE TEST MODE (2 epochs)")

    # Load background noise for silence generation
    print("Loading data...")
    noise_wavs = load_noise_wavs(args.data_dir)
    print(f"  Loaded {len(noise_wavs)} background noise files")

    # Load datasets
    unknown_cap = 300 if args.smoke else None
    train_ds = SpeechCommandsDataset(
        args.data_dir, "training", augment=True,
        unknown_cap=unknown_cap, noise_wavs=noise_wavs,
        time_shift=args.time_shift,
        time_stretch=args.time_stretch, split_stretch=args.split_stretch,
        distortion=args.distortion,
        stretch_range=tuple(args.stretch_range),
    )
    val_ds = SpeechCommandsDataset(
        args.data_dir, "validation", augment=False,
        unknown_cap=unknown_cap, noise_wavs=noise_wavs,
        time_shift=args.time_shift,
    )
    test_ds = SpeechCommandsDataset(
        args.data_dir, "testing", augment=False,
        unknown_cap=unknown_cap, noise_wavs=noise_wavs,
        time_shift=args.time_shift,
    )

    # Print dataset stats
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        labels = [s[1] for s in ds.samples]
        counts = Counter(labels)
        dist = ", ".join(f"{LABELS[k]}:{v}" for k, v in sorted(counts.items()))
        print(f"  {name}: {len(ds)} samples ({dist})")

    def make_loaders(bs):
        tl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=False)
        vl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=2, pin_memory=True)
        tel = DataLoader(test_ds, batch_size=bs, shuffle=False,
                         num_workers=2, pin_memory=True)
        return tl, vl, tel

    results = {}
    models_to_run = []
    if args.model in ("mel", "baselines", "all"):
        models_to_run.append(("MelCNN", MelCNNWithSpecAugment(hop_length=args.hop).to(device)))
    if args.model in ("mel_maxpool",):
        models_to_run.append(("MelMaxPool", MelCNNMaxPool(n_mels=args.n_embed or 40, hop_length=args.hop).to(device)))
    if args.model in ("mel_maxpool_nosa",):
        models_to_run.append(("MelMaxPoolNoSA", MelCNNMaxPool(n_mels=args.n_embed or 40, hop_length=args.hop, no_spec_aug=True).to(device)))
    if args.model in ("mel_maxpool_160",):
        models_to_run.append(("MelMaxPool160", MelCNNMaxPool(hop_length=160).to(device)))
    if args.model in ("mel_maxpool_80",):
        models_to_run.append(("MelMaxPool80", MelCNNMaxPool(hop_length=80).to(device)))
    if args.model in ("mel_maxpool_40",):
        models_to_run.append(("MelMaxPool40", MelCNNMaxPool(hop_length=40).to(device)))
    if args.model in ("mel_multiphase_80",):
        models_to_run.append(("MelMultiPhase80", MelCNNMultiPhase(hop_length=80).to(device)))
    if args.model in ("mel_multiphase_40",):
        models_to_run.append(("MelMultiPhase40", MelCNNMultiPhase(hop_length=40).to(device)))
    if args.model in ("raw", "baselines", "all"):
        models_to_run.append(("RawCNN", RawCNN().to(device)))
    if args.model in ("learned_spec", "all"):
        models_to_run.append(("LearnedSpecCNN", LearnedSpecCNN(n_freqs=args.n_embed or 40, window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("learned_spec_frozen",):
        models_to_run.append(("LearnedSpecFrozen", LearnedSpecCNN(n_freqs=args.n_embed or 40, window=args.window, hop_length=args.hop, freeze_freqs=True).to(device)))
    if args.model in ("learned_spec_linear",):
        models_to_run.append(("LearnedSpecLinear", LearnedSpecLinearCNN(window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("learned_spec_magreim",):
        models_to_run.append(("LearnedSpecMagReIm", LearnedSpecMagReImCNN(n_freqs=args.n_embed or 40, window=args.window_l1, hop_length=args.hop).to(device)))
    if args.model in ("learned_spec_magreim_frozen",):
        models_to_run.append(("LearnedSpecMagReImFrozen", LearnedSpecMagReImCNN(n_freqs=args.n_embed or 40, window=args.window_l1, hop_length=args.hop, freeze_freqs=True).to(device)))
    if args.model in ("learned_spec_multi",):
        models_to_run.append(("LearnedSpecMulti", LearnedSpecMultiCNN(window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("learned_spec_multi4",):
        models_to_run.append(("LearnedSpecMulti4", LearnedSpecMultiCNN(n_per_bin=4, window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("learned_spec_mod",):
        models_to_run.append(("LearnedSpecCNNMod", LearnedSpecCNNMod(hop_length=args.hop).to(device)))
    if args.model in ("learned_spec_mod2",):
        models_to_run.append(("LearnedSpecCNNMod2", LearnedSpecCNNMod2(hop_length=args.hop).to(device)))
    if args.model in ("learned_spec_conv",):
        models_to_run.append(("LearnedSpecCNNConv", LearnedSpecCNNConv(hop_length=args.hop).to(device)))
    if args.model in ("learned_spec_conv2",):
        models_to_run.append(("LearnedSpecCNNConv2", LearnedSpecCNNConv2(hop_length=args.hop).to(device)))
    if args.model in ("filterbank",):
        models_to_run.append(("FilterbankCNN", FilterbankCNN(window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("filterbank_mel",):
        models_to_run.append(("FilterbankMelInit", FilterbankMelInitCNN(window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("filterbank_sincos",):
        models_to_run.append(("FilterbankSinCos", FilterbankSinCosCNN(window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("filterbank_sincos_frozen",):
        models_to_run.append(("FilterbankSinCosFrozen", FilterbankSinCosCNN(n_freqs=args.n_embed or 40, window=args.window, hop_length=args.hop, freeze_filters=True).to(device)))
    if args.model in ("filterbank_sincos_magreim_frozen",):
        models_to_run.append(("FilterbankSinCosMagReImFrozen", FilterbankSinCosMagReImCNN(n_freqs=args.n_embed or 40, window=args.window, hop_length=args.hop, freeze_filters=True).to(device)))
    if args.model in ("filterbank_sincos_multi",):
        models_to_run.append(("FilterbankSinCosMulti", FilterbankSinCosMultiCNN(window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("filterbank_sincos_linear",):
        models_to_run.append(("FilterbankSinCosLinear", FilterbankSinCosLinearCNN(window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("filterbank_sincos_combined",):
        models_to_run.append(("FilterbankSinCosCombined", FilterbankSinCosCombinedCNN(window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("filterbank_linear",):
        models_to_run.append(("FilterbankLinear", FilterbankLinearCNN(window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("filterbank_mel_linear",):
        models_to_run.append(("FilterbankMelInitLinear", FilterbankMelInitLinearCNN(window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("rot_fixed", "rotation", "all"):
        models_to_run.append(("RotFixed", RotFixed(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool).to(device)))
    if args.model in ("rot_input", "rotation", "all"):
        models_to_run.append(("RotInput", RotInput(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool).to(device)))
    if args.model in ("rot_window", "rotation", "all"):
        models_to_run.append(("RotWindow", RotWindow(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool,
            window=args.window).to(device)))
    if args.model in ("rot_window_input", "rotation", "all"):
        models_to_run.append(("RotWinInput", RotWindowInput(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool,
            window=args.window).to(device)))
    if args.model in ("rot_decay_fixed", "decay", "all"):
        models_to_run.append(("RotDecayFixed", RotDecayFixed(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool).to(device)))
    if args.model in ("rot_decay_input", "decay", "all"):
        models_to_run.append(("RotDecayInput", RotDecayInput(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool).to(device)))
    if args.model in ("rot_s5_fixed", "rots5", "all"):
        models_to_run.append(("RotS5Fixed", RotS5Fixed(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool).to(device)))
    if args.model in ("rot_s5_input", "rots5", "all"):
        models_to_run.append(("RotS5Input", RotS5Input(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool).to(device)))
    if args.model in ("cumsum_s5", "cumsum", "all"):
        models_to_run.append(("CumsumS5", CumsumS5(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool).to(device)))
    if args.model in ("window_s5", "cumsum", "all"):
        models_to_run.append(("WindowS5", WindowS5(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool,
            window=args.window).to(device)))
    if args.model in ("minimal_strided",):
        models_to_run.append(("MinimalStridedWindow", MinimalStridedWindow(
            n_freqs=args.n_embed, window=args.window,
            hop_length=args.hop).to(device)))
    if args.model in ("multi_layer_minimal",):
        models_to_run.append(("MultiLayerMinimal", MultiLayerMinimalStrided(
            n_freqs=args.n_embed, window=args.window,
            ds_factor=args.ds_factor, n_layers=args.n_layers).to(device)))
    if args.model in ("multi_layer_mod",):
        models_to_run.append(("MultiLayerMod", MultiLayerMinimalMod(
            n_freqs=args.n_embed, window=args.window,
            ds_factor=args.ds_factor, n_layers=args.n_layers).to(device)))
    if args.model in ("multi_layer_v2",):
        models_to_run.append(("MultiLayerV2", MultiLayerMinimalV2(
            n_freqs=args.n_embed, window=args.window,
            n_layers=args.n_layers).to(device)))
    if args.model in ("multi_layer_mod_v2",):
        models_to_run.append(("MultiLayerModV2", MultiLayerMinimalModV2(
            n_freqs=args.n_embed, window=args.window,
            n_layers=args.n_layers).to(device)))
    if args.model in ("conv_cumsum_v2",):
        models_to_run.append(("ConvCumsumV2", ConvCumsumV2(
            n_freqs=args.n_embed, window=args.window,
            n_layers=args.n_layers).to(device)))
    if args.model in ("conv_cumsum_mod_v2",):
        models_to_run.append(("ConvCumsumModV2", ConvCumsumModV2(
            n_freqs=args.n_embed, window=args.window,
            n_layers=args.n_layers).to(device)))
    if args.model in ("cumsum_mag_deep",):
        models_to_run.append(("CumsumMagDeep", CumsumMagDeep(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers,
            stride_l1=args.stride_l1).to(device)))
    if args.model in ("cumsum_mag_deep_proj",):
        models_to_run.append(("CumsumMagDeepProj", CumsumMagDeep(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers,
            stride_l1=args.stride_l1, use_proj=True).to(device)))
    if args.model in ("cumsum_mag_deep_freeze_proj",):
        models_to_run.append(("CumsumMagDeepFreezeProj", CumsumMagDeep(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers,
            stride_l1=args.stride_l1, freeze_proj=True).to(device)))
    if args.model in ("cumsum_mag_deep_ds",):
        models_to_run.append(("CumsumMagDeepDS", CumsumMagDeep(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers,
            stride_l1=args.stride_l1, downsample=True).to(device)))
    if args.model in ("cumsum_complex",):
        models_to_run.append(("CumsumComplex", CumsumComplex(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers,
            stride_l1=args.stride_l1, readout=args.readout, readout_mult=args.readout_mult).to(device)))
    if args.model in ("cumsum_single",):
        models_to_run.append(("CumsumSingle", CumsumSingleLayer(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            stride_l1=args.stride_l1, readout_mult=args.readout_mult).to(device)))
    if args.model in ("cumsum_single_mp",):
        models_to_run.append(("CumsumSingleMP", CumsumSingleLayer(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            stride_l1=args.stride_l1, readout_mult=args.readout_mult,
            features='mag_phase').to(device)))
    if args.model in ("cumsum_e2e",):
        models_to_run.append(("CumsumE2E", CumsumEndToEnd(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers, readout=args.readout, readout_mult=args.readout_mult).to(device)))
    if args.model in ("cumsum_e2e_s320",):
        models_to_run.append(("CumsumE2E_s320", CumsumEndToEnd(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers, stride_l1=320, readout=args.readout, readout_mult=args.readout_mult).to(device)))
    if args.model in ("cumsum_e2e_s640",):
        models_to_run.append(("CumsumE2E_s640", CumsumEndToEnd(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers, stride_l1=640, readout=args.readout, readout_mult=args.readout_mult).to(device)))
    if args.model in ("cumsum_e2e_mag",):
        models_to_run.append(("CumsumE2EMag", CumsumEndToEndMag(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers).to(device)))
    if args.model in ("cumsum_e2e_mag_s320",):
        models_to_run.append(("CumsumE2EMag_s320", CumsumEndToEndMag(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers, stride_l1=320).to(device)))
    if args.model in ("cumsum_e2e_mag_s640",):
        models_to_run.append(("CumsumE2EMag_s640", CumsumEndToEndMag(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers, stride_l1=640).to(device)))
    if args.model in ("cumsum_e2e_mag_mod",):
        models_to_run.append(("CumsumE2EMagMod", CumsumEndToEndMagMod(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers).to(device)))
    if args.model in ("cumsum_e2e_mod",):
        models_to_run.append(("CumsumE2EMod", CumsumEndToEndMod(
            n_freqs=args.n_embed, window_l1=args.window_l1,
            window=args.window, n_layers=args.n_layers, readout=args.readout, readout_mult=args.readout_mult).to(device)))
    if args.model in ("cumsum_e2e_mag_s1",):
        models_to_run.append(("CumsumE2EMagS1", CumsumEndToEndMag(
            n_freqs=args.n_embed, window_l1=args.window, window=args.window,
            n_layers=args.n_layers, stride_l1=1).to(device)))
    if args.model in ("cumsum_e2e_mag_mod_s1",):
        models_to_run.append(("CumsumE2EMagModS1", CumsumEndToEndMagMod(
            n_freqs=args.n_embed, window_l1=args.window, window=args.window,
            n_layers=args.n_layers, stride_l1=1).to(device)))
    if args.model in ("mel_cumsum_fixed",):
        models_to_run.append(("MelCumsumFixed", MelCumsumFixed(
            n_embed=args.n_embed, n_layers=args.n_layers).to(device)))
    if args.model in ("mel_cumsum_fixed_tied",):
        models_to_run.append(("MelCumsumFixedTied", MelCumsumFixed(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, hop_length=args.hop, tie_layers=True,
            zero_freqs=args.zero_freqs).to(device)))
    if args.model in ("mel_cumsum_mod_tied",):
        models_to_run.append(("MelCumsumModTied", MelCumsumModTied(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, hop_length=args.hop,
            freq_bias=not args.no_freq_bias,
            freq_bottleneck=args.freq_bottleneck,
            future_phase=args.future_phase,
            phase_mode=args.phase_mode).to(device)))
    if args.model in ("mel_cumsum_fixed_tied_mp2",):
        models_to_run.append(("MelCumsumFixedTiedMP2", MelCumsumFixed(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, hop_length=args.hop, n_phases=2, tie_layers=True).to(device)))
    if args.model in ("mel_cumsum_mod_tied_mp2",):
        models_to_run.append(("MelCumsumModTiedMP2", MelCumsumModTied(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, hop_length=args.hop, n_phases=2).to(device)))
    if args.model in ("mel_cumsum_bidir_tied",):
        models_to_run.append(("MelCumsumBidirTied", MelCumsumBidirTied(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("mel_cumsum_residual_mod_tied",):
        models_to_run.append(("MelCumsumResidualModTied", MelCumsumResidualModTied(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, hop_length=args.hop).to(device)))
    if args.model in ("mel_cumsum_mod",):
        models_to_run.append(("MelCumsumMod", MelCumsumMod(
            n_embed=args.n_embed, n_layers=args.n_layers).to(device)))
    if args.model in ("mel_cumsum_fixed_w",):
        models_to_run.append(("MelCumsumFixedW", MelCumsumFixed(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window).to(device)))
    if args.model in ("mel_cumsum_mod_w",):
        models_to_run.append(("MelCumsumModW", MelCumsumMod(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window).to(device)))
    if args.model in ("mel_cumsum_fixed_mp2",):
        models_to_run.append(("MelCumsumFixedMP2", MelCumsumFixed(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, hop_length=80, n_phases=2).to(device)))
    if args.model in ("mel_cumsum_fixed_mp4",):
        models_to_run.append(("MelCumsumFixedMP4", MelCumsumFixed(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, hop_length=40, n_phases=4).to(device)))
    if args.model in ("mel_cumsum_mod_mp2",):
        models_to_run.append(("MelCumsumModMP2", MelCumsumMod(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, hop_length=80, n_phases=2).to(device)))
    if args.model in ("mel_cumsum_mod_mp4",):
        models_to_run.append(("MelCumsumModMP4", MelCumsumMod(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, hop_length=40, n_phases=4).to(device)))
    if args.model in ("mel_cumsum_magdeep",):
        models_to_run.append(("MelCumsumMagDeep", MelCumsumMagDeep(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window).to(device)))
    if args.model in ("mel_cumsum_magdeep_tied",):
        models_to_run.append(("MelCumsumMagDeepTied", MelCumsumMagDeep(
            n_embed=args.n_embed, n_layers=args.n_layers,
            window=args.window, tie_layers=True).to(device)))
    if args.model in ("cumsum_spec_cumsum_tied",):
        models_to_run.append(("CumsumSpecCumsumTied", CumsumSpecCumsumTied(
            n_freqs=args.n_embed or 40, n_embed=args.n_embed,
            n_layers=args.n_layers, window_l1=args.window_l1,
            hop_length=args.hop, window=args.window).to(device)))
    if args.model in ("cumsum_spec_cumsum_mod_tied",):
        models_to_run.append(("CumsumSpecCumsumModTied", CumsumSpecCumsumModTied(
            n_freqs=args.n_embed or 40, n_embed=args.n_embed,
            n_layers=args.n_layers, window_l1=args.window_l1,
            hop_length=args.hop, window=args.window).to(device)))
    if args.model in ("mel_cumsum_resnet",):
        ch0 = args.n_embed if args.n_embed != 64 else 16  # default 16 unless overridden
        channels = [ch0, max(ch0, 24), max(ch0+8, 32), max(ch0+24, 48)]
        models_to_run.append(("MelCumsumResNet", MelCumsumResNet(
            channels=channels, window=args.window).to(device)))
    if args.model in ("simple_strided",):
        models_to_run.append(("SimpleStridedWindow", SimpleStridedWindow(
            d_model=args.n_embed, n_layers=args.n_layers, window=args.window,
            stride=args.hop).to(device)))
    if args.model in ("strided_window_s5",):
        models_to_run.append(("StridedWindowS5", StridedWindowS5(
            d_model=args.n_embed, n_layers=args.n_layers, window=args.window,
            stride=args.hop).to(device)))
    if args.model in ("cumsum_s5_input", "cumsum_input", "all"):
        models_to_run.append(("CumsumS5Input", CumsumS5Input(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool).to(device)))
    if args.model in ("window_s5_input", "cumsum_input", "all"):
        models_to_run.append(("WindowS5Input", WindowS5Input(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool,
            window=args.window).to(device)))
    if args.model in ("cumsum_s5_mod", "cumsum_mod", "all"):
        models_to_run.append(("CumsumS5Mod", CumsumS5Mod(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool).to(device)))
    if args.model in ("window_s5_mod", "cumsum_mod", "all"):
        models_to_run.append(("WindowS5Mod", WindowS5Mod(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool,
            window=args.window).to(device)))
    if args.model in ("block_decay_s5", "block_decay", "all"):
        models_to_run.append(("BlockDecayS5", BlockDecayS5(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool,
            window=args.window).to(device)))
    if args.model in ("block_decay_s5_v2", "block_decay", "all"):
        models_to_run.append(("BlockDecayS5V2", BlockDecayS5V2(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool,
            window=args.window).to(device)))
    if args.model in ("block_decay_s5_overlap", "block_decay", "all"):
        models_to_run.append(("BlockDecayS5Overlap", BlockDecayS5Overlap(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool,
            window=args.window).to(device)))
    if args.model in ("block_decay_s5_mod", "block_decay", "all"):
        models_to_run.append(("BlockDecayS5Mod", BlockDecayS5Mod(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool,
            window=args.window).to(device)))
    if args.model in ("s5", "all"):
        models_to_run.append(("S5", S5Model(
            d_model=args.n_embed, n_layers=args.n_layers, pool=args.pool).to(device)))

    for model_name, model in models_to_run:
        is_cnn = model_name in ('MelCNN', 'RawCNN', 'LearnedSpecCNN', 'LearnedSpecLinear', 'LearnedSpecCNNMod', 'LearnedSpecCNNMod2', 'LearnedSpecCNNConv', 'LearnedSpecCNNConv2', 'FilterbankCNN', 'FilterbankMelInit')
        bs = args.batch_size or (128 if is_cnn else 16)
        train_loader, val_loader, test_loader = make_loaders(bs)
        model = train_model(model, train_loader, val_loader, args.epochs, device, model_name, fixed_lr=args.lr)

        # Evaluate on test set
        test_acc, preds, labels = evaluate(model, test_loader, device)
        per_class_acc, per_class_f1, macro_f1, cm = compute_metrics(preds, labels)

        results[model_name] = {
            "test_acc": test_acc,
            "macro_f1": macro_f1,
            "per_class_acc": per_class_acc,
            "per_class_f1": per_class_f1,
            "cm": cm,
        }

        print(f"\n{'='*60}")
        print(f"{model_name} — Test Results")
        print(f"{'='*60}")
        print(f"  Top-1 Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        print(f"  Macro F1:       {macro_f1:.4f}")
        print(f"\n  Per-class accuracy:")
        for name in LABELS:
            acc = per_class_acc[name]
            f1 = per_class_f1[name]
            print(f"    {name:>8s}: acc={acc:.3f}  f1={f1:.3f}")
        print(f"\n  Confusion matrix:")
        print_confusion_matrix(cm)

        # Frequency analysis for Mod models
        if hasattr(model, 'shared_freq_proj'):
            print(f"\n  Frequency Analysis:")
            model.eval()
            all_freqs = []
            with torch.no_grad():
                loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)
                for waveforms, labels in loader:
                    x = waveforms.unsqueeze(1).to(device)
                    x = model.mel_spec(x).squeeze(1)
                    x = (x + 1e-8).log().transpose(1, 2)
                    h = model.embed(x)
                    freqs = model.freq_ln(model.shared_freq_proj(h))  # (B, T, n_freqs)
                    # Mean over time per sample
                    all_freqs.append(freqs.mean(dim=1).cpu())  # (B, n_freqs)
            all_freqs = torch.cat(all_freqs)  # (N, n_freqs)
            # Cross-sample std per frequency (how much does each freq vary across inputs?)
            per_freq_std = all_freqs.std(dim=0)
            # Cross-sample mean per frequency
            per_freq_mean = all_freqs.mean(dim=0)
            print(f"    Per-freq mean (first 10): {per_freq_mean[:10].numpy()}")
            print(f"    Per-freq std  (first 10): {per_freq_std[:10].numpy()}")
            print(f"    Avg cross-sample std: {per_freq_std.mean().item():.4f}")
            print(f"    Freq range: {all_freqs.min().item():.4f} to {all_freqs.max().item():.4f}")

        # Frequency analysis for Fixed models
        if hasattr(model, 'freq_params'):
            print(f"\n  Fixed Frequencies:")
            for i, fp in enumerate(model.freq_params):
                vals = fp.data.cpu()
                print(f"    Layer {i}: min={vals.min().item():.4f} max={vals.max().item():.4f} mean={vals.mean().item():.4f}")

        if args.eval_grid:
            evaluate_perturbation_grid(model, test_ds, device, model_name)

        if args.eval_split:
            print(f"\n{'='*60}")
            print(f"Split-Stretch Eval: {model_name}")
            print(f"{'='*60}")
            evaluate_split_stretch(model, test_ds, device)

        if args.eval_dual:
            print(f"\n{'='*60}")
            print(f"Dual-Command Eval: {model_name}")
            print(f"{'='*60}")
            evaluate_dual_command(model, test_ds, device)

    # Summary table
    if len(results) >= 1:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"  {'Model':<12s} {'Accuracy':>10s} {'Macro F1':>10s} {'Params':>10s}")
        for model_name, model in models_to_run:
            r = results[model_name]
            print(f"  {model_name:<12s} {r['test_acc']*100:9.1f}% {r['macro_f1']:10.4f} {model.param_count():>10,}")


if __name__ == "__main__":
    main()
