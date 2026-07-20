#!/usr/bin/env python3
"""Benchmark: cumsum vs S5/SSM — streaming, training speed, and inference.

Three comparison scenarios:
  1. Sample-by-sample (CumsumEndToEnd vs S5): both raw waveform, both stream
     per-sample. CumsumEndToEnd uses running cumsum + circular buffer.
     S5 uses complex multiply recurrence h[t] = Lambda*h[t-1] + B*x[t].
  2. Training wall time: CumsumEndToEnd vs S5 vs BlockDecayS5V2.
  3. Hop-by-hop (MelCumsumFixed): mel-based model streams per-hop with
     O(1) state update per frame. Batch/streaming equivalence proof.

Plus: primitive operation comparison (torch.cumsum vs torch.associative_scan).
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(__file__))
from speech_commands import (
    SpeechCommandsDataset, load_noise_wavs,
    CumsumEndToEnd, MelCumsumFixed, MelCNN, S5Model, BlockDecayS5V2,
    TransposedBN, GLU,
    train_one_epoch, evaluate,
    NUM_CLASSES, SAMPLE_RATE, NUM_SAMPLES,
)


# ─── Streaming Wrappers ─────────────────────────────────────────────────────

class StreamingCumsumEndToEnd(nn.Module):
    """Per-sample streaming wrapper for CumsumEndToEnd.

    Layer 1: processes each raw audio sample, accumulating cumsum state.
      Every stride_l1 samples, emits a frame to layers 2+.
    Layers 2+: process each emitted frame through cumsum layers.
    Readout: running max over readout outputs, classify at end.
    """

    def __init__(self, model: CumsumEndToEnd):
        super().__init__()
        self.model = model
        self.n_freqs = model.n_freqs
        self.dim = 2 * model.n_freqs
        self.window_l1 = model.window_l1
        self.stride_l1 = model.stride_l1
        self.n_layers = model.n_layers

    def reset(self):
        device = self.model.log_freqs.device
        # Layer 1 state
        self.l1_running_cs = torch.zeros(self.n_freqs, dtype=torch.complex64, device=device)
        self.l1_buffer = torch.zeros(self.window_l1, self.n_freqs, dtype=torch.complex64, device=device)
        self.l1_freqs = self.model.log_freqs.exp().detach()
        # Layers 2+ state
        self.l2_running_cs = []
        self.l2_buffer = []
        for i in range(self.n_layers - 1):
            W = self.model.window
            self.l2_running_cs.append(torch.zeros(self.n_freqs, dtype=torch.complex64, device=device))
            self.l2_buffer.append(torch.zeros(W, self.n_freqs, dtype=torch.complex64, device=device))
        self.l2_freqs = [self.model.freq_params[i].detach() for i in range(self.n_layers - 1)]
        # Readout state
        if self.model.readout_mode == "mlp":
            self.running_max = torch.full((self.n_freqs,), -float('inf'), device=device)
        elif self.model.readout_mode == "mlp_direct":
            self.running_max = torch.full((NUM_CLASSES,), -float('inf'), device=device)
        else:  # mag
            self.running_max = torch.full((self.n_freqs,), -float('inf'), device=device)
        self.sample_t = 0  # raw sample counter
        self.frame_t = 0   # layer 2+ frame counter

    def step_sample(self, x_t):
        """Process one raw audio sample. Returns logits-ready frame or None."""
        t = self.sample_t
        freqs = self.l1_freqs

        # Derotate + cumsum update
        phase_t = torch.exp(1j * t * freqs)
        z = x_t.to(torch.complex64) * phase_t.conj()
        self.l1_running_cs = self.l1_running_cs + z

        # Window diff via circular buffer
        buf_idx = t % self.window_l1
        if t >= self.window_l1:
            d_t = (self.l1_running_cs - self.l1_buffer[buf_idx]) * phase_t
        else:
            d_t = self.l1_running_cs * phase_t
        self.l1_buffer[buf_idx] = self.l1_running_cs.clone()

        self.sample_t += 1

        # Only emit a frame every stride_l1 samples
        if self.sample_t % self.stride_l1 != 0:
            return None

        # Build layer 1 output
        h = torch.cat([d_t.real, d_t.imag], dim=-1)  # (dim,)
        # BN + GLU (layer 1)
        h_3d = h.unsqueeze(0).unsqueeze(0)  # (1,1,dim)
        h_3d = self.model.bn1(h_3d)
        h = h.unsqueeze(0) + self.model.glu1(h_3d).squeeze(0)
        h = h.squeeze(0)  # (dim,)

        # Process through layers 2+
        for i in range(self.n_layers - 1):
            proj = self.model.proj_layers[i](h)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            freqs_i = self.l2_freqs[i]
            ft = self.frame_t
            phase_ft = torch.exp(1j * ft * freqs_i)
            rotated = z * phase_ft.conj()

            self.l2_running_cs[i] = self.l2_running_cs[i] + rotated

            W = self.l2_buffer[i].shape[0]
            buf_idx = ft % W
            if ft >= W:
                d = (self.l2_running_cs[i] - self.l2_buffer[i][buf_idx]) * phase_ft
            else:
                d = self.l2_running_cs[i] * phase_ft
            self.l2_buffer[i][buf_idx] = self.l2_running_cs[i].clone()

            out = torch.cat([d.real, d.imag], dim=-1)
            out_3d = out.unsqueeze(0).unsqueeze(0)
            out_3d = self.model.bn_layers[i](out_3d)
            out = out_3d.squeeze(0).squeeze(0)
            h = h + self.model.glu_layers[i](out.unsqueeze(0)).squeeze(0)

        self.frame_t += 1

        # Readout and track running max
        if self.model.readout_mode in ("mlp", "mlp_direct"):
            r = self.model.readout(h.unsqueeze(0)).squeeze(0)
            self.running_max = torch.max(self.running_max, r)
        else:  # mag
            h_re, h_im = h.chunk(2, dim=-1)
            mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
            self.running_max = torch.max(self.running_max, mag)

        return h

    def classify(self):
        if self.model.readout_mode == "mlp":
            return self.model.fc(self.running_max).unsqueeze(0)
        elif self.model.readout_mode == "mlp_direct":
            return self.running_max.unsqueeze(0)
        else:
            return self.model.fc(self.running_max).unsqueeze(0)

    def streaming_forward(self, x):
        """Full streaming pass over raw waveform x: (16000,)."""
        self.reset()
        for t in range(x.shape[0]):
            self.step_sample(x[t:t+1])
        return self.classify()

    def state_size_bytes(self):
        W1 = self.window_l1
        W2 = self.model.window
        n = self.n_freqs
        # Layer 1: running_cs + buffer
        l1 = (1 + W1) * n * 8  # complex64
        # Layers 2+: running_cs + buffer each
        l2 = (self.n_layers - 1) * (1 + W2) * n * 8
        # running_max + counters
        extra = n * 4 + 8
        return l1 + l2 + extra


class StreamingMelCumsumFixed(nn.Module):
    """Per-hop streaming wrapper for MelCumsumFixed.

    Each mel frame (one per hop) is processed through cumsum layers with
    O(1) state update. Produces identical output to batch forward.
    """

    def __init__(self, model: MelCumsumFixed):
        super().__init__()
        self.model = model
        self.n_layers = model.n_layers
        self.n_freqs = model.fc.in_features
        self.n_embed = self.n_freqs * 2
        self.window = model.window
        self.freqs = [model.freq_params[i].detach() for i in range(self.n_layers)]

    def reset(self):
        device = self.freqs[0].device
        W = self.window if self.window is not None else 1
        self.running_cs = [torch.zeros(self.n_freqs, dtype=torch.complex64, device=device)
                           for _ in range(self.n_layers)]
        self.buffer = [torch.zeros(W, self.n_freqs, dtype=torch.complex64, device=device)
                       for _ in range(self.n_layers)]
        self.max_mag = torch.zeros(self.n_freqs, device=device)
        self.t = 0

    def step(self, h_t):
        """Process one embedded mel frame through all cumsum layers."""
        t = self.t
        model = self.model

        for i in range(self.n_layers):
            proj_layer = model.shared_proj if model.tie_layers else model.proj_layers[i]
            glu_layer = model.shared_glu if model.tie_layers else model.glu_layers[i]
            bn = model.bn_layers[i]

            proj = proj_layer(h_t)
            z_re, z_im = proj.chunk(2, dim=-1)
            z = torch.complex(z_re, z_im)

            freqs = self.freqs[i]
            phase_t = torch.exp(1j * t * freqs)
            rotated_t = z * phase_t.conj()

            self.running_cs[i] = self.running_cs[i] + rotated_t

            W = self.buffer[i].shape[0]
            buf_idx = t % W
            if self.window is not None and t >= W:
                d_t = (self.running_cs[i] - self.buffer[i][buf_idx]) * phase_t
            else:
                d_t = self.running_cs[i] * phase_t
            self.buffer[i][buf_idx] = self.running_cs[i].clone()

            out = torch.cat([d_t.real, d_t.imag], dim=-1)
            out_3d = out.unsqueeze(0).unsqueeze(0)
            out_3d = bn(out_3d)
            out = out_3d.squeeze(0).squeeze(0)
            h_t = h_t + glu_layer(out.unsqueeze(0)).squeeze(0)

        h_re, h_im = h_t.chunk(2, dim=-1)
        mag = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-8)
        self.max_mag = torch.max(self.max_mag, mag)
        self.t += 1
        return h_t

    def classify(self):
        return self.model.fc(self.max_mag).unsqueeze(0)

    def streaming_forward(self, mel_log):
        """Full streaming pass. mel_log: (1, 40, T) log mel spectrogram."""
        self.reset()
        x = mel_log.squeeze(0).transpose(0, 1)  # (T, 40)
        h_seq = self.model.embed(x)
        for t in range(h_seq.shape[0]):
            self.step(h_seq[t])
        return self.classify()

    def state_size_bytes(self):
        W = self.window if self.window is not None else 1
        per_layer = (1 + W) * self.n_freqs * 8
        total = self.n_layers * per_layer
        total += self.n_freqs * 4 + 4
        return total


class StreamingS5(nn.Module):
    """Per-sample streaming wrapper for S5Model (forward direction only).

    Uses the forward SSM recurrence: h[t] = Lambda_bar * h[t-1] + B_bar @ x[t]
    Drops the backward SSM (would need full sequence).
    NOTE: This produces DIFFERENT outputs from the batch bidirectional model.
    We verify equivalence against a forward-only batch pass instead.
    """

    def __init__(self, model: S5Model):
        super().__init__()
        self.model = model
        self.n_blocks = len(model.blocks)
        self.d_model = model.fc.in_features
        self.pool_size = model.pool_size
        # Precompute discretized SSM parameters per block
        self.block_params = []
        for block in model.blocks:
            ssm_f = block.ssm.ssm_f
            Lambda = torch.complex(ssm_f.Lambda_re.clamp(max=-1e-4), ssm_f.Lambda_im)
            dt = ssm_f.log_dt.exp()
            Lambda_bar = torch.exp(Lambda * dt)
            Bmat = torch.complex(ssm_f.B_re, ssm_f.B_im)
            B_bar = ((Lambda_bar - 1.0) / Lambda).unsqueeze(-1) * Bmat
            Cmat = torch.complex(ssm_f.C_re, ssm_f.C_im)
            D = block.ssm.D
            self.block_params.append((Lambda_bar.detach(), B_bar.detach(), Cmat.detach(), D.detach()))

    def reset(self):
        device = self.block_params[0][0].device
        n = self.block_params[0][0].shape[0]  # state dim
        self.h_states = [torch.zeros(n, dtype=torch.complex64, device=device)
                         for _ in range(self.n_blocks)]
        self.output_sum = torch.zeros(self.d_model, device=device)
        self.n_frames = 0
        self.pool_buf = []

    def step_sample(self, x_t):
        """Process one raw audio sample x_t: scalar tensor."""
        # Input projection — x_t is scalar
        u = self.model.input_proj(x_t.view(1))  # (d_model,)

        # Pooling: accumulate pool_size samples, average, then process
        self.pool_buf.append(u)
        if len(self.pool_buf) < self.pool_size:
            return None
        u = torch.stack(self.pool_buf).mean(dim=0)  # (d_model,)
        self.pool_buf = []

        # Process through S5 blocks (forward SSM only)
        for i, block in enumerate(self.model.blocks):
            Lambda_bar, B_bar, Cmat, D = self.block_params[i]

            # BN1 (eval mode, single frame)
            x_bn = block.bn1(u.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            # Forward SSM recurrence: h = Lambda_bar * h + B_bar @ x
            Bu = torch.einsum('h,nh->n', x_bn.to(torch.complex64), B_bar)
            self.h_states[i] = Lambda_bar * self.h_states[i] + Bu
            y = 2.0 * torch.einsum('n,hn->h', self.h_states[i], Cmat.conj()).real
            y = y + x_bn * D

            # Residual + dropout (no dropout in eval)
            u = u + y

            # BN2 + GLU + residual
            x_bn2 = block.bn2(u.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            u = u + block.glu(x_bn2.unsqueeze(0)).squeeze(0)

        # BN out
        u = self.model.bn_out(u.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # Accumulate for mean pooling
        self.output_sum = self.output_sum + u
        self.n_frames += 1
        return u

    def classify(self):
        mean_out = self.output_sum / max(self.n_frames, 1)
        return self.model.fc(mean_out).unsqueeze(0)

    def streaming_forward(self, x):
        """Full streaming pass over raw waveform x: (16000,)."""
        self.reset()
        for t in range(x.shape[0]):
            self.step_sample(x[t])
        return self.classify()

    def state_size_bytes(self):
        n = self.block_params[0][0].shape[0]
        # Per block: h state (n complex64)
        ssm_state = self.n_blocks * n * 8
        # output_sum (d_model float32) + pool_buf (up to pool_size * d_model float32)
        extra = self.d_model * 4 + self.pool_size * self.d_model * 4
        return ssm_state + extra


# ─── Equivalence Verification ────────────────────────────────────────────────

@torch.no_grad()
def verify_s5_streaming(model, test_loader, device, n_samples=20):
    """Verify S5 forward-only batch == per-sample streaming.

    We compare against a modified batch pass using only the forward SSM,
    since the streaming wrapper drops the backward SSM.
    """
    model.eval()
    streamer = StreamingS5(model)
    streamer.eval()

    max_diff = 0.0
    n_checked = 0

    for waveforms, labels in test_loader:
        for i in range(min(waveforms.shape[0], n_samples - n_checked)):
            wav = waveforms[i].to(device)

            # Forward-only batch pass (manually)
            with torch.no_grad():
                x = model.input_proj(wav.unsqueeze(-1))  # (16000, d_model)
                if model.pool_size > 1:
                    x = model.pool(x.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
                for bi, block in enumerate(model.blocks):
                    Lambda_bar, B_bar, Cmat, D = streamer.block_params[bi]
                    x_bn = block.bn1(x.unsqueeze(0)).squeeze(0)
                    Bu = torch.einsum('th,nh->tn', x_bn.to(torch.complex64), B_bar)
                    # Sequential scan for forward-only
                    T_seq = Bu.shape[0]
                    n = Lambda_bar.shape[0]
                    h = torch.zeros(n, dtype=torch.complex64, device=device)
                    h_all = []
                    for t in range(T_seq):
                        h = Lambda_bar * h + Bu[t]
                        h_all.append(h)
                    h_all = torch.stack(h_all)
                    y = 2.0 * torch.einsum('tn,hn->th', h_all, Cmat.conj()).real
                    y = y + x_bn * D
                    x = x + y
                    x_bn2 = block.bn2(x.unsqueeze(0)).squeeze(0)
                    x = x + block.glu(x_bn2.unsqueeze(0)).squeeze(0)
                x = model.bn_out(x.unsqueeze(0)).squeeze(0)
                batch_logits = model.fc(x.mean(dim=0)).unsqueeze(0)

            # Streaming pass
            stream_logits = streamer.streaming_forward(wav)

            diff = (batch_logits - stream_logits).abs().max().item()
            max_diff = max(max_diff, diff)
            n_checked += 1
            if n_checked >= n_samples:
                break
        if n_checked >= n_samples:
            break

    return max_diff, n_checked


@torch.no_grad()
def verify_e2e_equivalence(model, test_loader, device, n_samples=20):
    """Verify CumsumEndToEnd batch == per-sample streaming."""
    model.eval()
    streamer = StreamingCumsumEndToEnd(model)
    streamer.eval()

    max_diff = 0.0
    n_checked = 0

    for waveforms, labels in test_loader:
        for i in range(min(waveforms.shape[0], n_samples - n_checked)):
            wav = waveforms[i].to(device)  # (16000,)
            batch_logits = model(wav.unsqueeze(0))
            stream_logits = streamer.streaming_forward(wav)
            diff = (batch_logits - stream_logits).abs().max().item()
            max_diff = max(max_diff, diff)
            n_checked += 1
            if n_checked >= n_samples:
                break
        if n_checked >= n_samples:
            break

    return max_diff, n_checked


@torch.no_grad()
def verify_mel_equivalence(model, test_loader, device, n_samples=20):
    """Verify MelCumsumFixed batch == per-hop streaming."""
    model.eval()
    streamer = StreamingMelCumsumFixed(model)
    streamer.eval()

    max_diff = 0.0
    n_checked = 0

    for waveforms, labels in test_loader:
        for i in range(min(waveforms.shape[0], n_samples - n_checked)):
            wav = waveforms[i:i+1].to(device)
            batch_logits = model(wav)

            x = wav.unsqueeze(1)
            x = model.mel_spec(x).squeeze(1)
            x = (x + 1e-8).log()

            stream_logits = streamer.streaming_forward(x)
            diff = (batch_logits - stream_logits).abs().max().item()
            max_diff = max(max_diff, diff)
            n_checked += 1
            if n_checked >= n_samples:
                break
        if n_checked >= n_samples:
            break

    return max_diff, n_checked


# ─── Inference Timing ────────────────────────────────────────────────────────

def benchmark_inference(model, test_loader, device, n_warmup=50, n_iter=200):
    """Time single-sample batch inference for any model."""
    model.eval()
    with torch.no_grad():
        waveforms, _ = next(iter(test_loader))
        wav = waveforms[:1].to(device)

    use_cuda = device.type == 'cuda'

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(wav)

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies = []
    for _ in range(n_iter):
        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                _ = model(wav)
            e.record()
            torch.cuda.synchronize()
            latencies.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(wav)
            latencies.append((time.perf_counter() - t0) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else 0.0
    return np.median(latencies), peak_mem


# ─── Training ────────────────────────────────────────────────────────────────

def train_and_time(model, model_name, train_loader, val_loader, device, epochs=2):
    """Train model, return (model, val_acc, wall_seconds)."""
    if model_name in ('S5', 'BlockDecayS5V2'):
        ssm_keys = ['Lambda', 'log_dt', 'B_re', 'B_im', 'C_re', 'C_im',
                     'log_lambda', 'angles']
        ssm_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and any(k in n for k in ssm_keys)]
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(k in n for k in ssm_keys)]
        optimizer = torch.optim.AdamW([
            {'params': ssm_params, 'lr': 1e-3, 'weight_decay': 0.0},
            {'params': other_params, 'lr': 1e-2, 'weight_decay': 0.05},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0
    best_state = None

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        epoch_secs = time.perf_counter() - epoch_t0
        print(f"  [{model_name}] Epoch {epoch}/{epochs}  loss={train_loss:.4f}  "
              f"train={train_acc:.4f}  val={val_acc:.4f}  {epoch_secs:.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    wall_secs = time.perf_counter() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc, wall_secs


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Cumsum vs S5 benchmark')
    parser.add_argument('--train', action='store_true', help='Train models (for accuracy + training time)')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--smoke', action='store_true', help='Quick test')
    parser.add_argument('--skip_ssm', action='store_true', help='Skip S5/BlockDecay')
    parser.add_argument('--n_warmup', type=int, default=50)
    parser.add_argument('--n_iter', type=int, default=200)
    args = parser.parse_args()

    if args.smoke:
        args.n_warmup = 5
        args.n_iter = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data ──
    print("\nLoading data...")
    noise_wavs = load_noise_wavs(args.data_dir)
    test_ds = SpeechCommandsDataset(args.data_dir, 'testing', noise_wavs=noise_wavs)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=2, pin_memory=True)

    if args.train:
        train_ds = SpeechCommandsDataset(args.data_dir, 'training',
                                          augment=True, noise_wavs=noise_wavs)
        val_ds = SpeechCommandsDataset(args.data_dir, 'validation', noise_wavs=noise_wavs)
        cumsum_tl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
        cumsum_vl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
        ssm_tl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
        ssm_vl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # ── Models ──
    print("\nSetting up models...")
    e2e_model = CumsumEndToEnd(n_freqs=40, window_l1=400, window=20, n_layers=4).to(device)
    mel_model = MelCumsumFixed(n_embed=80, n_layers=8, window=10, hop_length=80, tie_layers=True).to(device)
    melcnn_model = MelCNN(hop_length=80).to(device)
    s5_model = S5Model(d_model=96, n_layers=6, pool=4).to(device) if not args.skip_ssm else None
    bd_model = BlockDecayS5V2(d_model=64, n_layers=6, pool=4, window=80).to(device) if not args.skip_ssm else None

    models = {
        'CumsumE2E': e2e_model,
        'MelCumsumFixed': mel_model,
        'MelCNN': melcnn_model,
    }
    if s5_model: models['S5'] = s5_model
    if bd_model: models['BlockDecayS5V2'] = bd_model

    for name, m in models.items():
        print(f"  {name}: {m.param_count():,} params")

    # ══════════════════════════════════════════════════════════════
    # Scenario 1: Sample-by-sample streaming (CumsumE2E vs S5)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("Scenario 1: Sample-by-sample streaming")
    print(f"{'=' * 60}")

    print("\nCumsumEndToEnd — per-sample streaming equivalence:")
    max_diff, n_checked = verify_e2e_equivalence(e2e_model, test_loader, device)
    passed = max_diff < 1e-3
    print(f"  Checked {n_checked} samples")
    print(f"  Max absolute difference: {max_diff:.2e}  {'PASS' if passed else 'FAIL'}")

    e2e_streamer = StreamingCumsumEndToEnd(e2e_model)
    print(f"  Streaming state size: {e2e_streamer.state_size_bytes() / 1024:.1f} KB")
    print(f"    Layer 1: running_cs + buffer({e2e_model.window_l1}) x {e2e_model.n_freqs} complex")
    print(f"    Layers 2-{e2e_model.n_layers}: running_cs + buffer({e2e_model.window}) x {e2e_model.n_freqs} complex each")

    if s5_model is not None:
        print("\nS5 — per-sample streaming equivalence (forward SSM only):")
        max_diff_s5, n_checked_s5 = verify_s5_streaming(s5_model, test_loader, device)
        passed_s5 = max_diff_s5 < 1e-3
        print(f"  Checked {n_checked_s5} samples")
        print(f"  Max absolute difference: {max_diff_s5:.2e}  {'PASS' if passed_s5 else 'FAIL'}")

        s5_streamer = StreamingS5(s5_model)
        print(f"  Streaming state size: {s5_streamer.state_size_bytes() / 1024:.1f} KB")

    # ══════════════════════════════════════════════════════════════
    # Scenario 2: Training wall time
    # ══════════════════════════════════════════════════════════════
    if args.train:
        print(f"\n{'=' * 60}")
        print(f"Scenario 2: Training wall time ({args.epochs} epochs)")
        print(f"{'=' * 60}")

        train_results = {}

        print(f"\nCumsumEndToEnd (bs=128):")
        e2e_model, e2e_acc, e2e_secs = train_and_time(
            e2e_model, 'CumsumE2E', cumsum_tl, cumsum_vl, device, args.epochs)
        train_results['CumsumE2E'] = (e2e_acc, e2e_secs, 128)

        print(f"\nMelCumsumFixed (bs=128):")
        mel_model, mel_acc, mel_secs = train_and_time(
            mel_model, 'MelCumsumFixed', cumsum_tl, cumsum_vl, device, args.epochs)
        train_results['MelCumsumFixed'] = (mel_acc, mel_secs, 128)

        print(f"\nMelCNN (bs=128):")
        melcnn_model, melcnn_acc, melcnn_secs = train_and_time(
            melcnn_model, 'MelCNN', cumsum_tl, cumsum_vl, device, args.epochs)
        train_results['MelCNN'] = (melcnn_acc, melcnn_secs, 128)

        if s5_model is not None:
            print(f"\nS5 (bs=16):")
            s5_model, s5_acc, s5_secs = train_and_time(
                s5_model, 'S5', ssm_tl, ssm_vl, device, args.epochs)
            train_results['S5'] = (s5_acc, s5_secs, 16)

        if bd_model is not None:
            print(f"\nBlockDecayS5V2 (bs=16):")
            bd_model, bd_acc, bd_secs = train_and_time(
                bd_model, 'BlockDecayS5V2', ssm_tl, ssm_vl, device, args.epochs)
            train_results['BlockDecayS5V2'] = (bd_acc, bd_secs, 16)

        # Summary table
        print(f"\n  {'Model':<20} {'Params':>8}  {'Batch':>5}  {'Wall Time':>10}  {'Per Epoch':>10}  {'Val Acc':>8}")
        print(f"  {'-'*20} {'-'*8}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}")
        for name in ['CumsumE2E', 'MelCumsumFixed', 'MelCNN', 'S5', 'BlockDecayS5V2']:
            if name not in train_results:
                continue
            acc, secs, bs = train_results[name]
            m = models[name]
            print(f"  {name:<20} {m.param_count():>8,}  {bs:>5}  {secs:>8.1f}s  "
                  f"{secs/args.epochs:>8.1f}s  {acc*100:>6.1f}%")

    # ══════════════════════════════════════════════════════════════
    # Scenario 3: Hop-by-hop streaming (MelCumsumFixed)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("Scenario 3: Hop-by-hop streaming (MelCumsumFixed)")
    print(f"{'=' * 60}")

    print("\nMelCumsumFixed — per-hop streaming equivalence:")
    max_diff, n_checked = verify_mel_equivalence(mel_model, test_loader, device)
    passed = max_diff < 1e-4
    print(f"  Checked {n_checked} samples")
    print(f"  Max absolute difference: {max_diff:.2e}  {'PASS' if passed else 'FAIL'}")

    mel_streamer = StreamingMelCumsumFixed(mel_model)
    print(f"  Streaming state size: {mel_streamer.state_size_bytes() / 1024:.1f} KB")
    print(f"    {mel_model.n_layers} layers x (running_cs + buffer({mel_model.window})) x {mel_streamer.n_freqs} complex")
    print(f"  Hop length: {mel_model.mel_spec.hop_length} samples = {mel_model.mel_spec.hop_length/SAMPLE_RATE*1000:.1f}ms")

    # ══════════════════════════════════════════════════════════════
    # Inference timing (all models, batch)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("Inference Latency (single sample, batch mode)")
    print(f"{'=' * 60}")

    print(f"\n  {'Model':<20} {'Params':>8}  {'Latency':>10}  {'Throughput':>12}  {'Peak Mem':>9}")
    print(f"  {'-'*20} {'-'*8}  {'-'*10}  {'-'*12}  {'-'*9}")

    for name in ['CumsumE2E', 'MelCumsumFixed', 'MelCNN', 'S5', 'BlockDecayS5V2']:
        if name not in models:
            continue
        m = models[name]
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        lat, mem = benchmark_inference(m, test_loader, device, args.n_warmup, args.n_iter)
        mem_s = f"{mem:>7.1f}MB" if device.type == 'cuda' else f"{'—':>9}"
        print(f"  {name:<20} {m.param_count():>8,}  {lat:>8.2f}ms  "
              f"{1000/lat:>9.0f} seq/s  {mem_s}")

    # ══════════════════════════════════════════════════════════════
    # Primitive Operation Comparison: torch.cumsum vs Triton fused scan
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("Primitive Operation: torch.cumsum vs Triton fused scan")
    print(f"{'=' * 60}")
    print("\n  cumsum model trains with torch.cumsum (built-in CUDA kernel)")
    print("  S5 model trains with Triton fused scan (single kernel, full autograd)\n")

    from speech_commands import _triton_scan_complex

    print(f"  {'T':>6}  {'N':>4}  {'cumsum':>10}  {'triton_scan':>12}  {'ratio':>8}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*10}  {'-'*12}  {'-'*8}")

    for T, N in [(100, 40), (100, 48), (200, 80), (1000, 48), (4000, 48)]:
        x = torch.randn(1, T, N, device=device)
        gates = torch.complex(
            torch.rand(1, T, N, device=device) * 0.9,
            torch.rand(1, T, N, device=device) * 0.1)
        values = torch.complex(
            torch.randn(1, T, N, device=device),
            torch.randn(1, T, N, device=device))

        # Warmup
        for _ in range(args.n_warmup):
            torch.cumsum(x, dim=1)
            _triton_scan_complex(gates, values)
        torch.cuda.synchronize()

        # Benchmark cumsum
        times_cs = []
        for _ in range(args.n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cumsum(x, dim=1)
            torch.cuda.synchronize()
            times_cs.append((time.perf_counter() - t0) * 1e6)

        # Benchmark Triton scan
        times_ts = []
        for _ in range(args.n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _triton_scan_complex(gates, values)
            torch.cuda.synchronize()
            times_ts.append((time.perf_counter() - t0) * 1e6)

        cs_us = sorted(times_cs)[len(times_cs)//2]
        ts_us = sorted(times_ts)[len(times_ts)//2]
        ratio = ts_us / cs_us
        print(f"  {T:>6}  {N:>4}  {cs_us:>8.1f}us  {ts_us:>10.1f}us  {ratio:>6.1f}x")

    print("\n  Note: cumsum = single CUDA kernel, works on all hardware (GPU/CPU/TPU/edge)")
    print("  Note: Triton scan = fused kernel with complex multiply, requires Triton")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print("  CumsumEndToEnd:   streams per-sample, O(1) update (cumsum), trains with torch.cumsum")
    print("  S5:               streams per-sample, O(1) update (complex multiply), trains with O(log T) parallel scan")
    print("  BlockDecayS5V2:   streams per-sample, O(1) update (complex multiply), trains with cumsum + block scan")
    print("  MelCumsumFixed:   streams per-hop, O(1) update per mel frame")
    print()


if __name__ == '__main__':
    main()
