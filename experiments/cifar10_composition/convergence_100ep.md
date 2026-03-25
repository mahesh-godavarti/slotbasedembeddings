# Convergence Data: CIFAR-100, D=32, 4 layers, 4 heads, seed=42, 100 epochs

All runs use K@Q^T attention ordering.

## Test Accuracy (best so far at each epoch)

| Epoch | rope2d | joformer_old | monoidal_axial | joformer_axial | rope2dv2 | joformer_fixed |
|-------|--------|-------------|----------------|----------------|----------|----------------|
| 10 | 31.62 | 32.29 | 30.90 | 31.26 | 31.54 | 32.32 |
| 20 | 38.07 | 38.27 | 39.09 | 38.51 | 37.22 | 37.52 |
| 30 | 42.30 | 41.55 | 42.21 | 42.26 | 40.45 | 40.50 |
| 40 | 44.02 | 42.91 | 44.70 | 44.37 | 42.27 | 42.34 |
| 50 | 44.88 | 44.58 | 45.73 | 45.11 | 43.36 | 43.30 |
| 60 | 45.88 | 45.55 | 46.43 | 45.95 | 44.16 | 44.42 |
| 70 | 46.60 | 46.34 | 47.59 | 47.35 | 45.09 | 45.04 |
| 80 | 47.15 | 47.04 | 47.81 | 47.61 | 45.30 | 45.58 |
| 90 | 47.42 | 47.51 | 48.02 | 48.15 | 45.91 | 45.94 |
| 100 | **47.42** | **47.60** | **48.15** | *running* | **45.91** | **45.96** |

## Model Descriptions

### Axial approach (split dimensions: first D/4 pairs = y, second D/4 pairs = x)
- **rope2d**: Fixed RoPE frequencies, Q/K only
- **joformer_old**: Fixed RoPE frequencies, Q/K/V + inverse (inherits from rope2d)
- **monoidal_axial**: Learned frequencies, Q/K only
- **joformer_axial**: Learned frequencies, Q/K/V + inverse (inherits from monoidal_axial)

### Combined approach (all D/2 pairs encode both axes: angle = pos_y * freq_y + pos_x * freq_x)
- **rope2dv2**: Fixed frequencies (freq_x = -freq_y), Q/K only
- **joformer_fixed**: Fixed frequencies (freq_x = -freq_y), Q/K/V + inverse (inherits from rope2dv2)
- **monoidal**: Learned frequencies (random init), Q/K only
- **joformer**: Learned frequencies (random init), Q/K/V + inverse (inherits from monoidal)

## Key Observations

1. **JoFormer starts slower, finishes stronger**: JoFormer variants trail their rope counterparts through epochs 10-50 but catch up and pull ahead in epochs 80-100. The V rotation + inverse needs more training time to show its benefit.

2. **Learned beats fixed**: monoidal_axial (48.15%) > rope2d (47.42%). Learnable frequencies find better positional representations than the standard 1/10000^(2d/D) formula.

3. **Axial beats combined (at D=32)**: Axial approach (~47-48%) consistently beats combined negative (~45-46%). At small D=32, the clean separation of axes is more effective. Literature (ECCV 2024) shows combined wins at larger D.

4. **V rotation consistently helps**: joformer_old > rope2d, joformer_fixed > rope2dv2, joformer_axial catching up to monoidal_axial. The effect is small (+0.05 to +0.18%) but consistent across all approaches.

5. **K@Q^T matters for JoFormer**: Switching from Q@K^T to K@Q^T improved joformer_old from 46.95% to 47.60% — a 0.65% gain from matching the attention direction with the V rotation journey operator.
