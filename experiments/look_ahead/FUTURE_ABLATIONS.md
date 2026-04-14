# Future Ablations: Correction Mechanism Variants

## RNN-style correction

Replace the FFN correction with a classical RNN combination rule:

```
Current (8C² params):
    corr[t] = FFN(LN(h[t-1] + x[t]))
    px[t] = x[t] + corr[t]
    h[t] = Block(px[t], {px[0], ..., px[t]})

Proposed RNN-style (2C² params):
    px[t] = tanh(W_h · h[t-1] + W_x · x[t] + b)
    h[t] = Block(px[t], {px[0], ..., px[t]})
    y[t] = softmax(W_y · LN(h[t]) + b_y)
```

### What this tests

1. Separate projections (W_h · h[t-1] + W_x · x[t]) vs shared input (FFN(LN(h[t-1] + x[t])))
2. Full replacement (px = tanh(mix)) vs additive reset (px = x + corr)
3. 2C² vs 8C² — 4x cheaper correction. At matched FLOPs, C can be larger.

### What we already know

- SA correction (FFN + cross-attention, 12C²): tied with FFN-only (8C²). More complex didn't help.
- Tied FFN (sharing block's FFN, 0 extra params): ~9 PPL worse than separate FFN. Too little capacity.
- Pure variant (x + h[t-1] + corr): worse than additive (x + corr). Direct state feedback hurts.
- Concat (24C²): slightly better than add (20C²) at D=1, but not at D>1. Extra params not worth it.

### Hypothesis

The FFN at 8C² is likely the sweet spot. The RNN-style 2C² may be too little capacity (like tied FFN), while SA at 12C² is too much (like concat). But worth testing because:
- If RNN-style works nearly as well, we have a simpler, cheaper architecture
- The separate projections for h[t-1] and x[t] might be better than adding them
- At matched FLOPs, the wider C from saving 6C² could compensate

### LSTM-style correction

Could also try gated correction:

```
LSTM-style (4C² params):
    gate[t] = sigmoid(W_g · [h[t-1], x[t]] + b_g)
    corr[t] = tanh(W_c · [h[t-1], x[t]] + b_c)
    px[t] = x[t] + gate[t] * corr[t]
    h[t] = Block(px[t], {px[0], ..., px[t]})
```

This adds a gate to control how much correction to apply. At 4C² it's between RNN-style (2C²) and our FFN (8C²).

### GRU-style correction

```
GRU-style (4C² params):
    z[t] = sigmoid(W_z · [h[t-1], x[t]] + b_z)       # update gate
    corr[t] = tanh(W_c · [z[t] * h[t-1], x[t]] + b_c) # gated correction
    px[t] = x[t] + corr[t]
    h[t] = Block(px[t], {px[0], ..., px[t]})
```

### Priority

Low. The current FFN correction works well and we have extensive ablation data showing it's the best variant tested. These are "what if" experiments for completeness, not urgent.
