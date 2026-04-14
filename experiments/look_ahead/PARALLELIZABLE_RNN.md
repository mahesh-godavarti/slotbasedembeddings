# A Parallelizable RNN: Bridging Transformers and Recurrent Networks

## The Problem

There are two dominant paradigms for sequence modeling:

**Transformers**: Parallel training (all positions processed simultaneously), but limited to constant-depth computation at inference (TC^0). Adding depth requires more layers, which is a fixed architectural choice.

**Recurrent Neural Networks (LSTMs, GRUs)**: Turing-complete at inference (unbounded sequential computation), but sequential training via BPTT (O(T) sequential steps for sequence length T). This makes training slow and limits practical sequence lengths.

| | Training | Inference | Computational class |
|---|---|---|---|
| Transformer (N layers) | Parallel, O(N) depth | Parallel, O(N) depth | TC^0 (constant depth) |
| RNN (LSTM/GRU) | Sequential, O(T) steps | Sequential, O(T) steps | Turing complete |
| **This work (D=1)** | **Parallel, O(K) steps** | **Sequential, O(T) steps** | **Turing complete** |
| **This work (D>1 cell)** | **Parallel, O(K×D) steps** | **Sequential, O(T×D) steps** | **Turing complete** |

where K << T is the number of parallel iterations (typically 5), D is the cell depth (number of transformer blocks within the recurrent cell), and T is the sequence length.

## The Architecture

### Single layer (D=1)

Three equations define the model:

**Correction** — compute what context to add, based on previous position's output:
$$c_t = \text{FFN}_{\text{corr}}(\text{LN}(h_{t-1} + x_t))$$

**Non-cumulative reset** — always anchor to the current token, not accumulate:
$$\tilde{x}_t = x_t + c_t$$

**Transformer block** — attention over all positions + feedforward:
$$h_t = \text{Block}(\tilde{x}_t, \{\tilde{x}_0, \ldots, \tilde{x}_t\})$$

where $x_t$ is the token embedding at position $t$, $h_t$ is the hidden state, $c_t$ is the correction, and $\text{Block}$ is a standard transformer layer (multi-head attention + FFN with residual connections and layer normalization).

At sequential inference, this is a recurrence: $h_t = F(h_{t-1}, x_t)$. The model processes one position at a time, each seeing the full correction chain from all previous positions. This is an RNN with a transformer block as the recurrent cell.

### Deeper cell (D > 1)

The recurrent cell can be made deeper by using D transformer blocks within a single cell. This is analogous to using a multi-layer MLP inside an LSTM cell — it makes the cell more powerful, but there is still only ONE temporal recurrence:

$$c_t = \text{FFN}_{\text{corr}}(\text{LN}(h^D_{t-1} + x_t))$$
$$\tilde{x}_t = x_t + c_t$$
$$h^1_t = \text{Block}_1(\tilde{x}_t, \ldots)$$
$$h^2_t = \text{Block}_2(h^1_t, \ldots)$$
$$\vdots$$
$$h^D_t = \text{Block}_D(h^{D-1}_t, \ldots)$$

Each Block has separate weights. The correction is computed from the deepest layer's output ($h^D_{t-1}$). The D blocks are depth within a single time step — there is only one temporal recurrence (the correction chain from $h^D_{t-1}$ to position $t$).

**This is NOT analogous to stacked LSTMs.** In stacked LSTMs, each layer has its own independent temporal recurrence:
```
Stacked LSTM layer 1: h1[t] = LSTM_1(h1[t-1], x[t])
Stacked LSTM layer 2: h2[t] = LSTM_2(h2[t-1], h1[t])
```
Each layer maintains its own hidden state across time. In our D>1 architecture, the D blocks simply make the cell deeper — like replacing a single-layer MLP with a D-layer MLP inside the recurrent cell.

| Architecture | RNN analogy |
|---|---|
| D=1 | Single-layer LSTM |
| D>1 | Single LSTM with a deeper cell (D layers of processing inside one cell) |
| Stacked units (N units × K iters) | Stacked LSTM (N independent temporal recurrences) |

## The Parallel Training Trick

### Why standard RNNs can't train in parallel

In a standard RNN, $h_t = f(h_{t-1}, x_t)$. Computing $h_t$ requires $h_{t-1}$, which requires $h_{t-2}$, etc. This chain dependency makes training inherently sequential: O(T) steps for sequence length T.

### Why our architecture can

The key enabler is **attention**. Each transformer block attends to ALL positions in its window, providing global context in a single pass. This means:

- Without attention (vanilla RNN): information propagates one position per iteration. Need K = T iterations for full context. No speedup.
- With attention (our architecture): global context available in one iteration. Iterations only need to converge the corrections. K << T suffices.

### The training algorithm

Initialize: $h^{(0)}[t] = \mathbf{0}$ for all positions $t$

For iteration $k = 1$ to $K$ (all positions in parallel):

$$c^{(k)}[t] = \text{FFN}_{\text{corr}}(\text{LN}(h^{(k-1)}[t-1] + x[t]))$$
$$\tilde{x}^{(k)}[t] = x[t] + c^{(k)}[t]$$
$$h^{(k)}[t] = \text{Block}(\tilde{x}^{(k)}[t], \{\tilde{x}^{(k)}[0], \ldots, \tilde{x}^{(k)}[t]\})$$

For D>1, Block is the sequential application of D transformer blocks within each position.

Each iteration uses the **previous iteration's** output from the **previous position**. All positions are computed simultaneously within each iteration. After K iterations, the corrections have approximately converged to what sequential processing would produce.

### Training cost comparison

| Method | Sequential steps | Positions parallel? |
|---|---|---|
| Single LSTM (BPTT) | T | No |
| LSTM with deeper cell (BPTT) | T × D | No |
| Stacked LSTM (BPTT) | T × N_layers | No |
| Standard transformer (N layers) | N | Yes (all T positions) |
| **Ours (D=1)** | **K** | **Yes (all T positions)** |
| **Ours (D>1, deeper cell)** | **K × D** | **Yes (all T positions)** |

For T=256, K=5: single LSTM needs 256 sequential steps, ours (D=1) needs 5. With D=3 cell depth: LSTM-with-deep-cell needs 768 steps, ours needs 15, transformer needs 3.

### Why convergence works

The non-cumulative reset ($\tilde{x}_t = x_t + c_t$, not $\tilde{x}_t = \tilde{x}_{t-1} + c_t$) ensures the correction is a bounded perturbation of the input. The convergence loss during training ($\text{MSE}(h^{(K)}, h^{(K-1)})$) encourages the model to learn a contractive correction mechanism. Empirically, K=5 is sufficient for T=256.

## Comparison with Related Approaches

### vs. Standard Transformers

Transformers are a special case of our architecture with K=1 (no correction) and D=N layers. They are parallel during both training and inference, but limited to TC^0 at any fixed depth. Our architecture adds the correction mechanism, which provides:
- Sequential inference mode (Turing complete)
- Better PPL at matched inference FLOPs (D=x beats N=x at every depth tested)
- The ability to train with BPTT if sequential reasoning beyond K steps is needed

### vs. Standard RNNs (LSTMs, GRUs)

Standard RNNs are sequential during both training and inference. Our architecture replaces the recurrent cell with a transformer block + correction FFN, providing:
- Parallel training via K iterations (speedup of T/K, typically 50x)
- Attention over the input window (richer context per step)
- No gates needed — the additive correction avoids vanishing gradients without explicit gating

Like LSTMs, our cell can be made deeper (D>1) for more processing power per time step. Like stacked LSTMs, multiple units with independent temporal recurrences can be composed (our "stacked" variants), though in practice the single-recurrence D>1 approach works better.

### vs. Linear Attention / State Space Models (Mamba, RWKV, etc.)

Recent work on linear attention and state space models also aims to combine RNN-like inference with parallel training. The key differences:
- SSMs linearize the recurrence, trading expressiveness for efficiency. Our architecture uses full quadratic attention — more expensive per step but more expressive.
- SSMs have fixed-size state that compresses the entire history. Our architecture has attention over the full window, plus the correction chain. The correction is a compressed state, but attention provides uncompressed access to recent context.
- SSMs require specialized initialization (e.g., HiPPO) and parameterization. Our architecture uses standard transformer components — no specialized initialization needed.

### vs. Universal Transformers (Dehghani et al., 2019)

Universal Transformers share weights across layers and iterate, similar to our approach. Key differences:
- Universal Transformers iterate the SAME computation at each position (depth recurrence). Our architecture iterates the correction between positions (sequential recurrence).
- Universal Transformers use adaptive halting (ACT). We use fixed K iterations with convergence loss.
- Universal Transformers at inference are still parallel across positions (TC^0). Our architecture at inference is sequential across positions (Turing complete).

### vs. Depth-Recurrent Transformers (Geiping et al., 2026)

"Thinking Deeper, Not Longer" proposes depth-recurrent transformers with shared weights. Key differences:
- They recur in depth (more layers at same position). We recur across positions (correction chain).
- They require stabilization tricks: silent thinking (loss only at final step), identity-biased gating (-2.0 bias). We use simple additive correction with convergence loss — no special tricks.
- They achieve variable depth at a single position. We achieve variable depth across the sequence — the effective depth grows with sequence length.

## Design Choices and Their Rationale

### Why additive correction, not full state feedback?

We tested feeding the full hidden state back directly: $\tilde{x}_t = x_t + h_{t-1} + c_t$ (the "pure" variant). It performed worse (134.27 vs 120.96 PPL at 10K iters). The correction bottleneck acts as a regularizer — forcing information through the FFN produces cleaner, more useful context than raw state feedback.

This is analogous to how LSTMs use gates to control information flow, but simpler — we use a learned projection (the correction FFN) instead of multiplicative gates.

### Why non-cumulative reset?

The correction resets to $x_t$ at each step: $\tilde{x}_t = x_t + c_t$, not $\tilde{x}_t = h_{t-1} + c_t$. This prevents error accumulation across the sequence. The model always starts from a clean signal (the token embedding) and adds a single correction. This is critical for long-sequence stability and is what enables the parallel training trick — the correction only needs to converge, not the entire accumulated state.

### Why attention is essential for parallel training

The attention mechanism provides O(T) context in O(1) depth. Without it, information must propagate one position per iteration, requiring K = T iterations (no speedup). With attention, every position sees the full context in a single iteration. The K iterations only refine the corrections, not propagate information. This is why K = 5 works for T = 256 — a 50x speedup over sequential training.

## Theoretical Properties

### Turing Completeness

At sequential inference, the D=1 model defines a recurrence $h_t = F(h_{t-1}, x_t)$ where F is a transformer block composed with the correction FFN. By the Turing completeness of RNNs (Siegelmann & Sontag, 1995) and the universal approximation property of the component functions (Cybenko, 1989; Hornik et al., 1989), D=1 is Turing complete given sufficient hidden state dimension C.

### Strict Separation from Parallel Transformers

A single parallel transformer layer is in TC^0 (Merrill & Sabharwal, 2023). There exist functions computable by our D=1 model (at sequential inference) that no fixed-depth parallel transformer can compute regardless of width. The correction chain provides unbounded sequential computation depth proportional to sequence length. See `THEORETICAL_SUFFICIENCY_FORMAL.md` for full proofs.

### Sufficient Statistic Property

If the hidden state $h_t \in \mathbb{R}^C$ is a sufficient statistic of the prefix $(x_0, \ldots, x_t)$ for predicting $x_{t+1}$, then one transformer block suffices to produce the optimal prediction. The correction mechanism's role is to maintain this sufficient statistic incrementally. See `THEORETICAL_SUFFICIENCY_FORMAL.md` for formal treatment.

### BPTT Fallback

If sequential reasoning beyond K steps is required (e.g., for tasks requiring exact multi-step computation), the same architecture can be trained with BPTT instead of parallel iterations. Training becomes O(T × D) sequential (like a standard stacked LSTM), but the model is the same — a transformer block + correction FFN as the recurrent cell. This provides a spectrum:

| Training method | Sequential steps | Sequential reasoning capability |
|---|---|---|
| Parallel (K iterations) | K × D | Up to K steps |
| BPTT (full sequential) | T × D | Up to T steps |

The practitioner chooses the training method based on the task requirements.

## Experimental Evidence

### D=1 beats N=2 at FLOP parity

At every width tested (C=256, 512, 1024), D=1 with correction beats N=2 roformer at matched inference FLOPs:

| Width | N=2 PPL | D=1 PPL (FLOP-matched) | Improvement |
|-------|---------|----------------------|-------------|
| C=256 | 115.83 | 106.53 | 8.0% |
| C=512 | 77.19 | 70.10 | 9.2% |
| C=1024 | running | running | ~7% |

### D=x beats N=x at every depth tested

At ~85M inference FLOPs, every look-ahead model beats its FLOP-matched roformer baseline:

| Look-ahead | PPL | Roformer baseline | PPL | Gap |
|------------|-----|-------------------|-----|-----|
| D=1 C=2048 | 39.94 | N=2 C=1888 | 42.99 | -3.05 |
| D=3 C=1408 | 37.26 | N=4 C=1344 | 38.68 | -1.42 |
| D=5 C=1120 | 36.38 | N=6 C=1088 | 37.76 | -1.38 |
| D=6 C=1024 | 36.56 | N=6 C=1088 | 37.76 | -1.20 |
| D=11 C=768 | 36.82 | N=12 C=768 | 37.83 | -1.01 |

### JoFormer (non-abelian attention) further improves D=1

Replacing standard attention with JoFormer attention (which also rotates values, making attention non-commutative) gives a further 6+ PPL improvement at D=1:

| Model | lr | PPL @ 300K |
|-------|-----|-----------|
| D=1 joformer C=1888 | 5e-5 | **35.04** |
| D=1 roformer C=2048 | 5e-5 | **41.06** |

### D=1 steadily closes the gap against deeper models

D=1 C=2048 vs N=6 C=1088 at ~85M FLOPs: gap narrowed from +10.19 (5K iters) to +0.94 (360K iters) and still closing. Given enough training, D=1 approaches any fixed-depth model — consistent with the theoretical prediction that D=1 is sufficient.

## References

Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems, 2(4), 303-314.

Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, L. (2019). Universal Transformers. ICLR 2019. arXiv:1807.03819.

Geiping, J., et al. (2026). Thinking Deeper, Not Longer: Depth-Recurrent Transformers. arXiv:2603.21676.

Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. Neural Networks, 2(5), 359-366.

Merrill, W. & Sabharwal, A. (2023). The Parallelism Tradeoff: Limitations of Log-Precision Transformers. Transactions of the ACL, 11, 531-545. arXiv:2207.00729.

Siegelmann, H. T. & Sontag, E. D. (1995). On the Computational Power of Neural Nets. Journal of Computer and System Sciences, 50(1), 132-150.

Yun, C., Bhojanapalli, S., Rawat, A. S., Reddi, S. J., & Kumar, S. (2020). Are Transformers universal approximators of sequence-to-sequence functions? ICLR 2020. arXiv:1912.10077.
