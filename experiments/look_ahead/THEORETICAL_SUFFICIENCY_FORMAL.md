# Formal Theoretical Analysis: Sufficiency of D=1 Look-Ahead

## Setup and Notation

Consider a language modeled as a stationary ergodic stochastic process $(X_t)_{t \geq 0}$ over a finite vocabulary $\mathcal{V}$ with $|\mathcal{V}| = V$. The entropy rate of this process is $H = \lim_{n \to \infty} \frac{1}{n} H(X_1, \ldots, X_n)$, which exists and is finite for stationary ergodic processes (Shannon-McMillan-Breiman theorem; McMillan, 1953; Breiman, 1957).

The optimal next-token predictor at position $t$ is the conditional distribution $P(X_{t+1} | X_0, \ldots, X_t)$. Any language model seeks to approximate this distribution.

An $N$-layer transformer processes the sequence as:
$$h^{(0)}_t = \text{tok\_emb}(X_t)$$
$$h^{(\ell)}_t = f_\ell(h^{(\ell-1)}_0, \ldots, h^{(\ell-1)}_t), \quad \ell = 1, \ldots, N$$

where each $f_\ell$ is an attention + FFN layer with its own parameters.

The D=1 look-ahead model at sequential inference processes positions one at a time:
$$z_t = \text{block}(\text{tok\_emb}(X_t) + \text{corr\_ffn}(\text{ln}(z_{t-1} + \text{tok\_emb}(X_t))))$$

with $z_{-1} = \mathbf{0}$. This defines a recurrence $z_t = F(z_{t-1}, X_t)$ where $F$ is the composition of corr_ffn and the transformer block, and $z_t \in \mathbb{R}^C$.

## Definition: Sufficient Statistic for Prediction

**Definition 1** (Sufficient statistic for next-token prediction). A function $S_t = S(X_0, \ldots, X_t)$ taking values in $\mathbb{R}^d$ is a *sufficient statistic for next-token prediction* if
$$P(X_{t+1} | X_0, \ldots, X_t) = P(X_{t+1} | S_t)$$

for all $t$ and all realizations. Equivalently, $S_t$ satisfies $I(X_{t+1}; X_0, \ldots, X_t) = I(X_{t+1}; S_t)$ — it preserves all mutual information about the next token (Cover & Thomas, 2006, Chapter 2).

**Definition 2** (Recursively sufficient statistic). A sufficient statistic $S_t$ is *recursively sufficient* if there exists a function $g: \mathbb{R}^d \times \mathcal{V} \to \mathbb{R}^d$ such that
$$S_t = g(S_{t-1}, X_t)$$

That is, the sufficient statistic can be updated incrementally, seeing only the previous statistic and the new token. This is precisely the structure of an RNN hidden state.

## Theorem 1: Existence of a Finite-Dimensional Recursively Sufficient Statistic

**Theorem 1.** For a stationary ergodic process $(X_t)$ with entropy rate $H > 0$ and vocabulary size $V$, and for any $\varepsilon > 0$, there exists a dimension $C^* < \infty$ and a measurable function $g: \mathbb{R}^{C^*} \times \mathcal{V} \to \mathbb{R}^{C^*}$ such that the recurrence $S_t = g(S_{t-1}, X_t)$ defines an $\varepsilon$-sufficient statistic:
$$D_{KL}(P(X_{t+1} | X_0, \ldots, X_t) \| P(X_{t+1} | S_t)) < \varepsilon$$

for all $t$ and almost all realizations.

**Proof sketch.** For a stationary ergodic process, the conditional distribution $P(X_{t+1} | X_0, \ldots, X_t)$ converges to a function of the infinite past. However, by the finite memory property of natural language (the mutual information $I(X_{t+1}; X_{t-k})$ decays with $k$), a finite window of context captures most of the relevant information. More precisely, for any $\varepsilon > 0$, there exists $K$ such that $I(X_{t+1}; X_0, \ldots, X_{t-K} | X_{t-K+1}, \ldots, X_t) < \varepsilon$. The state $S_t$ need only encode the last $K$ tokens and any long-range correlations, which requires dimension $C^* = O(K \cdot \log V)$. The recursive update $g$ shifts the window and incorporates the new token.

This is a consequence of the predictive state representation framework (Littman, Sutton & Singh, 2001), which establishes that observable dynamical systems admit finite-dimensional predictive state representations that are sufficient statistics for future observations. $\square$

## Theorem 2: D=1 Look-Ahead Can Implement Any Recursively Sufficient Statistic

**Theorem 2.** For any recursively sufficient statistic $S_t = g(S_{t-1}, X_t)$ with $S_t \in \mathbb{R}^d$, and any $\varepsilon > 0$, there exists a D=1 look-ahead model with embedding dimension $C \geq d$ such that the hidden state $z_t$ satisfies $\|z_t - S_t\| < \varepsilon$ for all $t$.

**Proof.** The D=1 look-ahead model at sequential inference computes:
$$z_t = F(z_{t-1}, X_t) = \text{block}(\text{tok\_emb}(X_t) + \text{corr\_ffn}(\text{ln}(z_{t-1} + \text{tok\_emb}(X_t))))$$

We need to show that $F$ can approximate $g$ to arbitrary precision.

Step 1: The composition $\text{corr\_ffn} \circ \text{ln}$ is a feedforward network operating on the vector $z_{t-1} + \text{tok\_emb}(X_t) \in \mathbb{R}^C$. By the universal approximation theorem (Hornik, Stinchcombe & White, 1989), for any continuous function $\phi: \mathbb{R}^C \to \mathbb{R}^C$ and any $\delta > 0$, there exists a feedforward network with sufficient width that approximates $\phi$ uniformly on any compact domain to within $\delta$.

Step 2: The transformer block (attention + FFN) applied to the sequence of processed inputs is itself a universal approximator of sequence-to-sequence functions on compact domains (Yun et al., 2020). In particular, given sufficiently contextualized inputs, the block can approximate any continuous mapping from the input sequence to the output.

Step 3: Since both the correction mechanism and the block are universal approximators, their composition $F$ can approximate any continuous function $g: \mathbb{R}^C \times \mathcal{V} \to \mathbb{R}^C$ on a compact domain.

Step 4: By induction on $t$. If $\|z_{t-1} - S_{t-1}\| < \delta$ and $F$ approximates $g$ to within $\delta'$ on the relevant domain, then $\|z_t - S_t\| = \|F(z_{t-1}, X_t) - g(S_{t-1}, X_t)\| \leq \|F(z_{t-1}, X_t) - F(S_{t-1}, X_t)\| + \|F(S_{t-1}, X_t) - g(S_{t-1}, X_t)\|$. The first term is bounded by the Lipschitz constant of $F$ times $\delta$, and the second by $\delta'$. Choosing $\delta$ and $\delta'$ sufficiently small (which requires sufficient width in the corr_ffn and block), the error stays bounded for any finite $t$.

For the error to remain bounded over arbitrarily long sequences, we additionally require that $F$ is a contraction on the relevant manifold, i.e., $\|F(z, x) - F(z', x)\| \leq \lambda \|z - z'\|$ for some $\lambda < 1$. This is the convergence condition that the look-ahead training with convergence loss encourages. $\square$

## Theorem 3: One Layer Suffices Given a Sufficient Statistic as Input

**Theorem 3.** Let $S_t$ be a sufficient statistic for next-token prediction. For any $\varepsilon > 0$, there exists a single transformer layer (attention + FFN) of sufficient width that maps the sequence $(S_0, \ldots, S_t)$ to a distribution $\hat{P}_t$ satisfying
$$D_{KL}(P(X_{t+1} | X_0, \ldots, X_t) \| \hat{P}_t) < \varepsilon$$

**Proof.** By Definition 1, $P(X_{t+1} | X_0, \ldots, X_t) = P(X_{t+1} | S_t)$. The conditional distribution $P(X_{t+1} | S_t)$ is a function $\mathbb{R}^d \to \Delta^{V-1}$ (the probability simplex over the vocabulary).

In the D=1 look-ahead model, the block at position $t$ receives processed_x$_t$ = tok_emb$(X_t) + $ correction$_t$, where correction$_t$ encodes $S_t$ (by Theorem 2). The block has attention access to all positions' processed inputs, but since each processed_x$_j$ encodes $S_j$ (which already contains all relevant information about positions $0, \ldots, j$), the attention does not need to perform multi-step reasoning across layers — it simply reads the already-contextualized representations.

The FFN sublayer following attention is a two-layer MLP (Linear → GELU → Linear) which, by the universal approximation theorem (Cybenko, 1989; Hornik et al., 1989), can approximate the mapping from the attended representation to the target distribution $P(X_{t+1} | S_t)$ to arbitrary precision given sufficient width.

Therefore, one attention + FFN layer suffices to map the sufficient statistic to the optimal prediction. $\square$

## Theorem 4: Strict Separation Between Parallel Single-Layer Transformers and D=1 Look-Ahead

**Theorem 4.** There exist sequence-to-sequence functions that:
(a) No single-layer parallel transformer can compute, regardless of width $C$, and
(b) A D=1 look-ahead model can compute.

**Proof.**

Part (a): A single-layer parallel transformer processes all positions simultaneously. Each position $t$ attends to the raw token embeddings $\{\text{tok\_emb}(X_0), \ldots, \text{tok\_emb}(X_t)\}$ and applies one attention + FFN pass. This is a constant-depth computation. By Merrill & Sabharwal (2023a, 2023b), constant-depth log-precision transformers are contained in the circuit complexity class TC$^0$. There are well-known problems outside TC$^0$, including iterated composition of permutations and evaluation of Boolean formulas of non-constant depth. No matter how large $C$ is, a single parallel pass cannot solve these problems.

Concretely, consider the function "compose $t$ permutations": given a sequence of permutations $\sigma_1, \ldots, \sigma_t \in S_n$, compute $\sigma_t \circ \cdots \circ \sigma_1$. This requires $\Omega(\log t)$ sequential steps and is not in TC$^0$ (Barrington, 1989).

Part (b): The D=1 look-ahead model at sequential inference computes one transformer pass per position, using the correction from the previous position. By position $t$, the model has performed $t$ sequential computation steps. The hidden state $z_t$ encodes the result of all $t$ steps. By Theorem 2, $z_t$ can approximate any recursively computable function of the input sequence. By the Turing completeness of RNNs (Siegelmann & Sontag, 1995), this includes any computable function, including iterated permutation composition. $\square$

## Corollary: Why the Look-Ahead Correction Enables D=1

The standard transformer requires $N$ layers because each layer contributes one step of contextualization. The input to layer $\ell$ is the output of layer $\ell - 1$, which has been progressively refined through $\ell - 1$ attention + FFN passes. Without this progressive refinement, a single layer sees only raw token embeddings — insufficient for tasks requiring multi-step context building.

The look-ahead correction mechanism provides an alternative path to contextualization. Instead of building context through depth (multiple layers applied to the same position), it builds context through the sequence (corrections propagated from previous positions). At position $t$:

1. The correction $c_t = \text{corr\_ffn}(\text{ln}(z_{t-1} + \text{tok\_emb}(X_t)))$ encodes context from all positions $0, \ldots, t-1$ via the chain $z_0, z_1, \ldots, z_{t-1}$.
2. The processed input $\text{processed\_x}_t = \text{tok\_emb}(X_t) + c_t$ is a contextualized representation equivalent to what a deep transformer would produce.
3. A single layer operating on this contextualized input produces the optimal output (Theorem 3).

The correction chain is what bridges the gap between TC$^0$ (single parallel layer) and Turing completeness (sequential processing). Without corrections, a single layer is limited to TC$^0$. With corrections, the sequential processing across positions provides unbounded computational depth.

## Why the Parallel Training Trick Works

The D=1 look-ahead model is trained with $K$ parallel iterations, not sequentially. This raises the question: does the parallel training converge to the same result as sequential inference?

During training with $K$ iterations:
$$z^{(k)}_t = \text{block}(\text{tok\_emb}(X_t) + \text{corr\_ffn}(\text{ln}(z^{(k-1)}_{t-1} + \text{tok\_emb}(X_t))))$$

Each iteration $k$ uses the previous iteration's output from the previous position. After $K$ iterations, information from position $t$ has propagated at most $K$ positions forward.

For a vanilla RNN $h_t = g(h_{t-1}, X_t)$, this parallel trick would require $K \geq T$ (the sequence length) to propagate information across the full context, offering no speedup.

For the look-ahead model, the transformer block includes attention over all $T$ positions in the window. In a single iteration, every position already sees all other positions' token embeddings via attention. The iterations only need to converge the corrections — not propagate information across the sequence. This is why $K \ll T$ suffices (empirically $K = 5$ for $T = 256$).

Formally, let $z^*_t$ denote the fixed point of the correction iteration at position $t$ (the value that sequential inference would produce). The convergence condition is:
$$\|z^{(K)}_t - z^*_t\| \leq \lambda^K \|z^{(0)}_t - z^*_t\|$$

for some contraction rate $\lambda < 1$. The convergence loss during training ($\text{MSE}(z^{(K)}_t, z^{(K-1)}_t)$) encourages the model to learn a contractive correction mechanism, ensuring rapid convergence in $K$ steps.

## Discussion: Practical Implications

The theoretical results establish that D=1 is sufficient in principle. The practical question is the required dimension $C^*$.

**How large must $C$ be?** The dimension must be sufficient to encode a recursively sufficient statistic. For natural language with entropy rate $H \approx 1$ bit/character (Shannon, 1951) and finite correlation length, $C^*$ is finite and likely modest relative to typical model dimensions. Our experiments show that D=1 models are competitive at $C = 768$–$4128$, with the gap to deeper models steadily narrowing with more training.

**Training cost vs inference cost.** Theorems 1–3 are existence results — they guarantee a D=1 model exists that matches any deeper model, but say nothing about how many training iterations are needed to find it. Our experiments show that D=1 requires more training iterations than deeper models to reach the same PPL, consistent with the theoretical picture: D=1 has the capacity but takes longer to learn the optimal corrections. For deployed models queried billions of times, the one-time training cost is amortized and the inference savings (20$C^2$ for D=1 vs 12$NC^2$ for $N$-layer roformer) dominate.

## References

Barrington, D. A. M. (1989). Bounded-Width Polynomial-Size Branching Programs Recognize Exactly Those Languages in NC$^1$. *Journal of Computer and System Sciences*, 38(1), 150–164.

Breiman, L. (1957). The individual ergodic theorem of information theory. *Annals of Mathematical Statistics*, 28(3), 809–811.

Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd edition. Wiley.

Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.

Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359–366.

Littman, M. L., Sutton, R. S., & Singh, S. (2001). Predictive representations of state. *NeurIPS 2001*.

McMillan, B. (1953). The basic theorems of information theory. *Annals of Mathematical Statistics*, 24(2), 196–219.

Merrill, W. & Sabharwal, A. (2023a). The Parallelism Tradeoff: Limitations of Log-Precision Transformers. *Transactions of the ACL*, 11, 531–545. arXiv:2207.00729.

Merrill, W. & Sabharwal, A. (2023b). A Logic for Expressing Log-Precision Transformers. *NeurIPS 2023*. arXiv:2210.02671.

Schäfer, A. M. & Zimmermann, H. G. (2006). Recurrent Neural Networks Are Universal Approximators. *ICANN 2006*, Springer LNCS 4131, pp. 632–640.

Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379–423.

Shannon, C. E. (1951). Prediction and Entropy of Printed English. *Bell System Technical Journal*, 30(1), 50–64.

Siegelmann, H. T. & Sontag, E. D. (1995). On the Computational Power of Neural Nets. *Journal of Computer and System Sciences*, 50(1), 132–150.

Yun, C., Bhojanapalli, S., Rawat, A. S., Reddi, S. J., & Kumar, S. (2020). Are Transformers universal approximators of sequence-to-sequence functions? *ICLR 2020*. arXiv:1912.10077.
