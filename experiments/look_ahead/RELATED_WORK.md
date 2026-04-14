# Related Work: Width vs Depth in Transformers

## The Width-Depth Tradeoff

The question of optimal width-to-depth ratio in transformers has been studied extensively. Kaplan et al. (2020) established neural scaling laws showing that model performance depends primarily on total parameters and training compute, but the allocation between width and depth matters for efficiency. Their analysis suggested relatively wide, shallow architectures are more compute-efficient than commonly assumed.

More recently, Fahim & Karim (2026) directly challenge the assumption that deeper transformers are superior in "The Depth Delusion: Why Transformers Should Be Wider, Not Deeper" [arXiv:2601.20994]. They show that increasing width yields better performance-to-parameter efficiency tradeoffs than stacking more layers, contradicting the scaling trends in GPT-3, PaLM, and LLaMA. Their key contribution is a critical depth formula:

- **D_crit ~ W^0.44** (sublinear in width)
- Practical rule: **never exceed D ~ 2.5 × ln(W)**
- Beyond D_crit, adding layers *increases* loss despite adding parameters
- Optimal depth scales as D* ~ C^0.12, width as W* ~ C^0.34 -- width should grow **2.8x faster** than depth

Critical depth estimates at specific widths:

| Width | D_crit |
|-------|--------|
| 512 | ~15 |
| 1024 | ~17 |
| 1536 | ~18 |

They find existing models are substantially over-deep:

| Model | Depth | Width | D_crit | Over-deep factor |
|-------|-------|-------|--------|-----------------|
| GPT-3 175B | 96 | 12,288 | ~23 | 4.25x |
| PaLM 540B | 118 | 18,432 | ~24 | 5.0x |
| LLaMA-2 70B | 80 | 8,192 | ~22 | 3.7x |
| Mistral 7B | 32 | 4,096 | ~20 | 1.6x |

The mechanism is gradient starvation: gradients decay exponentially through depth with persistence length tau(W) ~ W^0.44. Beyond D_crit, early layers cannot learn effectively. Validated across 30 architectures from 27M to 7.1B parameters (R^2 = 0.922).

At 7B scale, a 32-layer x 4096-wide model (6.92B params) beat a 64-layer x 2816-wide model (7.08B params) by 0.12 nats despite having fewer parameters.

### Inference-Efficient Scaling Laws

Li et al. (2026) in "Scaling Laws Meet Model Architecture" [ICLR 2026, arXiv:2510.18245] optimize specifically for inference efficiency. Their "Surefire" architectures achieve 42% higher inference throughput and 2.1% higher accuracy vs LLaMA-3.2 under identical training budgets. At 1B params: 2560 hidden size, 16 layers; at 3B params: 4096 hidden size, 28 layers. Inference time correlates more strongly with depth than width -- for autoregressive generation, each token must pass through every layer sequentially, making depth the latency bottleneck.

Yehudai et al. (2025) provide theoretical support for width over depth in "Depth-Width Tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers" [arXiv:2503.01805], proving that with linear width, constant depth suffices for solving graph-based problems, with significant speedups through hardware parallelization.

## Depth is Underutilized

Several recent papers demonstrate that current deep transformers do not efficiently use their depth.

### Linearity of Deep Layers

Razzhigaev et al. (2024) in "Your Transformer is Secretly Linear" [arXiv:2405.12250] show that adjacent layers have Procrustes similarity scores of 0.99 -- near-perfect linear relationships. Removing or linearly approximating the most linear blocks barely affects performance. The norm of each block's contribution to the residual stream is remarkably low, suggesting most layers perform near-identity transformations.

### Depth Efficiency Analysis

Csordas, Manning & Potts (2025) show in "Do Language Models Use Their Depth Efficiently?" [NeurIPS 2025, arXiv:2505.13898] that second-half layers contribute substantially less than first-half layers, with minimal impact from removing them. They find no evidence of compositional depth use in multihop reasoning -- layers replicate operations across depths rather than performing fundamentally different computations.

Hu, Zhou & Zhang (2025) confirm this in "What Affects the Effective Depth of Large Language Models?" [arXiv:2512.14064], showing that while effective layer count increases with model size, the effective depth ratio remains stable across scales. Models fail to increase effective depth even with harder tasks, indicating current LLMs underuse available depth.

Petty et al. (2024) reach similar conclusions in "The Impact of Depth on Compositional Generalization in Transformer Language Models" [NAACL 2024, arXiv:2310.19956], finding that deeper models show diminishing returns in compositional generalization and can be made shallower without sacrificing performance within fixed parameter budgets.

### Attention Redundancy

Men et al. (2024) in "What Matters in Transformers? Not All Attention is Needed" [arXiv:2406.15786] show that LLaMA-2 70B can drop 32 of 80 attention layers with only 0.1% performance decrease. Dropping 48/80 still works. 50% attention layer removal gives 48.4% speedup with only 2.4% performance drop. Deeper layers are more redundant than shallower ones.

## Making Shallower Models Work

Several approaches have been proposed to reduce depth while maintaining quality.

Saratchandran, Teney & Lucey (2025) propose "Leaner Transformers: More Heads, Less Depth" [arXiv:2505.20802], proving that increasing attention heads improves layer conditioning, enabling transformers with more heads but fewer layers. They achieve 30-50% parameter reduction while maintaining or improving accuracy across vision, language, and long-sequence reasoning tasks.

Yu & Zhang (2023) identify a key obstacle to depth in "Why 'Classic' Transformers are Shallow and How to Make Them Go Deep" [arXiv:2312.06182]: token similarity escalation, where tokens become increasingly similar after successive attention applications. They propose similarity elimination to address this.

## Layer Pruning

Post-hoc layer pruning is a practical approach to reducing depth in deployed models.

Huang et al. (2026) present "GradPruner: Gradient-guided Layer Pruning" [arXiv:2601.19503], achieving 40% parameter reduction with only 0.99% accuracy decrease on Llama3.1-8B and Mistral-7B, reducing inference costs by approximately 39%.

Lu et al. (2024) find in "Reassessing Layer Pruning in LLMs" [arXiv:2411.15558] that simple reverse-order pruning (removing the final 25% of layers) outperforms sophisticated metrics, and partial-layer fine-tuning is sufficient for recovery. They create Llama-3.1-6.3B-It requiring only 12.74-14.96M training tokens while outperforming larger baselines.

Kim et al. (2024) in "Shortened LLaMA" [arXiv:2402.02834] show 20-35% of layers can be removed with LoRA fine-tuning sufficient for recovery.

Zhu et al. (2026) in "FlattenGPT" [arXiv:2602.08858] merge adjacent layers via flattening, achieving ~20% depth compression while retaining 90-96% zero-shot performance.

## Weight-Sharing and Depth Recurrence

Weight-sharing across transformer layers is a well-established technique for parameter efficiency.

### Depth-Recurrent Transformers

Most directly related to our work, Geiping et al. (2026) in "Thinking Deeper, Not Longer" [arXiv:2603.21676] propose a depth-recurrent transformer that shares weights across iterations using a single shared block applied 20+ times. They identify three key stability mechanisms:
- **Silent thinking**: loss computed only at the final iteration step
- **LayerScale**: learned per-channel scaling of residual contributions
- **Identity-biased gating**: -2.0 bias that retains ~88% of the previous state

They demonstrate a "computational frontier" where accuracy jumps from chance to near-perfect as recurrence depth increases, and show generalization to 1.75x training depth.

### Universal Transformers

Dehghani et al. (2019) introduced weight sharing across depth in "Universal Transformers" [arXiv:1807.03819], iteratively applying a single layer with adaptive halting. Our approach differs by using a correction mechanism rather than simple iteration.

## Our Contribution

The literature establishes that (1) depth is overprovisioned in current transformers by 2-5x, (2) wider models are desirable for inference efficiency, (3) removing layers post-hoc is surprisingly effective, and (4) weight-sharing across depth is viable but requires stabilization mechanisms.

However, no prior work offers a mechanism to make wider models competitive with deeper ones during training from scratch while maintaining the same inference cost as a standard transformer of that width.

The look-ahead correction mechanism addresses this gap. During training, D transformer blocks are iterated K times, with each position receiving a correction derived from the previous position's output. At sequential inference, K=1 suffices because each position naturally receives corrections from all preceding positions' full processing. The model is a D-layer transformer at inference but was trained with K×D effective depth.

Key results:

- At ~340M inference FLOPs, D=6 C=2048 (29.04 PPL) matches N=12 C=1536 (29.01 PPL) -- 6 wide layers with correction equals 12 standard layers.
- D=12 C=1408 beats N=12 C=1408 by 0.92 PPL at 200K iters and 1.06 PPL at 400K iters, at matched depth and width (only 5% FLOP overhead from the correction FFN).
- D=23 C=1024 beats N=24 C=1024 by 0.53 PPL at FLOP parity.
- The correction adds value across all depths tested (D=1 through D=23) and all widths (C=446 through C=2048).
- Converting any pretrained transformer to use the correction mechanism and fine-tuning briefly improves quality: N=12 → D=12 gains 1.20 PPL, N=24 → D=24 gains 0.43 PPL.

Unlike layer pruning (which removes capacity post-hoc), depth-recurrent approaches (which require stabilization tricks), or architectural changes (which require retraining), the correction mechanism is additive -- it can be applied to existing models through fine-tuning, it requires no special training tricks (no silent thinking, no gating), and it enables training wider models that are competitive from the start.

## References

Csordas, R., Manning, C. D., & Potts, C. (2025). Do Language Models Use Their Depth Efficiently? NeurIPS 2025. arXiv:2505.13898.

Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, L. (2019). Universal Transformers. ICLR 2019. arXiv:1807.03819.

Fahim, M. M. M., & Karim, M. R. (2026). The Depth Delusion: Why Transformers Should Be Wider, Not Deeper. arXiv:2601.20994.

Geiping, J., et al. (2026). Thinking Deeper, Not Longer: Depth-Recurrent Transformers. arXiv:2603.21676.

Hu, Y., Zhou, C., & Zhang, M. (2025). What Affects the Effective Depth of Large Language Models? arXiv:2512.14064.

Huang, W., Cheng, A., & Wang, Y. (2026). GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs. arXiv:2601.19503.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.

Kim, B.-H., et al. (2024). Shortened LLaMA: Depth Pruning for Large Language Models. arXiv:2402.02834.

Li, Y., et al. (2026). Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs. ICLR 2026. arXiv:2510.18245.

Lu, Y., Cheng, H., Fang, Y., Wang, Z., Wei, J., Xu, D., Xuan, Q., Yang, X., & Zhu, Z. (2024). Reassessing Layer Pruning in LLMs: New Insights and Methods. arXiv:2411.15558.

Men, X., et al. (2024). What Matters in Transformers? Not All Attention is Needed. arXiv:2406.15786.

Petty, J., van Steenkiste, S., Dasgupta, I., Sha, F., Garrette, D., & Linzen, T. (2024). The Impact of Depth on Compositional Generalization in Transformer Language Models. NAACL 2024. arXiv:2310.19956.

Razzhigaev, A., et al. (2024). Your Transformer is Secretly Linear. arXiv:2405.12250.

Saratchandran, H., Teney, D., & Lucey, S. (2025). Leaner Transformers: More Heads, Less Depth. arXiv:2505.20802.

Yehudai, G., Sanford, C., Bechler-Speicher, M., Fischer, O., Gilad-Bachrach, R., & Globerson, A. (2025). Depth-Width Tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers. arXiv:2503.01805.

Yu, Y., & Zhang, Y. (2023). Why "Classic" Transformers are Shallow and How to Make Them Go Deep. arXiv:2312.06182.

Zhu, X., et al. (2026). FlattenGPT: Depth Compression for Transformer with Layer Flattening. arXiv:2602.08858.
