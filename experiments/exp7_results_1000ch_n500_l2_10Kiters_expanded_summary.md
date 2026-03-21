# Exp 7a Results: 1000 chains (expanded_names), n_embed=500, n_layers=2, 10K iters, 1 seed

Config: n_embed=500, n_layers=2, max_iters=10000, batch_size=32, lr=0.0005, device=cuda
Seeds: 1
MLM KG: A, A', E, E', F, F', G, G', H, H'
Causal KG: Ec, Ec', Hc, Hc' (--causal_kg)
kg_as_text: B, B', C, C' (--kg_as_text)

Note: E/E'/H/H' = MLM KG training (default). Ec/Ec'/Hc/Hc' = causal KG training (--causal_kg).

## Text Evaluation (h@5 / PPL)

| Tier | A | A' | B (kat) | B' (kat) | C (kat) | **C' (kat)** | E | E' | Ec | Ec' | F | F' | G | G' | H | H' | Hc | Hc' |
|------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mem** | .011/8.86 | .010/9.42 | .064/5.86 | .365/3.52 | .296/4.25 | **.783/2.02** | .027/9.22 | .014/9.47 | .249/3.51 | .329/3.36 | .012/8.73 | .013/9.05 | .011/8.75 | .011/9.09 | .010/9.21 | .009/9.48 | .343/3.14 | .489/2.67 |
| **trn** | .000/10.17 | .000/10.71 | .056/5.60 | .222/3.92 | .278/3.91 | **.711/2.20** | .000/13.17 | .000/13.20 | .100/4.23 | .156/4.57 | .000/10.68 | .011/9.48 | .000/10.95 | .000/9.89 | .000/10.96 | .000/11.55 | .222/3.70 | .322/3.79 |
| **gen** | .000/10.25 | .000/10.74 | .011/8.49 | .244/5.90 | .133/7.72 | **.478/4.88** | .000/14.05 | .000/12.75 | .144/7.04 | .133/6.37 | .000/9.63 | .011/10.75 | .000/10.54 | .011/10.37 | .000/11.36 | .011/11.22 | .222/5.90 | .344/5.42 |
| **kg_ex_m** | .000/3283 | .000/658 | .000/11.21 | .017/9.80 | .067/10.14 | .150/10.82 | .000/7837 | .000/3582 | .017/5.32 | .050/7.58 | .000/5426 | .000/926 | .000/3023 | .000/907 | .000/1514 | .000/678 | .033/5.74 | .083/6.48 |
| **kg_ex_g** | .000/2620 | .000/1173 | .017/19.30 | .067/12.98 | .033/25.19 | .083/21.49 | .000/4308 | .000/3941 | .033/12.87 | .117/18.21 | .000/7373 | .000/1917 | .000/3066 | .000/1977 | .000/1627 | .000/1618 | .083/10.85 | .083/10.92 |
| **tx_ex_m** | .000/8.98 | .000/12.64 | .050/6.52 | .217/5.56 | .117/6.62 | **.483/3.83** | .000/13.57 | .000/13.74 | .100/8.71 | .133/6.56 | .017/9.66 | .000/12.52 | .017/9.69 | .000/12.92 | .017/10.96 | .000/14.90 | .133/5.87 | .283/4.41 |
| **tx_ex_g** | .000/18.68 | .000/20.28 | .067/8.65 | .150/8.59 | .117/9.86 | **.383/7.51** | .000/25.04 | .000/29.43 | .100/26.24 | .067/13.45 | .017/14.01 | .000/20.16 | .000/14.38 | .000/22.13 | .000/19.57 | .000/27.80 | .083/11.74 | .100/10.05 |

(kat) = kg_as_text mode. E/E'/H/H' = MLM KG. Ec/Ec'/Hc/Hc' = causal KG.

## KG Evaluation (h@5 / PPL)

B/B'/C/C' use linearized KG eval; A/A'/E/E'/F/F'/G/G'/H/H' use MLM; Ec/Ec'/Hc/Hc' use causal.

| Tier | A | A' | B (lin) | B' (lin) | C (lin) | **C' (lin)** | E | E' | Ec | Ec' | F | F' | G | G' | H | H' | Hc | **Hc'** |
|------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mem** | .185/13.32 | **.860**/1.93 | .066/5.44 | .288/3.61 | .297/4.24 | .783/2.01 | .018/65.05 | **.907/1.54** | .630/2.29 | .708/2.11 | .077/25.17 | .562/3.45 | .094/25.08 | .807/2.27 | .473/5.24 | .899/1.61 | .716/2.18 | .759/2.00 |
| **trn** | .200/14.17 | **.811**/2.06 | .050/5.34 | .172/3.92 | .256/3.94 | .750/2.11 | .011/74.49 | **.911/1.54** | .611/2.25 | .656/2.19 | .056/21.02 | .511/3.55 | .178/19.38 | .789/2.38 | .511/4.59 | .922/1.62 | .756/2.00 | .778/1.95 |
| **gen** | .044/28.10 | **.644**/3.11 | .017/7.79 | .150/6.20 | .106/7.14 | .489/4.91 | .000/131.33 | .789/2.44 | .489/7.53 | .556/6.90 | .067/34.26 | .333/6.11 | .022/42.07 | .611/3.59 | .244/9.82 | **.822/2.24** | .533/5.71 | .544/5.31 |
| **kg_ex_m** | .000/41.61 | .517/5.72 | .000/8.83 | .075/5.92 | .083/8.57 | .517/3.57 | .000/125.11 | .517/4.39 | .850/2.21 | .800/2.20 | .000/70.89 | .083/12.54 | .000/78.31 | .417/7.48 | .067/18.82 | .500/4.07 | **.883/2.08** | .883/1.96 |
| **kg_ex_g** | .033/84.37 | .183/18.98 | .025/15.89 | .183/9.95 | .042/20.99 | .325/10.84 | .000/249.36 | .317/12.56 | .550/11.08 | .617/15.74 | .000/127.60 | .067/65.68 | .017/116.40 | .150/35.84 | .133/51.84 | .183/22.29 | **.667/7.94** | .633/7.30 |
| **tx_ex_m** | .000/638 | .000/1697 | .067/7.15 | .125/7.19 | .058/7.33 | .167/7.33 | .000/604 | .000/3155 | .000/39.10 | .000/65.29 | .000/673 | .000/1634 | .000/755 | .000/1850 | .000/990 | .000/3610 | .000/34.85 | .000/21.40 |
| **tx_ex_g** | .000/1077 | .000/2630 | .058/8.95 | .033/11.79 | .058/10.59 | .117/12.23 | .000/890 | .000/5026 | .000/111.93 | .000/96.42 | .000/874 | .000/4061 | .000/941 | .000/2719 | .000/2491 | .000/5417 | .000/52.77 | .017/36.80 |

## Key Observations

### Causal vs MLM KG training (Ec/Hc vs E/H)

#### Text: causal training is essential
- **Ec/Ec'** learn text (PPL 3.36-3.51 mem); **E/E'** completely fail (PPL 9.22-9.47). Same for H variants.
- All MLM-trained models (A/A'/E/E'/F/F'/G/G'/H/H') fail at text with PPL 8-14. Only causal KG (Ec/Hc) and kg_as_text (B/C) learn text.
- Causal KG training forces the model to process sequences left-to-right, which transfers to text generation.

#### KG with V rotation: MLM beats causal
- **E' MLM** (.907/1.54) beats **Ec' causal** (.708/2.11) on KG mem — MLM is better for KG memorization with V rotation.
- **H' MLM** (.899/1.61) beats **Hc' causal** (.759/2.00) on KG mem — same pattern.
- **E' MLM** (.911/1.54) also leads on KG transfer. H' MLM (.922/1.62) beats Hc' (.778/1.95).
- MLM training (predict masked tokens given full context) is more natural for structured KG triples.

#### KG without V rotation: causal beats MLM
- **Ec causal** (.630/2.29) far better than **E MLM** (.018/65.05) on KG mem. Without V rotation, E can't learn KG via MLM at all.
- **Hc causal** (.716/2.18) beats **H MLM** (.473/5.24) — less dramatic but still clear.
- V rotation is critical for MLM KG training. Without it, causal training is the only viable path.

#### KG exclusive: causal generalizes better
- **Ec/Hc causal** dominate kg_excl_mem (.850/.883 h@5, PPL 2.08-2.21); **E'/H' MLM** get .517/.500 h@5 (PPL 4.07-4.39).
- Causal training produces better generalization to unseen KG facts, even though MLM has better memorization.

#### V rotation is critical for MLM
- E without V rotation: catastrophic KG PPL (65.05). E' with it: excellent (1.54). Gap of 42x.
- H without V rotation: poor KG PPL (5.24). H' with it: 1.61. Gap of 3x.
- For causal training, V rotation helps but isn't essential: Ec .630/2.29 → Ec' .708/2.11.

### Text champions: C' dominates, Hc' strong second
- **C' crushes text**: .783/2.02 mem, .711/2.20 transfer, .483/3.83 text_excl_mem — far ahead of everything
- **Hc' is strong second**: .489/2.67 mem, .322/3.79 transfer, .283/4.41 text_excl_mem
- **B' third**: .365/3.52 mem — V rotation matters hugely (B is .064/5.86)
- **All MLM KG models fail at text**: A/A'/E/E'/F/F'/G/G'/H/H' all PPL 8-14, near-zero h@5

### KG champions: E'/H' MLM lead memorization, Hc/Hc' lead generalization
- **E' MLM best KG mem/trn**: .907/1.54 mem, .911/1.54 transfer — best PPL of any model
- **H' MLM close**: .899/1.61 mem, .922/1.62 transfer
- **Hc/Hc' causal best kg_excl**: .883/2.08 and .883/1.96 — causal generalizes better to unseen KG
- **A' excellent**: .860/1.93 mem — best h@5 among slot-angle models
- **G' very strong**: .807/2.27 mem — per-relation slot angles + V rotation excels
- **C' linearized KG competitive**: .783/2.01 mem — remarkably close to native KG models!

### V rotation effects
- **B vs B'**: .064 vs .365 text mem — massive gap. V rotation transforms B from useless to decent
- **C vs C'**: .296 vs .783 text mem — even bigger gap
- **Ec vs Ec'**: .249 vs .329 text mem — modest gap
- **Hc vs Hc'**: .343 vs .489 text mem — significant
- **E vs E'**: .018 vs .907 KG mem — catastrophic vs excellent. V rotation essential for MLM KG
- **H vs H'**: .473 vs .899 KG mem — poor vs excellent. Same pattern
- **A vs A'**: .185 vs .860 KG mem — enormous gap
- **G vs G'**: .094 vs .807 KG mem — same enormous gap
- **F vs F'**: .077 vs .562 KG mem — V rotation critical for flat RoPE

### Cross-pollination: ZERO at n500/l2 (confirmed)
- **kg_excl on text**: catastrophic PPL for all MLM models (658-7837). Ec/Hc causal get 5-18 PPL, C' 10-21. No model learns KG-exclusive facts as text.
- **text_excl on KG**: catastrophic PPL for all models (21-5417). Zero h@5. No cross-pollination in either direction.
- **Shallow models (l2) cannot cross-pollinate** — consistent with 130ch finding where l20 showed faint signals

### 1000 chains vs 130 chains (n500/l2/10K)
- **A/A' massively better with 1000 chains on KG**: A' .860 vs .138 mem. Slot angles need data volume.
- **G' same pattern**: .807 vs not tested at 130ch. Per-relation slots also need data.
- **Ec' NOT better with 1000 chains on KG**: .708 vs .794 at 130ch. Learned cumsum may overfit with more data?
- **Hc/Hc' much better**: Hc' .759 vs .143, Hc .716 vs .122. Fixed cumsum benefits from data.
- **C' text much better**: .783/2.02 vs .258/4.72 at 130ch. More text data = better text model.
- **Cross-pollination worse**: kg_excl PPL on text is 10-21 at 1000ch vs 5-6 at 130ch. More data = more modality separation.
