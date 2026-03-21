# Exp 7a Results: 1000 chains (expanded_names), n_embed=500, n_layers=2, 10K iters, 1 seed — SOFTMAX attention

Config: n_embed=500, n_layers=2, max_iters=10000, batch_size=32, lr=0.0005, device=cuda, --softmax
Seeds: 1
MLM KG: A, A', E, E', F, F', G, G', H, H'
Causal KG: Ec, Ec', Hc, Hc' (--causal_kg)
kg_as_text: B, B', C, C' (--kg_as_text)

Note: This run uses softmax attention instead of the default softplus (log(exp(x)+1)).
Purpose: Test whether softmax fixes MLM KG interference with causal text learning.

## Text Evaluation (h@5 / PPL)

| Tier | A | A' | B (kat) | B' (kat) | C (kat) | C' (kat) | E | E' | Ec | Ec' | F | F' | G | G' | H | H' | Hc | Hc' |
|------|---|---|---------|----------|---------|----------|---|---|---|-----|---|---|---|---|---|---|---|------|
| **mem** | .009/10.16 | .009/9.79 | .195/5.54 | .375/4.19 | .201/5.64 | .520/3.61 | .012/11.92 | .009/9.95 | .158/4.03 | .053/6.27 | .009/9.72 | .010/10.08 | .010/9.68 | .010/9.71 | .008/10.25 | .009/9.90 | .182/4.11 | .237/4.03 |
| **trn** | .000/11.86 | .000/10.53 | .222/5.34 | .256/4.51 | .133/5.92 | .544/3.55 | .011/16.75 | .011/13.34 | .067/4.48 | .011/7.59 | .000/10.91 | .000/11.66 | .000/10.68 | .011/10.94 | .011/13.81 | .000/12.54 | .156/4.69 | .200/4.57 |
| **gen** | .000/10.72 | .000/10.83 | .100/8.09 | .244/6.24 | .089/8.12 | .311/6.00 | .000/15.33 | .000/13.02 | .111/7.18 | .000/7.77 | .000/10.43 | .000/10.66 | .000/10.56 | .000/10.45 | .000/13.39 | .011/11.93 | .089/6.18 | .144/6.18 |
| **kg_ex_m** | .000/681 | .000/369 | .017/12.38 | .017/11.12 | .033/15.33 | .067/12.49 | .000/1773 | .000/3458 | .067/6.05 | .017/12.51 | .000/1038 | .000/530 | .000/2269 | .000/605 | .000/735 | .000/437 | .033/6.13 | .033/7.34 |
| **kg_ex_g** | .000/1123 | .000/665 | .000/19.46 | .017/15.02 | .000/35.17 | .067/19.13 | .000/2597 | .000/3289 | .050/14.86 | .000/18.28 | .000/2267 | .000/1139 | .000/2145 | .000/1251 | .000/1337 | .000/1128 | .017/13.97 | .017/12.56 |
| **tx_ex_m** | .000/11.01 | .000/11.33 | .150/6.76 | .100/6.84 | .117/8.07 | .167/6.15 | .000/20.29 | .000/10.90 | .017/9.60 | .000/12.01 | .000/10.16 | .000/11.83 | .000/10.51 | .000/11.72 | .000/11.97 | .000/11.99 | .067/8.47 | .083/7.53 |
| **tx_ex_g** | .000/21.97 | .000/17.64 | .133/9.61 | .117/9.17 | .033/13.26 | .150/9.28 | .000/37.54 | .000/18.87 | .033/22.82 | .000/17.29 | .000/15.34 | .000/18.59 | .000/14.35 | .000/16.33 | .050/26.58 | .017/22.88 | .067/14.90 | .083/13.25 |

(kat) = kg_as_text mode.

## KG Evaluation (h@5 / PPL)

B/B'/C/C' linearized; A/A'/E/E'/F/F'/G/G'/H/H' MLM; Ec/Ec'/Hc/Hc' causal.

| Tier | A | A' | B (lin) | B' (lin) | C (lin) | C' (lin) | E | E' | Ec | Ec' | F | F' | G | G' | H | H' | Hc | Hc' |
|------|---|---|---------|----------|---------|----------|---|---|---|-----|---|---|---|---|---|---|---|------|
| **mem** | .184/11.42 | .891/1.78 | .224/5.09 | .392/3.60 | .208/5.38 | .610/3.04 | .555/4.81 | **.958/1.33** | .670/2.30 | .768/2.05 | .105/19.01 | .567/3.39 | .107/18.59 | .831/2.09 | .773/2.30 | .863/1.88 | .713/2.21 | .749/2.07 |
| **trn** | .178/13.00 | .856/2.00 | .244/4.94 | .306/3.98 | .161/5.47 | .617/2.98 | .633/4.20 | **.967/1.33** | .667/2.42 | .667/2.11 | .078/16.99 | .522/3.56 | .111/17.82 | .867/2.12 | .778/2.43 | .900/2.03 | .767/2.00 | .789/2.12 |
| **gen** | .111/23.41 | .656/3.55 | .117/7.55 | .261/5.85 | .106/7.64 | .350/5.66 | .378/8.76 | .867/1.73 | .422/6.99 | .478/6.47 | .078/27.86 | .356/5.10 | .056/34.07 | .544/4.27 | .644/3.25 | **.767/2.48** | .556/5.45 | .478/5.72 |
| **kg_ex_m** | .017/34.48 | .617/3.79 | .017/10.43 | .108/6.44 | .033/10.31 | .292/5.61 | .133/18.47 | .750/2.55 | .900/1.93 | .883/1.98 | .017/47.79 | .200/7.84 | .000/51.23 | .467/6.26 | .300/11.52 | .350/6.22 | **.933/1.88** | .850/2.04 |
| **kg_ex_g** | .050/65.01 | .317/11.86 | .033/16.44 | .092/9.57 | .008/22.69 | .233/9.86 | .050/60.33 | .400/7.80 | .533/8.72 | .650/11.61 | .000/83.59 | .117/25.79 | .033/71.35 | .267/18.35 | .167/34.45 | .233/38.77 | **.567/8.93** | .600/6.31 |
| **tx_ex_m** | .000/501 | .000/1261 | .108/7.22 | .067/8.55 | .117/8.25 | .033/8.60 | .000/790 | .000/1724 | .000/50.46 | .000/139.10 | .000/512 | .000/1076 | .000/507 | .000/1345 | .000/828 | .000/1628 | .000/33.66 | .000/27.99 |
| **tx_ex_g** | .000/804 | .000/1727 | .100/10.04 | .067/11.87 | .033/11.96 | .100/10.60 | .000/1301 | .000/3844 | .000/97.84 | .000/139.88 | .000/753 | .000/1620 | .000/585 | .000/1632 | .000/1194 | .000/1374 | .000/46.90 | .000/56.24 |

## Key Observations

### Softmax does NOT fix MLM-text interference
- All MLM models (A/A'/E/E'/F/F'/G/G'/H/H') still completely fail at text: PPL 9-12, near-zero h@5
- Switching from softplus to softmax does not enable MLM-trained models to learn causal text
- The bidirectional vs causal mask conflict is at the weight level, not caused by attention normalization

### Softmax helps E' KG dramatically
- **E' KG mem**: .958/1.33 (softmax) vs .907/1.54 (softplus) — best KG result we've seen at l2
- **E KG mem**: .555/4.81 (softmax) vs .018/65.05 (softplus) — massive rescue. Softmax helps E without V rotation
- **H KG mem**: .773/2.30 (softmax) vs .473/5.24 (softplus) — significant improvement
- Softmax makes MLM KG learning more robust, especially without V rotation

### Softmax hurts text learning for all models
- **C'**: .520/3.61 (softmax) vs .783/2.02 (softplus) — much worse
- **B'**: .375/4.19 (softmax) vs .365/3.52 (softplus) — similar h@5 but worse PPL
- **Ec**: .158/4.03 (softmax) vs .249/3.51 (softplus) — worse on both
- **Hc'**: .237/4.03 (softmax) vs .489/2.67 (softplus) — much worse
- Softplus is fundamentally better for causal text generation

### B improves with softmax
- **B text mem**: .195/5.54 (softmax) vs .064/5.86 (softplus) — h@5 tripled
- **B lin KG mem**: .224/5.09 (softmax) vs .066/5.44 (softplus) — also better
- Standard RoPE without V rotation benefits from normalized attention

### Causal KG models learn text with softmax (but worse than softplus)
- Ec: .158/4.03 text mem — learning text, but worse than softplus (.249/3.51)
- Hc: .182/4.11 — learning text, but worse than softplus (.343/3.14)
- Hc': .237/4.03 — learning text, but worse than softplus (.489/2.67)

### Conclusion
- **Softmax is not the fix for MLM-text interference**. The problem is inherent to mixing bidirectional and causal objectives on shared weights.
- **Softplus is better for text generation**. The non-normalized attention works better for autoregressive prediction.
- **Softmax helps KG memorization** with MLM training, especially models without V rotation.
- **The trade-off**: softmax = better KG (especially without V rotation), softplus = better text.
