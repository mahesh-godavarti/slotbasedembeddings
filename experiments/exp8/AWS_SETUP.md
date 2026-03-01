# Exp 8: AWS Setup

## 1. Launch EC2 Instance

- **Instance type**: g5.2xlarge (8 vCPU, 32GB RAM, 1x A10G GPU 24GB VRAM, ~$1.21/hr)
- **AMI**: Ubuntu 24.04 LTS
- **Storage**: 50GB gp3
- **Key pair**: Create or select existing (e.g., ML-server.pem)
- **Security group**:
  - Inbound: SSH (port 22) from 0.0.0.0/0 (protected by key pair)
  - Outbound: All traffic (default — needed for Claude Code API, pip)

## 2. SSH into the Instance

```bash
ssh -i ~/AWS/ML-server.pem ubuntu@<public-ip>
```

## 3. Install Claude Code

Native install (no Node.js required):

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Verify:

```bash
claude --version
```

## 4. Authenticate Claude Code

Run `claude` — it will print a URL. Copy the URL and open it in your browser on your Mac to complete OAuth login. Requires a Pro, Max, Teams, Enterprise, or Console account.

```bash
claude
# It prints: Open this URL to authenticate: https://claude.ai/oauth/...
# Copy that URL, open in your Mac browser, log in, done.
```

## 5. Sync Exp 8 to the Instance

Run this from your Mac (separate terminal):

```bash
rsync -avz -e "ssh -i ~/AWS/ML-server.pem" \
  ~/Dropbox/ACarrot/Papers/journey_groupoids_tmlr_v7/experiments/exp8/ \
  ubuntu@<public-ip>:~/exp8/
```

## 6. Set Up Python Environment

```bash
cd ~/exp8
python3 -m venv venv
source venv/bin/activate
pip install torch numpy tokenizers tqdm
```

## 7. Smoke Test

```bash
source venv/bin/activate
python word_experiment.py --models B "B'" --smoke --wiki_lines 1000 --vocab_size 4000
```

## 8. Start Claude Code

```bash
cd ~/exp8
claude
```

## Data Layout

All data lives in `exp8/data/` with relative paths (no hardcoded `~/AWS/...`):

```
exp8/
├── word_experiment.py
├── AWS_SETUP.md
└── data/
    ├── wiki.en.txt              (3.0 GB)
    ├── wordnet-synonyms.txt     (5.1 MB)
    ├── framenet.txt             (3.4 MB)
    ├── questions-words_for_training.txt (590 KB)
    ├── wordanalogies.txt        (49 KB)
    └── BATS_3.0/                (240 KB)
```

## Example Experiment Commands

```bash
# Full run with B/C models
python word_experiment.py --models B "B'" C "C'" --seeds 3 --n_embed 100 --n_layers 20 --iters 10000 --vocab_size 16000

# KG-as-text mode
python word_experiment.py --models B "B'" C "C'" --kg_as_text --seeds 3 --n_embed 100 --n_layers 20 --iters 10000

# All KG models
python word_experiment.py --models A "A'" D "D'" F "F'" --seeds 3 --n_embed 100 --n_layers 20 --iters 10000

# Causal KG models
python word_experiment.py --models E "E'" H "H'" I "I'" --causal_kg --seeds 3 --n_embed 100 --n_layers 20 --iters 10000
```

## Notes

- The A10G GPU gives significant speedup over Mac CPU (~2-3x faster than T4)
- The code auto-detects GPU via `device=cuda`
- Remember to **stop the instance** when done to avoid charges
- Data paths are relative to the script — no path changes needed between local and AWS
