# Running Experiments on AWS

## 1. Launch EC2 Instance

- **Instance type**: g4dn.xlarge (4 vCPU, 16GB RAM, 1x T4 GPU, ~$0.53/hr)
- **AMI**: Ubuntu 24.04 LTS
- **Storage**: 30GB gp3
- **Key pair**: Create or select existing (e.g., ML-server.pem)
- **Security group**:
  - Inbound: SSH (port 22) from 0.0.0.0/0 (protected by key pair)
  - Outbound: All traffic (default — needed for Claude Code API, pip, npm)

## 2. SSH into the Instance

```bash
ssh -i ~/AWS/ML-server.pem ubuntu@<public-ip>
```

## 3. Install Node.js and Claude Code

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @anthropic-ai/claude-code
```

## 4. Authenticate Claude Code

Run `claude` — it will print a URL. Copy the URL and open it in your browser on your Mac to complete OAuth login. No API key needed.

```bash
claude
# It prints: Open this URL to authenticate: https://claude.ai/oauth/...
# Copy that URL, open in your Mac browser, log in, done.
```

## 5. Copy Experiments to the Instance

Run this from your Mac (separate terminal):

```bash
rsync -avz -e "ssh -i ~/AWS/ML-server.pem" \
  ~/Dropbox/ACarrot/Papers/journey_groupoids_tmlr_v7/experiments/ \
  ubuntu@<public-ip>:~/experiments/
```

## 6. Start Claude Code

```bash
cd ~/experiments
claude
```

Then tell Claude Code to set up the Python environment and run experiments. It will install Python, pip, torch, numpy, create the venv, and run everything.

## Example Experiment Commands

Once Claude Code is running, you can tell it:

- "Set up the Python venv with torch and numpy, then run a smoke test"
- "Run: python kg_text_experiment.py --models B B' C C' --kg_as_text --seeds 3 --n_embed 100 --n_layers 20 --iters 10000 --exp 7a"

## Notes

- The T4 GPU should give ~10-20x speedup over Mac CPU
- The code auto-detects GPU via `device=cuda`
- Remember to **stop the instance** when done to avoid charges
- Your public IP may change if you switch WiFi networks
