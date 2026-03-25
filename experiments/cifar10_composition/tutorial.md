# Auto-Resume Training on ThunderCompute

ThunderCompute can preempt (reboot) your instance at any time. When this happens, all running processes are killed. This tutorial explains how to set up automatic checkpoint saving and resume so training picks up where it left off after a reboot.

## Overview

The system has three parts:

1. **Resumable checkpoints** — the training script saves full state every epoch
2. **`auto_resume.sh`** — a script that finds interrupted runs and restarts them
3. **sshd wrapper** — ensures `auto_resume.sh` runs on boot before anything else

The boot sequence:

```
tini (PID 1)
  └→ /usr/sbin/sshd  (our wrapper script)
       ├→ auto_resume.sh &  (background: resumes training)
       └→ exec sshd.real    (SSH works normally)
```

Only one mechanism triggers the resume. The lock file `/tmp/.auto_resume_training.lock` (cleared on every reboot since `/tmp` is ephemeral) prevents double-runs if anything else also calls `auto_resume.sh`.

---

## Part 1: Make Your Training Script Resumable

### What to save

Every epoch, save a `latest_<run_id>.pt` checkpoint containing:

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),       # if using AMP
    'best_acc1': best_acc1,                          # or best_loss, etc.
    'best_acc5': best_acc5,
    'pe_type': args.pe_type,                         # run identifier
    'args': vars(args),                              # ALL CLI args
}, latest_ckpt)
```

Saving `vars(args)` is critical — `auto_resume.sh` reads it to reconstruct the exact launch command.

This checkpoint is **overwritten every epoch**, so disk cost is constant (one file per model, ~10-15x model size due to optimizer state).

Keep your existing `best_*.pt` checkpoint (saved on accuracy improvement) as a separate file.

### How to resume

Add a `--resume` flag. On startup, check for the latest checkpoint and load all state:

```python
parser.add_argument("--resume", action="store_true")

# ... after creating model, optimizer, scheduler, scaler ...

best_acc1 = 0
best_acc5 = 0
start_epoch = 1

latest_ckpt = os.path.join(args.save_dir, f'latest_{args.pe_type}.pt')
if args.resume and os.path.exists(latest_ckpt):
    print(f"Resuming from {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_acc1 = ckpt['best_acc1']
    best_acc5 = ckpt['best_acc5']
    print(f"  Resumed at epoch {start_epoch}, best_top1={best_acc1:.2f}%")

for epoch in range(start_epoch, args.epochs + 1):
    # ... training loop ...
```

Key points:
- Create model/optimizer/scheduler/scaler with the **same hyperparameters** first, then load state dicts. This ensures the object structure matches.
- `start_epoch = ckpt['epoch'] + 1` — the saved epoch is the last **completed** epoch.
- The scheduler state dict stores the internal step counter, so the LR schedule continues correctly.

### What you lose on preemption

At most one epoch of work. If the machine dies mid-epoch, the checkpoint from the previous epoch is used. For long epochs (e.g., 30 min on ImageNet), you could save mid-epoch too, but epoch-level granularity is usually fine.

---

## Part 2: The Auto-Resume Script

`auto_resume.sh` scans the checkpoint directory for `latest_*.pt` files, checks if each run is incomplete (epoch < total epochs), and relaunches with `--resume`.

The script:
- Reads `args` from the checkpoint to reconstruct the exact command
- Launches each run on its original GPU
- Uses a lock file to prevent double-runs
- Logs to `logs/auto_resume.log`

Full script at `/home/ubuntu/cifar10_composition/auto_resume.sh`.

### How it works

```bash
# For each latest_*.pt checkpoint:
#   1. Extract pe_type from filename
#   2. Read saved args from checkpoint (via Python one-liner)
#   3. Check if training is complete (epoch >= total epochs) — skip if so
#   4. Reconstruct the CLI command from saved args
#   5. Launch with nohup, appending to a log file
```

### Testing manually

```bash
# See what it would do (dry run — just check for checkpoints):
ls checkpoints/latest_*.pt

# Run it:
./auto_resume.sh

# Check the log:
cat logs/auto_resume.log
```

---

## Part 3: The Boot Trigger

### Why the obvious approaches don't work on ThunderCompute

ThunderCompute instances are containers with `tini` as PID 1, not full VMs with systemd. This rules out the standard Linux approaches:

**cron `@reboot`** — The `cron` daemon is not installed by default. You can `apt-get install cron`, and it works within a session, but:
- `cron` is an init.d service. On a normal Linux system, systemd (or SysV init) starts it on boot. Here, `tini` only starts `sshd` — it doesn't run init.d scripts.
- Even if you manually start cron (`sudo service cron start`), that only lasts until the next reboot. On reboot, tini starts fresh and doesn't know about cron.
- You could add `service cron start` to another startup hook, but then you need a startup hook to install your startup hook — turtles all the way down.

**systemd services** — `systemctl enable my-service` fails with "System has not been booted with systemd as init system." Systemd exists on the filesystem (some unit files are present), but it's not PID 1 and can't manage services.

**`/etc/rc.local`** — Doesn't exist, and nothing would execute it even if you created it.

**`.profile` / `.bashrc`** — These run on SSH login, not on boot. If the machine reboots at 3 AM, training sits idle until someone SSHs in. Also, `.bashrc` runs on every new shell (including subshells, `screen` sessions, etc.), so you need boot-detection logic to avoid re-triggering.

### What does work: the sshd wrapper

The only process `tini` reliably starts on boot is `sshd`. We replace `/usr/sbin/sshd` with a wrapper script that calls `auto_resume.sh` in the background, then `exec`s the real sshd binary.

We replace `/usr/sbin/sshd` with a wrapper script that calls `auto_resume.sh` in the background, then `exec`s the real sshd binary.

### Setup

```bash
# 1. Backup the real sshd binary
sudo cp /usr/sbin/sshd /usr/sbin/sshd.real

# 2. Replace with wrapper (can't overwrite while running, must rm first)
sudo rm /usr/sbin/sshd
sudo tee /usr/sbin/sshd > /dev/null << 'WRAPPER'
#!/bin/bash
# sshd wrapper: runs auto_resume.sh on boot, then execs real sshd.

LOCK="/tmp/.auto_resume_training.lock"
SCRIPT="/home/ubuntu/cifar10_composition/auto_resume.sh"

if [ -f "$SCRIPT" ] && [ ! -f "$LOCK" ]; then
    bash "$SCRIPT" >> /home/ubuntu/cifar10_composition/logs/auto_resume.log 2>&1 &
fi

exec /usr/sbin/sshd.real "$@"
WRAPPER
sudo chmod +x /usr/sbin/sshd
```

### Why this is safe

- `exec` replaces the wrapper process with the real sshd — no extra process lingering
- The `&` on `auto_resume.sh` means sshd starts immediately, SSH is not delayed
- The lock file prevents double-runs if you also call `auto_resume.sh` manually
- If `auto_resume.sh` doesn't exist (e.g., you delete it), the wrapper just starts sshd normally

### Reverting

```bash
sudo rm /usr/sbin/sshd
sudo mv /usr/sbin/sshd.real /usr/sbin/sshd
```

---

## Preventing Clashes

There is exactly **one trigger** (sshd wrapper) and **one lock** (`/tmp/.auto_resume_training.lock`). The lock file lives in `/tmp`, which is cleared on every reboot.

| Scenario | What happens |
|----------|-------------|
| Normal reboot | Lock doesn't exist → wrapper runs `auto_resume.sh` → lock created → training resumes |
| Manual run of `auto_resume.sh` after boot | Lock exists → script prints "already ran" and exits |
| `auto_resume.sh` doesn't exist | Wrapper skips it, starts sshd normally |
| Training already finished | `auto_resume.sh` reads checkpoint, sees epoch >= total → skips |

No cron jobs. No `.profile` hooks. No systemd services. One wrapper, one lock.

---

## Adding a New Training Run

To make a new training run auto-resumable:

1. Launch it with checkpoint saving enabled (as described in Part 1)
2. Let it complete at least one epoch so `latest_<run_id>.pt` exists
3. That's it — `auto_resume.sh` will find it automatically on next reboot

To stop a run from auto-resuming, delete its `latest_*.pt` checkpoint.

---

## File Inventory

| File | Purpose |
|------|---------|
| `/usr/sbin/sshd` | Wrapper script (boot trigger) |
| `/usr/sbin/sshd.real` | Real sshd binary |
| `auto_resume.sh` | Finds and restarts interrupted runs |
| `checkpoints/latest_*.pt` | Resumable checkpoints (one per model) |
| `checkpoints/best_*.pt` | Best-accuracy checkpoints (unchanged) |
| `logs/auto_resume.log` | Resume script output |
| `logs/imagenet_*_resumed.log` | Training output after resume |

---

## Caveats

- **DataLoader randomness**: After resume, the DataLoader's random shuffling will differ from the original run (different batch ordering within the resumed epoch). This has negligible impact on final results but means the run is not bit-for-bit reproducible after a preemption.
- **Disk space**: Each `latest_*.pt` is ~10-15x model parameter count (optimizer momentum + variance buffers). For ViT-S (22M params), this is ~264MB per checkpoint.
- **Instance rebuild**: If ThunderCompute fully rebuilds the instance (new filesystem), the sshd wrapper and sshd.real will be lost. The checkpoints survive only if the filesystem persists. In practice, ThunderCompute preserves `/home/ubuntu` across reboots.
