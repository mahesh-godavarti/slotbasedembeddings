# Auto-Resume Design for Preemptible ThunderCompute Instances

## Problem

We run long training experiments (10-200K iters, hours to days) on ThunderCompute instances. These instances can be preempted/rebooted without warning. We need experiments to resume automatically after a reboot, without requiring manual SSH login.

## Constraints

1. **Container environment** -- no systemd, no init.d, no cron daemon
2. **PID 1 is `tini -s -- /usr/sbin/sshd -D -e`** -- tini starts sshd, that's it
3. **No control over container entrypoint** -- we can't change the Docker CMD/ENTRYPOINT
4. **We CAN modify files inside the container** -- /usr/sbin/sshd, /etc/profile.d/, home directory, etc.
5. **Checkpointing is already implemented** -- training saves rolling checkpoints at every eval. Training script auto-resumes from checkpoint if checkpoint_dir has a checkpoint.
6. **Multiple experiments may run simultaneously** (one per GPU)
7. **NEVER rename/move/swap source files** -- we destroyed blocks.py twice doing this. Use PYTHONPATH override if needed for flash attention.
8. **The existing venv is at `/home/ubuntu/exp8/venv/`**

## Approach Options

### Option A: `.bashrc` / `.profile` trigger

**How**: Add a launcher call to .bashrc. On SSH login, it checks if experiments should be running and starts them.

**Pros**: Simple, no system modification
**Cons**: Requires manual SSH login after reboot. Defeats the purpose.

**Verdict**: Rejected -- user needs fully automatic restart.

### Option B: cron @reboot

**How**: Add `@reboot /path/to/launcher.sh` to crontab.

**Pros**: Clean, standard approach
**Cons**: No cron daemon in the container. `which cron` returns nothing.

**Verdict**: Rejected -- not available.

### Option C: sshd wrapper (RECOMMENDED)

**How**: Rename `/usr/sbin/sshd` to `/usr/sbin/sshd.real`. Create a new `/usr/sbin/sshd` shell script that:
1. Starts the experiment watchdog in background
2. exec's `/usr/sbin/sshd.real` with all original arguments

When tini starts "sshd", it actually runs our wrapper, which starts the watchdog and then becomes the real sshd.

**Pros**: Runs on every container boot. No SSH login needed. The watchdog starts before sshd is even ready.
**Cons**: Modifying a system binary. If the wrapper breaks, SSH access is lost (recovery: ThunderCompute console or new instance). Must be robust.

**Verdict**: Recommended. The risk is manageable with careful implementation.

### Option D: Modify /etc/bash.bashrc with singleton check

**How**: Add launcher to system bashrc, but use flock to ensure it only runs once.

**Pros**: Less risky than modifying sshd
**Cons**: Still requires SSH login to trigger

**Verdict**: Rejected -- same problem as Option A.

## Recommended Implementation (Option C)

### Components

#### 1. Experiment Registry (`/home/ubuntu/look_ahead6/active_experiments.json`)

```json
[
    {
        "name": "d6_c2048_scratch",
        "gpu": 1,
        "command": "bash /home/ubuntu/look_ahead6/run_d6_c2048_scratch.sh 1",
        "log": "/home/ubuntu/look_ahead6/logs/d6_c2048_scratch.log",
        "enabled": true
    },
    {
        "name": "d2_c1536_scratch",
        "gpu": 0,
        "command": "bash /home/ubuntu/look_ahead6/run_d2_c1536.sh",
        "log": "/home/ubuntu/look_ahead6/logs/d2_c1536_scratch.log",
        "enabled": true
    }
]
```

To add/remove experiments, edit this file. Set `enabled: false` to pause without removing.

#### 2. Watchdog (`/home/ubuntu/look_ahead6/watchdog.sh`)

A loop that:
- Runs every 60 seconds
- Reads active_experiments.json
- For each enabled experiment, checks if its process is running (grep for a unique string in the command)
- If not running, launches it in background
- Logs all actions to a watchdog log

Must use `flock` to ensure only one watchdog instance runs:
```bash
exec 200>/tmp/watchdog.lock
flock -n 200 || exit 0  # another watchdog already running
```

Key details:
- Wait 30s on startup for GPUs to initialize
- Check nvidia-smi before launching (GPU must be available)
- Don't launch if checkpoint indicates training completed (iter >= max_iters)
- Log to `/home/ubuntu/look_ahead6/logs/watchdog.log`

#### 3. sshd Wrapper (`/usr/sbin/sshd`)

```bash
#!/bin/bash
# Start experiment watchdog in background
nohup /home/ubuntu/look_ahead6/watchdog.sh > /dev/null 2>&1 &

# Become the real sshd
exec /usr/sbin/sshd.real "$@"
```

Setup (run once):
```bash
mv /usr/sbin/sshd /usr/sbin/sshd.real
# Create wrapper (the script above)
chmod +x /usr/sbin/sshd
```

### Safety measures

1. **flock on watchdog** -- prevents multiple watchdog instances even if sshd wrapper runs multiple times
2. **GPU check before launch** -- don't launch if nvidia-smi fails or GPU memory is in use
3. **Completed experiment detection** -- check if checkpoint iter >= max_iters, don't relaunch finished experiments
4. **Experiment registry** -- single source of truth, easy to edit, no hardcoded commands
5. **All training scripts already handle checkpoint resume** -- they check checkpoint_dir on startup and resume from latest
6. **Watchdog is a separate process from sshd** -- if watchdog crashes, sshd still works. If sshd restarts, flock prevents duplicate watchdogs.

### What to test

1. Verify sshd.real works: `sudo /usr/sbin/sshd.real -D -e` (in another terminal)
2. Verify wrapper works: SSH in, check watchdog started
3. Kill a training process, verify watchdog restarts it within 60s
4. Simulate reboot: kill all training + watchdog, run wrapper manually
5. Verify flock: try starting watchdog twice, second should exit immediately

### Past failures and lessons

1. **sshd wrapper launched training 6 times in 30 seconds** (2026-03-23) -- fixed by flock
2. **File swap scripts destroyed blocks.py twice** -- never swap files, use experiment registry instead
3. **Stale checkpoints from crashed runs caused wrong model loading** -- watchdog should check GPU assignment matches
4. **blocks.py got deleted by trap in run_d23_flash.sh** -- no traps in launch scripts, no file manipulation
