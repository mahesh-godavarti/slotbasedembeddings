# Lesson: Queuing Experiments Reliably

## Problem

When experiments are already running and we want to queue follow-up work, we can't modify the running scripts (they're already loaded in memory). So we create separate "watcher" processes that poll until the current job finishes, then launch the next one.

## What went wrong

We used `pgrep -f 'width_extend_gpu0'` to watch for the script wrapper. This is fragile because:
1. The pgrep matches the script name, not the actual training process
2. When the script finishes, the pgrep returns false immediately
3. Race conditions, `set -e`, or stale parent processes can kill the watcher silently
4. The follow-up jobs never launched — both GPUs sat idle

## The fix

Always pgrep on the **actual last training process** in the script, not the script wrapper. For example:

```bash
# BAD: watches the script name
while pgrep -f 'width_extend_gpu0' > /dev/null 2>&1; do sleep 30; done

# GOOD: watches the actual last training command
while pgrep -f 'n_embed 1120.*n_layers 5' > /dev/null 2>&1; do sleep 30; done
```

The training process is what actually runs on the GPU. When it finishes, the GPU is free and the next job can start.

## Best practices

1. **Plan all work upfront** and put everything in one script before launching. This avoids the queuing problem entirely.

2. **When that's not possible**, queue by watching the actual training process (model name, n_embed, n_layers pattern), not the wrapper script.

3. **Always verify** that queued jobs actually started — check `nvidia-smi` after the expected completion time. Don't assume the queue worked.

4. **Always write a script file**, then `nohup bash script.sh`. Never use `nohup bash -c '...'` with complex commands — quoting in nested shells is fragile and fails silently. The watcher + launch should be a single `.sh` file on disk.
