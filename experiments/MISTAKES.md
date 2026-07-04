# Mistakes Log

## 2026-05-28: Failed to extract training loss lines from experiment output

### What happened
When running Model I at n250_l8, I told the user there were no loss lines being printed
during training, and that "the experiment only prints loss at evaluation time." This was
completely wrong — `train_model_mixed` prints `text_loss` and `kg_loss` every 500 iters
(line 3083 of kg_text_experiment.py).

### Root cause
The loss lines were in the output file but interleaved with tqdm `\r` carriage returns.
My extraction attempts used naive grep/string matching that failed to find them. Instead
of trying harder or checking the code to confirm the print statements existed, I concluded
the lines weren't being printed and gave the user false information.

I then incorrectly blamed Python stdout buffering with `tee`, killed a run that had 8K+
iters of useful training data, and restarted with `PYTHONUNBUFFERED=1`. The restart
happened to work because I also improved my parsing code, not because of the buffering fix.

### What I should have done
1. Read the training code first to confirm loss is printed every `eval_interval` iters.
2. When grep didn't find the lines, recognized it as a parsing problem, not a missing-data
   problem.
3. Used proper `\r` → `\n` replacement (like parse_log.py already does) from the start.
4. Never told the user "loss is only printed at evaluation time" without verifying.
5. Never killed a running experiment based on a wrong diagnosis.

### Fix
Created `read_training_loss.py` — a dedicated script that properly handles tqdm `\r`
carriage returns and extracts clean loss lines from any experiment output file. Use this
instead of ad-hoc grep/tail commands.

```bash
python read_training_loss.py <output_file>           # all loss lines
python read_training_loss.py <output_file> --tail 5  # last 5
python read_training_loss.py <output_file> --watch   # poll every 30s
```
