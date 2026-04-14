# Instructions for New Machine (7dfsikdj)

## Context

We're testing whether D=1 (one wide layer with correction) can catch up to N=12 (twelve standard layers) given enough training. At ~341M FLOPs and 200K iters, D=1 C=4128 (33.40) is 4.39 PPL behind N=12 C=1536 (29.01), but the gap has been narrowing steadily throughout training. We want to extend both runs to see if D=1 crosses over.

At ~85M FLOPs and 100K iters, D=1 C=2048 (39.94) is 2.11 behind N=12 C=768 (37.83), also narrowing.

## Task 1: Extend 341M experiments to 400K iters

Both checkpoints are in look_ahead6 on this machine. Resume from 200K to 400K.

### N=12 C=1536 (resume from 200K)
```bash
cd /home/ubuntu/look_ahead6
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models roformer --n_embed 1536 --n_layers 12 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 400000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_n12_c1536 \
    --gpu 0 \
    --amp 2>&1 | tee logs/roformer_n12_c1536_ext400k.log
```

### D=1 C=4128 (resume from 200K)
```bash
cd /home/ubuntu/look_ahead6
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 4128 --n_layers 5 --block_size 256 --batch_size 32 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 400000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d1_c4128 \
    --gpu 1 \
    --amp 2>&1 | tee logs/d1_c4128_ext400k.log
```

## Task 2: Extend 85M experiments to 200K iters

### N=12 C=768 (resume from 100K)
```bash
cd /home/ubuntu/look_ahead6
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models roformer --n_embed 768 --n_layers 12 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --n_head 16 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_n12_c768 \
    --gpu 0 \
    --amp 2>&1 | tee logs/roformer_n12_c768_ext200k.log
```

### D=1 C=2048 (resume from 100K)
```bash
cd /home/ubuntu/look_ahead6
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/exp8/venv/bin/python train_wiki_streaming.py train \
    --models block_head_corr_ffn_add --n_embed 2048 --n_layers 5 --block_size 256 --batch_size 64 \
    --lr 2e-4 --softmax --convergence_weight 0.1 --d_block 1 --n_head 16 --k_min 2 \
    --max_iters 200000 --eval_interval 5000 \
    --data_dir /home/ubuntu/look_ahead/look_ahead/data_owt \
    --checkpoint_dir checkpoints_d1_c2048 \
    --gpu 1 \
    --amp 2>&1 | tee logs/d1_c2048_ext200k.log
```

## Execution order

1. Start Task 1 (341M extensions) on both GPUs immediately — this is the most important.
2. When Task 1 finishes, run Task 2 (85M extensions).

## Checking progress
```bash
bash check_progress.sh logs/LOGFILE.log
```

## Key numbers to watch

341M gap at 200K: D=1 4.39 behind N=12. Look for continued narrowing.
85M gap at 100K: D=1 2.11 behind N=12. Already narrowing fast.

## Important notes

- Venv: `/home/ubuntu/exp8/venv/bin/python`
- Data: `/home/ubuntu/look_ahead/look_ahead/data_owt`
- All code runs from `/home/ubuntu/look_ahead6/`
- look_ahead7 is a strict superset of look_ahead6 (same blocks.py, same train_wiki_streaming.py, models.py has SA classes added). Either directory works.
- Checkpoints auto-resume from checkpoint_dir if files exist.
- Do NOT modify blocks.py or models.py.
