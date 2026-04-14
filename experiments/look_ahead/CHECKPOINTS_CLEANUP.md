# Checkpoint Cleanup

## Already Deleted (2026-04-11)

| Date | Size | Directory | Experiment |
|------|------|-----------|------------|
| 2026-03-26 | 4.1G | `checkpoints_d23_ksched` | D=23 K-schedule, done |
| 2026-03-27 | 3.3G | `checkpoints_n12` | N=12 C=1024, done |
| 2026-03-27 | 4.3G | `checkpoints_d24_converted` | D=24 finetune from N=24, done |
| 2026-03-27 | 5.5G | `checkpoints_d23` | D=23 C=1024, done |

## Safe to Delete

All experiments below are completed with results recorded in FULL_RESULTS.md.

### Scaling experiments (~33G, 18 dirs, March 28-30)

| Date | Size | Directory | Experiment |
|------|------|-----------|------------|
| 2026-03-28 | 1.2G | `checkpoints_scaling_n1` | Scaling N=1 C=1024 |
| 2026-03-28 | 1.2G | `checkpoints_scaling_n1_cont` | Scaling N=1 continuation |
| 2026-03-29 | 1.2G | `checkpoints_scaling_n1_cont2` | Scaling N=1 continuation 2 |
| 2026-03-28 | 1.4G | `checkpoints_scaling_n2` | Scaling N=2 C=1024 |
| 2026-03-28 | 1.4G | `checkpoints_scaling_n2_cont` | Scaling N=2 continuation |
| 2026-04-07 | 1.4G | `checkpoints_scaling_n2_cont2` | Scaling N=2 continuation 2 |
| 2026-03-28 | 1.6G | `checkpoints_scaling_n3` | Scaling N=3 C=1024 |
| 2026-03-28 | 1.6G | `checkpoints_scaling_n3_cont` | Scaling N=3 continuation |
| 2026-03-29 | 1.6G | `checkpoints_scaling_n3_cont2` | Scaling N=3 continuation 2 |
| 2026-03-28 | 2.2G | `checkpoints_scaling_n6` | Scaling N=6 C=1024 |
| 2026-03-29 | 2.2G | `checkpoints_scaling_n6_cont` | Scaling N=6 continuation |
| 2026-03-29 | 2.2G | `checkpoints_scaling_n6_cont2` | Scaling N=6 continuation 2 |
| 2026-03-28 | 1.3G | `checkpoints_scaling_d1` | Scaling D=1 C=1024 |
| 2026-03-28 | 1.3G | `checkpoints_scaling_d1_cont` | Scaling D=1 continuation |
| 2026-03-29 | 1.3G | `checkpoints_scaling_d1_cont2` | Scaling D=1 continuation 2 |
| 2026-03-29 | 1.3G | `checkpoints_scaling_d1_cont3` | Scaling D=1 continuation 3 |
| 2026-03-29 | 1.3G | `checkpoints_scaling_d1_fresh` | Scaling D=1 fresh |
| 2026-03-28 | 1.5G | `checkpoints_scaling_d2` | Scaling D=2 C=1024 |
| 2026-03-28 | 1.5G | `checkpoints_scaling_d2_cont` | Scaling D=2 continuation |
| 2026-03-29 | 1.5G | `checkpoints_scaling_d2_cont2` | Scaling D=2 continuation 2 |
| 2026-03-29 | 1.5G | `checkpoints_scaling_d2_cont3` | Scaling D=2 continuation 3 |
| 2026-03-29 | 1.5G | `checkpoints_scaling_d2_fresh` | Scaling D=2 fresh |
| 2026-03-28 | 1.7G | `checkpoints_scaling_d3` | Scaling D=3 C=1024 |
| 2026-03-28 | 1.7G | `checkpoints_scaling_d3_cont` | Scaling D=3 continuation |
| 2026-03-29 | 1.7G | `checkpoints_scaling_d3_cont2` | Scaling D=3 continuation 2 |
| 2026-03-29 | 1.7G | `checkpoints_scaling_d3_cont3` | Scaling D=3 continuation 3 |
| 2026-03-29 | 1.7G | `checkpoints_scaling_d3_fresh` | Scaling D=3 fresh |
| 2026-03-28 | 2.3G | `checkpoints_scaling_d6` | Scaling D=6 C=1024 |
| 2026-03-29 | 2.3G | `checkpoints_scaling_d6_cont` | Scaling D=6 continuation |
| 2026-03-29 | 2.3G | `checkpoints_scaling_d6_cont2` | Scaling D=6 continuation 2 |
| 2026-03-30 | 2.3G | `checkpoints_scaling_d6_cont3` | Scaling D=6 continuation 3 |
| 2026-03-30 | 2.3G | `checkpoints_scaling_d6_fresh` | Scaling D=6 fresh |

### 341M FLOP budget experiments (~54G, March 28 - April 4)

| Date | Size | Directory | Experiment | Final PPL |
|------|------|-----------|------------|-----------|
| 2026-03-28 | 3.4G | `checkpoints_d12_converted` | D=12 finetune from N=12 | 32.21 |
| 2026-03-28 | 3.5G | `checkpoints_n13` | N=13 C=1024 | 32.82 |
| 2026-03-31 | 3.7G | `checkpoints_n14` | N=14 C=1024 | -- |
| 2026-03-31 | 1.4G | `checkpoints_blockhead_d24` | Old blockhead D=24 | -- |
| 2026-04-01 | 4.4G | `checkpoints_d2_c2176` | D=2 C=2176 341M | -- |
| 2026-04-02 | 6.5G | `checkpoints_n6_c2048` | N=6 C=2048 341M | 30.35 |
| 2026-04-02 | 7.0G | `checkpoints_d6_c2048_scratch` | D=6 C=2048 341M | 29.04 |
| 2026-04-03 | 8.8G | `checkpoints_n2_c3776` | N=2 C=3776 341M | 36.10 |
| 2026-04-03 | 7.6G | `checkpoints_n4_c2656` | N=4 C=2656 341M | 31.95 |
| 2026-04-03 | 6.2G | `checkpoints_n24_c1088` | N=24 C=1088 341M | 28.68 |
| 2026-04-03 | 6.6G | `checkpoints_n12_c1536` | N=12 C=1536 341M, ext to 650K | 25.09 |
| 2026-04-04 | 9.1G | `checkpoints_d1_c4128` | D=1 C=4128 341M, ext to 600K | 29.20 |

### 85M FLOP budget experiments (~30G, March 31 - April 9)

| Date | Size | Directory | Experiment | Final PPL |
|------|------|-----------|------------|-----------|
| 2026-03-31 | 1.8G | `checkpoints_d6_c2048` | D=6 C=2048 intermediate | -- |
| 2026-03-31 | 2.6G | `checkpoints_d2_c1536` | D=2 C=1536 | -- |
| 2026-03-31 | 3.0G | `checkpoints_d1_c1952` | D=1 C=1952 | -- |
| 2026-04-06 | 2.0G | `checkpoints_d11_c768` | D=11 C=768 85M | 36.82 |
| 2026-04-06 | 2.7G | `checkpoints_d3_c1408` | D=3 C=1408 85M | 37.26 |
| 2026-04-07 | 2.3G | `checkpoints_d6_c1024` | D=6 C=1024 85M | 36.56 |
| 2026-04-07 | 2.4G | `checkpoints_d5_c1120` | D=5 C=1120 85M | 36.38 |
| 2026-04-07 | 5.8G | `checkpoints_n6_c1088` | N=6 C=1088 bs256, ext to 360K | 32.14 |
| 2026-04-08 | 9.1G | `checkpoints_d1_c4128_lr1e5` | D=1 C=4128 lr=1e-5 | -- |
| 2026-04-09 | 5.7G | `checkpoints_d1_pure_c2048` | Pure variant | -- |
| 2026-04-09 | 3.3G | `checkpoints_d1_c2048_bs64` | D=1 bs64 experiment | 65.94 |
| 2026-04-09 | 3.1G | `checkpoints_n2_c1888_bs64` | N=2 bs64 experiment | 66.91 |
| 2026-04-09 | 3.8G | `checkpoints_n1_c2656` | N=1 C=2656 | -- |
| 2026-04-09 | 11G | `checkpoints_n1_c5344` | N=1 C=5344 | -- |

### Width scaling experiments (~6G, March 31 - April 7)

| Date | Size | Directory | Experiment |
|------|------|-----------|------------|
| 2026-03-31 | 1.5G | `checkpoints_width_d2_c1024_scratch` | D=2 C=1024 width |
| 2026-03-31 | 283M | `checkpoints_width_d2_c256` | D=2 C=256 width |
| 2026-03-31 | 283M | `checkpoints_width_d2_c256_scratch` | D=2 C=256 width |
| 2026-03-31 | 597M | `checkpoints_width_n2_c512` | N=2 C=512 width |
| 2026-03-31 | 629M | `checkpoints_width_d2_c512` | D=2 C=512 width |
| 2026-03-31 | 629M | `checkpoints_width_d2_c512_scratch` | D=2 C=512 width |
| 2026-04-06 | 483M | `checkpoints_width_d1_c560_scratch` | D=1 C=560 width |
| 2026-04-07 | 275M | `checkpoints_width_n2_c256` | N=2 C=256 width |
| 2026-04-07 | 299M | `checkpoints_width_d1_c280_scratch` | D=1 C=280 width |
| 2026-04-07 | 597M | `checkpoints_width_n2_c512_b1024` | N=2 C=512 b1024 |
| 2026-04-07 | 644M | `checkpoints_width_d1_c560_b1024` | D=1 C=560 b1024 |
| 2026-04-07 | 1.5G | `checkpoints_width_d1_c1120_scratch` | D=1 C=1120 width |

### Completed bs512 (April 10)

| Date | Size | Directory | Experiment | Final PPL |
|------|------|-----------|------------|-----------|
| 2026-04-10 | 3.1G | `checkpoints_n2_c1888_bs512` | N=2 C=1888 bs512 | 38.23 |

## Keep (active or may extend)

| Date | Size | Directory | Status |
|------|------|-----------|--------|
| 2026-04-11 | 2.5G | `checkpoints_d1_c2048_bs1024` | Running |
| 2026-04-11 | 1.8G | `checkpoints_n6_c1088_bs1024` | Running |
| 2026-04-11 | 3.3G | `checkpoints_d1_c2048_bs512` | Done 200K, may extend |
| 2026-04-11 | 2.4G | `checkpoints_n6_c1088_bs512` | Done 200K |
| 2026-04-08 | 11G | `checkpoints_d1_c2048` | bs256 run, ext to 400K |
