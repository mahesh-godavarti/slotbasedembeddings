#!/bin/bash
# Grid sweep: H, H', J, J' at n_embed 100, 200, 400 (500K iters each)
# 12 models total, run sequentially on single GPU

echo "=== Starting H/J grid sweep ==="
echo "=== n_embed=100 ==="
bash /home/ubuntu/exp8/run_HJ_n100.sh
echo "=== n_embed=100 DONE ==="

echo "=== n_embed=200 ==="
bash /home/ubuntu/exp8/run_HJ_n200.sh
echo "=== n_embed=200 DONE ==="

echo "=== n_embed=400 ==="
bash /home/ubuntu/exp8/run_HJ_n400.sh
echo "=== n_embed=400 DONE ==="

echo "=== Grid sweep complete ==="
