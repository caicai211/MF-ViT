#!/bin/bash

conda activate MotionRGBD

# K400
python tools/train.py --script mf_vit --config track3_rtd_k400_ep10 --save_dir ./output --mode multiple --nproc_per_node 2
python tools/train.py --script mf_vit --config track3_rgbt_k400_ep10 --save_dir ./output --mode multiple --nproc_per_node 2
python tools/train.py --script mf_vit --config track3_rgbd_k400_ep10 --save_dir ./output --mode multiple --nproc_per_node 2