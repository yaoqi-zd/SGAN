#!/bin/bash

# train
CUDA_VISIBLE_DEVICES=$1 python tools/stage2/train_sgan.py  --cfg_file $2

# # inference with loader
# CUDA_VISIBLE_DEVICES=$1 python tools/stage2/infer_cam.py --cfg_file $2

# evaluate seed mIoU
# python tools/eval_mIoU.py --cfg_file $2