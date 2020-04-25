#!/bin/bash
current_time=`date +%Y-%m-%d-%H-%M-%S`
dataset_root="/data/yaoqi/Dataset/VOCdevkit/VOC2012/"
gpu=$1
exp=deeplabv2_weakly
pretrain="/data/yaoqi/segmentation/wsss/data/pretrained/vgg16_20M_mc.caffemodel"

# train
python ../../tools/train_anti_noise.py \
   --solver config/deeplabv2_weakly_solver.prototxt \
   --weights ${pretrain} \
   --gpu ${gpu} 2>&1 | tee log/train_${current_time}.log

# retrain
# python ../../tools/train_anti_noise.py \
#         --solver config/solver-f.prototxt \
#         --weights ${pretrain} \
#         --gpu ${gpu} 2>&1 | tee log/train_${current_time}.log

# validate
# split="val"  # trainaugSEC / val
# total_parts=4
# for ((i=1;i<=${total_parts};i++));
# do
# python ../../tools/test_ms.py --model model/seed_collab_semi/model_iter_8000.caffemodel \
#     --net config/${exp}.prototxt \
#     --imgPath ${dataset_root}/JPEGImages \
#     --png_savepath result/seg_res_collab_semi \
#     --id_path ${dataset_root}/ImageSets/Segmentation/${split}.txt \
#     --total_parts ${total_parts} \
#     --cur_part ${i} \
#     --gpu ${gpu} --smooth --multi_scale &
# done
