#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 main_DeSD_ssl.py \
 --arch='res3d50' \
 --data_path='/media/new_userdisk0/data_SSL/' \
 --list_path='SSL_data_deeplesion.txt' \
 --output_dir='snapshots/DeSD_res3d50_192_300/' \
 --batch_size_per_gpu=32 \
 --optimizer='sgd' \
 --use_fp16=False \
 --momentum_teacher=0.996 \
 --epochs=300 \
 --out_dim=60000 \
 --warmup_teacher_temp=0.04 \
 --teacher_temp=0.07 \
 --warmup_teacher_temp_epochs=20 \
 --weight_decay=0.000001 \
 --weight_decay_end=0.000001 \
 --clip_grad=0 \
 --lr=0.3 \
 --min_lr=0.048 \
 --use_bn_in_head=True


