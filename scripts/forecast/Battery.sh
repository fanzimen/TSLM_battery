#!/bin/sh

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=288
patch_len=96
# ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
ckpt_path=checkpoints/checkpoint.pth
data=battery
file_name=upsampled_CALCE_CS2_35_label

for subset_rand_ratio in 1
do
# train
torchrun --nnodes=1 --nproc_per_node=4 run.py \
  --task_name forecast \
  --is_training 0 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/$data/ \
  --data_path $file_name.txt \
  --data custom \
  --model_id battery_sr_$subset_rand_ratio \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 0 \
  --subset_rand_ratio $subset_rand_ratio \
  --use_ims \
  --use_multi_gpu
done
