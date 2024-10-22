#!/bin/sh

export CUDA_VISIBLE_DEVICES=2

model_name=Timer
ckpt_path=checkpoints/Timer_anomaly_detection_1.0.ckpt
seq_len=768
d_model=256
d_ff=512
e_layers=4
patch_len=96
subset_rand_ratio=0.01
dataset_dir="./dataset/UCR_Anomaly_FullData"

# ergodic datasets
# for file_path in "$dataset_dir"/*
# do
# data=$(basename "$file_path")
# data="trend-054_UCR_Anomaly_DISTORTEDWalkingAceleration5_2700_5920_5979.txt"
# data="trend-220_UCR_Anomaly_STAFFIIIDatabase_43217_250720_251370.txt"
# data="mos-001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
# data="mav-001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
data="004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt"
# data="fte-011_UCR_Anomaly_DISTORTEDECG1_10000_11800_12100.txt"
python -u run.py \
  --task_name anomaly_detection_mse \
  --is_training 0 \
  --root_path ./dataset/UCR_Anomaly_FullData \
  --data_path $data \
  --model_id UCRA_$data \
  --ckpt_path $ckpt_path \
  --model $model_name \
  --data UCRA \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len $patch_len \
  --e_layers $e_layers \
  --train_test 0 \
  --batch_size 128 \
  --subset_rand_ratio $subset_rand_ratio \
  --train_epochs 10 \
  --use_ims
# done