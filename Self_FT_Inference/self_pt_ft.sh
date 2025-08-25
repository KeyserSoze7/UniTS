model_name=UniTS
exp_name=UniTS_supervised_x64
wandb_mode=disabled
project_name=supervised_learning
d_model=64

random_port=$((RANDOM % 9000 + 1000))

# Prompt tuning
CUDA_VISIBLE_DEVICES=3,7,8,9 torchrun --nnodes 1 --master_port $random_port /media/RTCIN15TB/Interns/RDT1/ekq3kor/UNITS/UniTS/run.py \
  --is_training 1 \
  --model_id $exp_name \ 
  --model $model_name \
  --lradj prompt_tuning \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 3e-3 \
  --weight_decay 0 \
  --prompt_tune_epoch 5 \
  --train_epochs 0 \
  --acc_it 32 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --clip_grad 100 \
  --pretrained_weight /media/RTCIN15TB/Interns/RDT1/ekq3kor/UNITS/UniTS/checkpoints/ALL_task_UniTS_supervised_x64_UniTS_All_ftM_dm64_el3_Exp_0/checkpoint.pth \
  --task_data_config_path  /media/RTCIN15TB/Interns/RDT1/ekq3kor/UNITS/UniTS/Self_Implemetation/data_provider.yaml