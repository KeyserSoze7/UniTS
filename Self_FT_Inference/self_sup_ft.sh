model_name=UniTS
exp_name=UniTS_supervised_x64
wandb_mode=disabled
project_name=supervised_learning

random_port=$((RANDOM % 9000 + 1000))

# Supervised learning
CUDA_VISIBLE_DEVICES=3,7,8,9 torchrun --nnodes 1 --nproc-per-node=1  --master_port $random_port  /media/RTCIN9TBA/Interns/RDT1/ibl3kor/UniTS/run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model 64 \
  --des 'Exp' \
  --learning_rate 1e-4 \
  --weight_decay 5e-6 \
  --train_epochs 5 \
  --batch_size 4 \
  --acc_it 4 \
  --debug $wandb_mode \
  --project_name $project_name \
  --clip_grad 100 \
  --pretrained_weight /media/RTCIN9TBA/Interns/RDT1/ibl3kor/UniTS/checkpoints/units_x64_supervised_checkpoint.pth \
  --task_data_config_path /media/RTCIN9TBA/Interns/RDT1/ibl3kor/UniTS/data_provider/tsv.yaml
