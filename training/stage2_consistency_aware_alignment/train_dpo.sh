export CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

SAVE_NAME=llama-2-7b-stage2 # [llama-2-7b-stage2, llama-2-13b-stage2]
MODEL_PATH=../checkpoints/llama-2-7b-stage1/ # [./checkpoints/llama-2-7b-stage1/, ./checkpoints/llama-2-13b-stage1/]
TRAIN_FILE=../../data/dpo.json

EPOCH=2
BETA=0.1
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port $MASTER_PORT \
    --deepspeed_config_file stage3_ds_config.json \
    dpo_tune.py \
    --model_name_or_path $MODEL_PATH \
    --gradient_checkpointing \
    --tokenizer_name $MODEL_PATH \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16\
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --beta $BETA \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --num_train_epochs $EPOCH \
    --output_dir ../checkpoints/$SAVE_NAME \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1