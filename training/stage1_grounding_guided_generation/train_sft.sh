SAVE_NAME=llama-2-7b-stage1 # [llama-2-7b-stage1, llama-2-13b-stage1]
MODEL_PATH=meta-llama/Llama-2-7b # [meta-llama/Llama-2-7b, meta-llama/Llama-2-13b]
DATA_PATH=../data/sft.json

NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
LR=2e-5
EPOCHS=5
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port $MASTER_PORT \
    --deepspeed_config_file ./stage2_ds_config.json \
    finetune.py \
    --use_flash_attn \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name $MODEL_PATH \
    --use_slow_tokenizer \
    --train_file $DATA_PATH \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs $EPOCHS \
    --output_dir ../checkpoints/$SAVE_NAME \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --gradient_checkpointing \
    --use_special_tokens
