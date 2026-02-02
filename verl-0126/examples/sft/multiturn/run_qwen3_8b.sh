#!/bin/bash
set -x
export TZ='Asia/Shanghai'
formatted_day=$(date "+%Y%m%d")
formatted_time=$(date "+%Y%m%d-%H-%M-%S")




PROJECT_DIR="$(pwd)"
experiment_name=qwen3-8B_cold_start-${formatted_time}

nproc_per_node=8
save_path=/extrahome0/Research_HUB/verl/output_dir/HPO/${formatted_day}/SFT-${formatted_time}/checkpoints
log_path=$PROJECT_DIR/output_dir/HPO/${formatted_day}/SFT-${formatted_time}/$experiment_name.log

mkdir -p $save_path
mkdir -p $PROJECT_DIR/output_dir/HPO/${formatted_day}/SFT-${formatted_time}
# Shift the arguments so $@ refers to the rest
shift 2
# use_remove_padding=true


export SWANLAB_MODE=cloud
export SWANLAB_LOG_DIR=$PROJECT_DIR/output_dir/HPO/${formatted_day}/SFT-${formatted_time}
export SWANLAB_API_KEY=7XJghZVNJYoRHjyZBnlSu

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='dynamic_grmi_grpo' \
    trainer.experiment_name=$experiment_name \
    data.train_files=/userhome/Research_HUB/verl/data_dir/Cold_Start/train.parquet \
    data.val_files=/userhome/Research_HUB/verl/data_dir/Cold_Start/test.parquet \
    data.multiturn.enable=true \
    +data.ignore_input_ids_mismatch=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size=512 \
    data.micro_batch_size_per_gpu=32 \
    data.max_length=32768 \
    data.truncation=right \
    model.partial_pretrain=/extrahome0/HF_models/Qwen/Qwen3-8B \
    trainer.default_local_dir=$save_path \
    trainer.total_epochs=5 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.max_ckpt_to_keep=5 \
    ulysses_sequence_parallel_size=2 \
    optim.lr=1e-4 \
    use_remove_padding=true $@ 2>&1 | tee $log_path  &