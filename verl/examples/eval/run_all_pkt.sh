#!/bin/bash

# roles prompt
ROLE_NAME=SCI-Agent-v8

# benchmark path
aime24=data_dir/benchmarks/aime24.parquet
aime25=data_dir/benchmarks/aime25.parquet
amc=data_dir/benchmarks/amc23.parquet
minerva=data_dir/benchmarks/minerva.parquet
olympiad=data_dir/benchmarks/olympiad.parquet
vail_files="['$aime24','$aime25']"


export SWANLAB_MODE=cloud
export SWANLAB_API_KEY=7XJghZVNJYoRHjyZBnlSu
export VERL_LOG_LEVEL=INFO
export PYTHON_LOG_LEVEL=INFO
export RAY_DEDUP_LOGS=0
export RAY_COLOR_PREFIX=0
export VERL_LOGGING_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Tmax hyper-parameter
MAX_ITER=6

# gird search
temperatures=(0.6 0.8 1.0 1.2 1.4 1.6)
top_ps=(1.0 0.95 0.85 0.9)
top_ks=(20 50 100)


total=$(( ${#temperatures[@]} * ${#top_ps[@]} * ${#top_ks[@]} ))
echo "🚀 Starting hyperparameter grid search over $total combinations..."
echo "Temperature: [${temperatures[*]}]"
echo "Top-p: [${top_ps[*]}]"
echo "Top-k: [${top_ks[*]}]"
echo ""

count=0

  for top_p in "${top_ps[@]}"; do
      for top_k in "${top_ks[@]}"; do
          for temp in "${temperatures[@]}"; do
            count=$((count + 1))
            echo "=================================================="
            echo "🧪 Experiment $count / $total: temp=$temp, top_p=$top_p, top_k=$top_k"
            echo "=================================================="


            formatted_day=$(date "+%Y%m%d")
            formatted_time=$(date "+%Y%m%d-%H-%M-%S")

            experiment_name=4GPU-${ROLE_NAME}-Temp${temp}-TopP${top_p}-TopK${top_k}-Maxiter-${MAX_ITER}

            base_dir=/userhome/Research_HUB/verl/output_dir/AIME_output_dir/${formatted_day}/Test-${formatted_time}
            rolloutdir=${base_dir}/rollouts
            checkpointdir=${base_dir}/checkpoints
            valrolloutdir=${base_dir}/val-rollouts
            log_path=${base_dir}/${experiment_name}.log

            mkdir -p "$rolloutdir" "$checkpointdir" "$valrolloutdir"

            export SWANLAB_LOG_DIR="$base_dir"

            echo "Log will be saved to: $log_path"

            python3 -m verl.trainer.main_ppo \
                trainer.val_before_train=True \
                trainer.val_only=True \
                algorithm.adv_estimator=grpo \
                custom_reward_function.path=/userhome/Research_HUB/verl/verl/utils/reward_score/Cmath.py \
                custom_reward_function.name=compute_score2 \
                +actor_rollout_ref.rollout.extra.roles="$ROLE_NAME" \
                +actor_rollout_ref.rollout.extra.role_config_path=/userhome/Research_HUB/verl/data_dir/AgentRoles \
                +actor_rollout_ref.rollout.extra.max_iter="$MAX_ITER" \
                actor_rollout_ref.model.path=/extrahome0/HF_models/Qwen/Qwen3-8B \
                data.train_files=/userhome/Research_HUB/verl/data_dir/CMath/train.parquet \
                data.val_files="$vail_files" \
                trainer.validation_data_dir="$valrolloutdir" \
                trainer.rollout_data_dir="$rolloutdir" \
                actor_rollout_ref.rollout.prompt_length=1024 \
                actor_rollout_ref.rollout.response_length=31744 \
                data.prompt1_length=1024 \
                data.response1_length=31744 \
                data.prompt2_length=4096 \
                data.response2_length=28672 \
                data.prompt3_length=6144 \
                data.response3_length=26624 \
                actor_rollout_ref.rollout.temperature=1.4 \
                actor_rollout_ref.rollout.top_p=1.0 \
                actor_rollout_ref.rollout.top_k=20 \
                actor_rollout_ref.rollout.val_kwargs.temperature="$temp" \
                actor_rollout_ref.rollout.val_kwargs.top_p="$top_p" \
                actor_rollout_ref.rollout.val_kwargs.top_k="$top_k" \
                actor_rollout_ref.rollout.val_kwargs.n=8 \
                actor_rollout_ref.rollout.val_kwargs.do_sample=True \
                data.train_batch_size=128 \
                data.filter_overlong_prompts=True \
                data.truncation='error' \
                actor_rollout_ref.actor.optim.lr=1e-6 \
                actor_rollout_ref.model.use_remove_padding=True \
                actor_rollout_ref.actor.ppo_mini_batch_size=4 \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.actor.use_kl_loss=True \
                actor_rollout_ref.actor.kl_loss_coef=0.001 \
                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                actor_rollout_ref.actor.entropy_coeff=0 \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                actor_rollout_ref.actor.fsdp_config.param_offload=False \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
                actor_rollout_ref.rollout._target_=verl.workers.config.rollout.RolloutConfig \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
                actor_rollout_ref.rollout.n=5 \
                actor_rollout_ref.rollout.mode='sync' \
                actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                algorithm.use_kl_in_reward=False \
                trainer.critic_warmup=0 \
                trainer.logger='["console","swanlab"]' \
                trainer.project_name='Test_AIME25' \
                trainer.experiment_name="$experiment_name" \
                trainer.n_gpus_per_node=8 \
                trainer.nnodes=1 \
                trainer.save_freq=20 \
                trainer.test_freq=5 \
                trainer.total_epochs=15 \
                2>&1 | tee "$log_path"

            echo "✅ Completed experiment $count: temp=$temp, top_p=$top_p, top_k=$top_k"
            echo ""
        done
    done
done

echo "🎉 All $total hyperparameter experiments finished!"