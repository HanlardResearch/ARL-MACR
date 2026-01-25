
export TZ='Asia/Shanghai'
formatted_day=$(date "+%Y%m%d")
formatted_time=$(date "+%Y%m%d-%H-%M-%S")

modelpath=/extrahome0/HF_models/Qwen/Qwen3-8B
#cp /userhome/Research_HUB/verl/data_dir/Model-Conifg/Qwen3-8B-default-config.json $modelpath/config.json
cat $modelpath/config.json


# Checklist
# 1 模型 actor_rollout_ref.model.path
# 2 名字 experiment_name
# 3 数据 data.val_files
# 4 奖励 custom_reward_function.name
# 5 长度 prompt_length
# 6 提示 /userhome/Research_HUB/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
# This is run name
loss_mode=grpo
ROLE_NAME=SCI-Agent-v9  # 如果 $1 为空，默认使用 default_role
MAX_ITER=3            # 如果 $2 为空，默认使用 3

experiment_name=GPU7-HPO-Diffculty1_Qwen3-8B-$ROLE_NAME-$formatted_time
rolloutdir=/userhome/Research_HUB/verl/output_dir/HPO/${formatted_day}/Train-${formatted_time}/rollouts
checkpointdir=/extrahome0/Research_HUB/verl/output_dir/HPO/${formatted_day}/Train-${formatted_time}/checkpoints
valrolloutdir=/userhome/Research_HUB/verl/output_dir/HPO/${formatted_day}/Train-${formatted_time}/val-rollouts
log_path=/userhome/Research_HUB/verl/output_dir/HPO/${formatted_day}/Train-${formatted_time}/$experiment_name.log
mkdir -p $rolloutdir
mkdir -p $checkpointdir
mkdir -p $valrolloutdir
export SWANLAB_MODE=cloud
export SWANLAB_LOG_DIR=/userhome/Research_HUB/verl/output_dir/HPO/${formatted_day}/Train-${formatted_time}
export SWANLAB_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXX
echo $log_path
set -x

########################## LOG ##########################
export VERL_LOG_LEVEL=INFO
export PYTHON_LOG_LEVEL=INFO
export RAY_DEDUP_LOGS=0
export RAY_COLOR_PREFIX=0  # 部分版本支持
export VERL_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1


python3 -m verl.trainer.main_ppo \
    trainer.val_before_train=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=hpo \
    algorithm.adv_estimator=grpo \
    +actor_rollout_ref.actor.HPO_epsilon=1e-4 \
    +actor_rollout_ref.actor.HPO_Hvalue=0.8 \
    custom_reward_function.path=/userhome/Research_HUB/verl/verl/utils/reward_score/Cmath.py \
    custom_reward_function.name=compute_score4 \
    +actor_rollout_ref.rollout.extra.roles=$ROLE_NAME \
    +actor_rollout_ref.rollout.extra.role_config_path=/userhome/Research_HUB/verl/data_dir/AgentRoles \
    +actor_rollout_ref.rollout.extra.max_iter=$MAX_ITER \
    actor_rollout_ref.model.path=$modelpath \
    trainer.validation_data_dir=$valrolloutdir \
    data.train_files=/userhome/Research_HUB/verl/data_dir/POLARIS/Polaris_train_filter_Diffculty1.parquet \
    data.val_files="[ '/userhome/Research_HUB/verl/data_dir/gsm8k-style/aime2024.parquet', '/userhome/Research_HUB/verl/data_dir/gsm8k-style/aime2025.parquet' ]" \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=31744 \
    data.prompt1_length=1024 \
    data.response1_length=31744 \
    data.prompt2_length=4096 \
    data.response2_length=28672 \
    data.prompt3_length=6144 \
    data.response3_length=26624 \
    actor_rollout_ref.nccl_timeout=7200 \
    trainer.default_local_dir=$checkpointdir \
    trainer.rollout_data_dir=$rolloutdir \
    actor_rollout_ref.rollout.temperature=1.45 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.4 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.85 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout._target_=verl.workers.config.rollout.RolloutConfig \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.mode='sync' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='GRMI-HPO' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=1 \
    $@ 2>&1 | tee $log_path  &
