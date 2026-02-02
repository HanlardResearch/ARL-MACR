# run on 8xH100
# make sure your current working directory is the root of the project
export TZ='Asia/Shanghai'
formatted_day=$(date "+%Y%m%d")
formatted_time=$(date "+%Y%m%d-%H-%M-%S")

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"




ROLE_NAME=SCI-Agent-v9  # 如果 $1 为空，默认使用 default_role
MAX_ITER=2             # 如果 $2 为空，默认使用 3

experiment_name=qwen3-8B_Polaris-dynamic-grmi-$formatted_time
rolloutdir=$PROJECT_DIR/output_dir/HPO/${formatted_day}/Train-${formatted_time}/rollouts
checkpointdir=/extrahome0/Research_HUB/verl/output_dir/HPO/${formatted_day}/Train-${formatted_time}/checkpoints
valrolloutdir=$PROJECT_DIR/output_dir/HPO/${formatted_day}/Train-${formatted_time}/val-rollouts
log_path=$PROJECT_DIR/output_dir/HPO/${formatted_day}/Train-${formatted_time}/$experiment_name.log
mkdir -p $rolloutdir
mkdir -p $checkpointdir
mkdir -p $valrolloutdir
export SWANLAB_MODE=cloud
export SWANLAB_LOG_DIR=$PROJECT_DIR/output_dir/HPO/${formatted_day}/Train-${formatted_time}
export SWANLAB_API_KEY=7XJghZVNJYoRHjyZBnlSu
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
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    trainer.val_before_train=False \
    custom_reward_function.path=/userhome/Research_HUB/verl/verl/utils/reward_score/Cmath.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$modelpath \
    trainer.validation_data_dir=$valrolloutdir \
    trainer.rollout_data_dir=$rolloutdir \
    actor_rollout_ref.rollout.temperature=1.4 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.4 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.85 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=sglang \
    data.train_batch_size=256 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    data.max_prompt_length=1024 \
    data.max_response_length=31744 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/extrahome0/Research_HUB/verl/output_dir/HPO/20260201/SFT-20260201-10-03-21/checkpoints/global_step_25/huggingface \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='dynamic_grmi_grpo' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=1 \
    data.train_files=/userhome/Research_HUB/verl/data_dir/POLARIS/Polaris_train_filter_Diffculty1_add_sys_pmt.parquet \
    data.val_files="[ '/userhome/Research_HUB/verl/data_dir/gsm8k-style/aime2024_add_sys_pmt.parquet', '/userhome/Research_HUB/verl/data_dir/gsm8k-style/aime2025_add_sys_pmt.parquet' ]" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/grmi_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    trainer.total_epochs=5    $@ 2>&1 | tee $log_path  &

