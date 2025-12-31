# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import inspect
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator,Dict,List

import cloudpickle as pickle
import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, LoRAConfig
from vllm.lora.request import LoRARequest
# from vllm.model_executor.model_loader.utils import process_weights_after_loading
try:
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
except Exception:
    def process_weights_after_loading(*args, **kwargs):
        """Compatibility shim for vLLM versions without this helper."""
        return

try:
    # https://github.com/vllm-project/vllm/commit/96b9aa5aa076e64c68765232aec343e4d0006e2a
    from vllm.config import CompilationMode

    _use_compilation_mode = True
except ImportError:
    from vllm.config import CompilationLevel

    _use_compilation_mode = False

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from packaging import version as vs

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.import_utils import deprecated
from verl.utils.model import get_lora_rank_from_adapter
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


@deprecated(
    "vLLM spmd mode is deprecated. Please set `actor_rollout_ref.rollout.mode=async` to use vllm native server mode."
)
class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code

        lora_adapter_path = getattr(model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            lora_rank = get_lora_rank_from_adapter(lora_adapter_path)
        else:
            lora_rank = model_config.lora_rank

        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(lora_rank)}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        # This parameter verification is borrowed from vllm:
        # https://github.com/vllm-project/vllm/blob/561253b37faadaafe68168ea32d8d8157621a6b4/vllm/config/scheduler.py#L249
        if max_num_batched_tokens < max_model_len and not self.config.enable_chunked_prefill:
            raise ValueError(
                f"max_num_batched_tokens ({max_num_batched_tokens}) is smaller than max_model_len ({max_model_len}). "
                "Please increase max_num_batched_tokens or enable chunked prefill."
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_args = {"cudagraph_capture_sizes": cudagraph_capture_sizes}
                if _use_compilation_mode:
                    compilation_args["mode"] = CompilationMode.VLLM_COMPILE
                else:
                    compilation_args["level"] = CompilationLevel.PIECEWISE
                compilation_config["compilation_config"] = CompilationConfig(**compilation_args)
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)


    # def _build_role_prompt(self, role: str, problem: str, content: str = "", feedback: str = "") -> str:
    #     """构建 SciAgent 风格的角色化 prompt"""
    #     if role == "generate":
    #         return f"You are an expert problem solver. Please answer the following scientific question:\n{problem}"
    #     elif role == "review":
    #         return (
    #             f"You are a rigorous reviewer. Please evaluate whether the following solution is logically sound and free of factual errors:\n"
    #             f"[Question]\n{problem}\n"
    #             f"[Solution]\n{content}\n"
    #             f"Respond only with: '<Pass>' or '<Fail>: [specific reason]'"
    #         )
    #     elif role == "improve":
    #         return (
    #             f"You are an improvement specialist. Please revise the following solution based on the review feedback:\n"
    #             f"[Question]\n{problem}\n"
    #             f"[Original Solution]\n{content}\n"
    #             f"[Review Feedback]\n{feedback}\n"
    #             f"Output only the revised, complete solution."
    #         )
    #     else:
    #         raise ValueError(f"Unsupported role: {role}")


    def _encode_prompts_to_dataproto(self, prompts: list[str], meta_info: dict) -> DataProto:
        """将字符串 prompt 列表编码为 DataProto，适配 generate_sequences 输入格式"""
        tokenized = [self.model_config.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        max_len = max(len(t) for t in tokenized)
        pad_id = self.pad_token_id

        input_ids = torch.full((len(tokenized), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        position_ids = torch.zeros_like(input_ids)

        for i, t in enumerate(tokenized):
            input_ids[i, -len(t):] = torch.tensor(t, dtype=torch.long)
            attention_mask[i, -len(t):] = 1
            position_ids[i, -len(t):] = torch.arange(len(t), dtype=torch.long)

        batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=len(prompts))

        non_tensor_batch = {}
        meta_info.setdefault("eos_token_id", self.model_config.tokenizer.eos_token_id)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def _decode_responses(self, output_dataproto: DataProto) -> list[str]:
        """从 DataProto 解码生成的 response 为字符串"""
        resps = output_dataproto.batch["responses"]
        decoded = []
        for resp in resps:
            # 移除 pad 和 eos
            resp = resp[resp != self.pad_token_id]
            eos_id = self.model_config.tokenizer.eos_token_id
            if eos_id in resp:
                resp = resp[:torch.where(resp == eos_id)[0][0]]
            decoded.append(self.model_config.tokenizer.decode(resp.tolist(), skip_special_tokens=True))
        return decoded

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def _get_response_mask_for_sciagent(self, all_responses: List[List[int]], eos_token_id: int, device) -> torch.Tensor:
        """
        构建符合 SciAgent 多轮对话格式的 response_mask
        格式: | LLM生成 | 工具调用 | LLM生成 | ... | 填充 |
        对应: | 1,1,1.. | 0,0,0.. | 1,1,1.. | ... | 0,0,0.. |

        Args:
            all_responses: 包含各角色生成内容的列表 [[gen_tokens], [rev_tokens], [imp_tokens], ...]
            eos_token_id: EOS token ID，用于识别段落边界

        Returns:
            response_mask: [response_length] tensor，1表示LLM生成，0表示工具调用/填充
        """
        flattened_responses = []
        mask_values = []

        # 第一轮：Generate Agent 生成 (1)
        if all_responses:
            flattened_responses.extend(all_responses[0])
            mask_values.extend([1] * len(all_responses[0]))

        # 后续轮次：Review (1) + Improve (1) 交替
        for i in range(1, len(all_responses), 2):
            # Review 部分
            if i < len(all_responses):
                flattened_responses.extend(all_responses[i])
                mask_values.extend([1] * len(all_responses[i]))  # Review 也是LLM生成

            # Improve 部分
            if i + 1 < len(all_responses):
                flattened_responses.extend(all_responses[i + 1])
                mask_values.extend([1] * len(all_responses[i + 1]))  # Improve 也是LLM生成

        # 转换为 tensor 并 pad 到 config.response_length
        response_length = self.config.response_length
        if len(mask_values) > response_length:
            mask_values = mask_values[:response_length]
            flattened_responses = flattened_responses[:response_length]

        # 填充部分设为0
        padded_mask = mask_values + [0] * (response_length - len(mask_values))
        return torch.tensor(padded_mask, dtype=torch.long, device=device)

    @GPUMemoryLogger(role="SCIAgent", logger=logger)
    @torch.no_grad()
    def generate_sequences(
            self,
            prompts: DataProto,
            max_rounds: int = 1,
    ) -> DataProto:
        """
        批量化的 SciAgent 多智能体协作流程，生成符合原始 generate_sequences 格式的 DataProto

        支持任意 batch size，对每个样本独立执行多轮对话，但通过批量化调用提高效率。
        """
        # 原始输入
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        pmt_attention_mask = prompts.batch["attention_mask"]
        pmt_position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        # 1. 提取所有问题
        prompt_token_ids_list = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]
        problems = [self.model_config.tokenizer.decode(prompt_token_ids) for prompt_token_ids in prompt_token_ids_list]

        # 2. 准备存储结构
        all_prompts = [[] for _ in range(batch_size)]  # 每个样本的提示列表
        all_responses = [[] for _ in range(batch_size)]  # 每个样本的响应列表

        current_answers = [""] * batch_size  # 当前每个样本的答案
        completed = [False] * batch_size  # 标记样本是否已完成

        # 3. 第一轮：Generate（批量生成初步答案）
        gen_prompts = [self._build_role_prompt("generate", problem) for problem in problems]
        gen_input = self._encode_prompts_to_dataproto(gen_prompts, prompts.meta_info.copy())

        if int(os.environ["RANK"]) == 0:
            torch.save(gen_prompts, "/code/tmp/debug/Agent1_input.pt")
        gen_output = self._generate_sequences(gen_input)
        # logger.info("[debug] line 443 gen_output = self._generate_sequences(gen_input) # 3. 第一轮: Generate")

        # 处理生成结果
        for i in range(batch_size):
            gen_resp = gen_output.batch["responses"][i]
            gen_resp_clean = gen_resp[gen_resp != self.pad_token_id].tolist()
            all_responses[i].append(gen_resp_clean)
            all_prompts[i].append(gen_prompts[i])

            current_answers[i] = self._decode_responses(DataProto(
                batch=TensorDict({"responses": gen_output.batch["responses"][i:i + 1]}, batch_size=1),
                non_tensor_batch={},
                meta_info=prompts.meta_info
            ))[0]
        # logger.info(f'[debug] line 445 current_answers(处理生成结果) {int(os.environ["RANK"])}, {current_answers[0]}')
        if int(os.environ["RANK"]) == 0:
            torch.save(current_answers, "/code/tmp/debug/Agent1_output.pt")

        # 4. 迭代执行 Review-Improve
        # 准备需要 review 的样本索引
        review_indices = [i for i in range(batch_size) ]
        # 4.1 批量 Review
        rev_problems = [problems[i] for i in review_indices]
        rev_answers = [current_answers[i] for i in review_indices]
        rev_prompts = [
            self._build_role_prompt("review", problem, answer)
            for problem, answer in zip(rev_problems, rev_answers)
        ]
        if int(os.environ["RANK"]) == 0:
            torch.save(rev_prompts, "/code/tmp/debug/Agent2_input.pt")

        rev_input = self._encode_prompts_to_dataproto(rev_prompts, prompts.meta_info.copy())
        rev_output = self._generate_sequences(rev_input)
        rev_feedbacks = self._decode_responses(rev_output)
        if int(os.environ["RANK"]) == 0:
            torch.save(rev_feedbacks, "/code/tmp/debug/Agent2_output.pt")
        # logger.info("[debug] line 473 rev_output = self._generate_sequences(rev_input) # 4. 迭代执行 Review-Improve 轮次")
        # 处理 review 结果
        for idx_pos, sample_idx in enumerate(review_indices):
            rev_resp = rev_output.batch["responses"][idx_pos]
            rev_resp_clean = rev_resp[rev_resp != self.pad_token_id].tolist()
            all_responses[sample_idx].append(rev_resp_clean)
            all_prompts[sample_idx].append(rev_prompts[sample_idx])

        # 4.2 准备需要 improve 的样本索引
        imp_indices = [i for i in range(batch_size) ]
        # 4.3 批量 Improve
        imp_problems = [problems[i] for i in imp_indices]
        imp_answers = [current_answers[i] for i in imp_indices]
        imp_feedbacks = [
            rev_feedbacks[review_indices.index(i)] for i in imp_indices
        ]
        imp_prompts = [
            self._build_role_prompt("improve", problem, answer, feedback)
            for problem, answer, feedback in zip(imp_problems, imp_answers, imp_feedbacks)
        ]
        if int(os.environ["RANK"]) == 0:
            torch.save(imp_prompts, "/code/tmp/debug/Agent3_input.pt")
        imp_input = self._encode_prompts_to_dataproto(imp_prompts, prompts.meta_info.copy())
        imp_output = self._generate_sequences(imp_input)
        imp_answers_new = self._decode_responses(imp_output)
        if int(os.environ["RANK"]) == 0:
            torch.save(imp_answers_new, "/code/tmp/debug/Agent3_output.pt")
        # logger.info("[debug] line 503 imp_output = self._generate_sequences(imp_input) # 4. 迭代执行 Review-Improve 轮次")
        # 处理 improve 结果
        for idx_pos, sample_idx in enumerate(imp_indices):
            imp_resp = imp_output.batch["responses"][idx_pos]
            imp_resp_clean = imp_resp[imp_resp != self.pad_token_id].tolist()
            all_responses[sample_idx].append(imp_resp_clean)
            current_answers[sample_idx] = imp_answers_new[idx_pos]


        # 5. 拼接 (Stitching) - 构建线性序列: 角色提示 | Pmt1 | Res1 | Pmt2 | Res2 | Pmt3 | Res3
        # 定义过渡用的 Prompt 模板 (用于连接各个阶段)
        # Pmt2: 连接 Res1 和 Res2 (Review 阶段指令)
        pmt2_transition_txt = (
            "\n<|im_start|>user\n"
            "Now, you are the [Review Agent], Please identify any errors in reasoning, calculations, or application of scientific principles.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        # Pmt3: 连接 Res2 和 Res3 (Improve 阶段指令)
        pmt3_transition_txt = (
            "\n<|im_start|>user\n"
            "Now, you are the [Improve Agent], Please revise solutions based on specific review feedback while preserving correct elements.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        tokenizer = self.model_config.tokenizer
        device = idx.device
        # 将过渡文本转为 Tensor (假设所有样本共用相同的指令模板)
        pmt2_ids = torch.tensor(tokenizer.encode(pmt2_transition_txt, add_special_tokens=False), device=device)
        pmt3_ids = torch.tensor(tokenizer.encode(pmt3_transition_txt, add_special_tokens=False), device=device)
        final_input_ids_list = []
        final_response_mask_list = []
        final_agent_boundaries = []
        # 临时存储拼接后的 response 部分 (用于 responses_tensor)
        final_responses_part_list = []

        max_seq_len = 0
        max_resp_len = 0

        for i in range(batch_size):
            # 1. 获取各个片段
            # Pmt1 (去除 padding)
            # p1_ids = idx[i]

            p1_ids = gen_input[i] # 修改为 [role]+[pmt1] // Hanlard 1201-1745


            if self.pad_token_id is not None:
                p1_ids = p1_ids[p1_ids != self.pad_token_id]

            # Res1, Res2, Res3
            # 注意: all_responses[i] 应该包含 [Gen_Tokens, Rev_Tokens, Imp_Tokens]
            # 如果某一步被跳过(Completed)，需要根据实际情况处理，这里假设流程完整
            res1_ids = torch.tensor(all_responses[i][0], device=device) if len(
                all_responses[i]) > 0 else torch.tensor([], device=device)
            res2_ids = torch.tensor(all_responses[i][1], device=device) if len(
                all_responses[i]) > 1 else torch.tensor([], device=device)
            res3_ids = torch.tensor(all_responses[i][2], device=device) if len(
                all_responses[i]) > 2 else torch.tensor([], device=device)

            # 2. 构建线性序列 (Input IDs)
            # 结构: [Role] + [Pmt1] + [Res1] + [Pmt2] + [Res2] + [Pmt3] + [Res3]
            # 注意：responses_tensor 通常只存 Pmt1 之后的内容，或者根据需求存完整内容。
            # 这里按照惯例，input_ids 是全长，responses_tensor 对应 input_ids 中除去 Pmt1 的部分(或者包含 Pmt1 但用 mask 遮盖)
            # 题目要求 "responses_tensor 改把三段输出直接拼接在一起"，且 mask 对应 000|111...

            parts = [
                # (p1_ids, 0),  # Pmt1, type=0 (Prompt)
                (p1_ids, 2),  # Role+Pmt1, type=1 (Prompt) # 修改为 [role]+[pmt1] // Hanlard 1201-1745
                (res1_ids, 1),  # Res1, type=1 (Response)
                (pmt2_ids, 0),  # Pmt2, type=0
                (res2_ids, 1),  # Res2, type=1
                (pmt3_ids, 0),  # Pmt3, type=0
                (res3_ids, 1)  # Res3, type=1
            ]

            full_ids = []
            full_mask = []
            boundaries = []

            # current_idx = 0

            # 记录 Responses 部分的起始偏移量 (用于构建 responses_tensor)
            # 如果 responses_tensor 不包含 Pmt1，则从 len(p1_ids) 开始截取

            current_idx = len(p1_ids) # 修改为从response部分开始计算偏移 // Hanlard 1201-1745

            for token_ids, type_id in parts:
                length = len(token_ids)
                if length == 0: continue

                full_ids.append(token_ids)

                # 构建 Mask: 1 for Response, 0 for Prompt
                # 注意：这里构建的是对应整个 input_ids 的 mask
                mask_val = 1 if type_id>0  else 0
                full_mask.append(torch.full((length,), mask_val, dtype=torch.bool, device=device))

                # 记录 Agent Boundaries (Start, End)
                # 仅记录 Response 部分 (type_id == 1)
                if type_id == 1:
                    boundaries.append((current_idx, current_idx + length))

                current_idx += length

            # 合并
            sample_input_ids = torch.cat(full_ids)
            sample_mask = torch.cat(full_mask)

            # 提取 responses_tensor 部分 (除去 Pmt1 的其余部分，或者根据框架需求保留 Pmt1)
            # 既然要求 "responses_tensor" 对应 mask "000|1111|000...",
            # 通常意味着 responses_tensor 和 input_ids 长度一致 (或者对齐右侧)。
            # 这里我们让 responses_tensor = input_ids (对齐)

            final_input_ids_list.append(sample_input_ids)
            final_response_mask_list.append(sample_mask)
            final_agent_boundaries.append(boundaries)

            max_seq_len = max(max_seq_len, len(sample_input_ids))

        # 6. Padding & Stack

        # 初始化 Tensors
        padded_input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long, device=device)
        padded_response_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)
        padded_attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)

        # 填充数据
        for i in range(batch_size):
            seq_len = len(final_input_ids_list[i])
            padded_input_ids[i, :seq_len] = final_input_ids_list[i]
            padded_response_mask[i, :seq_len] = final_response_mask_list[i]
            padded_attention_mask[i, :seq_len] = 1  # Valid tokens

        responses_tensor = padded_input_ids  # 复用 input_ids 作为 responses_tensor
        response_mask = padded_response_mask

        # 6.1 这里的 final_attention_mask 根据模型具体要求，如果是 FlashAttn 通常只需要 input_ids 和 1D attention_mask
        # 这里为了保持接口一致，返回 1D mask (Pad Mask) 或根据原代码逻辑调整
        # 原代码是返回 block mask，这里我们返回 1D padding mask (padded_attention_mask)
        # 因为线性序列对于标准 LLM 来说 1D mask + internal causal logic 足够。
        final_attention_mask = padded_attention_mask

        if int(os.environ["RANK"]) == 0:
            torch.save(responses_tensor, "/code/tmp/debug/responses_tensor.pt")
            torch.save(response_mask, "/code/tmp/debug/response_mask.pt")
            torch.save(final_agent_boundaries, "/code/tmp/debug/agent_boundaries.pt")

        # 6.2 位置编码 (Position IDs)
        # 简单线性增长即可，因为拼接后就是单条对话
        position_ids = torch.arange(max_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size,
                                                                                                      -1)
        # 兼容 MROPE (如果原 position_ids 是 3D)
        final_position_ids = position_ids
        if prompts.batch["position_ids"].dim() == 3:
            # 简单扩展为 (bs, 4, seq_len) - 假设不需要特殊的 rotary view 调整，或者沿用线性
            final_position_ids = position_ids.unsqueeze(1).expand(-1, 4, -1)


        # 7. 构建最终 batch
        final_batch = TensorDict({
            "prompts": idx,
            "responses": responses_tensor,  # 也就是 input_ids (full sequence)
            "response_mask": response_mask,  # 000|111|000|111...
            "agent_boundaries": final_agent_boundaries,
            "input_ids": torch.cat([idx, responses_tensor], dim=-1),
            "attention_mask": final_attention_mask,
            "position_ids": final_position_ids
        }, batch_size=batch_size)

        if int(os.environ["RANK"]) == 0:
            torch.save({
                "prompts": idx,
                "responses": responses_tensor,  # 也就是 input_ids (full sequence) # 根据dp_actor.py line 245 logp是根据responses长度截取的
                "response_mask": response_mask,  # 000|111|000|111...
                "agent_boundaries": final_agent_boundaries,
                "input_ids": torch.cat([idx, responses_tensor], dim=-1),
                "attention_mask": final_attention_mask,
                "position_ids": final_position_ids
            }, "/code/tmp/debug/final_batch.pt")

        return DataProto(batch=final_batch, non_tensor_batch=prompts.non_tensor_batch, meta_info=prompts.meta_info)

    def _build_role_prompt(self, role: str, problem: str, content: str = "", feedback: str = "") -> str:
        # ... (保持原有的 _build_role_prompt 不变) ...
        # 注意：这个函数用于生成独立的 input_ids 给模型推理，
        # 而 Pmt2/Pmt3 的拼接文本是在 generate_sequences 内部手动构建的。
        """构建 SciAgent 风格的角色化 prompt"""

        def cleantext(text: str) -> str:
            if text == "":
                return text
            # 简单清理，避免 prompt 注入干扰
            return text.replace("<|im_start|>user", "").replace("<|im_start|>assistant", "").replace("<|im_end|>",
                                                                                                     "").strip()

        clean_problem = cleantext(problem)
        clean_content = cleantext(content)
        clean_feedback = cleantext(feedback)

        pmt0 = ('You are simulating the **Math Olympiad Worker System** from **SciAgent**, '
                'a unified multi-agent architecture for expert-level mathematical problem solving. '
                'You will **sequentially enact three distinct agent roles** in strict order for the given problem.\n\n'
                'For each role, follow the exact instructions below and produce only the required output format.\n\n---'
                '\n\n**[1. Generate Agent]**  '
                '\nYou are an expert mathematical problem solver. Provide a **complete, rigorous, and self-contained '
                'solution** to the problem. Present your reasoning **step by step**, including all necessary definitions, '
                'lemmas, logical deductions, and justifications. Do not self-criticize or hedge—write with full confidence. '
                'Conclude with a clearly boxed final answer.'
                '\n\n**[2. Review Agent]**  \nYou are a meticulous mathematical reviewer. '
                'Evaluate the **proposed solution** against the following criteria:  '
                '\n1. Logical correctness and internal consistency  '
                '\n2. Proper use of mathematical definitions and theorems  '
                '\n3. Completeness of argument (no unjustified leaps)  '
                '\n4. Accuracy of calculations or symbolic manipulations  '
                '\n\nRespond **EXACTLY** with one of the following:  '
                '\n- `<Pass>` if the solution is fully correct and Olympiad-rigorous  '
                '\n- `<Fail>: [concise, specific reason]` if any flaw exists (e.g., gap in logic, incorrect claim, missing case)'
                '\n\n**[3. Improve Agent]**  '
                '\nYou are a refinement specialist. Given the **original solution** and the **review feedback**, '
                'produce a **revised solution** that:  \n- Fixes all issues identified in the review  '
                '\n- Preserves all correct parts of the original reasoning  '
                '\n- Enhances clarity, rigor, or completeness where needed  '
                '\n- Maintains a clear step-by-step structure  '
                '\n\nOutput **ONLY** the improved solution—no preface, meta-commentary, or labels.'
                )


        pmt1 =f"<|im_start|>system\n"\
                f"{pmt0}\n"\
                f"<|im_end|>\n"\
                f"<|im_start|>user\n"\
                f"[PROBLEM]\n {clean_problem}\n[/PROBLEM]\n\n"\
                f"Now, you are the [Generate Agent], Please provide a complete solution demonstrating expert-level scientific reasoning.\n"\
                f"<|im_end|>\n"\
                f"<|im_start|>assistant\n"

        pmt2 = f"{pmt1}"\
               f"[PROPOSED SOLUTION]\n{clean_content}\n[/PROPOSED SOLUTION]\n\n" \
               f"<|im_end|>\n" \
               f"<|im_start|>user\n" \
               f"Now, you are the [Review Agent], Please identify any errors in reasoning, calculations, or application of scientific principles.\n" \
               f"<|im_end|>\n" \
               f"<|im_start|>assistant\n"

        pmt3 = f"{pmt2}" \
               f"[REVIEW FEEDBACK]\n{clean_feedback}\n[/REVIEW FEEDBACK]\n\n" \
               f"<|im_end|>\n" \
               f"<|im_start|>user\n" \
               f"Now, you are the [Improve Agent], Please revise solutions based on specific review feedback while preserving correct elements.\n" \
               f"<|im_end|>\n" \
               f"<|im_start|>assistant\n"

        if role == "generate":
            return pmt1
        elif role == "review":
            return pmt2
        elif role == "improve":
            return pmt3
        else:
            raise ValueError(f"Unsupported role: {role}")




    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def _generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,1,1,0,0,1,1,1]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if not self.config.free_cache_engine:
            return

        self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)
            vllm_config = self.inference_engine.llm_engine.vllm_config.model_config
            device = next(model.parameters()).device
            process_weights_after_loading(model, vllm_config, device)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank)}
            if self.model_config.lora_rank > 0
            else {}
        )

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, sock = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        if self.config.quantization is not None:
            if self.config.quantization == "fp8":
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_fp8_patches()
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {self.config.quantization}")
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            # In async mode, make sure the old lora is removed before adding the new one
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model_runner = self.inference_engine.worker.model_runner
            model = model_runner.model
            patch_vllm_moe_model_weight_loader(model)

            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
