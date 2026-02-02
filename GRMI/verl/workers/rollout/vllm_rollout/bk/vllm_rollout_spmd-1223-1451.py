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

1223: 历史错误累计
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
from typing import Any, Generator, Dict, List
import re
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
import torch.nn.functional as F
from verl.MultiAgent.multi_agent_role_prompt import get_key_prompts


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
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length, pad_sequence_to_length
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

    def _encode_prompts_to_dataproto_with_maxLen(self, prompts: list[str], meta_info: dict, max_pmt_legth) -> DataProto:
        """将字符串 prompt 列表编码为 DataProto，适配 generate_sequences 输入格式"""
        tokenized = [self.model_config.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        max_len = min( max(len(t) for t in tokenized), max_pmt_legth )
        pad_id = self.pad_token_id

        input_ids = torch.full((len(tokenized), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        position_ids = torch.zeros_like(input_ids)

        for i, t in enumerate(tokenized):
            if len(t) > max_len:
                t = t[-max_len:]

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
    def _decode_responses_wt_MaxLen(self, output_dataproto: DataProto, max_response_len) -> list[str]:
        """从 DataProto 解码生成的 response 为字符串"""
        resps = output_dataproto.batch["responses"]
        decoded = []
        for resp in resps:
            # 移除 pad 和 eos
            resp = resp[resp != self.pad_token_id]
            eos_id = self.model_config.tokenizer.eos_token_id
            if eos_id in resp:
                resp = resp[:torch.where(resp == eos_id)[0][0]].tolist()
                if len(resp) > max_response_len:
                    resp = resp[-max_response_len:]
            decoded.append(self.model_config.tokenizer.decode(resp, skip_special_tokens=True))
        return decoded

    def _decode_responses_wt_MaxLen_with_simple_think(self, output_dataproto: DataProto, max_response_len) -> list[str]:
        """从 DataProto 解码生成的 response 为字符串"""

        def simply_think_content(text: str) -> str:
            """
            如果存在 <think>...</think> 标签：
                - 提取其内部文本
                - 保留前 500 个词 + 后 500 个词（若总词数 > 1000）
                - 中间替换为 '... <SKIP MIDDLE CONTENT> ...'
                - 将原 <think>...</think> 替换为这个摘要版内容（不带标签）
            如果不存在 <think> 标签：
                - 对全文做同样处理（前500 + 后500 词）
            """
            if not text.strip():
                return text

            # 处理单个 <think> 块的摘要
            def summarize_think_content(content: str, max_half=500) -> str:
                words = content.split()
                total = len(words)
                if total <= 2 * max_half:
                    return ' '.join(words)
                else:
                    first_part = ' '.join(words[:max_half])
                    last_part = ' '.join(words[-max_half:])
                    return f"<think>{first_part} ... <SKIP MIDDLE CONTENT> ... {last_part}</think>"

            # 检查是否存在 <think>...</think>
            pattern = r'<think>(.*?)</think>'
            matches = list(re.finditer(pattern, text, flags=re.DOTALL))

            if matches:
                # 从后往前替换（避免位置偏移）
                result = text
                for match in reversed(matches):
                    inner_content = match.group(1)  # 提取 <think> 内部内容
                    summarized = summarize_think_content(inner_content)
                    # 替换整个 <think>...</think> 为摘要内容（无标签）
                    result = result[:match.start()] + summarized + result[match.end():]
                return result.strip()
            else:
                # 全文处理
                return summarize_think_content(text)

        resps = output_dataproto.batch["responses"]
        decoded = []
        for resp in resps:
            # 移除 pad 和 eos
            resp = resp[resp != self.pad_token_id]
            eos_id = self.model_config.tokenizer.eos_token_id
            if eos_id in resp:
                resp = resp[:torch.where(resp == eos_id)[0][0]].tolist()
                if len(resp) > max_response_len:
                    resp = resp[-max_response_len:]
            out_str=self.model_config.tokenizer.decode(resp, skip_special_tokens=True)
            simple_str= simply_think_content(out_str)
            decoded.append(simple_str)
        return decoded

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def _get_response_mask_for_sciagent(self, all_responses: List[List[int]], eos_token_id: int,
                                        device) -> torch.Tensor:
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

    def post_process_for_static_length(self, input_bacth) -> torch.Tensor:
        """
        对tensor进行静态长度对齐处理

        Args:
            prompts: (batch_size, current_prompt_len)
            responses: (batch_size, current_response_len)
            attention_mask: (batch_size, current_total_len)
            position_ids: (batch_size, current_total_len)

        Returns:
            处理后的tensors
        """
        prompts =  input_bacth['prompts']
        responses =  input_bacth['responses']
        attention_mask =  input_bacth['attention_mask']
        position_ids =  input_bacth['position_ids']

        batch_size = prompts.size(0)
        device = prompts.device

        # 获取当前长度
        current_prompt_len = prompts.size(1)
        current_response_len = responses.size(1)
        current_total_len = current_prompt_len + current_response_len

        # 计算填充数量
        prompt_pad_left = max(0, self.config.prompt_length - current_prompt_len)
        response_pad_right = max(0, self.config.response_length - current_response_len)

        # 处理prompts (左侧填充)
        if prompt_pad_left > 0:
            pad_tensor = torch.full((batch_size, prompt_pad_left), self.pad_token_id,
                                    dtype=prompts.dtype, device=device)
            processed_prompts = torch.cat([pad_tensor, prompts], dim=1)
        else:
            processed_prompts = prompts[:, :self.config.prompt_length]

        # 处理responses (右侧填充)
        if response_pad_right > 0:
            pad_tensor = torch.full((batch_size, response_pad_right), self.pad_token_id,
                                    dtype=responses.dtype, device=device)
            processed_responses = torch.cat([responses, pad_tensor], dim=1)
        else:
            processed_responses = responses[:, :self.config.response_length]

        # 拼接input_ids
        processed_input_ids = torch.cat([processed_prompts, processed_responses], dim=1)

        # 分离attention_mask的prompt和response部分
        prompt_mask = attention_mask[:, :current_prompt_len]
        response_mask = attention_mask[:, current_prompt_len:]

        # 处理attention_mask
        # prompt部分左侧补0
        if prompt_pad_left > 0:
            pad_zeros = torch.zeros((batch_size, prompt_pad_left),
                                    dtype=attention_mask.dtype, device=device)
            processed_prompt_mask = torch.cat([pad_zeros, prompt_mask], dim=1)
        else:
            processed_prompt_mask = prompt_mask[:, :self.config.prompt_length]

        # response部分右侧补0
        if response_pad_right > 0:
            pad_zeros = torch.zeros((batch_size, response_pad_right),
                                    dtype=attention_mask.dtype, device=device)
            processed_response_mask = torch.cat([response_mask, pad_zeros], dim=1)
        else:
            processed_response_mask = response_mask[:, :self.config.response_length]

        processed_attention_mask = torch.cat([processed_prompt_mask, processed_response_mask], dim=1)

        # 处理position_ids
        prompt_pos = position_ids[:, :current_prompt_len]
        response_pos = position_ids[:, current_prompt_len:]

        # prompt部分左侧补0
        if prompt_pad_left > 0:
            pad_zeros = torch.zeros((batch_size, prompt_pad_left),
                                    dtype=position_ids.dtype, device=device)
            processed_prompt_pos = torch.cat([pad_zeros, prompt_pos], dim=1)
        else:
            processed_prompt_pos = prompt_pos[:, :self.config.prompt_length]

        # response部分：右侧接着数数
        if response_pos.size(1) > 0:
            # 获取每个样本response部分的最后一个位置
            last_pos = response_pos.max(dim=1, keepdim=True).values
        else:
            # 如果response部分为空，则从0开始
            last_pos = torch.zeros((batch_size, 1), dtype=position_ids.dtype, device=device)

        if response_pad_right > 0:
            # 创建递增序列
            increment = torch.arange(1, response_pad_right + 1,
                                     dtype=position_ids.dtype, device=device)
            increment = increment.unsqueeze(0).expand(batch_size, -1)
            pad_pos = last_pos + increment

            processed_response_pos = torch.cat([response_pos, pad_pos], dim=1)
        else:
            processed_response_pos = response_pos[:, :self.config.response_length]

        processed_position_ids = torch.cat([processed_prompt_pos, processed_response_pos], dim=1)

        return {
            "input_ids": processed_input_ids.to(torch.long),
            "attention_mask": processed_attention_mask.to(torch.long),
            "position_ids": processed_position_ids.to(torch.long),
            "prompts": processed_prompts.to(torch.long),
            "responses": processed_responses.to(torch.long)
        }
    def remove_think_tags(self, text: str) -> str:
        """移除所有 <think>...</think> 标签及其内容；
        如果没有找到任何 <think>...</think> 标签，则返回 text 的最后 500 个单词。
        """
        if text == "":
            return text

        pattern = r'<think>.*?</think>'
        # 检查是否存在匹配项
        if re.search(pattern, text, flags=re.DOTALL):
            # 如果有匹配，移除所有 <think>...</think> 并返回清理后的内容
            cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
            return cleaned.strip()
        else:
            # 如果没有匹配，返回最后 500 个单词
            words = text.split()
            last_500_words = words[-500:]
            return ' '.join(last_500_words)

    def extract_fail_feedback(self, review_text: str) -> str:
        """
        从 Review 输出中提取 <Fail> 部分的反馈内容。
        假设格式如：<Fail>: 错误描述\n 或包含关键错误信息。
        若无明确 <Fail>，但整体是失败反馈，也可返回全文。
        """
        review_text = self.remove_think_tags(review_text)
        review_texts = review_text.split("\n")
        res = []
        for text in review_texts:
            if "<Fail>: [" in text and len(text) < 500:
                res.append(text)
        if res:
            return "\n".join(res)
        else:
            return ""
    @GPUMemoryLogger(role="SCIAgent", logger=logger)
    @torch.no_grad()
    def generate_sequences(
            self,
            prompts: DataProto,
    ) -> DataProto:
        """
        批量化的 SciAgent 多智能体协作流程，生成符合原始 generate_sequences 格式的 DataProto
        支持 Review-Improve 循环 (最大5次迭代)，直到 Review 通过
        """


        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        batch_size = idx.size(0)
        device = idx.device

        #######################################  多重长度约束  ###########################################
        max_seq_length = self.config.response_length + self.config.prompt_length
        # assert self.config.prompt1_length + self.config.response1_length <= self.config.prompt2_length
        # assert self.config.prompt2_length + self.config.response2_length <= self.config.prompt3_length
        # assert self.config.prompt3_length + self.config.response3_length <= max_seq_length
        ###############################################################################################

        # 提取所有问题
        prompt_token_ids_list = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]
        problems = [self.model_config.tokenizer.decode(prompt_token_ids) for prompt_token_ids in prompt_token_ids_list]

        ####################################### <Agent-1: Generate> #######################################
        gen_prompts = [self._build_CMATHrole_zh_prompt_selfdefine("generate", problem) for problem in problems]
        gen_input = self._encode_prompts_to_dataproto_with_maxLen(gen_prompts,
                                                                  prompts.meta_info.copy(), self.config.prompt1_length)
        gen_output = self._generate_sequences(gen_input)
        agent1_outputs = self._decode_responses_wt_MaxLen_with_simple_think(gen_output, self.config.response1_length)  # str
        ####################################### <Agent-1> ###################################################

        # ====== 初始化循环变量 ======
        current_answers = agent1_outputs.copy()  # 当前答案 (动态更新)
        last_review_outputs = [""] * batch_size  # 保存最后一次review输出 (用于拼接)
        final_answers = [""] * batch_size  # 最终输出 (Agent-3)
        historical_fails = [[] for _ in range(batch_size)]  # List[List[str]]
        passed = [False] * batch_size  # 标记是否已通过review


        # ====== Review-Improve 循环 ======
        for iter_num in range(int(self.config.extra['max_iter'])):
            # 1. 收集需要Review的样本索引 (未通过的样本)
            active_indices = [i for i in range(batch_size) if not passed[i]]
            if not active_indices:  # 所有样本已通过
                break

            # 2. 批量执行 Review (Agent-2)
            active_problems = [problems[i] for i in active_indices]
            active_answers = [current_answers[i] for i in active_indices]
            rev_prompts = [self._build_CMATHrole_zh_prompt_selfdefine("review", p, a) for p, a in
                           zip(active_problems, active_answers)]
            rev_input = self._encode_prompts_to_dataproto_with_maxLen(rev_prompts, prompts.meta_info.copy(),
                                                                      self.config.prompt2_length)
            rev_output = self._generate_sequences(rev_input)
            active_review_outputs = self._decode_responses_wt_MaxLen_with_simple_think(rev_output, self.config.response2_length)
            active_review_outputs = [f"(Review Times: <{iter_num+1}>)\n"+x for x in active_review_outputs]

            print(f"vllm_rollout_spmd.py line-629 (Review Times: <{iter_num+1}>): {len(active_indices)} active samples")

            # 3. 处理Review结果 & 标记通过样本
            improve_indices = []  # 需要Improve的样本索引
            for idx_in_active, global_idx in enumerate(active_indices):
                review_output = active_review_outputs[idx_in_active]
                last_review_outputs[global_idx] = review_output  # 保存最后一次review输出

                if "<Pass>" in review_output and "<Fail>" not in review_output:
                    passed[global_idx] = True  # 标记为已通过
                else:
                    improve_indices.append(global_idx)  # 默认加一次improve
                    # ✅ 新增：提取当前 Fail 并加入历史记录
                    fail_feedback = self.extract_fail_feedback(review_output)
                    if fail_feedback:
                        # 可加上轮次标记，便于追溯
                        annotated_fail = f"(Iteration {iter_num + 1}) {fail_feedback}"
                        historical_fails[global_idx].append(annotated_fail)


            # 4. 批量执行 Improve (Agent-3) 仅对未通过样本
            if improve_indices:
                imp_problems = [problems[i] for i in improve_indices]
                imp_answers = [current_answers[i] for i in improve_indices]
                # 从review输出中提取反馈文本
                # imp_feedbacks = [
                #     last_review_outputs[i]#.replace("[FAIL]", "").strip()
                #     for i in improve_indices
                # ]
                # ✅ 构造累积的历史失败文本
                imp_historical_fails = []
                for i in improve_indices:
                    if historical_fails[i]:
                        hist_str = "\n".join(historical_fails[i])
                        fail_context = f"Historical Fails:\n{hist_str}\n\nCurrent Fails:\n{last_review_outputs[i]}"
                    else:
                        fail_context = f"Current Fails:\n{last_review_outputs[i]}"
                    imp_historical_fails.append(fail_context)

                imp_prompts = [
                    self._build_CMATHrole_zh_prompt_selfdefine("improve", p, a, f)
                    for p, a, f in zip(imp_problems, imp_answers, imp_historical_fails)
                ]
                imp_input = self._encode_prompts_to_dataproto_with_maxLen(imp_prompts,
                                                                          prompts.meta_info.copy(),
                                                                          self.config.prompt3_length)
                imp_output = self._generate_sequences(imp_input)
                improved_answers = self._decode_responses_wt_MaxLen_with_simple_think(imp_output, self.config.response3_length)
                # 更新当前答案
                for idx_in_imp, global_idx in enumerate(improve_indices):
                    current_answers[global_idx] = improved_answers[idx_in_imp]

        # ====== 最终处理 ======
        # 1. 收集已通过样本 (需要最终润色)
        pass_indices = [i for i in range(batch_size) if passed[i]]
        if pass_indices:
            #######################
            final_imp_feedbacks = []
            for i in pass_indices:
                if historical_fails[i]:
                    hist_str = "\n".join(historical_fails[i])
                    feedback = f"Historical Fails:\n{hist_str}\n\nFinal Review:\n{last_review_outputs[i]}"
                else:
                    feedback = last_review_outputs[i]
                final_imp_feedbacks.append(feedback)
            ########################
            # final_imp_prompts = [
            #     self._build_CMATHrole_zh_prompt_selfdefine("improve", problems[i], current_answers[i], last_review_outputs[i])
            #     for i in pass_indices  #
            # ]
            final_imp_prompts = [
                self._build_CMATHrole_zh_prompt_selfdefine("improve", problems[i], current_answers[i], f)
                for i, f in zip(pass_indices, final_imp_feedbacks)
            ]

            final_imp_input = self._encode_prompts_to_dataproto_with_maxLen(final_imp_prompts,
                                                                            prompts.meta_info.copy(),
                                                                            self.config.prompt3_length)
            final_imp_output = self._generate_sequences(final_imp_input)
            final_pass_answers = self._decode_responses_wt_MaxLen_with_simple_think(final_imp_output,self.config.response3_length)
            for idx_in_pass, global_idx in enumerate(pass_indices):
                final_answers[global_idx] = final_pass_answers[idx_in_pass]

        # 2. 未通过样本：使用最后一次改进的答案
        for i in range(batch_size):
            if not passed[i]:
                final_answers[i] = current_answers[i]

        # ====== 设置输出变量 (用于拼接) ======
        agent2_outputs = last_review_outputs  # 最后一次Review输出 (含[PASS]/[FAIL])
        # agent2_outputs = ["\n---\n".join(faillist) for faillist in historical_fails]  # 最后一次Review输出 (含[PASS]/[FAIL])
        # for ii,faillist in enumerate(historical_fails):
        #     print(ii,"\n---\n".join(faillist))


        agent3_outputs = final_answers  # 最终答案 (已润色)

        # 拼接 (Stitching) - 构建线性序列: [角色提示] | Pmt1 | Res1 | Pmt2 | Res2 | Pmt3 | Res3
        pmt1_transition_txt,pmt2_transition_txt,pmt3_transition_txt = self._build_CMATHrole_zh_prompt_selfdefine(role="all")

        tokenizer = self.model_config.tokenizer
        # 将过渡文本转为 Tensor (假设所有样本共用相同的指令模板)
        pmt1_transition_tensor = torch.tensor(tokenizer.encode(pmt1_transition_txt, add_special_tokens=False),
                                              dtype=torch.long,
                                              device=device)
        pmt2_ids = torch.tensor(tokenizer.encode(pmt2_transition_txt, add_special_tokens=False), dtype=torch.long, device=device)
        pmt3_ids = torch.tensor(tokenizer.encode(pmt3_transition_txt, add_special_tokens=False), dtype=torch.long, device=device)

        batch_global_response_ids = []
        batch_global_response_mask = []

        for i in range(batch_size):
            res1_ids = torch.tensor(tokenizer.encode(agent1_outputs[i], add_special_tokens=False), dtype=torch.long,device=device)
            res2_ids = torch.tensor(tokenizer.encode(agent2_outputs[i], add_special_tokens=False), dtype=torch.long,device=device)
            res3_ids = torch.tensor(tokenizer.encode(agent3_outputs[i], add_special_tokens=False), dtype=torch.long,device=device)
            global_response_parts = [
                (res1_ids, 1),  # Res1, type=1 (Response)
                (pmt2_ids, 0),  # Pmt2, type=0
                (res2_ids, 1),  # Res2, type=1
                (pmt3_ids, 0),  # Pmt3, type=0
                (res3_ids, 1)  # Res3, type=1
            ]
            global_response_ids = []
            global_response_mask = []
            for token_ids, mask_val in global_response_parts:
                length = len(token_ids)
                global_response_ids.append(token_ids)
                global_response_mask.append(torch.full((length,), mask_val, dtype=torch.long, device=device))

            batch_global_response_ids.append(torch.cat(global_response_ids, dim=-1))
            batch_global_response_mask.append(torch.cat(global_response_mask, dim=-1))

        valid_response_len = [x.shape[-1] for x in batch_global_response_ids]
        max_res_len = max(valid_response_len)


        batch_global_response_ids = [
            F.pad(x, (0, max_res_len - x.size(-1)), value=self.pad_token_id)[:self.config.response_length] for x in
            batch_global_response_ids]
        batch_global_response_mask = [
            F.pad(x, (0, max_res_len - x.size(-1)), value=0)[:self.config.response_length] for x in
            batch_global_response_mask]

        #  to [bs, res-len]
        batch_global_response_ids = torch.stack(batch_global_response_ids, dim=0).to(torch.long)
        batch_global_response_mask = torch.stack(batch_global_response_mask, dim=0).to(torch.long)

        # Agent-1 prompt is global prompt
        batch_global_prompt_ids = torch.cat([gen_input.batch["input_ids"].to(torch.long).to(device),
                                             pmt1_transition_tensor.unsqueeze(0).expand(batch_size, -1)], dim=1).to(torch.long)


        batch_global_prompt_mask = (batch_global_prompt_ids != self.pad_token_id).int().to(device)
        batch_global_prompt_position_ids = gen_input.batch['position_ids'].to(device)

        batch_seq_ids = torch.cat([batch_global_prompt_ids, batch_global_response_ids.to(device)], dim=1).to(torch.long)
        baTch_seq_mask = torch.cat([batch_global_prompt_mask, batch_global_response_mask.to(device)], dim=1).to(torch.long)



        res_len = batch_global_response_ids.size(1)
        last_prompt_pos = batch_global_prompt_position_ids[:, -1:]  # [bs, 1]
        response_position_ids = last_prompt_pos + torch.arange(1, res_len + 1 + pmt1_transition_tensor.size(-1),
                                                               device=last_prompt_pos.device).unsqueeze(0)
        batch_seq_position_ids = torch.cat([batch_global_prompt_position_ids, response_position_ids.to(device)],
                                           dim=1).to(torch.long)

        ## 根据 self.config.prompt_length 和 self.config.response_length 静态长度后处理
        # 7. 构建最终 batch
        terdict = {
            "prompts": batch_global_prompt_ids.to(torch.long),  # Global Prompt
            "responses": batch_global_response_ids.to(torch.long),  # Global Response
            "input_ids": batch_seq_ids.to(torch.long),
            "attention_mask": baTch_seq_mask.to(torch.long),
            "position_ids": batch_seq_position_ids.to(torch.long)
        }
        terdict = self.post_process_for_static_length(terdict)


        assert terdict["input_ids"].shape[-1] == max_seq_length

        final_batch = TensorDict(terdict, batch_size=batch_size)

        return DataProto(batch=final_batch, non_tensor_batch=prompts.non_tensor_batch, meta_info=prompts.meta_info)



    def _build_CMATHrole_zh_prompt_selfdefine(self, role: str, problem: str = "", content: str = "", feedback: str = "") -> str:
        """构建 SciAgent 风格的角色化 prompt"""

        key1,key2,key3 = get_key_prompts(self.config.extra["roles"], self.config.extra['role_config_path'])

        pmt1_transition_txt = "[PROPOSED SOLUTION]\n"
        # Pmt2: 连接 Res1 和 Res2 (Review 阶段指令)
        pmt2_transition_txt =  "[/PROPOSED SOLUTION]\n\n<|im_start|>user\n" \
                               f"{key2}" \
                               "<|im_end|>\n" \
                               "<|im_start|>assistant\n[REVIEW FEEDBACK]\n"

        # Pmt3: 连接 Res2 和 Res3 (Improve 阶段指令)
        pmt3_transition_txt = (
            "[/REVIEW FEEDBACK]\n\n<|im_start|>user\n"
            f"{key3}" 
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        if role == "all":
            return pmt1_transition_txt, pmt2_transition_txt, pmt3_transition_txt


        def cleantext(text: str) -> str:
            if text == "":
                return text
            # 简单清理，避免 prompt 注入干扰
            return text.replace("<|im_start|>user", "").replace("<|im_start|>assistant", "").replace("<|im_end|>",
                                                                                                     "").strip()

        # def remove_think_tags(text: str) -> str:
        #     """移除所有 <think>...</think> 标签及其内容"""
        #     if text == "":
        #         return text
        #     # 使用正则表达式移除标签及其中间内容
        #     pattern = r'<think>.*?</think>'
        #     return re.sub(pattern, '', text, flags=re.DOTALL).strip()


        clean_problem = cleantext(problem)
        clean_content = self.remove_think_tags(cleantext(content))
        clean_feedback = self.remove_think_tags(cleantext(feedback))


        pmt1 = f"<|im_start|>user\n" \
               f"{key1}" \
               f"{clean_problem}" \
               f"<|im_end|>\n" \
               f"<|im_start|>assistant\n"

        pmt2 = f"{pmt1}" \
               f"[PROPOSED SOLUTION]\n{clean_content}\n" \
               f"[/PROPOSED SOLUTION]\n\n" \
               f"<|im_end|>\n" \
               f"<|im_start|>user\n" \
               f"{key2}" \
               f"<|im_end|>\n" \
               f"<|im_start|>assistant\n"

        pmt3 = f"{pmt2}" \
               f"[REVIEW FEEDBACK]\n{clean_feedback}\n[/REVIEW FEEDBACK]\n\n" \
               f"<|im_end|>\n" \
               f"<|im_start|>user\n" \
               f"{key3}" \
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
            # print("vllm_rollout_spmd.py line-1346:",kwargs)
        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id,
                                                lora_path="/simon-stub-path")
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
