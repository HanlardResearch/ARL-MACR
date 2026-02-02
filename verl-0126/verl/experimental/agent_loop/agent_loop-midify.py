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
import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4
import re
import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.experimental.reward_loop import RewardLoopWorker
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
import torch.nn.functional as F
from verl.MultiAgent.multi_agent_role_prompt import get_key_prompts
from typing import Any, Generator, Dict, List



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(self.server_handles)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        _, _, server = self.weighted_serveres[0]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=uuid4().hex,  # use new request_id for each turn
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfig,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (DictConfigWrap): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
            dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
            dataset_config (DictConfig): Dataset config.
        """
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.dataset_config = dataset_config
        self.apply_chat_template_kwargs = dataset_config.get("apply_chat_template_kwargs", {})
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Extract images and videos from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys "images" and "videos".
        """
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = await self.dataset_cls.process_vision_info(
                messages, image_patch_size=self.processor.image_processor.patch_size, config=self.dataset_config
            )
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            # split the videos and according metadatas
            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadatas=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        self.config = config

        # for recipe to change
        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(config, server_handles)

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_router_address = reward_router_address

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        use_reward_loop = True if self.config.reward_model.use_reward_loop else None
        self.use_reward_loop = use_reward_loop
        if use_reward_loop and not hasattr(self, "reward_loop_worker"):
            self.reward_loop_worker = RewardLoopWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(self.config, self.reward_router_address)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def _encode_prompts_to_dataproto(self, prompts: list[str], meta_info: dict) -> DataProto:
        """将字符串 prompt 列表编码为 DataProto，适配 generate_sequences 输入格式"""
        tokenized = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
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
        meta_info.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def _encode_prompts_to_dataproto_with_maxLen(self, prompts: list[str], meta_info: dict, max_pmt_legth) -> DataProto:
        """将字符串 prompt 列表编码为 DataProto，适配 generate_sequences 输入格式"""
        tokenized = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        max_len = min(max(len(t) for t in tokenized), max_pmt_legth)
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
        meta_info.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def _decode_responses(self, output_dataproto: DataProto) -> list[str]:
        """从 DataProto 解码生成的 response 为字符串"""
        resps = output_dataproto.batch["responses"]
        decoded = []
        for resp in resps:
            # 移除 pad 和 eos
            resp = resp[resp != self.pad_token_id]
            eos_id = self.tokenizer.eos_token_id
            if eos_id in resp:
                resp = resp[:torch.where(resp == eos_id)[0][0]]
            decoded.append(self.tokenizer.decode(resp.tolist(), skip_special_tokens=True))
        return decoded

    def _decode_responses_wt_MaxLen(self, output_dataproto: DataProto, max_response_len) -> list[str]:
        """从 DataProto 解码生成的 response 为字符串"""
        resps = output_dataproto.batch["responses"]
        decoded = []
        for resp in resps:
            # 移除 pad 和 eos
            resp = resp[resp != self.pad_token_id]
            eos_id = self.tokenizer.eos_token_id
            if eos_id in resp:
                resp = resp[:torch.where(resp == eos_id)[0][0]].tolist()
                if len(resp) > max_response_len:
                    resp = resp[-max_response_len:]
            decoded.append(self.tokenizer.decode(resp, skip_special_tokens=True))
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
            eos_id = self.tokenizer.eos_token_id
            if eos_id in resp:
                resp = resp[:torch.where(resp == eos_id)[0][0]].tolist()
                if len(resp) > max_response_len:
                    resp = resp[-max_response_len:]
            out_str = self.tokenizer.decode(resp, skip_special_tokens=True)
            simple_str = simply_think_content(out_str)
            decoded.append(simple_str)
        return decoded


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
        response_length = self.config.actor_rollout_ref.rollout.response_length
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
        prompts = input_bacth['prompts']
        responses = input_bacth['responses']
        attention_mask = input_bacth['attention_mask']
        position_ids = input_bacth['position_ids']

        batch_size = prompts.size(0)
        device = prompts.device

        # 获取当前长度
        current_prompt_len = prompts.size(1)
        current_response_len = responses.size(1)
        current_total_len = current_prompt_len + current_response_len

        # 计算填充数量
        prompt_pad_left = max(0, self.config.actor_rollout_ref.rollout.prompt_length - current_prompt_len)
        response_pad_right = max(0, self.config.actor_rollout_ref.rollout.response_length - current_response_len)

        # 处理prompts (左侧填充)
        if prompt_pad_left > 0:
            pad_tensor = torch.full((batch_size, prompt_pad_left), self.pad_token_id,
                                    dtype=prompts.dtype, device=device)
            processed_prompts = torch.cat([pad_tensor, prompts], dim=1)
        else:
            processed_prompts = prompts[:, :self.config.actor_rollout_ref.rollout.prompt_length]

        # 处理responses (右侧填充)
        if response_pad_right > 0:
            pad_tensor = torch.full((batch_size, response_pad_right), self.pad_token_id,
                                    dtype=responses.dtype, device=device)
            processed_responses = torch.cat([responses, pad_tensor], dim=1)
        else:
            processed_responses = responses[:, :self.config.actor_rollout_ref.rollout.response_length]

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
            processed_prompt_mask = prompt_mask[:, :self.config.actor_rollout_ref.rollout.prompt_length]

        # response部分右侧补0
        if response_pad_right > 0:
            pad_zeros = torch.zeros((batch_size, response_pad_right),
                                    dtype=attention_mask.dtype, device=device)
            processed_response_mask = torch.cat([response_mask, pad_zeros], dim=1)
        else:
            processed_response_mask = response_mask[:, :self.config.actor_rollout_ref.rollout.response_length]

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
            processed_prompt_pos = prompt_pos[:, :self.config.actor_rollout_ref.rollout.prompt_length]

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
            processed_response_pos = response_pos[:, :self.config.actor_rollout_ref.rollout.response_length]

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

    def extract_think_content(self, text: str) -> str:
        """提取所有 <think>...</think> 标签中的内容（不包含标签本身）；
        如果没有找到任何 <think>...</think> 标签，则返回 text 的最后 500 个单词。
        """
        if text == "":
            return text
        pattern = r'<think>(.*?)</think>'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            # 提取所有匹配的内容，保留原始换行和空格
            return ''.join(matches).strip()
        else:
            # 如果没有匹配，返回原文本
            return text

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

    import re

    def split_thought_and_output(self, text):
        """
        Split the input text into 'thought' and 'non-thought' parts.

        Args:
            text (str): The raw model output possibly containing <think>...</think>.

        Returns:
            tuple: (thought_str, non_thought_str)
                   If no valid <think>...</think> found, thought_str = '', non_thought_str = original text.
        """
        if not isinstance(text, str):
            return '', str(text)

        # Normalize: make search case-insensitive by converting to lower for matching,
        # but preserve original text for output.
        lower_text = text.lower()

        # Find positions of <think> and </think>
        think_start = lower_text.find('<think>')
        think_end = lower_text.find('</think>')

        # Case 1: Both tags present and in correct order
        if think_start != -1 and think_end != -1 and think_start < think_end:
            thought = text[think_start + 8:think_end]  # +8 to skip '<think>'
            non_thought = text[:think_start] + text[think_end + 9:]  # +9 to skip '</think>'
            return thought.strip(), non_thought.strip()

        # Case 2: Only <think> present (no closing) → treat rest as thought?
        # But safer: assume unclosed => not valid, so no thought
        elif think_start != -1 and (think_end == -1 or think_end < think_start):
            # Heuristic: if <think> exists but no </think>, maybe it's unclosed.
            # We assume everything after <think> is thought, but this is risky.
            # However, in practice, models often omit closing tag accidentally.
            # So: if <think> exists and no </think>, take from <think> to end as thought.
            # But only if it's likely intended.
            # We'll do: if <think> found and no </think>, extract from <think> onward as thought.
            thought = text[think_start + 8:]
            non_thought = text[:think_start]
            return thought.strip(), non_thought.strip()

        # Case 3: Only </think> present (no opening) → ignore it, whole text is non-thought
        elif think_end != -1 and think_start == -1:
            return '', text.strip()

        # Case 4: No tags at all
        else:
            return '', text.strip()

    @tqbridge()
    def generate_sequences(
            self,
            prompts: DataProto,
    ) -> DataProto:
        """
        批量化的 SciAgent 多智能体协作流程，生成符合原始 generate_sequences 格式的 DataProto
        支持 Review-Improve 循环 (最大5次迭代)，直到 Review 通过
        """
        print(prompts)
        torch.save(prompts, "/code/tmp/debug/prompts.pt")
        print("/code/tmp/debug/prompts.pt")

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        batch_size = idx.size(0)
        device = idx.device

        #######################################  多重长度约束  ###########################################
        max_seq_length = self.config.actor_rollout_ref.rollout.response_length + self.config.actor_rollout_ref.rollout.prompt_length
        ###############################################################################################

        # 提取所有问题
        prompt_token_ids_list = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]
        problems = [self.tokenizer.decode(prompt_token_ids) for prompt_token_ids in prompt_token_ids_list]

        ####################################### <Agent-1: Generate> #######################################
        gen_prompts = [self._build_CMATHrole_zh_prompt_selfdefine("generate", problem) for problem in problems]
        gen_input = self._encode_prompts_to_dataproto_with_maxLen(gen_prompts,
                                                                  prompts.meta_info.copy(),
                                                                  self.config.actor_rollout_ref.rollout.prompt1_length)
        gen_output = self._generate_sequences(gen_input)
        agent1_outputs = self._decode_responses_wt_MaxLen(gen_output,
                                                          self.config.actor_rollout_ref.rollout.response1_length)  # str
        ####################################### <Agent-1> ###################################################

        ####################################### <Summary Agent> ################################################
        # # 1. Split into think and no-think contents
        # agent1_thk_list = []
        # agent1_non_thk_list = []
        # summary_prompt="""Please summary the reasoning and thought process, the output should be less than 200 words. \nThought process:"""
        # for agent1_output in agent1_outputs:
        #     thk_str, non_thk_str = self.split_thought_and_output(agent1_output)
        #     agent1_thk_list.append(f"{summary_prompt}{thk_str}")
        #     agent1_non_thk_list.append(non_thk_str)
        # # 2. Summary the thinking content
        # summary_input = self._encode_prompts_to_dataproto_with_maxLen(agent1_thk_list,
        #                                                           prompts.meta_info.copy(), self.config.actor_rollout_ref.rollout.prompt1_length)
        # summary_output = self._generate_sequences(summary_input)
        # summary_output_str_list_with_thk = self._decode_responses_wt_MaxLen(summary_output, self.config.actor_rollout_ref.rollout.response1_length)  # str
        # summarry_thinks = [self.split_thought_and_output(x)[1] for x in summary_output_str_list_with_thk]
        # # 3. replace the think content as summaried think content, concat the summaried think content and no-think contents
        # agent1_outputs = [f"<think>{x}</think>{y}"   for x,y in zip(summarry_thinks, agent1_non_thk_list)]
        ####################################### <Summary Agent> ############################################

        # ====== 初始化循环变量 ======
        current_answers = agent1_outputs.copy()  # 当前答案 (动态更新)
        last_review_outputs = [""] * batch_size  # 保存最后一次review输出 (用于拼接)
        final_answers = [""] * batch_size  # 最终输出 (Agent-3)
        historical_fails = [[] for _ in range(batch_size)]  # List[List[str]]
        passed = [False] * batch_size  # 标记是否已通过review

        # ====== Review-Improve 循环 ======
        for iter_num in range(int(self.config.actor_rollout_ref.rollout.extra['max_iter'])):
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
                                                                      self.config.actor_rollout_ref.rollout.prompt2_length)
            rev_output = self._generate_sequences(rev_input)
            active_review_outputs = self._decode_responses_wt_MaxLen(rev_output,
                                                                     self.config.actor_rollout_ref.rollout.response2_length)
            active_review_outputs = [f"(Review Times: <{iter_num + 1}>)\n" + x for x in active_review_outputs]

            print(
                f"vllm_rollout_spmd.py line-629 (Review Times: <{iter_num + 1}>): {len(active_indices)} active samples")

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
                                                                          self.config.actor_rollout_ref.rollout.prompt3_length)
                imp_output = self._generate_sequences(imp_input)
                improved_answers = self._decode_responses_wt_MaxLen(imp_output,
                                                                    self.config.actor_rollout_ref.rollout.response3_length)
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
                                                                            self.config.actor_rollout_ref.rollout.prompt3_length)
            final_imp_output = self._generate_sequences(final_imp_input)
            final_pass_answers = self._decode_responses_wt_MaxLen(final_imp_output,
                                                                  self.config.actor_rollout_ref.rollout.response3_length)
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
        pmt1_transition_txt, pmt2_transition_txt, pmt3_transition_txt = self._build_CMATHrole_zh_prompt_selfdefine(
            role="all")

        tokenizer = self.tokenizer
        # 将过渡文本转为 Tensor (假设所有样本共用相同的指令模板)
        pmt1_transition_tensor = torch.tensor(tokenizer.encode(pmt1_transition_txt, add_special_tokens=False),
                                              dtype=torch.long,
                                              device=device)
        pmt2_ids = torch.tensor(tokenizer.encode(pmt2_transition_txt, add_special_tokens=False), dtype=torch.long,
                                device=device)
        pmt3_ids = torch.tensor(tokenizer.encode(pmt3_transition_txt, add_special_tokens=False), dtype=torch.long,
                                device=device)

        batch_global_response_ids = []
        batch_global_response_mask = []

        for i in range(batch_size):
            res1_ids = torch.tensor(tokenizer.encode(agent1_outputs[i], add_special_tokens=False), dtype=torch.long,
                                    device=device)
            res2_ids = torch.tensor(tokenizer.encode(agent2_outputs[i], add_special_tokens=False), dtype=torch.long,
                                    device=device)
            res3_ids = torch.tensor(tokenizer.encode(agent3_outputs[i], add_special_tokens=False), dtype=torch.long,
                                    device=device)
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
            F.pad(x, (0, max_res_len - x.size(-1)), value=self.pad_token_id)[
                :self.config.actor_rollout_ref.rollout.response_length] for x in
            batch_global_response_ids]
        batch_global_response_mask = [
            F.pad(x, (0, max_res_len - x.size(-1)), value=0)[:self.config.actor_rollout_ref.rollout.response_length] for
            x in
            batch_global_response_mask]

        #  to [bs, res-len]
        batch_global_response_ids = torch.stack(batch_global_response_ids, dim=0).to(torch.long)
        batch_global_response_mask = torch.stack(batch_global_response_mask, dim=0).to(torch.long)

        # Agent-1 prompt is global prompt
        batch_global_prompt_ids = torch.cat([gen_input.batch["input_ids"].to(torch.long).to(device),
                                             pmt1_transition_tensor.unsqueeze(0).expand(batch_size, -1)], dim=1).to(
            torch.long)

        batch_global_prompt_mask = (batch_global_prompt_ids != self.pad_token_id).int().to(device)
        batch_global_prompt_position_ids = gen_input.batch['position_ids'].to(device)

        batch_seq_ids = torch.cat([batch_global_prompt_ids, batch_global_response_ids.to(device)], dim=1).to(torch.long)
        baTch_seq_mask = torch.cat([batch_global_prompt_mask, batch_global_response_mask.to(device)], dim=1).to(
            torch.long)

        res_len = batch_global_response_ids.size(1)
        last_prompt_pos = batch_global_prompt_position_ids[:, -1:]  # [bs, 1]
        response_position_ids = last_prompt_pos + torch.arange(1, res_len + 1 + pmt1_transition_tensor.size(-1),
                                                               device=last_prompt_pos.device).unsqueeze(0)
        batch_seq_position_ids = torch.cat([batch_global_prompt_position_ids, response_position_ids.to(device)],
                                           dim=1).to(torch.long)

        ## 根据 self.config.actor_rollout_ref.rollout.prompt_length 和 self.config.actor_rollout_ref.rollout.response_length 静态长度后处理
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

    def _build_CMATHrole_zh_prompt_selfdefine(self, role: str, problem: str = "", content: str = "",
                                              feedback: str = "") -> str:
        """构建 SciAgent 风格的角色化 prompt"""

        key1, key2, key3 = get_key_prompts(self.config.actor_rollout_ref.rollout.extra["roles"],
                                           self.config.actor_rollout_ref.rollout.extra['role_config_path'])

        pmt1_transition_txt = "[PROPOSED SOLUTION]\n"
        # Pmt2: 连接 Res1 和 Res2 (Review 阶段指令)
        pmt2_transition_txt = "[/PROPOSED SOLUTION]\n\n<|im_start|>user\n" \
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

    @tqbridge()
    async def _generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

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
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)

        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=self.config.data,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, **kwargs)

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                experts_tensor = torch.from_numpy(output.routed_experts)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs)
        await self._compute_score(
            output,
            prompts=prompt_output["input_ids"],
            responses=response_output["input_ids"],
            attention_mask=attention_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            kwargs=kwargs,
        )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        images = output.multi_modal_data.get("images")
        videos = output.multi_modal_data.get("videos")
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = self.processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        video_grid_thw = multi_modal_inputs.get("video_grid_thw")

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, output, prompts, responses, attention_mask, input_ids, position_ids, kwargs):
        """Compute reward score for single sample."""
        enable_async_reward = (
            self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
        ) or not self.config.reward_model.enable

        if output.reward_score is None and enable_async_reward and self.use_reward_loop:
            batch = TensorDict(
                {
                    "prompts": prompts,  # [1, prompt_length]
                    "responses": responses,  # [1, response_length]
                    "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                    "input_ids": input_ids,  # [1, prompt_length + response_length]
                    "position_ids": position_ids,
                },
                batch_size=1,
            )
            non_tensor_batch = {
                **{k: np.array([v]) for k, v in kwargs.items()},
                "__num_turns__": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            }

            data = DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
            )
            result = await self.reward_loop_worker.compute_score.remote(data)
            output.reward_score = result["reward_score"]
            output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    def create_transferqueue_client(
        self,
    ):
        """Create a client for data system (TransferQueue)."""
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)

        self.tq_client = create_transferqueue_client(
            client_id=f"AgentLoopWorker_{client_name}",
            config=self.config.transfer_queue,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        rm_resource_pool: RayResourcePool = None,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
            rollout_resource_pool (RayResourcePool): Resource pool for actor rollout (Colocate or Standalone mode).
            rm_resource_pool (RayResourcePool): Resource pool for reward model (Standalone mode).
        """
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward_loop import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

        self._initialize_llm_servers(rollout_resource_pool)
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self, rollout_resource_pool: RayResourcePool):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and rollout_config.name != "trtllm":
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        elif self.worker_group and rollout_config.name == "trtllm":
            self._run_all(
                [
                    server.init_hybrid_colocated(self.worker_group, rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses, rollout_config.name)

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_router_address)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        # Fix for Issue #4147: Always call wake_up() to ensure weight sync
        # The wake_up()/sleep() methods internally check free_cache_engine
        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        # Fix for Issue #4147: Always call sleep() to ensure proper cleanup
        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        return timing

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def clear_kv_cache(self):
        """Clear all rollout kv cache, but don`t sleep."""
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])

    def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        self._run_all([replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        self._run_all([replica.stop_profile() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
