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
Generate responses given a dataset of prompts
"""

import os

import aiohttp
import hydra
import numpy as np
import ray
import itertools
from typing import Any, Dict, List

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import asyncio
from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletion

from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.replica import get_rollout_replica_class
from verl.MultiAgent.multi_agent_role_prompt import get_key_prompts
from tqdm import tqdm

async def start_server(config):
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size
    rollout_config = config.actor_rollout_ref.rollout
    # rollout_config.max_model_len = 32768
    model_config = config.actor_rollout_ref.model
    # create standalone rollout server
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    return server_handles, server_addresses


async def submit_request(server_address, **chat_complete_request):
    try:
        extra_headers = chat_complete_request.pop("extra_headers", {})
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(
            url=f"http://{server_address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},
            json=chat_complete_request,
        ) as resp:
            data = await resp.json()
            return ChatCompletion(**data)
    finally:
        await session.close()


async def generate_per_replica(server_address, model_path: str, n_samples: int, sampling_params: dict, chat_lst: list):
    # here we should sample n_samples for each chat_lst.
    # we use aiohttp to avoid hang in AsyncOpenAI when the number of requests is large.

    # client = AsyncOpenAI(
    #     api_key="123-abc",
    #     base_url=f"http://{server_address}/v1",
    # )

    chat_complete_request = [
        {
            "model": model_path,
            "messages": messages,
            **sampling_params,
        }
        for messages in chat_lst
        for _ in range(n_samples)
    ]

    tasks = [submit_request(server_address, **req) for req in chat_complete_request]
    results = await asyncio.gather(*tasks)
    return results


async def generate(
    server_addresses: list, model_path: str, n_samples: int, sampling_params: dict, chat_numpy: np.ndarray
):
    num_replicas = len(server_addresses)
    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]
    assert len(server_addresses) == len(chat_sub_array)
    results = await asyncio.gather(
        *[
            generate_per_replica(server_addresses[i], model_path, n_samples, sampling_params, chat_sub_array[i])
            for i in range(num_replicas)
        ]
    )
    return results

async def _generate_and_process(
    server_addresses,
    model_path: str,
    n_samples: int,
    sampling_params: Dict[str, Any],
    chat_numpy: np.ndarray
):
    gen_results = await generate(
        server_addresses, model_path, n_samples, sampling_params, chat_numpy
    )
    
    # Flatten the list of lists
    flattened_results = list(itertools.chain.from_iterable(gen_results))
    
    # Extract content
    contents = np.array([result.choices[0].message.content for result in flattened_results])
    
    # Reshape to (-1, n_samples)
    reshaped = np.reshape(contents, (-1, n_samples))
    
    return reshaped.tolist()

def run_generation(
    server_addresses,
    model_path: str,
    n_samples: int,
    sampling_params: Dict[str, Any],
    chat_numpy: np.ndarray
) -> List[List[str]]:
    return asyncio.run(_generate_and_process(
        server_addresses, model_path, n_samples, sampling_params, chat_numpy
    ))

import re
from typing import List, Dict, Union, Tuple

def _build_CMATHrole_zh_prompt_selfdefine(
    config, 
    role: str, 
    problem: str = "", 
    content: str = "", 
    feedback: str = ""
) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]]:
    """
    构建 SciAgent 风格的角色化 messages
    """
    key1, key2, key3 = get_key_prompts(config.actor_rollout_ref.rollout.extra.roles)

    def cleantext(text: str) -> str:
        if not text:
            return text
        return (
            text.replace("<|im_start|>user", "")
                .replace("<|im_start|>assistant", "")
                .replace("<|im_end|>", "")
                .strip()
        )

    def remove_think_tags(text: str) -> str:
        if not text:
            return text
        pattern = r'<think>.*?</think>'
        if re.search(pattern, text, flags=re.DOTALL):
            return re.sub(pattern, '', text, flags=re.DOTALL).strip()
        else:
            words = text.split()
            return ' '.join(words[-500:])

    clean_problem = cleantext(problem)
    clean_content = remove_think_tags(cleantext(content))
    clean_feedback = remove_think_tags(cleantext(feedback))

    # Stage 1: Generate
    pmt1_msgs = [
        {"role": "user", "content": key1 + clean_problem}
    ]

    # Stage 2: Review
    pmt2_msgs = [
        {"role": "user", "content": key1 + clean_problem},
        {"role": "assistant", "content": f"[PROPOSED SOLUTION]\n{clean_content}\n[/PROPOSED SOLUTION]"},
        {"role": "user", "content": key2}
    ]

    # Stage 3: Improve
    pmt3_msgs = [
        {"role": "user", "content": key1 + clean_problem},
        {"role": "assistant", "content": f"[PROPOSED SOLUTION]\n{clean_content}\n[/PROPOSED SOLUTION]"},
        {"role": "user", "content": key2},
        {"role": "assistant", "content": f"[REVIEW FEEDBACK]\n{clean_feedback}\n[/REVIEW FEEDBACK]"},
        {"role": "user", "content": key3}
    ]

    if role == "all":
        return pmt1_msgs, pmt2_msgs, pmt3_msgs

    if role == "generate":
        return pmt1_msgs
    elif role == "review":
        return pmt2_msgs
    elif role == "improve":
        return pmt3_msgs
    else:
        raise ValueError(f"Unsupported role: {role}")

def remove_think_tags(text: str) -> str:
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
        
def _build_CMATHrole_zh_prompt_selfdefine_text(config, role: str, problem: str = "", content: str = "", feedback: str = "") -> str:
    """构建 SciAgent 风格的角色化 prompt"""

    key1,key2,key3 = get_key_prompts(config.actor_rollout_ref.rollout.extra.roles)

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
    clean_content = remove_think_tags(cleantext(content))
    clean_feedback = remove_think_tags(cleantext(feedback))

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

def extract_fail_feedback(review_text: str) -> str:
    """
    从 Review 输出中提取 <Fail> 部分的反馈内容。
    假设格式如：<Fail>: 错误描述\n 或包含关键错误信息。
    若无明确 <Fail>，但整体是失败反馈，也可返回全文。
    """
    review_text = remove_think_tags(review_text)
    review_texts = review_text.split("\n")
    res = []
    for text in review_texts:
        if "<Fail>: [" in text and len(text) < 500:
            res.append(text)
    if res:
        return "\n".join(res)
    else:
        return ""
            
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}})
    
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    # start native server
    server_handles, server_addresses = asyncio.run(start_server(config))

    n_samples = config.actor_rollout_ref.rollout.n

    if config.actor_rollout_ref.rollout.temperature == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."
    assert n_samples >= 1, "n_samples should always >= 1"

    base_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        # "max_model_len": config.actor_rollout_ref.rollout.max_model_len,
        # "top_k": config.actor_rollout_ref.rollout.top_k,
    }

    prompt_lengths = [
        config.actor_rollout_ref.rollout.prompt1_length,
        config.actor_rollout_ref.rollout.prompt2_length,
        config.actor_rollout_ref.rollout.prompt3_length,
    ]
    
    response_lengths = [
        config.actor_rollout_ref.rollout.response1_length,
        config.actor_rollout_ref.rollout.response2_length,
        config.actor_rollout_ref.rollout.response3_length,
    ]

    agent_sampling_params_list = [
        {**base_params, "max_completion_tokens": res_length, "truncate_prompt_tokens": pmt_length}
        for pmt_length, res_length in zip(prompt_lengths, response_lengths)
    ]

    agent_1_sampling_params, agent_2_sampling_params, agent_3_sampling_params = agent_sampling_params_list
    batch_size = n_samples
    from omegaconf import ListConfig

    train_files = config.data.train_files
    if not isinstance(train_files, list | ListConfig):
        train_files = [train_files]

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    datasets = []
    for train_file in train_files:
        dataset = pd.read_parquet(train_file)
        datasets.append(dataset)
    pmt1_transition_txt,pmt2_transition_txt,pmt3_transition_txt = _build_CMATHrole_zh_prompt_selfdefine_text(config, role="all")
    # concat dataset
    dataset = pd.concat(datasets, axis=0, ignore_index=True)
    dataset['responses'] = None
    chat_lst = dataset[config.data.prompt_key].tolist()
    for idx, chat in enumerate(tqdm(chat_lst)):
        chat_numpy = np.array(chat.tolist())
        problems = [chat_numpy.copy()] * batch_size
        last_review_outputs = [""] * batch_size  # 保存最后一次review输出 (用于拼接)
        final_answers = [""] * batch_size  # 最终输出 (Agent-3)
        historical_fails = [[] for _ in range(batch_size)]  # List[List[str]]
        passed = [False] * batch_size  # 标记是否已通过review
        gen_prompts = [_build_CMATHrole_zh_prompt_selfdefine(config, "generate", problem[0]['content']) for problem in problems]
        
        # run generate
        # print("run generate for agent 1")
        agent1_outputs = run_generation(
            server_addresses, config.actor_rollout_ref.model.path, 1, agent_1_sampling_params, gen_prompts
        )
        current_answers = agent1_outputs.copy()  # 当前答案 (动态更新)
        # ====== Review-Improve 循环 ======
        for iter_num in range(int(config.actor_rollout_ref.rollout.extra.max_iter)):
            # 1. 收集需要Review的样本索引 (未通过的样本)
            active_indices = [i for i in range(batch_size) if not passed[i]]
            if not active_indices:  # 所有样本已通过
                break

            # 2. 批量执行 Review (Agent-2)
            active_problems = [problems[i] for i in active_indices]
            active_answers = [current_answers[i] for i in active_indices]
            rev_prompts = [_build_CMATHrole_zh_prompt_selfdefine(config, "review", p[0]['content'], a[0]) for p, a in
                            zip(active_problems, active_answers)]
            # rev_input = self._encode_prompts_to_dataproto_with_maxLen(rev_prompts, prompts.meta_info.copy(),
            #                                                             self.config.prompt2_length)
            # rev_output = self._generate_sequences(rev_input)
            # active_review_outputs = self._decode_responses_wt_MaxLen(rev_output, self.config.response2_length)
            active_review_outputs = run_generation(
                server_addresses, config.actor_rollout_ref.model.path, 1, agent_2_sampling_params, rev_prompts
            )
            active_review_outputs = [f"(Review Times: <{iter_num+1}>)\n"+x[0] for x in active_review_outputs]

            # print(f"vllm_rollout_spmd.py line-629 (Review Times: <{iter_num+1}>): {len(active_indices)} active samples")

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
                    fail_feedback = extract_fail_feedback(review_output)
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
                    _build_CMATHrole_zh_prompt_selfdefine(config, "improve", p[0]['content'], a[0], f)
                    for p, a, f in zip(imp_problems, imp_answers, imp_historical_fails)
                ]
                # imp_input = self._encode_prompts_to_dataproto_with_maxLen(imp_prompts,
                #                                                             prompts.meta_info.copy(),
                #                                                             self.config.prompt3_length)
                # imp_output = self._generate_sequences(imp_input)
                # improved_answers = self._decode_responses_wt_MaxLen(imp_output, self.config.response3_length)
                improved_answers = run_generation(
                    server_addresses, config.actor_rollout_ref.model.path, 1, agent_3_sampling_params, imp_prompts
                )
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
            #     _build_CMATHrole_zh_prompt_selfdefine(config, "improve", problems[i], current_answers[i], last_review_outputs[i])
            #     for i in pass_indices  #
            # ]
            final_imp_prompts = [
                _build_CMATHrole_zh_prompt_selfdefine(config, "improve", problems[i][0]['content'], current_answers[i][0], f)
                for i, f in zip(pass_indices, final_imp_feedbacks)
            ]

            # final_imp_input = self._encode_prompts_to_dataproto_with_maxLen(final_imp_prompts,
            #                                                                 prompts.meta_info.copy(),
            #                                                                 self.config.prompt3_length)
            # final_imp_output = self._generate_sequences(final_imp_input)
            # final_pass_answers = self._decode_responses_wt_MaxLen(final_imp_output,self.config.response3_length)
            final_pass_answers = run_generation(
                server_addresses, config.actor_rollout_ref.model.path, 1, agent_3_sampling_params, final_imp_prompts
            )
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

        global_responses = [
            pmt1_transition_txt
            +res_1[0]
            +pmt2_transition_txt
            +res_2
            +pmt3_transition_txt
            +res_3[0]
            for res_1, res_2, res_3 in zip(agent1_outputs, agent2_outputs, agent3_outputs)
        ]

        # add to the data frame
        dataset.at[idx, "responses"] = global_responses

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {config.data.output_path}")
    dataset.to_parquet(config.data.output_path)

if __name__ == "__main__":
    main()
