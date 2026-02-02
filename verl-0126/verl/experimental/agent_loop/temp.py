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
    max_seq_length = self.config.actor_rollout_ref.rollout.response_length + self.config.actor_rollout_ref.rollout.prompt_length
    ###############################################################################################

    # 提取所有问题
    prompt_token_ids_list = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]
    problems = [self.tokenizer.decode(prompt_token_ids) for prompt_token_ids in prompt_token_ids_list]

    ####################################### <Agent-1: Generate> #######################################
    gen_prompts = [self._build_CMATHrole_zh_prompt_selfdefine("generate", problem) for problem in problems]
    gen_input = self._encode_prompts_to_dataproto_with_maxLen(gen_prompts,
                                                              prompts.meta_info.copy(), self.config.actor_rollout_ref.rollout.prompt1_length)
    gen_output = self._generate_sequences(gen_input)
    agent1_outputs = self._decode_responses_wt_MaxLen(gen_output, self.config.actor_rollout_ref.rollout.response1_length)  # str
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
        active_review_outputs = self._decode_responses_wt_MaxLen(rev_output, self.config.actor_rollout_ref.rollout.response2_length)
        active_review_outputs = [f"(Review Times: <{iter_num + 1}>)\n" + x for x in active_review_outputs]

        print(f"vllm_rollout_spmd.py line-629 (Review Times: <{iter_num + 1}>): {len(active_indices)} active samples")

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
            improved_answers = self._decode_responses_wt_MaxLen(imp_output, self.config.actor_rollout_ref.rollout.response3_length)
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
        final_pass_answers = self._decode_responses_wt_MaxLen(final_imp_output, self.config.actor_rollout_ref.rollout.response3_length)
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
        F.pad(x, (0, max_res_len - x.size(-1)), value=self.pad_token_id)[:self.config.actor_rollout_ref.rollout.response_length] for x in
        batch_global_response_ids]
    batch_global_response_mask = [
        F.pad(x, (0, max_res_len - x.size(-1)), value=0)[:self.config.actor_rollout_ref.rollout.response_length] for x in
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
    baTch_seq_mask = torch.cat([batch_global_prompt_mask, batch_global_response_mask.to(device)], dim=1).to(torch.long)

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

    key1, key2, key3 = get_key_prompts(self.config.actor_rollout_ref.rollout.extra["roles"], self.config.actor_rollout_ref.rollout.extra['role_config_path'])

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
