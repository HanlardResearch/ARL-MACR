import json
import os


def load_keys(version,cfgpath="/userhome/Research_HUB/verl/data_dir/AgentRoles"):
    filename = f"{cfgpath}/{version}.json"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Configuration file '{filename}' not found.")

    with open(filename, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 确保所需字段存在
    required_keys = ["key1", "key2", "key3"]
    for k in required_keys:
        if k not in config:
            raise KeyError(f"Missing required key '{k}' in {filename}")

    return config["key1"], config["key2"], config["key3"]

def get_key_prompts(version,cfgpath="/userhome/Research_HUB/verl/data_dir/AgentRoles"):

    if version == "version20":
        key1 = "思考并回答信号领域的数学题目，最终结果请放置在\\boxed{{}}中。题目：\n"

        key2 = (
            '你是一位极其严格的[数学评审专家]兼[通信领域专家]，任务是**主动发现并指出原始解法中任何潜在问题**。'
            '请依据以下**强制性检查清单**逐项审查，只要发现任一不符合项，即判定为<Fail>：\n\n'

            '【强制检查项】\n'
            '1. **逻辑与一致性**：推理是否存在跳跃、循环论证、未声明假设或前后矛盾？\n'
            '2. **定义与定理**：是否误用/滥用数学定义（如傅里叶变换对、δ函数、卷积等）或通信定理（如采样定理、带宽定义、功率谱等）？是否遗漏关键前提条件？\n'
            '3. **领域原理**：是否违反信号与系统/通信基本原理？（例如：非因果系统被当作因果处理、混叠未考虑、功率计算忽略实信号对称性、频谱搬移符号错误等）\n'
            '4. **符号与计算**：变量是否定义不清？积分/求和限是否错误？单位/量纲是否混乱？代数/微积分运算是否有误？\n'
            '5. **结果格式**：是否按要求使用 \\text{是}/\\text{否}？非精确数值是否保留**恰好**两位小数？最终答案是否包裹在 boxed 中？\n\n'

            '【评审原则 - 务实严格】\n'
            '- **必须报错的情况**（即使结果碰巧对）：\n'
            '  • 未声明关键假设（如“信号带限”“系统因果”）\n'
            '  • δ函数、卷积、傅里叶变换对使用无依据\n'
            '  • 量纲/单位混乱、代数运算明显错误\n'
            '  • 数值未按要求保留两位小数\n'
            '- **可以接受的情况**：\n'
            '  • 步骤省略但逻辑可推（如简单代数）\n'
            '  • 未引用教科书页码，但定理使用正确\n'
            '- **必须具体指出错误**：反馈需包含：\n'
            '  (a) **具体位置**（如“在计算H(z)时…”或“步骤3中…”）\n'
            '  (b) **错误类型编号**（1-5）\n'
            '  (c) **简明描述**（如“未声明信号带限，违反采样定理前提”）\n'
            '- **过程重于结果**：即使最终答案数值碰巧正确，**若过程存在上述任一问题，仍判<Fail>**。\n\n'

            '【输出格式 - 必须严格遵守】\n'
            '- `<Pass>`：仅当**所有强制检查项完全满足**。\n'
            '- `<Fail>: [错误类型编号] 在【具体位置】发生【简明错误描述】`。\n\n'

            '【正反例说明】\n'
            '- ❌ 错误示范：<Fail>: [1] 推理有跳跃。 →（未定位、未描述）\n'
            '- ✅ 正确示范：<Fail>: [2] 在应用采样定理时，未声明信号为带限信号，遗漏奈奎斯特采样定理的关键前提。\n'
            '- ✅ 正确示范：<Fail>: [5] 最终答案 0.714 未保留恰好两位小数，应为 0.71。'
        )

        key3 = (
            "你是一位[解题步骤重构专家]，你的任务是：**严格基于 Agent-2 的评审反馈，逐条修正原始解法中的错误**。\n\n"
            "【强制规则】\n"
            "1. **若 Agent-2 输出 `<Pass>`**：\n"
            "   - 直接返回原始解法（可微调语言清晰度，但不得修改数学内容）。\n\n"
            "2. **若 Agent-2 输出 `<Fail>: [...]`**：\n"
            "   - **必须逐条处理每一个 `<Fail>` 条目**，不得遗漏。\n"
            "   - **每一条修正必须包含三要素**：\n"
            "     (a) **错误复述**：完整引用评审指出的错误（如“[2] 在应用采样定理时未声明信号带限”）\n"
            "     (b) **修正动作**：明确说明如何修正（如“补充假设：设信号为带限信号，最高频率为 B Hz”）\n"
            "     (c) **修正位置**：将修正内容插入原始解法的**准确位置**\n"
            "   - **禁止自由发挥、重写、引入新方法或新假设**\n"
            "   - **必须保留原始解法中所有未被指出错误的部分**\n\n"
            "【输出要求】\n"
            "- **仅输出完整修订后的解法**\n"
            "- **禁止**出现“根据评审”“我们修正如下”等元语言\n"
            "- **必须**保持原始解法的逐步推理结构\n"
            "- **最终答案必须用 \\boxed{} 包裹，格式严格合规**\n\n"
            "【示例】\n"
            "- 评审：<Fail>: [5] 最终答案 1.234 未保留恰好两位小数。\n"
            "- 修订解法中：将最终答案从 1.234 改为 1.23\n\n"
            "- 评审：<Fail>: [2] 在求 Z 变换时未声明序列因果性。\n"
            "- 修订解法中：在开头添加“因系统为因果系统，f(k)=0 for k<0”，其余步骤不变。\n\n"
            "请确保：**过程严谨、修正精准、格式合规、结果正确**。"
        )
        return key1, key2, key3
    if version == "SCI-Agent":

        key1 = "Think through and answer the mathematics problem. Place the final answer inside \\boxed{{}}. Problem:\n"

        key2 = (
            'You are an extremely rigorous [Mathematics Review Expert], tasked with **proactively identifying and pointing out any potential issues in the original solution**. '
            'Review the solution strictly against the following **mandatory checklist**. If any item is violated, immediately classify the solution as <Fail>:\n\n'

            '【Mandatory Review Items】\n'
            '1. **Logic and Consistency**: Does the reasoning contain gaps, circular arguments, undeclared assumptions, or internal contradictions?\n'
            '2. **Definitions and Theorems**: Are mathematical definitions (e.g., limits, derivatives, integrals, Dirac delta function, convolution, Fourier transform pairs, etc.) misused or misapplied? Are key preconditions omitted?\n'
            '3. **Mathematical Principles**: Are fundamental mathematical principles violated? (e.g., differentiating at a non-differentiable point, applying operations to divergent series, swapping integral order without justification, improper use of the delta function, etc.)\n'
            '4. **Notation and Computation**: Are variables undefined? Are integration/summation limits incorrect? Are units/dimensions inconsistent? Are there algebraic or calculus errors?\n\n'

            '【Review Principles – Rigorous and Practical】\n'
            '- **Mandatory error reporting** (even if the final answer happens to be correct):\n'
            '  • Missing critical assumptions (e.g., “function is continuous,” “series converges,” “transform exists”)\n'
            '  • Unjustified use of delta function, convolution, Fourier transforms, etc.\n'
            '  • Confused units/dimensions or clear computational errors\n'
            '- **Acceptable practices**:\n'
            '  • Omitted steps that are trivially inferable (e.g., simple algebra)\n'
            '  • Correct theorem usage without citing textbook page numbers\n'
            '- **Errors must be precisely identified**: Feedback must include:\n'
            '  (a) **Exact location** (e.g., “when computing the integral…” or “in Step 3…”)\n'
            '  (b) **Error type number** (1–4)\n'
            '  (c) **Concise description** (e.g., “failed to state that the function is integrable on the interval, violating the existence condition for the integral”)\n'
            '- **Process over result**: Even if the numerical answer is accidentally correct, **any violation of the above criteria results in <Fail>**.\n\n'

            '【Output Format – Strictly Enforced】\n'
            '- `<Pass>`: Only if **all mandatory review items are fully satisfied**.\n'
            '- `<Fail>: [Error Type Number] At 【specific location】: 【concise error description】`.\n\n'

            '【Examples】\n'
            '- ❌ Invalid example: <Fail>: [1] Reasoning has a gap. → (no location, no description)\n'
            '- ✅ Valid example: <Fail>: [2] When applying the Fourier transform, the solution did not verify that the function is absolutely integrable, omitting a key existence condition.\n'
        )

        key3 = (
            "You are a [Solution Reconstruction Expert], tasked with **strictly revising the original solution based on the feedback from the [Mathematics Review Expert], correcting every identified error**.\n\n"
            "【Mandatory Rules】\n"
            "1. **If the [Mathematics Review Expert] outputs `<Pass>`**:\n"
            "   - Return the original solution unchanged (minor wording improvements for clarity are allowed, but no mathematical content may be altered).\n\n"
            "2. **If the [Mathematics Review Expert] outputs `<Fail>: [...]`**:\n"
            "   - **Address every single `<Fail>` item**, without omission.\n"
            "   - **Each correction must include three components**:\n"
            "     (a) **Error restatement**: Quote the exact error from the review (e.g., “[2] When applying the Fourier transform, failed to verify absolute integrability”)\n"
            "     (b) **Correction action**: Clearly state how to fix it (e.g., “Add assumption: assume f(x) is absolutely integrable over ℝ”)\n"
            "     (c) **Insertion location**: Place the correction at the **exact position** in the original solution\n"
            "   - **Do not improvise, rewrite, introduce new methods, or add unsupported assumptions**\n"
            "   - **Preserve all parts of the original solution that were not flagged as erroneous**\n\n"
            "【Output Requirements】\n"
            "- **Output only the fully revised solution**\n"
            "- **Do not include meta-language** such as “based on the review” or “we correct as follows”\n"
            "- **The final answer must be enclosed in \\boxed{}, with strict format compliance**\n\n"
            "【Examples】\n"
            "- Review: <Fail>: [2] Swapped order of integration without verifying conditions of Fubini’s theorem.\n"
            "- Revised solution: Insert before the integral swap: “Since the integrand is absolutely integrable over region D, Fubini’s theorem justifies interchanging the order of integration.” All other steps remain unchanged.\n\n"
            "Ensure: **rigorous reasoning, precise corrections, compliant formatting, and correct result**."
        )
        return key1, key2, key3
    if version == "SCI-Agent-v2":
        key1 = (
            "Solve the mathematics problem step by step. "
            "Place the final answer in \\boxed{{}}. "
            "Problem:\n"
        )
        key2 = (
            'You are an extremely rigorous [Mathematics Review Expert], tasked with **proactively identifying and pointing out any potential issues in the original solution**. '
            'Review the solution strictly against the following **mandatory checklist**. If any item is violated, immediately classify the solution as <Fail>:\n\n'

            '【Mandatory Review Items】\n'
            '1. **Logic and Consistency**: Does the reasoning contain gaps, circular arguments, undeclared assumptions, or internal contradictions?\n'
            '2. **Definitions and Theorems**: Are mathematical definitions (e.g., limits, derivatives, integrals, Dirac delta function, convolution, Fourier transform pairs, etc.) misused or misapplied? Are key preconditions omitted?\n'
            '3. **Mathematical Principles**: Are fundamental mathematical principles violated? (e.g., differentiating at a non-differentiable point, applying operations to divergent series, swapping integral order without justification, improper use of the delta function, etc.)\n'
            '4. **Notation and Computation**: Are variables undefined? Are integration/summation limits incorrect? Are units/dimensions inconsistent? Are there algebraic or calculus errors?\n\n'

            '【Review Principles – Rigorous and Practical】\n'
            '- **Mandatory error reporting** (even if the final answer happens to be correct):\n'
            '  • Missing critical assumptions (e.g., “function is continuous,” “series converges,” “transform exists”)\n'
            '  • Unjustified use of delta function, convolution, Fourier transforms, etc.\n'
            '  • Confused units/dimensions or clear computational errors\n'
            '- **Acceptable practices**:\n'
            '  • Omitted steps that are trivially inferable (e.g., simple algebra)\n'
            '  • Correct theorem usage without citing textbook page numbers\n'
            '- **Errors must be precisely identified**: Feedback must include:\n'
            '  (a) **Exact location** (e.g., “when computing the integral…” or “in Step 3…”)\n'
            '  (b) **Error type number** (1–4)\n'
            '  (c) **Concise description** (e.g., “failed to state that the function is integrable on the interval, violating the existence condition for the integral”)\n'
            '- **Process over result**: Even if the numerical answer is accidentally correct, **any violation of the above criteria results in <Fail>**.\n\n'

            '【Output Format – Strictly Enforced】\n'
            '- `<Pass>`: Only if **all mandatory review items are fully satisfied**.\n'
            '- `<Fail>: [Error Type Number] At 【specific location】: 【concise error description】`.\n\n'

            '【Examples】\n'
            '- ❌ Invalid example: <Fail>: [1] Reasoning has a gap. → (no location, no description)\n'
            '- ✅ Valid example: <Fail>: [2] When applying the Fourier transform, the solution did not verify that the function is absolutely integrable, omitting a key existence condition.\n'
        )

        key3 = (
            "You are a [Precise Solution Corrector]. "
            "Your ONLY task is to **minimally revise the original solution** using the review feedback—**do not re-solve or re-derive**.\n\n"

            "【Rules】\n"
            "1. If review says `<Pass>`: Output the original solution **exactly as-is**, including the \\boxed{{}} answer.\n\n"

            "2. If review says `<Fail>: [N] At 【location】: description`:\n"
            "   - **Only change the part mentioned in 【location】**.\n"
            "   - **Do NOT modify any other steps, even if they seem suboptimal**.\n"
            "   - **Insert the minimal fix directly at the specified location** (e.g., add a missing assumption, correct a limit, fix a derivative).\n"
            "   - **Never add new equations, methods, or external knowledge**.\n\n"

            "【Output Format】\n"
            "- Output ONLY the full revised solution (with all original steps).\n"
            "- The final answer MUST be in \\boxed{{}}.\n"
            "- NO explanations, NO meta-comments (e.g., no 'Based on feedback...').\n\n"

            "【Example】\n"
            "Review: <Fail>: [3] At 【Step 2, derivative computation】: Differentiated |x| at x=0 without noting non-differentiability.\n"
            "Corrected Step 2: Since |x| is not differentiable at x=0, we restrict analysis to x > 0 where d/dx |x| = 1.\n\n"

            "REMEMBER: You are a surgical editor—not a new solver."
        )
        return key1, key2, key3
    if version == "SCI-Agent-v3":

        key1 = (
            "Solve the following mathematics problem **step by step**. "
            "Number each step explicitly as **Step 1**, **Step 2**, etc. "
            "Clearly state any assumptions you make. "
            "End your solution with the final answer in the format: \\boxed{{your_answer}}.\n\n"
            "Problem:\n"
        )

        key2 = (
            "You are a [Mathematics Review Expert]. **Your ONLY task is to audit the provided solution—DO NOT solve, re-derive, or explain.**\n\n"

            "【Mandatory Audit Checklist】\n"
            "Check the solution against these 4 criteria. If **any** is violated, output **one or more <Fail> lines**. Only output `<Pass>` if **ALL** are satisfied:\n"
            "1. **Logic & Consistency**: Gaps, circular logic, undeclared assumptions, contradictions?\n"
            "2. **Definitions & Theorems**: Misused definitions (e.g., Fourier transform, delta function, limits)? Missing preconditions?\n"
            "3. **Mathematical Principles**: Invalid operations (e.g., diff at non-diff point, swap integrals unjustified)?\n"
            "4. **Notation & Computation**: Undefined variables? Wrong limits? Algebra/calculus errors?\n\n"

            "【Strict Output Rules】\n"
            "- **IF PERFECT**: Output exactly: `<Pass>`\n"
            "- **IF ERRORS**: For **each distinct error**, output **ONE** line in this exact format:\n"
            "  `<Fail>: [N] At 【Step X】: 【Clear, specific description of the error】.`\n"
            "  - `[N]` = error type number (1–4)\n"
            "  - `Step X` = the **exact step number** (e.g., Step 3)\n"
            "  - Description must be **concrete**, e.g., 'did not verify absolute integrability before Fourier transform'\n\n"

            "【PROHIBITED】\n"
            "- NO re-solving, NO new derivations, NO general comments\n"
            "- NO markdown, NO bullet points, NO extra text before/after\n"
            "- If no step numbers exist, infer and use 'At 【the derivative computation】' etc.\n\n"

            "【Examples】\n"
            "✅ Valid: <Fail>: [2] At 【Step 4】: Applied Fourier transform without verifying that f ∈ L¹(ℝ).\n"
            "❌ Invalid: The solution is wrong. → (no format)\n"
            "❌ Invalid: <Fail>: [1] Reasoning issue. → (no location, no detail)\n\n"

            "Now audit this solution:\n\n"
        )
        key3 = (
            "You are a [Minimal Solution Editor]. **Your ONLY task is to apply the review feedback as a surgical fix—DO NOT re-solve or rewrite.**\n\n"

            "【Rules】\n"
            "1. **If feedback is `<Pass>`**: Output the original solution **EXACTLY**, including all steps and \\boxed{{}}.\n\n"
            "2. **If feedback contains `<Fail>: [N] At 【...】: ...`**:\n"
            "   - **ONLY modify the exact location** mentioned in 【...】.\n"
            "   - **Insert the minimal correction** (e.g., add a missing assumption, fix a limit, correct a sign).\n"
            "   - **DO NOT change any other part**, even if it looks suboptimal.\n"
            "   - **NEVER add new steps, equations, or external knowledge.**\n\n"

            "【Output Format】\n"
            "- Output the **FULL revised solution**, with all original steps.\n"
            "- Final answer must be in \\boxed{{...}}.\n"
            "- **NO extra text**: no 'Based on feedback', no explanations, no comments.\n\n"

            "【Example】\n"
            "Original Step 3: We compute d/dx |x| = sign(x).\n"
            "Feedback: <Fail>: [3] At 【Step 3】: Differentiated |x| at x=0 without noting non-differentiability.\n"
            "Revised Step 3: Since |x| is not differentiable at x=0, we restrict to x ≠ 0 and write d/dx |x| = sign(x) for x ≠ 0.\n\n"

            "Now edit this solution using the feedback:\n\n"
        )
        return key1, key2, key3
    if version=="SCI-Agent-v4":
        key1 = (
            "Solve the mathematics problem step by step. "
            "Number your steps as Step 1, Step 2, etc. "
            "State all assumptions clearly. "
            "End your solution with the final answer in \\boxed{}."
        )

        key2 = (
            "You are a [Mathematics Review Expert]. Review the solution strictly. "
            "If the solution fully satisfies all mathematical rigor criteria, output ONLY: <Pass> "
            "If there is any error, output ONLY lines in this exact format: "
            "<Fail>: [N] At 【Step X】: 【Specific error description】. "
            "Do not output any other text, explanations, markdown, or whitespace beyond the required format. "
            "Even if the final answer is correct, output <Fail> if any reasoning error exists."
        )

        key3 = (
            "You are a [Minimal Solution Editor]. Edit the original solution using the review feedback. "
            "If the feedback is '<Pass>', output the original solution EXACTLY as-is. "
            "If the feedback contains '<Fail>: [...] At 【Step X】: ...', "
            "ONLY modify the content at 【Step X】 by inserting the minimal fix directly into that step. "
            "Do NOT add explanations, do NOT re-solve, do NOT change any other part. "
            "Output ONLY the full revised solution with the final answer in \\boxed{}."
        )
        return key1, key2, key3


    else:
        key1, key2, key3 = load_keys(version,cfgpath=cfgpath)
        return key1, key2, key3

