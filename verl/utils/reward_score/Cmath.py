import os
import json
import requests
import re
from retry import retry
from openai import OpenAI
from pathlib import Path
# from anthropic import AnthropicBedrock
# from zai import ZhipuAiClient
def find_files(directory='.', endwith='.json', exclude=None):
    """exclude 可以是字符串或 Path，也可以是可迭代对象"""
    if exclude is None:
        exclude = []
    if isinstance(exclude, (str, Path)):
        exclude = [Path(exclude)]

    exclude = {e.resolve() for e in exclude}          # 统一成绝对路径集合
    return [
        p for p in Path(directory).rglob(f'*{endwith}')
        if not any(ex in p.resolve().parents for ex in exclude)
    ]


s = requests.session()
def load_json(path):
    res = None
    with open(path, 'r', encoding='utf-8') as f:
        res = json.load(f)
    return res 

def load_jsonl(path):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        datas = f.readlines()
        for i, line in enumerate(datas):
            res.append(json.loads(line))
    return res
def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def write_jsonl(data, path, mode = 'a'):
    with open(path, mode, encoding='utf-8') as f:
        if type(data) == list:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        elif type(data) == dict:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

            
def get_res_from_deepseek(messages, temperature = 0.0):
    deepseek_key = 'api-key'
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages,
    stream=False,
    temperature=temperature,
    )
    return response.choices[0].message.content

def get_url_model_vllm(messages, temperature=0.6, url = 'http://192.168.242.10:50001/api', thinking = True):
    data = json.dumps({"messages": messages,
                        "max_new_tokens": 1024,
                        "thinking": thinking,
                        "temperature" : temperature
                       }).encode('utf-8')
    response = requests.put(url, data=data, headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        response_json = response.json()
        return response_json['text']
    else:
        print(f"更新失败，状态码：{response.status_code}")
        return None



def extract_boxed_content(text):
    # 查找所有\boxed{}的位置
    start_indices = [m.start() for m in re.finditer(r'\\boxed{', text)]
    if not start_indices:
        return None
    
    # 存储所有成功解析的内容
    results = []
    
    # 处理每个\boxed{}实例
    for start_idx in start_indices:
        # 找到\boxed{的结束位置（{之后）
        start_pos = start_idx + len(r'\boxed{')
        count = 1  # 已经有一个左括号
        content = []
        i = start_pos
        
        while i < len(text) and count > 0:
            char = text[i]
            
            # 处理转义字符
            if char == '\\':
                if i + 1 < len(text):
                    # 添加转义字符和下一个字符
                    content.append(char + text[i+1])
                    i += 2  # 跳过已处理的下一个字符
                else:
                    content.append(char)
                    i += 1
                continue
            
            # 处理括号嵌套
            if char == '{':
                count += 1
            elif char == '}':
                count -= 1
                # 当count为0时，不要添加这个右括号
                if count == 0:
                    i += 1
                    break
            
            # 只添加尚未匹配结束的内容
            if count > 0:
                content.append(char)
            
            i += 1
        
        # 成功匹配时保存内容
        if count == 0:
            content_str = ''.join(content)
            content_str = content_str.strip()
            # 只保存非空内容
            if content_str.strip():
                results.append(content_str)
    
    # 返回最后一个非空内容
    return results[-1] if results else None




"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union
import re
from pylatexenc import latex2text
import sympy
from sympy.parsing import sympy_parser
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from math_verify import parse, verify

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"



@dataclass
class RewardConfig:
    # Use LLM as ORM to evaluate correctness.
    use_math_orm: bool = True
    
    # General reward constants.
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    format_error_reward: float = -1.0
    unk_error_reward: float = -1.0


class RewardType(Enum):
    """
    Enum class representing the different types of rewards that can be assigned.

    Attributes:
        MATH (str): Represents a math-related problem type.
        CODE (str): Represents a coding-related problem type.
        UNK (str): Represents an unknown or unclassified problem type.
    """
    MATH = 'MATH'
    CODE = 'CODE'
    UNK = 'UNK'


@dataclass
class RewardInput:
    """Data structure for input required to calculate rewards.

    Attributes:
        problem (str): The original problem text or prompt provided to the model.
        model_response (str): The response generated by the model that needs evaluation.
        problem_type (RewardType): The category of the problem (e.g., math, code) to be evaluated.
        ground_truth (dict): Additional contextual information necessary for evaluation:
            - For math problems: This may include the ground truth answer.
            - For coding problems: This may include unit tests to validate the solution.
    """
    problem: str
    model_response: str
    problem_type: RewardType = RewardType.UNK
    ground_truth: dict = field(default_factory=dict)


@dataclass
class RewardOutput:
    """Data structure for the output of reward calculations.

    Attributes:
        reward (float): The computed reward value based on the evaluation of the model's response.
        is_correct (bool): A boolean flag indicating whether the model's response is deemed correct.
    """
    reward: float
    is_correct: bool


class RewardFn:
    """Abstract base class for defining reward calculation strategies.

    This class should be subclassed to implement specific reward calculation logic.
    The __call__ method must be overridden to provide the functionality for evaluating
    the input and returning the corresponding reward output.
    """
    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, input: RewardInput) -> RewardOutput:
        raise NotImplementedError("Subclasses must implement this method.")

# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search(r"^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer

def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string


    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string


    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string


    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(rf"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution

def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct

def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None

def grade_answer_verl(solution_str, ground_truth):
    if not ground_truth:
        return False
    if '\\boxed' in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = extract_answer(solution_str)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) \
        or grade_answer_sympy(given_answer, ground_truth)



ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        # problem = input.problem
        model_response = input.model_response
        
        # Extract solution.
        # if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
        #     model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        # else:
        #     return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        model_answer = extract_answer(model_response)
        # print(model_answer)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = True):
    solution_str = f"\\boxed{{{solution_str}}}"
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct


def math_verify_reward_function(solution_str, ground_truth):
    # solution_str = solution_str.split("</think>")[1]
    solution_str = f"\\boxed{{{solution_str}}}"
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    # print(math_verify_parsed)
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return 0.0

def ver(prediction, ground_truth):
    gt = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    deepscaler_result = deepscaler_reward_fn(prediction,gt)
    math_new_result = math_verify_reward_function(prediction,gt)
    if deepscaler_result or math_new_result>0.5:
        return 1.0
    else:
        return 0.0


from math_verify.errors import TimeoutException
def compute_score(solution_str: str, ground_truth: str, timeout_score: float = 0, data_source=None, extra_info=None,reward_router_address=None, reward_model_tokenizer=None,) -> bool:
    ret_score = 0.0
    try:
        ret_score =  ver(solution_str, ground_truth)
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score
    return ret_score


def compute_score_wo_timeout(solution_str: str, ground_truth: str, timeout_score: float = 0, data_source=None, extra_info=None,reward_router_address=None, reward_model_tokenizer=None,) -> bool:
    try:
        ret_score =  ver(solution_str, ground_truth)
    except:
        ret_score = 0.0
    return ret_score

def multi_agent_format_verify(solution_str):
    cond1 = "[REVIEW FEEDBACK]" in solution_str
    cond2 = "[/REVIEW FEEDBACK]" in solution_str

    if cond1 and cond2:
        return True
    else:
        return False


def compute_score2(solution_str: str, ground_truth: str, timeout_score: float = 0, data_source=None, extra_info=None,                reward_router_address=None, reward_model_tokenizer=None,) -> bool:

    assert extra_info is not None


    format_right = multi_agent_format_verify(solution_str)

    if not format_right:
        # print("multi_agent_format Error",solution_str)
        if extra_info['split']=='train':
            res = {"score": 0.0, "reward_info": f"multi_agent_format Error"}
        else:
            test_score = compute_score(solution_str, ground_truth, timeout_score)
            res = {"score": test_score, "reward_info": f"multi_agent_format Error"}
        return res
    agent1_out = solution_str.split("[/PROPOSED SOLUTION]")[0]
    agent2_out = solution_str.split("[/REVIEW FEEDBACK]")[0].split("[REVIEW FEEDBACK]")[1]
    agent3_out = solution_str.split("[/REVIEW FEEDBACK]")[1]


    try:
        score_agent1 = ver(agent1_out, ground_truth)
        score_agent3 = ver(agent3_out, ground_truth)
        if score_agent1 < 0.5: # agent-1 答错了
            if "<Pass>" in agent2_out:  # agent-2 判断失误
                if score_agent3 < 0.5: # agent-3 改进无效
                    ret_score = 0.0
                    reward_info="Cond1: Agent-1 <Wrong>, Agent-2 <Pass>, Agent-3 <Wrong>"
                else: # agent-3 改进有效
                    ret_score = 0.5
                    reward_info = "Cond2: Agent-1 <Wrong>, Agent-2 <Pass>, Agent-3 <Right>"
            elif "<Fail>" in agent2_out: # agent-2 判断正确
                if score_agent3 < 0.5: # agent-3 改进无效
                    ret_score = 0.2
                    reward_info="Cond3: Agent-1 <Wrong>, Agent-2 <Fail>, Agent-3 <Wrong>"
                else:
                    ret_score = 0.9 # agent-3 改进有效
                    reward_info = "Cond4: Agent-1 <Wrong>, Agent-2 <Fail>, Agent-3 <Right>"
            else:
                ret_score = 0.0
                reward_info = "Cond5: Agent-1 <Wrong>, Agent-2 <FmtErr>"
        else: # agent-1 答对了
            if "<Fail>" in agent2_out:  # agent-2 判断失误
                if score_agent3 < 0.5: # agent-3 越改越差
                    ret_score = 0.0
                    reward_info="Cond6: Agent-1 <Right>, Agent-2 <Fail>, Agent-3 <Wrong>"
                else: # agent-3 改进有效
                    ret_score = 0.6
                    reward_info="Cond7: Agent-1 <Right>, Agent-2 <Fail>, Agent-3 <Right>"
            elif "<Pass>" in agent2_out: # agent-2 判断正确
                if score_agent3 < 0.5: # agent-3 改进无效
                    ret_score = 0.4
                    reward_info="Cond8: Agent-1 <Right>, Agent-2 <Pass>, Agent-3 <Wrong>"
                else:
                    ret_score = 1.0 # agent-3 改进有效
                    reward_info="Cond9: Agent-1 <Right>, Agent-2 <Pass>, Agent-3 <Right>"
            else:
                ret_score = 0.0
                reward_info = "Cond10: Agent-1 <Right>, Agent-2 <FmtErr>"
        res = {"score": ret_score, "reward_info": reward_info}
    except TimeoutException:
        res = {"score": timeout_score, "reward_info": "Cond11: <Verify Timeout>"}
    except Exception as e:
        res = {"score": 0.0, "reward_info": f"Cond12: {e}"}

    # 测试条件下不使用多智能体打分规则，但是记录 reward_info
    if extra_info['split']=='test':
        train_score = res["score"]
        reward_info= res['reward_info']
        test_score = compute_score(solution_str, ground_truth, timeout_score)
        res =  {"score": test_score, "reward_info": f"Train Score: {train_score}, {reward_info}"}


    # print(res)
    return res


def edit_distance_reward(s1: str, s2: str) -> int:
    """
    计算两个字符串之间的编辑距离（Levenshtein 距离）

    参数:
        s1 (str): 第一个字符串
        s2 (str): 第二个字符串

    返回:
        int: 编辑距离
    """
    m, n = len(s1), len(s2)
    # 创建 (m+1) x (n+1) 的 DP 表
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i  # s1 的前 i 个字符转为空字符串，需要 i 次删除
    for j in range(n + 1):
        dp[0][j] = j  # 空字符串转为 s2 的前 j 个字符，需要 j 次插入

    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 字符相同，无需操作
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # 删除
                    dp[i][j - 1],  # 插入
                    dp[i - 1][j - 1]  # 替换
                )
    if len(s2) == 0:
        res = 0.0
    else:
        res = 1 - min(dp[m][n] / len(s2),1.0)
#     print(s1,s2,res)
    return res


def compute_score3(solution_str: str, ground_truth: str, timeout_score: float = 0, data_source=None, extra_info=None,  reward_router_address=None, reward_model_tokenizer=None,) -> bool:

    assert extra_info is not None

    format_right = multi_agent_format_verify(solution_str)

    if not format_right:
        # print("multi_agent_format Error",solution_str)
        res = {"score": 0.0, "reward_info": f"multi_agent_format Error"}
        return res

    agent1_out = solution_str.split("[/PROPOSED SOLUTION]")[0]
    agent2_out = solution_str.split("[/REVIEW FEEDBACK]")[0].split("[REVIEW FEEDBACK]")[1]
    agent3_out = solution_str.split("[/REVIEW FEEDBACK]")[1]

    try:
        score_agent1 = ver(agent1_out, ground_truth)
        score_agent3 = ver(agent3_out, ground_truth)

        # 0 in case parsing cannot be completed
        try:
            math_verify_parsed1 = parse(f"\\boxed{{{agent1_out}}}", parsing_timeout=5)[1]
        except:
            math_verify_parsed1 = ""
        try:
            math_verify_parsed3 = parse(f"\\boxed{{{agent3_out}}}", parsing_timeout=5)[1]
        except:
            math_verify_parsed3 = ""

        # print("ground_truth", ground_truth)
        # print("math_verify_parsed1", math_verify_parsed1)
        # print("math_verify_parsed3", math_verify_parsed3)
        edit_reward1 = edit_distance_reward(math_verify_parsed1, ground_truth)
        edit_reward3 = edit_distance_reward(math_verify_parsed3, ground_truth)

        if score_agent1 < 0.5:  # agent-1 答错了
            if "<Pass>" in agent2_out:  # agent-2 判断失误
                if score_agent3 < 0.5:  # agent-3 改进无效
                    ret_score = 0.0
                    reward_info = "Cond1: Agent-1 <Wrong>, Agent-2 <Pass>, Agent-3 <Wrong>"
                else:  # agent-3 改进有效
                    ret_score = 0.5
                    reward_info = "Cond2: Agent-1 <Wrong>, Agent-2 <Pass>, Agent-3 <Right>"
            elif "<Fail>" in agent2_out:  # agent-2 判断正确
                if score_agent3 < 0.5:  # agent-3 改进无效
                    ret_score = 0.2
                    reward_info = "Cond3: Agent-1 <Wrong>, Agent-2 <Fail>, Agent-3 <Wrong>"
                else:
                    ret_score = 0.9  # agent-3 改进有效
                    reward_info = "Cond4: Agent-1 <Wrong>, Agent-2 <Fail>, Agent-3 <Right>"
            else:
                ret_score = 0.0
                reward_info = "Cond5: Agent-1 <Wrong>, Agent-2 <FmtErr>"
        else:  # agent-1 答对了
            if "<Fail>" in agent2_out:  # agent-2 判断失误
                if score_agent3 < 0.5:  # agent-3 越改越差
                    ret_score = 0.0
                    reward_info = "Cond6: Agent-1 <Right>, Agent-2 <Fail>, Agent-3 <Wrong>"
                else:  # agent-3 改进有效
                    ret_score = 0.6
                    reward_info = "Cond7: Agent-1 <Right>, Agent-2 <Fail>, Agent-3 <Right>"
            elif "<Pass>" in agent2_out:  # agent-2 判断正确
                if score_agent3 < 0.5:  # agent-3 改进无效
                    ret_score = 0.4
                    reward_info = "Cond8: Agent-1 <Right>, Agent-2 <Pass>, Agent-3 <Wrong>"
                else:
                    ret_score = 1.0  # agent-3 改进有效
                    reward_info = "Cond9: Agent-1 <Right>, Agent-2 <Pass>, Agent-3 <Right>"
            else:
                ret_score = 0.0
                reward_info = "Cond10: Agent-1 <Right>, Agent-2 <FmtErr>"
        ret_score = ret_score + 0.1 * edit_reward1 + 0.15 * edit_reward3
        reward_info = reward_info + f" edit_reward1:{edit_reward1:.3} edit_reward3:{edit_reward3:.3}"
        res = {"score": ret_score, "reward_info": reward_info}
    except TimeoutException:
        res = {"score": timeout_score, "reward_info": "Cond11: <Verify Timeout>"}
    except Exception as e:
        res = {"score": 0.0, "reward_info": f"Cond12: {e}"}

    # 测试条件下不使用多智能体打分规则，但是记录 reward_info
    if extra_info['split']=='test':
        train_score = res["score"]
        reward_info= res['reward_info']
        test_score = compute_score(solution_str, ground_truth, timeout_score)
        res =  {"score": test_score, "reward_info": f"Train Score: {train_score}, {reward_info}"}
    # print(res)
    return res