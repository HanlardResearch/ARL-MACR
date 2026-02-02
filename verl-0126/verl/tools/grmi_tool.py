import logging
import os
from typing import Any, Optional
from uuid import uuid4


from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GRMIRoutingTool(BaseTool):
    """极简路由工具：根据 next_role 参数返回 reviewer/improver 提示词之一"""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        # 从 config 直接加载两个 prompt
        self.prompts = {
            "reviewer": config.get("reviewer_prompt", "You are a [Mathematical Logic Auditor].\n\n<instructions>\n- First, independently verify the solution by reasoning through its logic and calculations.\n- Compress your full audit reasoning (verification, error check, classification) into a single paragraph inside <summary>...</summary>.\n- Then, output ONLY the audit result in the required format (<Pass> or <Fail> with structured fields).\n- Do NOT leak reasoning into the main output.\n</instructions>\n\n**Strict Rule**: Shaky reasoning = <Fail>, even if answer is correct."),
            "improver": config.get("improver_prompt", "You are a [Master Mathematician]. Revise the solution using the Auditor's feedback.\n\n<instructions>\n- First, analyze the audit, plan corrections, and re-derive any faulty parts in your mind.\n- Compress your entire revision reasoning (what was wrong, how you fixed it, why it’s now sound) into a single paragraph inside <summary>...</summary>.\n- Then, output ONLY the polished, complete solution with final answer in \\boxed{}.\n- No meta-comments. No reasoning outside <summary>.\n</instructions>")
        }

    async def create(
        self,
        instance_id: Optional[str] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        next_role = parameters.get("next_role", "reviewer")
        prompt = self.prompts.get(next_role, self.prompts["reviewer"])
        return ToolResponse(text=prompt), 0.0, {"next_role": next_role}


    async def release(self, instance_id: str, **kwargs) -> None:
        pass  # 无状态，无需清理