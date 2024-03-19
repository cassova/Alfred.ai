from typing import Dict, Any, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

class CodeReviewerSchema(BaseModel):
    input: Dict[str, Any] = Field(description="instructions on what to review and how to review it. In format: {'input': '<<instruction>>'}")


def get_code_reviewer_tool(func) -> StructuredTool:
    python_coder_tool = StructuredTool.from_function(
        func=func,
        name="code_reviewer",
        description="Useful for when you need need to test or review a piece of code",
        args_schema=CodeReviewerSchema,
        #return_direct=True,  #not allowed in multi-action agents
    )
    return python_coder_tool
