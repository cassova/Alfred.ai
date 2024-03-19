from typing import Dict, Any, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

class CodeWriterSchema(BaseModel):
    input: Dict[str, Any] = Field(description="high-level design with detailed instructions. In format: {'input': '<<instruction>>'}")


def get_code_writer_tool(func) -> StructuredTool:
    python_coder_tool = StructuredTool.from_function(
        func=func,
        name="code_writer",
        description="Useful for when you need to create or modify a piece of code",
        args_schema=CodeWriterSchema,
        #return_direct=True,  #not allowed in multi-action agents
    )
    return python_coder_tool
