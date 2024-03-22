from typing import Optional, Type, Any, Dict
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class CodeWriterSchema(BaseModel):
    input: str = Field(description="high-level design with detailed instructions")
    cwd: str = Field(description="file path to the package contents. for example: /temp/<package_name> or c:\\temp\\<package_name>")


class CodeWriterCallbackHandler(BaseCallbackHandler):
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        print(f"Starting Tool: {serialized} - {input_str}")

class CodeWriterTool(BaseTool):
    name: str = "CodeWriter"
    description: str = "Useful for when you need to create or modify a piece of code"
    args_schema: Type[BaseModel] = CodeWriterSchema
    _agent_executor: AgentExecutor = PrivateAttr(None)

    def __init__(self, agent_executor: AgentExecutor, **kwargs: Any):
        super().__init__(**kwargs)
        self._agent_executor = agent_executor

    def _run(
        self, input: str, cwd: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self._agent_executor.invoke({'input': input, 'cwd': cwd}, {'callbacks': [CodeWriterCallbackHandler]})
    
    async def _arun(
        self,
        input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("code_reviewer does not support async")
