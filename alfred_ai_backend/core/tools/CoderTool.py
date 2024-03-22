from typing import Optional, Type, Any, Dict
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class CoderSchema(BaseModel):
    input: str = Field(description="Technical design with detailed instructions on what to build")
    cwd: str = Field(description="File path to the package contents. for example: /temp/<package_name> or c:\\temp\\<package_name>")


class CoderCallbackHandler(BaseCallbackHandler):
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        print(f"Starting Tool: {serialized} - {input_str}")

class CoderTool(BaseTool):
    name: str = "Coder"
    description: str = "Useful for when you need to create or modify a piece of code"
    args_schema: Type[BaseModel] = CoderSchema
    _agent_executor: AgentExecutor = PrivateAttr(None)

    def __init__(self, agent_executor: AgentExecutor, **kwargs: Any):
        super().__init__(**kwargs)
        self._agent_executor = agent_executor

    def _run(
        self, input: str, cwd: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self._agent_executor.invoke({'input': input, 'cwd': cwd}, {'callbacks': [CoderCallbackHandler]})
    
    async def _arun(
        self,
        input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CoderTool does not support async")
