from typing import Optional, Type, Any
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class CodeTesterSchema(BaseModel):
    input: str = Field(description="instructions on how to test the code")


class CodeTesterTool(BaseTool):
    name: str = "CodeTester"
    description: str = "Useful for when you need need to run a test on a piece of code"
    args_schema: Type[BaseModel] = CodeTesterSchema
    _agent_executor: AgentExecutor = PrivateAttr(None)

    def __init__(self, agent_executor: AgentExecutor, **kwargs: Any):
        super().__init__( **kwargs)
        self._agent_executor = agent_executor

    def _run(
        self, input: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self._agent_executor.invoke({'input': input})
    
    async def _arun(
        self,
        input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("code_reviewer does not support async")
