from typing import Optional, Type, Any
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain.agents import load_tools, AgentExecutor
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_experimental.tools import PythonREPLTool
from alfred_ai_backend.core.utils.StatusMessaging import StatusMessaging
from alfred_ai_backend.core.utils.ToolConfig import ToolConfig
from alfred_ai_backend.core.utils.AgentLogger import AgentLogger
from alfred_ai_backend.models.Model import Model
from alfred_ai_backend.core.Config import Config
import sys

root_config = Config()
CONFIG_FILE_NAME = "debug_tool_config.yml"

class DebugSchema(BaseModel):
    input: str = Field(description="Detailed description of the issue, its location, the error message, the stack trace, and instructions on how to recreate the issue")
    cwd: str = Field(description="Current working directory from which the issue can be recreated and debugged")
    pkg_name: str = Field(description="The package name and subfolder where all code should reside")


class DebugTool(BaseTool):
    name: str = "Debugger"
    description: str = "Useful for when you need to debug and fix broken code. Use to find where in the code there is a problem."
    args_schema: Type[BaseModel] = DebugSchema

    _model: Model = PrivateAttr(None)
    _model_path: str = PrivateAttr(None)
    _agent_executor: AgentExecutor = PrivateAttr(None)
    _tool_config: ToolConfig = PrivateAttr(None)
    _parent: str = PrivateAttr("")

    def __init__(self, model_type: Type[Model], parent: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._parent = parent
        self._model_path = getattr(sys.modules[model_type.__module__], '__file__')
        self._tool_config = ToolConfig(CONFIG_FILE_NAME, self._model_path)
        self._model = model_type(self._tool_config)
        self._model.initialize_agent(
            user_input_variables=['input'],
            system_input_variables={'cwd':root_config.get('root_folder')},
            tools=self._get_tools(),
            chat_history=True
        )

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any
    ) -> str:
        """Use the tool."""
        resp = self._model.invoke_agent_executor(kwargs, {'callbacks': [AgentLogger("Debugger", self._parent), StatusMessaging("Debugger", self._parent)]})
        return resp.get('output', ' [[no response]]')
    
    async def _arun(
        self,
        input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # TODO: add async support
        raise NotImplementedError("DebugTool does not support async")
    
    def _get_tools(self):
        tools = load_tools(
            ["llm-math"],
            llm=self._model.get_llm(),
            allow_dangerous_tools=True
        )
        file_toolkit = FileManagementToolkit(
            root_dir=str(root_config.get('root_folder'))
        )
        tools += file_toolkit.get_tools()
        tools += [PythonREPLTool()]  # This addresses TypeError: unhashable type: 'PythonREPLTool'
        return tools
