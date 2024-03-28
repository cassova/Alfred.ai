from typing import Optional, Type, Any, Dict
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, load_tools
from langchain_core.callbacks import BaseCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from alfred_ai_backend.core.tools.DebugTool import DebugTool
from alfred_ai_backend.core.tools.ToolConfig import ToolConfig
from alfred_ai_backend.models.Model import Model
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_experimental.tools import PythonREPLTool
from alfred_ai_backend.core.Config import Config
import sys

root_config = Config()
CONFIG_FILE_NAME = "tester_tool_config.yml"

class TesterSchema(BaseModel):
    input: str = Field(description="Detailed instructions on what to test and how to run the tests")
    cwd: str = Field(description="Current working directory from where the tests are run")
    pkg_name: str = Field(description="The package name and subfolder where all code should reside")


class TesterCallbackHandler(BaseCallbackHandler):
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        # TODO: This doesn't seem to work, need to debug
        print(f"Starting Tool: {serialized} - {input_str}")


class TesterTool(BaseTool):
    name: str = "Tester"
    description: str = "Useful for when you need to test code to make sure it works, run unit tests, and look for issues"
    args_schema: Type[BaseModel] = TesterSchema

    _model: Model = PrivateAttr(None)
    _model_path: str = PrivateAttr(None)
    _agent_executor: AgentExecutor = PrivateAttr(None)
    _tool_config: ToolConfig = PrivateAttr(None)

    def __init__(self, model_type: Type[Model], **kwargs: Any):
        super().__init__(**kwargs)
        self._model_path = getattr(sys.modules[model_type.__module__], '__file__')
        self._tool_config = ToolConfig(CONFIG_FILE_NAME, self._model_path)
        self._model = model_type(self._tool_config)
        self._model.initialize_agent(
            user_input_variables=['input'],
            system_input_variables={'cwd':root_config.get('root_folder')},
            tools=self._get_tools(model_type),
            chat_history=True
        )

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any
    ) -> str:
        """Use the tool."""
        resp = self._model.invoke_agent_executor(kwargs, {'callbacks': [TesterCallbackHandler]})
        return resp.get('output', ' [[no response]]')
    
    async def _arun(
        self,
        input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # TODO: add async support
        raise NotImplementedError("TesterTool does not support async")
    
    def _get_tools(self, model_type: Type[Model]):
        tools = load_tools(
            ["llm-math"],
            llm=self._model.get_llm(),
            allow_dangerous_tools=True
        )
        file_toolkit = FileManagementToolkit(
            root_dir=str(root_config.get('root_folder')),
            selected_tools=["list_directory", "read_file", "file_search"],
        )
        debug_tool = DebugTool(model_type)

        tools += file_toolkit.get_tools()
        tools += [PythonREPLTool()]  # This addresses TypeError: unhashable type: 'PythonREPLTool'
        tools += [debug_tool]
        return tools
