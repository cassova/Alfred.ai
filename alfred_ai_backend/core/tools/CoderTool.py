from typing import Optional, Type, Any, Dict
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, load_tools
from langchain_core.callbacks import BaseCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from alfred_ai_backend.core.tools.ToolConfig import ToolConfig
from alfred_ai_backend.models.Model import Model
from langchain_community.agent_toolkits import FileManagementToolkit
from alfred_ai_backend.core.Config import Config


class CoderSchema(BaseModel):
    input: str = Field(description="Technical design with detailed instructions on what code to write or modify")
    cwd: str = Field(description="File path to the python package where code can be found and written to. For example: /temp/<package_name> or c:\\temp\\<package_name>")


class CoderCallbackHandler(BaseCallbackHandler):
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        # TODO: This doesn't seem to work, need to debug
        print(f"Starting Tool: {serialized} - {input_str}")


class CoderTool(BaseTool):
    name: str = "Coder"
    description: str = "Useful for when you need to create or modify code"
    args_schema: Type[BaseModel] = CoderSchema

    _model: Model = PrivateAttr(None)
    _agent_executor: AgentExecutor = PrivateAttr(None)
    _root_config: Config = PrivateAttr(None)
    _tool_config: ToolConfig = PrivateAttr(None)

    def __init__(self, root_config: Config, model_type: Type[Model], **kwargs: Any):
        super().__init__(**kwargs)
        self._root_config = root_config
        self._tool_config = ToolConfig("coder_config.yml")
        self._model = model_type(self._root_config, self._tool_config)
        self._model.initialize_agent(
            tools=self.get_tools(),
            chat_history=True
        )

    def _run(
        self,
        input: str,
        cwd: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return self._model.invoke_agent_executor({'input': input, 'cwd': cwd}, {'callbacks': [CoderCallbackHandler]})
    
    async def _arun(
        self,
        input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # TODO: add async support
        raise NotImplementedError("CoderTool does not support async")
    
    def get_tools(self):
        tools = load_tools(
            ["llm-math"],
            llm=self._model.get_llm(),
            allow_dangerous_tools=True
        )
        file_toolkit = FileManagementToolkit(
            root_dir=str(self._root_config.get('root_folder'))
        )
        tools += file_toolkit.get_tools()
        return tools
