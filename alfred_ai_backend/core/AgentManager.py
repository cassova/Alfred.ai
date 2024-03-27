from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, AgentExecutor
#from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List, Type
import logging
from alfred_ai_backend.core.Config import Config
from alfred_ai_backend.core.tools.CoderTool import CoderTool
from alfred_ai_backend.core.tools.ToolConfig import ToolConfig
from alfred_ai_backend.core.utils.code_tester_tool import CodeTesterTool
from alfred_ai_backend.models.Model import Model
#from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.globals import set_verbose, set_debug
from langchain.tools import StructuredTool, BaseTool
import sys

logger = logging.getLogger(__name__)
CONFIG_FILE_NAME = "agent_manager_config.yml"

class AgentManager():
    def __init__(self, root_config: Config, model_type: Type[Model]):
        self._model_path = getattr(sys.modules[model_type.__module__], '__file__')
        tool_config = ToolConfig(CONFIG_FILE_NAME, self._model_path)

        self._model_type = model_type
        self._model = model_type(root_config, tool_config)
        self._root_config = root_config

        if root_config.get('enable_langchain_debug_mode', False):
            set_debug(True)
        if root_config.get('enable_langchain_verbose_mode', False):
            set_verbose(True)

        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True,
            output_key="output"
        )        
            
        # initialize agent
        tools = self._get_tools()
        self._agent_executor = self._model.initialize_agent(
            user_input_variables=['input'],
            system_input_variables={'cwd': self._root_config.get('root_folder')},
            tools=tools,
        )

    def _get_tools(self) -> List[BaseTool]:
        tools = load_tools(
            ["llm-math"],
            llm=self._model.get_llm(),
            allow_dangerous_tools=True
        )
        file_toolkit = FileManagementToolkit(
            root_dir=str(self._root_config.get('root_folder')),
            selected_tools=["list_directory", "read_file"],
        )
        tools += file_toolkit.get_tools()
        #tools = tools + [PythonREPLTool()]  # This addresses TypeError: unhashable type: 'PythonREPLTool'

        # This creates an sub-agent that can be used for a specific task
        coder_tool = CoderTool(self._root_config, self._model_type)
        #code_reviewer_tool = CodeTesterTool(self.llm_wrapper.create_tester_agent_executor())
        tools += [coder_tool]

        return tools


    def start_task(self, user_input_str: str) -> Dict[str, Any]:
        logger.info("*** Starting task ***")
        return self._model.invoke_agent_executor({'input': user_input_str})
