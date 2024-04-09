from langchain.agents import load_tools
#from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List, Type
import logging
from alfred_ai_backend.core.Config import Config
from alfred_ai_backend.core.tools.CoderTool import CoderTool
from alfred_ai_backend.core.tools.TesterTool import TesterTool
from alfred_ai_backend.core.utils.ToolConfig import ToolConfig
from alfred_ai_backend.core.utils.AgentLogger import AgentLogger
from alfred_ai_backend.models.Model import Model
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.globals import set_verbose, set_debug
from langchain.tools import BaseTool
import sys

logger = logging.getLogger(__name__)
root_config = Config()
CONFIG_FILE_NAME = "agent_manager_config.yml"

class AgentManager():
    """The main AI agent that manages other agents via tools"""

    def __init__(self, model_type: Type[Model]):
        self._model_path = getattr(sys.modules[model_type.__module__], '__file__')
        tool_config = ToolConfig(CONFIG_FILE_NAME, self._model_path)

        self._model_type = model_type
        self._model = model_type(tool_config)

        if root_config.get('enable_langchain_debug_mode', False):
            set_debug(True)
        if root_config.get('enable_langchain_verbose_mode', False):
            set_verbose(True)
            
        # initialize agent
        tools = self._get_tools()
        self._agent_executor = self._model.initialize_agent(
            user_input_variables=['input'],
            system_input_variables={'cwd': root_config.get('root_folder')},
            tools=tools,
        )

    def _get_tools(self) -> List[BaseTool]:
        tools = load_tools(
            ["llm-math"],
            llm=self._model.get_llm(),
            allow_dangerous_tools=True
        )
        file_toolkit = FileManagementToolkit(
            root_dir=str(root_config.get('root_folder')),
            selected_tools=["list_directory", "read_file", "file_search"],
        )
        tools += file_toolkit.get_tools()

        # This creates sub-agents that can be used for a specific task
        coder_tool = CoderTool(self._model_type, parent="AgentManager")
        tester_tool = TesterTool(self._model_type, parent="AgentManager")
        tools += [coder_tool, tester_tool]

        return tools

    def start_task(self, user_input_str: str) -> Dict[str, Any]:
        logger.info("*** Starting task ***")
        return self._model.invoke_agent_executor({'input': user_input_str}, {'callbacks': [AgentLogger("AgentManager")]})
