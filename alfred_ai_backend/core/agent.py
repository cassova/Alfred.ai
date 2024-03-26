from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, AgentExecutor
#from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List
import logging
from alfred_ai_backend.core.Config import Config
from alfred_ai_backend.core.tools.CoderTool import CoderTool
from alfred_ai_backend.core.utils.code_tester_tool import CodeTesterTool
from alfred_ai_backend.models.Model import Model
#from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.globals import set_verbose, set_debug
from langchain.tools import StructuredTool, BaseTool


logger = logging.getLogger(__name__)

class AgentWrapper():
    def __init__(self, config: Config, model: Model):
        # TODO: remove most of these `self` things since we don't need them
        self.model = model
        self.config = config

        if config.get('enable_langchain_debug_mode', False):
            set_debug(True)
        if config.get('enable_langchain_verbose_mode', False):
            set_verbose(True)

        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=5, return_messages=True, output_key="output"
        )        
            
        # initialize agent
        tools = self._get_tools()
        agent = self.model.initialize_agent(tools)

        # initialize executor
        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            memory=memory,
        )

    def _get_tools(self) -> List[BaseTool]:
        tools = load_tools(
            ["llm-math"],
            llm=self.model.get_llm(),
            allow_dangerous_tools=True
        )
        file_toolkit = FileManagementToolkit(
            root_dir=str(self.config.get('root_folder')),
            selected_tools=["list_directory"],
        )
        tools += file_toolkit.get_tools()
        #tools = tools + [PythonREPLTool()]  # This addresses TypeError: unhashable type: 'PythonREPLTool'

        # This creates an sub-agent that can be used for a specific task
        coder_tool = CoderTool()
        #code_reviewer_tool = CodeTesterTool(self.llm_wrapper.create_tester_agent_executor())
        tools += [coder_tool]

        return tools


    def start_task(self, user_input_str: str) -> Dict[str, Any]:
        logger.info("*** Starting task ***")
        return self.model.invoke_agent_executor(self._agent_executor, user_input_str)
