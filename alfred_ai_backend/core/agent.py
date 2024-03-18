from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any
import logging
from alfred_ai_backend.core.config import Config
from alfred_ai_backend.models.llm import LlmWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.globals import set_verbose, set_debug


logger = logging.getLogger(__name__)

# Source: https://github.com/pinecone-io/examples/blob/master/learn/generation/llm-field-guide/llama-2/llama-2-70b-chat-agent.ipynb
class AgentWrapper():
    def __init__(self, config: Config, llm_wrapper: LlmWrapper):
        # TODO: remove most of these `self` things since we don't need them
        self.llm_wrapper = llm_wrapper
        self.config = config

        if config.get('enable_langchain_debug_mode', False):
            set_debug(True)
        if config.get('enable_langchain_verbose_mode', False):
            set_verbose(True)

        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=5, return_messages=True, output_key="output"
        )
        tools = load_tools(
            ["llm-math", "terminal", "human"],
            llm=self.llm_wrapper.get_llm(),
            allow_dangerous_tools=True
        )
        file_toolkit = FileManagementToolkit(
            root_dir=str(config.get('root_folder'))
        )
        tools += file_toolkit.get_tools()
        #tools = tools + [PythonREPLTool()]  # This addresses TypeError: unhashable type: 'PythonREPLTool'
            
        # initialize agent
        agent = self.llm_wrapper.create_agent(tools)

        # initialize executor
        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            memory=memory,
        )

    def start_task(self, user_input_str: str) -> Dict[str, Any]:
        logger.info("*** Starting task ***")
        return self.llm_wrapper.invoke_agent_executor(self._agent_executor, user_input_str)
