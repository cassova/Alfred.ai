from typing import Sequence, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.tools.render import ToolsRenderer, render_text_description_and_args
from langchain.memory import ConversationBufferWindowMemory
import logging
from langchain_community.callbacks import get_openai_callback
from alfred_ai_backend.core.Config import Config
from alfred_ai_backend.core.tools.ToolConfig import ToolConfig
from alfred_ai_backend.models.Model import Model
from langchain_core.runnables import RunnableConfig


logger = logging.getLogger(__name__)

class ChatGpt(Model):
    def __init__(self, root_config: Config, tool_config: Optional[ToolConfig] = None):
        super().__init__(root_config, tool_config)
        
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "Could not import langchain_openai library. "
                "Please install the langchain_openai library to "
                "use this embedding model: pip install langchain-openai"
            )

        # TODO Need to fix the config here.... **************************************************
        self._llm = ChatOpenAI(**self._tool_config.get_init_config())
    
    def initialize_agent(
        self,
        tools: Sequence[BaseTool],
        tools_renderer: Optional[ToolsRenderer] = render_text_description_and_args,
        chat_history: Optional[bool] = True,
    ) -> AgentExecutor:

        if chat_history:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['cwd'], template=self._tool_config.get('system_prompt_template'))),
                MessagesPlaceholder(variable_name='chat_history', optional=True),
                HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template=self._tool_config.get('user_prompt_template'))),
                MessagesPlaceholder(variable_name='agent_scratchpad')
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['cwd'], template=self._tool_config.get('system_prompt_template'))),
                HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template=self._tool_config.get('user_prompt_template'))),
                MessagesPlaceholder(variable_name='agent_scratchpad')
            ])
        
        prompt = prompt.partial(
            cwd=self._root_config.get('root_folder'),
        )
        agent = create_openai_tools_agent(self._llm, tools, prompt)

        memory = None
        if chat_history:
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history", k=5, return_messages=True, output_key="output"
            )
        
        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            memory=memory,
        )

    def invoke_agent_executor(self, input: Dict[str, Any], inference_config: Optional[RunnableConfig] = None, **kwargs: Any  ) -> Dict[str, Any]:
        with get_openai_callback() as cb:
            response = self._agent_executor.invoke(input, inference_config, **kwargs)

            logger.info(f"Total Tokens: {cb.total_tokens}")
            logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
            logger.info(f"Completion Tokens: {cb.completion_tokens}")
            logger.info(f"Total Cost (USD): ${cb.total_cost}")
            return response
