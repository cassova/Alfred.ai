from typing import Sequence, Dict, Any
from langchain.agents import create_openai_functions_agent
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.tools.render import ToolsRenderer, render_text_description_and_args
import logging
from langchain_community.callbacks import get_openai_callback
from langchain.agents import AgentExecutor
from alfred_ai_backend.core.config import Config
from alfred_ai_backend.models.llm import LlmWrapper
from alfred_ai_backend.core.utils.redirect_stream import RedirectStdStreamsToLogger


logger = logging.getLogger(__name__)

class ChatGpt(LlmWrapper):
    def __init__(self, config: Config):
        super().__init__(__file__, config)
        
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "Could not import langchain_openai library. "
                "Please install the langchain_openai library to "
                "use this embedding model: pip install langchain-openai"
            )

        self.model_config = self.get_model_config()
        self.llm = ChatOpenAI(**self.model_config.get_init_config())
    
    def create_agent(
        self,
        tools: Sequence[BaseTool],
        tools_renderer: ToolsRenderer = render_text_description_and_args,
    ) -> Runnable:

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['cwd'], template=self.model_config.get('system_prompt_template'))),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template=self.model_config.get('user_prompt'))),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])
        
        prompt = prompt.partial(
            cwd=self.config.get('root_folder'),
        )

        agent = create_openai_functions_agent(self.llm, tools, prompt)
        return agent

    def invoke_agent_executor(self, agent_executor: AgentExecutor, user_input: str) -> Dict[str, Any]:
        with RedirectStdStreamsToLogger(logger):
            with get_openai_callback() as cb:
                inference_config = self.model_config.get_inference_config()
                if inference_config:
                    response = agent_executor.invoke({"input": user_input}, **inference_config)
                else:
                    response = agent_executor.invoke({"input": user_input})
                
                logger.info(f"Total Tokens: {cb.total_tokens}")
                logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
                logger.info(f"Completion Tokens: {cb.completion_tokens}")
                logger.info(f"Total Cost (USD): ${cb.total_cost}")
                return response
