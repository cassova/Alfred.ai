from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence
import logging
import yaml
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.tools.render import ToolsRenderer, render_text_description_and_args
from alfred_ai_backend.core.utils.redirect_stream import RedirectStdStreamsToLogger

logger = logging.getLogger(__name__)

class LlmWrapper(ABC):
    def __init__(self, model_file_name: str):
        self.llm = None
        self.model_config = ModelConfig(model_file_name)

    def get_model_config(self):
        return self.model_config

    def get_llm(self):
        return self.llm
    
    @abstractmethod
    def create_agent(self,
        tools: Sequence[BaseTool],
        tools_renderer: ToolsRenderer = render_text_description_and_args,
    ):
        pass
    
    def invoke_agent_executor(self, agent_executor: AgentExecutor, user_input: str) -> Dict[str, Any]:
        with RedirectStdStreamsToLogger(logger):
            inference_config = self.model_config.get_inference_config()
            if inference_config:
                return agent_executor.invoke({"input": user_input}, **inference_config)
            else:
                return agent_executor.invoke({"input": user_input})


class ModelConfig():
    def __init__(self, model_file_name: str):
        model_config_file_name = model_file_name.replace('.py','.yml')
        try:
            with open(model_config_file_name, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded model config from here: {model_config_file_name}")
        except Exception as e:
            logger.error(f"Unable to load model config.  Exepected location: [{model_config_file_name}]  Error message: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def get_init_config(self):
        return self.config.get('init')
    
    def get_inference_config(self):
        return self.config.get('inference')
