from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence
import logging
import yaml
import os
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.tools.render import ToolsRenderer, render_text_description_and_args
import alfred_ai_backend

logger = logging.getLogger(__name__)

class LlmWrapper(ABC):
    def __init__(self, model_file_name: str):
        self.llm = None
        self.model_file_name = model_file_name
        self.config = ModelConfig(model_file_name)

    @abstractmethod
    def create_system_prompt_template(self):
        pass

    @abstractmethod
    def create_user_prompt_template(self):
        pass

    def get_model_config(self):
        return self.config

    def get_llm(self):
        return self.llm
    
    def create_agent(self,
        tools: Sequence[BaseTool],
        prompt: ChatPromptTemplate,
        tools_renderer: ToolsRenderer = render_text_description_and_args,
    ):
        return create_structured_chat_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt,
            tools_renderer=tools_renderer
        )
    
    def invoke_agent_executor(self, agent_executor: AgentExecutor, user_input: str) -> Dict[str, Any]:
        return agent_executor.invoke({"input": user_input}, **self.config.get_inference_config())


class ModelConfig():
    def __init__(self, model_file_name: str):
        try:
            model_config_file_name = model_file_name.replace('.py','.yml')
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
