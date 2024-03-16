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
    def __init__(self, name: str):
        self._llm = None
        self._name = name
        self._config = ModelConfig(name)
    
    @abstractmethod
    def message(self):
        pass

    @abstractmethod
    def create_system_prompt_template(self):
        pass

    @abstractmethod
    def create_user_prompt_template(self):
        pass

    def get_model_config(self):
        return self._config
    
    def get_init_config(self):
        return self._config.get('init')

    def get_llm(self):
        return self._llm
    
    def create_agent(self,
        tools: Sequence[BaseTool],
        prompt: ChatPromptTemplate,
        tools_renderer: ToolsRenderer = render_text_description_and_args,
    ):
        return create_structured_chat_agent(
            llm=self._llm,
            tools=tools,
            prompt=prompt,
            tools_renderer=tools_renderer
        )
    
    def invoke_agent_executor(self, agent_executor: AgentExecutor, user_input: str) -> Dict[str, Any]:
        return agent_executor.invoke({"input": user_input}, **self._config.get('inference'))


class ModelConfig():
    def __init__(self, name: str):
        base_dir = alfred_ai_backend.__path__
        try:
            model_config_path = os.path.join(*base_dir + name.split('.')[1:]) + '.yml'
            with open(model_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded model config from here: {model_config_path}")
        except Exception as e:
            logger.error(f"Unable to load model config.  Exepected location: [{model_config_path}]  Error message: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
