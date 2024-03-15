from typing import Optional, Any
from alfred_ai_backend.config import Config
import logging
import yaml
import os


class LlmWrapper():
    def __init__(self, name: str):
        self._llm = None
        self._name = name
        self._config = ModelConfig(name)
        self._logger = logging.getLogger()
    
    @classmethod
    def message(cls):
        pass

    @classmethod
    def create_system_prompt_template(cls):
        pass

    @classmethod
    def create_user_prompt_template(cls):
        pass

    def get_model_config(self):
        return self._config
    
    def get_init_config(self):
        return self._config.get('init')

    def get_llm(self):
        return self._llm
    
    def create_agent(self):
        # TODO: just return create_structured_chat_agent: https://github.com/langchain-ai/langchain/blob/bbe164ad2876badee992382a25cdbc25703fbc6e/libs/langchain/langchain/agents/structured_chat/base.py#L153
        pass


class ModelConfig():
    def __init__(self, name: str):
        base_dir = os.path.dirname(__file__)
        try:
            model_config_path = os.path.join(*[base_dir] + name.split('.')) + '.yml'
            with open(model_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logging.info(f"Loaded model config from here: {model_config_path}")
        except Exception as e:
            logging.error(f"Unable to load model config.  Exepected location: [{model_config_path}]  Error message: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
