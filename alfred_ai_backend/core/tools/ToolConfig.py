import logging
import yaml
from typing import Any, Type
from pathlib import Path

logger = logging.getLogger(__name__)

class ToolConfig():
    #def __init__(self, model_type: Type[Model], tool_config_file_name: str):
    def __init__(self, tool_config_file_name: str, model_path: str):
        try:
            config_file_path = Path(Path(model_path).parent,'config',tool_config_file_name)
            with open(config_file_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded tool config from here: {config_file_path}")
        except Exception as e:
            logger.error(f"Unable to load tool config.  Exepected location: [{config_file_path}]  Error message: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def get_init_config(self):
        return self.config.get('init')
    
    def get_inference_config(self):
        return self.config.get('inference')
