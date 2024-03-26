import logging
import yaml
from typing import Any

logger = logging.getLogger(__name__)

class ToolConfig():
    def __init__(self, tool_config_file_name: str):
        try:
            with open(tool_config_file_name, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded tool config from here: {tool_config_file_name}")
        except Exception as e:
            logger.error(f"Unable to load tool config.  Exepected location: [{tool_config_file_name}]  Error message: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def get_init_config(self):
        return self.config.get('init')
    
    def get_inference_config(self):
        return self.config.get('inference')
