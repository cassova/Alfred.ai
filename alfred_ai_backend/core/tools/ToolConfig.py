import logging
import yaml
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

class ToolConfig():
    """Holds the configuration of a model's tool"""

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
        """Retrieve a configuration value by its key

        Args:
            key (str): Configuration key
            default (Any, optional): Default for when the value does not exist. Defaults to None.

        Returns:
            Any: The value in the configuration
        """
        return self.config.get(key, default)
    
    def get_init_config(self) -> Any:
        """Retrieves the model's initialization configuration

        Returns:
            Any: The configuration settings usually in key-value pairs
        """
        return self.config.get('init')
    
    def get_inference_config(self) -> Any:
        """Retrieves the model's inference configuration

        Returns:
            Any: The configuration settings usually in key-value pairs
        """
        return self.config.get('inference')
