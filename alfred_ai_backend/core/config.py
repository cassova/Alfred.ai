import yaml
from typing import Any
import logging

logger = logging.getLogger(__name__)
CONFIG_FILE_NAME = "config.yml"

class SingletonMeta(type):
    """Creates a single instance of the class"""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    """Holds the root config"""

    def __init__(self, filename: str = CONFIG_FILE_NAME):
        with open(filename, 'r') as f:
            self.config = yaml.safe_load(f)
            logger.info("Agent Manager config loaded successfully")
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def __contains__(self, key):
        return key in self.config
