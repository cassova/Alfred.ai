import yaml
from typing import Any
import logging

logger = logging.getLogger(__name__)

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    def __init__(self, filename: str = 'config.yml'):
        with open(filename, 'r') as f:
            self.config = yaml.safe_load(f)
            logger.info("Agent Manager config loaded successfully")
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def __contains__(self, key):
        return key in self.config
