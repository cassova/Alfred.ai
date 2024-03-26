from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Optional
import logging
import yaml
from langchain.agents import AgentExecutor
from langchain_core.tools import BaseTool
from langchain.tools.render import ToolsRenderer, render_text_description_and_args
from alfred_ai_backend.core.Config import Config
from langchain_core.runnables import RunnableConfig

from alfred_ai_backend.core.tools.ToolConfig import ToolConfig
#from alfred_ai_backend.core.utils.redirect_stream import RedirectStdStreamsToLogger
#from langchain_community.callbacks import wandb_tracing_enabled

logger = logging.getLogger(__name__)

class Model(ABC):
    def __init__(self, root_config: Config, tool_config: Optional[ToolConfig] = None):
        self._llm  = None
        self._agent_executor: AgentExecutor = None
        self._root_config = root_config
        self._tool_config = tool_config

    def get_llm(self):
        return self._llm

    @abstractmethod
    def initialize_agent(
        self,
        tools: Sequence[BaseTool],
        tools_renderer: Optional[ToolsRenderer] = render_text_description_and_args,
        chat_history: Optional[bool] = True,
    ):
        pass
    
    def invoke_agent_executor(self, input: Dict[str, Any], inference_config: Optional[RunnableConfig] = None, **kwargs: Any ) -> Dict[str, Any]:
        #with RedirectStdStreamsToLogger(logger):
            #with wandb_tracing_enabled():
        return self._agent_executor.invoke(input, inference_config, **kwargs)
