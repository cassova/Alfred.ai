from typing import Any, Dict, List, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    LLMResult,
)
import traceback
import logging



class AgentLogger(BaseCallbackHandler):
    """This handles the logging using langchain API

    more details here: https://python.langchain.com/docs/modules/callbacks/
    """
    def __init__(self, name: str, parent: str=None):
        if parent:
            self._name = f"{parent} > {name}"
        else:
            self._name = name
        self._running_tool = None
        self._chat_running = False
        self._logger = logging.getLogger(__name__)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self._chat_running = False
        self._logger.info(f"[{self._name}] - LLM Start")
        self._logger.debug(f"   Serialized: {serialized}")
        self._logger.debug(f"   Prompts: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        if self._chat_running:
            self._logger.info(f"[{self._name}] - Chat Model End")
        else:
            self._logger.info(f"[{self._name}] - LLM End")
        self._logger.debug(f"   Response: {response}")
        self._chat_running = False

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        if self._chat_running:
            self._logger.info(f"[{self._name}] - Chat Model ERROR")
        else:
            self._logger.error(f"[{self._name}] - LLM ERROR")
        self._logger.error(f"   Error: {error}")
        self._logger.error(f"   Traceback:\n{traceback.format_exc()}")
        self._chat_running = False
        
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        self._chat_running = True
        self._logger.info(f"[{self._name}] - Chat Model Start ({serialized['id'][-1]})")
        #self._logger.debug(f"   Serialized: {serialized}")
        self._logger.debug(f"   Messages: {messages}")

    # There is too much messaging with chains to make these logs useful
    # def on_chain_start(
    #     self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    # ) -> Any:
    #     """Run when chain starts running."""
    #     self._logger.info(f"[{self._name}] - Chain Start")
    #     self._logger.debug(f"   Serialized: {serialized}")
    #     self._logger.debug(f"   Inputs: {inputs}")

    # There is too much messaging with chains to make these logs useful
    # def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
    #     """Run when chain ends running."""
    #     self._logger.info(f"[{self._name}] - Chain End")
    #     self._logger.debug(f"   Output: {outputs}")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        self._logger.error(f"[{self._name}] - Chain ERROR")
        self._logger.error(f"   Error: {error}")
        self._logger.error(f"   Traceback:\n{traceback.format_exc()}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        self._running_tool = serialized.get('name', serialized.get('tool', 'unknown'))
        self._logger.info(f"[{self._name} > {self._running_tool}] - STARTED tool")
        self._logger.debug(f"   Tool Inputs: {input_str}")

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self._logger.info(f"[{self._name} > {self._running_tool}] - FINISEHD tool")
        self._logger.debug(f"   Tool Output: {output}")
        self._running_tool = None

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        self._logger.error(f"[{self._name} > {self._running_tool}] - Tool ERROR")
        self._logger.error(f"   Error: {error}")
        self._logger.error(f"   Traceback:\n{traceback.format_exc()}")
        self._running_tool = None

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self._logger.info(f"[{self._name}] - Agent Action")
        self._logger.debug(f"   Action: {action}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        self._logger.info(f"[{self._name}] - Agent Finish")
        self._logger.debug(f"   Finish: {finish}")
