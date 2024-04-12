from typing import Any, Dict, List, Union, Optional
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    LLMResult,
)
import traceback
import time
import sys

def get_colored_text(text: str, color: Optional[str]="yellow", **kwargs):
    _TEXT_COLOR_MAPPING = {
        "blue": "36;1",
        "yellow": "33;1",
        "pink": "38;5;200",
        "green": "32;1",
        "red": "31;1",
    }
    return f"\u001b[{_TEXT_COLOR_MAPPING[color]}m\033[1;3m{text}\u001b[0m"


class StatusMessaging(StdOutCallbackHandler):
    """This handles additional console status messaging using langchain API

    more details here: https://python.langchain.com/docs/modules/callbacks/
    """
    def __init__(self, name: str, parent: str=None):
        if parent:
            self._name = f"{parent} > {name}"
        else:
            self._name = name
        self._running_tool = None
        self._chat_running = False
        self._p = "  "*len(self._name.split(' > '))

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self._chat_running = False
        self._llm_timer = time.perf_counter()
        print(f"{self._p}[{self._name}] - Running LLM...", end="")
        sys.stdout.flush()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        t = f"{time.perf_counter() - self._llm_timer:0.1f} sec"
        print(f"Done ({t})")
        self._chat_running = False
        sys.stdout.flush()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        print(get_colored_text("ERROR","red"), end="")
        t = f"{time.perf_counter() - self._llm_timer:0.1f} sec"
        print(f" ({t})")
        print(f"{self._p}  Error: {error}")
        print(f"{self._p}  Traceback:\n{traceback.format_exc()}")
        self._chat_running = False
        
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        self._chat_running = True
        self._llm_timer = time.perf_counter()
        print(f"{self._p}[{self._name}] - Running Chat Model {serialized['id'][-1]}...", end="")
        sys.stdout.flush()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        # Chain messaging is just too much...
        # print(f"{self._p}[{self._name}] - Chain Start")
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        # Chain messaging is just too much...
        # print(f"{self._p}[{self._name}] - Chain End")
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        # Chain messaging is just too much...
        # errmsg = get_colored_text(f"Chain ERROR", "red")
        # print(f"{self._p}[{self._name}] - {errmsg}")
        # print(f"{self._p}  Error: {error}")
        # print(f"{self._p}  Traceback:\n{traceback.format_exc()}")
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        self._tool_timer = time.perf_counter()
        self._running_tool = serialized.get('name', serialized.get('tool', 'unknown'))
        print(f"{self._p}[{self._name}] - Starting tool {self._running_tool}")
        sys.stdout.flush()

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        t = f"{time.perf_counter() - self._tool_timer:0.1f} sec"
        print(f"{self._p}[{self._name}] - Finished with tool {self._running_tool} ({t})")
        self._running_tool = None
        sys.stdout.flush()

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        errmsg = get_colored_text(f"Tool ERROR ({self._running_tool})", "red")
        print(f"{self._p}[{self._name}] - {errmsg}", end="")
        t = f"{time.perf_counter() - self._tool_timer:0.1f} sec"
        print(f" ({t})")
        print(f"{self._p}  Error: {error}")
        print(f"{self._p}  Traceback:\n{traceback.format_exc()}")
        self._running_tool = None

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        print(f"{self._p}[{self._name}] - Beginning Agent Action: {action.get('tool', 'unknown')}")
        sys.stdout.flush()

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        print(f"{self._p}[{self._name}] - Finished Agent Execution")
        sys.stdout.flush()
