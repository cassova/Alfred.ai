from typing import Union, Dict, Any
from langchain_core.agents import AgentAction, AgentFinish
import ast
import json
import logging

logger = logging.getLogger(__name__)

def parse_tool_input(tool_input: Union[Dict,str]) -> Any:
    if isinstance(tool_input,str):
        try:
            tool_input = json.loads(tool_input)
        except Exception:
            try:
                tool_input = ast.literal_eval(tool_input)
            except Exception:
                #raise ValueError(f"Unable to parse tool_input: {tool_input}")
                logger.warn(f"Unable to parse tool_input but this might be OK if the tool supports a string. Continuing. Tool input: {tool_input}")
                pass
    
    # Took this code and comment from here: from langchain.agents.output_parsers.openai_tools import parse_ai_message_to_openai_tool_action
    # since we were getting this problem with calling some tools
    # HACK HACK HACK:
    # The code that encodes tool input into Open AI uses a special variable
    # name called `__arg1` to handle old style tools that do not expose a
    # schema and expect a single string argument as an input.
    # We unpack the argument here if it exists.
    # Open AI does not support passing in a JSON array as an argument.
    if "__arg1" in tool_input:
        tool_input = tool_input["__arg1"]
    else:
        tool_input = tool_input

    return tool_input


def parse_content(text: str) -> Union[AgentAction, AgentFinish]:
    if "</tool>" in text:
        tool, tool_input = text.split("</tool>")
        _tool = tool.split("<tool>")[1]
        _tool_input = tool_input.split("<tool_input>")[1]
        if "</tool_input>" in _tool_input:
            _tool_input = _tool_input.split("</tool_input>")[0]
        return AgentAction(tool=_tool, tool_input=parse_tool_input(_tool_input), log=text)
    elif "<final_answer>" in text:
        _, answer = text.split("<final_answer>")
        if "</final_answer>" in answer:
            answer = answer.split("</final_answer>")[0]
        return AgentFinish(return_values={"output": answer}, log=text)
    else:
        raise ValueError
