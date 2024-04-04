from typing import Union, List, Dict, Any
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.exceptions import OutputParserException
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
import json

class ClaudeAgentOutputParser(JsonOutputToolsParser):
    
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message

        try:
            # Try to parse tool or final answer from text
            text = message.content
            return self._parse_content(text)
        except ValueError:
            # Extract tool and return agent action
            json_results = super().parse_result(result, partial=partial)
            if self.first_tool_only:
                return AgentAction(
                    tool=json_results["type"],
                    tool_input=self._parse_tool_input(json_results["args"]),
                    log=text
                )
            return [
                AgentAction(
                    tool=r["type"],
                    tool_input=self._parse_tool_input(r["args"]),
                    log=text
                ) for r in json_results
            ]


    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")

    def get_format_instructions(self) -> str:
        raise NotImplementedError

    @property
    def _type(self) -> str:
        return "xml-agent"
    
    def _parse_tool_input(self, tool_input: Union[Dict,str]):
        if isinstance(tool_input,str):
            try:
                tool_input = json.loads(tool_input)
            except Exception:
                raise ValueError(f"Unable to parse tool_input: {tool_input}")
        
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

    
    def _parse_content(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "</tool>" in text:
            tool, tool_input = text.split("</tool>")
            _tool = tool.split("<tool>")[1]
            _tool_input = tool_input.split("<tool_input>")[1]
            if "</tool_input>" in _tool_input:
                _tool_input = _tool_input.split("</tool_input>")[0]
            return AgentAction(tool=_tool, tool_input=_tool_input, log=text)
        elif "<final_answer>" in text:
            _, answer = text.split("<final_answer>")
            if "</final_answer>" in answer:
                answer = answer.split("</final_answer>")[0]
            return AgentFinish(return_values={"output": answer}, log=text)
        else:
            raise ValueError
