from typing import Union, List, Dict, Any
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.exceptions import OutputParserException
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from alfred_ai_backend.models.anthropic.claude.utils import parse_tool_input, parse_content
import json


class ClaudeAgentOutputParser(JsonOutputToolsParser):
    
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        print(f"Got Claude result: {result}")
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message

        try:
            # Try to parse tool or final answer from text
            text = message.content
            return parse_content(text)
        except ValueError:
            # Extract tool and return agent action
            json_results = super().parse_result(result, partial=partial)
            if self.first_tool_only:
                return AgentAction(
                    tool=json_results["type"],
                    tool_input=parse_tool_input(json_results["args"]),
                    log=text
                )
            return [
                AgentAction(
                    tool=r["type"],
                    tool_input=parse_tool_input(r["args"]),
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
    
