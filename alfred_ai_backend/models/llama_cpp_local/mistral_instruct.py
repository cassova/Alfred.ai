from alfred_ai_backend.models.llama_cpp_local.utils import RedirectStdStreamsToLogger
from alfred_ai_backend.models.llm import LlmWrapper
from typing import Sequence, Union
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.tools.render import ToolsRenderer, render_text_description_and_args
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import parse_json_markdown
import logging


logger = logging.getLogger(__name__)
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<s> ", " </s>"

class MistralInstruct(LlmWrapper):
    def __init__(self):
        super().__init__('llama_cpp_local.mistral_instruct')

        # Load the model using the model's init config and send any std messages to logger
        with RedirectStdStreamsToLogger(self._logger):
            self._config = self.get_model_config()
            self._llm = LlamaCpp(**self.get_init_config())

    def create_system_prompt_template(self) -> str:
        return f"{B_SYS} {B_INST} {self._config.get('system_prompt_template')} {E_INST} "

    def create_user_prompt_template(self) -> str:
        return f"{B_INST} {self._config.get('user_prompt')} {E_INST}\n {self._config.get('user_prompt_context')}\n{self.get_response_prefix()}"

    def get_response_prefix(self) -> str:
        return self._config.get('user_prompt_starter_response')

    def create_agent(
        self,
        tools: Sequence[BaseTool],
        prompt: ChatPromptTemplate,
        tools_renderer: ToolsRenderer = render_text_description_and_args,
    ) -> Runnable:
        # This is based on langchain.agents.create_structured_chat_agent
        missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
            prompt.input_variables
        )
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")

        prompt = prompt.partial(
            tools=tools_renderer(list(tools)),
            tool_names=", ".join([t.name for t in tools]),
        )
        llm_with_stop = self._llm.bind(stop=["Observation"])

        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            )
            | prompt
            | llm_with_stop
            | MistralJsonOutputParser(self)
        )
        return agent
    

class MistralJsonOutputParser(JSONAgentOutputParser):
    def __init__(self, llm_wrapper: MistralInstruct):
        # Hack to avoid missing pydantic variable errors
        self.__dict__['_llm_wrapper'] = llm_wrapper

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            # Add the json prefix that we added at the end of our user prompt
            logger.debug(f"Parsing output: {text}")
            response_prefix = self.__dict__['_llm_wrapper'].get_response_prefix().replace('{{','{')
            text = text.strip().replace(r'\n','')
            if not text.startswith('```json') and self.__dict__['_llm_wrapper']:
                text = (response_prefix + text)
                logger.debug(f"Cleaned: {text.strip()}")

            response = parse_json_markdown(text)
            logger.debug(f"Parsed: {response}")
            if isinstance(response, list):
                # gpt turbo frequently ignores the directive to emit a single action
                logger.warning("Got multiple action responses: %s", response)
                response = response[0]
            if response["action"] == "Final Answer":
                return AgentFinish({"output": response["action_input"]}, text)
            else:
                return AgentAction(
                    response["action"], response.get("action_input", {}), text
                )
        except Exception as e:
            logger.error(f"Failed to parse with error: {e}")
            raise OutputParserException(f"Could not parse LLM output: {text}") from e