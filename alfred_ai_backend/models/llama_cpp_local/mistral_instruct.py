from alfred_ai_backend.core.utils.redirect_stream import RedirectStdStreamsToLogger
from alfred_ai_backend.models.llm import LlmWrapper
from typing import Sequence, Union, Dict, Any
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
from langchain.agents import AgentExecutor
import logging


logger = logging.getLogger(__name__)
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<s> ", " </s>"

class MistralInstruct(LlmWrapper):
    def __init__(self):
        super().__init__(__name__)

        # Load the model using the model's init config and send any std messages to logger
        with RedirectStdStreamsToLogger(logger):
            self._config = self.get_model_config()
            self._llm = LlamaCpp(**self._config.get_init_config())

    def create_system_prompt_template(self) -> str:
        return f"{B_SYS} {B_INST} {self._config.get('system_prompt_template')} {E_INST} "

    def create_user_prompt_template(self) -> str:
        return f"{B_INST} {self._config.get('user_prompt')} {E_INST}\n {self._config.get('user_prompt_context')}\n{self.get_response_prefix()}"

    def get_response_prefix(self) -> str:
        return self._config.get('user_prompt_starter_response').strip()

    def create_agent(
        self,
        tools: Sequence[BaseTool],
        prompt: ChatPromptTemplate,
        tools_renderer: ToolsRenderer = render_text_description_and_args,
    ) -> Runnable:
        # This is based on langchain.agents.create_structured_chat_agent but customized for Mistral
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
    
    def invoke_agent_executor(self, agent_executor: AgentExecutor, user_input: str) -> Dict[str, Any]:
        with RedirectStdStreamsToLogger(logger):
            return agent_executor.invoke({"input": user_input}, **self._config.get_inference_config())
    

class MistralJsonOutputParser(JSONAgentOutputParser):
    def __init__(self, llm_wrapper: MistralInstruct):
        super().__init__()
        # Hack to avoid missing pydantic variable errors
        self.__dict__['_llm_wrapper'] = llm_wrapper

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            # Add the json prefix that we added at the end of our user prompt
            logger.debug(f"Parsing response: {text}")
            response_prefix = self.__dict__['_llm_wrapper'].get_response_prefix().replace('{{','{')
            text = text.strip()
            if not text.startswith('```json') and self.__dict__['_llm_wrapper']:
                text = (response_prefix + text)

            response = parse_json_markdown(text)
            if isinstance(response, list):
                # gpt turbo frequently ignores the directive to emit a single action
                logger.warning("Got multiple action responses: %s", response)
                response = response[0]

            # Mistral frequently ignores action_input needing to be a string
            action_input = response.get("action_input", {})
            if isinstance(action_input, dict):
                action_input = action_input.get('values', str(action_input))
                if isinstance(action_input, list) and len(action_input)>0:
                    action_input = action_input[0]

            logger.debug(f"Converted action_input to string: {action_input}")

            if response["action"] == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                return AgentAction(
                    response["action"], action_input, text
                )
        except Exception as e:
            logger.error(f"Failed to parse with error: {e}")
            raise OutputParserException(f"Could not parse LLM output: {text}") from e
