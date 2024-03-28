from alfred_ai_backend.core.utils.RedirectStdStreamsToLogger import RedirectStdStreamsToLogger
from alfred_ai_backend.models.Model import Model
from typing import Sequence, Union, Optional
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools.render import ToolsRenderer, render_text_description_and_args
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.agents import AgentExecutor, AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import parse_json_markdown
from langchain.memory import ConversationBufferWindowMemory
from alfred_ai_backend.core.tools.ToolConfig import ToolConfig
import logging


logger = logging.getLogger(__name__)
B_INST, E_INST = "[INST]", "[/INST]"
BOS, EOS = "<s> ", " </s>"

class MistralInstruct(Model):
    def __init__(self, tool_config: Optional[ToolConfig] = None):
        super().__init__(tool_config)
        # Load the model using the model's init config and send any std messages to logger
        with RedirectStdStreamsToLogger(logger):
            self._llm = LlamaCpp(**self._tool_config.get.get_init_config())

    def initialize_agent(
        self,
        tools: Sequence[BaseTool],
        tools_renderer: Optional[ToolsRenderer] = render_text_description_and_args,
        chat_history: Optional[bool] = True,
    ) -> AgentExecutor:
        # This is based on langchain.agents.create_structured_chat_agent but customized for Mistral
        prompt = self._create_prompt_template()
        missing_vars = {"tools", "tool_names", "chat_history", "agent_scratchpad", "input"}.difference(
            prompt.input_variables
        )
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")

        prompt = prompt.partial(
            tools=tools_renderer(list(tools)),
            tool_names=", ".join([t.name for t in tools]),
        )
        llm_with_stop = self._llm.bind(stop=["Observation"])

        memory = None
        if chat_history:
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history", k=5, return_messages=True, output_key="output"
            )

        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            )
            | prompt
            | llm_with_stop
            | MistralJsonOutputParser(self)
        )

        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            memory=memory,
        )
        
    def _create_system_prompt_template(self) -> str:
        return f"{BOS} {B_INST} {self._tool_config.get('system_prompt_template')} {E_INST}"

    def _create_user_prompt_template(self) -> str:
        # TODO Need to fix the config here.... **************************************************
        return f"{self._tool_config.get('user_prompt_context')}\n{B_INST} {self._tool_config.get('user_prompt')} {E_INST}\n{self._tool_config.get('user_prompt_scratch_pad')}"
    
    def _get_response_prefix(self) -> str:
        # TODO Need to fix the config here.... **************************************************
        return self._tool_config.get('user_prompt_starter_response').strip()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['tools', 'tool_names'],
                    template=self._create_system_prompt_template()
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['input', 'chat_history', 'agent_scratchpad'],
                    template=self._create_user_prompt_template()
                )
            ),
        ])
        return prompt
    

class MistralJsonOutputParser(JSONAgentOutputParser):
    def __init__(self, llm_wrapper: MistralInstruct):
        super().__init__()
        # Hack to avoid missing pydantic variable errors
        # TODO: change this to a pydantic PrivateAttr
        self.__dict__['_llm_wrapper'] = llm_wrapper

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            # Add the json prefix that we added at the end of our user prompt
            logger.debug(f"Parsing response: {text}")
            # response_prefix = self.__dict__['_llm_wrapper']._get_response_prefix().replace('{{','{')
            # text = text.strip()
            # if not text.startswith('```json') and self.__dict__['_llm_wrapper']:
            #     text = (response_prefix + text)

            response = parse_json_markdown(text)
            logger.debug(f"Parsed response: {response}")

            if isinstance(response, list):
                # gpt turbo frequently ignores the directive to emit a single action
                logger.warning("Got multiple action responses: %s", response)
                response = response[0]

            # Mistral frequently ignores action_input needing to be a string
            action_input = response.get("action_input", {})
            # if isinstance(action_input, dict):
            #     action_input = action_input.get('values', str(action_input))
            #     if isinstance(action_input, list) and len(action_input)>0:
            #         action_input = action_input[0]

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
