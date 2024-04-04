from typing import Sequence, Dict, List, Any, Optional
from langchain.agents import AgentExecutor
from langchain_core.tools import BaseTool
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools.render import ToolsRenderer, render_text_description_and_args
from langchain.memory import ConversationBufferWindowMemory
from alfred_ai_backend.core.tools.ToolConfig import ToolConfig
from alfred_ai_backend.models.Model import Model
from langchain_core.runnables import RunnableConfig
import logging
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from alfred_ai_backend.models.anthropic.claude.ClaudeAgentOutputParser import ClaudeAgentOutputParser
# from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.runnables import Runnable, RunnablePassthrough

logger = logging.getLogger(__name__)

"""
NOTES:
Provider > Anthropic: https://python.langchain.com/docs/integrations/platforms/anthropic
Chat models > Anthropic: https://python.langchain.com/docs/integrations/chat/anthropic
Chat models > Anthropic Tools: https://python.langchain.com/docs/integrations/chat/anthropic_functions
Templates > extraction-anthropic-functions: https://python.langchain.com/docs/templates/extraction-anthropic-functions

This has a JS working example
JS: Langchain Expression Language > Cookbook > Agents: https://js.langchain.com/docs/expression_language/cookbook/agents
*** Python equivalent: https://python.langchain.com/docs/expression_language/cookbook/agent

Anthropic Function calling spec: https://docs.anthropic.com/claude/docs/functions-external-tools

Medium article implentation example: https://medium.com/@daniellefranca96/running-a-langchain-agent-on-bedrock-claude-using-the-model-function-calling-5f400a8f0d62
"""

class Claude(Model):
    def __init__(self, tool_config: Optional[ToolConfig] = None):
        super().__init__(tool_config)
        
        try:
            #from langchain_anthropic import ChatAnthropic
            from langchain_anthropic.experimental import ChatAnthropicTools  #beta and is going away
        except ImportError:
            raise ImportError(
                "Could not import langchain_anthropic library. "
                "Please install the langchain_anthropic library to "
                "use this embedding model: pip install langchain-anthropic "
                #" and pip install -qU langchain-anthropic defusedxml"
            )
        self._chat_history = True
        # self._llm = ChatAnthropic(**self._tool_config.get_init_config())
        self._llm = ChatAnthropicTools(**self._tool_config.get_init_config())
    
    def initialize_agent(
        self,
        user_input_variables: List[str],
        system_input_variables: Dict[str,str],
        tools: Sequence[BaseTool],
        tools_renderer: Optional[ToolsRenderer] = render_text_description_and_args,
        chat_history: Optional[bool] = True,
    ) -> AgentExecutor:
        self._chat_history = chat_history
        system_input_var_list = list(system_input_variables.keys()) if system_input_variables else []
        user_input_variables.append("agent_scratchpad")
        if chat_history:
            user_input_variables.append("chat_history")

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=system_input_var_list, template=self._tool_config.get('system_prompt_template'))),
            HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=user_input_variables, template=self._tool_config.get('user_prompt_template'))),
        ])

        # if system_input_variables:
        #     prompt = prompt.partial(**system_input_variables)

        memory = None
        if chat_history:
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=5,
                return_messages=True,
                input_key="input",
                output_key="output",
            )
        
        if system_input_variables:
            prompt = prompt.partial(**system_input_variables)

        llm_with_tools = self._llm.bind_tools(tools=tools, stop=["</tool_input>", "</final_answer>"])

        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: self._convert_intermediate_steps(
                    x["intermediate_steps"]
                ),
                #chat_history=lambda _: memory.load_memory_variables({})["chat_history"]
            )
            | prompt
            | llm_with_tools
            | ClaudeAgentOutputParser()
        )
        
        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            memory=memory,
        )
        return self._agent_executor


    def invoke_agent_executor(self, input: Dict[str, Any], inference_config: Optional[RunnableConfig] = None, **kwargs: Any  ) -> Dict[str, Any]:
        # TODO add token counting for Anthropic
        # with get_openai_callback() as cb:
        print("*** Invoking with the following input: ", input)
        response = self._agent_executor.invoke(input, inference_config, **kwargs)

        # logger.info(f"Total Tokens: {cb.total_tokens}")
        # logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
        # logger.info(f"Completion Tokens: {cb.completion_tokens}")
        # logger.info(f"Total Cost (USD): ${cb.total_cost}")
        return response


    def _convert_intermediate_steps(self, intermediate_steps: Any) -> str:
        """Logic for going from intermediate steps to a string to pass into model
        This is pretty tied to the prompt

        Source: https://python.langchain.com/docs/expression_language/cookbook/agent

        Args:
            intermediate_steps (Any): intermediate steps

        Returns:
            str: string conversion
        """
        log = ""
        for action, observation in intermediate_steps:
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{observation}</observation>"
            )
        return log
