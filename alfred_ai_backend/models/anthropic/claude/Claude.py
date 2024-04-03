from typing import Sequence, Dict, List, Any, Optional
from langchain.agents import AgentExecutor
from langchain_core.tools import BaseTool
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.tools.render import ToolsRenderer, render_text_description_and_args
from langchain.memory import ConversationBufferWindowMemory
from alfred_ai_backend.core.tools.ToolConfig import ToolConfig
from alfred_ai_backend.models.Model import Model
from langchain_core.runnables import RunnableConfig
from langchain.agents.output_parsers import XMLAgentOutputParser
import logging

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
            from langchain_anthropic import ChatAnthropic
            #from langchain_anthropic.experimental import ChatAnthropicTools  #beta and is going away
        except ImportError:
            raise ImportError(
                "Could not import langchain_anthropic library. "
                "Please install the langchain_anthropic library to "
                "use this embedding model: pip install langchain-anthropic "
                #" and pip install -qU langchain-anthropic defusedxml"
            )
        self._chat_history = True
        self._llm = ChatAnthropic(**self._tool_config.get_init_config())
        # self._llm = ChatAnthropicTools(**self._tool_config.get_init_config())
    
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
        system_input_var_list.append("tools")

        if chat_history:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=system_input_var_list, template=self._tool_config.get('system_prompt_template'))),
                MessagesPlaceholder(variable_name='chat_history', optional=True),
                HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=user_input_variables, template=self._tool_config.get('user_prompt_template'))),
                MessagesPlaceholder(variable_name='agent_scratchpad')
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=system_input_var_list, template=self._tool_config.get('system_prompt_template'))),
                HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=user_input_variables, template=self._tool_config.get('user_prompt_template'))),
                MessagesPlaceholder(variable_name='agent_scratchpad')
            ])
        
        if system_input_variables:
            prompt = prompt.partial(**system_input_variables)

        #agent = create_openai_tools_agent(self._llm, tools, prompt)
        #llm_with_tools = self._llm.bind_tools(tools=tools)

        memory = None
        if chat_history:
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=5,
                return_messages=True,
                input_key="input",  # Since we can have mulitple inputs we need to specify which to use for history for some reason
                output_key="output",
            )

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: self._convert_intermediate_steps(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda _: memory.load_memory_variables({})["chat_history"]  # no idea if this will work...
            }
            | prompt.partial(tools=self._convert_tools(tools))
            | self._llm.bind(stop=["</tool_input>", "</final_answer>"])
            | XMLAgentOutputParser()
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
        response = self._agent_executor.invoke(input, inference_config, **kwargs)

        # logger.info(f"Total Tokens: {cb.total_tokens}")
        # logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
        # logger.info(f"Completion Tokens: {cb.completion_tokens}")
        # logger.info(f"Total Cost (USD): ${cb.total_cost}")
        return response

    # May need to use this if we have to add 
    # self.conversation_memory.chat_memory.add_ai_message("hello, I'm an AI") in the invoke method above
    # Source: https://www.reddit.com/r/LangChain/comments/1bq1rgj/how_to_implement_claude_based_agents/
    # def _convert_chat_history(self, chat_history) -> str:
    #     pass


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

    def _convert_tools(self, tools: Sequence[BaseTool]) -> str:
        """Logic for converting tools to string to go in prompt

        Source: https://python.langchain.com/docs/expression_language/cookbook/agent

        Args:
            tools (Sequence[BaseTool]): Tools to be used

        Returns:
            str: string conversion
        """
        return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

