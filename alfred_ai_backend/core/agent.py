from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
import logging
# from alfred_ai_backend.core.output_parser import OutputParser
from langchain_core.messages import AIMessage
from alfred_ai_backend.models.llm import LlmWrapper
import json
from langchain.schema import AgentAction, AgentFinish


# TODO: logger = logging.getLogger(__name__)

# Source: https://github.com/pinecone-io/examples/blob/master/learn/generation/llm-field-guide/llama-2/llama-2-70b-chat-agent.ipynb
class AgentWrapper():
    def __init__(self, llm_wrapper: LlmWrapper):
        # TODO: remove most of these `self` things since we don't need them
        self._llm_wrapper = llm_wrapper
        self._llm = self._llm_wrapper.get_llm()

        self._memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=5, return_messages=True, output_key="output"
        )
        self._tools = load_tools(["llm-math"], llm=self._llm)
            
        # initialize agent
        self._agent = self._llm_wrapper.create_agent(
            tools=self._tools,
            prompt=self._load_prompt_from_config(),
        )

        # initialize executor
        self._agent_executor = AgentExecutor(
            agent=self._agent,
            tools=self._tools,
            verbose=True,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            memory=self._memory,
            #agent_kwargs={"output_parser": output_parser}
        )

    def _load_prompt_from_config(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._llm_wrapper.create_system_prompt_template()),
            ("human", self._llm_wrapper.create_user_prompt_template()),
        ])
        return prompt

    def start_task(self, user_input_str: str):
        return self._agent_executor.invoke({"input": user_input_str}, return_only_outputs=False, stop=['```'])