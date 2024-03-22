
class CoderAgent():


    def create_coder_agent_executor(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['cwd'], template=self.model_config.get('coder_system_prompt'))),
            HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template=self.model_config.get('coder_user_prompt'))),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])
        tools = load_tools(
            ["llm-math", "terminal"],
            llm=self.get_llm(),
            allow_dangerous_tools=True
        )
        file_toolkit = FileManagementToolkit(
            root_dir=str(self.config.get('root_folder'))
        )
        tools += file_toolkit.get_tools()
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            early_stopping_method="generate",
        )
        return agent_executor