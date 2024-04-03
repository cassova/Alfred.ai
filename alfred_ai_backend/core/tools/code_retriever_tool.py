from langchain.tools.retriever import create_retriever_tool
from alfred_ai_backend.core.CodeDatabase import CodeDatabase
from langchain.tools import Tool

def create_code_retriever_tool() -> Tool:
    """Creates a tool that will be used 

    Returns:
        Tool: Code retriever tool
    """

    # TODO: this is going to be broken: the retiever is empty at this stage
    # need to figure out the chicken-egg problem of having a tool defined at
    # initiation where the tool isn't ready because it hasn't initlaized the db
    

    code_database = CodeDatabase()
    retriever = code_database.get_retriever()

    tool = create_retriever_tool(
        retriever,
        "SearchCodeRepo",
        "Searches and returns parts of the code repository.",
    )
    return tool
