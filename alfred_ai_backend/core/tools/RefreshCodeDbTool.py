from typing import Optional, Type
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from alfred_ai_backend.core.CodeDatabase import CodeDatabase
from alfred_ai_backend.core.Config import Config
from pathlib import Path
import asyncio

root_config = Config()
code_database = CodeDatabase()

class RefreshCodeDbSchema(BaseModel):
    pkg_name: str = Field(description="The python package name (i.e. subfolder) where all code should reside")

class RefreshCodeDbTool(BaseTool):
    name: str = "RefreshCodeDb"
    description: str = "Useful for refreshing code database which is your understanding of a local python package. Use this once to "
    args_schema: Type[BaseModel] = RefreshCodeDbSchema

    def _run(
        self,
        pkg_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        repo_path = Path(root_config.get('root_folder'), pkg_name)
        code_database.refresh_database(repo_path)
        return "Update successful"
    
    async def _arun(
        self,
        pkg_name: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        repo_path = Path(root_config.get('root_folder'), pkg_name)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, code_database.refresh_database, repo_path=repo_path)
        return "Update successful"
