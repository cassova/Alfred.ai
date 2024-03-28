import yaml
from typing import Any
import logging
from alfred_ai_backend.core.Config import SingletonMeta
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, VectorStoreRetriever, VectorStore
from langchain_openai import OpenAIEmbeddings
import asyncio

logger = logging.getLogger(__name__)
CONFIG_FILE_NAME = "config.yml"

class CodeDatabase(metaclass=SingletonMeta):
    """Holds the Code Database"""

    def __init__(self):
        self._lock = asyncio.Lock()  # Initialize the lock
        self._db: VectorStore = None
        self._retriever: VectorStoreRetriever = None
        pass

    def get_retriever(self):
        if self._retriever:
            return self._retriever
        
        raise Exception("Code database needs to be refreshed")

    def refresh_database(self, repo_path: str):
        with self._lock:
            # Load the code documents
            loader = GenericLoader.from_filesystem(
                repo_path,
                glob="**/*",
                suffixes=[".py", ".yml"],
                exclude=["**/non-utf8-encoding.py"],
                parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
            )
            documents = loader.load()

            # Split into chunks
            python_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
            )
            texts = python_splitter.split_documents(documents)

            # Build the vector db
            self._db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
            self._retriever = self._db.as_retriever(
                search_type="mmr",  # Also test "similarity"
                search_kwargs={"k": 8},
            )

