from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_community.vectorstores import Chroma


def load_documents(file_paths: list[str]) -> list[Document]:
    ...


def split_documents(docs: list[Document]) -> list[Document]:
    ...


def embed_and_store(chunks: list[Document]) -> Chroma:
    ...


def load_existing_store() -> Chroma:
    ...


def setup_retriever(vectorstore: Chroma) -> BaseRetriever:
    ...


def trim_history(messages: list[dict], max_tokens: int, max_turns: int) -> list[dict]:
    ...


def format_history(messages: list[dict]) -> str:
    ...


def create_chain(retriever: BaseRetriever) -> Runnable:
    ...


def ask_question(chain, query: str, history: list[dict]) -> dict:
    ...
