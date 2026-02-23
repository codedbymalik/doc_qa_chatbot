from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableMap
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import tiktoken
from config import (
    LLM_MODEL,
    RERANKER_TOP_K,
    RETRIEVER_TOP_K,
    SIMILARITY_THRESHOLD,
    MAX_HISTORY_TURNS,
    MAX_HISTORY_TOKENS,
)

try:
    from langchain_cohere import CohereRerank
except ModuleNotFoundError:  # pragma: no cover - exercised in integration envs
    CohereRerank = None


load_dotenv()


class CohereRerankingRetriever(BaseRetriever):
    vectorstore: Chroma
    reranker: object

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=RERANKER_TOP_K,
        )

        filtered_docs: list[Document] = []
        for doc, score in docs_and_scores:
            if score >= SIMILARITY_THRESHOLD:
                metadata = dict(doc.metadata or {})
                metadata["similarity_score"] = float(score)
                doc.metadata = metadata
                filtered_docs.append(doc)

        if not filtered_docs:
            return []

        reranked_docs = self.reranker.compress_documents(
            documents=filtered_docs,
            query=query,
        )
        return list(reranked_docs)[:RETRIEVER_TOP_K]


def setup_retriever(vectorstore: Chroma) -> BaseRetriever:
    if CohereRerank is None:
        raise ImportError(
            "Cohere reranking requires langchain-cohere and cohere. "
            "Install them with: pip install langchain-cohere cohere"
        )

    reranker = CohereRerank(
        model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
        top_n=RETRIEVER_TOP_K,
    )
    return CohereRerankingRetriever(vectorstore=vectorstore, reranker=reranker)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer questions using only the provided context. If the answer cannot be found in the context, respond with: 'No relevant information found in the uploaded documents.'",
            ),
            (
                "human",
                "Chat History:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )


def trim_history(messages: list[dict], max_tokens: int, max_turns: int) -> list[dict]:
    if not messages or max_tokens <= 0 or max_turns <= 0:
        return []

    normalized_messages = [
        {"role": msg.get("role"), "content": str(msg.get("content", ""))}
        for msg in messages
        if msg.get("role") in {"user", "assistant"}
    ]
    if not normalized_messages:
        return []

    recent_messages = normalized_messages[-(max_turns * 2) :]

    paired_messages: list[dict] = []
    index = 0
    while index + 1 < len(recent_messages):
        first = recent_messages[index]
        second = recent_messages[index + 1]
        if first["role"] == "user" and second["role"] == "assistant":
            paired_messages.extend([first, second])
            index += 2
            continue
        index += 1

    if not paired_messages:
        return []

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def total_tokens(history_messages: list[dict]) -> int:
        return sum(
            len(
                encoding.encode(
                    f"{msg.get('role', '')}: {msg.get('content', '')}"
                )
            )
            for msg in history_messages
        )

    while paired_messages and total_tokens(paired_messages) > max_tokens:
        paired_messages = paired_messages[2:]

    return paired_messages


def format_history(messages: list[dict]) -> str:
    if not messages:
        return ""

    lines = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = str(role).capitalize() if role else "Unknown"
        lines.append(f"{prefix}: {content}")

    return "\n".join(lines)


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def create_chain(retriever: BaseRetriever) -> Runnable:
    answer_chain = build_prompt() | ChatOpenAI(model=LLM_MODEL) | StrOutputParser()

    chain = (
        RunnableMap(
            {
                "question": lambda data: data["question"],
                "chat_history": lambda data: data.get("chat_history", ""),
                "source_documents": lambda data: retriever.invoke(data["question"]),
            }
        )
        | RunnableMap(
            {
                "context": lambda data: format_docs(data["source_documents"]),
                "question": lambda data: data["question"],
                "chat_history": lambda data: data["chat_history"],
                "source_documents": lambda data: data["source_documents"],
            }
        )
        | RunnableMap(
            {
                "answer": answer_chain,
                "source_documents": lambda data: data["source_documents"],
            }
        )
    )
    return chain


def ask_question(chain, query: str, history: list[dict]) -> dict:
    trimmed_history = trim_history(history, MAX_HISTORY_TOKENS, MAX_HISTORY_TURNS)
    formatted_history = format_history(trimmed_history)

    result = chain.invoke({"question": query, "chat_history": formatted_history})
    source_documents = result.get("source_documents") or []
    answer = result.get("answer", "")

    if not source_documents:
        answer = "No relevant information found in the uploaded documents."

    return {
        "answer": answer,
        "sources": [
            {
                "file": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "excerpt": doc.page_content[:200],
            }
            for doc in source_documents
        ],
    }
