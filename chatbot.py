from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from config import LLM_MODEL, RETRIEVER_K, SIMILARITY_THRESHOLD


load_dotenv()


def setup_retriever(vectorstore: Chroma) -> "VectorStoreRetriever":
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVER_K, "score_threshold": SIMILARITY_THRESHOLD},
    )


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer questions using only the provided context. If the answer cannot be found in the context, respond with: 'No relevant information found in the uploaded documents.'",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def create_chain(retriever) -> "Runnable":
    answer_chain = build_prompt() | ChatOpenAI(model=LLM_MODEL) | StrOutputParser()

    chain = (
        RunnableMap(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "source_documents": retriever,
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


def ask_question(chain, query: str) -> dict:
    result = chain.invoke(query)
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
