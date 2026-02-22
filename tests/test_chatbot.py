import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

import chatbot


def test_ask_question_returns_answer_and_sources_keys():
    chain = MagicMock()
    chain.invoke.return_value = {
        "answer": "This is an answer from the chain.",
        "source_documents": [
            Document(
                page_content="Relevant context for the answer.",
                metadata={"source": "manual.pdf", "page": 2},
            )
        ],
    }

    result = chatbot.ask_question(chain, "What does the manual say?")

    assert set(result.keys()) == {"answer", "sources"}
    assert result["answer"] == "This is an answer from the chain."
    assert isinstance(result["sources"], list)


def test_ask_question_sources_include_file_page_and_excerpt():
    long_text = "x" * 350
    chain = MagicMock()
    chain.invoke.return_value = {
        "answer": "Answer text",
        "source_documents": [
            Document(page_content=long_text, metadata={"source": "guide.pdf", "page": 7})
        ],
    }

    result = chatbot.ask_question(chain, "Explain the section.")
    source = result["sources"][0]

    assert set(source.keys()) == {"file", "page", "excerpt"}
    assert source["file"] == "guide.pdf"
    assert source["page"] == 7
    assert source["excerpt"] == long_text[:200]
    assert len(source["excerpt"]) <= 200


def test_ask_question_returns_default_answer_when_no_sources():
    chain = MagicMock()
    chain.invoke.return_value = {
        "answer": "LLM hallucinated answer",
        "source_documents": [],
    }

    result = chatbot.ask_question(chain, "Unrelated question")

    assert result["answer"] == "No relevant information found in the uploaded documents."
    assert result["sources"] == []


def test_build_prompt_returns_chat_prompt_template():
    prompt = chatbot.build_prompt()
    assert isinstance(prompt, ChatPromptTemplate)


def test_build_prompt_contains_only_the_provided_context_instruction():
    prompt = chatbot.build_prompt()
    system_message = prompt.messages[0].prompt.template.lower()
    assert "only the provided context" in system_message


def test_format_docs_joins_content_with_double_newlines():
    docs = [
        Document(page_content="First page", metadata={}),
        Document(page_content="Second page", metadata={}),
    ]

    result = chatbot.format_docs(docs)

    assert result == "First page\n\nSecond page"


def test_format_docs_returns_empty_string_for_empty_input():
    assert chatbot.format_docs([]) == ""
