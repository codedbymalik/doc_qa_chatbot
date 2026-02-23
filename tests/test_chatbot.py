from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import chatbot


def _construct_retriever(vectorstore, reranker):
    if hasattr(chatbot.CohereRerankingRetriever, "model_construct"):
        return chatbot.CohereRerankingRetriever.model_construct(
            vectorstore=vectorstore,
            reranker=reranker,
        )
    return chatbot.CohereRerankingRetriever.construct(
        vectorstore=vectorstore,
        reranker=reranker,
    )


def test_trim_history_empty_input_returns_empty_list():
    assert chatbot.trim_history([], max_tokens=500, max_turns=4) == []


def test_trim_history_enforces_turn_limit():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]

    trimmed = chatbot.trim_history(history, max_tokens=10_000, max_turns=2)

    assert trimmed == history[-4:]
    assert len(trimmed) == 4


def test_trim_history_enforces_token_budget():
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    token_map = {
        "u1": 300,
        "a1": 250,
        "u2": 100,
        "a2": 100,
        "u3": 80,
        "a3": 80,
    }

    def fake_encode(text):
        total = sum(token_map[k] for k in token_map if k in text)
        return [0] * max(total, 1)

    fake_encoding = MagicMock()
    fake_encoding.encode.side_effect = fake_encode

    with patch("chatbot.tiktoken.encoding_for_model", return_value=fake_encoding):
        trimmed = chatbot.trim_history(history, max_tokens=400, max_turns=10)

    assert trimmed == history[2:]


def test_trim_history_never_orphans_user_assistant_pairs():
    history = [
        {"role": "assistant", "content": "orphan_assistant"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "orphan_user"},
    ]

    trimmed = chatbot.trim_history(history, max_tokens=10_000, max_turns=10)

    assert len(trimmed) % 2 == 0
    assert trimmed == history[1:5]


def test_trim_history_passthrough_under_turn_and_token_limits():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "how are you"},
        {"role": "assistant", "content": "good"},
    ]

    fake_encoding = MagicMock()
    fake_encoding.encode.return_value = [0] * 3

    with patch("chatbot.tiktoken.encoding_for_model", return_value=fake_encoding):
        trimmed = chatbot.trim_history(history, max_tokens=1_000, max_turns=10)

    assert trimmed == history


def test_format_history_empty_input_returns_empty_string():
    assert chatbot.format_history([]) == ""


def test_format_history_contains_labels():
    history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "It combines retrieval with generation."},
    ]

    formatted = chatbot.format_history(history)

    assert "User:" in formatted
    assert "Assistant:" in formatted


def test_format_history_preserves_order():
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]

    formatted = chatbot.format_history(history)

    first_q = formatted.find("first question")
    first_a = formatted.find("first answer")
    second_q = formatted.find("second question")
    second_a = formatted.find("second answer")

    assert -1 not in (first_q, first_a, second_q, second_a)
    assert first_q < first_a < second_q < second_a


def test_ask_question_returns_answer_and_sources_keys_with_source_page_metadata():
    chain = MagicMock()
    chain.invoke.return_value = {
        "answer": "Mocked answer.",
        "source_documents": [
            Document(page_content="Context excerpt", metadata={"source": "guide.pdf", "page": 4})
        ],
    }

    result = chatbot.ask_question(chain, "Explain it.", history=[])

    assert set(result.keys()) == {"answer", "sources"}
    assert isinstance(result["sources"], list)
    assert result["sources"]

    source_item = result["sources"][0]
    source_value = source_item.get("source", source_item.get("file"))
    assert source_value == "guide.pdf"
    assert source_item["page"] == 4


def test_similarity_threshold_filtering_excludes_low_score_from_llm_context():
    low_doc = Document(page_content="LOW_SCORE_CONTENT", metadata={"source": "a.pdf", "page": 1})
    high_doc = Document(page_content="HIGH_SCORE_CONTENT", metadata={"source": "b.pdf", "page": 2})

    vectorstore = MagicMock()
    vectorstore.similarity_search_with_relevance_scores.return_value = [
        (low_doc, chatbot.SIMILARITY_THRESHOLD - 0.1),
        (high_doc, chatbot.SIMILARITY_THRESHOLD + 0.1),
    ]

    reranker = MagicMock()
    reranker.compress_documents.side_effect = lambda documents, query: documents

    retriever = _construct_retriever(vectorstore=vectorstore, reranker=reranker)

    filtered_docs = retriever._get_relevant_documents(
        "Which chunk is relevant?",
        run_manager=MagicMock(),
    )
    llm_context = chatbot.format_docs(filtered_docs)

    assert high_doc in filtered_docs
    assert low_doc not in filtered_docs
    assert "HIGH_SCORE_CONTENT" in llm_context
    assert "LOW_SCORE_CONTENT" not in llm_context


def test_ask_question_returns_refusal_when_retriever_is_empty():
    chain = MagicMock()
    chain.invoke.return_value = {
        "answer": "This should be ignored",
        "source_documents": [],
    }

    result = chatbot.ask_question(chain, "Unrelated question", history=[])

    assert result["answer"] == "No relevant information found in the uploaded documents."
    assert result["sources"] == []
