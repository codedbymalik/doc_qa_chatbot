from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import chatbot
import document_processor


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


def test_end_to_end_pipeline_with_mocked_openai_and_chromadb(monkeypatch):
    raw_pages = [
        Document(page_content="Chapter 1\nRAG retrieves context before answering.", metadata={"page": 0})
    ]

    monkeypatch.setattr(document_processor.os.path, "exists", lambda _path: True)

    loader_cls = MagicMock()
    loader_instance = MagicMock()
    loader_instance.load.return_value = raw_pages
    loader_cls.return_value = loader_instance
    monkeypatch.setattr(document_processor, "PyPDFLoader", loader_cls)

    class FakeSemanticChunker:
        def __init__(self, *_args, **_kwargs):
            pass

        def split_documents(self, _docs):
            return [
                Document(
                    page_content="RAG retrieves context before answering.",
                    metadata={},
                )
            ]

    monkeypatch.setattr(document_processor, "SemanticChunker", FakeSemanticChunker)
    monkeypatch.setattr(document_processor, "OpenAIEmbeddings", MagicMock())

    fake_vectorstore = MagicMock()
    source_doc = Document(
        page_content="RAG retrieves context before answering.",
        metadata={"source": "sample.pdf", "page": 0},
    )
    fake_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (source_doc, chatbot.SIMILARITY_THRESHOLD + 0.2)
    ]

    with patch("document_processor.Chroma.from_documents", return_value=fake_vectorstore):
        docs = document_processor.load_documents(["/tmp/sample.pdf"])
        chunks = document_processor.split_documents(docs)
        vectorstore = document_processor.embed_and_store(chunks)

    reranker = MagicMock()
    reranker.compress_documents.side_effect = lambda documents, query: documents
    retriever = _construct_retriever(vectorstore=vectorstore, reranker=reranker)

    fake_llm = RunnableLambda(lambda _prompt: "Mocked final answer")
    fake_parser = RunnableLambda(lambda text: text)
    monkeypatch.setattr(chatbot, "ChatOpenAI", lambda model: fake_llm)
    monkeypatch.setattr(chatbot, "StrOutputParser", lambda: fake_parser)

    chain = chatbot.create_chain(retriever)
    result = chatbot.ask_question(chain, "How does this system work?", history=[])

    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "Mocked final answer"
    assert result["sources"]
    assert result["sources"][0]["file"] == "sample.pdf"
    assert result["sources"][0]["page"] == 0


def test_two_turn_history_from_turn_one_is_passed_to_turn_two():
    chain = MagicMock()
    chain.invoke.side_effect = [
        {
            "answer": "Turn 1 answer",
            "source_documents": [
                Document(page_content="Turn one source", metadata={"source": "a.pdf", "page": 1})
            ],
        },
        {
            "answer": "Turn 2 answer",
            "source_documents": [
                Document(page_content="Turn two source", metadata={"source": "a.pdf", "page": 2})
            ],
        },
    ]

    turn_one_result = chatbot.ask_question(chain, "Question one?", history=[])
    turn_two_history = [
        {"role": "user", "content": "Question one?"},
        {"role": "assistant", "content": turn_one_result["answer"]},
    ]
    chatbot.ask_question(chain, "Question two?", history=turn_two_history)

    first_payload = chain.invoke.call_args_list[0].args[0]
    second_payload = chain.invoke.call_args_list[1].args[0]

    assert first_payload["chat_history"] == ""
    assert second_payload["question"] == "Question two?"
    assert "User: Question one?" in second_payload["chat_history"]
    assert "Assistant: Turn 1 answer" in second_payload["chat_history"]
