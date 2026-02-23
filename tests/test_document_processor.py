import pytest
import tiktoken
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import document_processor


def test_load_documents_with_valid_pdf_mock():
    fake_pages = [
        Document(page_content="Chapter 1\nIntro text", metadata={"page": "0"}),
        Document(page_content="More details on page 2", metadata={"page": 1}),
    ]

    with patch("document_processor.os.path.exists", return_value=True), patch(
        "document_processor.PyPDFLoader"
    ) as loader_cls:
        loader_instance = MagicMock()
        loader_instance.load.return_value = fake_pages
        loader_cls.return_value = loader_instance

        docs = document_processor.load_documents(["/tmp/handbook.pdf"])

    assert len(docs) == 2
    assert docs[0].metadata["source"] == "handbook.pdf"
    assert docs[0].metadata["page"] == 0
    assert docs[1].metadata["source"] == "handbook.pdf"
    assert docs[1].metadata["page"] == 1


def test_load_documents_with_scanned_pdf_mock_raises_value_error():
    with patch("document_processor.os.path.exists", return_value=True), patch(
        "document_processor.PyPDFLoader"
    ) as loader_cls:
        loader_instance = MagicMock()
        loader_instance.load.return_value = [
            Document(page_content="   ", metadata={"page": 0}),
            Document(page_content="", metadata={"page": 1}),
        ]
        loader_cls.return_value = loader_instance

        with pytest.raises(ValueError, match="No extractable text found"):
            document_processor.load_documents(["/tmp/scanned.pdf"])


def test_split_documents_keeps_source_and_page_metadata_on_every_chunk(monkeypatch):
    docs = [
        Document(
            page_content="Chunking sample content",
            metadata={"source": "handbook.pdf", "page": 3, "chapter": "Chapter 2"},
        )
    ]

    class FakeSemanticChunker:
        def __init__(self, *_args, **_kwargs):
            pass

        def split_documents(self, _docs):
            return [
                Document(page_content="first split", metadata={}),
                Document(page_content="second split", metadata={}),
            ]

    monkeypatch.setattr(document_processor, "SemanticChunker", FakeSemanticChunker)
    monkeypatch.setattr(document_processor, "OpenAIEmbeddings", MagicMock())

    chunks = document_processor.split_documents(docs)

    assert len(chunks) == 2
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
        assert chunk.metadata["source"] == "handbook.pdf"
        assert chunk.metadata["page"] == 3


def test_split_documents_chunks_do_not_exceed_512_tokens(monkeypatch):
    docs = [
        Document(
            page_content="seed content",
            metadata={"source": "handbook.pdf", "page": 5, "chapter": "Chapter 5"},
        )
    ]

    class FakeSemanticChunker:
        def __init__(self, *_args, **_kwargs):
            pass

        def split_documents(self, _docs):
            return [
                Document(page_content=("alpha " * 350).strip(), metadata={}),
                Document(page_content=("beta " * 300).strip(), metadata={}),
            ]

    monkeypatch.setattr(document_processor, "SemanticChunker", FakeSemanticChunker)
    monkeypatch.setattr(document_processor, "OpenAIEmbeddings", MagicMock())

    chunks = document_processor.split_documents(docs)
    encoding = tiktoken.get_encoding("cl100k_base")

    assert chunks
    for chunk in chunks:
        token_count = len(encoding.encode(chunk.page_content))
        assert token_count <= 512


def test_embed_and_store_with_mocked_openai_embeddings_makes_no_real_api_calls():
    chunks = [
        Document(
            page_content="embedded text",
            metadata={"source": "handbook.pdf", "page": 0, "chapter": "Chapter 1"},
        )
    ]
    embedding_instance = MagicMock()
    store_instance = MagicMock()

    with patch(
        "document_processor.OpenAIEmbeddings", return_value=embedding_instance
    ) as embeddings_cls, patch(
        "document_processor.Chroma.from_documents", return_value=store_instance
    ) as from_documents:
        result = document_processor.embed_and_store(chunks)

    embeddings_cls.assert_called_once_with(model=document_processor.EMBEDDING_MODEL)
    from_documents.assert_called_once_with(
        chunks,
        embedding_instance,
        persist_directory=document_processor.CHROMA_PERSIST_DIR,
    )
    embedding_instance.embed_documents.assert_not_called()
    embedding_instance.embed_query.assert_not_called()
    assert result is store_instance
