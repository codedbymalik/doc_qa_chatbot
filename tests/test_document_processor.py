import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import document_processor
from config import CHROMA_PERSIST_DIR, CHUNK_OVERLAP, CHUNK_SIZE


def test_load_documents_raises_for_missing_path():
    with patch("document_processor.os.path.exists", return_value=False):
        with pytest.raises(ValueError, match="Invalid or non-PDF file"):
            document_processor.load_documents(["/tmp/missing.pdf"])


def test_load_documents_raises_for_non_pdf_file():
    with patch("document_processor.os.path.exists", return_value=True):
        with pytest.raises(ValueError, match="Invalid or non-PDF file"):
            document_processor.load_documents(["/tmp/not_a_pdf.txt"])


def test_load_documents_raises_for_no_extractable_text():
    with patch("document_processor.os.path.exists", return_value=True), patch(
        "document_processor.PyPDFLoader"
    ) as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.load.return_value = [Document(page_content="   ", metadata={"page": 0})]
        mock_loader_cls.return_value = mock_loader

        with pytest.raises(ValueError, match="No extractable text found"):
            document_processor.load_documents(["/tmp/blank.pdf"])


def test_load_documents_returns_documents_with_source_and_page_metadata():
    fake_documents = [
        Document(page_content="First page text", metadata={"page": "2"}),
        Document(page_content="Second page text", metadata={"page": "not-an-int"}),
    ]

    with patch("document_processor.os.path.exists", return_value=True), patch(
        "document_processor.PyPDFLoader"
    ) as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.load.return_value = fake_documents
        mock_loader_cls.return_value = mock_loader

        result = document_processor.load_documents(["/data/reports/sample.pdf"])

    assert len(result) == 2
    assert result[0].metadata["source"] == "sample.pdf"
    assert result[0].metadata["page"] == 2
    assert result[1].metadata["source"] == "sample.pdf"
    assert result[1].metadata["page"] == 1


def test_split_documents_preserves_source_and_page_metadata():
    docs = [
        Document(
            page_content=("lorem ipsum " * 500).strip(),
            metadata={"source": "sample.pdf", "page": 3},
        )
    ]

    chunks = document_processor.split_documents(docs)

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["source"] == "sample.pdf"
        assert chunk.metadata["page"] == 3


def test_split_documents_chunk_size_does_not_exceed_limit():
    docs = [
        Document(
            page_content=("alpha beta gamma " * 700).strip(),
            metadata={"source": "sample.pdf", "page": 1},
        )
    ]

    chunks = document_processor.split_documents(docs)

    assert chunks
    for chunk in chunks:
        assert len(chunk.page_content) <= CHUNK_SIZE + CHUNK_OVERLAP


def test_split_documents_raises_runtime_error_when_metadata_is_missing(monkeypatch):
    def fake_split_documents(_self, _docs):
        return [Document(page_content="chunk without metadata", metadata={})]

    monkeypatch.setattr(
        document_processor.RecursiveCharacterTextSplitter,
        "split_documents",
        fake_split_documents,
    )

    with pytest.raises(RuntimeError, match="Metadata lost during splitting"):
        document_processor.split_documents(
            [Document(page_content="text", metadata={"source": "file.pdf", "page": 0})]
        )


def test_embed_and_store_uses_chroma_from_documents_with_persist_directory():
    chunks = [Document(page_content="chunk text", metadata={"source": "doc.pdf", "page": 0})]
    mock_store = MagicMock()

    with patch("document_processor.OpenAIEmbeddings") as mock_embeddings_cls, patch(
        "document_processor.Chroma.from_documents", return_value=mock_store
    ) as mock_from_documents:
        result = document_processor.embed_and_store(chunks)

    mock_embeddings_cls.assert_called_once_with(model=document_processor.EMBEDDING_MODEL)
    mock_from_documents.assert_called_once()

    args, kwargs = mock_from_documents.call_args
    assert args[0] == chunks
    assert args[1] == mock_embeddings_cls.return_value
    assert kwargs["persist_directory"] == CHROMA_PERSIST_DIR

    mock_store.persist.assert_not_called()
    assert result is mock_store


def test_load_existing_store_returns_none_when_directory_is_missing():
    with patch("document_processor.os.path.exists", return_value=False), patch(
        "document_processor.os.path.isdir", return_value=False
    ), patch("document_processor.os.listdir") as mock_listdir, patch(
        "document_processor.OpenAIEmbeddings"
    ) as mock_embeddings_cls, patch("document_processor.Chroma") as mock_chroma_cls:
        result = document_processor.load_existing_store()

    assert result is None
    mock_listdir.assert_not_called()
    mock_embeddings_cls.assert_not_called()
    mock_chroma_cls.assert_not_called()


def test_load_existing_store_returns_chroma_instance_when_directory_has_data():
    with patch("document_processor.os.path.exists", return_value=True), patch(
        "document_processor.os.path.isdir", return_value=True
    ), patch("document_processor.os.listdir", return_value=["chroma.sqlite3"]), patch(
        "document_processor.OpenAIEmbeddings"
    ) as mock_embeddings_cls, patch(
        "document_processor.Chroma.__init__", return_value=None
    ) as mock_chroma_init:
        result = document_processor.load_existing_store()

    assert isinstance(result, document_processor.Chroma)
    mock_embeddings_cls.assert_called_once_with(model=document_processor.EMBEDDING_MODEL)
    mock_chroma_init.assert_called_once()

    _, kwargs = mock_chroma_init.call_args
    assert kwargs["persist_directory"] == CHROMA_PERSIST_DIR
    assert kwargs["embedding_function"] == mock_embeddings_cls.return_value
