from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, CHROMA_PERSIST_DIR


load_dotenv()


def load_documents(file_paths: list[str]) -> list[Document]:
    all_documents: list[Document] = []

    for path in file_paths:
        if not os.path.exists(path) or not path.lower().endswith(".pdf"):
            raise ValueError(f"Invalid or non-PDF file: {path}")

        documents = PyPDFLoader(path).load()
        has_extractable_text = any((doc.page_content or "").strip() for doc in documents)
        if not has_extractable_text:
            raise ValueError(f"No extractable text found in: {path}")

        source_name = os.path.basename(path)
        for index, doc in enumerate(documents):
            metadata = dict(doc.metadata or {})
            metadata["source"] = source_name
            raw_page = metadata.get("page", index)
            try:
                metadata["page"] = int(raw_page)
            except (TypeError, ValueError):
                metadata["page"] = index
            doc.metadata = metadata
            all_documents.append(doc)

    return all_documents


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        metadata = chunk.metadata or {}
        if "source" not in metadata or "page" not in metadata:
            raise RuntimeError("Metadata lost during splitting")

    return chunks


def embed_and_store(chunks: list[Document]) -> Chroma:
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(
        chunks,
        embedding,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def load_existing_store() -> Chroma | None:
    if os.path.isdir(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        )
    return None
