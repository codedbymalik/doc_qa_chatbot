from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import re
from config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    SEMANTIC_CHUNKER_BREAKPOINT,
)

try:
    from langchain_experimental.text_splitter import SemanticChunker
except ModuleNotFoundError:  # pragma: no cover - exercised in integration envs
    SemanticChunker = None


load_dotenv()


def _detect_chapter_heading(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "unknown"

    heading_pattern = re.compile(
        r"^(chapter|section|part)\s+[A-Za-z0-9.\-]+(?:[:\-\s].*)?$",
        re.IGNORECASE,
    )
    markdown_heading_pattern = re.compile(r"^#{1,6}\s+(.+)$")

    for line in lines[:12]:
        if heading_pattern.match(line):
            return line
        markdown_match = markdown_heading_pattern.match(line)
        if markdown_match:
            return markdown_match.group(1).strip()

    return "unknown"


def load_documents(file_paths: list[str]) -> list[Document]:
    all_documents: list[Document] = []

    for path in file_paths:
        if not os.path.exists(path) or not path.lower().endswith(".pdf"):
            raise ValueError(f"Invalid or non-PDF file: {path}")

        documents = PyPDFLoader(path).load()
        has_extractable_text = any((doc.page_content or "").strip() for doc in documents)
        if not has_extractable_text:
            raise ValueError(
                f"No extractable text found in uploaded PDF: {path}. "
                "This file may be scanned or image-only."
            )

        source_name = os.path.basename(path)
        for index, doc in enumerate(documents):
            metadata = dict(doc.metadata or {})
            metadata["source"] = source_name
            raw_page = metadata.get("page", index)
            try:
                metadata["page"] = int(raw_page)
            except (TypeError, ValueError):
                metadata["page"] = index
            metadata["chapter"] = _detect_chapter_heading(doc.page_content or "")
            doc.metadata = metadata
            all_documents.append(doc)

    return all_documents


def split_documents(docs: list[Document]) -> list[Document]:
    if SemanticChunker is None:
        raise ImportError(
            "Semantic chunking requires langchain-experimental. "
            "Install it with: pip install langchain-experimental"
        )

    splitter = SemanticChunker(
        OpenAIEmbeddings(model=EMBEDDING_MODEL),
        breakpoint_threshold_type=SEMANTIC_CHUNKER_BREAKPOINT,
    )
    chunks: list[Document] = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        chapter = doc.metadata.get("chapter", "unknown")
        if chapter == "unknown":
            chapter = _detect_chapter_heading(doc.page_content or "")

        base_doc = Document(
            page_content=doc.page_content,
            metadata={"source": source, "page": page, "chapter": chapter},
        )
        split_chunks = splitter.split_documents([base_doc])
        if not split_chunks and (doc.page_content or "").strip():
            split_chunks = [base_doc]

        current_chapter = chapter
        for chunk in split_chunks:
            metadata = dict(chunk.metadata or {})
            detected_chapter = _detect_chapter_heading(chunk.page_content or "")
            if detected_chapter != "unknown":
                current_chapter = detected_chapter

            metadata["source"] = source
            metadata["page"] = page
            metadata["chapter"] = current_chapter
            chunk.metadata = metadata
            chunks.append(chunk)

    for chunk in chunks:
        metadata = chunk.metadata or {}
        if "source" not in metadata or "page" not in metadata or "chapter" not in metadata:
            raise RuntimeError("Metadata lost during splitting")

    return chunks


def embed_and_store(chunks: list[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    persist_dir = CHROMA_PERSIST_DIR

    # If collection exists, check dimension compatibility
    if os.path.exists(persist_dir):
        try:
            existing = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            # Try a test embedding to catch dimension mismatch early
            existing.similarity_search("test", k=1)
        except Exception:
            # Dimension mismatch or corrupt store — wipe and recreate
            import shutil
            shutil.rmtree(persist_dir)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )


def load_existing_store() -> Chroma:
    if not os.path.isdir(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"No persisted vector store found at: {CHROMA_PERSIST_DIR}"
        )

    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
