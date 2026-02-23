# Document Q&A Chatbot (RAG)

## Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot for PDF documents. You upload PDFs, the app chunks and embeds the content into a vector database, retrieves the most relevant chunks for each question, reranks them, and then asks the LLM to answer using that retrieved context.

## Prerequisites
- Python 3.10+
- OpenAI API key (for embeddings and chat)
- Cohere API key (for reranking)

## Installation
```bash
git clone <repo-url>
cd doc_qa_chatbot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt

cp .env.example .env
```

Then edit `.env` and provide:
- `OPENAI_API_KEY`
- `COHERE_API_KEY`
- Optional overrides such as `CHROMA_PERSIST_DIR`, `MAX_HISTORY_TURNS`, and `MAX_HISTORY_TOKENS`

## Run
```bash
streamlit run main.py
```

## Run Tests
```bash
pytest tests/ -v
```

## Troubleshooting
- Missing API key / authentication error: verify `OPENAI_API_KEY` and `COHERE_API_KEY` are set correctly in `.env`.
- Scanned PDF with no extractable text: scanned/image-only PDFs need OCR first; convert with an OCR tool before upload.
- ChromaDB permission error: if persistence fails, check write permissions for `CHROMA_PERSIST_DIR`.
- Stale vector store after re-upload: remove the persisted Chroma directory and reprocess documents.
- **Embedding model dimension mismatch**: If you switch embedding models (e.g., from `text-embedding-3-small` to `text-embedding-3-large`), you must delete the `chroma_store/` directory and re-upload your documents. Collections are not compatible across models with different output dimensions. The app will now auto-detect and fix this.
- Chatbot forgets recent context: increase `MAX_HISTORY_TOKENS` cautiously; larger history increases token usage and per-query API cost.
- Slow ingestion: expected when using `text-embedding-3-large` plus semantic chunking, especially on large PDFs.

## Architecture (Text Diagram)
```text
PDF Upload
   -> Semantic Chunker
      -> OpenAI Embedder
         -> ChromaDB Vector Store
            -> Retriever
               -> Cohere Reranker
                  -> LLM
                     -> Answer + Sources
```
