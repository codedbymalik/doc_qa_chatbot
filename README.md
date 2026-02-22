## 1. Overview
This app is a document question-answering chatbot built with Retrieval-Augmented Generation (RAG). In plain English, RAG means the model first looks up relevant content from your uploaded PDFs, then uses that retrieved context to generate answers grounded in your documents instead of relying only on general model memory.

## 2. Prerequisites
- Python 3.10 or higher
- An OpenAI API key with access to `text-embedding-3-small` and `gpt-3.5-turbo`



## 3. Running the app
```bash
streamlit run main.py
```

## 4. Running tests
```bash
pytest tests/ -v
```

## 5. How it works
1. Upload: one or more PDF files are selected for processing.
2. Chunk: extracted text is split into overlapping chunks for retrieval quality.
3. Embed: each chunk is converted into a vector embedding.
4. Store: embeddings and metadata are saved in ChromaDB.
5. Retrieve: for each question, the most relevant chunks are fetched from the vector store.
6. Generate: the LLM answers using retrieved context only.

## 6. Troubleshooting
- `AuthenticationError`: This usually means the API key in `.env` is missing, invalid, or typoed. Verify `OPENAI_API_KEY` and restart the app.
- `No extractable text found`: The PDF is likely a scanned image rather than selectable text. Run OCR first (for example with `pymupdf` workflows or Adobe Acrobat), then re-upload.
- ChromaDB `OSError: permission denied` on `chroma_store`: Check filesystem permissions on the store directory, or set a writable `CHROMA_PERSIST_DIR` in `config.py`.
- `No relevant information found in the uploaded documents.`: The question may not match uploaded content, or retrieval may be too strict. Try lowering `SIMILARITY_THRESHOLD` in `config.py`.
- Stale vector store behavior after re-uploading: delete the `chroma_store/` directory and reprocess documents to rebuild a clean index.
