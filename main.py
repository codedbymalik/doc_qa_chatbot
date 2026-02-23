import os
import shutil
import tempfile

import openai
import streamlit as st
from dotenv import load_dotenv

from chatbot import ask_question, create_chain, setup_retriever
from config import MAX_HISTORY_TURNS
from document_processor import (
    embed_and_store,
    load_documents,
    load_existing_store,
    split_documents,
)

load_dotenv()

st.set_page_config(page_title="Document Q&A", layout="wide")

MISSING_API_KEY_ERROR = (
    "OPENAI_API_KEY is missing. Add it to your .env file before uploading or chatting."
)
AUTH_ERROR_MESSAGE = "OpenAI API key invalid or missing. Check your .env file."
OPENAI_ERROR_MESSAGE = "OpenAI API error. Please try again."


def build_chain_from_vectorstore() -> None:
    if st.session_state["vectorstore"] is None:
        st.session_state["chain"] = None
        return

    retriever = setup_retriever(st.session_state["vectorstore"])
    st.session_state["chain"] = create_chain(retriever)


def initialize_session_state(api_key_available: bool) -> None:
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
        existing_store = None
        if api_key_available:
            try:
                existing_store = load_existing_store()
            except FileNotFoundError:
                existing_store = None
            except openai.AuthenticationError:
                st.error(AUTH_ERROR_MESSAGE)
                st.stop()
            except openai.OpenAIError:
                st.error(OPENAI_ERROR_MESSAGE)
                st.stop()

        if existing_store is not None:
            st.session_state["vectorstore"] = existing_store
            st.info("Loaded existing document store.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "chain" not in st.session_state:
        st.session_state["chain"] = None

    if st.session_state["vectorstore"] is not None and st.session_state["chain"] is None:
        try:
            build_chain_from_vectorstore()
        except ImportError as error:
            st.error(str(error))
            st.stop()
        except openai.AuthenticationError:
            st.error(AUTH_ERROR_MESSAGE)
            st.stop()
        except openai.OpenAIError:
            st.error(OPENAI_ERROR_MESSAGE)
            st.stop()


def get_loaded_filenames() -> list[str]:
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        return []

    try:
        payload = vectorstore.get(include=["metadatas"])
    except Exception:
        return []

    metadatas = payload.get("metadatas") or []
    return sorted(
        {
            metadata.get("source")
            for metadata in metadatas
            if isinstance(metadata, dict) and metadata.get("source")
        }
    )


def process_documents(uploaded_files: list) -> bool:
    temp_dir = tempfile.mkdtemp()
    temp_paths: list[str] = []

    try:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as handle:
                handle.write(uploaded_file.getbuffer())
            temp_paths.append(file_path)

        with st.spinner("Embedding documents, please wait..."):
            try:
                docs = load_documents(temp_paths)
            except ValueError as error:
                st.error(str(error))
                return False

            chunks = split_documents(docs)

            try:
                vectorstore = embed_and_store(chunks)
            except OSError:
                st.error("Storage error: check directory permissions.")
                return False

        st.session_state["vectorstore"] = vectorstore
        try:
            build_chain_from_vectorstore()
        except ImportError as error:
            st.error(str(error))
            st.session_state["chain"] = None
            return False

        st.success(f"Processed {len(chunks)} chunks from {len(uploaded_files)} file(s).")
        return True
    except openai.AuthenticationError:
        st.error(AUTH_ERROR_MESSAGE)
        return False
    except openai.OpenAIError:
        st.error(OPENAI_ERROR_MESSAGE)
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for index, source in enumerate(sources, start=1):
            filename = source.get("file", "unknown")
            page = source.get("page", "unknown")
            st.write(f"{index}. {filename} (page {page})")


def main() -> None:
    api_key_available = bool(os.getenv("OPENAI_API_KEY"))
    initialize_session_state(api_key_available)
    messages = st.session_state["messages"]

    with st.sidebar:
        st.subheader("Upload and Process Documents")
        if not api_key_available:
            st.error(MISSING_API_KEY_ERROR)

        st.caption(
            f"Memory: {min(len(messages) // 2, MAX_HISTORY_TURNS)} turns retained"
        )

        loaded_filenames = get_loaded_filenames()
        if loaded_filenames:
            st.write("Loaded documents:")
            for filename in loaded_filenames:
                st.write(f"- {filename}")
        else:
            st.caption("Loaded documents: none")

        if st.button("Clear vector store"):
            st.session_state["vectorstore"] = None
            st.session_state["chain"] = None
            st.rerun()

        uploaded_files = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files and api_key_available:
            uploaded_filenames = sorted(file.name for file in uploaded_files)
            if uploaded_filenames != loaded_filenames:
                process_documents(uploaded_files)

    st.title("Document Q&A Chatbot")

    for message in messages:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            if role == "assistant":
                render_sources(message.get("sources") or [])

    query = st.chat_input("Ask a question about your documents...")
    if query:
        if st.session_state["chain"] is None:
            st.warning("Please upload and process documents first.")
            return

        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    result = ask_question(
                        st.session_state["chain"],
                        query,
                        history=st.session_state["messages"][:-1],
                    )
                except openai.AuthenticationError:
                    st.error(AUTH_ERROR_MESSAGE)
                    return
                except openai.OpenAIError:
                    st.error(OPENAI_ERROR_MESSAGE)
                    return

            answer = result.get("answer", "")
            sources = result.get("sources") or []
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.markdown(answer)
            render_sources(sources)


if __name__ == "__main__":
    main()
