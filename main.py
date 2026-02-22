import os
import shutil
import tempfile

import openai
import streamlit as st
from dotenv import load_dotenv

from chatbot import ask_question, create_chain, setup_retriever
from document_processor import (
    embed_and_store,
    load_documents,
    load_existing_store,
    split_documents,
)

load_dotenv()

st.set_page_config(page_title="Document Q&A", layout="wide")


def build_chain_from_vectorstore() -> None:
    if st.session_state["vectorstore"] is None:
        st.session_state["chain"] = None
        return

    retriever = setup_retriever(st.session_state["vectorstore"])
    st.session_state["chain"] = create_chain(retriever)


def initialize_session_state() -> None:
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
        try:
            existing_store = load_existing_store()
        except openai.AuthenticationError:
            st.error("OpenAI API key invalid or missing. Check your .env file.")
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
        except openai.AuthenticationError:
            st.error("OpenAI API key invalid or missing. Check your .env file.")
            st.stop()


def process_documents(uploaded_files: list) -> None:
    temp_dir = tempfile.mkdtemp()
    temp_paths: list[str] = []

    try:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as handle:
                handle.write(uploaded_file.getbuffer())
            temp_paths.append(file_path)

        docs = []
        skipped_errors = []
        for path in temp_paths:
            try:
                docs.extend(load_documents([path]))
            except ValueError as error:
                skipped_errors.append(str(error))

        if not docs:
            st.session_state["vectorstore"] = None
            st.session_state["chain"] = None
            for error in skipped_errors:
                st.error(error)
            st.stop()

        for error in skipped_errors:
            st.warning(f"Skipped file: {error}")

        chunks = split_documents(docs)
        processed_count = len(uploaded_files) - len(skipped_errors)
        st.success(
            f"Processed {len(chunks)} chunks from {processed_count} file(s)."
        )

        try:
            vectorstore = embed_and_store(chunks)
        except OSError:
            st.error("Storage error: check directory permissions.")
            st.stop()

        st.session_state["vectorstore"] = vectorstore
        build_chain_from_vectorstore()
    except openai.AuthenticationError:
        st.error("OpenAI API key invalid or missing. Check your .env file.")
        st.stop()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def render_sources(sources: list[dict]) -> None:
    with st.expander("Sources"):
        if not sources:
            st.write("No sources returned.")
            return

        for index, source in enumerate(sources, start=1):
            st.write(f"{index}. Filename: {source.get('file', 'unknown')}")
            st.write(f"Page: {source.get('page', 'unknown')}")
            st.write(f"Excerpt: {source.get('excerpt', '')}")


def main() -> None:
    initialize_session_state()

    with st.sidebar:
        st.subheader("Upload and Process Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF before processing.")
            else:
                process_documents(uploaded_files)

    st.title("Document Q&A Chatbot")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
                    result = ask_question(st.session_state["chain"], query)
                except openai.AuthenticationError:
                    st.error("OpenAI API key invalid or missing. Check your .env file.")
                    st.stop()

            answer = result.get("answer", "")
            sources = result.get("sources") or []
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )
            st.markdown(answer)
            render_sources(sources)


if __name__ == "__main__":
    main()
