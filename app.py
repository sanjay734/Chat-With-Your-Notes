"""
Chat With Your Notes — FREE Multi-Format RAG Assistant
100% local: HuggingFace embeddings + a tiny local LLM (via transformers) + FAISS.
No API key, no billing, no separate app to install.
"""

import streamlit as st
from document_processor import DocumentProcessor
from rag_system import RAGSystem, RAGMemoryManager
from config import SUPPORTED_FORMATS, LOCAL_LLM_MODEL
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Chat With Your Notes (Free RAG)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



@st.cache_resource
def initialize_rag_system():
    return RAGSystem()


@st.cache_resource
def initialize_doc_processor():
    return DocumentProcessor()


@st.cache_resource
def initialize_memory():
    return RAGMemoryManager()


def display_source_documents(sources):
    if not sources:
        return
    st.markdown("### 📄 Source Documents")
    for i, doc in enumerate(sources, 1):
        with st.expander(f"Source {i}: {doc['metadata'].get('source', 'Unknown')}"):
            st.markdown(f"**File:** {doc['metadata'].get('source', 'Unknown')}")
            st.markdown(f"**Type:** {doc['metadata'].get('file_type', 'Unknown').upper()}")
            if 'page' in doc['metadata']:
                st.markdown(f"**Page:** {doc['metadata']['page']}")
            st.markdown("---")
            st.markdown(f"**Content:**\n\n{doc['content']}")


def main():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = initialize_rag_system()
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = initialize_doc_processor()
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = initialize_memory()
    if "document_count" not in st.session_state:
        st.session_state.document_count = 0
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0

    st.title("🤖 Chat With Your Notes — Free RAG Assistant")
    st.caption("100% local · $0 cost · No API key · No separate app to install")

    # First run loads the local model into memory (downloads it once if needed) —
    # this can take a little while the very first time.
    is_ready, model_msg = st.session_state.rag_system.check_ollama_connection()
    if not is_ready:
        st.error(f"**Model failed to load:**\n\n{model_msg}")
    else:
        st.success(f"✅ {model_msg}")

    st.markdown("---")

    with st.sidebar:
        st.header("📋 Settings & Info")
        st.subheader("📤 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'docx', 'csv', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, CSV, TXT"
        )

        if uploaded_files:
            if st.button("Process Documents", use_container_width=True):
                with st.spinner("Processing documents locally..."):
                    total_new_chunks = 0
                    progress_bar = st.progress(0)

                    for idx, uploaded_file in enumerate(uploaded_files):
                        try:
                            chunks, file_type = st.session_state.doc_processor.process_uploaded_file(uploaded_file)
                            added_docs = st.session_state.rag_system.add_documents(chunks)
                            total_new_chunks += added_docs
                            st.session_state.document_count += 1
                            st.session_state.total_chunks += added_docs
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                    if total_new_chunks > 0:
                        st.success(f"✅ Processed {len(uploaded_files)} file(s)")
                        st.info(f"📊 Created {total_new_chunks} chunks (embedded locally, no API calls)")

        st.markdown("---")
        st.subheader("ℹ️ System Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents Loaded", st.session_state.document_count)
        with col2:
            st.metric("Total Chunks", st.session_state.total_chunks)

        store_info = st.session_state.rag_system.get_store_info()
        if store_info['loaded']:
            st.success("✅ " + store_info['status'])
        else:
            st.info("ℹ️ " + store_info['status'])

        st.markdown("---")
        st.subheader("⚙️ Model Info")
        st.caption(f"**LLM:** {LOCAL_LLM_MODEL} (runs in-process, no separate app)")
        st.caption("**Embeddings:** all-MiniLM-L6-v2 (HuggingFace, local)")
        st.caption("**Vector Store:** FAISS (local)")
        st.caption("**Cost:** $0 — everything runs on your machine")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Data", use_container_width=True):
                st.session_state.rag_system.clear_vector_store()
                st.session_state.memory_manager.clear_history()
                st.session_state.document_count = 0
                st.session_state.total_chunks = 0
                st.session_state.last_result = None
                st.success("Data cleared")
                st.rerun()
        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.memory_manager.clear_history()
                st.session_state.last_result = None
                st.success("Chat cleared")
                st.rerun()

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("💬 Ask Questions")

        if st.session_state.document_count == 0:
            st.warning("⚠️ Please upload and process documents first.")
        else:
            question = st.text_area(
                "Enter your question:",
                placeholder="What is the main topic of the document?",
                height=100,
                label_visibility="collapsed"
            )

            if st.button("🔍 Search", use_container_width=True) and question.strip():
                with st.spinner("Thinking locally (this can take a little while on CPU)..."):
                    answer, sources = st.session_state.rag_system.query(question)
                st.session_state.memory_manager.add_to_history(question, answer)
                st.session_state.last_result = {
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                }

            # Show the latest result from session state, so it stays on screen
            # even if the page reruns (clicking a widget reruns the whole script).
            last = st.session_state.get("last_result")
            if last:
                st.markdown("### 💡 Answer")
                with st.container(border=True):
                    st.markdown(last["answer"])

                if last["sources"]:
                    display_source_documents(last["sources"])
                else:
                    st.info("No relevant source documents found")

        st.markdown("---")
        st.subheader("📜 Chat History")
        history = st.session_state.memory_manager.get_history()
        if history:
            for idx, exchange in enumerate(history, 1):
                with st.expander(f"Q{idx}: {exchange['question'][:50]}..."):
                    st.markdown(f"**Question:** {exchange['question']}")
                    st.markdown(f"**Answer:** {exchange['answer']}")
        else:
            st.info("No chat history yet.")

    with col2:
        st.subheader("📊 Statistics")
        st.metric("System Status", "Ready" if st.session_state.document_count > 0 else "Waiting")

        st.markdown("---")
        st.markdown("**Supported Formats:**")
        for ext, name in SUPPORTED_FORMATS.items():
            st.caption(f"• .{ext} - {name}")

        st.markdown("---")
        st.markdown("**Why it's free:**")
        st.caption("🖥️ LLM runs directly in this app via transformers")
        st.caption("🖥️ Embeddings run on your CPU via HuggingFace")
        st.caption("💾 FAISS index stored locally on disk")
        st.caption("🚫 No API key, no per-query billing")


if __name__ == "__main__":
    main()
