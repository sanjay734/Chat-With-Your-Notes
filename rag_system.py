"""
RAG System — FREE, fully local, no separate app needed.

- Embeddings: sentence-transformers (all-MiniLM-L6-v2), runs on CPU
- Vector store: FAISS
- LLM: a small instruct model (SmolLM2) run directly with transformers,
  using the model's own chat format so it actually follows instructions.
"""

import os
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline as hf_pipeline

from config import (
    EMBEDDING_MODEL, LOCAL_LLM_MODEL,
    TOP_K_RETRIEVAL, GENERATION_TEMPERATURE, MAX_TOKENS_RESPONSE,
    SYSTEM_PROMPT, PERSIST_DIRECTORY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keep the prompt small so a tiny model on CPU stays fast and focused.
MAX_CONTEXT_CHARS = 4000


def build_messages(question: str, docs: List[Document]) -> list:
    """Build the chat messages (system + user) from the question and retrieved chunks."""
    parts, used = [], 0
    for i, d in enumerate(docs, 1):
        text = d.page_content.strip()
        if used + len(text) > MAX_CONTEXT_CHARS:
            text = text[: max(0, MAX_CONTEXT_CHARS - used)]
        if not text:
            break
        parts.append(f"[Excerpt {i}]\n{text}")
        used += len(text)
    context = "\n\n".join(parts)

    user_msg = (
        "Use only the document excerpts below to answer the question. "
        "If the answer is not in the excerpts, say you could not find it.\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def extract_reply(output) -> str:
    """Pull the assistant's text out of a transformers pipeline result."""
    try:
        gen = output[0]["generated_text"]
    except (IndexError, KeyError, TypeError):
        return ""
    if isinstance(gen, list):          # chat-style output: list of messages
        for msg in reversed(gen):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return (msg.get("content") or "").strip()
        return ""
    return str(gen).strip()            # plain string output


class RAGSystem:
    """Multi-format RAG system: 100% local and free."""

    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"Loading local LLM: {LOCAL_LLM_MODEL}")
        self.generator = hf_pipeline(
            "text-generation",
            model=LOCAL_LLM_MODEL,
            device=-1,  # CPU
        )

        self.vector_store = None
        self.retriever = None
        logger.info("RAG system ready")

    # Kept with this name so app.py's health check keeps working.
    def check_ollama_connection(self) -> Tuple[bool, str]:
        if self.generator is not None:
            return True, f"Local model '{LOCAL_LLM_MODEL}' loaded and ready"
        return False, "Model failed to load. Check the terminal for errors."

    def add_documents(self, chunks: List[dict]) -> int:
        if not chunks:
            return 0
        documents = [
            Document(page_content=c["content"], metadata=c["metadata"]) for c in chunks
        ]
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": TOP_K_RETRIEVAL}
        )
        return len(documents)

    def _generate(self, question: str, docs: List[Document]) -> str:
        messages = build_messages(question, docs)
        kwargs = {"max_new_tokens": MAX_TOKENS_RESPONSE}
        if GENERATION_TEMPERATURE > 0:
            kwargs.update(do_sample=True, temperature=GENERATION_TEMPERATURE)
        else:
            kwargs.update(do_sample=False)
        output = self.generator(messages, **kwargs)
        return extract_reply(output)

    def query(self, question: str) -> Tuple[str, List[dict]]:
        if self.retriever is None:
            return "No documents loaded. Please upload documents first.", []

        try:
            docs = self.retriever.invoke(question)
            sources = [{"content": d.page_content, "metadata": d.metadata} for d in docs]

            if not docs:
                return "I couldn't find anything relevant in your documents.", []

            answer = self._generate(question, docs)

            # Never show an empty box. If the model gave nothing, show the
            # closest matching text from the document instead.
            if not answer:
                best = docs[0].page_content.strip()
                answer = (
                    "The small model did not produce an answer this time. "
                    "Here is the closest matching text from your document:\n\n"
                    f"> {best[:800]}"
                )
            return answer, sources

        except Exception as e:
            logger.exception("Error while answering")
            return f"Something went wrong while answering: {e}", []

    def retrieve_similar_documents(self, query: str, k: int = 5) -> List[dict]:
        if self.vector_store is None:
            return []
        docs = self.vector_store.similarity_search(query, k=k)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

    def save_vector_store(self, path: str = PERSIST_DIRECTORY):
        if self.vector_store is None:
            return
        os.makedirs(path, exist_ok=True)
        self.vector_store.save_local(path)

    def load_vector_store(self, path: str = PERSIST_DIRECTORY):
        self.vector_store = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": TOP_K_RETRIEVAL}
        )

    def clear_vector_store(self):
        self.vector_store = None
        self.retriever = None

    def get_store_info(self) -> dict:
        if self.vector_store is None:
            return {"loaded": False, "documents": 0, "status": "No documents loaded"}
        return {"loaded": True, "status": "Vector store loaded and ready"}


class RAGMemoryManager:
    """Keeps a short chat history for the session."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.chat_history = []

    def add_to_history(self, question: str, answer: str):
        self.chat_history.append({"question": question, "answer": answer})
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def get_history(self) -> List[dict]:
        return self.chat_history

    def clear_history(self):
        self.chat_history = []
