"""
Configuration settings for the FREE Multi-Format RAG Assistant.

Everything here runs locally on your own machine — no separate app to install:
- Embeddings: HuggingFace sentence-transformers (downloaded once, then offline)
- LLM: A tiny HuggingFace model run directly via Python (downloaded once, then offline)
- Vector store: FAISS (always local)

No API key. No billing. No Ollama. No internet needed after the one-time downloads.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Embedding Model (runs locally via sentence-transformers) ────────────
# all-MiniLM-L6-v2: 384 dimensions, ~80MB, fast on CPU, great for RAG.
# Alternatives (swap the name below if you want):
#   - "sentence-transformers/all-mpnet-base-v2"   (higher quality, slower, ~420MB)
#   - "BAAI/bge-small-en-v1.5"                     (strong retrieval performance, ~130MB)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── LLM Configuration (runs locally via transformers — NO separate app needed) ──
# Downloaded automatically the first time you run the app, then cached forever
# in ~/.cache/huggingface — no internet needed after that.
#
# Pick based on your internet/RAM (smallest first):
#   "HuggingFaceTB/SmolLM2-135M-Instruct"  -> ~270MB download, fastest, weakest answers
#   "HuggingFaceTB/SmolLM2-360M-Instruct"  -> ~750MB download, good balance (DEFAULT)
#   "Qwen/Qwen2.5-0.5B-Instruct"            -> ~1GB download, noticeably better answers
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")

# ── Document Processing ──────────────────────────────────────────────────
CHUNK_SIZE = 1000       # Characters per chunk
CHUNK_OVERLAP = 200     # Overlap between chunks
MAX_CHUNKS = 500        # Maximum chunks to process per file

# ── RAG Retrieval ─────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 5           # Number of chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.5    # Minimum similarity score

# ── Streamlit / Upload Limits ────────────────────────────────────────────
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB max file size
SUPPORTED_FORMATS = {
    'pdf': 'PDF Files',
    'docx': 'Word Documents',
    'csv': 'CSV Files',
    'txt': 'Text Files'
}

# ── Vector Store ──────────────────────────────────────────────────────────
VECTOR_STORE_TYPE = "faiss"
PERSIST_DIRECTORY = "./data/vector_store"

# ── Generation Settings ──────────────────────────────────────────────────
GENERATION_TEMPERATURE = 0.3   # Lower = more focused/deterministic (local models
                                # tend to ramble more than GPT-4o-mini, so we keep
                                # this conservative by default)
MAX_TOKENS_RESPONSE = 300   # shorter answers = much faster on CPU

# ── Prompts ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful AI assistant specialized in analyzing and answering questions about documents.
You have access to a set of document chunks retrieved based on the user's query.

Guidelines:
1. Always base your answers on the provided document context
2. If the answer is not found in the documents, clearly state that
3. Be concise but comprehensive
4. Quote relevant sections when appropriate
5. Acknowledge when information might be incomplete"""

QA_PROMPT_TEMPLATE = """Use the following document excerpts to answer the question as accurately as possible.
If the answer isn't in the excerpts, say so clearly instead of guessing.

Document Excerpts:
{context}

Question: {question}

Answer:"""
