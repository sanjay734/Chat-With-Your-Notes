# Chat With Your Notes — FREE Multi-Format RAG Assistant (No Separate App)

100% free, fully local. No API key, no billing, **no Ollama or any other app to install** — the AI model runs directly inside this Python app.

## 💰 Why this is free and simple

| Component | Runs where | Cost |
|---|---|---|
| LLM (SmolLM2-360M) | Inside this app, via `transformers` | $0 |
| Embeddings (MiniLM) | Inside this app, via `sentence-transformers` | $0 |
| Vector store (FAISS) | Local disk | $0 |

The only internet usage is the **first run**, when the two small model files download automatically (total ~800MB–1GB). After that, everything works offline.

## 🚀 Setup

```bash
cd rag_assistant_free
python -m venv venv
venv\Scripts\activate        # Windows. Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

That's it — no separate app, no `ollama pull`, nothing else to install. The first time you run it, it'll pause for a minute or two while it downloads the model, then it's ready.

## Running it again later (any day after)
```bash
cd rag_assistant_free
venv\Scripts\activate
streamlit run app.py
```
Nothing re-downloads. Nothing reinstalls.

## 🐢 Poor / slow internet? Use an even smaller model

Edit `config.py` (or `.env`) and change:
```
LOCAL_LLM_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
```
This is the smallest option (~270MB download) — weaker answers, but works on very limited internet or hardware.

| Model | Download size | Answer quality |
|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | ~270MB | Basic |
| `HuggingFaceTB/SmolLM2-360M-Instruct` (default) | ~750MB | Good |
| `Qwen/Qwen2.5-0.5B-Instruct` | ~1GB | Better |

## 🔧 Troubleshooting

**First run is slow** → it's downloading the model. Just wait — subsequent runs are fast.

**Out of memory** → switch to `SmolLM2-135M-Instruct` in config.py.

**Answers seem weak/off-topic** → these are very small models, chosen specifically to be tiny and free. For noticeably better answers, try `Qwen2.5-0.5B-Instruct`, or use the paid OpenAI-based version.

---
**Built with:** Python · Streamlit · LangChain · HuggingFace Transformers · FAISS — no external services required.
