# AzeemBot — Personal RAG Chatbot

**AzeemBot** is a Retrieval-Augmented Generation (RAG) chatbot built for **Muhammad Daniyal Azeem**, an AI & Flutter developer based in Lahore, Pakistan.

It answers questions about Daniyal’s background, education, skills, projects (Clinic Portal, Wallpaper App, Student Performance Prediction, ETL pipelines, etc.), and contact details using his own curated knowledge base.

---

## Architecture (matches the RAG diagram)

1. **Additional documents** (JSONL knowledge chunks) → **Encode** with embedding model  
2. Embeddings → **Index** into **FAISS vector database**  
3. User **Query** → Encode → **Similarity search** → retrieve top-k similar documents  
4. Retrieved docs + Query → **Prompt** → **LLM** (DeepSeek / OpenAI-compatible) → **Final response**

### Stack

| Component            | Choice                          |
|----------------------|---------------------------------|
| Embeddings           | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store         | FAISS (IndexFlatIP, cosine)     |
| LLM                  | DeepSeek Chat (or any OpenAI-compatible API) |
| Interface            | Streamlit                       |
| Knowledge base       | 140 personal chunks (JSONL)     |

---

## Quick start (local)

```bash
cd azeembot
pip install -r requirements.txt
streamlit run app.py
```

Optional: set an API key for natural generation

```bash
export DEEPSEEK_API_KEY=sk-...
# or
export OPENAI_API_KEY=sk-...
```

Without a key the bot still works in **retrieval-only** mode (returns the most relevant knowledge chunk).

---

## Dataset

- `daniyal_azeem_chatbot_knowledge.jsonl` — 140 documents covering:
  - Profile & identity
  - Education & personal details
  - Skills & strengths
  - Projects (Clinic Portal, Wallpaper App, ML model, ETL, databases, LUMINA…)
  - 120+ ready Q&A pairs
  - Contact & availability

Source PDF is also included for reference.

---

## Features

- Named identity: **AzeemBot**
- Full RAG pipeline (load → embed → index → retrieve → prompt → generate)
- Conversation history
- Optional source transparency
- Works with or without LLM API key
- Clean, professional Streamlit UI

---

## Deployment

### Streamlit Community Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `app.py`
4. (Optional) Add `DEEPSEEK_API_KEY` in Secrets

### Hugging Face Spaces

1. Create a new Space (Streamlit SDK)
2. Upload the files
3. Add secret `DEEPSEEK_API_KEY` if desired

---

## Author

Muhammad Daniyal Azeem  
BS Artificial Intelligence @ UMT, Lahore  
GitHub: [github.com/azeemuhammad](https://github.com/azeemuhammad)  
Portfolio: [azeemuhammad.github.io](https://azeemuhammad.github.io)
