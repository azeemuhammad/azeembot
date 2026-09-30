# AzeemBot — Personal RAG Chatbot + Persistent Memory + Gemini

**AzeemBot** is a Retrieval-Augmented Generation (RAG) chatbot built for **Muhammad Daniyal Azeem**, an AI & Flutter developer based in Lahore, Pakistan.

It answers questions about Daniyal’s background, education, skills, projects, and contact details using his curated knowledge base.

**Features in this version:**
- **Google Gemini** as the default LLM (free tier via [Google AI Studio](https://aistudio.google.com/apikey))
- Also supports DeepSeek and OpenAI
- **Persistent memory** — store DOB, preferred name, standing instructions, free-form facts
- Facts can be written into the knowledge dataset
- Professional Streamlit UI + CLI

---

## Architecture

1. Knowledge documents (JSONL) → Embed with `all-MiniLM-L6-v2`  
2. Embeddings → FAISS vector index  
3. Query → similarity search → top-k chunks  
4. Chunks + **User Memory** + query → Prompt → **Gemini / other LLM** → answer  

### Persistent Memory

| You say | Bot does |
|---------|----------|
| `Store my DOB as 15/03/2003` | Saves DOB to `user_memory.json` |
| `What is my date of birth?` | Answers from memory |
| `Call me Boss` | Saves preferred name |
| `Always answer in short points` | Standing instruction |
| `Remember that my city is Lahore` | Free-form fact |

---

## Quick start

```bash
cd azeembot
pip install -r requirements.txt
```

### 1. Get a free Gemini API key

1. Open https://aistudio.google.com/apikey  
2. Create an API key  
3. Either paste it in the Streamlit sidebar, or:

```bash
export GEMINI_API_KEY=your_key_here
```

### 2. Run

```bash
streamlit run app.py
```

CLI:

```bash
python cli_chat.py
```

Without any key the bot still works (retrieval + memory only).

---

## LLM providers (sidebar)

| Provider | Default model | Env variable |
|----------|---------------|--------------|
| **Google Gemini** (default) | `gemini-3.8-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| DeepSeek | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |

Gemini is called via Google’s OpenAI-compatible endpoint:

```
https://generativelanguage.googleapis.com/v1beta/openai/
```

Native `google-generativeai` is used as automatic fallback if needed.

---

## Dataset

- `daniyal_azeem_chatbot_knowledge.jsonl` — 140+ chunks (profile, skills, projects, Q&A, contact)
- New user facts may be appended under category `user_memory`
- Source PDF included for reference

---

## Project structure

```
azeembot/
├── app.py                 # Streamlit UI
├── cli_chat.py            # Terminal chat
├── rag_engine.py          # RAG + Gemini / DeepSeek / OpenAI
├── memory.py              # Persistent memory store
├── daniyal_azeem_chatbot_knowledge.jsonl
├── user_memory.json       # created at runtime
├── requirements.txt
├── README.md
└── .streamlit/config.toml
```

---

## Deployment

### Streamlit Community Cloud

1. Push folder to GitHub  
2. Deploy `app.py` on https://share.streamlit.io  
3. Add secret `GEMINI_API_KEY` under Settings → Secrets  

### Hugging Face Spaces

1. New Space (Streamlit SDK)  
2. Upload files  
3. Add `GEMINI_API_KEY` as a Space secret  

> On ephemeral hosts, `user_memory.json` resets unless you attach persistent storage.

---

## Author

Muhammad Daniyal Azeem  
BS Artificial Intelligence @ UMT, Lahore  
GitHub: [github.com/azeemuhammad](https://github.com/azeemuhammad)  
Portfolio: [azeemuhammad.github.io](https://azeemuhammad.github.io)
