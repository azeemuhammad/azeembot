# AzeemBot — RAG Chatbot Project Report

**Student:** Muhammad Daniyal Azeem  
**Chatbot Name:** AzeemBot  
**Course Project:** RAG-based Personal Chatbot  

---

## 1. Project Overview

AzeemBot is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Muhammad Daniyal Azeem using his personal knowledge base. It follows the standard RAG architecture shown in the provided diagram:

1. Documents are encoded by an embedding model  
2. Embeddings are indexed in a vector database (FAISS)  
3. User query is encoded and used for similarity search  
4. Top similar documents + query form a prompt  
5. LLM generates the final response  

The chatbot has a clear personal identity (“AzeemBot”) and speaks naturally about Daniyal’s education, skills, projects, and contact information.

---

## 2. Dataset

| Source | Description |
|--------|-------------|
| `daniyal_azeem_chatbot_knowledge.jsonl` | 140 curated chunks |
| Categories | profile, skills, education, project, contact, availability, qa |
| Content | Profile intro, about me, skills, strengths, education, languages, personal details, 6+ projects (Clinic Portal, Wallpaper App, Student Performance Prediction, ETL, Database Design, LUMINA), 120+ Q&A pairs, contact info |

Data was lightly preprocessed (whitespace normalization, JSON validation) before embedding.

---

## 3. System Implementation

### 3.1 Document loading & preprocessing
- JSONL lines parsed into structured documents  
- Fields: `id`, `text`, `category`, optional `project_name`  

### 3.2 Embedding generation
- Model: `sentence-transformers/all-MiniLM-L6-v2`  
- 384-dimensional vectors, L2-normalized  

### 3.3 Vector database
- FAISS `IndexFlatIP` (inner product = cosine similarity on normalized vectors)  
- 140 vectors indexed at startup  

### 3.4 Retrieval
- Top-k = 5  
- Cosine similarity threshold = 0.25  
- Light re-ranking boost for profile / identity chunks  

### 3.5 Prompt engineering
- System persona: AzeemBot speaks as Daniyal’s assistant, uses first person where natural, refuses to invent facts  
- Context block of retrieved documents + optional short conversation history  

### 3.6 LLM generation
- Primary: DeepSeek Chat (OpenAI-compatible API)  
- Fallback: retrieval-only mode (extracts best matching chunk) when no API key is set  

### 3.7 Interface
- **Streamlit web app** (`app.py`) — preferred  
- CLI (`cli_chat.py`) for quick testing  
- Conversation history maintained in session state  
- Optional “show sources” for transparency  

---

## 4. How to Run Locally

```bash
cd azeembot
pip install -r requirements.txt
streamlit run app.py
```

Optional LLM key:

```bash
export DEEPSEEK_API_KEY=your_key_here
```

---

## 5. Deployment Instructions (Mandatory)

### Option A — Streamlit Community Cloud (recommended)

1. Create a GitHub repository and push the `azeembot` folder  
2. Visit https://share.streamlit.io  
3. Connect the repo, select `app.py` as the main file  
4. (Optional) Add secret `DEEPSEEK_API_KEY` under Settings → Secrets  
5. Deploy — you will get a public URL  

### Option B — Hugging Face Spaces

1. Create a new Space with SDK = Streamlit  
2. Upload all files from the `azeembot` folder  
3. Add `DEEPSEEK_API_KEY` as a Space secret if desired  
4. Space builds and publishes a public URL  

Local-only running is **not** sufficient for the assignment; a public deployment link is required at submission time.

---

## 6. Sample Queries & Expected Behavior

| Query | Expected answer focus |
|-------|------------------------|
| Who are you? | Intro as Muhammad Daniyal Azeem, AI/Flutter developer, UMT |
| What is Clinic Portal? | Multi-role Flutter clinic app, Firebase, JazzCash, etc. |
| What skills do you have? | Flutter, Dart, Firebase, Python, ML, SQL, C++… |
| How can I contact you? | Email, WhatsApp, GitHub, LinkedIn, portfolio |
| Are you looking for an internship? | Yes — Flutter/Mobile in Lahore (Johar Town) |

---

## 7. Project Structure

```
azeembot/
├── app.py                          # Streamlit UI
├── rag_engine.py                   # Full RAG pipeline
├── cli_chat.py                     # Optional CLI
├── daniyal_azeem_chatbot_knowledge.jsonl
├── Daniyal_Azeem_Chatbot_Knowledge_Base.pdf
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md
```

---

## 8. Conclusion

AzeemBot successfully implements a complete RAG pipeline on a personal dataset, provides a named chatbot identity, maintains conversation history, and is ready for public deployment on Streamlit Cloud or Hugging Face Spaces.
