"""
AzeemBot RAG Engine
Follows the classic RAG pipeline:
  1. Load & preprocess personal knowledge documents
  2. Generate embeddings (sentence-transformers)
  3. Index into FAISS vector store
  4. Retrieve similar documents via similarity search
  5. Build prompt with context + query
  6. Generate final answer with LLM (DeepSeek / OpenAI-compatible)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "daniyal_azeem_chatbot_knowledge.jsonl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # fast, strong for English semantic search
TOP_K = 5
SIMILARITY_THRESHOLD = 0.25  # cosine similarity floor


class AzeemRAG:
    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = data_path
        self.documents: List[Dict] = []
        self.texts: List[str] = []
        self.ids: List[str] = []
        self.categories: List[str] = []
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self._ready = False

    # ------------------------------------------------------------------
    # 1. Document loading & light preprocessing
    # ------------------------------------------------------------------
    def load_documents(self) -> None:
        docs = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "").strip()
                    if not text:
                        continue
                    # Light clean: collapse extra whitespace
                    text = " ".join(text.split())
                    docs.append({
                        "id": obj.get("id", f"doc_{len(docs)}"),
                        "text": text,
                        "category": obj.get("category", "general"),
                        "project_name": obj.get("project_name", None),
                    })
                except json.JSONDecodeError:
                    continue
        self.documents = docs
        self.texts = [d["text"] for d in docs]
        self.ids = [d["id"] for d in docs]
        self.categories = [d["category"] for d in docs]
        print(f"[AzeemRAG] Loaded {len(self.documents)} knowledge chunks.")

    # ------------------------------------------------------------------
    # 2. Embedding generation
    # ------------------------------------------------------------------
    def build_embeddings(self) -> None:
        if not self.texts:
            self.load_documents()
        print(f"[AzeemRAG] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so inner product == cosine
        )
        print(f"[AzeemRAG] Embeddings shape: {self.embeddings.shape}")

    # ------------------------------------------------------------------
    # 3. Index into FAISS (vector database)
    # ------------------------------------------------------------------
    def build_index(self) -> None:
        if self.embeddings is None:
            self.build_embeddings()
        dim = self.embeddings.shape[1]
        # IndexFlatIP works with normalized vectors → cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))
        self._ready = True
        print(f"[AzeemRAG] FAISS index ready with {self.index.ntotal} vectors.")

    def ensure_ready(self) -> None:
        if not self._ready:
            self.load_documents()
            self.build_embeddings()
            self.build_index()

    # ------------------------------------------------------------------
    # 4. Similarity search / retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        self.ensure_ready()
        # Fetch a few extra so we can re-rank
        fetch_k = min(top_k + 8, len(self.documents))
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        scores, indices = self.index.search(q_emb, fetch_k)
        results = []
        q_lower = query.lower()
        identity_q = any(k in q_lower for k in (
            "who are you", "about yourself", "introduce", "elevator",
            "who is daniyal", "who is muhammad", "background",
        ))
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < SIMILARITY_THRESHOLD:
                continue
            doc = self.documents[idx].copy()
            s = float(score)
            # Gentle boost for high-value identity / profile chunks
            if identity_q and doc.get("id") in (
                "profile_intro", "elevator_pitch", "about_me",
                "qa_001", "qa_006", "qa_007", "qa_092", "qa_101",
            ):
                s += 0.15
            elif identity_q and doc.get("category") == "profile":
                s += 0.08
            doc["score"] = s
            results.append(doc)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # 5. Prompt engineering
    # ------------------------------------------------------------------
    def build_prompt(
        self,
        query: str,
        retrieved: List[Dict],
        history: Optional[List[Dict]] = None,
    ) -> str:
        context_blocks = []
        for i, doc in enumerate(retrieved, 1):
            cat = doc.get("category", "")
            proj = doc.get("project_name")
            header = f"[{cat}]" if cat else f"[doc-{i}]"
            if proj:
                header += f" Project: {proj}"
            context_blocks.append(f"{header}\n{doc['text']}")

        context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant knowledge found."

        history_text = ""
        if history:
            # keep last 4 turns for context window safety
            recent = history[-4:]
            lines = []
            for turn in recent:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    lines.append(f"User: {content}")
                else:
                    lines.append(f"AzeemBot: {content}")
            history_text = "\n".join(lines)

        system_persona = (
            "You are AzeemBot, the personal AI assistant of Muhammad Daniyal Azeem "
            "(also called Daniyal). You speak in first person as Daniyal when answering "
            "questions about him — friendly, professional, concise, and natural. "
            "Never invent facts. Only use the provided knowledge context. "
            "If the answer is not in the context, say you don't have that information "
            "and politely suggest contacting Daniyal directly (email: azeemuhammad47@gmail.com). "
            "Keep answers warm and human — avoid sounding robotic or overly formal."
        )

        prompt = f"""{system_persona}

=== KNOWLEDGE CONTEXT ===
{context}
=== END CONTEXT ===

"""
        if history_text:
            prompt += f"Recent conversation:\n{history_text}\n\n"

        prompt += f"Current question: {query}\n\nAzeemBot:"
        return prompt

    # ------------------------------------------------------------------
    # 6. LLM generation (DeepSeek / any OpenAI-compatible endpoint)
    # ------------------------------------------------------------------
    def generate(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ) -> Tuple[str, List[Dict]]:
        retrieved = self.retrieve(query)
        prompt = self.build_prompt(query, retrieved, history)

        # Prefer DeepSeek if key present; otherwise fall back to pure retrieval answer
        key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key, base_url=base_url)
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are AzeemBot, personal assistant of Muhammad Daniyal Azeem. "
                            "Answer only from the given context. Speak as Daniyal in first person "
                            "when appropriate. Be natural and concise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.35,
                    max_tokens=600,
                )
                answer = resp.choices[0].message.content.strip()
                return answer, retrieved
            except Exception as e:
                # graceful fallback
                fallback = self._fallback_answer(query, retrieved)
                return f"{fallback}\n\n_(LLM unavailable: {str(e)[:80]})_", retrieved

        return self._fallback_answer(query, retrieved), retrieved

    def _extract_answer(self, text: str) -> str:
        """Pull the answer portion out of Q&A-style chunks."""
        for sep in ["Answer:", "A:"]:
            if sep in text:
                return text.split(sep, 1)[-1].strip()
        return text.strip()

    def _fallback_answer(self, query: str, retrieved: List[Dict]) -> str:
        """Useful answer when no LLM key is available."""
        if not retrieved:
            return (
                "I don't have that information in my knowledge base right now. "
                "Feel free to reach Daniyal at azeemuhammad47@gmail.com or +92 300 6868447."
            )

        q = query.lower().strip()

        # Prefer profile / intro / elevator / about_me chunks for identity questions
        identity_keywords = ("who are you", "tell me about yourself", "introduce yourself",
                             "about you", "who is daniyal", "who is muhammad", "elevator",
                             "short introduction", "background")
        if any(k in q for k in identity_keywords):
            for doc in retrieved:
                if doc.get("id") in ("profile_intro", "elevator_pitch", "about_me", "qa_001", "qa_006", "qa_007", "qa_092"):
                    return self._extract_answer(doc["text"])
            # also prefer any profile category
            for doc in retrieved:
                if doc.get("category") == "profile":
                    return self._extract_answer(doc["text"])

        # Prefer exact Q&A match when the question text is present
        for doc in retrieved:
            text = doc["text"]
            if text.lower().startswith("question:") and q in text.lower()[:120]:
                return self._extract_answer(text)

        # Default: best retrieved chunk
        return self._extract_answer(retrieved[0]["text"])


# Singleton used by the Streamlit app
_rag_instance: Optional[AzeemRAG] = None


def get_rag() -> AzeemRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = AzeemRAG()
        _rag_instance.ensure_ready()
    return _rag_instance
