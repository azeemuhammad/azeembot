"""
AzeemBot RAG Engine
Follows the classic RAG pipeline:
  1. Load & preprocess personal knowledge documents
  2. Generate embeddings (sentence-transformers)
  3. Index into FAISS vector store
  4. Retrieve similar documents via similarity search
  5. Build prompt with context + query + persistent memory
  6. Generate final answer with LLM (Gemini / DeepSeek / OpenAI-compatible)

Also supports:
  - Persistent user memory (facts, DOB, preferred name, standing instructions)
  - Dynamic addition of knowledge chunks into the JSONL dataset
  - Re-indexing after new knowledge is added
  - Google Gemini via OpenAI-compatible endpoint or native google-generativeai
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from memory import get_memory, MemoryStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "daniyal_azeem_chatbot_knowledge.jsonl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.25

# Provider presets
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.8-flash",
        "env_keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_keys": ("DEEPSEEK_API_KEY",),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_keys": ("OPENAI_API_KEY",),
    },
}


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
            normalize_embeddings=True,
        )
        print(f"[AzeemRAG] Embeddings shape: {self.embeddings.shape}")

    # ------------------------------------------------------------------
    # 3. Index into FAISS (vector database)
    # ------------------------------------------------------------------
    def build_index(self) -> None:
        if self.embeddings is None:
            self.build_embeddings()
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))
        self._ready = True
        print(f"[AzeemRAG] FAISS index ready with {self.index.ntotal} vectors.")

    def ensure_ready(self) -> None:
        if not self._ready:
            self.load_documents()
            self.build_embeddings()
            self.build_index()

    def reload_index(self) -> None:
        """Re-load documents and rebuild index (after knowledge was appended)."""
        self._ready = False
        self.documents = []
        self.texts = []
        self.ids = []
        self.categories = []
        self.embeddings = None
        self.index = None
        self.ensure_ready()

    # ------------------------------------------------------------------
    # 4. Similarity search / retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        self.ensure_ready()
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
    # 5. Prompt engineering (includes persistent memory)
    # ------------------------------------------------------------------
    def build_prompt(
        self,
        query: str,
        retrieved: List[Dict],
        history: Optional[List[Dict]] = None,
        memory: Optional[MemoryStore] = None,
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
            recent = history[-6:]
            lines = []
            for turn in recent:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    lines.append(f"User: {content}")
                else:
                    lines.append(f"AzeemBot: {content}")
            history_text = "\n".join(lines)

        memory_block = ""
        if memory:
            memory_block = memory.to_prompt_block()

        preferred = ""
        if memory:
            pname = memory.get_profile("preferred_name")
            if pname:
                preferred = f" Address the user as {pname} when appropriate."

        system_persona = (
            "You are AzeemBot, the personal AI assistant of Muhammad Daniyal Azeem "
            "(also called Daniyal). You speak in first person as Daniyal when answering "
            "questions about him — friendly, professional, concise, and natural. "
            "Never invent facts about Daniyal. Only use the provided knowledge context. "
            "If the answer is not in the context, say you don't have that information "
            "and politely suggest contacting Daniyal (email: azeemuhammad47@gmail.com). "
            "You MUST obey any standing instructions and user memory provided below. "
            "When the user asks you to remember, store, or change personal info, confirm "
            "clearly that it has been saved."
            f"{preferred}"
        )

        prompt = f"""{system_persona}

{memory_block}
=== KNOWLEDGE CONTEXT ===
{context}
=== END CONTEXT ===

"""
        if history_text:
            prompt += f"Recent conversation:\n{history_text}\n\n"

        prompt += f"Current question: {query}\n\nAzeemBot:"
        return prompt

    def _resolve_credentials(
        self,
        api_key: Optional[str],
        base_url: str,
        model: str,
        provider: str = "gemini",
    ) -> Tuple[Optional[str], str, str]:
        """Pick API key, base_url, model from args / env / provider preset."""
        preset = PROVIDERS.get(provider, PROVIDERS["gemini"])
        key = api_key
        if not key:
            for env_name in preset.get("env_keys", ()):
                key = os.getenv(env_name)
                if key:
                    break
            if not key:
                key = (
                    os.getenv("GEMINI_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                    or os.getenv("DEEPSEEK_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                )
        url = base_url or preset["base_url"]
        mdl = model or preset["model"]
        # Auto-fix base_url if user selected Gemini but left DeepSeek URL
        if key and provider == "gemini" and "deepseek" in (url or ""):
            url = PROVIDERS["gemini"]["base_url"]
            if model in ("deepseek-chat", "", None):
                mdl = PROVIDERS["gemini"]["model"]
        return key, url, mdl

    # ------------------------------------------------------------------
    # 6. LLM generation (Gemini preferred, also DeepSeek / OpenAI)
    # ------------------------------------------------------------------
    def generate(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-3.8-flash",
        memory: Optional[MemoryStore] = None,
        provider: str = "gemini",
    ) -> Tuple[str, List[Dict]]:
        mem = memory or get_memory()

        # 1) Try direct answer from memory (DOB, name, etc.)
        mem_answer = mem.answer_from_memory(query)
        if mem_answer:
            return mem_answer, []

        # 2) Detect store / update intents and persist
        action_summary, confirm_msg = mem.process_user_message(query)
        if confirm_msg:
            if action_summary and any(k in action_summary.lower() for k in ("dob", "name", "saved", "fact")):
                try:
                    fact_text = confirm_msg.replace("**", "")
                    mem.append_to_knowledge_base(
                        text=f"User memory: {fact_text}",
                        category="user_memory",
                    )
                except Exception:
                    pass
            return confirm_msg, []

        # 3) Normal RAG path
        retrieved = self.retrieve(query)
        prompt = self.build_prompt(query, retrieved, history, memory=mem)

        key, url, mdl = self._resolve_credentials(api_key, base_url, model, provider=provider)

        if key:
            # Prefer OpenAI-compatible client (works for Gemini, DeepSeek, OpenAI)
            try:
                answer = self._call_openai_compatible(key, url, mdl, prompt)
                return answer, retrieved
            except Exception as e1:
                # Fallback: native Gemini SDK if provider is gemini
                if provider == "gemini" or "generativelanguage" in (url or ""):
                    try:
                        answer = self._call_gemini_native(key, mdl, prompt)
                        return answer, retrieved
                    except Exception as e2:
                        fallback = self._fallback_answer(query, retrieved)
                        return (
                            f"{fallback}\n\n_(LLM unavailable: {str(e1)[:60]} | native: {str(e2)[:60]})_",
                            retrieved,
                        )
                fallback = self._fallback_answer(query, retrieved)
                return f"{fallback}\n\n_(LLM unavailable: {str(e1)[:80]})_", retrieved

        return self._fallback_answer(query, retrieved), retrieved

    def _call_openai_compatible(self, api_key: str, base_url: str, model: str, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        system_content = (
            "You are AzeemBot, personal assistant of Muhammad Daniyal Azeem. "
            "Answer only from the given context and user memory. "
            "Speak as Daniyal in first person when appropriate. "
            "Strictly follow any standing instructions in the memory block. "
            "Be natural and concise."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.35,
            max_tokens=700,
        )
        return resp.choices[0].message.content.strip()

    def _call_gemini_native(self, api_key: str, model: str, prompt: str) -> str:
        """Native google-generativeai fallback."""
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # Map common / deprecated aliases to a live model
        model_name = model or "gemini-3.8-flash"
        aliases = {
            "gemini-2.0-flash": "gemini-3.8-flash",
            "gemini-2.0-flash-lite": "gemini-3.5-flash-lite",
            "gemini-1.5-flash": "gemini-2.5-flash",
            "gemini-1.5-pro": "gemini-2.5-pro",
            "gemini-flash": "gemini-3.8-flash",
            "gemini": "gemini-3.8-flash",
            "deepseek-chat": "gemini-3.8-flash",
        }
        model_name = aliases.get(model_name, model_name)
        system = (
            "You are AzeemBot, personal assistant of Muhammad Daniyal Azeem. "
            "Answer only from the given context and user memory. "
            "Speak as Daniyal in first person when appropriate. Be natural and concise."
        )
        gm = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system,
        )
        resp = gm.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.35,
                max_output_tokens=700,
            ),
        )
        return (resp.text or "").strip()

    def _extract_answer(self, text: str) -> str:
        for sep in ["Answer:", "A:"]:
            if sep in text:
                return text.split(sep, 1)[-1].strip()
        return text.strip()

    def _fallback_answer(self, query: str, retrieved: List[Dict]) -> str:
        if not retrieved:
            return (
                "I don't have that information in my knowledge base right now. "
                "Feel free to reach Daniyal at azeemuhammad47@gmail.com or +92 300 6868447. "
                "You can also ask me to remember something for next time "
                "(e.g. “remember my DOB is 15/03/2003”)."
            )

        q = query.lower().strip()

        identity_keywords = (
            "who are you", "tell me about yourself", "introduce yourself",
            "about you", "who is daniyal", "who is muhammad", "elevator",
            "short introduction", "background",
        )
        if any(k in q for k in identity_keywords):
            for doc in retrieved:
                if doc.get("id") in (
                    "profile_intro", "elevator_pitch", "about_me",
                    "qa_001", "qa_006", "qa_007", "qa_092",
                ):
                    return self._extract_answer(doc["text"])
            for doc in retrieved:
                if doc.get("category") == "profile":
                    return self._extract_answer(doc["text"])

        for doc in retrieved:
            text = doc["text"]
            if text.lower().startswith("question:") and q in text.lower()[:120]:
                return self._extract_answer(text)

        return self._extract_answer(retrieved[0]["text"])

    def add_knowledge(
        self,
        text: str,
        category: str = "user_memory",
        reindex: bool = True,
    ) -> str:
        """Public helper: append to dataset and optionally rebuild index."""
        mem = get_memory()
        doc_id = mem.append_to_knowledge_base(text, category=category)
        if reindex:
            self.reload_index()
        return doc_id


# Singleton used by the Streamlit app
_rag_instance: Optional[AzeemRAG] = None


def get_rag() -> AzeemRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = AzeemRAG()
        _rag_instance.ensure_ready()
    return _rag_instance
