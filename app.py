"""
AzeemBot — Personal RAG Chatbot for Muhammad Daniyal Azeem
Professional Streamlit interface with:
  - Conversation history
  - Persistent memory (DOB, name, facts, standing instructions)
  - Google Gemini (default) + DeepSeek / OpenAI support
  - Retrieval transparency
"""

import os
import streamlit as st

st.set_page_config(
    page_title="AzeemBot | Daniyal's AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 0.6rem 0 0.3rem 0;
    }
    .main-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #64748b;
        font-size: 0.95rem;
        margin: 0;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
    .source-card {
        background: #f1f5f9;
        border-left: 3px solid #3b82f6;
        padding: 0.55rem 0.75rem;
        margin: 0.35rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #334155;
    }
    .memory-badge {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 0.35rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 0.2rem 0;
        display: inline-block;
    }
    .footer-note {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 1.5rem;
    }
    .hint-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        font-size: 0.85rem;
        color: #0c4a6e;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AzeemBot knowledge base…")
def load_rag():
    from rag_engine import get_rag
    return get_rag()


def get_mem():
    from memory import get_memory
    return get_memory()


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hey! I'm **AzeemBot** — Daniyal's personal assistant.\n\n"
                    "Ask me anything about his background, skills, projects, education, or contact.\n\n"
                ),
            }
        ]
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False


PROVIDER_OPTIONS = {
    "Google Gemini (recommended)": {
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.8-flash",
        "key_env": "GEMINI_API_KEY",
        "key_label": "Gemini API Key",
    },
    "DeepSeek": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
        "key_label": "DeepSeek API Key",
    },
    "OpenAI": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "key_label": "OpenAI API Key",
    },
}


def sidebar():
    mem = get_mem()
    with st.sidebar:
        st.markdown("### About AzeemBot")
        st.markdown(
            "Personal RAG chatbot built on Daniyal's own knowledge base "
            "(profile, projects, skills, 120+ Q&A pairs) **+ your private memory**."
        )
        st.markdown("---")
        st.markdown("**Muhammad Daniyal Azeem**")
        st.caption("AI & Flutter Developer · Lahore, Pakistan")
        st.caption("BS Artificial Intelligence @ UMT (2023–Present)")
        st.markdown("")
        st.markdown("📧 azeemuhammad47@gmail.com")
        st.markdown("📱 +92 300 6868447")
        st.markdown("🔗 [GitHub](https://github.com/azeemuhammad) · [Portfolio](https://azeemuhammad.github.io)")
        st.markdown("---")

        # ---- Memory panel ----
        st.markdown("### 🧠 Your Memory")
        profile = mem.get_all_profile()
        facts = mem.data.get("facts") or []
        instructions = mem.data.get("instructions") or []

        if profile:
            for k, v in profile.items():
                st.markdown(
                    f'<div class="memory-badge"><b>{k.replace("_", " ").title()}</b>: {v}</div>',
                    unsafe_allow_html=True,
                )
        if facts:
            with st.expander(f"Facts ({len(facts)})", expanded=False):
                for f in facts[-10:]:
                    st.caption(f"• {f}")
        if instructions:
            with st.expander(f"Instructions ({len(instructions)})", expanded=False):
                for i in instructions:
                    st.caption(f"• {i}")

        if not profile and not facts and not instructions:
            st.caption("Nothing stored yet. Try: “remember my DOB is 15/03/2003”")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear memory", use_container_width=True):
                mem.clear_all()
                st.rerun()
        with col2:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Conversation cleared. What would you like to know?",
                    }
                ]
                st.rerun()

        st.markdown("---")

        # ---- LLM provider (Gemini default) ----
        st.markdown("### 🔑 LLM Provider")
        provider_label = st.selectbox(
            "Provider",
            options=list(PROVIDER_OPTIONS.keys()),
            index=0,
            help="Gemini is free-tier friendly. Get a key at https://aistudio.google.com/apikey",
        )
        cfg = PROVIDER_OPTIONS[provider_label]

        default_key = (
            os.getenv(cfg["key_env"], "")
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
            or os.getenv("DEEPSEEK_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
        api_key = st.text_input(
            cfg["key_label"] + " (optional)",
            type="password",
            help="Leave empty for retrieval-only mode. With a key, answers become more natural.",
            value=default_key,
        )
        gemini_models = [
            "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]
        if cfg["provider"] == "gemini":
            model_name = st.selectbox(
                "Model name",
                options=gemini_models,
                index=0,
                help="If one model returns 404, try another (e.g. gemini-2.5-flash).",
            )
        else:
            model_name = st.text_input("Model name", value=cfg["model"])
        base_url = cfg["base_url"]

        st.session_state.show_sources = st.checkbox("Show retrieved sources", value=False)

        st.markdown("---")
        st.caption("RAG · Memory · Gemini / FAISS · all-MiniLM-L6-v2")
        return api_key, base_url, model_name, cfg["provider"]


def main():
    init_session()
    api_key, base_url, model_name, provider = sidebar()
    mem = get_mem()

    st.markdown(
        """
        <div class="main-header">
            <h1>AzeemBot</h1>
            <p>Personal AI assistant of Muhammad Daniyal Azeem</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hint-box">'
        "💡 <b>Tip:</b> I can store info permanently. Try "
        "“Store my DOB as 15/03/2003”, “Call me Boss”, or “Always reply in Urdu”. "
        "Use a <b>Gemini API key</b> (sidebar) for natural answers — free at "
        "<a href='https://aistudio.google.com/apikey' target='_blank'>Google AI Studio</a>."
        "</div>",
        unsafe_allow_html=True,
    )

    rag = load_rag()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and st.session_state.show_sources:
                with st.expander("Sources used"):
                    for s in msg["sources"]:
                        score = s.get("score", 0)
                        cat = s.get("category", "")
                        preview = s.get("text", "")[:160] + ("…" if len(s.get("text", "")) > 160 else "")
                        st.markdown(
                            f'<div class="source-card"><b>{cat}</b> · score {score:.2f}<br>{preview}</div>',
                            unsafe_allow_html=True,
                        )

    if prompt := st.chat_input("Ask about Daniyal, or tell me something to remember…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                history_for_rag = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                answer, retrieved = rag.generate(
                    query=prompt,
                    history=history_for_rag,
                    api_key=api_key if api_key else None,
                    base_url=base_url,
                    model=model_name,
                    memory=mem,
                    provider=provider,
                )
                st.markdown(answer)

                if st.session_state.show_sources and retrieved:
                    with st.expander("Sources used"):
                        for s in retrieved:
                            score = s.get("score", 0)
                            cat = s.get("category", "")
                            preview = s.get("text", "")[:160] + ("…" if len(s.get("text", "")) > 160 else "")
                            st.markdown(
                                f'<div class="source-card"><b>{cat}</b> · score {score:.2f}<br>{preview}</div>',
                                unsafe_allow_html=True,
                            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": retrieved if st.session_state.show_sources else None,
            }
        )
        st.rerun()

    st.markdown(
        '<p class="footer-note">Built with RAG + Persistent Memory · Gemini / FAISS · all-MiniLM-L6-v2</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
