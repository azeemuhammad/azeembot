"""
Simple CLI interface for AzeemBot (useful for testing without Streamlit).
Supports persistent memory + Gemini / DeepSeek / OpenAI.
Usage:
  export GEMINI_API_KEY=your_key
  python cli_chat.py
"""

import os
from rag_engine import get_rag
from memory import get_memory


def main():
    print("=" * 55)
    print("  AzeemBot — Daniyal's personal RAG assistant")
    print("  Memory + Gemini enabled")
    print("  Type 'quit' or 'exit' to leave")
    print("  Examples:")
    print("    Store my DOB as 15/03/2003")
    print("    Call me Boss")
    print("    What is my date of birth?")
    print("    Always answer in short points")
    print("=" * 55)

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if api_key:
        print("  API key detected — LLM mode ON")
    else:
        print("  No API key — retrieval + memory only")
        print("  Set GEMINI_API_KEY for natural answers")

    rag = get_rag()
    mem = get_memory()
    history = []

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if query.lower() in ("memory", "show memory"):
            print("\n--- Memory ---")
            print("Profile:", mem.get_all_profile())
            print("Facts:", mem.data.get("facts"))
            print("Instructions:", mem.data.get("instructions"))
            continue
        if query.lower() in ("clear memory",):
            mem.clear_all()
            print("Memory cleared.")
            continue

        answer, sources = rag.generate(
            query,
            history=history,
            memory=mem,
            api_key=api_key,
            provider="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-2.0-flash",
        )
        print(f"\nAzeemBot: {answer}")
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
