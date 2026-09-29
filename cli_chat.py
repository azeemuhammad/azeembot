"""
Simple CLI interface for AzeemBot (useful for testing without Streamlit).
Usage:  python cli_chat.py
"""

from rag_engine import get_rag


def main():
    print("=" * 55)
    print("  AzeemBot — Daniyal's personal RAG assistant")
    print("  Type 'quit' or 'exit' to leave")
    print("=" * 55)
    rag = get_rag()
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

        answer, sources = rag.generate(query, history=history)
        print(f"\nAzeemBot: {answer}")
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
