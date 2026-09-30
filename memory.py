"""
AzeemBot Persistent Memory
Stores user facts, preferences, and instructions so the bot can remember
and follow them across sessions (e.g. DOB, preferred name, custom rules).
Also supports appending new knowledge into the main JSONL dataset.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MEMORY_PATH = Path(__file__).parent / "user_memory.json"
KNOWLEDGE_PATH = Path(__file__).parent / "daniyal_azeem_chatbot_knowledge.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MemoryStore:
    """Simple durable key-value + facts store for AzeemBot."""

    def __init__(self, path: Path = MEMORY_PATH):
        self.path = path
        self.data: Dict[str, Any] = {
            "profile": {},          # fixed keys: name, dob, email, preferred_name, ...
            "facts": [],            # free-form remembered facts
            "instructions": [],     # standing instructions the bot must follow
            "history_notes": [],    # short notes from past turns (optional)
            "updated_at": None,
        }
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        self.data["updated_at"] = _now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Profile helpers
    # ------------------------------------------------------------------
    def set_profile(self, key: str, value: str) -> None:
        key = key.strip().lower().replace(" ", "_")
        self.data.setdefault("profile", {})[key] = value.strip()
        self.save()

    def get_profile(self, key: str, default: str = "") -> str:
        return self.data.get("profile", {}).get(key.strip().lower().replace(" ", "_"), default)

    def get_all_profile(self) -> Dict[str, str]:
        return dict(self.data.get("profile", {}))

    # ------------------------------------------------------------------
    # Free-form facts & instructions
    # ------------------------------------------------------------------
    def add_fact(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        facts: List[str] = self.data.setdefault("facts", [])
        # avoid exact duplicates
        if fact not in facts:
            facts.append(fact)
            # keep last 100 facts
            self.data["facts"] = facts[-100:]
            self.save()

    def add_instruction(self, instruction: str) -> None:
        instruction = instruction.strip()
        if not instruction:
            return
        inst: List[str] = self.data.setdefault("instructions", [])
        if instruction not in inst:
            inst.append(instruction)
            self.data["instructions"] = inst[-50:]
            self.save()

    def clear_facts(self) -> None:
        self.data["facts"] = []
        self.save()

    def clear_instructions(self) -> None:
        self.data["instructions"] = []
        self.save()

    def clear_all(self) -> None:
        self.data = {
            "profile": {},
            "facts": [],
            "instructions": [],
            "history_notes": [],
            "updated_at": None,
        }
        self.save()

    # ------------------------------------------------------------------
    # Intent detection: store / update / retrieve from natural language
    # ------------------------------------------------------------------
    def process_user_message(self, message: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect memory-related intents.
        Returns (action_summary, response_hint) or (None, None) if nothing to store.
        action_summary is a short human-readable confirmation.
        """
        text = message.strip()
        lower = text.lower()

        # --- Standing instructions ---
        # e.g. "always answer in urdu", "from now on call me Boss"
        inst_patterns = [
            r"(?:please\s+)?(?:always|from now on|make sure to|remember to|you should always)\s+(.+)",
            r"(?:rule|instruction)\s*:\s*(.+)",
        ]
        for pat in inst_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                instr = m.group(1).strip().rstrip(".")
                if len(instr) > 3:
                    self.add_instruction(instr)
                    return f"Instruction saved: “{instr}”", (
                        f"Got it. I will follow this from now on: **{instr}**"
                    )

        # --- Preferred name / call me ---
        name_m = re.search(
            r"(?:call me|my (?:preferred )?name is|change my name to|set my name to|mera naam)\s+([A-Za-z\u0600-\u06FF\s]{2,40})",
            text,
            re.IGNORECASE,
        )
        if name_m:
            name = name_m.group(1).strip()
            self.set_profile("preferred_name", name)
            self.add_fact(f"User prefers to be called {name}")
            return f"Preferred name set to {name}", f"Sure! I’ll call you **{name}** from now on."

        # --- Date of birth ---
        dob_patterns = [
            r"(?:my |mera )?(?:date of birth|dob|birthday|birth date|paidaish|date of birth)\s*(?:is|=|:)?\s*([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4}|[0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{2,4})",
            r"(?:store|save|remember)\s+(?:my )?(?:date of birth|dob|birthday)\s*(?:as|=|:)?\s*(.+)",
            r"(?:dob|birthday)\s*(?:ko )?(?:store|save|yaad)\s*(?:karo|rakho)?\s*(?:as|=|:)?\s*(.+)",
        ]
        for pat in dob_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                dob = m.group(1).strip().rstrip(".")
                self.set_profile("dob", dob)
                self.add_fact(f"User's date of birth is {dob}")
                return f"DOB stored: {dob}", f"Saved. Your date of birth is **{dob}**. Ask me anytime and I’ll tell you."

        # --- Generic "remember / store / save that ..." ---
        remember_m = re.search(
            r"(?:remember|store|save|yaad rakh|yaad rakho|store karo|save karo)\s+(?:that\s+|this\s+|ye\s+|k\s+)?(.+)",
            text,
            re.IGNORECASE,
        )
        if remember_m:
            fact = remember_m.group(1).strip().rstrip(".")
            # skip pure questions
            if "?" not in fact and len(fact) > 4:
                # try to parse "X is Y" into profile
                kv = re.match(r"(?:my |mera )?([a-zA-Z_\s]{2,30})\s+(?:is|=|:)\s+(.+)", fact, re.IGNORECASE)
                if kv:
                    key = kv.group(1).strip().lower().replace(" ", "_")
                    val = kv.group(2).strip()
                    if key in ("name", "full_name", "email", "phone", "city", "location", "age", "dob", "birthday"):
                        self.set_profile(key, val)
                    self.add_fact(f"{key.replace('_', ' ').title()}: {val}")
                    return f"Saved {key}: {val}", f"Got it — I’ve stored **{key.replace('_', ' ')}** as **{val}**."
                self.add_fact(fact)
                return f"Fact saved: {fact[:60]}", f"Remembered: **{fact}**"

        # --- Explicit key=value or "my X is Y" ---
        kv_m = re.search(
            r"(?:my |mera )?([a-zA-Z_][a-zA-Z0-9_\s]{1,25})\s+(?:is|=|:)\s+([^\n?]{2,80})",
            text,
            re.IGNORECASE,
        )
        if kv_m and any(w in lower for w in ("store", "save", "remember", "set", "update", "change", "yaad", "rakho")):
            key = kv_m.group(1).strip().lower().replace(" ", "_")
            val = kv_m.group(2).strip().rstrip(".")
            self.set_profile(key, val)
            self.add_fact(f"{key.replace('_', ' ').title()}: {val}")
            return f"Saved {key}={val}", f"Updated. **{key.replace('_', ' ').title()}** is now **{val}**."

        return None, None

    def answer_from_memory(self, query: str) -> Optional[str]:
        """If the user is asking for something we already stored, answer directly."""
        q = query.lower().strip()

        # Preferred name
        if any(p in q for p in ("what is my name", "mera naam", "call me", "preferred name", "my name")):
            name = self.get_profile("preferred_name") or self.get_profile("name")
            if name:
                return f"Your preferred name is **{name}**."

        # DOB
        if any(p in q for p in ("date of birth", "dob", "birthday", "paidaish", "birth date", "mera dob")):
            dob = self.get_profile("dob") or self.get_profile("birthday")
            if dob:
                return f"Your date of birth is **{dob}**."

        # Generic profile lookup: "what is my X"
        m = re.search(r"(?:what is|what's|tell me|batao|kya hai)\s+(?:my |mera )?([a-zA-Z_\s]{2,30})\??", q)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            val = self.get_profile(key)
            if val:
                return f"Your **{key.replace('_', ' ')}** is **{val}**."

        # Search facts
        for fact in self.data.get("facts", []):
            if any(tok in fact.lower() for tok in q.split() if len(tok) > 3):
                # weak match — only if query is clearly retrieval
                if any(w in q for w in ("what", "tell", "batao", "kya", "remember", "yaad")):
                    return f"From memory: **{fact}**"

        return None

    # ------------------------------------------------------------------
    # Context injection for the LLM
    # ------------------------------------------------------------------
    def to_prompt_block(self) -> str:
        lines: List[str] = []
        profile = self.get_all_profile()
        if profile:
            lines.append("User profile (remembered):")
            for k, v in profile.items():
                lines.append(f"  - {k.replace('_', ' ').title()}: {v}")
        facts = self.data.get("facts") or []
        if facts:
            lines.append("Remembered facts:")
            for f in facts[-15:]:
                lines.append(f"  - {f}")
        instructions = self.data.get("instructions") or []
        if instructions:
            lines.append("Standing instructions you MUST follow:")
            for i in instructions:
                lines.append(f"  - {i}")
        if not lines:
            return ""
        return "=== USER MEMORY (always respect) ===\n" + "\n".join(lines) + "\n=== END MEMORY ===\n"

    # ------------------------------------------------------------------
    # Optionally append a new knowledge chunk into the main dataset
    # ------------------------------------------------------------------
    def append_to_knowledge_base(
        self,
        text: str,
        category: str = "user_memory",
        doc_id: Optional[str] = None,
    ) -> str:
        """Append a new line to the JSONL knowledge file. Returns the id used."""
        if not text.strip():
            raise ValueError("Empty text")
        doc_id = doc_id or f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        obj = {
            "id": doc_id,
            "text": " ".join(text.strip().split()),
            "category": category,
        }
        with open(KNOWLEDGE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return doc_id


# Singleton
_memory: Optional[MemoryStore] = None


def get_memory() -> MemoryStore:
    global _memory
    if _memory is None:
        _memory = MemoryStore()
    return _memory
