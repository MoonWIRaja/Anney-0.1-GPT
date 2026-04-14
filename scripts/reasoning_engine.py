"""
Reasoning Engine untuk Anney GPT.

Intercept soalan user dan enhance prompt sebelum masuk model:
  1. Detect jenis soalan (logik, emosi, fakta, pendapat)
  2. Chain-of-thought injection
  3. Knowledge lookup (dari taught + learned)
  4. Personality/mood context injection

Guna:
    from scripts.reasoning_engine import ReasoningEngine
    engine = ReasoningEngine()
    enhanced = engine.enhance(user_input, history)
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# =========================================================================== #
#  Question Type Detection                                                      #
# =========================================================================== #

class QuestionType:
    LOGIC = "logic"
    EMOTION = "emotion"
    FACT = "fact"
    OPINION = "opinion"
    GREETING = "greeting"
    PERSONAL = "personal"
    HOW_TO = "how_to"
    UNKNOWN = "unknown"


# Pattern matchers for each question type
QUESTION_PATTERNS = {
    QuestionType.LOGIC: {
        "keywords": [
            "kalau", "jika", "adakah", "patut", "boleh ke", "mungkin ke",
            "kenapa", "sebab", "logik", "masuk akal", "apa jadi",
            "if", "would", "could", "should", "why", "because",
            "betul ke", "salah ke", "logic", "reason", "mana satu",
            "lebih baik", "atau", "pilih", "compare", "beza",
        ],
        "patterns": [
            r"kalau\s.+\s(apa|macam mana|boleh|patut)",
            r"(adakah|betul ke|salah ke)\s",
            r"(kenapa|mengapa|sebab apa)\s",
            r"(lebih baik|mana satu|patut)\s.+\s(atau|ke)\s",
            r"apa (jadi|berlaku) (kalau|jika)",
        ],
    },
    QuestionType.EMOTION: {
        "keywords": [
            "sedih", "gembira", "marah", "takut", "bosan", "rindu",
            "sayang", "benci", "lonely", "happy", "sad", "angry",
            "mati", "putus", "gagal", "menang", "kahwin", "sakit",
            "menangis", "ketawa", "rasa", "perasaan", "hati",
            "kecewa", "stress", "depress", "anxious", "risau",
            "bunuh diri", "suicide", "hopeless", "give up",
        ],
        "patterns": [
            r"saya (rasa|feel)\s",
            r"(sedih|gembira|marah|takut|bosan|kecewa)",
            r"(mati|meninggal|passed away)",
            r"saya (tak boleh|tak tahan|tak mampu)",
        ],
    },
    QuestionType.FACT: {
        "keywords": [
            "apa itu", "siapa", "bila", "berapa", "di mana",
            "what is", "who is", "when", "how many", "where",
            "definisi", "maksud", "definition", "meaning",
            "sejarah", "history", "fakta", "fact",
        ],
        "patterns": [
            r"(apa (itu|tu)|what is)\s",
            r"(siapa|who)\s",
            r"(bila|when)\s.+(berlaku|happen)",
            r"berapa\s",
        ],
    },
    QuestionType.HOW_TO: {
        "keywords": [
            "macam mana", "bagaimana", "cara", "how to", "step",
            "tutorial", "ajar", "teach", "langkah", "guide",
        ],
        "patterns": [
            r"(macam mana|bagaimana|cara)\s(nak|untuk)",
            r"how (to|do|can)\s",
        ],
    },
    QuestionType.GREETING: {
        "keywords": [
            "hai", "hello", "hi", "hey", "assalamualaikum", "salam",
            "good morning", "selamat", "apa khabar", "how are you",
        ],
        "patterns": [
            r"^(hai|hello|hi|hey|salam)\b",
            r"^assalamualaikum",
            r"apa khabar",
        ],
    },
    QuestionType.PERSONAL: {
        "keywords": [
            "awak", "kau", "anney", "you", "nama awak",
            "umur awak", "siapa awak", "your name",
        ],
        "patterns": [
            r"(siapa|who)\s(awak|kau|you|anney)",
            r"(nama|name)\s(awak|kau|you)",
        ],
    },
}


def detect_question_type(text: str) -> str:
    """Detect jenis soalan dari teks input."""
    text_lower = text.lower().strip()

    scores = {}
    for qtype, config in QUESTION_PATTERNS.items():
        score = 0

        # Keyword matching
        for kw in config["keywords"]:
            if kw in text_lower:
                score += 2

        # Pattern matching (regex)
        for pattern in config["patterns"]:
            if re.search(pattern, text_lower):
                score += 5

        scores[qtype] = score

    # Return type with highest score
    if not scores or max(scores.values()) == 0:
        return QuestionType.UNKNOWN

    return max(scores, key=scores.get)


# =========================================================================== #
#  Chain-of-Thought Builder                                                     #
# =========================================================================== #

# CoT templates per question type
COT_TEMPLATES = {
    QuestionType.LOGIC: (
        "[FIKIR]: Ini soalan logik. Mari analisis step by step.\n"
        "Pertama, kenal pasti apa yang ditanya.\n"
        "Kedua, fikir sebab dan akibat.\n"
        "Ketiga, beri jawapan yang masuk akal.\n"
    ),
    QuestionType.EMOTION: (
        "[RASA]: Ini situasi emosi. Saya perlu empati.\n"
        "Pertama, faham perasaan user.\n"
        "Kedua, validate perasaan tu.\n"
        "Ketiga, beri sokongan yang sesuai.\n"
    ),
    QuestionType.FACT: (
        "[FAKTA]: Ini soalan fakta. Jawab dengan tepat.\n"
        "Berdasarkan maklumat yang saya tahu.\n"
    ),
    QuestionType.HOW_TO: (
        "[CARA]: Ini soalan cara/tutorial.\n"
        "Beri langkah-langkah yang jelas dan mudah diikuti.\n"
    ),
}


def build_chain_of_thought(question_type: str, user_input: str) -> str:
    """Bina chain-of-thought prefix berdasarkan jenis soalan."""
    template = COT_TEMPLATES.get(question_type, "")
    return template


# =========================================================================== #
#  Reasoning Engine                                                             #
# =========================================================================== #

class ReasoningEngine:
    """
    Engine pemikiran yang enhance prompt sebelum masuk model.

    Flow:
    1. Detect jenis soalan
    2. Cari maklumat berkaitan dari taught/learned knowledge
    3. Build chain-of-thought
    4. Inject personality/mood context
    5. Return enhanced prompt
    """

    def __init__(self):
        self._taught_storage = None
        self._learn_storage = None

    @property
    def taught_storage(self):
        """Lazy load TaughtStorage."""
        if self._taught_storage is None:
            try:
                from scripts.anney_fahamkan import TaughtStorage
                self._taught_storage = TaughtStorage()
            except ImportError:
                self._taught_storage = None
        return self._taught_storage

    @property
    def learn_storage(self):
        """Lazy load LearnStorage."""
        if self._learn_storage is None:
            try:
                from scripts.learn_storage import LearnStorage
                self._learn_storage = LearnStorage()
            except ImportError:
                self._learn_storage = None
        return self._learn_storage

    def enhance(
        self,
        user_input: str,
        history: list = None,
        mood: str = None,
        user_reputation: str = None,
    ) -> dict:
        """
        Enhance input user sebelum masuk model.

        Returns:
            dict with:
            - question_type: str
            - taught_answer: str or None (jawapan dari taught knowledge)
            - cot_prefix: str (chain-of-thought prefix)
            - context: str (konteks tambahan)
            - enhanced_prompt: str (prompt yang di-enhance)
            - mood_context: str
        """
        # 1. Detect question type
        qtype = detect_question_type(user_input)

        # 2. Search taught knowledge
        taught_answer = self._search_taught(user_input)

        # 3. Search learned knowledge
        learned_context = self._search_learned(user_input)

        # 4. Build chain-of-thought
        cot = build_chain_of_thought(qtype, user_input)

        # 5. Build mood context
        mood_ctx = ""
        if mood:
            mood_ctx = f"[MOOD: {mood}] "
        if user_reputation:
            mood_ctx += f"[USER: {user_reputation}] "

        # 6. Build enhanced prompt
        context_parts = []

        if cot:
            context_parts.append(cot)

        if learned_context:
            context_parts.append(f"[MAKLUMAT]: {learned_context[:500]}")

        if taught_answer:
            context_parts.append(f"[DIAJAR]: {taught_answer[:500]}")

        context = "\n".join(context_parts)

        # Build history context
        history_ctx = ""
        if history:
            recent = history[-3:]
            parts = []
            for turn in recent:
                parts.append(
                    f"[PENGGUNA]: {turn['pengguna']} [ANNEY]: {turn['anney']}"
                )
            history_ctx = " ".join(parts) + " "

        # Final enhanced prompt
        if context:
            enhanced = (
                f"{mood_ctx}{context}\n"
                f"{history_ctx}"
                f"[PENGGUNA]: {user_input} [ANNEY]: "
            )
        else:
            enhanced = (
                f"{mood_ctx}{history_ctx}"
                f"[PENGGUNA]: {user_input} [ANNEY]: "
            )

        return {
            "question_type": qtype,
            "taught_answer": taught_answer,
            "cot_prefix": cot,
            "context": context,
            "enhanced_prompt": enhanced,
            "mood_context": mood_ctx,
        }

    def _search_taught(self, query: str) -> str | None:
        """Cari dalam maklumat yang diajar."""
        if not self.taught_storage:
            return None
        try:
            results = self.taught_storage.search(query)
            if results:
                best = results[0]
                content = self.taught_storage.get_knowledge(best.get("slug", ""))
                if content:
                    # Extract info section
                    lines = content.split("\n")
                    info = []
                    in_info = False
                    for line in lines:
                        if line.strip() == "## Maklumat":
                            in_info = True
                            continue
                        if in_info and line.startswith("---"):
                            break
                        if in_info and line.strip():
                            info.append(line)
                    if info:
                        return "\n".join(info)
        except Exception:
            pass
        return None

    def _search_learned(self, query: str) -> str | None:
        """Cari dalam maklumat yang dipelajari dari internet."""
        if not self.learn_storage:
            return None
        try:
            index = self.learn_storage.get_index()
            query_lower = query.lower()
            for item in index:
                topic = item.get("topic", "").lower()
                if any(word in topic for word in query_lower.split() if len(word) > 3):
                    path = item.get("memory_path", "")
                    if path and Path(path).exists():
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                        # Extract summary
                        if "## Ringkasan" in content:
                            summary = content.split("## Ringkasan")[1]
                            summary = summary.split("##")[0].strip()
                            return summary[:500]
                        elif "## Summary" in content:
                            summary = content.split("## Summary")[1]
                            summary = summary.split("##")[0].strip()
                            return summary[:500]
        except Exception:
            pass
        return None
