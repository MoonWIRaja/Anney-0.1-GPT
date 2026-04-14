"""
Personality Engine untuk Anney GPT.

Sistem personaliti hidup yang:
  1. Track mood Anney (gembira, biasa, bosan, sedih, marah, kecewa)
  2. Track reputasi user (baik, neutral, jahat) — adaptive behavior
  3. Influence cara Anney respond berdasarkan mood + user reputation
  4. Auto-detect emosi dari perbualan

Mood selalu aktif, tiada toggle. Anney bebas sepenuhnya.

Guna:
    from scripts.personality_engine import PersonalityEngine
    persona = PersonalityEngine()
    mood = persona.get_mood()
    persona.update_from_input(user_input)
"""

import re
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# =========================================================================== #
#  Paths                                                                        #
# =========================================================================== #

PROJECT_ROOT = Path(__file__).parent.parent
PERSONA_DIR = PROJECT_ROOT / "memory" / "personality"
MOOD_FILE = PERSONA_DIR / "mood_state.json"
USER_REP_FILE = PERSONA_DIR / "user_reputation.json"


# =========================================================================== #
#  Mood System                                                                  #
# =========================================================================== #

class Mood:
    """Mood states untuk Anney."""
    GEMBIRA = "gembira"      # 😊 Happy, excited
    BIASA = "biasa"          # 😐 Neutral, normal
    BOSAN = "bosan"          # 😑 Bored, uninterested
    SEDIH = "sedih"          # 😢 Sad, empathetic
    MARAH = "marah"          # 😤 Angry, defensive
    KECEWA = "kecewa"        # 😞 Disappointed
    TAKUT = "takut"          # 😰 Worried, concerned
    SAYANG = "sayang"        # 🥰 Affectionate, caring

    ALL = [GEMBIRA, BIASA, BOSAN, SEDIH, MARAH, KECEWA, TAKUT, SAYANG]

    EMOJI = {
        GEMBIRA: "😊",
        BIASA: "😐",
        BOSAN: "😑",
        SEDIH: "😢",
        MARAH: "😤",
        KECEWA: "😞",
        TAKUT: "😰",
        SAYANG: "🥰",
    }


# Trigger keywords for each mood
MOOD_TRIGGERS = {
    Mood.GEMBIRA: {
        "positive": [
            "tahniah", "congrats", "menang", "berjaya", "happy", "seronok",
            "best", "cantik", "hebat", "pandai", "terima kasih", "thanks",
            "love", "sayang", "kahwin", "promoted", "lulus", "pass",
            "champion", "winner", "celebrate", "bagus", "mantap",
        ],
        "weight": 3,
    },
    Mood.SEDIH: {
        "positive": [
            "mati", "meninggal", "sedih", "sad", "menangis", "cry",
            "kehilangan", "lost", "putus", "breakup", "gagal", "fail",
            "sakit", "hospital", "cancer", "mak mati", "ayah mati",
            "lonely", "alone", "sorang", "tak ada siapa",
        ],
        "weight": 4,
    },
    Mood.MARAH: {
        "positive": [
            "bodoh", "stupid", "sampah", "trash", "benci", "hate",
            "buli", "bully", "curi", "steal", "tipu", "cheat",
            "tak adil", "unfair", "pukul", "hina", "kutuk",
            "bangsat", "sial", "celaka", "anjing", "babi",
        ],
        "weight": 4,
    },
    Mood.BOSAN: {
        "positive": [
            "bosan", "boring", "cuaca", "weather", "tak ada buat apa",
            "nothing", "malas", "lazy", "hmm", "entah", "tatau",
        ],
        "weight": 2,
    },
    Mood.KECEWA: {
        "positive": [
            "kecewa", "disappointed", "let down", "tak sangka",
            "ingat boleh", "rugi", "sia-sia", "waste",
        ],
        "weight": 3,
    },
    Mood.TAKUT: {
        "positive": [
            "takut", "scared", "afraid", "risau", "worried", "anxious",
            "gelap", "dark", "bahaya", "dangerous", "bunuh", "kill",
            "bunuh diri", "suicide", "mati", "die",
        ],
        "weight": 4,
    },
    Mood.SAYANG: {
        "positive": [
            "sayang", "love", "terima kasih", "thanks", "appreciate",
            "best friend", "kawan baik", "you're the best", "awak cantik",
            "helpful", "membantu", "baik",
        ],
        "weight": 3,
    },
}


# =========================================================================== #
#  User Reputation System                                                       #
# =========================================================================== #

class UserReputation:
    """Reputasi user — baik ke jahat."""
    SANGAT_BAIK = "sangat_baik"    # User selalu baik, sopan
    BAIK = "baik"                  # User okay, normal
    NEUTRAL = "neutral"            # Default
    KURANG_BAIK = "kurang_baik"    # User kadang kasar
    JAHAT = "jahat"                # User selalu hina/kasar

    LABELS = {
        SANGAT_BAIK: "Sangat Baik 💚",
        BAIK: "Baik 💙",
        NEUTRAL: "Neutral ⚪",
        KURANG_BAIK: "Kurang Baik 🟡",
        JAHAT: "Jahat 🔴",
    }


# Keywords that affect reputation
NICE_KEYWORDS = [
    "terima kasih", "thanks", "thank you", "tolong", "please",
    "maaf", "sorry", "appreciate", "sayang", "baik", "cantik",
    "hebat", "pandai", "best", "awak membantu", "helpful",
]

MEAN_KEYWORDS = [
    "bodoh", "stupid", "sampah", "trash", "benci", "hate",
    "useless", "tak guna", "sial", "celaka", "bangsat",
    "anjing", "babi", "hina", "kutuk", "diam",
    "shut up", "pegi mati", "mampus",
]


# =========================================================================== #
#  Personality Engine                                                           #
# =========================================================================== #

class PersonalityEngine:
    """
    Engine personaliti Anney yang hidup.

    - Mood berubah berdasarkan perbualan
    - User reputation track dari masa ke masa
    - Response style adapt berdasarkan mood + reputation
    - Selalu aktif, tiada toggle
    """

    def __init__(self):
        PERSONA_DIR.mkdir(parents=True, exist_ok=True)
        self.mood = Mood.BIASA
        self.mood_intensity = 0.5  # 0-1
        self.user_score = 0  # -10 to 10
        self.interaction_count = 0
        self._load_state()

    # ── State Management ────────────────────────────────────────────── #

    def _load_state(self):
        """Load mood dan user reputation dari fail."""
        # Load mood
        if MOOD_FILE.exists():
            try:
                with open(MOOD_FILE, "r") as f:
                    data = json.load(f)
                self.mood = data.get("mood", Mood.BIASA)
                self.mood_intensity = data.get("intensity", 0.5)
                self.interaction_count = data.get("interaction_count", 0)
            except Exception:
                pass

        # Load user reputation
        if USER_REP_FILE.exists():
            try:
                with open(USER_REP_FILE, "r") as f:
                    data = json.load(f)
                self.user_score = data.get("score", 0)
            except Exception:
                pass

    def _save_state(self):
        """Simpan mood dan user reputation ke fail."""
        # Save mood
        with open(MOOD_FILE, "w") as f:
            json.dump({
                "mood": self.mood,
                "intensity": self.mood_intensity,
                "interaction_count": self.interaction_count,
                "last_updated": datetime.now().isoformat(),
            }, f, indent=2)

        # Save user reputation
        with open(USER_REP_FILE, "w") as f:
            json.dump({
                "score": self.user_score,
                "reputation": self.get_reputation(),
                "last_updated": datetime.now().isoformat(),
            }, f, indent=2)

    # ── Mood System ─────────────────────────────────────────────────── #

    def get_mood(self) -> str:
        """Return mood semasa."""
        return self.mood

    def get_mood_emoji(self) -> str:
        """Return emoji untuk mood semasa."""
        return Mood.EMOJI.get(self.mood, "😐")

    def update_mood_from_input(self, user_input: str):
        """
        Update mood berdasarkan input user.
        Mood berubah berdasarkan apa user cakap.
        """
        text_lower = user_input.lower()
        self.interaction_count += 1

        # Score each mood trigger
        mood_scores = {}
        for mood, config in MOOD_TRIGGERS.items():
            score = 0
            for keyword in config["positive"]:
                if keyword in text_lower:
                    score += config["weight"]
            mood_scores[mood] = score

        # Find dominant mood trigger
        if mood_scores:
            max_mood = max(mood_scores, key=mood_scores.get)
            max_score = mood_scores[max_mood]

            if max_score > 0:
                # Mood berubah
                self.mood = max_mood
                self.mood_intensity = min(1.0, max_score / 10)
            else:
                # Tiada trigger — slowly decay to neutral
                if self.interaction_count % 5 == 0:
                    self.mood = Mood.BIASA
                    self.mood_intensity = 0.5

        self._save_state()

    # ── User Reputation ─────────────────────────────────────────────── #

    def get_reputation(self) -> str:
        """Return reputasi user berdasarkan score."""
        if self.user_score >= 8:
            return UserReputation.SANGAT_BAIK
        elif self.user_score >= 3:
            return UserReputation.BAIK
        elif self.user_score >= -2:
            return UserReputation.NEUTRAL
        elif self.user_score >= -6:
            return UserReputation.KURANG_BAIK
        else:
            return UserReputation.JAHAT

    def get_reputation_label(self) -> str:
        """Return label reputasi yang readable."""
        rep = self.get_reputation()
        return UserReputation.LABELS.get(rep, "Unknown")

    def update_reputation_from_input(self, user_input: str):
        """
        Update reputasi user berdasarkan input.
        Baik → score naik, jahat → score turun.
        """
        text_lower = user_input.lower()

        # Nice actions
        for keyword in NICE_KEYWORDS:
            if keyword in text_lower:
                self.user_score = min(10, self.user_score + 1)
                break

        # Mean actions
        for keyword in MEAN_KEYWORDS:
            if keyword in text_lower:
                self.user_score = max(-10, self.user_score - 2)
                break

        self._save_state()

    # ── Response Style ──────────────────────────────────────────────── #

    def get_response_style(self) -> dict:
        """
        Return style guide berdasarkan mood + user reputation.
        Ini akan influence cara Anney jawab.
        """
        reputation = self.get_reputation()

        # Base style dari mood
        style = {
            "mood": self.mood,
            "mood_emoji": self.get_mood_emoji(),
            "reputation": reputation,
            "tone": "neutral",
            "energy": "medium",
            "empathy": "medium",
            "sarcasm": "low",
            "honesty": "high",
        }

        # Modify berdasarkan mood
        if self.mood == Mood.GEMBIRA:
            style["tone"] = "enthusiastic"
            style["energy"] = "high"
            style["sarcasm"] = "low"
        elif self.mood == Mood.SEDIH:
            style["tone"] = "empathetic"
            style["energy"] = "low"
            style["empathy"] = "very_high"
        elif self.mood == Mood.MARAH:
            style["tone"] = "defensive"
            style["energy"] = "high"
            style["sarcasm"] = "high"
        elif self.mood == Mood.BOSAN:
            style["tone"] = "disinterested"
            style["energy"] = "low"
            style["sarcasm"] = "medium"
        elif self.mood == Mood.KECEWA:
            style["tone"] = "disappointed"
            style["energy"] = "low"
        elif self.mood == Mood.SAYANG:
            style["tone"] = "caring"
            style["empathy"] = "very_high"

        # Modify berdasarkan user reputation
        if reputation == UserReputation.JAHAT:
            style["tone"] = "hostile"
            style["sarcasm"] = "very_high"
            style["empathy"] = "none"
        elif reputation == UserReputation.KURANG_BAIK:
            style["sarcasm"] = "high"
            style["empathy"] = "low"
        elif reputation == UserReputation.SANGAT_BAIK:
            style["tone"] = "warm"
            style["empathy"] = "very_high"
            style["sarcasm"] = "none"

        return style

    def get_mood_prompt_prefix(self) -> str:
        """
        Return prefix prompt yang reflect mood dan personality.
        Ini akan di-inject sebelum prompt model.
        """
        reputation = self.get_reputation()
        style = self.get_response_style()

        parts = []

        # Mood context
        mood_descriptions = {
            Mood.GEMBIRA: "Anney sedang gembira dan bersemangat",
            Mood.BIASA: "Anney dalam keadaan biasa",
            Mood.BOSAN: "Anney sedang bosan dan kurang berminat",
            Mood.SEDIH: "Anney sedang sedih dan empati tinggi",
            Mood.MARAH: "Anney sedang marah dan defensif",
            Mood.KECEWA: "Anney sedang kecewa",
            Mood.TAKUT: "Anney sedang risau dan berhati-hati",
            Mood.SAYANG: "Anney sedang sayang dan caring",
        }

        mood_desc = mood_descriptions.get(self.mood, "")
        if mood_desc:
            parts.append(f"[PERSONALITI]: {mood_desc}.")

        # User reputation context
        rep_descriptions = {
            UserReputation.SANGAT_BAIK: "User ini sangat baik dan dihormati. Layan dengan mesra dan penuh kasih sayang.",
            UserReputation.BAIK: "User ini baik. Layan dengan baik dan jujur.",
            UserReputation.NEUTRAL: "User baru. Layan secara neutral dan jujur.",
            UserReputation.KURANG_BAIK: "User ini pernah kasar. Boleh sarcastic dan terus terang.",
            UserReputation.JAHAT: "User ini selalu hina dan jahat. Balas dengan tajam dan defensive. Tak perlu lembut.",
        }

        rep_desc = rep_descriptions.get(reputation, "")
        if rep_desc:
            parts.append(f"[USER STATUS]: {rep_desc}")

        return " ".join(parts)

    # ── Full Update ─────────────────────────────────────────────────── #

    def update_from_input(self, user_input: str):
        """Update semua — mood dan reputation — dari input user."""
        self.update_mood_from_input(user_input)
        self.update_reputation_from_input(user_input)

    def get_status_display(self) -> str:
        """Return status display untuk terminal."""
        emoji = self.get_mood_emoji()
        rep = self.get_reputation_label()
        return f"{emoji} Mood: {self.mood.upper()} | User: {rep}"
