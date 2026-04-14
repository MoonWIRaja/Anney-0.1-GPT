"""
AnneyBelajar — Modul Pembelajaran Kendiri dari Internet.

Modul utama yang menggabungkan semua komponen:
  1. WebResearcher — cari maklumat dari internet
  2. KnowledgeProcessor — proses dan ringkaskan secara neutral
  3. LearnStorage — simpan ke memori dan training data

Boleh digunakan sebagai:
  - Command CLI: `/AnneyBelajar <topik>`
  - Standalone:  python scripts/anney_belajar.py "quantum computing"
  - Import:      from scripts.anney_belajar import AnneyBelajar

Guna:
    python scripts/anney_belajar.py "sejarah Melaka"
    python scripts/anney_belajar.py "quantum computing" --lang en
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.web_research import WebResearcher
from scripts.knowledge_processor import KnowledgeProcessor
from scripts.learn_storage import LearnStorage


# =========================================================================== #
#  AnneyBelajar                                                                 #
# =========================================================================== #

class AnneyBelajar:
    """
    Sistem pembelajaran kendiri Anney.

    Flow:
    1. Terima topik dari pengguna
    2. Cari maklumat dari internet (percuma, open source)
    3. Proses dan ringkaskan secara neutral
    4. Simpan ke memori dan training data
    5. Paparkan laporan pembelajaran
    """

    def __init__(self):
        self.researcher = WebResearcher()
        self.processor = KnowledgeProcessor()
        self.storage = LearnStorage()

    def detect_language(self, text: str) -> str:
        """
        Detect bahasa input — BM atau EN.
        Simple heuristic based on common words.
        """
        text_lower = text.lower()

        malay_markers = [
            "apa", "ini", "itu", "dan", "yang", "dalam", "untuk",
            "saya", "kita", "dia", "adalah", "ada", "tidak",
            "bagaimana", "kenapa", "macam", "belajar", "tentang",
            "sejarah", "cara", "boleh", "perlu", "hendak", "mahu",
        ]

        en_markers = [
            "what", "the", "and", "how", "why", "about", "learn",
            "this", "that", "for", "with", "from", "into", "have",
            "does", "can", "should", "would", "which", "where",
        ]

        ms_score = sum(1 for w in malay_markers if w in text_lower.split())
        en_score = sum(1 for w in en_markers if w in text_lower.split())

        # Kalau ada karakter khas BM
        if any(c in text_lower for c in ["ñ"]):
            ms_score += 2

        return "ms" if ms_score >= en_score else "en"

    def belajar(
        self,
        topic: str,
        lang: str = None,
        verbose: bool = True,
    ) -> dict:
        """
        Jalankan proses pembelajaran untuk topik tertentu.

        Args:
            topic:   Topik yang hendak dipelajari
            lang:    Bahasa output (None = auto-detect)
            verbose: Tunjuk progress di terminal

        Returns:
            dict dengan:
              - report: KnowledgeReport
              - paths: dict dengan paths yang disimpan
              - terminal_report: str untuk paparan terminal
        """
        # ── 0. Detect bahasa ─────────────────────────────────────── #
        if lang is None:
            lang = self.detect_language(topic)

        if verbose:
            if lang == "ms":
                print(f"\n  🎓 Anney sedang belajar tentang '{topic}'...\n")
            else:
                print(f"\n  🎓 Anney is learning about '{topic}'...\n")

        # ── 1. Cari maklumat ─────────────────────────────────────── #
        results = self.researcher.research(
            topic=topic,
            request_lang=lang,
            max_sources=10,
            verbose=verbose,
        )

        if not results:
            if verbose:
                if lang == "ms":
                    print("\n  ❌ Tiada maklumat dijumpai. Pastikan ada sambungan internet.")
                else:
                    print("\n  ❌ No information found. Please check your internet connection.")
            return {"report": None, "paths": {}, "terminal_report": ""}

        # ── 2. Proses maklumat ───────────────────────────────────── #
        report = self.processor.process(
            results=results,
            topic=topic,
            lang=lang,
            verbose=verbose,
        )

        # ── 3. Simpan ke memori dan training data ────────────────── #
        if verbose:
            print("  💾 Menyimpan ke memori...", flush=True)

        paths = self.storage.save(report, verbose=verbose)

        # ── 4. Jana terminal report ──────────────────────────────── #
        terminal_report = report.to_terminal_report()

        if verbose:
            # Tambah info simpanan
            rel_memory = paths.get("memory", "").replace(str(Path(__file__).parent.parent) + "/", "")
            terminal_report += f"\n  💾 Disimpan ke: {rel_memory}"
            terminal_report += "\n  📖 Training data dikemaskini untuk retrain model"
            terminal_report += "\n  " + "═" * 50

            print(terminal_report)

        return {
            "report": report,
            "paths": paths,
            "terminal_report": terminal_report,
        }

    def senarai_ilmu(self) -> list[str]:
        """Senarai semua topik yang telah dipelajari."""
        return self.storage.get_learned_topics()

    def sudah_belajar(self, topic: str) -> bool:
        """Semak kalau topik sudah pernah dipelajari."""
        return self.storage.has_learned(topic)


# =========================================================================== #
#  CLI handler — dipanggil dari chat.py                                         #
# =========================================================================== #

def handle_anney_belajar_command(user_input: str) -> None:
    """
    Handler untuk command /AnneyBelajar dalam CLI chat.

    Args:
        user_input: Input penuh pengguna (contoh: "/AnneyBelajar quantum computing")
    """
    # Parse topik dari command
    parts = user_input.split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():
        print("\n  ⚠ Sila nyatakan topik yang hendak dipelajari.")
        print("  Contoh: /AnneyBelajar sejarah Melaka")
        print("  Contoh: /AnneyBelajar quantum computing")
        return

    topic = parts[1].strip()

    # Detect bahasa dari topik + context
    anney = AnneyBelajar()
    lang = anney.detect_language(topic)

    # Semak kalau sudah pernah belajar
    if anney.sudah_belajar(topic):
        if lang == "ms":
            print(f"\n  ℹ Anney sudah pernah belajar tentang '{topic}'.")
            print("  Mengemaskini maklumat...\n")
        else:
            print(f"\n  ℹ Anney has already learned about '{topic}'.")
            print("  Updating information...\n")

    # Jalankan pembelajaran
    result = anney.belajar(topic=topic, lang=lang, verbose=True)

    if result["report"] is None:
        return

    # Paparkan nota akhir
    report = result["report"]
    if lang == "ms":
        print(f"\n  ✅ Anney telah selesai belajar tentang '{topic}'!")
        print(f"  📊 {len(report.key_facts)} fakta utama dikumpulkan dari {len(report.sources)} sumber")
        print(f"  🧠 Ilmu ini akan digunakan untuk meningkatkan model Anney pada retrain seterusnya")
    else:
        print(f"\n  ✅ Anney has finished learning about '{topic}'!")
        print(f"  📊 {len(report.key_facts)} key facts gathered from {len(report.sources)} sources")
        print(f"  🧠 This knowledge will be used to improve Anney's model on the next retrain")

    print()


def handle_senarai_ilmu_command() -> None:
    """Handler untuk command /SenaraiIlmu — senarai semua yang telah dipelajari."""
    storage = LearnStorage()
    index = storage.get_index()

    if not index:
        print("\n  📭 Anney belum belajar apa-apa lagi.")
        print("  Guna /AnneyBelajar <topik> untuk mula belajar.")
        return

    print(f"\n  📚 Senarai Ilmu Anney ({len(index)} topik)")
    print("  " + "─" * 48)

    for item in index:
        confidence = item.get("confidence", "?")
        date = item.get("date", "?")
        sources = item.get("sources_count", 0)
        topic = item.get("topic", "?")
        print(f"  📖 {topic}")
        print(f"     {date} | {sources} sumber | {confidence}")

    print("  " + "─" * 48)
    print()


# =========================================================================== #
#  Standalone execution                                                         #
# =========================================================================== #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AnneyBelajar — Pembelajaran Kendiri")
    parser.add_argument("topic", nargs="?", help="Topik yang hendak dipelajari")
    parser.add_argument("--lang", choices=["ms", "en"], default=None, help="Bahasa output")
    parser.add_argument("--list", action="store_true", help="Senarai semua ilmu")
    args = parser.parse_args()

    if args.list:
        handle_senarai_ilmu_command()
    elif args.topic:
        anney = AnneyBelajar()
        anney.belajar(topic=args.topic, lang=args.lang, verbose=True)
    else:
        print("Guna: python scripts/anney_belajar.py \"sejarah Melaka\"")
        print("      python scripts/anney_belajar.py --list")
