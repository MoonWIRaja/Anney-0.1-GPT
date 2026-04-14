"""
AnneyFahamkan — Suap Maklumat Terus kepada Anney.

Beza dengan AnneyBelajar (cari dari internet), AnneyFahamkan terima
maklumat terus dari pengguna melalui chat. Anney akan proses, simpan
ke memori, dan boleh jawab soalan berdasarkan maklumat yang diajar.

Flow:
  1. User taip /AnneyFahamkan <tajuk>
  2. User beri maklumat step-by-step (baris demi baris)
  3. User taip /selesai untuk habiskan
  4. Anney proses, simpan ke memori + training data
  5. Bila ada soalan berkaitan, Anney boleh jawab

Guna:
    /AnneyFahamkan cara masuk server Minecraft Burhan
    > 1. Buka Minecraft Java Edition
    > 2. Klik Multiplayer
    > 3. Klik Add Server
    > 4. Masukkan IP: play.burhan.my
    > 5. Klik Done
    > 6. Double-click server untuk masuk
    /selesai
"""

import os
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

TAUGHT_DIR = PROJECT_ROOT / "memory" / "library-items" / "taught"
INDEX_FILE = TAUGHT_DIR / "_index.json"
TRAINING_DIR = PROJECT_ROOT / "data" / "raw"
CHAT_DATA_DIR = PROJECT_ROOT / "data_samples"


# =========================================================================== #
#  Taught Knowledge Report                                                      #
# =========================================================================== #

class TaughtKnowledge:
    """Satu unit maklumat yang diajar oleh pengguna."""

    def __init__(
        self,
        title: str,
        content_lines: list[str],
        tags: list[str] = None,
        lang: str = "ms",
    ):
        self.title = title
        self.content_lines = content_lines
        self.tags = tags or []
        self.lang = lang
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.time = datetime.now().strftime("%H:%M")
        self.slug = self._make_slug(title)

        # Auto-extract tags dari tajuk dan content
        if not self.tags:
            self.tags = self._auto_tags()

    def to_markdown(self) -> str:
        """Tukar ke format Markdown untuk simpanan memori."""
        lines = [
            f"# {self.title}",
            f"*Diajar: {self.date} {self.time}*",
            f"*Tags: {', '.join(self.tags)}*" if self.tags else "",
            "",
            "## Maklumat",
            "",
        ]

        for line in self.content_lines:
            lines.append(line)

        lines.append("")
        lines.append("---")
        lines.append(f"*Sumber: Diajar terus oleh pengguna melalui /AnneyFahamkan*")

        return "\n".join(lines)

    def to_training_text(self) -> str:
        """Tukar ke format teks untuk training data model."""
        parts = [f"# {self.title}\n"]

        for line in self.content_lines:
            parts.append(line)

        return "\n".join(parts)

    def to_chat_training(self) -> list[dict]:
        """
        Tukar ke format chat training (JSONL) supaya model boleh
        jawab soalan berkaitan maklumat ini.
        """
        content = "\n".join(self.content_lines)
        title_lower = self.title.lower()

        # Auto-generate variasi soalan berdasarkan tajuk
        qa_pairs = []

        if self.lang == "ms":
            questions = [
                f"Macam mana {title_lower}?",
                f"Cara {title_lower}",
                f"Bagaimana {title_lower}?",
                f"Terangkan {title_lower}",
                f"Apa itu {title_lower}?",
                f"Boleh jelaskan {title_lower}?",
            ]
        else:
            questions = [
                f"How to {title_lower}?",
                f"What is {title_lower}?",
                f"Explain {title_lower}",
                f"Can you tell me about {title_lower}?",
                f"How do I {title_lower}?",
            ]

        for q in questions:
            qa_pairs.append({
                "pengguna": q,
                "anney": content,
            })

        # Tambah soalan dari tags
        for tag in self.tags:
            if self.lang == "ms":
                qa_pairs.append({
                    "pengguna": f"Apa yang kau tahu tentang {tag}?",
                    "anney": f"Tentang {tag}, saya tahu maklumat berikut:\n\n{content}",
                })
            else:
                qa_pairs.append({
                    "pengguna": f"What do you know about {tag}?",
                    "anney": f"About {tag}, I know the following:\n\n{content}",
                })

        return qa_pairs

    def _auto_tags(self) -> list[str]:
        """Extract tags automatik dari tajuk dan content."""
        # Gabungkan tajuk dan content
        all_text = self.title + " " + " ".join(self.content_lines)
        all_text_lower = all_text.lower()

        # Stop words BM + EN
        stop_words = {
            "dan", "atau", "yang", "ini", "itu", "ke", "di", "dari",
            "untuk", "dengan", "dalam", "ada", "tidak", "akan", "telah",
            "boleh", "kena", "perlu", "cara", "macam", "mana", "apa",
            "the", "and", "or", "is", "are", "to", "in", "on", "for",
            "of", "a", "an", "how", "what", "this", "that", "with",
        }

        # Extract kata penting
        words = re.findall(r"\b[a-zA-Z\u00C0-\u024F]{3,}\b", all_text)
        word_freq = {}
        for w in words:
            w_lower = w.lower()
            if w_lower not in stop_words and len(w_lower) > 3:
                word_freq[w_lower] = word_freq.get(w_lower, 0) + 1

        # Ambil top 5 kata paling kerap
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        tags = [w for w, _ in sorted_words[:5]]

        # Tambah tajuk penuh sebagai tag
        if self.title.lower() not in tags:
            tags.insert(0, self.title.lower())

        return tags

    @staticmethod
    def _make_slug(title: str) -> str:
        """Tukar tajuk ke slug selamat."""
        slug = title.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        return slug[:60]


# =========================================================================== #
#  Taught Storage                                                               #
# =========================================================================== #

class TaughtStorage:
    """Simpan dan urus maklumat yang diajar."""

    def __init__(self):
        TAUGHT_DIR.mkdir(parents=True, exist_ok=True)
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        CHAT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def save(self, knowledge: TaughtKnowledge, verbose: bool = True) -> dict:
        """
        Simpan maklumat yang diajar ke semua lokasi.

        Returns:
            dict dengan paths yang disimpan
        """
        paths = {}

        # ── 1. Simpan ke memory ──────────────────────────────────── #
        memory_path = TAUGHT_DIR / f"{knowledge.slug}.md"
        with open(memory_path, "w", encoding="utf-8") as f:
            f.write(knowledge.to_markdown())
        paths["memory"] = str(memory_path)

        if verbose:
            rel = memory_path.relative_to(PROJECT_ROOT)
            print(f"  💾 Disimpan ke memori: {rel}")

        # ── 2. Simpan ke training data (raw text) ────────────────── #
        training_path = TRAINING_DIR / f"taught_{knowledge.slug}.md"
        with open(training_path, "w", encoding="utf-8") as f:
            f.write(knowledge.to_training_text())
        paths["training"] = str(training_path)

        if verbose:
            rel = training_path.relative_to(PROJECT_ROOT)
            print(f"  📖 Training data: {rel}")

        # ── 3. Simpan ke chat training data (JSONL) ──────────────── #
        chat_path = CHAT_DATA_DIR / "taught_qa.jsonl"
        qa_pairs = knowledge.to_chat_training()

        # Append ke fail sedia ada
        with open(chat_path, "a", encoding="utf-8") as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        paths["chat_training"] = str(chat_path)

        if verbose:
            print(f"  💬 Chat training: {len(qa_pairs)} soalan-jawapan ditambah")

        # ── 4. Update index ──────────────────────────────────────── #
        self._update_index(knowledge, paths)
        if verbose:
            print(f"  📋 Index dikemaskini")

        # ── 5. Update library-items README ────────────────────────── #
        self._update_readme(knowledge)

        return paths

    def get_index(self) -> list[dict]:
        """Baca index semua maklumat yang diajar."""
        if INDEX_FILE.exists():
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def search(self, query: str) -> list[dict]:
        """
        Cari dalam maklumat yang diajar berdasarkan query.
        Return list of matching entries.
        """
        index = self.get_index()
        query_lower = query.lower()
        results = []

        for item in index:
            title = item.get("title", "").lower()
            tags = [t.lower() for t in item.get("tags", [])]
            score = 0

            # Scoring
            if query_lower in title:
                score += 10
            for tag in tags:
                if query_lower in tag or tag in query_lower:
                    score += 5
            # Word overlap
            query_words = set(query_lower.split())
            title_words = set(title.split())
            tag_words = set(tags)
            overlap = query_words & (title_words | tag_words)
            score += len(overlap) * 3

            if score > 0:
                results.append({**item, "_score": score})

        results.sort(key=lambda x: x["_score"], reverse=True)
        return results

    def get_knowledge(self, slug: str) -> str | None:
        """Baca kandungan maklumat yang diajar."""
        path = TAUGHT_DIR / f"{slug}.md"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def get_all_titles(self) -> list[str]:
        """Senarai semua tajuk yang diajar."""
        index = self.get_index()
        return [item.get("title", "") for item in index]

    def _update_index(self, knowledge: TaughtKnowledge, paths: dict):
        """Kemaskini fail index."""
        index = self.get_index()

        entry = {
            "title": knowledge.title,
            "slug": knowledge.slug,
            "date": knowledge.date,
            "time": knowledge.time,
            "lang": knowledge.lang,
            "tags": knowledge.tags,
            "lines_count": len(knowledge.content_lines),
            "memory_path": paths.get("memory", ""),
            "training_path": paths.get("training", ""),
        }

        # Replace kalau sudah ada
        updated = False
        for i, item in enumerate(index):
            if item.get("slug") == knowledge.slug:
                index[i] = entry
                updated = True
                break

        if not updated:
            index.append(entry)

        index.sort(key=lambda x: x.get("date", ""), reverse=True)

        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _update_readme(self, knowledge: TaughtKnowledge):
        """Kemaskini README library-items."""
        readme_path = PROJECT_ROOT / "memory" / "library-items" / "README.md"

        if not readme_path.exists():
            content = "# Library Items\n\nKoleksi pengetahuan yang tersimpan.\n\n"
        else:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()

        # Tambah section Taught kalau belum ada
        if "## Taught" not in content:
            content += "\n\n## Taught (Diajar oleh Pengguna)\n\n"

        # Tambah entry baru
        entry_marker = f"- [{knowledge.title}]"
        if entry_marker not in content:
            entry_line = (
                f"- [{knowledge.title}](taught/{knowledge.slug}.md) "
                f"— {knowledge.date}\n"
            )
            content = content.replace(
                "## Taught (Diajar oleh Pengguna)\n\n",
                f"## Taught (Diajar oleh Pengguna)\n\n{entry_line}",
            )

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)


# =========================================================================== #
#  Language Detection (reuse from anney_belajar)                                #
# =========================================================================== #

def detect_language(text: str) -> str:
    """Detect bahasa — BM atau EN."""
    text_lower = text.lower()

    malay_markers = [
        "apa", "ini", "itu", "dan", "yang", "dalam", "untuk",
        "saya", "kita", "dia", "adalah", "ada", "tidak",
        "bagaimana", "kenapa", "macam", "belajar", "tentang",
        "sejarah", "cara", "boleh", "perlu", "hendak", "mahu",
        "buka", "klik", "masuk", "tekan", "pilih", "tulis",
        "server", "masukkan", "kemudian", "lepas", "tunggu",
    ]

    en_markers = [
        "what", "the", "and", "how", "why", "about", "learn",
        "this", "that", "for", "with", "from", "into", "have",
        "does", "can", "should", "would", "which", "where",
        "open", "click", "type", "enter", "then", "next",
    ]

    words = text_lower.split()
    ms_score = sum(1 for w in malay_markers if w in words)
    en_score = sum(1 for w in en_markers if w in words)

    return "ms" if ms_score >= en_score else "en"


# =========================================================================== #
#  CLI Handler — dipanggil dari chat.py                                         #
# =========================================================================== #

def handle_anney_fahamkan_command(user_input: str, input_func=None) -> None:
    """
    Handler untuk /AnneyFahamkan dalam CLI chat.

    Flow:
    1. Parse tajuk dari command
    2. Minta pengguna beri maklumat baris demi baris
    3. User taip /selesai untuk habiskan
    4. Proses dan simpan

    Args:
        user_input: Input penuh (contoh: "/AnneyFahamkan cara masuk server")
        input_func: Custom input function (untuk testing). Default: built-in input()
    """
    if input_func is None:
        input_func = input

    # Parse tajuk
    parts = user_input.split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():
        print("\n  ⚠ Sila nyatakan tajuk maklumat yang hendak diajar.")
        print("  Contoh: /AnneyFahamkan cara masuk server Minecraft Burhan")
        print("  Contoh: /AnneyFahamkan what is Python programming")
        return

    title = parts[1].strip()
    lang = detect_language(title)

    # Semak kalau sudah ada
    storage = TaughtStorage()
    existing = storage.search(title)
    if existing:
        if lang == "ms":
            print(f"\n  ℹ Anney sudah ada maklumat berkaitan '{title}'.")
            print("  Maklumat baru akan ditambah/dikemaskini.\n")
        else:
            print(f"\n  ℹ Anney already has info about '{title}'.")
            print("  New info will be added/updated.\n")

    # Minta maklumat
    if lang == "ms":
        print(f"\n  📝 Anney bersedia belajar tentang: {title}")
        print("  ─────────────────────────────────────────────")
        print("  Sila taip maklumat baris demi baris.")
        print("  Taip /selesai bila sudah siap.")
        print("  ─────────────────────────────────────────────\n")
    else:
        print(f"\n  📝 Anney is ready to learn about: {title}")
        print("  ─────────────────────────────────────────────")
        print("  Please type information line by line.")
        print("  Type /done when finished.")
        print("  ─────────────────────────────────────────────\n")

    # Kumpul maklumat
    content_lines = []
    line_num = 0

    while True:
        try:
            line = input_func("  📎 ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  ❌ Dibatalkan.")
            return

        # Semak kalau selesai
        if line.lower() in ("/selesai", "/done", "/habis", "/finish"):
            break

        # Semak kalau batal
        if line.lower() in ("/batal", "/cancel"):
            print("\n  ❌ Dibatalkan. Tiada maklumat disimpan.")
            return

        if line:
            content_lines.append(line)
            line_num += 1

    # Validate
    if not content_lines:
        if lang == "ms":
            print("\n  ⚠ Tiada maklumat diberikan. Dibatalkan.")
        else:
            print("\n  ⚠ No information provided. Cancelled.")
        return

    # Proses dan simpan
    if lang == "ms":
        print(f"\n  🧠 Memproses {len(content_lines)} baris maklumat...\n")
    else:
        print(f"\n  🧠 Processing {len(content_lines)} lines of information...\n")

    knowledge = TaughtKnowledge(
        title=title,
        content_lines=content_lines,
        lang=lang,
    )

    paths = storage.save(knowledge, verbose=True)

    # Report
    if lang == "ms":
        print(f"\n  ═══════════════════════════════════════════════")
        print(f"    ✅ ANNEY SUDAH FAHAM!")
        print(f"  ═══════════════════════════════════════════════")
        print(f"  📚 Tajuk: {title}")
        print(f"  📝 Baris maklumat: {len(content_lines)}")
        print(f"  🏷  Tags: {', '.join(knowledge.tags)}")
        print(f"  💬 {len(knowledge.to_chat_training())} soalan-jawapan dijana")
        print(f"  🧠 Anney sekarang boleh jawab soalan tentang '{title}'")
        print(f"  ═══════════════════════════════════════════════\n")
    else:
        print(f"\n  ═══════════════════════════════════════════════")
        print(f"    ✅ ANNEY NOW UNDERSTANDS!")
        print(f"  ═══════════════════════════════════════════════")
        print(f"  📚 Title: {title}")
        print(f"  📝 Lines of info: {len(content_lines)}")
        print(f"  🏷  Tags: {', '.join(knowledge.tags)}")
        print(f"  💬 {len(knowledge.to_chat_training())} Q&A pairs generated")
        print(f"  🧠 Anney can now answer questions about '{title}'")
        print(f"  ═══════════════════════════════════════════════\n")


def handle_senarai_faham_command() -> None:
    """Handler untuk /SenaraiAjar — senarai semua yang diajar."""
    storage = TaughtStorage()
    index = storage.get_index()

    if not index:
        print("\n  📭 Anney belum diajar apa-apa lagi.")
        print("  Guna /AnneyFahamkan <tajuk> untuk mula mengajar.")
        return

    print(f"\n  📚 Senarai Maklumat Diajar ({len(index)} topik)")
    print("  " + "─" * 48)

    for item in index:
        date = item.get("date", "?")
        lines = item.get("lines_count", 0)
        title = item.get("title", "?")
        tags = ", ".join(item.get("tags", [])[:3])
        print(f"  📖 {title}")
        print(f"     {date} | {lines} baris | Tags: {tags}")

    print("  " + "─" * 48)
    print()


def handle_cari_faham_command(query: str) -> str | None:
    """
    Cari dalam maklumat yang diajar.
    Dipanggil secara automatik bila user tanya soalan.

    Returns:
        String jawapan dari maklumat diajar, atau None kalau tiada match.
    """
    storage = TaughtStorage()
    results = storage.search(query)

    if not results:
        return None

    # Ambil top result
    best = results[0]
    slug = best.get("slug", "")
    content = storage.get_knowledge(slug)

    if content:
        # Extract bahagian maklumat sahaja
        lines = content.split("\n")
        info_lines = []
        in_info = False
        for line in lines:
            if line.strip() == "## Maklumat":
                in_info = True
                continue
            if in_info and line.startswith("---"):
                break
            if in_info and line.strip():
                info_lines.append(line)

        if info_lines:
            return "\n".join(info_lines)

    return None


# =========================================================================== #
#  Standalone execution                                                         #
# =========================================================================== #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AnneyFahamkan — Ajar Anney Terus")
    parser.add_argument("title", nargs="?", help="Tajuk maklumat")
    parser.add_argument("--list", action="store_true", help="Senarai semua ajar")
    parser.add_argument("--search", help="Cari dalam maklumat diajar")
    args = parser.parse_args()

    if args.list:
        handle_senarai_faham_command()
    elif args.search:
        result = handle_cari_faham_command(args.search)
        if result:
            print(f"\n{result}\n")
        else:
            print("\n  Tiada maklumat berkaitan dijumpai.\n")
    elif args.title:
        handle_anney_fahamkan_command(f"AnneyFahamkan {args.title}")
    else:
        print("Guna: python scripts/anney_fahamkan.py \"cara masuk server\"")
        print("      python scripts/anney_fahamkan.py --list")
        print("      python scripts/anney_fahamkan.py --search \"server minecraft\"")
