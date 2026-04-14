"""
Penyimpanan Pembelajaran AnneyBelajar.

Simpan hasil pembelajaran ke dalam sistem memori:
  - memory/library-items/learned/{topik}.md — ilmu yang dipelajari
  - memory/library-items/learned/_index.json — index semua ilmu
  - data/raw/learned_{topik}.md — training data untuk model

Guna:
    from scripts.learn_storage import LearnStorage
    storage = LearnStorage()
    storage.save(report)
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

LEARN_DIR = PROJECT_ROOT / "memory" / "library-items" / "learned"
INDEX_FILE = LEARN_DIR / "_index.json"
TRAINING_DIR = PROJECT_ROOT / "data" / "raw"


# =========================================================================== #
#  Learn Storage                                                                #
# =========================================================================== #

class LearnStorage:
    """Simpan dan urus hasil pembelajaran."""

    def __init__(self):
        # Pastikan folder wujud
        LEARN_DIR.mkdir(parents=True, exist_ok=True)
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    def save(self, report, verbose: bool = True) -> dict:
        """
        Simpan KnowledgeReport ke memori dan training data.

        Args:
            report:  KnowledgeReport dari KnowledgeProcessor
            verbose: Tunjuk progress

        Returns:
            dict dengan paths yang disimpan
        """
        slug = self._make_slug(report.topic)
        paths = {}

        # ── 1. Simpan ke memory ──────────────────────────────────── #
        memory_path = LEARN_DIR / f"{slug}.md"
        markdown_content = report.to_markdown()

        with open(memory_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        paths["memory"] = str(memory_path)

        if verbose:
            print(f"  💾 Disimpan ke memori: {memory_path.relative_to(PROJECT_ROOT)}")

        # ── 2. Simpan ke training data ───────────────────────────── #
        training_path = TRAINING_DIR / f"learned_{slug}.md"
        training_content = report.to_training_text()

        with open(training_path, "w", encoding="utf-8") as f:
            f.write(training_content)
        paths["training"] = str(training_path)

        if verbose:
            print(f"  📖 Ditambah ke training data: {training_path.relative_to(PROJECT_ROOT)}")

        # ── 3. Update index ──────────────────────────────────────── #
        self._update_index(report, slug, paths)

        if verbose:
            print(f"  📋 Index dikemaskini")

        # ── 4. Update library-items README ────────────────────────── #
        self._update_readme(report, slug)

        return paths

    def get_index(self) -> list[dict]:
        """Baca index semua ilmu yang dipelajari."""
        if INDEX_FILE.exists():
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def has_learned(self, topic: str) -> bool:
        """Semak kalau topik sudah pernah dipelajari."""
        slug = self._make_slug(topic)
        return (LEARN_DIR / f"{slug}.md").exists()

    def get_learned_topics(self) -> list[str]:
        """Senarai semua topik yang telah dipelajari."""
        index = self.get_index()
        return [item.get("topic", "") for item in index]

    def _update_index(self, report, slug: str, paths: dict):
        """Kemaskini fail index."""
        index = self.get_index()

        # Cari kalau sudah ada entry untuk topik ini
        entry = {
            "topic": report.topic,
            "slug": slug,
            "date": report.date,
            "lang": report.lang,
            "sources_count": len(report.sources),
            "confidence": report.confidence,
            "facts_count": len(report.key_facts),
            "memory_path": paths.get("memory", ""),
            "training_path": paths.get("training", ""),
            "total_chars": report.total_chars,
        }

        # Replace kalau sudah ada, tambah kalau baru
        updated = False
        for i, item in enumerate(index):
            if item.get("slug") == slug:
                index[i] = entry
                updated = True
                break

        if not updated:
            index.append(entry)

        # Sort by date (terbaru dulu)
        index.sort(key=lambda x: x.get("date", ""), reverse=True)

        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _update_readme(self, report, slug: str):
        """Kemaskini README library-items dengan entry baru."""
        readme_path = PROJECT_ROOT / "memory" / "library-items" / "README.md"

        if not readme_path.exists():
            # Buat README baru
            content = (
                "# Library Items\n\n"
                "Koleksi pengetahuan yang tersimpan.\n\n"
                "## Learned (Dipelajari dari Internet)\n\n"
            )
        else:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()

        # Tambah section Learned kalau belum ada
        if "## Learned" not in content:
            content += "\n\n## Learned (Dipelajari dari Internet)\n\n"

        # Tambah entry baru kalau belum ada
        entry_marker = f"- [{report.topic}]"
        if entry_marker not in content:
            entry_line = (
                f"- [{report.topic}](learned/{slug}.md) "
                f"— {report.date} ({report.confidence})\n"
            )
            # Masukkan selepas header "## Learned"
            content = content.replace(
                "## Learned (Dipelajari dari Internet)\n\n",
                f"## Learned (Dipelajari dari Internet)\n\n{entry_line}",
            )

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _make_slug(topic: str) -> str:
        """Tukar topik ke slug yang selamat untuk nama fail."""
        slug = topic.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        slug = slug[:60]  # Hadkan panjang
        return slug
