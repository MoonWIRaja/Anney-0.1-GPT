"""
Knowledge Processor untuk AnneyBelajar.

Proses data mentah dari penyelidikan web dan hasilkan:
  - Ringkasan tersusun dan neutral
  - Fakta utama
  - Skor keyakinan

Prinsip neutral:
  - Bahasa objektif ("menurut...", "dinyatakan bahawa...")
  - Pelbagai sumber, bukan satu sahaja
  - Kontroversi: nyatakan semua pihak
  - Labelkan fakta vs pandangan

Guna:
    from scripts.knowledge_processor import KnowledgeProcessor
    processor = KnowledgeProcessor()
    report = processor.process(results, topic="quantum computing", lang="ms")
"""

import re
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


# =========================================================================== #
#  Structured Knowledge Report                                                  #
# =========================================================================== #

class KnowledgeReport:
    """Hasil pemprosesan pengetahuan — report yang tersusun."""

    def __init__(
        self,
        topic: str,
        lang: str,
        summary: str,
        key_facts: list[str],
        sections: list[dict],
        sources: list[dict],
        confidence: str,
        total_chars: int,
        date: str,
    ):
        self.topic = topic
        self.lang = lang
        self.summary = summary
        self.key_facts = key_facts
        self.sections = sections  # [{title, content}]
        self.sources = sources    # [{source, title, url}]
        self.confidence = confidence  # "rendah" / "sederhana" / "tinggi"
        self.total_chars = total_chars
        self.date = date

    def to_markdown(self) -> str:
        """Tukar report ke format Markdown untuk disimpan."""
        lang = self.lang

        # Pilih label berdasarkan bahasa
        if lang == "ms":
            lbl = {
                "studied": "Dipelajari",
                "sources_count": "Sumber",
                "sources_label": "sumber dari internet",
                "confidence_label": "Tahap Keyakinan",
                "summary_title": "Ringkasan",
                "facts_title": "Fakta Utama",
                "details_title": "Maklumat Terperinci",
                "ref_title": "Sumber Rujukan",
                "chars_label": "aksara diproses",
            }
        else:
            lbl = {
                "studied": "Studied",
                "sources_count": "Sources",
                "sources_label": "sources from the internet",
                "confidence_label": "Confidence Level",
                "summary_title": "Summary",
                "facts_title": "Key Facts",
                "details_title": "Detailed Information",
                "ref_title": "References",
                "chars_label": "characters processed",
            }

        lines = [
            f"# {self.topic}",
            f"*{lbl['studied']}: {self.date}*",
            f"*{lbl['sources_count']}: {len(self.sources)} {lbl['sources_label']}*",
            f"*{lbl['confidence_label']}: {self.confidence}*",
            "",
            f"## {lbl['summary_title']}",
            "",
            self.summary,
            "",
            f"## {lbl['facts_title']}",
            "",
        ]

        for fact in self.key_facts:
            lines.append(f"- {fact}")

        lines.append("")

        if self.sections:
            lines.append(f"## {lbl['details_title']}")
            lines.append("")
            for section in self.sections:
                lines.append(f"### {section['title']}")
                lines.append("")
                lines.append(section["content"])
                lines.append("")

        lines.append(f"## {lbl['ref_title']}")
        lines.append("")
        for i, src in enumerate(self.sources, 1):
            url_part = f" — {src['url']}" if src.get("url") else ""
            lines.append(f"{i}. [{src['source']}] {src['title']}{url_part}")

        lines.append("")
        lines.append("---")
        lines.append(f"*{self.total_chars:,} {lbl['chars_label']}*")

        return "\n".join(lines)

    def to_training_text(self) -> str:
        """Tukar report ke format teks untuk training data model."""
        parts = [
            f"# {self.topic}\n",
            self.summary,
            "\n",
        ]

        for section in self.sections:
            parts.append(f"\n## {section['title']}\n")
            parts.append(section["content"])

        if self.key_facts:
            parts.append("\n## Fakta Penting\n" if self.lang == "ms" else "\n## Key Facts\n")
            for fact in self.key_facts:
                parts.append(f"- {fact}")

        return "\n".join(parts)

    def to_terminal_report(self) -> str:
        """Format report untuk paparan terminal."""
        lang = self.lang

        if lang == "ms":
            lbl = {
                "title": "LAPORAN PEMBELAJARAN ANNEY",
                "topic": "Topik",
                "date": "Tarikh",
                "sources": "Sumber",
                "sources_suffix": "sumber",
                "platforms": "platform",
                "confidence": "Keyakinan",
                "summary_h": "Ringkasan",
                "facts_h": "Fakta Utama",
                "refs_h": "Sumber Rujukan",
                "saved": "Disimpan ke",
            }
        else:
            lbl = {
                "title": "ANNEY LEARNING REPORT",
                "topic": "Topic",
                "date": "Date",
                "sources": "Sources",
                "sources_suffix": "sources",
                "platforms": "platforms",
                "confidence": "Confidence",
                "summary_h": "Summary",
                "facts_h": "Key Facts",
                "refs_h": "References",
                "saved": "Saved to",
            }

        # Kira platform unik
        platforms = set()
        for src in self.sources:
            name = src.get("source", "")
            if "wikipedia" in name.lower():
                platforms.add("Wikipedia")
            elif "duckduckgo" in name.lower():
                platforms.add("DuckDuckGo")
            else:
                platforms.add("Web")

        lines = [
            "",
            "  " + "═" * 50,
            f"    📊 {lbl['title']}",
            "  " + "═" * 50,
            "",
            f"  📚 {lbl['topic']}: {self.topic}",
            f"  📅 {lbl['date']}: {self.date}",
            f"  📡 {lbl['sources']}: {len(self.sources)} {lbl['sources_suffix']} dari {len(platforms)} {lbl['platforms']}",
            f"  🎯 {lbl['confidence']}: {self.confidence}",
            "",
            "  ─── " + lbl['summary_h'] + " " + "─" * (44 - len(lbl['summary_h'])),
        ]

        # Wrap summary to ~70 chars per line
        for para in self.summary.split("\n\n"):
            wrapped = _wrap_text(para, width=68, indent="  ")
            lines.append(wrapped)
            lines.append("")

        lines.append("  ─── " + lbl['facts_h'] + " " + "─" * (44 - len(lbl['facts_h'])))
        for fact in self.key_facts:
            lines.append(f"  • {fact}")
        lines.append("")

        lines.append("  ─── " + lbl['refs_h'] + " " + "─" * (44 - len(lbl['refs_h'])))
        for i, src in enumerate(self.sources, 1):
            lines.append(f"  {i}. [{src['source']}] {src['title']}")
        lines.append("")

        lines.append("  " + "═" * 50)

        return "\n".join(lines)


# =========================================================================== #
#  Knowledge Processor                                                          #
# =========================================================================== #

class KnowledgeProcessor:
    """Proses data mentah dari penyelidikan dan hasilkan report neutral."""

    # Frasa neutral berdasarkan bahasa
    NEUTRAL_PHRASES = {
        "ms": [
            "Menurut sumber yang dikaji",
            "Berdasarkan maklumat yang dikumpulkan",
            "Dinyatakan bahawa",
            "Secara umumnya",
            "Menurut kajian",
            "Berdasarkan pelbagai sumber",
        ],
        "en": [
            "According to available sources",
            "Based on the information gathered",
            "It is stated that",
            "Generally",
            "According to research",
            "Based on multiple sources",
        ],
    }

    def process(
        self,
        results: list,
        topic: str,
        lang: str = "ms",
        verbose: bool = True,
    ) -> KnowledgeReport:
        """
        Proses hasil penyelidikan dan hasilkan KnowledgeReport.

        Args:
            results:  list of ResearchResult dari WebResearcher
            topic:    Topik asal
            lang:     Bahasa output (ms/en)
            verbose:  Tunjuk progress

        Returns:
            KnowledgeReport
        """
        if verbose:
            print("  📝 Memproses maklumat...", flush=True)

        # ── 1. Kumpul semua teks ──────────────────────────────────── #
        all_content = []
        sources = []
        total_chars = 0

        for r in results:
            if r.content and len(r.content.strip()) > 50:
                all_content.append({
                    "source": r.source,
                    "title": r.title,
                    "content": self._clean_content(r.content),
                    "language": r.language,
                    "url": r.url,
                })
                sources.append({
                    "source": r.source,
                    "title": r.title,
                    "url": r.url,
                })
                total_chars += len(r.content)

        if not all_content:
            return self._empty_report(topic, lang)

        # ── 2. Extract key facts ──────────────────────────────────── #
        key_facts = self._extract_key_facts(all_content, topic, lang)

        # ── 3. Build sections ─────────────────────────────────────── #
        sections = self._build_sections(all_content, topic, lang)

        # ── 4. Build summary ──────────────────────────────────────── #
        summary = self._build_summary(all_content, sections, key_facts, topic, lang)

        # ── 5. Calculate confidence ───────────────────────────────── #
        confidence = self._calculate_confidence(results, all_content, lang)

        if verbose:
            print("  📝 Memproses maklumat... ✓", flush=True)

        return KnowledgeReport(
            topic=topic,
            lang=lang,
            summary=summary,
            key_facts=key_facts,
            sections=sections,
            sources=sources,
            confidence=confidence,
            total_chars=total_chars,
            date=datetime.now().strftime("%Y-%m-%d"),
        )

    def _clean_content(self, text: str) -> str:
        """Bersihkan teks content."""
        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        # Buang URLs berlebihan
        text = re.sub(r"https?://\S+", "", text)
        # Buang karakter pelik
        text = re.sub(r"[^\w\s.,;:!?'\"-–—()\[\]{}%@#&*/+=$€£¥°²³]", "", text)
        return text.strip()

    def _extract_key_facts(self, contents: list[dict], topic: str, lang: str) -> list[str]:
        """Extract fakta utama dari semua sumber."""
        facts = []
        seen = set()

        for item in contents:
            text = item["content"]
            sentences = self._split_sentences(text)

            for sentence in sentences:
                s = sentence.strip()
                # Cari ayat yang mengandungi topik dan fakta
                topic_lower = topic.lower()
                s_lower = s.lower()

                if len(s) < 30 or len(s) > 300:
                    continue

                # Topik mesti ada dalam ayat, atau bermula dengan kata kunci
                if topic_lower not in s_lower and not self._is_factual(s):
                    continue

                # Deduplicate
                s_hash = hashlib.md5(s_lower.encode()).hexdigest()[:8]
                if s_hash in seen:
                    continue
                seen.add(s_hash)

                facts.append(s)

                if len(facts) >= 10:
                    break

            if len(facts) >= 10:
                break

        # Kalau tak cukup fakta, ambil ayat pertama dari setiap sumber
        if len(facts) < 3:
            for item in contents:
                sentences = self._split_sentences(item["content"])
                for s in sentences[:2]:
                    if len(s.strip()) > 30:
                        s_hash = hashlib.md5(s.strip().lower().encode()).hexdigest()[:8]
                        if s_hash not in seen:
                            seen.add(s_hash)
                            facts.append(s.strip())
                            break

        return facts[:10]

    def _build_sections(self, contents: list[dict], topic: str, lang: str) -> list[dict]:
        """Bina sections dari content yang dikumpulkan."""
        sections = []

        # Kumpulkan semua teks berdasarkan source
        for item in contents[:5]:  # Hadkan kepada 5 sumber terbaik
            content = item["content"]

            # Cuba pecah berdasarkan heading sedia ada
            heading_parts = re.split(r"\n==+\s*(.+?)\s*==+\n|\n#+\s*(.+?)\n", content)

            if len(heading_parts) > 1:
                # Ada headings — guna sebagai sections
                current_title = item["title"]
                current_content = []

                for part in heading_parts:
                    if not part:
                        continue
                    part = part.strip()
                    if len(part) < 50 and not any(c in part for c in ".!?,;"):
                        # Ini heading
                        if current_content:
                            combined = "\n\n".join(current_content)
                            if len(combined) > 100:
                                sections.append({
                                    "title": current_title,
                                    "content": combined[:2000],
                                })
                        current_title = part
                        current_content = []
                    else:
                        current_content.append(part)

                if current_content:
                    combined = "\n\n".join(current_content)
                    if len(combined) > 100:
                        sections.append({
                            "title": current_title,
                            "content": combined[:2000],
                        })
            else:
                # Tiada heading — satu section penuh
                if len(content) > 100:
                    sections.append({
                        "title": f"{item['title']} ({item['source']})",
                        "content": content[:2000],
                    })

        return sections[:8]  # Hadkan kepada 8 sections

    def _build_summary(
        self,
        contents: list[dict],
        sections: list[dict],
        key_facts: list[str],
        topic: str,
        lang: str,
    ) -> str:
        """Bina ringkasan neutral dari semua maklumat."""
        # Kumpulkan perenggan pembukaan dari setiap sumber
        opening_paras = []
        for item in contents[:4]:
            paragraphs = item["content"].split("\n\n")
            for p in paragraphs[:2]:
                p = p.strip()
                if len(p) > 100:
                    opening_paras.append(p)
                    break

        if not opening_paras:
            if lang == "ms":
                return f"Maklumat tentang '{topic}' sedang dikumpulkan dari pelbagai sumber."
            else:
                return f"Information about '{topic}' is being gathered from multiple sources."

        # Bina summary dari perenggan pembukaan (dedup dan gabungkan)
        summary_parts = []
        phrases = self.NEUTRAL_PHRASES.get(lang, self.NEUTRAL_PHRASES["en"])

        for i, para in enumerate(opening_paras[:3]):
            # Tambah frasa neutral pada permulaan
            if i == 0:
                summary_parts.append(para)
            else:
                prefix = phrases[min(i, len(phrases) - 1)]
                # Elakkan redundancy — hanya tambah kalau bukan duplicate
                if not any(
                    _text_similarity(para, existing) > 0.5
                    for existing in summary_parts
                ):
                    summary_parts.append(f"{prefix}, {para[0].lower()}{para[1:]}")

        return "\n\n".join(summary_parts)

    def _calculate_confidence(self, results: list, contents: list[dict], lang: str) -> str:
        """Kira tahap keyakinan berdasarkan bilangan dan kualiti sumber."""
        n_sources = len(contents)
        n_wiki = sum(1 for r in results if "wikipedia" in r.source.lower())
        total_chars = sum(len(c["content"]) for c in contents)

        score = 0
        score += min(n_sources * 10, 40)  # Max 40 dari bilangan sumber
        score += min(n_wiki * 15, 30)     # Max 30 dari Wikipedia
        score += min(total_chars // 1000, 30)  # Max 30 dari jumlah teks

        if lang == "ms":
            if score >= 60:
                return "Tinggi ✅"
            elif score >= 30:
                return "Sederhana ⚡"
            else:
                return "Rendah ⚠️"
        else:
            if score >= 60:
                return "High ✅"
            elif score >= 30:
                return "Medium ⚡"
            else:
                return "Low ⚠️"

    def _empty_report(self, topic: str, lang: str) -> KnowledgeReport:
        """Report kosong kalau tiada hasil."""
        if lang == "ms":
            summary = f"Tiada maklumat yang mencukupi tentang '{topic}' dijumpai dari sumber internet."
        else:
            summary = f"Insufficient information about '{topic}' was found from internet sources."

        return KnowledgeReport(
            topic=topic, lang=lang, summary=summary,
            key_facts=[], sections=[], sources=[],
            confidence="Rendah ⚠️" if lang == "ms" else "Low ⚠️",
            total_chars=0,
            date=datetime.now().strftime("%Y-%m-%d"),
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Pecahkan teks kepada ayat."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _is_factual(sentence: str) -> bool:
        """Check kalau ayat mengandungi ciri fakta."""
        factual_markers = [
            r"\d{4}",           # tahun
            r"\d+%",            # peratusan
            r"\d+\.\d+",       # nombor perpuluhan
            r"(?:km|kg|m|cm)",  # unit
            r"(?:pertama|kedua|ketiga|largest|smallest|first|second)",  # ordinal
            r"(?:adalah|ialah|merupakan|is|was|are|were)",  # definisi
        ]
        for pattern in factual_markers:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False


# =========================================================================== #
#  Helper functions                                                             #
# =========================================================================== #

def _text_similarity(a: str, b: str) -> float:
    """Kira kesamaan ringkas antara dua teks (Jaccard similarity)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _wrap_text(text: str, width: int = 70, indent: str = "") -> str:
    """Wrap teks kepada lebar tertentu."""
    words = text.split()
    lines = []
    current_line = indent

    for word in words:
        if len(current_line) + len(word) + 1 > width:
            lines.append(current_line)
            current_line = indent + word
        else:
            if current_line == indent:
                current_line += word
            else:
                current_line += " " + word

    if current_line.strip():
        lines.append(current_line)

    return "\n".join(lines)
