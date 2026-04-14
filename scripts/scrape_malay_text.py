"""
Skrip untuk memuat turun teks Bahasa Melayu dari Wikipedia API.

Wikipedia BM tersedia secara percuma melalui API REST dan MediaWiki API.
Skrip ini memuat turun artikel dari kategori tertentu dan menyimpannya
sebagai fail .md dalam data/raw/.

Guna:
    python scripts/scrape_malay_text.py --articles 50 --output data/raw
    python scripts/scrape_malay_text.py --categories Malaysia,Sains --output data/raw
"""

import os
import re
import sys
import time
import json
import argparse
import requests
from pathlib import Path

# Tambah root ke sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =========================================================================== #
#  Wikipedia API                                                                #
# =========================================================================== #

WIKI_API = "https://ms.wikipedia.org/w/api.php"
USER_AGENT = "AnneyGPT/0.1 (latihan model bahasa BM; github.com/pengguna/anney)"

HEADERS = {"User-Agent": USER_AGENT}

# Senarai artikel permulaan yang baik untuk teks Melayu
DEFAULT_SEED_ARTICLES = [
    "Malaysia",
    "Bahasa Melayu",
    "Kuala Lumpur",
    "Sejarah Malaysia",
    "Alam sekitar",
    "Sains",
    "Teknologi",
    "Pendidikan",
    "Matematik",
    "Fizik",
    "Biologi",
    "Kimia",
    "Geografi",
    "Ekonomi",
    "Budaya Malaysia",
    "Masakan Malaysia",
    "Sukan",
    "Muzik",
    "Sastera Melayu",
    "Islam",
    "Kerajaan Malaysia",
    "Alam semula jadi",
    "Hutan",
    "Lautan",
    "Bumi",
    "Sejarah dunia",
    "Komputer",
    "Internet",
    "Perubatan",
    "Pertanian",
]


def clean_wiki_text(text: str) -> str:
    """
    Bersihkan teks Wikipedia dari markup dan kandungan tidak berguna.
    """
    # Buang tag XML/HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Buang bahagian tipikal Wikipedia yang tidak berguna
    untuk_buang = [
        r"== Lihat juga ==.*",
        r"== Rujukan ==.*",
        r"== Pautan luar ==.*",
        r"== Nota ==.*",
        r"== Bibliografi ==.*",
    ]
    for pattern in untuk_buang:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Buang template Wikipedia {{ ... }}
    depth = 0
    result = []
    i = 0
    while i < len(text):
        if text[i:i+2] == "{{":
            depth += 1
            i += 2
        elif text[i:i+2] == "}}":
            depth -= 1
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    text = "".join(result)

    # Buang wikilink tapi kekalkan teks: [[Teks|Papar]] → Papar, [[Teks]] → Teks
    text = re.sub(r"\[\[(?:[^\|\]]*\|)?([^\]]*)\]\]", r"\1", text)

    # Buang tanda lain
    text = re.sub(r"\[https?://[^\]]*\]", "", text)
    text = re.sub(r"'{2,3}", "", text)

    # Bersih whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def get_article_content(title: str) -> str | None:
    """
    Dapatkan kandungan teks artikel Wikipedia BM.

    Args:
        title: Tajuk artikel Wikipedia

    Returns:
        Teks artikel, atau None jika gagal
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
        "utf8": True,
    }

    try:
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                return None
            extract = page.get("extract", "")
            if extract and len(extract) > 200:
                return clean_wiki_text(extract)

    except Exception as e:
        print(f"  Ralat memuat '{title}': {e}")

    return None


def get_random_articles(n: int = 10) -> list[str]:
    """
    Dapatkan senarai tajuk artikel rawak dari Wikipedia BM.
    """
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": min(n, 500),
        "format": "json",
    }

    try:
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("random", [])
        return [p["title"] for p in pages]
    except Exception as e:
        print(f"Ralat mendapatkan artikel rawak: {e}")
        return []


def save_article(title: str, content: str, output_dir: str) -> str:
    """
    Simpan kandungan artikel sebagai fail .md.
    """
    # Sanitize nama fail
    safe_name = re.sub(r"[^\w\s\-]", "", title)
    safe_name = re.sub(r"\s+", "_", safe_name.strip())
    safe_name = safe_name[:80]  # Hadkan panjang nama fail

    filename = f"{safe_name}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(content)

    return filepath


# =========================================================================== #
#  Main                                                                         #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Muat turun teks BM dari Wikipedia untuk latihan Anney GPT"
    )
    parser.add_argument(
        "--articles", type=int, default=30,
        help="Bilangan artikel untuk dimuat turun (default: 30)"
    )
    parser.add_argument(
        "--output", default="data/raw",
        help="Direktori output untuk fail .md (default: data/raw)"
    )
    parser.add_argument(
        "--mode", choices=["seed", "random", "both"], default="both",
        help="Mod: 'seed' (artikel tetap), 'random' (rawak), 'both' (default)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Kelewatan antara permintaan (saat) untuk hormati Wikipedia (default: 1.0)"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Tentukan senarai artikel
    articles_to_fetch = []

    if args.mode in ("seed", "both"):
        articles_to_fetch.extend(DEFAULT_SEED_ARTICLES)

    if args.mode in ("random", "both"):
        n_random = max(0, args.articles - len(articles_to_fetch))
        if n_random > 0:
            print(f"Mendapatkan {n_random} artikel rawak...")
            random_titles = get_random_articles(n_random)
            articles_to_fetch.extend(random_titles)

    # Hadkan kepada bilangan yang diminta
    articles_to_fetch = articles_to_fetch[:args.articles]

    print(f"\nMemuat turun {len(articles_to_fetch)} artikel ke '{args.output}'")
    print("=" * 60)

    berhasil = 0
    total_chars = 0

    for i, title in enumerate(articles_to_fetch, 1):
        print(f"[{i:3d}/{len(articles_to_fetch)}] {title}...", end=" ")

        content = get_article_content(title)

        if content:
            filepath = save_article(title, content, args.output)
            chars = len(content)
            total_chars += chars
            berhasil += 1
            print(f"✓ ({chars:,} aksara)")
        else:
            print("✗ (dilewati)")

        # Hormati Wikipedia — jangan hantar terlalu banyak permintaan
        if i < len(articles_to_fetch):
            time.sleep(args.delay)

    print("\n" + "=" * 60)
    print(f"Selesai! {berhasil}/{len(articles_to_fetch)} artikel berjaya dimuat turun")
    print(f"Jumlah teks: {total_chars:,} aksara")
    print(f"Fail disimpan di: {args.output}")

    if berhasil == 0:
        print("\n⚠ Tiada artikel berjaya dimuat turun.")
        print("  Pastikan anda mempunyai sambungan internet.")
    elif total_chars < 50_000:
        print("\n⚠ Teks agak sedikit. Cuba tambah lebih banyak artikel atau")
        print("  masukkan data teks Melayu anda sendiri ke data/raw/ sebagai fail .md")


if __name__ == "__main__":
    main()
