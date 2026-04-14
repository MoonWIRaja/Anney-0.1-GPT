"""
Engine Penyelidikan Web untuk AnneyBelajar.

Cari maklumat dari pelbagai sumber percuma:
  - Wikipedia Bahasa Melayu (ms.wikipedia.org)
  - Wikipedia English (en.wikipedia.org)
  - DuckDuckGo Search
  - Web Scraping (top results)

Semua sumber percuma dan open source, tiada API key diperlukan.

Guna:
    from scripts.web_research import WebResearcher
    researcher = WebResearcher()
    results = researcher.research("quantum computing", lang="ms")
"""

import os
import re
import sys
import time
import json
import requests
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))


# =========================================================================== #
#  Konfigurasi                                                                  #
# =========================================================================== #

USER_AGENT = (
    "AnneyGPT/0.1 (self-learning engine; "
    "neutral knowledge acquisition)"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/json",
    "Accept-Language": "ms,en;q=0.9",
}

REQUEST_TIMEOUT = 15
RATE_LIMIT_DELAY = 1.0  # saat antara request


# =========================================================================== #
#  Data class untuk hasil penyelidikan                                          #
# =========================================================================== #

class ResearchResult:
    """Satu unit maklumat dari penyelidikan."""

    def __init__(
        self,
        source: str,
        title: str,
        url: str,
        content: str,
        language: str,
        snippet: str = "",
    ):
        self.source = source
        self.title = title
        self.url = url
        self.content = content
        self.language = language
        self.snippet = snippet

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "language": self.language,
            "snippet": self.snippet,
        }

    def __repr__(self):
        return f"ResearchResult(source={self.source!r}, title={self.title!r}, lang={self.language!r})"


# =========================================================================== #
#  Wikipedia API                                                                #
# =========================================================================== #

class WikipediaSource:
    """Cari dan ambil kandungan dari Wikipedia."""

    def __init__(self, lang: str = "ms"):
        self.lang = lang
        self.api_url = f"https://{lang}.wikipedia.org/w/api.php"

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Cari tajuk artikel yang berkaitan."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "utf8": True,
        }
        try:
            resp = requests.get(
                self.api_url, params=params,
                headers=HEADERS, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("query", {}).get("search", [])
            return [r["title"] for r in results]
        except Exception as e:
            print(f"  ⚠ Wikipedia {self.lang} search error: {e}")
            return []

    def get_article(self, title: str) -> Optional[ResearchResult]:
        """Dapatkan kandungan penuh artikel."""
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
            resp = requests.get(
                self.api_url, params=params,
                headers=HEADERS, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})

            for page_id, page in pages.items():
                if page_id == "-1":
                    return None
                extract = page.get("extract", "")
                if extract and len(extract) > 200:
                    # Bersihkan teks
                    content = self._clean_text(extract)
                    url = f"https://{self.lang}.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                    return ResearchResult(
                        source=f"Wikipedia ({self.lang.upper()})",
                        title=title,
                        url=url,
                        content=content,
                        language=self.lang,
                        snippet=content[:300] + "..." if len(content) > 300 else content,
                    )
        except Exception as e:
            print(f"  ⚠ Wikipedia {self.lang} article error for '{title}': {e}")
        return None

    def research(self, query: str, max_articles: int = 3) -> list[ResearchResult]:
        """Cari dan ambil artikel berkaitan."""
        results = []
        titles = self.search(query, limit=max_articles + 2)

        for title in titles[:max_articles]:
            article = self.get_article(title)
            if article:
                results.append(article)
            time.sleep(RATE_LIMIT_DELAY * 0.5)

        return results

    @staticmethod
    def _clean_text(text: str) -> str:
        """Bersihkan teks Wikipedia."""
        # Buang bahagian tidak berguna
        for section in [
            "== Lihat juga ==", "== See also ==",
            "== Rujukan ==", "== References ==",
            "== Pautan luar ==", "== External links ==",
            "== Nota ==", "== Notes ==",
            "== Bibliografi ==", "== Bibliography ==",
        ]:
            idx = text.find(section)
            if idx != -1:
                text = text[:idx]

        # Bersih whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


# =========================================================================== #
#  DuckDuckGo Search                                                            #
# =========================================================================== #

class DuckDuckGoSource:
    """Cari maklumat menggunakan DuckDuckGo (percuma, tanpa API key)."""

    # DuckDuckGo Instant Answer API (percuma)
    INSTANT_API = "https://api.duckduckgo.com/"
    # DuckDuckGo HTML search (fallback)
    HTML_SEARCH = "https://html.duckduckgo.com/html/"

    def research(self, query: str, max_results: int = 5) -> list[ResearchResult]:
        """Cari maklumat dari DuckDuckGo."""
        results = []

        # Cuba Instant Answer API dulu
        instant = self._instant_answer(query)
        if instant:
            results.append(instant)

        # Kemudian HTML search untuk lebih banyak hasil
        html_results = self._html_search(query, max_results=max_results)
        results.extend(html_results)

        return results[:max_results]

    def _instant_answer(self, query: str) -> Optional[ResearchResult]:
        """Guna DuckDuckGo Instant Answer API."""
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1,
        }
        try:
            resp = requests.get(
                self.INSTANT_API, params=params,
                headers=HEADERS, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            # Abstract — biasanya dari Wikipedia
            abstract = data.get("Abstract", "")
            abstract_url = data.get("AbstractURL", "")
            abstract_source = data.get("AbstractSource", "DuckDuckGo")

            if abstract and len(abstract) > 100:
                return ResearchResult(
                    source=f"DuckDuckGo ({abstract_source})",
                    title=data.get("Heading", query),
                    url=abstract_url,
                    content=abstract,
                    language="en",  # DDG biasanya English
                    snippet=abstract[:300],
                )

            # Related topics
            related = data.get("RelatedTopics", [])
            if related:
                combined = []
                for topic in related[:5]:
                    text = topic.get("Text", "")
                    if text:
                        combined.append(text)
                if combined:
                    content = "\n\n".join(combined)
                    return ResearchResult(
                        source="DuckDuckGo (Related)",
                        title=data.get("Heading", query),
                        url=data.get("AbstractURL", ""),
                        content=content,
                        language="en",
                        snippet=content[:300],
                    )
        except Exception as e:
            print(f"  ⚠ DuckDuckGo Instant API error: {e}")
        return None

    def _html_search(self, query: str, max_results: int = 5) -> list[ResearchResult]:
        """Scrape DuckDuckGo HTML search results."""
        results = []
        try:
            resp = requests.post(
                self.HTML_SEARCH,
                data={"q": query},
                headers={
                    **HEADERS,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            html = resp.text

            # Parse results — cari pattern DuckDuckGo HTML results
            # Pattern: <a class="result__a" href="...">title</a>
            # dan <a class="result__snippet">snippet</a>
            link_pattern = re.compile(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                re.DOTALL,
            )
            snippet_pattern = re.compile(
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                re.DOTALL,
            )

            links = link_pattern.findall(html)
            snippets = snippet_pattern.findall(html)

            for i, (url, title) in enumerate(links[:max_results]):
                # Bersihkan HTML tags
                title_clean = re.sub(r"<[^>]+>", "", title).strip()
                snippet = ""
                if i < len(snippets):
                    snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

                if title_clean and url:
                    # DuckDuckGo redirect URLs
                    if "uddg=" in url:
                        real_url = re.search(r"uddg=([^&]+)", url)
                        if real_url:
                            from urllib.parse import unquote
                            url = unquote(real_url.group(1))

                    results.append(ResearchResult(
                        source="DuckDuckGo",
                        title=title_clean,
                        url=url,
                        content=snippet,  # Hanya snippet, content penuh dari scraping
                        language="en",
                        snippet=snippet,
                    ))

        except Exception as e:
            print(f"  ⚠ DuckDuckGo HTML search error: {e}")

        return results


# =========================================================================== #
#  Web Scraper — ambil teks penuh dari URL                                      #
# =========================================================================== #

class WebScraper:
    """Scrape teks dari halaman web."""

    # Domain yang selamat untuk scrape
    SAFE_DOMAINS = [
        "wikipedia.org", "britannica.com", "sciencedaily.com",
        "bbc.com", "reuters.com", "thestar.com.my",
        "bernama.com", "astroawani.com", "malaysiakini.com",
        "nationalgeographic.com", "nasa.gov", "who.int",
    ]

    def scrape(self, url: str, max_chars: int = 5000) -> Optional[str]:
        """Ambil teks utama dari halaman web."""
        try:
            # Pastikan domain selamat
            domain = urlparse(url).netloc.lower()
            is_safe = any(safe in domain for safe in self.SAFE_DOMAINS)
            if not is_safe:
                # Masih cuba, tapi lebih berhati-hati
                pass

            resp = requests.get(
                url, headers=HEADERS, timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()

            # Hanya proses HTML
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None

            html = resp.text
            text = self._extract_text(html)

            if text and len(text) > 200:
                return text[:max_chars]

        except Exception as e:
            print(f"  ⚠ Scrape error for {url}: {e}")
        return None

    @staticmethod
    def _extract_text(html: str) -> str:
        """Extract teks bermakna dari HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
        except ImportError:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
            except ImportError:
                # Fallback: regex basic
                text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text)
                return text.strip()

        # Buang script, style, nav, footer, sidebar
        for tag in soup.find_all(["script", "style", "nav", "footer", "aside", "header"]):
            tag.decompose()

        # Ambil teks dari paragraf, headings, dan list items
        meaningful_tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "td", "th"])
        paragraphs = []
        for tag in meaningful_tags:
            text = tag.get_text(strip=True)
            if text and len(text) > 30:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)


# =========================================================================== #
#  Web Researcher — koordinator utama                                           #
# =========================================================================== #

class WebResearcher:
    """
    Koordinator penyelidikan web.
    Cari dari pelbagai sumber dan kumpulkan hasilnya.
    """

    def __init__(self):
        self.wiki_ms = WikipediaSource(lang="ms")
        self.wiki_en = WikipediaSource(lang="en")
        self.ddg = DuckDuckGoSource()
        self.scraper = WebScraper()

    def research(
        self,
        topic: str,
        request_lang: str = "ms",
        max_sources: int = 10,
        verbose: bool = True,
        progress_callback=None,
    ) -> list[ResearchResult]:
        """
        Jalankan penyelidikan menyeluruh tentang sesuatu topik.

        Args:
            topic:             Topik yang hendak dikaji
            request_lang:      Bahasa permintaan asal (ms/en)
            max_sources:       Bilangan sumber maksimum
            verbose:           Tunjuk progress di terminal
            progress_callback: Fungsi callback untuk progress updates

        Returns:
            list of ResearchResult
        """
        all_results = []

        def _progress(msg):
            if verbose:
                print(msg, flush=True)
            if progress_callback:
                progress_callback(msg)

        # ── 1. Wikipedia BM ─────────────────────────────────────────── #
        _progress("  📡 Mencari di Wikipedia BM...", )
        wiki_ms_results = self.wiki_ms.research(topic, max_articles=3)
        all_results.extend(wiki_ms_results)
        _progress(f"  📡 Wikipedia BM... ✓ ({len(wiki_ms_results)} artikel)")
        time.sleep(RATE_LIMIT_DELAY)

        # ── 2. Wikipedia EN ─────────────────────────────────────────── #
        _progress("  📡 Mencari di Wikipedia EN...")
        wiki_en_results = self.wiki_en.research(topic, max_articles=3)
        all_results.extend(wiki_en_results)
        _progress(f"  📡 Wikipedia EN... ✓ ({len(wiki_en_results)} artikel)")
        time.sleep(RATE_LIMIT_DELAY)

        # ── 3. DuckDuckGo ───────────────────────────────────────────── #
        _progress("  🔍 Mencari di DuckDuckGo...")
        ddg_results = self.ddg.research(topic, max_results=5)
        all_results.extend(ddg_results)
        _progress(f"  🔍 DuckDuckGo... ✓ ({len(ddg_results)} hasil)")
        time.sleep(RATE_LIMIT_DELAY)

        # ── 4. Web Scraping (top DDG results) ────────────────────────── #
        scrape_count = 0
        urls_scraped = set()

        # Scrape DDG results yang ada URL tapi content pendek
        scrapeable = [
            r for r in ddg_results
            if r.url
            and len(r.content) < 500
            and r.url not in urls_scraped
            and "wikipedia.org" not in r.url  # Wikipedia dah diambil
        ]

        if scrapeable:
            _progress("  🌐 Membaca halaman web...")
            for result in scrapeable[:4]:
                full_text = self.scraper.scrape(result.url)
                if full_text:
                    result.content = full_text
                    scrape_count += 1
                    urls_scraped.add(result.url)
                time.sleep(RATE_LIMIT_DELAY)
            _progress(f"  🌐 Halaman web... ✓ ({scrape_count} halaman)")

        # ── 5. Deduplicate ───────────────────────────────────────────── #
        seen_titles = set()
        unique = []
        for r in all_results:
            key = r.title.lower().strip()
            if key not in seen_titles and r.content:
                seen_titles.add(key)
                unique.append(r)

        final = unique[:max_sources]

        _progress(f"\n  📊 Jumlah: {len(final)} sumber unik dikumpulkan")

        return final


# =========================================================================== #
#  Standalone test                                                              #
# =========================================================================== #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Web Researcher")
    parser.add_argument("topic", help="Topik untuk dicari")
    parser.add_argument("--lang", default="ms", help="Bahasa (ms/en)")
    args = parser.parse_args()

    researcher = WebResearcher()
    results = researcher.research(args.topic, request_lang=args.lang)

    print(f"\n{'=' * 60}")
    print(f"Hasil penyelidikan: {args.topic}")
    print(f"{'=' * 60}")

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.source} — {r.title}")
        print(f"    URL: {r.url}")
        print(f"    Content: {r.content[:200]}...")
        print(f"    Lang: {r.language}")
