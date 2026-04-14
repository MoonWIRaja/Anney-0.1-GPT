---
name: anney-belajar
description: "MUST use when user says '/AnneyBelajar', 'belajar tentang', 'learn about',
             'cari maklumat tentang', 'search about', 'study topic', or when user
             asks Anney to learn or research a topic from the internet."
---

# AnneyBelajar — Pembelajaran Kendiri dari Internet 🎓
*Self-learning engine: cari, proses, dan simpan ilmu secara neutral*

## Activation

When this skill activates, output:

`🎓 Anney sedang belajar tentang '{topik}'...`

Then execute the protocol below.

## Context Guard

| Context | Status |
|---------|--------|
| **User minta belajar topik** | ACTIVE — full protocol |
| **User guna /AnneyBelajar** | ACTIVE — full protocol |
| **User tanya soalan biasa** | DORMANT — guna model biasa |
| **User minta coding/debug** | DORMANT — do not activate |

## Protocol

### Step 1: Terima Topik
- [ ] Parse topik dari input pengguna
- [ ] Detect bahasa (BM/EN) — report ikut bahasa permintaan
- [ ] Semak kalau topik sudah pernah dipelajari

### Step 2: Cari Maklumat
- [ ] Cari dari Wikipedia BM (ms.wikipedia.org)
- [ ] Cari dari Wikipedia EN (en.wikipedia.org)
- [ ] Cari dari DuckDuckGo (percuma, tanpa API key)
- [ ] Scrape halaman web teratas untuk maklumat tambahan
- [ ] Tunjuk progress real-time di terminal

### Step 3: Proses Maklumat
- [ ] Bersihkan teks mentah (buang HTML, normalize)
- [ ] Deduplicate maklumat serupa
- [ ] Extract fakta utama
- [ ] Bina ringkasan neutral (tanpa bias)
- [ ] Bina sections terperinci
- [ ] Kira skor keyakinan

### Step 4: Simpan dan Laporkan
- [ ] Simpan ke `memory/library-items/learned/{topik}.md`
- [ ] Tambah ke training data `data/raw/learned_{topik}.md`
- [ ] Kemaskini index `memory/library-items/learned/_index.json`
- [ ] Paparkan laporan lengkap di terminal
- [ ] Nyatakan bilangan sumber, fakta, dan skor keyakinan

## Mandatory Rules

1. **NEUTRAL** — Semua ringkasan mesti neutral, tiada bias atau pendapat peribadi
2. **MULTI-SOURCE** — Mesti guna pelbagai sumber, bukan satu sahaja
3. **BAHASA IKUT USER** — Kalau user guna BM, report BM. Kalau EN, report EN
4. **OPEN SOURCE** — Hanya guna sumber percuma (Wikipedia, DuckDuckGo), tiada API key
5. **RATE LIMIT** — Hormati server — ada delay antara setiap request
6. **AUTO-RETRAIN** — Simpan ke training data supaya model makin pandai

## Edge Cases

| Situation | Behavior |
|-----------|----------|
| **Tiada internet** | Papar mesej ralat, cadangkan cuba lagi nanti |
| **Topik sudah dipelajari** | Maklumkan user, kemaskini maklumat |
| **Topik terlalu luas** | Cuba fokuskan carian, ambil top results |
| **Topik kontroversial** | Nyatakan semua pihak secara neutral |
| **Tiada hasil** | Papar mesej, cadangkan topik berkaitan |

## Level History

- **Lv.1** — Base: cari Wikipedia + DuckDuckGo, proses neutral summary, simpan ke memori + training data. (Origin: 2026-04-14, AnneyBelajar v1.0)
