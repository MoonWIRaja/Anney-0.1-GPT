# Anney 0.1 GPT

Panduan mula guna dan cara start projek ini.

Repo ini ada 3 bahagian utama:

1. `Anney 0.1 GPT` — model bahasa kecil Bahasa Melayu berasaskan PyTorch
2. `MemoryAnney` — sistem memori kerja dalam folder `memory/` untuk session recap, reminder, keputusan, projek aktif, dan diary
3. `AnneyBelajar` — sistem self-learning dari internet (Wikipedia, DuckDuckGo) — belajar apa sahaja topik secara neutral
4. `AnneyFahamkan` — terima maklumat terus dari pengguna melalui chat — Anney ingat dan boleh jawab balik

## Cara Faham Repo Ini

Kalau ringkas:

- `memory/` = tempat urus kerja dan konteks projek
- `data/`, `tokenizer/`, `model/`, `training/`, `scripts/`, `cli/` = pipeline model GPT
- `scripts/anney_belajar.py` = sistem self-learning dari internet

Kalau anda nak mula guna projek ini, mula dengan "Mula Dari Zero" di bawah.

---

## 🚀 Mula Dari Zero (Local & Clone dari GitHub)

Arahan A-Z ini adalah untuk anda yang baru sahaja *clone* atau muat turun repo ini dari GitHub. Memandangkan GitHub tidak menyimpan data berat model (seperti `.pt`), anda tidak boleh terus mulakan perbualan (chat). Anda perlu latih (train) model ini dahulu!

### Langkah 1: Clone Repository
Buka terminal dan clone repo ini ke komputer anda:
```bash
git clone https://github.com/MoonWIRaja/Anney-0.1-GPT.git
cd "Anney 0.1 GPT"
```

### Langkah 2: Guna Python Yang Tepat (Python 3.12)
Gunakan **Python 3.12**. (Jangan guna versi yang terlalu baru seperti Python 3.14).
Semak versi anda:
```bash
python3.12 --version
```

### Langkah 3: Pasang Keperluan Berkemungkinan (Dependencies)
Pasang semua pakej pip (PyTorch dll.) yang diperlukan:
```bash
python3.12 -m pip install -r requirements.txt
```
*(Tip: Kalau anda guna MacOS dengan Homebrew dan kena "externally-managed-environment", tambah `--user --break-system-packages`)*

### Langkah 4: Bina "Otak" Semula (A to Z)
Anda baru clone repo. Anda ada data mentah latihan, tapi tidak ada model AI. Anda wajib jalankan turutan ini:

**A. Latih Tokenizer** (Ini mengajar AI patah perkataan bahasa melayu):
```bash
python3.12 tokenizer/train_tokenizer.py
```

**B. Siapkan Data Latihan Utama** (Menyemak kosa kata & saiz jadual - 50 Jutaan Parameter):
```bash
python3.12 scripts/prepare_data.py
```
*(Ia akan proses file `jsonl` dan `.md` dari Github tadi menjadi fail `.pt` di komputer anda).*

**C. Mulakan Latihan Otak (Pre-Train)**:
Langkah ini paling lama. Ia mendidik AI daripada zero untuk faham konteks susunan melayu.
```bash
python3.12 scripts/pretrain.py
```
*(Biarkan ia berjalan. Kalau tak berhenti, biarkan selagi loss terus turun. Atau, ubah fail `configs/train_config.yaml` ikut kapasiti PC anda).*

**D. Latihan Ciri Khas Teras Peribadi (Fine-Tune)**:
Setelah selesai Pretrain, kita latih Chat Logik, Emosi dan personaliti Unfiltered Anney.
```bash
python3.12 scripts/finetune.py
```

### Langkah 5: Sembang dengan Anney (Anda Berjaya!)
Bila Model (D) telah siap, lancarkan perbualan interaktif.
```bash
python3.12 cli/chat.py --checkpoint checkpoints/finetune/best_model.pt
```

---

## Cara Start MemoryAnney

Kalau anda nak guna sistem memori projek ini dulu, mula dari sini:

- `memory/master-memory.md`
- `memory/main/main-memory.md`
- `memory/main/current-session.md`
- `memory/main/reminders.md`

### Fungsi folder `memory/`

- `memory/main/current-session.md` = status sesi semasa
- `memory/main/reminders.md` = perkara yang belum selesai
- `memory/main/decisions.md` = kenapa sesuatu keputusan dibuat
- `memory/projects/active/` = fail projek aktif
- `memory/daily-diary/` = catatan sesi

### Cara guna memory system

1. Buka `memory/master-memory.md`
2. Semak `memory/main/current-session.md`
3. Semak `memory/main/reminders.md`
4. Sambung kerja ikut konteks yang direkod

### Kalau nak sambung sesi kerja

Semak fail ini dahulu:

```text
memory/master-memory.md
memory/main/current-session.md
memory/projects/active/anney-0.1-gpt.md
memory/main/reminders.md
```

---

## Cara Start Training Model

Selepas dependency siap dipasang, ikut urutan ini.

### Langkah 1: Semak sample data

Sample permulaan sudah ada di:

```text
data/raw/sample_melayu.md
```

Kalau nak hasil lebih baik, tambah lebih banyak fail `.md` Bahasa Melayu ke dalam `data/raw/`.

### Langkah 2: Latih tokenizer

```bash
python3.12 tokenizer/train_tokenizer.py
```

Output penting:

- `tokenizer/sp_model/anney.model`
- `tokenizer/sp_model/anney.vocab`

### Langkah 3: Pra-proses data

```bash
python3.12 scripts/prepare_data.py
```

Output penting:

- `data/processed/train.pt`
- `data/processed/val.pt`

### Langkah 4: Mula pretraining

```bash
python3.12 scripts/pretrain.py
```

Checkpoint akan masuk ke:

```text
checkpoints/pretrain/
```

### Langkah 5: Fine-tune untuk chat

```bash
python3.12 scripts/finetune.py
```

Checkpoint akan masuk ke:

```text
checkpoints/finetune/
```

### Langkah 6: Start chat model

```bash
python3.12 cli/chat.py --checkpoint checkpoints/finetune/best_model.pt
```

Kalau baru habis pretrain dan belum fine-tune, anda boleh uji dulu dengan:

```bash
python3.12 cli/chat.py --checkpoint checkpoints/pretrain/best_model.pt
```

---

## AnneyBelajar — Self-Learning dari Internet 🎓

Anney boleh belajar sendiri apa sahaja topik dari internet. Maklumat dicari dari pelbagai sumber secara **neutral**, kemudian disimpan ke memori dan training data supaya model makin pandai.

### Cara Guna

#### Dalam CLI Chat

Selepas start chat (`python3.12 cli/chat.py`), taip:

```
/AnneyBelajar sejarah Melaka
/AnneyBelajar quantum computing
/SenaraiIlmu
```

#### Terus dari Terminal (Standalone)

```bash
python3.12 scripts/anney_belajar.py "sejarah Melaka"
python3.12 scripts/anney_belajar.py "quantum computing" --lang en
python3.12 scripts/anney_belajar.py --list
```

### Apa Yang Berlaku Bila Guna `/AnneyBelajar`

1. **Cari** — cari maklumat dari Wikipedia BM, Wikipedia EN, dan DuckDuckGo (percuma, tanpa API key)
2. **Proses** — bersihkan teks, extract fakta utama, bina ringkasan neutral
3. **Simpan** — simpan ke `memory/library-items/learned/` dan `data/raw/` (training data)
4. **Report** — paparkan laporan lengkap dengan fakta, sumber, dan skor keyakinan

### Prinsip Neutral Learning

- Bahasa objektif — "menurut kajian...", "dinyatakan bahawa..."
- Pelbagai sumber, bukan satu sahaja
- Kalau ada kontroversi, nyatakan semua pihak
- Labelkan jelas antara fakta dan pandangan
- Skor keyakinan: Rendah / Sederhana / Tinggi

### Auto-Retrain

Setiap kali Anney belajar topik baru, teks ilmu itu akan disimpan ke `data/raw/learned_{topik}.md`. Bila anda retrain model (tokenizer → prepare_data → pretrain), ilmu baru ini akan masuk sekali ke dalam model.

### Bahasa Ikut Pengguna

Kalau anda taip topik dalam Bahasa Melayu, report akan keluar dalam BM. Kalau English, report akan keluar dalam EN.

---

## AnneyFahamkan — Suap Maklumat Terus 📝

Selain belajar dari internet, anda juga boleh ajar Anney terus melalui chat. Beri maklumat step-by-step, dan Anney akan ingat serta boleh jawab bila ditanya.

### Cara Guna

#### Dalam CLI Chat

```
/AnneyFahamkan cara masuk server Minecraft Burhan
```

Kemudian taip maklumat baris demi baris:

```
📎 1. Buka Minecraft Java Edition
📎 2. Klik Multiplayer
📎 3. Klik Add Server
📎 4. Masukkan IP: play.burhan.my
📎 5. Klik Done dan join
📎 /selesai
```

#### Senarai Semua yang Diajar

```
/SenaraiAjar
```

### Apa Yang Berlaku

1. **Simpan ke memori** — `memory/library-items/taught/{tajuk}.md`
2. **Simpan ke training data** — `data/raw/taught_{tajuk}.md`
3. **Jana Q&A** — auto-buat 12+ variasi soalan-jawapan ke `data_samples/taught_qa.jsonl`
4. **Auto-jawab** — bila sesiapa tanya soalan berkaitan, Anney jawab terus dari apa yang diajar

### Contoh Auto-Jawab

Selepas diajar, kalau user tanya:

```
[ANDA]: macam mana nak masuk server minecraft?
[ANNEY]: 1. Buka Minecraft Java Edition
         2. Klik Multiplayer
         3. Klik Add Server
         4. Masukkan IP: play.burhan.my
         5. Klik Done dan join
```

Anney jawab automatik dari maklumat yang telah diajar — tanpa perlu internet.

### Command Tambahan

- `/selesai` atau `/done` — habiskan sesi mengajar
- `/batal` atau `/cancel` — batal tanpa simpan

## Urutan Paling Ringkas Untuk Start

Kalau anda cuma nak start cepat:

```bash
python3.12 -m pip install --user --break-system-packages -r requirements.txt
python3.12 tokenizer/train_tokenizer.py
python3.12 scripts/prepare_data.py
python3.12 scripts/pretrain.py
```

---

## Fail Penting

### Sistem memori

- `memory/master-memory.md`
- `memory/main/current-session.md`
- `memory/main/reminders.md`
- `memory/main/decisions.md`
- `memory/projects/active/anney-0.1-gpt.md`

### Sistem model

- `configs/model_config.yaml`
- `configs/train_config.yaml`
- `tokenizer/train_tokenizer.py`
- `scripts/prepare_data.py`
- `scripts/pretrain.py`
- `scripts/finetune.py`
- `cli/chat.py`

### Sistem AnneyBelajar

- `scripts/anney_belajar.py` — modul utama self-learning
- `scripts/web_research.py` — engine carian (Wikipedia + DuckDuckGo)
- `scripts/knowledge_processor.py` — pemprosesan neutral
- `scripts/learn_storage.py` — penyimpanan ke memori dan training data
- `memory/plugins/anney-skills/skills/anney-belajar/SKILL.md` — skill definition
- `memory/library-items/learned/` — ilmu yang telah dipelajari
- `memory/library-items/learned/_index.json` — index semua ilmu

### Sistem AnneyFahamkan

- `scripts/anney_fahamkan.py` — modul utama terima maklumat dari pengguna
- `memory/plugins/anney-skills/skills/anney-fahamkan/SKILL.md` — skill definition
- `memory/library-items/taught/` — maklumat yang telah diajar
- `memory/library-items/taught/_index.json` — index semua ajar
- `data_samples/taught_qa.jsonl` — soalan-jawapan auto-generated

---

## Kalau Ada Error Masa Start

### `python3: command not found`

Guna interpreter yang ada pada mesin anda, tetapi disyorkan `python3.12`.

### `No matching distribution found for torch`

Biasanya ini sebab Python terlalu baru. Guna:

```bash
python3.12 -m pip install --user --break-system-packages -r requirements.txt
```

### `externally-managed-environment`

Ini biasa berlaku bila Python dipasang melalui Homebrew dan `pip` tak benarkan install terus ke environment sistem.

Kalau anda memang nak terus guna tanpa `.venv`, guna:

```bash
python3.12 -m pip install --user --break-system-packages -r requirements.txt
```

Ini akan pasang package ke user site-packages anda, bukan paksa tulis terus ke install utama Homebrew.

### `Vocabulary size too high`

Kod tokenizer sudah disediakan supaya boleh fallback untuk dataset kecil. Tapi untuk hasil lebih baik, tambah lebih banyak teks ke `data/raw/`.

### Model jawab teruk atau tak stabil

Itu normal kalau data sangat kecil. Tambah lebih banyak data, kemudian latih semula.

---

## Cara Kerja Harian Yang Disyorkan

Setiap kali buka projek ini:

1. Pastikan anda guna `python3.12`
2. Baca `memory/main/current-session.md`
3. Baca `memory/main/reminders.md`
4. Sambung kerja
5. Bila ada perubahan penting, update fail dalam `memory/`

---

## Status Semasa Repo

Setakat ini repo ini sudah:

- ada struktur projek yang kemas
- ada sample data dalam `data/raw/`
- ada subsistem MemoryAnney dalam `memory/`
- boleh latih tokenizer
- boleh pra-proses data
- sedia untuk pretraining
- ada sistem AnneyBelajar untuk self-learning dari internet 🎓
- ada sistem AnneyFahamkan untuk terima maklumat dari pengguna 📝
- auto-retrain — ilmu baru masuk ke training data
- auto-jawab — Anney boleh jawab soalan dari maklumat yang diajar

Kalau anda nak terus mula sekarang, command paling sesuai ialah:

```bash
python3.12 scripts/pretrain.py
```

Kalau nak Anney belajar topik baru dari internet:

```bash
python3.12 scripts/anney_belajar.py "sejarah Melaka"
```

Kalau nak ajar Anney terus:

```bash
python3.12 scripts/anney_fahamkan.py "cara masuk server Minecraft"
```
