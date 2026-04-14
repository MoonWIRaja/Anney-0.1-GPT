---
name: anney-fahamkan
description: "MUST use when user says '/AnneyFahamkan', 'fahamkan', 'ajar anney',
             'teach anney', 'ingat ini', 'remember this', or when user wants to
             manually feed information for Anney to remember and learn."
---

# AnneyFahamkan — Ajar Anney Terus 📝
*Suap maklumat terus supaya Anney faham dan ingat*

## Activation

When this skill activates, output:

`📝 Anney bersedia belajar tentang '{tajuk}'...`

Then execute the protocol below.

## Context Guard

| Context | Status |
|---------|--------|
| **User nak ajar Anney maklumat** | ACTIVE — full protocol |
| **User guna /AnneyFahamkan** | ACTIVE — full protocol |
| **User tanya soalan biasa** | DORMANT — search taught knowledge |
| **User minta belajar dari internet** | DORMANT — guna AnneyBelajar |

## Protocol

### Step 1: Terima Tajuk
- [ ] Parse tajuk dari command
- [ ] Detect bahasa (BM/EN)
- [ ] Semak kalau tajuk sudah ada

### Step 2: Kumpul Maklumat
- [ ] Paparkan prompt untuk user taip maklumat
- [ ] Terima input baris demi baris
- [ ] User taip /selesai atau /done untuk habiskan

### Step 3: Proses
- [ ] Auto-extract tags dari tajuk dan content
- [ ] Jana variasi soalan-jawapan untuk training
- [ ] Bina TaughtKnowledge object

### Step 4: Simpan
- [ ] Simpan ke `memory/library-items/taught/{slug}.md`
- [ ] Simpan ke `data/raw/taught_{slug}.md` (training text)
- [ ] Tambah Q&A ke `data_samples/taught_qa.jsonl` (chat training)
- [ ] Kemaskini index

### Step 5: Confirm
- [ ] Papar report (tajuk, baris, tags, Q&A count)
- [ ] Maklumkan Anney sekarang boleh jawab soalan berkaitan

## Auto-Answer

Bila user tanya soalan biasa, sistem akan:
1. Cari dalam maklumat diajar (search by title + tags)
2. Kalau ada match, jawab terus dari maklumat diajar
3. Kalau tiada, fall through ke model GPT biasa

## Mandatory Rules

1. **INGAT** — Semua maklumat yang diajar mesti disimpan
2. **BOLEH JAWAB** — Selepas diajar, Anney mesti boleh jawab soalan berkaitan
3. **AUTO-TRAINING** — Simpan Q&A pairs untuk retrain
4. **MULTI-FORMAT** — Simpan dalam 3 format: memory, training text, chat JSONL

## Edge Cases

| Situation | Behavior |
|-----------|----------|
| **User batal (/batal)** | Batal, tiada yang disimpan |
| **Tiada input** | Maklumkan, batal |
| **Tajuk sudah ada** | Kemaskini maklumat sedia ada |

## Level History

- **Lv.1** — Base: terima maklumat, simpan, auto-jawab. (Origin: 2026-04-14, AnneyFahamkan v1.0)
