# Project: Anney 0.1 GPT

## Metadata
- **Status**: Active
- **Created**: 2026-04-14
- **Last Accessed**: 2026-04-14
- **Type**: Python / PyTorch / NLP

## Summary
Membina model bahasa kecil Bahasa Melayu dari kosong menggunakan PyTorch, dengan aliran lengkap tokenizer -> data prep -> pretraining -> SFT -> CLI chat.

## Current Goals
- Kemaskan struktur repo supaya boleh terus digunakan
- Kekalkan MemoryAnney sebagai subsistem operasi projek
- Sediakan sample data dan struktur output yang diperlukan
- Sahkan skrip utama boleh dimuat dan dijalankan secara asas

## Recent Progress
### 2026-04-14
- Audit keseluruhan README dan kod pipeline
- Betulkan struktur folder `data/`, `tokenizer/sp_model/`, dan `checkpoints/`
- Integrasikan MemoryAnney ke dalam `memory/`
- Seed sample training data ke `data/raw/sample_melayu.md`
- Lulus smoke test tokenizer, data preparation, dan forward pass model

## Next Steps
- Jalankan smoke test tokenizer dan pra-proses data
- Mulakan pretraining kecil untuk pengesahan end-to-end jika diperlukan
