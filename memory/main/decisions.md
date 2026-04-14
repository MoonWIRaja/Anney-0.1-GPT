# Decision Log
*Append-only record of non-obvious decisions.*
*Only log decisions where future-us would wonder "why did we do it this way?"*

---

<!-- Decisions are added below this line. Format: -->
<!-- ## YYYY-MM-DD -- Short title -->
<!-- **Context**: What situation prompted this decision -->
<!-- **Decision**: What was chosen (and what was rejected) -->
<!-- **Rationale**: Why -- trade-offs, constraints, evidence -->

## 2026-04-14 -- Place MemoryAnney under `memory/`
**Context**: Repo ini sudah mempunyai struktur kod model sendiri. Import MemoryAnney terus ke root akan mencampurkan fail operasi projek dengan fail latihan model.
**Decision**: Integrasikan MemoryAnney sebagai subsistem khusus di bawah folder `memory/`, bukan ekstrak mentah di root.
**Rationale**: Ini mengekalkan sempadan yang jelas antara runtime/model code dan memory/workflow layer, sambil mengekalkan kebanyakan struktur MemoryAnney asal.

## 2026-04-14 -- Allow flexible tokenizer vocab on tiny corpora
**Context**: Sample dataset dalam `data/raw/sample_melayu.md` terlalu kecil untuk mencapai `vocab_size=4000`, menyebabkan latihan tokenizer gagal.
**Decision**: Kekalkan target vocab 4000 tetapi aktifkan `hard_vocab_limit=False` dalam SentencePiece.
**Rationale**: Pengguna masih boleh guna vocab penuh pada korpus besar, sementara sample project tetap boleh berjalan pada dataset kecil tanpa perlu ubah config setiap kali.
