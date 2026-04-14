# MemoryAnney for Anney 0.1 GPT

Layer memori operasi untuk repo ini.

Folder ini memisahkan dua perkara:

1. `Anney 0.1 GPT` sebagai projek model bahasa PyTorch
2. `MemoryAnney` sebagai sistem memori kerja untuk sesi pembangunan, keputusan, reminder, projek aktif, dan diari

## Apa Yang Ada Di Sini

- `master-memory.md`: pintu masuk sistem memori
- `main/`: memori teras, sesi semasa, reminder, keputusan, dan post-mortem
- `projects/`: jejak projek aktif dan arkib
- `daily-diary/`: log sesi harian
- `library/` dan `library-items/`: pengetahuan boleh guna semula
- `plugins/anney-skills/`: skill metadata dan protokol automasi
- `Project Resources/`: format plan kerja dan rujukan projek

## Kedudukan Dalam Repo

Sistem ini sengaja diletakkan di bawah `memory/` supaya:

- tidak bercampur dengan kod latihan model
- tidak termasuk secara tidak sengaja ke dalam `data/raw/`
- mudah diurus sebagai subsistem operasi repo

## Cara Guna

1. Rujuk `memory/master-memory.md` sebagai entry point
2. Guna `memory/main/current-session.md` untuk kesinambungan sesi
3. Guna `memory/main/reminders.md` untuk open loops
4. Guna `memory/projects/active/` untuk status projek aktif

## Nota

- Folder ini bukan sebahagian daripada runtime model GPT
- Ia adalah lapisan pengurusan kerja dan konteks untuk membangunkan sistem Anney
- `MemoryAnney.zip` kekal sebagai sumber asal import; kandungan di sini sudah dikurasi untuk repo ini
