# Master Memory - Anney Workspace

Entry point untuk sistem memori projek ini.

## Core Loading Order

Apabila kita mahu pulihkan konteks kerja Anney, baca fail ini mengikut turutan:

1. `main/main-memory.md`
2. `main/reminders.md`
3. `main/current-session.md`
4. `projects/project-list.md`

## Direct Commands

- `Anney` -> pulihkan konteks kerja
- `save` -> simpan kemajuan penting ke fail memori
- `save diary` -> tulis log sesi
- `remind me` -> tambah reminder
- `log decision` -> rekod keputusan penting
- `list projects` -> lihat status projek aktif

## Installed Components

- Unified memory core in `main/`
- Reminder and decision tracking in `main/`
- Project tracking in `projects/`
- Daily diary archive in `daily-diary/`
- Reusable knowledge library in `library/`
- Skill metadata in `plugins/anney-skills/`

## Repo Context

Repo ini membina model bahasa kecil Bahasa Melayu menggunakan PyTorch.
Sistem memori ini mengurus konteks pembangunan repo, bukan inference model secara langsung.
