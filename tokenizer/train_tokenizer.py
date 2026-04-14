"""
Latihan tokenizer SentencePiece untuk Anney GPT.

Skrip ini:
  1. Membaca semua fail .md dalam data/raw/
  2. Membersihkan teks (asas)
  3. Melatih model BPE SentencePiece
  4. Menyimpan model tokenizer ke tokenizer/sp_model/

Guna:
    python tokenizer/train_tokenizer.py --config configs/train_config.yaml
"""

import os
import re
import sys
import argparse
import tempfile
import yaml
import sentencepiece as spm
from pathlib import Path

# Tambah root ke sys.path supaya import berfungsi dari mana-mana direktori
sys.path.insert(0, str(Path(__file__).parent.parent))


# =========================================================================== #
#  Pembersihan teks                                                             #
# =========================================================================== #

def clean_text(text: str) -> str:
    """
    Pembersihan teks asas untuk teks Melayu dari fail Markdown.

    Apa yang dibuang:
      - Sintaks Markdown (header #, bold **, italic *, code ``, link [], image ![)
      - URL dan jalan fail
      - Baris kosong berlebihan
      - Ruang lebihan

    Apa yang dikekalkan:
      - Tanda baca asas
      - Aksara Melayu (termasuk aksara pinjaman)
      - Struktur perenggan
    """
    # Buang blok kod (``` ... ```)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)

    # Buang imej Markdown: ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)

    # Buang pautan Markdown: [text](url) — kekalkan teks
    text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)

    # Buang header Markdown (#, ##, dll.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Buang bold/italic (**text**, *text*, __text__, _text_)
    text = re.sub(r"\*{1,2}([^*]*)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]*)_{1,2}", r"\1", text)

    # Buang URL
    text = re.sub(r"https?://\S+", " ", text)

    # Buang tanda Markdown lain (>, -, *)  di awal baris
    text = re.sub(r"^[>*\-+]\s+", "", text, flags=re.MULTILINE)

    # Tukar baris kosong berganda kepada satu baris kosong
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Tukar ruang berganda kepada satu ruang
    text = re.sub(r"[ \t]+", " ", text)

    # Trim setiap baris
    lines = [line.strip() for line in text.splitlines()]
    text  = "\n".join(lines)

    return text.strip()


# =========================================================================== #
#  Baca dan gabung fail .md                                                    #
# =========================================================================== #

def read_markdown_files(data_dir: str) -> str:
    """
    Baca semua fail .md dalam direktori dan kembalikan teks yang digabung.

    Args:
        data_dir: Laluan direktori yang mengandungi fail .md

    Returns:
        Teks yang digabung dan dibersihkan
    """
    md_files = sorted(Path(data_dir).rglob("*.md"))

    if not md_files:
        raise FileNotFoundError(
            f"Tiada fail .md dijumpai dalam '{data_dir}'. "
            f"Sila letakkan fail teks Melayu anda dalam folder data/raw/"
        )

    print(f"Dijumpai {len(md_files)} fail .md dalam '{data_dir}'")

    all_text = []
    total_chars = 0

    for filepath in md_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            cleaned = clean_text(raw)
            if cleaned.strip():
                all_text.append(cleaned)
                total_chars += len(cleaned)
                print(f"  ✓ {filepath.name} ({len(cleaned):,} aksara selepas pembersihan)")
        except Exception as e:
            print(f"  ✗ {filepath.name}: {e}")

    if not all_text:
        raise ValueError("Semua fail .md kosong atau tidak boleh dibaca.")

    print(f"\nJumlah: {total_chars:,} aksara dari {len(all_text)} fail")

    return "\n\n".join(all_text)


# =========================================================================== #
#  Latih tokenizer                                                              #
# =========================================================================== #

def train_tokenizer(
    text: str,
    output_dir: str,
    vocab_size: int = 4000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
) -> None:
    """
    Latih model SentencePiece dan simpan ke output_dir.

    Args:
        text:               Teks latihan (rentetan panjang)
        output_dir:         Direktori output untuk simpan model
        vocab_size:         Saiz vocab (biasanya 4000-8000 untuk model kecil)
        character_coverage: Liputan aksara (0.9995 sesuai untuk BM)
        model_type:         'bpe' atau 'unigram'
        pad_id:             ID token pad
        unk_id:             ID token unknown
        bos_id:             ID token begin-of-sentence
        eos_id:             ID token end-of-sentence
    """
    os.makedirs(output_dir, exist_ok=True)

    # Tulis teks ke fail sementara (SentencePiece perlukan fail input)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    model_prefix = os.path.join(output_dir, "anney")

    print(f"\nMelatih tokenizer SentencePiece...")
    print(f"  vocab_size         : {vocab_size}")
    print(f"  model_type         : {model_type}")
    print(f"  character_coverage : {character_coverage}")
    print(f"  output             : {model_prefix}.model / .vocab")

    try:
        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            # Token khas untuk format chat Anney
            user_defined_symbols=["[PENGGUNA]:", "[ANNEY]:"],
            normalization_rule_name="nfkc",
            remove_extra_whitespaces=True,
            input_sentence_size=1_000_000,
            shuffle_input_sentence=True,
            # Benarkan corpus kecil menghasilkan vocab sebenar yang lebih rendah
            # daripada sasaran config supaya sample project tetap boleh berjalan.
            hard_vocab_limit=False,
        )
        print(f"\n✓ Tokenizer berjaya dilatih!")
        print(f"  Model : {model_prefix}.model")
        print(f"  Vocab : {model_prefix}.vocab")

    finally:
        os.unlink(tmp_path)

    # Simpan metadata
    import json
    meta = {
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "pad_id": pad_id, "unk_id": unk_id,
        "bos_id": bos_id, "eos_id": eos_id,
        "model_path": f"{model_prefix}.model",
    }
    meta_path = os.path.join(output_dir, "tokenizer_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta  : {meta_path}")

    # Uji tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    test_text = "Selamat datang ke Anney GPT, model bahasa Melayu."
    encoded = sp.encode(test_text, out_type=str)
    print(f"\nUji tokenizer:")
    print(f"  Input  : {test_text}")
    print(f"  Output : {encoded}")
    print(f"  Vocab size sebenar: {sp.get_piece_size()}")


# =========================================================================== #
#  Main                                                                         #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Latih tokenizer SentencePiece untuk Anney GPT")
    parser.add_argument("--config",   default="configs/train_config.yaml", help="Laluan config YAML")
    parser.add_argument("--data_dir", default="data/raw",                  help="Direktori fail .md")
    parser.add_argument("--output",   default="tokenizer/sp_model",        help="Output direktori")
    args = parser.parse_args()

    # Muat config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    tok_cfg = cfg.get("tokenizer", {})

    # Baca dan bersih teks
    print(f"Membaca fail .md dari '{args.data_dir}'...")
    text = read_markdown_files(args.data_dir)

    # Latih tokenizer
    train_tokenizer(
        text=text,
        output_dir=args.output,
        vocab_size=tok_cfg.get("vocab_size", 4000),
        character_coverage=tok_cfg.get("character_coverage", 0.9995),
        model_type=tok_cfg.get("model_type", "bpe"),
        pad_id=tok_cfg.get("pad_id", 0),
        unk_id=tok_cfg.get("unk_id", 1),
        bos_id=tok_cfg.get("bos_id", 2),
        eos_id=tok_cfg.get("eos_id", 3),
    )


if __name__ == "__main__":
    main()
