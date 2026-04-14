"""
Pra-proses data untuk latihan Anney GPT.

Skrip ini:
  1. Membaca semua fail .md dari data/raw/
  2. Membersihkan dan menggabungkan teks
  3. Tokenize menggunakan model SentencePiece yang telah dilatih
  4. Bahagikan kepada train dan validation set
  5. Simpan sebagai tensor PyTorch (.pt) ke data/processed/

Mesti dijalankan SELEPAS melatih tokenizer.

Guna:
    python scripts/prepare_data.py --config configs/train_config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import sentencepiece as spm
from pathlib import Path

# Tambah root ke sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.train_tokenizer import read_markdown_files, clean_text


def tokenize_text(text: str, sp: spm.SentencePieceProcessor) -> list:
    """
    Tokenize teks kepada senarai ID token.

    Tambah token BOS di awal dan EOS di penghujung setiap perenggan
    untuk membantu model belajar sempadan ayat.
    """
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # Pecah kepada perenggan
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    all_tokens = []
    for para in paragraphs:
        if len(para) < 10:  # Abaikan perenggan terlalu pendek
            continue
        tokens = sp.encode(para, out_type=int)
        if tokens:
            all_tokens.extend([bos_id] + tokens + [eos_id])

    return all_tokens


def prepare_data(
    data_dir: str,
    tokenizer_path: str,
    output_dir: str,
    val_split: float = 0.1,
) -> None:
    """
    Proses data dan simpan sebagai fail .pt.

    Args:
        data_dir:       Direktori fail .md
        tokenizer_path: Laluan model SentencePiece (.model)
        output_dir:     Direktori output untuk fail .pt
        val_split:      Pecahan data untuk validation (0.0-0.5)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Muat tokenizer
    print(f"Memuatkan tokenizer dari: {tokenizer_path}")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    print(f"Vocab size: {sp.get_piece_size()}")

    # Baca dan bersih teks
    print(f"\nMembaca fail .md dari: {data_dir}")
    text = read_markdown_files(data_dir)

    # Tokenize
    print("\nMenokenkan teks...")
    tokens = tokenize_text(text, sp)
    total_tokens = len(tokens)
    print(f"Jumlah token: {total_tokens:,}")

    if total_tokens < 1000:
        print("⚠ Amaran: Terlalu sedikit token untuk latihan yang berkesan.")
        print("  Cuba tambah lebih banyak data teks Melayu ke data/raw/")

    # Bahagikan train/val
    n_val   = max(1, int(total_tokens * val_split))
    n_train = total_tokens - n_val

    train_tokens = tokens[:n_train]
    val_tokens   = tokens[n_train:]

    print(f"\nPembahagian data:")
    print(f"  Train : {n_train:,} token ({(1-val_split)*100:.0f}%)")
    print(f"  Val   : {n_val:,} token ({val_split*100:.0f}%)")

    # Simpan sebagai tensor
    train_tensor = torch.tensor(train_tokens, dtype=torch.long)
    val_tensor   = torch.tensor(val_tokens,   dtype=torch.long)

    train_path = os.path.join(output_dir, "train.pt")
    val_path   = os.path.join(output_dir, "val.pt")

    torch.save(train_tensor, train_path)
    torch.save(val_tensor, val_path)

    print(f"\nData disimpan:")
    print(f"  Train : {train_path}")
    print(f"  Val   : {val_path}")

    # Ringkasan metadata
    meta = {
        "total_tokens":   total_tokens,
        "train_tokens":   n_train,
        "val_tokens":     n_val,
        "vocab_size":     sp.get_piece_size(),
        "tokenizer_path": tokenizer_path,
        "val_split":      val_split,
    }

    import json
    meta_path = os.path.join(output_dir, "data_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta  : {meta_path}")

    print("\n✓ Pra-proses data selesai!")


def main():
    parser = argparse.ArgumentParser(
        description="Pra-proses data untuk latihan Anney GPT"
    )
    parser.add_argument(
        "--config",    default="configs/train_config.yaml",
        help="Laluan config YAML"
    )
    parser.add_argument(
        "--data_dir",  default="data/raw",
        help="Direktori fail .md (default: data/raw)"
    )
    parser.add_argument(
        "--tokenizer", default="tokenizer/sp_model/anney.model",
        help="Laluan model tokenizer (default: tokenizer/sp_model/anney.model)"
    )
    parser.add_argument(
        "--output",    default="data/processed",
        help="Direktori output (default: data/processed)"
    )
    args = parser.parse_args()

    # Muat config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    val_split = cfg.get("val_split", 0.1)

    # Semak tokenizer wujud
    if not os.path.isfile(args.tokenizer):
        print(f"✗ Tokenizer tidak dijumpai: {args.tokenizer}")
        print("  Sila latih tokenizer dahulu:")
        print("  python tokenizer/train_tokenizer.py")
        sys.exit(1)

    prepare_data(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        val_split=val_split,
    )


if __name__ == "__main__":
    main()
