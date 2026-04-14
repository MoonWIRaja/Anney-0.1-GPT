"""
Dataset untuk pretraining dan supervised fine-tuning (SFT).

PretrainDataset  — untuk next-token prediction pada teks mentah
SFTDataset       — untuk fine-tuning format [PENGGUNA]/[ANNEY]
"""

import json
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from typing import Tuple


# =========================================================================== #
#  Pretraining Dataset                                                          #
# =========================================================================== #

class PretrainDataset(Dataset):
    """
    Dataset untuk pretraining.

    Data yang dimuat ialah tensor 1D token ID (hasil tokenisasi keseluruhan
    korpus). Setiap sample ialah tetingkap gelongsor panjang (context_length + 1),
    di mana input = [:context_length] dan sasaran = [1:context_length+1].
    """

    def __init__(self, data_path: str, context_length: int):
        """
        Args:
            data_path:      Laluan ke fail .pt yang mengandungi tensor 1D token ID
            context_length: Panjang konteks model
        """
        self.data = torch.load(data_path, map_location="cpu")
        self.context_length = context_length

        if len(self.data) <= context_length:
            raise ValueError(
                f"Data terlalu pendek ({len(self.data)} token) "
                f"untuk context_length={context_length}. "
                f"Perlukan sekurang-kurangnya {context_length + 1} token."
            )

    def __len__(self) -> int:
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]
        x = chunk[:-1].clone()   # input: token 0..T-1
        y = chunk[1:].clone()    # sasaran: token 1..T
        return x, y


# =========================================================================== #
#  SFT Dataset                                                                  #
# =========================================================================== #

class SFTDataset(Dataset):
    """
    Dataset untuk supervised fine-tuning format chat.

    Format data (JSONL):
        {"pengguna": "Apa itu Malaysia?", "anney": "Malaysia ialah sebuah negara..."}

    Format prompt yang digunakan:
        [PENGGUNA]: <soalan> [ANNEY]: <jawapan>

    Loss dikira HANYA pada bahagian [ANNEY] (bukan pada bahagian prompt).
    Ini membolehkan model belajar untuk menjawab, bukan menghafal soalan.
    """

    PROMPT_TEMPLATE = "[PENGGUNA]: {pengguna} [ANNEY]: "
    FULL_TEMPLATE   = "[PENGGUNA]: {pengguna} [ANNEY]: {anney}"

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        context_length: int,
    ):
        """
        Args:
            data_path:      Laluan ke fail JSONL data chat
            tokenizer_path: Laluan ke model SentencePiece (.model)
            context_length: Panjang konteks model
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        self.context_length = context_length

        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    if "pengguna" in sample and "anney" in sample:
                        self.samples.append(sample)
                    else:
                        print(f"[SFTDataset] Baris {line_num} diabaikan: tiada kunci 'pengguna'/'anney'")
                except json.JSONDecodeError as e:
                    print(f"[SFTDataset] Baris {line_num} ralat JSON: {e}")

        if not self.samples:
            raise ValueError(f"Tiada sample yang sah dalam {data_path}")

        print(f"[SFTDataset] Loaded {len(self.samples)} samples dari {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        pengguna = sample["pengguna"].strip()
        anney    = sample["anney"].strip()

        # Tokenize keseluruhan perbualan
        full_text   = self.FULL_TEMPLATE.format(pengguna=pengguna, anney=anney)
        prompt_text = self.PROMPT_TEMPLATE.format(pengguna=pengguna)

        full_tokens   = self.sp.encode(full_text,   out_type=int)
        prompt_tokens = self.sp.encode(prompt_text, out_type=int)

        # Potong jika terlalu panjang (simpan sekurang-kurangnya 1 token respons)
        max_len = self.context_length + 1
        if len(full_tokens) > max_len:
            full_tokens = full_tokens[:max_len]

        # Pastikan ada sekurang-kurangnya 2 token (untuk buat x dan y)
        if len(full_tokens) < 2:
            # Fallback — buat tensor kosong dengan ignore
            x = torch.zeros(self.context_length, dtype=torch.long)
            y = torch.full((self.context_length,), -1, dtype=torch.long)
            return x, y

        tokens_tensor = torch.tensor(full_tokens, dtype=torch.long)
        x = tokens_tensor[:-1]   # input
        y = tokens_tensor[1:].clone()   # sasaran

        # Mask bahagian prompt dalam y — gunakan -1 sebagai ignore_index
        # Model tidak perlu belajar menjana semula soalan pengguna
        prompt_len = min(len(prompt_tokens) - 1, len(y))
        if prompt_len > 0:
            y[:prompt_len] = -1

        # Pad kepada context_length dengan token 0 (pad) untuk x, -1 untuk y
        current_len = len(x)
        pad_len = self.context_length - current_len

        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)])
        elif pad_len < 0:
            x = x[:self.context_length]
            y = y[:self.context_length]

        return x, y
