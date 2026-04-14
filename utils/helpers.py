"""
Fungsi pembantu pelbagai guna untuk Anney GPT.
"""

import re
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set seed untuk reproducibility merentas Python, NumPy, dan PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_tokens_in_file(path: str) -> int:
    """Kira jumlah token dalam fail .pt yang mengandungi tensor 1D."""
    data = torch.load(path, map_location="cpu")
    return len(data)


def get_device() -> torch.device:
    """Pulangkan peranti terbaik yang tersedia."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def human_readable_size(num_bytes: int) -> str:
    """Tukar saiz bytes kepada rentetan yang mudah dibaca."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


# =========================================================================== #
#  JSON Extraction — untuk mod output JSON                                      #
# =========================================================================== #

def extract_json(text: str) -> Optional[dict]:
    """
    Cuba extract objek JSON dari teks output model.

    Strategi (ikut tertib):
    1. Cari blok ```json ... ``` (markdown code fence)
    2. Cari { ... } pertama yang valid
    3. Cuba parse keseluruhan teks sebagai JSON

    Args:
        text: Teks output dari model

    Returns:
        dict jika berjaya, None jika gagal
    """
    # Strategi 1: markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategi 2: Cari kurung kurawal pertama yang seimbang
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategi 3: Cuba parse seluruh teks (selepas trim)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    return None


def format_json_prompt(user_query: str, schema_hint: str = None) -> str:
    """
    Bina prompt khas untuk mendorong model keluarkan JSON.

    Args:
        user_query:  Pertanyaan pengguna
        schema_hint: Hint tentang struktur JSON yang dikehendaki

    Returns:
        Prompt yang diformatkan untuk JSON output
    """
    schema_part = f"\nStruktur JSON: {schema_hint}" if schema_hint else ""
    return (
        f"[PENGGUNA]: {user_query}{schema_part}\n"
        f"Sila jawab dalam format JSON sahaja. "
        f"[ANNEY]: "
    )
