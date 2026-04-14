"""
ModelConfig — konfigurasi model Anney GPT.
Semua hyperparameter boleh diubah melalui fail YAML atau JSON.
"""

from dataclasses import dataclass, asdict
import json
import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 4000
    context_length: int = 256
    n_embed: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    bias: bool = False

    # ------------------------------------------------------------------ #
    #  Loader helpers                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # ------------------------------------------------------------------ #
    #  Saver helpers                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f)

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        lines = ["ModelConfig:"]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
