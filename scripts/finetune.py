"""
Skrip supervised fine-tuning (SFT) untuk Anney GPT.

Fine-tunes model yang telah di-pretrain pada data chat format:
    [PENGGUNA]: <soalan> [ANNEY]: <jawapan>

Guna:
    # Fine-tune dari checkpoint pretrain terbaik
    python scripts/finetune.py --base_checkpoint checkpoints/pretrain/best_model.pt

    # Resume fine-tuning
    python scripts/finetune.py --resume checkpoints/finetune/ckpt_epoch005_step200.pt

    # Gunakan data chat sendiri
    python scripts/finetune.py --sft_data path/ke/data_chat.jsonl
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Tambah root ke sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.gpt import AnneyGPT
from training.dataset import SFTDataset
from training.trainer import Trainer
from utils.helpers import set_seed
from utils.logger import get_logger


def load_pretrained_model(checkpoint_path: str, device: torch.device):
    """
    Load model dari checkpoint pretrain.

    Returns:
        (model, model_config, tokenizer_path)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_config = ModelConfig.from_dict(ckpt["model_config"])
    model        = AnneyGPT(model_config)
    model.load_state_dict(ckpt["model_state_dict"])

    tokenizer_path = ckpt.get("tokenizer_path", "tokenizer/sp_model/anney.model")

    return model, model_config, tokenizer_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Anney GPT untuk chat")
    parser.add_argument(
        "--config",           default="configs/train_config.yaml",
        help="Laluan config YAML"
    )
    parser.add_argument(
        "--base_checkpoint",  default="checkpoints/pretrain/best_model.pt",
        help="Checkpoint pretrain sebagai titik permulaan"
    )
    parser.add_argument(
        "--sft_data",         default=None,
        help="Laluan data SFT JSONL (override config)"
    )
    parser.add_argument(
        "--resume",           default=None,
        help="Resume fine-tuning dari checkpoint SFT"
    )
    parser.add_argument(
        "--checkpoint_dir",   default="checkpoints/finetune",
        help="Direktori untuk simpan checkpoint SFT"
    )
    parser.add_argument(
        "--seed",             type=int, default=42
    )
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_file = os.path.join(args.checkpoint_dir, "finetune.log")
    logger   = get_logger("anney.finetune", log_file)

    logger.info("=" * 60)
    logger.info("  Anney 0.1 GPT — Supervised Fine-Tuning (SFT)")
    logger.info("=" * 60)

    # Muat config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sft_cfg = cfg["finetune"]

    # Tentukan sumber data SFT
    sft_data_path = args.sft_data or sft_cfg.get("sft_data", "data_samples/chat_samples.jsonl")

    if not os.path.isfile(sft_data_path):
        logger.error(f"Data SFT tidak dijumpai: {sft_data_path}")
        sys.exit(1)

    # Pilih checkpoint sumber
    if args.resume:
        # Resume SFT yang tergendala
        source_ckpt = args.resume
        logger.info(f"Resume SFT dari: {source_ckpt}")
    elif os.path.isfile(args.base_checkpoint):
        source_ckpt = args.base_checkpoint
        logger.info(f"Memulakan SFT dari pretrain checkpoint: {source_ckpt}")
    else:
        logger.warning(
            f"Checkpoint pretrain tidak dijumpai: {args.base_checkpoint}\n"
            "Melatih dari awal (tidak disyorkan — pretrain dahulu!)"
        )
        source_ckpt = None

    # Tentukan device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Peranti: {device}")

    # Load model
    if source_ckpt:
        model, model_cfg, tokenizer_path = load_pretrained_model(source_ckpt, device)
        logger.info(f"Model dimuat dari checkpoint")
    else:
        # Mulai dari awal
        from model.config import ModelConfig
        model_cfg      = ModelConfig.from_yaml("configs/model_config.yaml")
        model          = AnneyGPT(model_cfg)
        tokenizer_path = cfg["tokenizer_model"]

    logger.info(f"\n{model.parameter_summary()}")

    # Buat dataset SFT
    logger.info(f"\nMemuat data SFT dari: {sft_data_path}")
    full_dataset = SFTDataset(
        data_path=sft_data_path,
        tokenizer_path=tokenizer_path,
        context_length=model_cfg.context_length,
    )

    # Bahagikan train/val (90/10)
    n_val   = max(1, int(len(full_dataset) * 0.1))
    n_train = len(full_dataset) - n_val

    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"  Train samples: {n_train}")
    logger.info(f"  Val samples  : {n_val}")

    # DataLoader
    batch_size = sft_cfg.get("batch_size", 8)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0,
    )

    # Trainer
    # SFT menggunakan learning rate lebih rendah dari pretrain
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=sft_cfg,
        model_config=model_cfg,
        tokenizer_path=tokenizer_path,
        checkpoint_dir=args.checkpoint_dir,
        log_file=log_file,
        resume_from=args.resume,
    )

    # Mula fine-tuning
    trainer.train()

    logger.info("\n✓ Fine-tuning selesai!")
    logger.info(f"  Model terbaik: {args.checkpoint_dir}/best_model.pt")
    logger.info(f"  Untuk bersembang: python cli/chat.py --checkpoint {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()
