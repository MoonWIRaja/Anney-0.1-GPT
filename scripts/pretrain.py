"""
Skrip pretraining utama untuk Anney GPT.

Menjalankan fasa next-token prediction pada dataset teks Melayu
yang telah diproses.

Guna:
    # Pretraining dari awal
    python scripts/pretrain.py

    # Resume dari checkpoint
    python scripts/pretrain.py --resume checkpoints/ckpt_epoch010_step500.pt

    # Gunakan config berbeza
    python scripts/pretrain.py --config configs/train_config.yaml --model_config configs/model_config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Tambah root ke sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.gpt import AnneyGPT
from training.dataset import PretrainDataset
from training.trainer import Trainer
from utils.helpers import set_seed, get_device
from utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description="Pretrain Anney GPT")
    parser.add_argument(
        "--config",       default="configs/train_config.yaml",
        help="Laluan config latihan YAML"
    )
    parser.add_argument(
        "--model_config", default="configs/model_config.yaml",
        help="Laluan config model YAML"
    )
    parser.add_argument(
        "--resume",       default=None,
        help="Laluan checkpoint untuk resume latihan"
    )
    parser.add_argument(
        "--seed",         type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints/pretrain",
        help="Direktori untuk simpan checkpoint"
    )
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup logger
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_file = os.path.join(args.checkpoint_dir, "pretrain.log")
    logger   = get_logger("anney.pretrain", log_file)

    logger.info("=" * 60)
    logger.info("  Anney 0.1 GPT — Pretraining")
    logger.info("=" * 60)

    # Muat config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg    = cfg["pretrain"]
    tokenizer_path = cfg["tokenizer_model"]

    # Semak fail yang diperlukan
    train_data_path = cfg["train_data"]
    val_data_path   = cfg["val_data"]

    for path, name in [
        (train_data_path, "Train data"),
        (val_data_path,   "Val data"),
        (tokenizer_path,  "Tokenizer"),
    ]:
        if not os.path.isfile(path):
            logger.error(f"{name} tidak dijumpai: {path}")
            logger.error("Sila jalankan pra-proses dahulu:")
            logger.error("  python tokenizer/train_tokenizer.py")
            logger.error("  python scripts/prepare_data.py")
            sys.exit(1)

    # Muat model config
    model_cfg = ModelConfig.from_yaml(args.model_config)
    logger.info(f"\n{model_cfg}")

    # Buat dataset
    logger.info(f"\nMemuat dataset...")
    try:
        train_dataset = PretrainDataset(train_data_path, model_cfg.context_length)
    except ValueError as e:
        logger.error(str(e))
        logger.error(
            "Train data terlalu kecil untuk context_length semasa. "
            "Tambah lebih banyak data atau kecilkan context_length dalam configs/model_config.yaml."
        )
        sys.exit(1)

    try:
        val_dataset = PretrainDataset(val_data_path, model_cfg.context_length)
    except ValueError as e:
        logger.warning(str(e))
        logger.warning(
            "Validation data terlalu kecil untuk context_length semasa. "
            "Untuk meneruskan ujian awal, train dataset akan digunakan semula sebagai validation fallback."
        )
        logger.warning(
            "Untuk validation yang betul, tambah data lagi dan jalankan semula: "
            "python3.12 scripts/prepare_data.py"
        )
        val_dataset = train_dataset

    logger.info(f"  Train samples: {len(train_dataset):,}")
    logger.info(f"  Val samples  : {len(val_dataset):,}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,   # 0 untuk CPU stability
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Buat model
    model = AnneyGPT(model_cfg)
    logger.info(f"\n{model.parameter_summary()}")

    # Buat trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        model_config=model_cfg,
        tokenizer_path=tokenizer_path,
        checkpoint_dir=args.checkpoint_dir,
        log_file=log_file,
        resume_from=args.resume,
    )

    # Mula latihan
    trainer.train()

    logger.info("\n✓ Pretraining selesai!")
    logger.info(f"  Checkpoint terbaik: {args.checkpoint_dir}/best_model.pt")
    logger.info(f"  Untuk jalankan chat: python cli/chat.py --checkpoint {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()
