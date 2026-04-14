"""
Trainer untuk Anney GPT.

Ciri-ciri:
  - Training loop stabil untuk CPU (dan GPU secara automatik jika tersedia)
  - Gradient accumulation untuk simulasi batch besar pada RAM terhad
  - Logging loss train/val dengan format yang jelas
  - Checkpoint berkala dengan semua state penting
  - Early stopping untuk elak overfitting pada dataset kecil
  - Resume training dari mana-mana checkpoint
  - device-agnostic — checkpoint yang sama boleh digunakan di CPU atau GPU
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from utils.logger import get_logger
from model.config import ModelConfig


class EarlyStopping:
    """Henti latihan awal jika val_loss tidak bertambah baik."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        """
        Returns True jika latihan patut dihentikan.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class Trainer:
    """
    Trainer utama untuk pretraining dan SFT Anney GPT.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        model_config: ModelConfig,
        tokenizer_path: str,
        checkpoint_dir: str = "checkpoints",
        log_file: str = None,
        resume_from: str = None,
    ):
        self.model_config    = model_config
        self.tokenizer_path  = tokenizer_path
        self.checkpoint_dir  = checkpoint_dir
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.config          = config

        # Logger
        self.logger = get_logger("anney.trainer", log_file)

        # Device — automatik pilih GPU jika ada, CPU jika tiada
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Menggunakan peranti: {self.device}")

        # Pindah model ke peranti
        self.model = model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Scheduler — Cosine annealing dengan warmup manual
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["max_epochs"],
            eta_min=config["learning_rate"] * 0.1,
        )

        # State latihan
        self.global_step  = 0
        self.start_epoch  = 0
        self.best_val_loss = float("inf")

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            min_delta=config.get("min_delta", 0.001),
        )

        # Resume dari checkpoint jika diberikan
        if resume_from and os.path.isfile(resume_from):
            self._load_checkpoint(resume_from)
        elif resume_from:
            self.logger.warning(f"Checkpoint tidak dijumpai: {resume_from}. Mulai dari awal.")

    # ======================================================================= #
    #  Checkpoint                                                               #
    # ======================================================================= #

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = "") -> str:
        """
        Simpan checkpoint lengkap.

        Checkpoint mengandungi:
          - model state dict
          - optimizer state dict
          - scheduler state dict
          - model config (dict)
          - tokenizer path
          - epoch semasa
          - global step
          - val loss terbaik

        Penggunaan map_location semasa load memastikan checkpoint
        boleh dipindah antara CPU dan GPU.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        tag_str  = f"_{tag}" if tag else ""
        filename = f"ckpt_epoch{epoch:03d}_step{self.global_step}{tag_str}.pt"
        path     = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "epoch":                epoch,
            "global_step":          self.global_step,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss":             val_loss,
            "best_val_loss":        self.best_val_loss,
            "model_config":         self.model_config.to_dict(),
            "tokenizer_path":       self.tokenizer_path,
            "train_config":         self.config,
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint disimpan: {path}")

        # Simpan sebagai best jika val_loss terbaik
        if val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model dikemas kini: val_loss={val_loss:.4f}")

        return path

    def _load_checkpoint(self, path: str) -> None:
        """
        Load checkpoint dengan map_location supaya boleh resume
        di peranti berbeza (CPU → GPU atau sebaliknya).
        """
        self.logger.info(f"Loading checkpoint dari: {path}")

        # map_location=self.device memastikan tensor dipindah ke peranti semasa
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.global_step   = ckpt.get("global_step", 0)
        self.start_epoch   = ckpt.get("epoch", 0) + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        self.logger.info(
            f"Resume dari epoch {ckpt.get('epoch', 0)}, "
            f"step {self.global_step}, "
            f"best_val_loss={self.best_val_loss:.4f}"
        )

    # ======================================================================= #
    #  Evaluasi                                                                 #
    # ======================================================================= #

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Kira purata val loss tanpa mengira gradient."""
        self.model.eval()
        total_loss = 0.0
        count      = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            if loss is not None:
                total_loss += loss.item()
                count      += 1

        self.model.train()
        return total_loss / max(count, 1)

    # ======================================================================= #
    #  Training Loop                                                            #
    # ======================================================================= #

    def train(self) -> None:
        """
        Main training loop.

        Gradient accumulation disokong — update berat setiap
        grad_accum_steps batch untuk simulasi batch lebih besar.
        """
        grad_accum = self.config.get("grad_accum_steps", 1)
        log_every  = self.config.get("log_every", 20)
        save_every = self.config.get("save_every", 10)
        max_epochs = self.config["max_epochs"]

        self.logger.info(
            f"Mula latihan | epoch: {self.start_epoch}→{max_epochs} | "
            f"batch: {self.config.get('batch_size')} | "
            f"grad_accum: {grad_accum} | lr: {self.config['learning_rate']}"
        )

        self.model.train()

        for epoch in range(self.start_epoch, max_epochs):
            epoch_loss = 0.0
            batch_count = 0
            t_start = time.time()

            self.optimizer.zero_grad()

            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                # Forward + loss
                _, loss = self.model(x, y)

                # Bahagi loss dengan grad_accum untuk gradient yang setara
                scaled_loss = loss / grad_accum
                scaled_loss.backward()

                # Update berat selepas grad_accum langkah
                if (i + 1) % grad_accum == 0:
                    # Clip gradient untuk kestabilan
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % log_every == 0:
                        self.logger.info(
                            f"Epoch {epoch:3d} | Step {self.global_step:6d} | "
                            f"Loss: {loss.item():.4f} | "
                            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                        )

                epoch_loss  += loss.item()
                batch_count += 1

            # Langkah terakhir jika bilangan batch bukan gandaan grad_accum
            remaining = len(self.train_loader) % grad_accum
            if remaining != 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Evaluasi
            avg_train_loss = epoch_loss / max(batch_count, 1)
            val_loss       = self._evaluate()
            elapsed        = time.time() - t_start

            self.logger.info(
                f"{'─' * 60}\n"
                f"  Epoch {epoch:3d} selesai | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Masa: {elapsed:.1f}s\n"
                f"{'─' * 60}"
            )

            # Kemaskini scheduler
            self.scheduler.step()

            # Simpan checkpoint berkala
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, val_loss)

            # Early stopping check
            if self.early_stopping(val_loss):
                self.logger.info(
                    f"Early stopping pada epoch {epoch}. "
                    f"Val loss tidak bertambah baik selama "
                    f"{self.early_stopping.patience} epoch."
                )
                self._save_checkpoint(epoch, val_loss, tag="early_stop")
                break

        # Simpan checkpoint akhir
        self.logger.info("Latihan selesai!")
        self._save_checkpoint(epoch, val_loss, tag="final")
