"""
Anney 0.1 GPT — Implementasi model bahasa causal Transformer.

Komponen:
  - Token embeddings
  - Positional embeddings
  - Masked self-attention (causal)
  - LayerNorm
  - MLP / feed-forward block
  - Residual connections
  - Linear output head (berat berkongsi dengan token embedding)

Direka untuk mudah dibaca dan dipelajari.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.config import ModelConfig


# =========================================================================== #
#  Causal Self-Attention                                                        #
# =========================================================================== #

class CausalSelfAttention(nn.Module):
    """
    Multi-head masked self-attention.
    Causal mask memastikan token pada posisi t hanya boleh
    'melihat' token pada posisi 0..t (bukan masa hadapan).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embed % config.n_heads == 0, (
            f"n_embed ({config.n_embed}) mesti boleh dibahagi dengan n_heads ({config.n_heads})"
        )

        self.n_heads = config.n_heads
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_heads

        # Projeksi Q, K, V dalam satu linear layer untuk kecekapan
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        # Projeksi output
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask — daftar sebagai buffer supaya tidak dianggap parameter
        # dan akan ikut .to(device) secara automatik
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, panjang urutan, embedding dim

        # Hitung Q, K, V serentak kemudian pisah
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        # Reshape untuk multi-head: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Skor perhatian (scaled dot-product attention)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale              # (B, n_heads, T, T)

        # Gunakan causal mask — set nilai masa hadapan kepada -inf
        att = att.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Gabungkan kepala
        y = att @ v                                          # (B, n_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # (B, T, C)

        return self.resid_dropout(self.c_proj(y))


# =========================================================================== #
#  MLP / Feed-Forward Block                                                     #
# =========================================================================== #

class MLP(nn.Module):
    """
    Feed-forward block dengan pengembangan 4x dan aktivasi GELU.
    Ini adalah 'ingatan' model — tempat pengetahuan faktual disimpan.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embed

        self.fc1     = nn.Linear(config.n_embed, hidden_dim, bias=config.bias)
        self.act     = nn.GELU()
        self.fc2     = nn.Linear(hidden_dim, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


# =========================================================================== #
#  Transformer Block                                                            #
# =========================================================================== #

class TransformerBlock(nn.Module):
    """
    Satu lapisan Transformer:
      x → LayerNorm → Attention → Tambah (residual)
        → LayerNorm → MLP      → Tambah (residual)

    Pre-norm (LayerNorm sebelum sub-layer) lebih stabil untuk latihan.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embed)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # Residual + Attention
        x = x + self.mlp(self.ln2(x))    # Residual + MLP
        return x


# =========================================================================== #
#  Anney GPT — Model Utama                                                      #
# =========================================================================== #

class AnneyGPT(nn.Module):
    """
    Model bahasa GPT kecil untuk Bahasa Melayu.

    Arkitek:
        Token Embedding + Positional Embedding
              ↓
        Dropout
              ↓
        [TransformerBlock] × n_layers
              ↓
        LayerNorm akhir
              ↓
        Linear head → logit untuk setiap token dalam vocab
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Komponen utama disimpan dalam ModuleDict supaya kemas
        self.transformer = nn.ModuleDict(dict(
            tok_emb  = nn.Embedding(config.vocab_size, config.n_embed),
            pos_emb  = nn.Embedding(config.context_length, config.n_embed),
            drop     = nn.Dropout(config.dropout),
            blocks   = nn.ModuleList([
                TransformerBlock(config) for _ in range(config.n_layers)
            ]),
            ln_final = nn.LayerNorm(config.n_embed),
        ))

        # Head output — tiada bias, berkongsi berat dengan token embedding
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight tying: embedding dan lm_head berkongsi tensor berat yang sama.
        # Ini mengurangkan bilangan parameter dengan ketara dan terbukti
        # meningkatkan kualiti model kecil.
        self.transformer.tok_emb.weight = self.lm_head.weight

        # Inisialisasi berat
        self.apply(self._init_weights)

        # Skala inisialisasi khas untuk projeksi residual (ikut GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---------------------------------------------------------------------- #
    #  Forward pass                                                            #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None
    ):
        """
        Args:
            idx:     Token ID input,   bentuk (B, T)
            targets: Token ID sasaran, bentuk (B, T) — untuk mengira loss

        Returns:
            logits: (B, T, vocab_size)
            loss:   cross-entropy loss jika targets diberikan, else None
        """
        B, T = idx.shape
        assert T <= self.config.context_length, (
            f"Urutan terlalu panjang: {T} > {self.config.context_length}"
        )

        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        # Embeddings
        tok_emb = self.transformer.tok_emb(idx)   # (B, T, n_embed)
        pos_emb = self.transformer.pos_emb(pos)   # (T, n_embed)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Lalui setiap blok Transformer
        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_final(x)          # (B, T, n_embed)
        logits = self.lm_head(x)                  # (B, T, vocab_size)

        # Kira loss jika targets diberikan
        loss = None
        if targets is not None:
            # Ratakan untuk cross_entropy: (B*T, vocab_size) vs (B*T,)
            # ignore_index=-1 digunakan dalam SFT untuk mask bahagian prompt
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ---------------------------------------------------------------------- #
    #  Penjanaan teks                                                          #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        """
        Jana token baru satu per satu menggunakan strategi sampling.

        Args:
            idx:            Urutan token awal, bentuk (1, T)
            max_new_tokens: Bilangan token baru untuk dijanakan
            temperature:    > 1 lebih rawak, < 1 lebih yakin
            top_k:          Sampel daripada k token teratas sahaja
            top_p:          Nucleus sampling — ambil token dengan cumulative prob ≤ p

        Returns:
            Urutan token yang dipanjangkan, bentuk (1, T + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Potong jika melebihi context length
            idx_cond = (
                idx if idx.size(1) <= self.config.context_length
                else idx[:, -self.config.context_length:]
            )

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # Ambil token terakhir sahaja

            # Top-K sampling
            if top_k is not None:
                top_k_clamped = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, top_k_clamped)
                threshold = topk_vals[:, [-1]]        # nilai terkecil dalam top-k
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Top-P (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Buang token di luar nukleus
                sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(sorted_remove, float("-inf"))
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    # ---------------------------------------------------------------------- #
    #  Utiliti                                                                 #
    # ---------------------------------------------------------------------- #

    def count_parameters(self) -> int:
        """Kira jumlah parameter yang boleh dilatih."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> str:
        total = self.count_parameters()
        lines = [
            f"{'=' * 45}",
            f"  Anney 0.1 GPT — Ringkasan Parameter",
            f"{'=' * 45}",
            f"  vocab_size    : {self.config.vocab_size:,}",
            f"  context_length: {self.config.context_length}",
            f"  n_embed       : {self.config.n_embed}",
            f"  n_heads       : {self.config.n_heads}",
            f"  n_layers      : {self.config.n_layers}",
            f"{'=' * 45}",
            f"  Jumlah parameter: {total:,}",
            f"{'=' * 45}",
        ]
        return "\n".join(lines)
