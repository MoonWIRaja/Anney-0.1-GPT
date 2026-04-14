"""
CLI Inferens Anney GPT — bersembang dengan model secara terminal.

Ciri:
  - Sembang interaktif format [PENGGUNA]/[ANNEY]
  - Kawalan penjanaan: temperature, top_k, top_p, max_new_tokens
  - Mod JSON output dengan auto-extraction dan validasi
  - Load mana-mana checkpoint
  - Paparan maklumat model
  - AnneyBelajar — self-learning dari internet
  - AnneyFahamkan — terima maklumat terus dari pengguna

Guna:
    python cli/chat.py
    python cli/chat.py --checkpoint checkpoints/finetune/best_model.pt
    python cli/chat.py --temperature 0.8 --top_k 40 --max_new_tokens 200
    python cli/chat.py --json_mode

Arahan dalam sembang:
    /help                    — tunjuk arahan
    /info                    — maklumat model
    /clear                   — kosongkan sejarah
    /set temp 0.7            — tukar temperature
    /json                    — togol mod JSON
    /AnneyBelajar <topik>    — belajar topik dari internet
    /AnneyFahamkan <tajuk>   — suap maklumat terus kepada Anney
    /SenaraiIlmu             — senarai semua ilmu yang dipelajari
    /SenaraiAjar             — senarai semua maklumat yang diajar
    /quit                    — keluar
"""

import os
import sys
import json
import argparse
import torch
import sentencepiece as spm
from pathlib import Path

# Tambah root ke sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.gpt import AnneyGPT
from utils.helpers import extract_json, format_json_prompt, get_device


# =========================================================================== #
#  Load model dari checkpoint                                                   #
# =========================================================================== #

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load model, tokenizer, dan config dari checkpoint.

    Menggunakan map_location untuk memastikan ia berfungsi sama ada
    checkpoint dilatih di CPU atau GPU.

    Returns:
        (model, sp_tokenizer, model_config, checkpoint_meta)
    """
    if not os.path.isfile(checkpoint_path):
        print(f"✗ Checkpoint tidak dijumpai: {checkpoint_path}")
        print("  Sila latih model dahulu:")
        print("  python scripts/pretrain.py")
        sys.exit(1)

    print(f"Memuatkan checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Bina model dari config yang tersimpan dalam checkpoint
    model_cfg = ModelConfig.from_dict(ckpt["model_config"])
    model     = AnneyGPT(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = ckpt.get("tokenizer_path", "tokenizer/sp_model/anney.model")

    if not os.path.isfile(tokenizer_path):
        # Cuba cari dalam lokasi standard
        for fallback in [
            "tokenizer/sp_model/anney.model",
            "../tokenizer/sp_model/anney.model",
        ]:
            if os.path.isfile(fallback):
                tokenizer_path = fallback
                break
        else:
            print(f"✗ Tokenizer tidak dijumpai: {tokenizer_path}")
            sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    meta = {
        "epoch":       ckpt.get("epoch", "?"),
        "global_step": ckpt.get("global_step", "?"),
        "val_loss":    ckpt.get("val_loss", "?"),
    }

    return model, sp, model_cfg, meta


# =========================================================================== #
#  Penjanaan jawapan                                                            #
# =========================================================================== #

def generate_response(
    model:          AnneyGPT,
    sp:             spm.SentencePieceProcessor,
    prompt_text:    str,
    device:         torch.device,
    max_new_tokens: int   = 150,
    temperature:    float = 0.9,
    top_k:          int   = 50,
    top_p:          float = None,
    json_mode:      bool  = False,
) -> str:
    """
    Jana respons model untuk prompt yang diberikan.

    Args:
        model:          Model AnneyGPT
        sp:             SentencePiece tokenizer
        prompt_text:    Teks prompt (sudah diformatkan)
        device:         Peranti (CPU/GPU)
        max_new_tokens: Bilangan token baru maksimum
        temperature:    Suhu pensampelan
        top_k:          Top-K sampling
        top_p:          Nucleus sampling (None = tidak digunakan)
        json_mode:      Jika True, cuba extract JSON dari output

    Returns:
        Teks respons model
    """
    # Tokenize prompt
    input_ids = sp.encode(prompt_text, out_type=int)

    # Hadkan kepada context_length - max_new_tokens untuk beri ruang respons
    max_ctx = model.config.context_length - max_new_tokens - 1
    if max_ctx < 1:
        max_ctx = max(1, model.config.context_length // 2)

    if len(input_ids) > max_ctx:
        # Kekalkan token awal (untuk format prompt) dan token akhir
        keep_start = 10
        input_ids  = input_ids[:keep_start] + input_ids[-(max_ctx - keep_start):]

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # Decode HANYA token baru (bukan prompt)
    new_tokens = output_ids[0, len(input_ids):].tolist()
    response   = sp.decode(new_tokens)

    # Potong pada penanda [PENGGUNA] jika model menjana perbualan baru
    if "[PENGGUNA]" in response:
        response = response.split("[PENGGUNA]")[0]

    # Bersih trailing whitespace
    response = response.strip()

    # Mod JSON — cuba extract JSON dari respons
    if json_mode:
        parsed = extract_json(response)
        if parsed is not None:
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        # Jika gagal extract JSON, kembalikan teks biasa dengan nota
        return f"[Nota: output bukan JSON yang sah]\n{response}"

    return response


# =========================================================================== #
#  Paparan maklumat                                                             #
# =========================================================================== #

def print_model_info(model: AnneyGPT, meta: dict, device: torch.device) -> None:
    cfg = model.config
    print("\n" + "─" * 50)
    print("  Maklumat Model")
    print("─" * 50)
    print(f"  Peranti       : {device}")
    print(f"  Parameter     : {model.count_parameters():,}")
    print(f"  Vocab size    : {cfg.vocab_size:,}")
    print(f"  Context length: {cfg.context_length}")
    print(f"  n_embed       : {cfg.n_embed}")
    print(f"  n_heads       : {cfg.n_heads}")
    print(f"  n_layers      : {cfg.n_layers}")
    print(f"  Epoch         : {meta['epoch']}")
    print(f"  Global step   : {meta['global_step']}")
    print(f"  Val loss      : {meta['val_loss']}")
    print("─" * 50)


def print_settings(max_new_tokens, temperature, top_k, top_p, json_mode) -> None:
    print("\n  Tetapan semasa:")
    print(f"  max_new_tokens : {max_new_tokens}")
    print(f"  temperature    : {temperature}")
    print(f"  top_k          : {top_k}")
    print(f"  top_p          : {top_p if top_p else 'tidak aktif'}")
    print(f"  json_mode      : {'aktif' if json_mode else 'tidak aktif'}")


def print_help() -> None:
    print("""
  Arahan yang tersedia:
  ──────────────────────────────────────────────────────
  /help                   — tunjuk senarai arahan ini
  /info                   — maklumat model dan tetapan
  /clear                  — kosongkan sejarah perbualan
  /set temp  <val>        — tukar temperature (0.1–2.0)
  /set topk  <val>        — tukar top_k (1–200)
  /set topp  <val>        — aktif/set top_p (0.5–1.0)
  /set tokens <val>       — tukar max_new_tokens
  /json                   — togol mod JSON output
  /AnneyBelajar <topik>   — belajar topik dari internet 🎓
  /AnneyFahamkan <tajuk>  — suap maklumat terus kepada Anney 📝
  /SenaraiIlmu            — senarai semua ilmu dipelajari 📚
  /SenaraiAjar            — senarai semua maklumat yang diajar 📖
  /quit atau /exit        — keluar dari sembang
  ──────────────────────────────────────────────────────
""")


# =========================================================================== #
#  Main CLI loop                                                                #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="CLI sembang dengan Anney GPT")
    parser.add_argument(
        "--checkpoint",     default="checkpoints/pretrain/best_model.pt",
        help="Laluan checkpoint model"
    )
    parser.add_argument(
        "--max_new_tokens", type=int,   default=150,
        help="Bilangan token baru maksimum (default: 150)"
    )
    parser.add_argument(
        "--temperature",    type=float, default=0.9,
        help="Temperature pensampelan (default: 0.9)"
    )
    parser.add_argument(
        "--top_k",          type=int,   default=50,
        help="Top-K sampling (default: 50)"
    )
    parser.add_argument(
        "--top_p",          type=float, default=None,
        help="Nucleus sampling — gunakan atau tidak (default: tiada)"
    )
    parser.add_argument(
        "--json_mode",      action="store_true",
        help="Mod output JSON"
    )
    args = parser.parse_args()

    # Setup
    device = get_device()
    model, sp, model_cfg, meta = load_model_from_checkpoint(args.checkpoint, device)

    # Tetapan semasa (boleh diubah semasa sembang)
    max_new_tokens = args.max_new_tokens
    temperature    = args.temperature
    top_k          = args.top_k
    top_p          = args.top_p
    json_mode      = args.json_mode

    # Sejarah perbualan (untuk konteks berbilang giliran — versi ringkas)
    history = []

    # ─── Init Personality & Reasoning Engine ───────────────────── #
    persona = None
    reasoning = None
    try:
        from scripts.personality_engine import PersonalityEngine
        persona = PersonalityEngine()
    except ImportError:
        pass
    try:
        from scripts.reasoning_engine import ReasoningEngine
        reasoning = ReasoningEngine()
    except ImportError:
        pass

    # Banner
    print("\n" + "═" * 50)
    print("       ANNEY 0.1 GPT — Sembang Terminal")
    print("═" * 50)
    print("  Model bahasa kecil Bahasa Melayu")
    if persona:
        print(f"  Personaliti: AKTIF {persona.get_mood_emoji()}")
        print(f"  User Status: {persona.get_reputation_label()}")
    print("  Taip /help untuk senarai arahan")
    print("  Taip /quit untuk keluar")
    print("═" * 50)
    print_model_info(model, meta, device)

    print("\n  Anney bersedia. Mulakan perbualan!\n")

    # Main loop
    while True:
        try:
            # Show mood emoji in prompt
            mood_emoji = persona.get_mood_emoji() if persona else ""
            user_input = input("  [ANDA]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Selamat tinggal!")
            break

        if not user_input:
            continue

        # ─── Arahan pengguna ───────────────────────────────────────── #

        if user_input.startswith("/"):
            # Case-insensitive untuk command biasa, tapi kekalkan case untuk AnneyBelajar topic
            cmd_lower = user_input[1:].lower().split()[0] if user_input[1:].split() else ""
            cmd_parts = user_input[1:].lower().split()
            cmd = cmd_parts[0] if cmd_parts else ""

            if cmd in ("quit", "exit", "q"):
                print("\n  Selamat tinggal!")
                break

            elif cmd == "help":
                print_help()

            elif cmd == "info":
                print_model_info(model, meta, device)
                print_settings(max_new_tokens, temperature, top_k, top_p, json_mode)

            elif cmd == "clear":
                history.clear()
                print("  [Sejarah perbualan dikosongkan]")

            elif cmd == "json":
                json_mode = not json_mode
                status = "aktif" if json_mode else "tidak aktif"
                print(f"  Mod JSON: {status}")

            elif cmd == "anneybelajar":
                # ─── AnneyBelajar: Self-Learning ─────────────────── #
                try:
                    from scripts.anney_belajar import handle_anney_belajar_command
                    handle_anney_belajar_command(user_input[1:])  # Hantar tanpa '/'
                except ImportError as e:
                    print(f"\n  ❌ Modul AnneyBelajar tidak dijumpai: {e}")
                    print("  Pastikan scripts/anney_belajar.py wujud.")
                except Exception as e:
                    print(f"\n  ❌ Ralat AnneyBelajar: {e}")

            elif cmd == "anneyfahamkan":
                # ─── AnneyFahamkan: Suap Maklumat ────────────────── #
                try:
                    from scripts.anney_fahamkan import handle_anney_fahamkan_command
                    handle_anney_fahamkan_command(user_input[1:])  # Hantar tanpa '/'
                except ImportError as e:
                    print(f"\n  ❌ Modul AnneyFahamkan tidak dijumpai: {e}")
                    print("  Pastikan scripts/anney_fahamkan.py wujud.")
                except Exception as e:
                    print(f"\n  ❌ Ralat AnneyFahamkan: {e}")

            elif cmd == "senaraiilmu":
                # ─── Senarai Ilmu ───────────────────────────────── #
                try:
                    from scripts.anney_belajar import handle_senarai_ilmu_command
                    handle_senarai_ilmu_command()
                except ImportError as e:
                    print(f"\n  ❌ Modul AnneyBelajar tidak dijumpai: {e}")
                except Exception as e:
                    print(f"\n  ❌ Ralat: {e}")

            elif cmd == "senaraiiajar" or cmd == "senaraiajar":
                # ─── Senarai Ajar ───────────────────────────────── #
                try:
                    from scripts.anney_fahamkan import handle_senarai_faham_command
                    handle_senarai_faham_command()
                except ImportError as e:
                    print(f"\n  ❌ Modul AnneyFahamkan tidak dijumpai: {e}")
                except Exception as e:
                    print(f"\n  ❌ Ralat: {e}")

            elif cmd == "set" and len(cmd_parts) >= 3:
                param = cmd_parts[1]
                try:
                    val = float(cmd_parts[2])
                    if param == "temp":
                        temperature = max(0.1, min(2.0, val))
                        print(f"  temperature = {temperature}")
                    elif param == "topk":
                        top_k = max(1, int(val))
                        print(f"  top_k = {top_k}")
                    elif param == "topp":
                        top_p = max(0.5, min(1.0, val))
                        print(f"  top_p = {top_p}")
                    elif param == "tokens":
                        max_new_tokens = max(10, min(500, int(val)))
                        print(f"  max_new_tokens = {max_new_tokens}")
                    else:
                        print(f"  Param tidak dikenali: {param}. Cuba: temp, topk, topp, tokens")
                except ValueError:
                    print(f"  Nilai tidak sah: {cmd_parts[2]}")
            else:
                print(f"  Arahan tidak dikenali: {user_input}. Taip /help untuk bantuan.")

            continue

        # ─── Update personality dari input ────────────────────────── #

        if persona:
            persona.update_from_input(user_input)

        # ─── Jana respons ──────────────────────────────────────────── #

        # Cuba guna reasoning engine untuk enhance prompt
        enhanced = None
        if reasoning:
            try:
                mood = persona.get_mood() if persona else None
                rep = persona.get_reputation() if persona else None
                enhanced = reasoning.enhance(
                    user_input, history, mood=mood, user_reputation=rep,
                )
            except Exception:
                pass

        # Kalau ada taught answer dari reasoning engine, jawab terus
        if enhanced and enhanced.get("taught_answer"):
            taught_answer = enhanced["taught_answer"]
            me = persona.get_mood_emoji() if persona else ""
            print(f"\n  [ANNEY {me}]: {taught_answer}")
            history.append({
                "pengguna": user_input,
                "anney":    taught_answer,
            })
            print()
            continue

        # Bina prompt — guna enhanced prompt kalau ada
        if json_mode:
            prompt = format_json_prompt(user_input)
        elif enhanced:
            prompt = enhanced["enhanced_prompt"]
        else:
            # Fallback — bina prompt biasa
            context_turns = history[-3:] if history else []
            context_parts = []
            for turn in context_turns:
                context_parts.append(
                    f"[PENGGUNA]: {turn['pengguna']} [ANNEY]: {turn['anney']}"
                )
            context_str = " ".join(context_parts)

            if context_str:
                prompt = f"{context_str} [PENGGUNA]: {user_input} [ANNEY]: "
            else:
                prompt = f"[PENGGUNA]: {user_input} [ANNEY]: "

        me = persona.get_mood_emoji() if persona else ""
        print(f"\n  [ANNEY {me}]: ", end="", flush=True)

        try:
            response = generate_response(
                model=model,
                sp=sp,
                prompt_text=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                json_mode=json_mode,
            )

            print(response)

            # Simpan dalam sejarah
            history.append({
                "pengguna": user_input,
                "anney":    response,
            })

        except Exception as e:
            print(f"\n  [Ralat penjanaan: {e}]")

        print()  # Baris kosong antara giliran


if __name__ == "__main__":
    main()
