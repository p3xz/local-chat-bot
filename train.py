#!/usr/bin/env python3
"""
Optional LoRA fine-tune. Only use this if you have 2000+ pairs.

Tuned for 16 GB RAM / 4 GB VRAM. Default model is Qwen2.5-1.5B-Instruct
in 4-bit, r=8, sequence 256, batch 1, gradient checkpointing, 8-bit optimizer.
A 3B model will likely run out of memory on 4 GB VRAM, do not use it here
unless you move to a bigger GPU or Colab. On CPU only, do not run this,
use chat.py instead.

Training data is ONLY your txt chats in data/pairs.jsonl. No other data
is used. The model learns reply style and facts from those chats only.

    pip install -r requirements-train.txt
    python train.py --epochs 2

Output: output/lora-adapter/ (local only, gitignored)
To chat with it, load the adapter in Ollama / LM Studio, or with transformers.
"""

import argparse
import json
from pathlib import Path

BASE = Path(__file__).parent
PAIRS_FILE = BASE / "data" / "pairs.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=256)
    args = ap.parse_args()

    if not PAIRS_FILE.exists():
        print("No data/pairs.jsonl. Run: python parse_chats.py")
        return

    pairs = [json.loads(l) for l in PAIRS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Pairs: {len(pairs)}")
    if len(pairs) < 2000:
        print("Warning: under 2000 pairs. Fine-tuning will be weak.")
        print("Use chat.py (prompt-based) instead, or collect more chat first.")
        answer = input("Continue anyway? [y/N] ").strip().lower()
        if answer != "y":
            return

    # Unsloth import is inside main so `python train.py --help` works without GPU deps
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
    )
    if "3B" in args.model or "3b" in args.model:
        print("Warning: a 3B model will likely OOM on 4 GB VRAM.")
        print("Use the default 1.5B model for this machine.")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    # Alpaca-style training text, one example per pair
    def format_pair(p):
        return (
            "Below is a chat. Reply in the same style as the second person.\n\n"
            f"### User:\n{p['prompt']}\n\n### Reply:\n{p['reply']}"
        )

    from datasets import Dataset

    ds = Dataset.from_list([{"text": format_pair(p)} for p in pairs])

    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            optim="adamw_8bit",
            gradient_checkpointing=True,
            output_dir=str(BASE / "output"),
            logging_steps=10,
            save_steps=100,
        ),
    )
    trainer.train()
    out = BASE / "output" / "lora-adapter"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
