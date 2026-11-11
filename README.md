# local-chat-bot

A private, local-only chatbot that learns to reply in the style of one person from your own chat exports.

This is a coding project, not a public product. It runs entirely on your own machine. Your chat files never leave your machine, and they are never committed to this repo.

## What it does

1. `parse_chats.py` reads your chat `.txt` files from `data/` and turns them into clean prompt / reply pairs.
2. `build_style.py` reads those pairs and writes a style profile: average reply length, common words, slang, emoji use, how often they use Hinglish, short replies, etc.
3. `chat.py` runs a local chat in your terminal. It uses Ollama with a small local model, plus the style profile and a few real example replies, so it talks in that style. No training needed, works with as little as 200 messages.
4. `train.py` is the optional real fine-tune. Only use it if you have 2000+ good pairs. It fine-tunes a small open model with LoRA using Unsloth. This is the JustZando-style route, but modern: you fine-tune an existing model instead of training a Transformer from zero.

## Setup

```bash
# 1. Python 3.10+
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Ollama (for the no-training chat)
# Install from https://ollama.com, then:
ollama pull qwen2.5:3b
# or: ollama pull llama3.2:3b
```

## Your chat files

Put your 3 chat exports in `data/` as plain `.txt` files. Do NOT commit them. They are gitignored.

```
data/
  chat1.txt
  chat2.txt
  chat3.txt
```

Supported formats, the parser tries all of them:
- WhatsApp export: `12/11/2026, 10:15 pm - Name: message here`
- Simple lines: `Name: message here`
- `Name (10:15): message here`

Edit the two names at the top of `parse_chats.py`:
- `YOU_NAME` = your name as it appears in the export
- `THEM_NAME` = the other person's name as it appears in the export

Messages that are `<Media omitted>`, deleted messages, stickers, calls, and system lines are skipped automatically.

## Run it

```bash
# Step 1: parse your 3 files into pairs
python parse_chats.py

# Step 2: build the style profile
python build_style.py

# Step 3: chat locally
python chat.py
```

`parse_chats.py` writes `data/pairs.jsonl` (local only, gitignored) and prints how many pairs it found. `build_style.py` writes `style.json` and prints a short summary.

If you have fewer than 200 pairs, the style bot will still work, it just will be less accurate. If you have fewer than 50, add more chat or use the sample data first to test the pipeline.

## Optional: real fine-tune

Only after the prompt-based chat works and you have 2000+ pairs:

```bash
pip install -r requirements-train.txt
python train.py --model Qwen/Qwen2.5-3B-Instruct --epochs 2
```

This needs a GPU with at least 8 GB VRAM, or a free Colab GPU. On CPU only, do not run this, use `chat.py` instead. The fine-tuned adapter is saved to `output/lora-adapter/`, local only.

## Test without your chats

```bash
cp sample_data/sample_chat.txt data/chat1.txt
python parse_chats.py
python build_style.py
python chat.py
```

The sample file is fictional, made up for testing. It is not any real person's chat.

## Privacy rules for this project

- Keep it local. Do not publish the bot as if it is that person.
- Do not share it with other people pretending it is them.
- Do not put real chat files, names, photos, or phone numbers in this repo.
- `data/`, `*.jsonl`, `output/`, and model files are all gitignored for this reason.

## Files

- `parse_chats.py` - chat export parser, WhatsApp and simple formats
- `build_style.py` - style profile from pairs
- `chat.py` - local terminal chat via Ollama
- `train.py` - optional LoRA fine-tune with Unsloth
- `sample_data/sample_chat.txt` - fictional sample chat for testing
- `requirements.txt` - runtime deps
- `requirements-train.txt` - training deps only
