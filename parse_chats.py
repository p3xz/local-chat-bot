#!/usr/bin/env python3
"""
Parse chat .txt exports in data/ into clean prompt/reply pairs.

Put your files in data/ as chat1.txt, chat2.txt, chat3.txt (any .txt works).
Edit YOU_NAME and THEM_NAME below to match the names in your export.

Output (local only, gitignored): data/pairs.jsonl
Each line: {"prompt": "<your message>", "reply": "<their reply>"}

Usage:
    python parse_chats.py
    python parse_chats.py --you "Namish" --them "Alex"
"""

import argparse
import json
import re
from pathlib import Path

YOU_NAME = "Namish"
THEM_NAME = "Alex"

DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "pairs.jsonl"

# WhatsApp: 12/11/2026, 10:15 pm - Name: message
WHATSAPP_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(?:am|pm)?\s*-\s*(.+?):\s*(.*)$",
    re.IGNORECASE,
)
# Simple: Name: message
SIMPLE_RE = re.compile(r"^([^:]{1,40}):\s*(.*)$")
# Name (10:15): message
TIME_RE = re.compile(r"^(.+?)\s*\(\d{1,2}:\d{2}(?:\s*(?:am|pm))?\)\s*:\s*(.*)$", re.IGNORECASE)

SKIP_SUBSTRINGS = [
    "<media omitted>",
    "media omitted",
    "this message was deleted",
    "message deleted",
    "missed voice call",
    "missed video call",
    "voice call",
    "video call",
    "created group",
    "added you",
    "changed the group",
    "messages are end-to-end encrypted",
    "sticker omitted",
    "image omitted",
    "audio omitted",
    "video omitted",
    "document omitted",
]


def clean_text(text: str) -> str:
    text = text.strip()
    # WhatsApp URLs / links are kept, but strip trailing spaces
    return re.sub(r"\s+", " ", text)


def is_skip(text: str) -> bool:
    low = text.lower()
    if not text:
        return True
    if len(text) > 500:
        return True
    return any(s in low for s in SKIP_SUBSTRINGS)


def parse_line(line: str):
    """Return (speaker, message) or None."""
    line = line.strip()
    if not line:
        return None
    for pattern in (WHATSAPP_RE, TIME_RE, SIMPLE_RE):
        m = pattern.match(line)
        if m:
            return clean_text(m.group(1)), clean_text(m.group(2))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--you", default=YOU_NAME)
    ap.add_argument("--them", default=THEM_NAME)
    ap.add_argument("--data", default=str(DATA_DIR))
    args = ap.parse_args()

    data_dir = Path(args.data)
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {data_dir}")
        print("Put your chat exports there as chat1.txt, chat2.txt, chat3.txt")
        print("Or test with: cp sample_data/sample_chat.txt data/chat1.txt")
        return

    pairs = []
    total_lines = 0
    total_messages = 0

    for path in files:
        print(f"Reading {path.name} ...")
        messages = []  # list of (speaker, text) in order
        current_speaker = None
        current_text = ""

        def flush():
            nonlocal current_speaker, current_text
            if current_speaker and current_text and not is_skip(current_text):
                messages.append((current_speaker, clean_text(current_text)))
            current_speaker = None
            current_text = ""

        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            total_lines += 1
            parsed = parse_line(raw)
            if parsed:
                flush()
                current_speaker, current_text = parsed
            elif current_speaker:
                # continuation line of the previous message
                current_text += " " + raw.strip()
        flush()

        total_messages += len(messages)
        print(f"  {len(messages)} messages")

        # Build pairs: your message -> their next reply
        for i in range(len(messages) - 1):
            speaker, text = messages[i]
            next_speaker, next_text = messages[i + 1]
            if speaker.lower() == args.you.lower() and next_speaker.lower() == args.them.lower():
                if not is_skip(text) and not is_skip(next_text):
                    pairs.append({"prompt": text, "reply": next_text})

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nFiles: {len(files)}, lines: {total_lines}, messages: {total_messages}")
    print(f"Pairs found (you -> them): {len(pairs)}")
    print(f"Wrote {OUT_FILE}")
    if len(pairs) < 200:
        print("Note: under 200 pairs, the style bot will work but will be rough.")
        print("Add more chat if you can. For a real fine-tune you want 2000+ pairs.")
    elif len(pairs) < 2000:
        print("Good for the prompt-based chat. For fine-tuning, aim for 2000+ pairs.")
    else:
        print("Good for both the prompt-based chat and fine-tuning.")


if __name__ == "__main__":
    main()
