#!/usr/bin/env python3
"""
Build a style profile from data/pairs.jsonl (created by parse_chats.py).

Output: style.json (safe to commit, it has no real messages in full,
only counts and a few short example replies, capped at 120 chars each).

Usage:
    python build_style.py
"""

import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent
PAIRS_FILE = BASE / "data" / "pairs.jsonl"
OUT_FILE = BASE / "style.json"


def main():
    if not PAIRS_FILE.exists():
        print("No data/pairs.jsonl found. Run: python parse_chats.py")
        return

    pairs = [json.loads(line) for line in PAIRS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not pairs:
        print("No pairs found in data/pairs.jsonl")
        return

    replies = [p["reply"] for p in pairs]
    words = []
    for r in replies:
        words.extend(re.findall(r"[a-zA-Z']+", r.lower()))

    lengths = [len(r.split()) for r in replies]
    avg_len = sum(lengths) / len(lengths)
    short = sum(1 for n in lengths if n <= 3) / len(lengths)
    emoji_count = sum(len(re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", r)) for r in replies)

    common = Counter(words).most_common(30)
    # drop very common English filler for the top list
    stop = {"the", "a", "an", "and", "to", "of", "in", "is", "it", "you", "i", "me", "my", "that", "this"}
    top_words = [w for w, _ in common if w not in stop][:15]

    examples = []
    for p in pairs:
        r = p["reply"]
        if 2 <= len(r.split()) <= 20 and len(r) <= 120:
            examples.append({"prompt": p["prompt"][:120], "reply": r})
        if len(examples) >= 30:
            break

    profile = {
        "pairs": len(pairs),
        "avg_reply_words": round(avg_len, 1),
        "short_reply_share": round(short, 2),
        "emoji_per_reply": round(emoji_count / len(replies), 2),
        "top_words": top_words,
        "examples": examples,
        "notes": [
            f"Average reply is {avg_len:.1f} words.",
            f"{short*100:.0f}% of replies are 3 words or fewer." if short > 0.2 else "Replies are usually longer than 3 words.",
            "Uses emojis often." if emoji_count / len(replies) > 0.5 else "Rarely uses emojis.",
        ],
    }

    OUT_FILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pairs: {len(pairs)}")
    print(f"Average reply: {avg_len:.1f} words")
    print(f"Short replies (<=3 words): {short*100:.0f}%")
    print(f"Top words: {', '.join(top_words)}")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
