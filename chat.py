#!/usr/bin/env python3
"""
Local terminal chat. No training needed.

Tuned for 16 GB RAM / 4 GB VRAM. Default model is qwen2.5:1.5b,
about 1 GB in Ollama, so it runs comfortably on 16 GB RAM.

Chats-only mode (default): the bot only uses your txt files in data/.
It retrieves the most similar past replies from data/pairs.jsonl for
each message, and it must not invent facts, names, places, or events
that are not in those chats. If something is not in the chats, it says
it does not know, in the same style. Use --no-strict to turn that off.

Uses Ollama with a small local model, plus style.json built by build_style.py.
Everything runs on your machine. History is saved to chat_history.jsonl locally.

Setup:
    ollama pull qwen2.5:1.5b
    python parse_chats.py
    python build_style.py
    python chat.py

Usage:
    python chat.py
    python chat.py --model qwen2.5:1.5b --temperature 0.8
    python chat.py --no-strict
"""

import argparse
import json
from pathlib import Path

BASE = Path(__file__).parent
STYLE_FILE = BASE / "style.json"
PAIRS_FILE = BASE / "data" / "pairs.jsonl"
HISTORY_FILE = BASE / "chat_history.jsonl"


def load_pairs():
    if not PAIRS_FILE.exists():
        return []
    out = []
    for line in PAIRS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def retrieve(pairs, user_text, k=3):
    """Simple word-overlap retrieval over your txt chats only. No external data."""
    import re

    user_words = set(re.findall(r"[a-zA-Z']+", user_text.lower()))
    if not user_words:
        return []
    scored = []
    for p in pairs:
        prompt_words = set(re.findall(r"[a-zA-Z']+", p["prompt"].lower()))
        score = len(user_words & prompt_words)
        if score:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


def load_style():
    if not STYLE_FILE.exists():
        return None
    return json.loads(STYLE_FILE.read_text(encoding="utf-8"))


def build_system_prompt(style, strict=True, retrieved=None):
    strict_line = ""
    if strict:
        strict_line = (
            "CHATS-ONLY RULE: Only use information, facts, names, places, and events "
            "that appear in the provided chat examples. Do not invent new facts about "
            "this person. If the user asks about something not in the chats, reply in "
            "their style saying you do not know or do not remember, do not make it up. "
        )
    if not style:
        return (
            "You are a friend chatting on a messaging app. "
            "Reply in short, natural messages, like a real person texting. "
            "Do not write long paragraphs or essays. " + strict_line
        )
    examples = "\n".join(
        f"User: {e['prompt']}\nYou: {e['reply']}" for e in style.get("examples", [])[:12]
    )
    retrieved_text = ""
    if retrieved:
        retrieved_text = "\nClosest matching chats from the txt files for this message:\n" + "\n".join(
            f"User: {p['prompt']}\nYou: {p['reply']}" for p in retrieved
        )
    return (
        "You are roleplaying as one specific person in a private text chat, "
        "for a local coding project. Reply exactly in their style, based on the examples below. "
        "Do not break character, do not explain that you are an AI, do not write long paragraphs. "
        + strict_line
        + f"Average reply length is about {style.get('avg_reply_words', 8)} words. "
        + f"Common words they use: {', '.join(style.get('top_words', []))}. "
        + "Keep replies short, natural, and in the same language mix as the examples.\n\n"
        + f"Examples:\n{examples}"
        + retrieved_text
    )


def ask_via_api(model, system, history, temperature):
    import urllib.request

    messages = [{"role": "system", "content": system}] + history
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--no-strict", action="store_true", help="Allow answers beyond the txt chats")
    args = ap.parse_args()
    strict = not args.no_strict

    style = load_style()
    pairs = load_pairs()
    if not style:
        print("No style.json found. Run: python parse_chats.py && python build_style.py")
        print("Continuing with a generic short-reply style.")
    if strict and not pairs:
        print("Chats-only mode is on, but no data/pairs.jsonl found.")
        print("Run: python parse_chats.py first, or the bot will have nothing to copy from.")
    history = []

    print(f"Local chat started. Model: {args.model} | 16GB RAM / 4GB VRAM profile")
    print(f"Chats-only mode: {'ON' if strict else 'OFF'} | Pairs loaded: {len(pairs)}")
    print("Type 'quit' to exit. History is saved locally to chat_history.jsonl")
    print("-" * 50)

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user.lower() in ("quit", "exit"):
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        # Rebuild the prompt for this message, with the closest chats retrieved
        retrieved = retrieve(pairs, user) if strict else []
        system_now = build_system_prompt(style, strict=strict, retrieved=retrieved)
        # Keep only the last 10 turns in context, so 16 GB RAM is enough
        recent = history[-20:]
        try:
            reply = ask_via_api(args.model, system_now, recent, args.temperature)
        except Exception as e:
            print(f"\nCould not reach Ollama: {e}")
            print("Is Ollama running? Try: ollama pull qwen2.5:1.5b")
            history.pop()
            continue
        print(f"Them: {reply}\n")
        history.append({"role": "assistant", "content": reply})
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"user": user, "reply": reply}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
