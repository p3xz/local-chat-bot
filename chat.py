#!/usr/bin/env python3
"""
Local terminal chat. No training needed.

Uses Ollama with a small local model, plus style.json built by build_style.py.
Everything runs on your machine. History is saved to chat_history.jsonl locally.

Setup:
    ollama pull qwen2.5:3b
    python parse_chats.py
    python build_style.py
    python chat.py

Usage:
    python chat.py
    python chat.py --model llama3.2:3b
    python chat.py --model qwen2.5:3b --temperature 0.8
"""

import argparse
import json
from pathlib import Path

BASE = Path(__file__).parent
STYLE_FILE = BASE / "style.json"
HISTORY_FILE = BASE / "chat_history.jsonl"


def load_style():
    if not STYLE_FILE.exists():
        return None
    return json.loads(STYLE_FILE.read_text(encoding="utf-8"))


def build_system_prompt(style):
    if not style:
        return (
            "You are a friend chatting on a messaging app. "
            "Reply in short, natural messages, like a real person texting. "
            "Do not write long paragraphs or essays."
        )
    examples = "\n".join(
        f"User: {e['prompt']}\nYou: {e['reply']}" for e in style.get("examples", [])[:12]
    )
    return (
        "You are roleplaying as one specific person in a private text chat, "
        "for a local coding project. Reply exactly in their style, based on the examples below. "
        "Do not break character, do not explain that you are an AI, do not write long paragraphs. "
        f"Average reply length is about {style.get('avg_reply_words', 8)} words. "
        f"Common words they use: {', '.join(style.get('top_words', []))}. "
        "Keep replies short, natural, and in the same language mix as the examples.\n\n"
        f"Examples:\n{examples}"
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
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    style = load_style()
    if not style:
        print("No style.json found. Run: python parse_chats.py && python build_style.py")
        print("Continuing with a generic short-reply style.")
    system = build_system_prompt(style)
    history = []

    print(f"Local chat started. Model: {args.model}")
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
        try:
            reply = ask_via_api(args.model, system, history, args.temperature)
        except Exception as e:
            print(f"\nCould not reach Ollama: {e}")
            print("Is Ollama running? Try: ollama pull qwen2.5:3b")
            history.pop()
            continue
        print(f"Them: {reply}\n")
        history.append({"role": "assistant", "content": reply})
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"user": user, "reply": reply}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
