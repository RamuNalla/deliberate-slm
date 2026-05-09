"""
Compress verified CoT rationale text to Chain-of-Draft via Groq.

Reads `data/processed/verified_cot_train.jsonl` and writes `data/processed/cod_train_data.jsonl`.

Usage:
  export GROQ_API_KEY=...   # or set in repository root `.env`
  python3 scripts/compress_to_cod.py

  python3 scripts/compress_to_cod.py --limit 50 --sleep 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")

COMPRESSOR_PROMPT = """
You are a Rationale Compressor. Rewrite the following thinking process into a 'Chain-of-Draft'.
Rules:
1. Use less than 20 words.
2. Use symbols, arrows, and equations (e.g., ->, +, *, =) instead of sentences.
3. Keep the logic intact but strip all conversational filler.

Example:
Input: First I take 50 and find 20 percent which is 10. Then I subtract 10 from 50 to get 40. Then I add 15.
Output: 50*0.2=10 -> 50-10=40 -> 40+15=55.

Thinking Process:
{thought}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compress verified CoT to Chain-of-Draft (Groq).")
    p.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data/processed/verified_cot_train.jsonl",
        help="verified_cot_train.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/processed/cod_train_data.jsonl",
        help="output cod_train_data.jsonl",
    )
    p.add_argument("--limit", type=int, default=200, help="max lines to compress (default: 200)")
    p.add_argument(
        "--groq-model",
        default=os.environ.get("GROQ_VERIFIER_MODEL") or "llama-3.3-70b-versatile",
        help="Groq model id (default: llama-3.3-70b-versatile)",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="seconds to pause after each API call (default: 0)",
    )
    return p.parse_args()


def compress_thought(client: OpenAI, model: str, thought: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": COMPRESSOR_PROMPT.format(thought=thought)}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    return (raw or "").strip()


def extract_thought_and_answer(output_text: str) -> tuple[str, str] | None:
    """Return (thinking_body, answer_text) from model output."""
    low = "<think>"
    hi = "</think>"
    if low not in output_text or hi not in output_text:
        return None
    thought = output_text.split(low, 1)[1].split(hi, 1)[0]
    parts = output_text.split("<answer>", 1)
    if len(parts) < 2:
        return None
    ans = parts[1].split("</answer>", 1)[0].strip()
    return thought.strip(), ans


def main() -> None:
    args = parse_args()
    inp = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    outp = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            f"Missing GROQ_API_KEY. Set it in {REPO_ROOT / '.env'} or your environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    model = args.groq_model

    cod_data: list[dict[str, str]] = []
    with open(inp, encoding="utf-8") as f:
        lines = f.readlines()

    n = min(args.limit, len(lines))
    for idx, line in enumerate(lines[:n], start=1):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        extracted = extract_thought_and_answer(data.get("output", ""))
        if extracted is None:
            print(f"[{idx}/{n}] skip: could not parse redacted_thinking / answer", file=sys.stderr)
            continue
        thought, answer = extracted

        print(f"[{idx}/{n}] compressing…")
        try:
            cod_thought = compress_thought(client, model, thought)
        except Exception as e:
            print(f"[{idx}/{n}] Groq error: {e}", file=sys.stderr)
            sys.exit(1)

        cod_data.append({
            "instruction": data["instruction"],
            "output": f"<think> {cod_thought} </think> <answer> {answer} </answer>",
        })

        if args.sleep > 0:
            time.sleep(args.sleep)

    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for entry in cod_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(cod_data)} samples → {outp}")


if __name__ == "__main__":
    main()
