import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI  # Works with most DeepSeek-compatible APIs

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = REPO_ROOT / "configs" / "teacher_prompt.txt"
OUTPUT_PATH = REPO_ROOT / "data" / "teacher" / "raw_rationales.jsonl"

with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
    system_prompt = f.read().strip()


def generate_rationale(client: OpenAI, question: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.6,
        )
        content = response.choices[0].message.content
        return content.strip() if content else None
    except Exception as e:
        print(f"Error: {e}")
        return None


def gsm8k_ground_truth(answer_field: str) -> str:
    """GSM8K answers look like '... reasoning #### 42' — extract the final number/text."""
    return answer_field.split("####")[-1].strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate teacher rationales for GSM8K via Groq (respects rate limits)."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        metavar="N",
        help="How many training rows to process (default: 500; use e.g. 1000 for a larger run).",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        metavar="I",
        help="Start index into GSM8K train split (resume partial runs without re-querying earlier rows).",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to wait after each API call (~30 req/min on Groq free tier; default 2).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of appending.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"JSONL output path (default: {OUTPUT_PATH.relative_to(REPO_ROOT)}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GROQ_API_KEY (e.g. in .env next to configs/ or export GROQ_API_KEY=...)."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    print("Downloading / loading GSM8K (openai/gsm8k, main, train)...")
    raw_dataset = load_dataset("openai/gsm8k", "main", split="train")

    n_total = len(raw_dataset)
    start = max(0, args.start)
    end = min(start + max(1, args.limit), n_total)
    if start >= n_total:
        raise ValueError(f"--start {start} is past dataset length ({n_total}).")

    out_path = args.output.resolve() if args.output.is_absolute() else (REPO_ROOT / args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "w" if args.overwrite else "a"

    print(
        f"Processing indices [{start}, {end}) ({end - start} questions) "
        f"→ {out_path.relative_to(REPO_ROOT)}, mode={file_mode}"
    )

    written = 0
    with open(out_path, file_mode, encoding="utf-8") as f:
        for i in range(start, end):
            row = raw_dataset[i]
            question = row["question"]
            answer = gsm8k_ground_truth(row["answer"])

            print(f"[{i - start + 1}/{end - start}] Generating rationale...")

            rationale = generate_rationale(client, question)
            if rationale:
                entry = {
                    "question": question,
                    "teacher_output": rationale,
                    "ground_truth": answer,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                written += 1

            time.sleep(args.sleep)

    print(f"Done. Wrote {written} new lines to {out_path}.")


if __name__ == "__main__":
    main()
