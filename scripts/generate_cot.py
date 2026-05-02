import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI  # Works with most DeepSeek-compatible APIs

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = REPO_ROOT / "configs" / "teacher_prompt.txt"
OUTPUT_PATH = REPO_ROOT / "data" / "teacher" / "raw_rationales.jsonl"

with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
    system_prompt = f.read().strip()


def gsm8k_ground_truth(answer_field: str) -> str:
    """GSM8K answers look like '... reasoning #### 42' — extract the final number/text."""
    return answer_field.split("####")[-1].strip()


def jsonl_line_count(path: Path) -> int:
    """Count non-empty JSONL rows (resume index assumes 1 row per GSM8K index in order)."""
    if not path.is_file():
        return 0
    n = 0
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                n += 1
    return n


def generate_rationale_groq(client: OpenAI, model: str, question: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model=model,
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


def generate_rationale_gemini(model_name: str, question: str) -> str | None:
    import google.generativeai as genai

    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_prompt,
    )
    try:
        response = model.generate_content(
            question,
            generation_config={"temperature": 0.6},
        )
        try:
            text = response.text
        except ValueError:
            # Blocked / no candidates
            print("Gemini returned no text (blocked or empty candidates).")
            return None
        return text.strip() if text else None
    except Exception as e:
        print(f"Error: {e}")
        if "429" in str(e):
            print("Rate limit hit, sleeping for 30 seconds...")
            time.sleep(30)
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate GSM8K teacher rationales via Groq (OpenAI-compat) or Google Gemini.",
    )
    p.add_argument(
        "--provider",
        choices=("groq", "gemini"),
        default="groq",
        help="LLM backend (default: groq). Use gemini to continue after another provider.",
    )
    p.add_argument(
        "--groq-model",
        default="llama-3.3-70b-versatile",
        help="Groq chat model id (default: llama-3.3-70b-versatile).",
    )
    p.add_argument(
        "--gemini-model",
        default=os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash",
        help=(
            "Gemini model id (default: GEMINI_MODEL env or gemini-2.5-flash). "
            "Older ids like gemini-1.5-flash were removed from the API (404)."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        metavar="N",
        help="Process N rows starting at --start / resume (ignored if --until is set).",
    )
    p.add_argument(
        "--until",
        type=int,
        default=None,
        metavar="U",
        help="Stop before GSM8K index U (exclusive). E.g. --start 205 --until 600 runs indices 205..599.",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        metavar="I",
        help="Start index into GSM8K train split (overridden by --resume).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Set start index to current JSONL row count so the next GSM8K row aligns with Llama/other runs. "
            "Assumes one line per GSM8K example in index order with no gaps."
        ),
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Seconds after each API attempt (Groq default: 2; Gemini default: 5).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate output JSONL instead of appending.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"JSONL output path (default: {OUTPUT_PATH.relative_to(REPO_ROOT)}).",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    if args.overwrite and args.resume:
        raise RuntimeError(
            "Do not combine --overwrite with --resume: resume counts existing lines, "
            "then --overwrite would wipe the file. Omit --overwrite to append."
        )

    out_path = args.output.resolve() if args.output.is_absolute() else (REPO_ROOT / args.output)

    start = args.start
    if args.resume:
        cnt = jsonl_line_count(out_path)
        if args.start != 0:
            print(
                f"Note: --resume uses JSONL row count ({cnt}), not --start {args.start}."
            )
        start = cnt
        print(f"Resume: {cnt} rows in {out_path.name} → GSM8K start index {start}.")

    if args.provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GROQ_API_KEY (e.g. in .env or export GROQ_API_KEY=...)."
            )
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model_id = args.groq_model
        caller: Callable[[str], str | None] = lambda q: generate_rationale_groq(
            client, model_id, q
        )
    else:
        import google.generativeai as genai

        gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini (e.g. in .env)."
            )
        genai.configure(api_key=gemini_key)
        model_id = args.gemini_model
        caller = lambda q: generate_rationale_gemini(model_id, q)

    pause = (
        args.sleep
        if args.sleep is not None
        else (5.0 if args.provider == "gemini" else 2.0)
    )

    print("Downloading / loading GSM8K (openai/gsm8k, main, train)...")
    raw_dataset = load_dataset("openai/gsm8k", "main", split="train")

    n_total = len(raw_dataset)
    if args.until is not None:
        end = min(args.until, n_total)
    else:
        end = min(start + max(1, args.limit), n_total)

    if start >= n_total:
        raise ValueError(f"Start index {start} is past dataset length ({n_total}).")
    if end <= start:
        raise ValueError(f"Nothing to process: end {end} <= start {start}.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "w" if args.overwrite else "a"

    print(
        f"Provider={args.provider} model={model_id} | GSM8K indices [{start}, {end}) "
        f"({end - start} questions) → {out_path.relative_to(REPO_ROOT)}, mode={file_mode}"
    )

    written = 0
    with open(out_path, file_mode, encoding="utf-8") as f:
        for i in range(start, end):
            row = raw_dataset[i]
            question = row["question"]
            answer = gsm8k_ground_truth(row["answer"])

            print(f"[{i - start + 1}/{end - start}] (GSM8K #{i + 1}) Generating rationale...")

            rationale = caller(question)
            if rationale:
                entry = {
                    "question": question,
                    "teacher_output": rationale,
                    "ground_truth": answer,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                written += 1

            time.sleep(pause)

    print(f"Done. Wrote {written} new lines to {out_path}.")


if __name__ == "__main__":
    main()
