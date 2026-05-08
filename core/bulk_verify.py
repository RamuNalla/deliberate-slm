"""
Run the same Groq best-path verifier as core/verifier.py over bulk_scaling_data JSON.

Each row must look like bulk_scaling_eval output:
  { "question": str, "ground_truth": str, "candidates": [str, ...] }

The judge still compares candidate paths (verifier.py prompt); ground_truth is kept in
the output for downstream GSM8K-style checks but is not injected into the verifier prompt.

Usage:
  Put GROQ_API_KEY in the repository root `.env` (loaded automatically; no shell export needed).
  python core/bulk_verify.py

  python core/bulk_verify.py \\
    --input data/inference_scaling_candidates/bulk_scaling_data.json \\
    --output data/inference_scaling_candidates/bulk_scaling_verified.json
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

from verifier import (
    REPO_ROOT,
    shuffled_display_order,
    verify_best_path_groq,
)

load_dotenv(REPO_ROOT / ".env")

DEFAULT_INPUT = REPO_ROOT / "data/inference_scaling_candidates/bulk_scaling_data.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/inference_scaling_candidates/bulk_scaling_verified.json"


def load_bulk_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Groq verifier for bulk_scaling_data JSON (question, ground_truth, candidates)."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"bulk_scaling_data JSON (default: {DEFAULT_INPUT.relative_to(REPO_ROOT)})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"where to write results (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    p.add_argument(
        "--groq-model",
        default=os.environ.get("GROQ_VERIFIER_MODEL") or "llama-3.3-70b-versatile",
        help="Groq model id (default: llama-3.3-70b-versatile; override with GROQ_VERIFIER_MODEL).",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Seconds to pause after each API call (default: 1.5).",
    )
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        metavar="S",
        help=(
            "If set, randomize candidate order shown to the verifier (per row, reproducible). "
            "Same behavior as core/verifier.py --shuffle-seed."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input if args.input.is_absolute() else (REPO_ROOT / args.input)
    output_path = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        env_path = REPO_ROOT / ".env"
        print(
            f"Missing GROQ_API_KEY. Add a line like GROQ_API_KEY=... to {env_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    model = args.groq_model

    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_bulk_rows(input_path)
    final_decisions: list[dict] = []

    for i, entry in enumerate(rows):
        question = entry.get("question", "")
        ground_truth = entry.get("ground_truth")
        candidates = entry.get("candidates") or []

        if not isinstance(candidates, list) or len(candidates) == 0:
            print(f"[{i + 1}/{len(rows)}] skip: missing or empty candidates", file=sys.stderr)
            final_decisions.append({
                "question": question,
                "ground_truth": ground_truth,
                "error": "no_candidates",
                "raw_verifier_reply": None,
                "best_path_index": None,
                "best_reasoning": None,
                "n_candidates": 0,
            })
            continue

        print(
            f"[{i + 1}/{len(rows)}] Verifying: {question[:80]!r}{'…' if len(question) > 80 else ''}"
        )

        orig_indices: list[int] | None = None
        to_verify = candidates
        if args.shuffle_seed is not None:
            to_verify, orig_indices = shuffled_display_order(
                candidates, i, args.shuffle_seed
            )

        raw, display_idx = verify_best_path_groq(client, model, question, to_verify)

        verifier_display_idx: int | None
        if display_idx is None:
            pick = candidates[0]
            best_orig_idx = 0
            verifier_display_idx = None
            print(
                f"  Could not parse index from reply {raw!r}; defaulting to original path {best_orig_idx}",
                file=sys.stderr,
            )
        else:
            verifier_display_idx = display_idx
            if orig_indices is not None:
                best_orig_idx = orig_indices[display_idx]
            else:
                best_orig_idx = display_idx
            pick = candidates[best_orig_idx]

        record: dict = {
            "question": question,
            "ground_truth": ground_truth,
            "best_path_index": best_orig_idx,
            "raw_verifier_reply": raw,
            "verifier_display_index": verifier_display_idx,
            "best_reasoning": pick,
            "n_candidates": len(candidates),
        }
        if orig_indices is not None:
            record["candidate_order_shown_to_verifier"] = orig_indices
        final_decisions.append(record)

        time.sleep(args.sleep)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_decisions, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(final_decisions)} entries → {output_path}")


if __name__ == "__main__":
    main()
