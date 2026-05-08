"""
Oracle best-of-N curve from bulk GSM8K-style JSON (question, ground_truth, candidates).

Uses `data/inference_scaling_candidates/bulk_scaling_data.json` by default (same schema as
`core/bulk_scaling_eval.py`). Per-path correctness = extracted `<answer>` matches ground truth
(string normalize + numeric fallback). Optional: verifier point from `bulk_scaling_verified.json`.

Usage:
  python3 scripts/plot_final.py
  python3 scripts/plot_final.py --output evaluation/final_scaling_curve.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data/inference_scaling_candidates/bulk_scaling_data.json"
DEFAULT_VERIFIED = REPO_ROOT / "data/inference_scaling_candidates/bulk_scaling_verified.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "final_scaling_curve.png"

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_first_answer(text: str) -> str:
    m = ANSWER_PATTERN.search(text)
    return (m.group(1).strip() if m else "").strip()


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower().replace(",", "").replace("$", ""))


def _last_number(s: str) -> float | None:
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", s.replace(",", "")))
    if not matches:
        return None
    try:
        return float(matches[-1].group())
    except ValueError:
        return None


def answer_matches(candidate_text: str, ground_truth: str) -> bool:
    pred = extract_first_answer(candidate_text)
    if not pred:
        return False
    g = ground_truth.strip()
    if pred.strip() == g:
        return True
    if _norm_text(pred) == _norm_text(g):
        return True
    pn, gn = _last_number(pred), _last_number(g)
    if pn is not None and gn is not None and abs(pn - gn) < 1e-5:
        return True
    return False


def path_correctness_flags(entry: dict) -> list[bool]:
    gt = entry.get("ground_truth", "")
    cands = entry.get("candidates") or []
    return [answer_matches(c, str(gt)) for c in cands]


def oracle_curve(rows: list[dict], max_n: int) -> list[float]:
    """Fraction of questions with ≥1 correct path in first N."""
    total = len(rows)
    if total == 0:
        return []
    out: list[float] = []
    for n in range(1, max_n + 1):
        hit = 0
        for entry in rows:
            flags = path_correctness_flags(entry)
            if any(flags[:n]):
                hit += 1
        out.append(100.0 * hit / total)
    return out


def path0_accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    hit = sum(1 for e in rows if path_correctness_flags(e)[:1] == [True])
    return 100.0 * hit / len(rows)


def verifier_accuracy_at_max(
    rows: list[dict], verified_path: Path
) -> float | None:
    if not verified_path.is_file():
        return None
    with open(verified_path, encoding="utf-8") as f:
        verified = json.load(f)
    if not isinstance(verified, list):
        return None
    by_q = {v.get("question", "").strip(): v for v in verified}
    scored = 0
    hits = 0
    for entry in rows:
        q = entry.get("question", "").strip()
        v = by_q.get(q)
        if not v:
            continue
        br = v.get("best_reasoning")
        if not isinstance(br, str):
            continue
        gt = entry.get("ground_truth", "")
        scored += 1
        if answer_matches(br, str(gt)):
            hits += 1
    return 100.0 * hits / scored if scored else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle best-of-N plot from bulk_scaling_data JSON.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="bulk_scaling_data.json")
    p.add_argument(
        "--verified",
        type=Path,
        default=DEFAULT_VERIFIED,
        help="optional bulk_scaling_verified.json for verifier point at N_max",
    )
    p.add_argument("--no-verified", action="store_true", help="omit verifier scatter point")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data if args.data.is_absolute() else REPO_ROOT / args.data
    out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"Expected non-empty JSON array in {data_path}")

    max_n = max(len(e.get("candidates") or []) for e in data)
    if max_n == 0:
        raise SystemExit("No candidates in data.")

    n_range = list(range(1, max_n + 1))
    oracle_acc = oracle_curve(data, max_n)
    baseline = path0_accuracy(data)

    vpath = args.verified if args.verified.is_absolute() else REPO_ROOT / args.verified
    verifier_pt: float | None = None
    if not args.no_verified:
        verifier_pt = verifier_accuracy_at_max(data, vpath)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(
        n_range,
        oracle_acc,
        marker="o",
        linestyle="-",
        linewidth=3,
        markersize=8,
        label="Oracle best-of-N",
        color="tab:blue",
    )
    plt.axhline(
        y=baseline,
        color="tab:gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Path 0 only ({baseline:.1f}%)",
    )
    if verifier_pt is not None:
        plt.scatter(
            [max_n],
            [verifier_pt],
            color="tab:red",
            s=120,
            zorder=5,
            label=f"Verifier @ N={max_n} ({verifier_pt:.1f}%)",
        )

    plt.title("Inference-Time Scaling (GSM8K-style bulk data)", fontsize=16, fontweight="bold")
    plt.xlabel("Number of reasoning paths (N)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, min(105, max(oracle_acc + [baseline, verifier_pt or 0]) + 10))
    plt.xticks(n_range)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    gain = oracle_acc[-1] - oracle_acc[0]
    if abs(gain) > 1e-6:
        plt.annotate(
            f"{gain:+.1f}% gain (path 0 → N={max_n})",
            xy=(max_n, oracle_acc[-1]),
            xytext=(max(max_n - 1.8, 1.5), oracle_acc[-1] - 12),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=11,
            fontweight="bold",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved → {out_path}")
    print(f"  Questions: {len(data)}")
    print(f"  Oracle by N (%): {list(zip(n_range, [round(x, 1) for x in oracle_acc]))}")
    print(f"  Path-0 baseline: {baseline:.1f}%")
    if verifier_pt is not None:
        print(f"  Verifier @ N={max_n}: {verifier_pt:.1f}%")


if __name__ == "__main__":
    main()
