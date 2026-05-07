"""
Plot inference-time scaling from repository data.

We do not store a pre-made table of accuracy vs N. This script derives:

- **Oracle best-of-N**: for each N, fraction of questions where at least one of the
  first N candidates has a correct extracted `<answer>`.
- **Oracle | path 0 wrong**: same, but only among questions where candidate 0 is
  incorrect — shows whether extra samples *rescue* failures (often the curve you
  expect to climb with N).
- **Single path (path 0)**: N=1 baseline.
- **Verifier @ N_max** (optional): score `best_reasoning` in `scaling_verified.json`.

If the overall oracle line is flat, the heatmap usually explains it: correctness
often appears already at index 0, or *no* path is ever correct under your gold checks.
Ground truth follows `QUESTION_CORRECTNESS`; extend when you add benchmarks.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
DEFAULT_CANDIDATES = REPO_ROOT / "data/inference_scaling_candidates/scaling_candidates.json"
DEFAULT_VERIFIED = REPO_ROOT / "data/inference_scaling_candidates/scaling_verified.json"
DEFAULT_FIGURE = EVAL_ROOT / "inference_scaling_graph.png"

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_first_answer(text: str) -> str:
    m = ANSWER_PATTERN.search(text)
    return (m.group(1).strip() if m else "").strip()


def _floats(s: str) -> list[float]:
    out: list[float] = []
    for m in re.finditer(r"\d+(?:\.\d+)?", s.replace(",", "")):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def gold_clock(answer_blob: str) -> bool:
    """Strikes: 3 strokes in 3s → gaps = 7.5s for 6 strokes (canonical puzzle)."""
    nums = _floats(answer_blob)
    if not nums:
        return False
    return any(abs(v - 7.5) < 0.25 for v in nums)


def gold_bat_ball(answer_blob: str) -> bool:
    nums = _floats(answer_blob)
    return any(abs(v - 0.05) < 0.001 for v in nums)


def gold_widgets(answer_blob: str) -> bool:
    a = answer_blob.lower().strip()
    if re.search(r"\b1\s+minute\b", a) or re.match(r"^\s*1\s*$", a):
        return False
    if re.search(r"\b5\s+minutes?\b", a) or re.match(r"^\s*5\s*$", a):
        return True
    nums = _floats(answer_blob)
    return any(abs(v - 5.0) < 0.25 for v in nums) and len(nums) <= 4


QUESTION_CORRECTNESS: dict[str, callable[..., bool]] = {
    "If a clock strikes 3 times in 3 seconds, how many seconds will it take to strike 6 times?": gold_clock,
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?": gold_bat_ball,
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?": gold_widgets,
}


def candidate_correct(question: str, candidate_text: str) -> bool:
    checker = QUESTION_CORRECTNESS.get(question.strip())
    if checker is None:
        return False
    ans = extract_first_answer(candidate_text)
    return bool(checker(ans))


def oracle_accuracy_through_n(rows: list[dict], n: int) -> float:
    hits = 0
    scored = 0
    for row in rows:
        q = row.get("question", "").strip()
        if q not in QUESTION_CORRECTNESS:
            continue
        cand = row.get("candidates") or []
        if not isinstance(cand, list) or not cand:
            continue
        scored += 1
        head = cand[: max(1, min(n, len(cand)))]
        if any(candidate_correct(q, c) for c in head):
            hits += 1
    return 100.0 * hits / scored if scored else 0.0


def path0_accuracy(rows: list[dict]) -> float:
    return oracle_accuracy_through_n(rows, 1)


def oracle_accuracy_path0_wrong_subset(rows: list[dict], n: int) -> float | None:
    """Among questions where path 0 is wrong, fraction where any of first N is correct."""
    subset: list[dict] = []
    for row in rows:
        q = row.get("question", "").strip()
        if q not in QUESTION_CORRECTNESS:
            continue
        cand = row.get("candidates") or []
        if not isinstance(cand, list) or not cand:
            continue
        if not candidate_correct(q, cand[0]):
            subset.append(row)
    if not subset:
        return None
    hits = 0
    for row in subset:
        q = row.get("question", "").strip()
        cand = row.get("candidates") or []
        head = cand[: max(1, min(n, len(cand)))]
        if any(candidate_correct(q, c) for c in head):
            hits += 1
    return 100.0 * hits / len(subset)


def count_path0_wrong(rows: list[dict]) -> int:
    k = 0
    for row in rows:
        q = row.get("question", "").strip()
        if q not in QUESTION_CORRECTNESS:
            continue
        cand = row.get("candidates") or []
        if isinstance(cand, list) and cand and not candidate_correct(q, cand[0]):
            k += 1
    return k


def correctness_matrix(rows: list[dict], max_n: int) -> tuple[np.ndarray, list[str]]:
    """Shape (n_questions, max_n); NaN if that path index does not exist."""
    mat: list[list[float]] = []
    labels: list[str] = []
    for row in rows:
        q = row.get("question", "").strip()
        if q not in QUESTION_CORRECTNESS:
            continue
        cand = row.get("candidates") or []
        if not isinstance(cand, list):
            continue
        row_vals: list[float] = []
        for j in range(max_n):
            if j < len(cand):
                row_vals.append(1.0 if candidate_correct(q, cand[j]) else 0.0)
            else:
                row_vals.append(float("nan"))
        mat.append(row_vals)
        labels.append(_short_q(q))
    return np.array(mat, dtype=float), labels


def _short_q(q: str) -> str:
    low = q.lower()
    if "clock" in low and "strike" in low:
        return "Clock strikes"
    if "bat" in low and "ball" in low:
        return "Bat & ball"
    if "widget" in low or "machine" in low:
        return "Widgets / machines"
    return (q[:42] + "…") if len(q) > 42 else q


def verifier_accuracy(rows: list[dict], verified: list[dict]) -> float | None:
    hits = 0
    scored = 0
    by_q = {v.get("question", "").strip(): v for v in verified}
    for row in rows:
        q = row.get("question", "").strip()
        if q not in QUESTION_CORRECTNESS:
            continue
        v = by_q.get(q)
        if not v:
            continue
        br = v.get("best_reasoning")
        if not isinstance(br, str):
            continue
        scored += 1
        if candidate_correct(q, br):
            hits += 1
    return 100.0 * hits / scored if scored else None


def load_json_array(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {path}")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot scaling metrics from candidates + optional verifier JSON.")
    p.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("--verified", type=Path, default=DEFAULT_VERIFIED, help="optional; omit with --no-verified")
    p.add_argument("--no-verified", action="store_true", help="do not load verifier results")
    p.add_argument("--output", type=Path, default=DEFAULT_FIGURE, help="PNG path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cand_path = args.candidates if args.candidates.is_absolute() else REPO_ROOT / args.candidates
    out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    rows = load_json_array(cand_path)
    max_n = max((len(r.get("candidates") or []) for r in rows), default=0)
    if max_n == 0:
        raise SystemExit("No candidates in input JSON.")

    n_paths = list(range(1, max_n + 1))
    oracle_curve = [oracle_accuracy_through_n(rows, n) for n in n_paths]
    rescue_curve = [oracle_accuracy_path0_wrong_subset(rows, n) for n in n_paths]
    single_baseline = path0_accuracy(rows)
    n_fail = count_path0_wrong(rows)

    verified_list: list[dict] = []
    verifier_point: float | None = None
    if not args.no_verified:
        vpath = args.verified if args.verified.is_absolute() else REPO_ROOT / args.verified
        if vpath.is_file():
            verified_list = load_json_array(vpath)
            verifier_point = verifier_accuracy(rows, verified_list)

    mat, q_labels = correctness_matrix(rows, max_n)

    flat_oracle = max(oracle_curve) - min(oracle_curve) < 1e-3

    fig, (ax_line, ax_heat) = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        height_ratios=[1.05, 0.95],
        constrained_layout=True,
    )

    ax_line.plot(
        n_paths,
        oracle_curve,
        marker="o",
        linestyle="-",
        color="tab:blue",
        linewidth=2,
        label="Oracle best-of-N (all questions)",
    )
    ax_line.axhline(
        y=single_baseline,
        color="tab:gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Path 0 only: {single_baseline:.1f}%",
    )
    if any(r is not None for r in rescue_curve):
        rc = [r if r is not None else np.nan for r in rescue_curve]
        ax_line.plot(
            n_paths,
            rc,
            marker="s",
            linestyle="--",
            color="tab:orange",
            linewidth=1.8,
            label=f"Oracle best-of-N | path 0 wrong (n={n_fail} Q)",
        )

    if verifier_point is not None:
        ax_line.scatter(
            [max_n],
            [verifier_point],
            color="tab:red",
            s=140,
            zorder=5,
            label=f"Verifier @ N={max_n}: {verifier_point:.1f}%",
        )

    ax_line.set_title(
        "Inference-time scaling — read the caption if the curve is flat",
        fontsize=13,
        fontweight="medium",
    )
    ax_line.set_xlabel("Number of candidate paths (N)")
    ax_line.set_ylabel("Accuracy (%)")
    ax_line.set_xticks(n_paths)
    ax_line.set_ylim(-2, min(105, max(oracle_curve + [single_baseline, verifier_point or 0]) + 15))
    ax_line.grid(True, linestyle="--", alpha=0.7)
    ax_line.legend(loc="upper left", fontsize=8)

    caption = []
    if flat_oracle:
        caption.append(
            "Overall oracle is flat → expanding N adds no NEW correct paths in this file: "
            "either path 0 is already correct, or every sampled path misses gold (often both)."
        )
    if n_fail:
        caption.append(
            f"Orange line: among the {n_fail} question(s) where path 0 is wrong — "
            "if this stays flat, later paths never recover the right ⟨answer⟩ under your scorer."
        )
    else:
        caption.append("Orange line omitted — path 0 is correct for every graded question.")

    caption.append(
        "Heatmap below: green = gold-correct extracted ⟨answer⟩; gray cell = fewer than N paths on disk."
    )
    ax_line.text(
        0.01,
        0.02,
        "\n".join(caption),
        transform=ax_line.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.88, edgecolor="0.6"),
    )

    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("cor", [(0.86, 0.2, 0.18), (0.95, 0.92, 0.88), (0.2, 0.72, 0.35)])
    im = ax_heat.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        vmin=-0.05,
        vmax=1.05,
        interpolation="nearest",
    )
    ax_heat.set_yticks(range(len(q_labels)))
    ax_heat.set_yticklabels(q_labels)
    ax_heat.set_xticks(range(max_n))
    ax_heat.set_xticklabels([str(i) for i in range(max_n)])
    ax_heat.set_xlabel("Candidate path index")
    ax_heat.set_title("Correctness per question × path (gold-scored)")
    plt.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02, label="1 = correct")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")
    print(f"  Path-0 baseline: {single_baseline:.1f}%")
    print(f"  Questions with wrong path 0 (subplot orange): {n_fail}")
    print(f"  Oracle-by-N (all): {list(zip(n_paths, [round(x, 1) for x in oracle_curve]))}")
    if any(r is not None for r in rescue_curve):
        print(f"  Oracle-by-N | path0 wrong: {[round(r, 1) if r is not None else None for r in rescue_curve]}")
    if verifier_point is not None:
        print(f"  Verifier @ N={max_n}: {verifier_point:.1f}%")
    else:
        print("  Verifier: not plotted (--no-verified or missing JSON)")


if __name__ == "__main__":
    main()
