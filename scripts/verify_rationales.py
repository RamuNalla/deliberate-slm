import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "data" / "teacher" / "raw_rationales.jsonl"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "verified_cot_train.jsonl"


def extract_answer(text):
    """Extracts content inside <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def verify() -> None:
    verified_data = []

    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            teacher_ans = extract_answer(data["teacher_output"])

            # Simple check: Does teacher answer match ground truth?
            # Note: For complex math, you might need a more robust parser
            if teacher_ans == data["ground_truth"]:
                verified_data.append(
                    {
                        "instruction": data["question"],
                        "output": data["teacher_output"],
                    }
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in verified_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Verification complete. Kept {len(verified_data)} high-quality samples.")


if __name__ == "__main__":
    verify()
