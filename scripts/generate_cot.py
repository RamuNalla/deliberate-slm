import json
import os
from pathlib import Path

from openai import OpenAI  # Works with most DeepSeek-compatible APIs

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = REPO_ROOT / "configs" / "teacher_prompt.txt"
OUTPUT_PATH = REPO_ROOT / "data" / "teacher" / "raw_rationales.jsonl"

with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
    system_prompt = f.read().strip()


def generate_rationale(client: OpenAI, question: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GROQ_API_KEY in your environment (e.g. export GROQ_API_KEY=...)."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # Load your seed questions (e.g., from GSM8K)
    # For this example, we'll use a small dummy list
    questions = [
        {"question": "If 3x + 5 = 20, what is x?", "ground_truth": "5"},
        # Add more questions here
    ]

    output_data = []
    for item in questions:
        print(f"Processing: {item['question']}")
        rationale = generate_rationale(client, item["question"])
        output_data.append(
            {
                "question": item["question"],
                "teacher_output": rationale,
                "ground_truth": item["ground_truth"],
            }
        )

    # Save raw teacher data
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
