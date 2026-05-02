import json
import re

def extract_answer(text):
    """Extracts content inside <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def verify():
    verified_data = []
    
    with open("data/teacher/raw_rationales.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            teacher_ans = extract_answer(data["teacher_output"])
            
            # Simple check: Does teacher answer match ground truth?
            # Note: For complex math, you might need a more robust parser
            if teacher_ans == data["ground_truth"]:
                verified_data.append({
                    "instruction": data["question"],
                    "output": data["teacher_output"]
                })
    
    # Save the cleaned data for training
    with open("data/processed/verified_cot_train.jsonl", "w") as f:
        for entry in verified_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Verification complete. Kept {len(verified_data)} high-quality samples.")

if __name__ == "__main__":
    verify()