import torch
from unsloth import FastLanguageModel
import json

# 1. Load your HF Model
model_id = "your-username/deliberate-qwen-2.5-3b-reasoning"
model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_id, load_in_4bit = True)
FastLanguageModel.for_inference(model)

test_questions = [
    "If a clock strikes 3 times in 3 seconds, how many seconds will it take to strike 6 times?",
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?"
]

def generate_candidates(question, n=5):
    prompt = f"### Question:\n{question}\n\n### Reasoning:\n"
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    candidates = []
    for i in range(n):
        # We use temperature=0.7 to get diverse thinking paths
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
        resp = tokenizer.batch_decode(outputs)[0].split("### Reasoning:\n")[1]
        candidates.append(resp)
    return candidates

results = []
for q in test_questions:
    print(f"Generating 5 paths for: {q}")
    paths = generate_candidates(q, n=5)
    results.append({"question": q, "candidates": paths})

with open("scaling_candidates.json", "w") as f:
    json.dump(results, f)