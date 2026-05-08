import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
import json
import time

# 1. Load Model
model_id = "nallaramu/deliberate-qwen-2.5-3b-reasoning"
model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_id, load_in_4bit=True)
FastLanguageModel.for_inference(model)

# 2. Load 20 questions from the TEST set
test_set = load_dataset("openai/gsm8k", "main", split="test")
sample_size = 20
questions = test_set.select(range(sample_size))

results = []

print(f"Starting bulk generation for {sample_size} questions...")

for i, item in enumerate(questions):
    q = item['question']
    gt = item['answer'].split("####")[-1].strip()
    
    print(f"[{i+1}/{sample_size}] Generating 5 paths...")
    prompt = f"### Question:\n{q}\n\n### Reasoning:\n"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    candidates = []
    for _ in range(5):
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.8, do_sample=True)
        resp = tokenizer.batch_decode(outputs)[0].split("### Reasoning:\n")[1]
        candidates.append(resp)
    
    results.append({
        "question": q,
        "ground_truth": gt,
        "candidates": candidates
    })

with open("bulk_scaling_data.json", "w") as f:
    json.dump(results, f)

print("Done! Data saved to bulk_scaling_data.json")