import json
import os
import time
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.groq.com/openai/v1")

def verify_sample(question, gt, candidates):
    # We ask the judge: "Which of these candidates got the answer {gt} correctly?"
    prompt = f"""Question: {question}
Ground Truth Answer: {gt}

Below are 5 candidate reasoning paths. For each path, determine if the FINAL ANSWER matches the Ground Truth.
Output a comma-separated list of boolean values (True/False) for each path.
Example Output: True, False, True, False, False

Candidates:
"""
    for i, c in enumerate(candidates):
        prompt += f"\n--- Path {i} ---\n{c}\n"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

with open("bulk_scaling_data.json", "r") as f:
    data = json.load(f)

final_results = []
for entry in data:
    print(f"Verifying: {entry['question'][:50]}...")
    verdict = verify_sample(entry['question'], entry['ground_truth'], entry['candidates'])
    # Convert "True, False..." string to a list of booleans
    bool_list = [x.strip().lower() == "true" for x in verdict.split(",")]
    entry["verdicts"] = bool_list
    final_results.append(entry)
    time.sleep(2) # Avoid rate limits

with open("bulk_results_final.json", "w") as f:
    json.dump(final_results, f)