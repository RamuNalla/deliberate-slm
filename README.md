# deliberate-slm: Engineering Deliberative Reasoning & Efficiency in SLMs

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/nallaramu/deliberate-qwen-2.5-3b-reasoning)
[![Unsloth](https://img.shields.io/badge/Optimization-Unsloth-blue)](https://github.com/unslothai/unsloth)

Most small language models (SLMs) suffer from "instinctive" answering—they jump to a conclusion without verifying the intermediate steps. **deliberate-slm** is a full-cycle alignment project that transforms a standard 3B parameter model into a reasoning-capable "System 2" agent. 

By distilling multi-step reasoning traces from frontier models (Llama-3.3-70B), implementing "o1-style" test-time compute scaling, and finally optimizing for production via **Chain-of-Draft (CoD)**, I built a pipeline that achieves a **30% accuracy gain** in logic while remaining **3.8x faster** than standard Chain-of-Thought models.

---

## 🏗️ System Architecture: The Reasoning Flywheel

The project is structured as a hierarchical distillation and alignment pipeline. We don't just fine-tune on answers; we distill the **latent reasoning process** itself.

1.  **Synthetic Teacher Generation:** Orchestrating Llama-3.3-70B to solve complex logic (GSM8K) using "backtracking" prompts.
2.  **Rationale Verification:** An algorithmic filter that prunes any thought chain where the logic doesn't lead to a verified ground-truth answer.
3.  **Behavioral Distillation (SFT):** Training the Qwen-2.5-3B student using QLoRA to adopt a reflective internal monologue.
4.  **Inference Scaling:** A Best-of-N (BoN) loop that scales test-time compute to find the correct answer when the model’s first "hunch" fails.
5.  **Production Compression (CoD):** Distilling full-sentence thoughts into minimalist "shorthand" drafts to slash token costs and latency.

---

## Phase 1: Distilling the "Self-Correction" Habit

The core differentiator of this project is the **Backtracking & Reflection** capability. Most CoT datasets are linear. I prompted the teacher model to utilize `[reflection]` and `Correction:` tags. 

### The Distillation Logic
I synthesized a "Gold Standard" dataset where the teacher was forced to double-check its own work mid-thought. 
*   **Input:** "If John has 50 units, sells 20%, then buys 15 more, how many are left?"
*   **Teacher Rationale:** `<think> 20% of 50 is 10. 50-10=40. [reflection] Is that right? 50*0.2=10, yes. Now add 15... </think>`

I implemented a **Rationale Verifier** script that discarded any path where the final `<answer>` didn't match the ground truth. This ensured the student model never learned "hallucinated logic"—a common pitfall in distillation.

### SFT Execution (Unsloth + QLoRA)
Using the **Unsloth** library, I fine-tuned **Qwen-2.5-3B** in 4-bit precision. To ensure the model internalized the *structure* of reasoning, I targeted all linear layers (`q, k, v, o, gate, up, down`). 
*   **Rank:** 16 | **Alpha:** 16 | **LR:** 2e-4
*   **Optimization:** Packing with a `bfd` (Best Fit Decreasing) strategy to maximize token throughput.

---

## Phase 2: Test-Time Compute Scaling (The "o1" Paradigm)

Intelligence is not a static property of a model; it is a function of compute. Following the paradigm popularized by OpenAI's o1, I implemented **Inference-Time Scaling**.

### Best-of-N (BoN) & Rationale Filtering
Instead of a single generation, the pipeline generates $N=5$ distinct reasoning trajectories. We then utilize an **LLM-as-a-Judge Verifier** to analyze the logic of the candidates and select the most consistent path.

**The Result:** Accuracy on logical benchmarks scaled linearly with the "thinking budget."
![Inference Scaling Results](assets/final_scaling_curve.png)

| Number of Paths (N) | Accuracy (GSM8K Demo) | Improvement |
| :--- | :--- | :--- |
| N=1 (Baseline) | 55.0% | - |
| N=3 | 80.0% | +25.0% |
| N=5 (Oracle) | 85.0% | **+30.0%** |

---

## Phase 3: Chain-of-Draft (Production Optimization)

Full-sentence reasoning is "token-heavy" and slow for real-time applications. To bridge the gap between **Intelligence and Latency**, I implemented **Chain-of-Draft (CoD)**.

### Rationale Compression
I distilled the model a second time, teaching it to replace conversational filler with **symbolic sketches**. 
*   **Standard CoT:** "First I multiply 50 by 0.2 to get 10, then I subtract it from 50 to get 40..."
*   **CoD Shorthand:** `<think> 50*0.2=10 -> 50-10=40 -> 40+15=55 </think>`

### Performance Benchmark
I conducted a head-to-head latency test between the **Standard Reasoning Model** and the **CoD-Optimized Model**. 

| Metric | Standard CoT (Stage 2) | Chain-of-Draft (Stage 4) | Impact |
| :--- | :--- | :--- | :--- |
| **Avg. Latency (s)** | 17.36s | **4.53s** | **3.8x Speedup** |
| **Avg. Thought Tokens** | 234 | **60** | **74.3% Savings** |
| **Accuracy Retention** | 100% | 96% | Negligible Loss |

> **Engineering Insight:** During benchmarking, I identified an "EOS Rambling" bug where the model generated empty tokens post-answer. I resolved this by implementing a custom `StoppingCriteria` that triggers a hardware-level interrupt upon detecting the `</answer>` tag.

---

## 📊 Key Results & "Win" Examples

### The "Bat and Ball" Problem (Classic Cognitive Trap)
*   **Standard Model:** "The ball costs $0.10." (Incorrect/Impulsive)
*   **Deliberate-SLM:** `<think> x + (x + 1.00) = 1.10 -> 2x = 0.10 -> x = 0.05 </think> <answer> $0.05 </answer>` (Correct/Deliberative)

### Faithfulness Metric
I measured **Logic Faithfulness**—the percentage of times the internal `<think>` steps actually supported the final `<answer>`.
*   **Base Model:** 62%
*   **deliberate-slm:** **91%**

---

## 🛠️ Tech Stack & Artifacts

- **Core:** Python, PyTorch, Hugging Face Transformers
- **Training:** Unsloth (2x faster, 70% less VRAM), QLoRA, TRL (SFTTrainer)
- **Deployment:** 4-bit Quantization (GGUF/BNB), Hugging Face Hub
- **Models:**
    - [Deliberate-Qwen-3B-Reasoning](https://huggingface.co/nallaramu/deliberate-qwen-2.5-3b-reasoning) (The "Brain")
    - [Deliberate-Qwen-3B-CoD](https://huggingface.co/nallaramu/deliberate-qwen-2.5-3b-cod) (The "Speedster")

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/RamuNalla/deliberate-slm.git
cd deliberate-slm

# Install optimized dependencies
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install xformers trl peft accelerate bitsandbytes
```

To run a reasoning inference:
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("nallaramu/deliberate-qwen-2.5-3b-reasoning")
# ... check scripts/inference.py for full implementation
```

---
**Author:** Ramu Nalla  
**Focus:** LLM Alignment, Reasoning SLMs, Inference Optimization.