import json
import matplotlib.pyplot as plt
import numpy as np

with open("bulk_results_final.json", "r") as f:
    data = json.load(f)

total = len(data)
# Calculate Oracle Accuracy for each N
# Accuracy @ N = (Count questions where at least one of the first N paths is True) / Total
oracle_acc = []
for n in range(1, 6):
    correct_at_n = 0
    for entry in data:
        if any(entry["verdicts"][:n]):
            correct_at_n += 1
    oracle_acc.append((correct_at_n / total) * 100)

# Plotting
n_range = [1, 2, 3, 4, 5]
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(n_range, oracle_acc, marker='o', linestyle='-', linewidth=3, markersize=8, label="Oracle Best-of-N")

# Formatting for a "Research Paper" look
plt.title("Inference-Time Scaling (n=20, GSM8K Test)", fontsize=16, fontweight='bold')
plt.xlabel("Number of Reasoning Paths (N)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(0, 100)
plt.xticks(n_range)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc="lower right")

# Annotate the gain
gain = oracle_acc[-1] - oracle_acc[0]
plt.annotate(f'+{gain:.1f}% Accuracy Gain', xy=(5, oracle_acc[-1]), xytext=(3.5, oracle_acc[-1]-10),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, fontweight='bold')

plt.savefig("final_scaling_curve.png")
plt.show()