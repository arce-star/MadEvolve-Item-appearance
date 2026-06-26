#!/usr/bin/env python3
"""Run4 进化轨迹 — Score 曲线"""
import sys, os, json, sqlite3
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/run4_quick/20260625_015652"
import time; ts = time.strftime("%Y%m%d_%H%M%S")
out = sys.argv[2] if len(sys.argv) > 2 else f"fig/run4/evolution_progress_{ts}.png"

db = sqlite3.connect(os.path.join(results_dir, "evolution.db"))
rows = db.execute("SELECT generation, combined_score, created_at FROM programs WHERE combined_score IS NOT NULL AND combined_score > -1e9 ORDER BY created_at").fetchall()
db.close()

scores = [r[1] for r in rows]
best = list(np.maximum.accumulate(scores))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(best)), best, linewidth=2, color='#2196F3', label='Best Score (cumulative)')
ax.scatter(range(len(scores)), scores, s=10, alpha=0.4, color='gray', label='Individual candidates')
ax.set_xlabel("Programs Evaluated")
ax.set_ylabel("Combined Score (R²+IC+ICIR)")
ax.set_title(f"Run4 Evolution Progress\nBaseline={scores[0]:.4f} → Best={best[-1]:.4f}", fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
print(f"✓ {out}")
