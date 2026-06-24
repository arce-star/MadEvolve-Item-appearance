#!/usr/bin/env python3
"""
种群动态可视化 — 重放注册序列, 展示 MAP-Elites + EliteVault + Island 的实时状态
用法: python code/visualize_population.py <results_dir> --run-name <name>
输出: fig/<name>/population_frames/ (每帧一张图)
"""

import argparse, hashlib, json, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIMS = ["complexity", "diversity", "performance"]
BINS = 10

# ──────────────────────── Helpers ────────────────────────

def _hash_embedding(code: str, dim=64) -> np.ndarray:
    h = int(hashlib.sha256(code.encode()).hexdigest()[:8], 16) % (2**31 - 1)
    rng = np.random.RandomState(h)
    emb = rng.randn(dim).astype(np.float64)
    return emb / (np.linalg.norm(emb) + 1e-10)

def compute_features(code: str, score: float, ref_embeddings: list) -> dict:
    emb = _hash_embedding(code)
    complexity = float(len(code))
    diversity = 1.0
    if ref_embeddings:
        dists = [1.0 - float(np.dot(emb, r) / (np.linalg.norm(emb)*np.linalg.norm(r)+1e-10))
                 for r in ref_embeddings[-20:]]
        diversity = float(np.mean(dists)) if dists else 1.0
    return {"complexity": complexity, "diversity": diversity, "performance": score}

def discretize(values, val, bins=BINS):
    lo, hi = np.min(values), np.max(values)
    if hi <= lo: hi = lo + 1.0
    return min(int((val - lo) / (hi - lo) * bins), bins - 1)

# ──────────────────────── Simulated Population ────────────────────────

class GridSim:
    def __init__(self):
        self.grid = {}       # (x,y,z) → {'id','score','features'}
        self.all_vals = {d: [] for d in DIMS}

    def register(self, pid, score, features):
        for d in DIMS:
            self.all_vals[d].append(features[d])
        coords = tuple(discretize(self.all_vals[d], features[d]) for d in DIMS)
        if coords not in self.grid or score > self.grid[coords]["score"]:
            self.grid[coords] = {"id": pid[:8], "score": score, "features": features}
            return "replace" if coords in self.grid else "new_cell"
        return "rejected"

class IslandSim:
    def __init__(self, n_islands=2, capacity=15):
        self.n = n_islands
        self.cap = capacity
        self.islands = [[] for _ in range(n_islands)]  # [{id,score},...]
        self.assignments = {}
        self.next_isl = 0

    def register(self, pid, score, parent_id):
        if parent_id and parent_id in self.assignments:
            idx = self.assignments[parent_id]
        else:
            idx = self.next_isl
            self.next_isl = (self.next_isl + 1) % self.n
        self.islands[idx].append({"id": pid[:8], "score": score})
        self.assignments[pid] = idx
        # sort & cap
        self.islands[idx].sort(key=lambda x: x["score"], reverse=True)
        removed = []
        while len(self.islands[idx]) > self.cap:
            removed.append(self.islands[idx].pop(-1))
        for r in removed:
            self.assignments.pop(r["id"], None)
        return idx, len(removed)

class VaultSim:
    def __init__(self, max_size=30):
        self.max = max_size
        self.vault = {}  # pid → score

    def register(self, pid, score):
        if pid in self.vault:
            self.vault[pid] = max(self.vault[pid], score)
            return "updated"
        if len(self.vault) < self.max:
            self.vault[pid] = score
            return "added"
        worst = min(self.vault, key=self.vault.get)
        if score > self.vault[worst]:
            del self.vault[worst]
            self.vault[pid] = score
            return "replaced"
        return "rejected"

# ──────────────────────── Visualization ────────────────────────

def draw_frame(step, total, pid, score, grid, islands, vault, out_path):
    fig = plt.figure(figsize=(18, 6))

    # ── MAP-Elites Grid ──
    ax1 = fig.add_subplot(1, 3, 1)
    heatmap = np.full((BINS, BINS), np.nan)
    for (xi, yi, _), entry in grid.grid.items():
        if np.isnan(heatmap[yi, xi]) or entry["score"] > heatmap[yi, xi]:
            heatmap[yi, xi] = entry["score"]
    # Show complexity vs performance (most intuitive slice)
    slice_z = {}  # aggregate z dim
    for (xi, yi, zi), entry in grid.grid.items():
        key = (xi, yi)
        if key not in slice_z or entry["score"] > slice_z[key]["score"]:
            slice_z[key] = entry
    for (xi, yi), entry in slice_z.items():
        heatmap[yi, xi] = entry["score"]
    im = ax1.imshow(heatmap, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax1, shrink=0.8, label="Best Score")
    filled = np.sum(~np.isnan(heatmap))
    ax1.set_title(f"MAP-Elites Grid\n{filled}/{BINS*BINS} filled")
    ax1.set_xlabel("complexity"); ax1.set_ylabel("performance")

    # ── Island Model ──
    ax2 = fig.add_subplot(1, 3, 2)
    colors = ["#2196F3", "#FF9800"]
    for i, isl in enumerate(islands.islands):
        xs = list(range(len(isl)))
        ys = [m["score"] for m in isl]
        label = f"Island {i+1} ({len(isl)})"
        ax2.bar(xs, np.array(ys)[:len(xs)], color=colors[i], alpha=0.6, label=label)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_title(f"Island Model (cap={islands.cap})")
    ax2.set_xlabel("Rank within island"); ax2.set_ylabel("Score")
    ax2.legend(fontsize=9)

    # ── Elite Vault ──
    ax3 = fig.add_subplot(1, 3, 3)
    vault_items = sorted(vault.vault.items(), key=lambda x: x[1], reverse=True)
    xs = list(range(len(vault_items)))
    ys = [v[1] for v in vault_items]
    bars = ax3.bar(xs, ys, color="#4CAF50" if len(vault_items) >= vault.max else "#8BC34A", alpha=0.7)
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.set_title(f"Elite Vault ({len(vault_items)}/{vault.max})")
    ax3.set_xlabel("Rank"); ax3.set_ylabel("Score")

    fig.suptitle(f"Population Dynamics — Step {step}/{total}  |  "
                 f"New: {pid[:8]} (${score:,.0f})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ──────────────────────── Main ────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--every", type=int, default=1,
                        help="Snapshot every N programs (default=1)")
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Max frames (subsample if too many)")
    args = parser.parse_args()

    db_path = os.path.join(args.results_dir, "evolution.db")
    if not os.path.exists(db_path):
        print(f"ERROR: {db_path} not found"); sys.exit(1)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT program_id, code, combined_score, parent_id, generation, created_at "
        "FROM programs WHERE combined_score IS NOT NULL AND combined_score > -1e9 "
        "ORDER BY created_at",
        conn,
    )
    conn.close()

    if len(df) < 2:
        print("Need at least 2 programs"); sys.exit(1)

    print(f"Replaying {len(df)} programs...")

    grid = GridSim()
    islands = IslandSim(n_islands=2, capacity=15)
    vault = VaultSim(max_size=30)
    ref_embeddings = []

    # Determine snapshot indices
    total = len(df)
    step_indices = list(range(0, total, max(1, total // args.max_frames)))
    if step_indices[-1] != total - 1:
        step_indices.append(total - 1)

    fig_dir = os.path.join(PROJECT_ROOT, "fig", args.run_name, "population_frames")
    os.makedirs(fig_dir, exist_ok=True)

    frame_num = 0
    for i, (_, row) in enumerate(df.iterrows()):
        pid = row["program_id"]
        score = row["combined_score"]
        code = row["code"]
        parent_id = row["parent_id"]

        features = compute_features(code, score, ref_embeddings)
        ref_embeddings.append(_hash_embedding(code))

        grid.register(pid, score, features)
        islands.register(pid, score, parent_id)
        vault.register(pid, score)

        if i in step_indices:
            out = os.path.join(fig_dir, f"frame_{frame_num:04d}.png")
            draw_frame(i + 1, total, pid, score, grid, islands, vault, out)
            print(f"  Frame {frame_num}: step {i+1}/{total} → {os.path.basename(out)}")
            frame_num += 1

    print(f"\n{frame_num} frames saved to {fig_dir}/")
    print("To make GIF: convert -delay 50 -loop 0 population_frames/*.png population.gif")


if __name__ == "__main__":
    main()
