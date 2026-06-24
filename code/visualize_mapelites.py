#!/usr/bin/env python3
"""
MAP-Elites 网格可视化 — 从 evolution.db 重建并绘制热力图
用法: python code/visualize_mapelites.py <results_dir> --run-name run1
输出: fig/<run_name>/mapelites_*.png (3张: complexity-performance / diversity-performance / complexity-diversity)
"""

import argparse, json, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MAP-Elites grid config
DIMENSIONS = ["complexity", "diversity", "performance"]
BINS = 10
DIM_PAIRS = [
    ("complexity", "performance", "viridis"),
    ("diversity", "performance", "plasma"),
    ("complexity", "diversity", "magma"),
]
DIM_LABELS = {
    "complexity": "Code Length (chars)",
    "diversity": "Diversity (cosine distance)",
    "performance": "Performance (score)",
}


def compute_features(code: str, score: float, all_codes: list) -> dict:
    """Recompute MAP-Elites features from code and score."""
    features = {}
    # complexity: character count
    features["complexity"] = len(code)

    # diversity: mean cosine distance to all other programs
    # Using hash-based embedding (same as our modified vectorizer)
    import hashlib
    emb = _hash_embedding(code)
    if len(all_codes) > 1:
        others = [_hash_embedding(c) for c in all_codes if c != code]
        if others:
            dists = [1.0 - np.dot(emb, o) / (np.linalg.norm(emb) * np.linalg.norm(o) + 1e-10)
                     for o in others[:20]]  # sample up to 20 for speed
            features["diversity"] = np.mean(dists)
        else:
            features["diversity"] = 0.0
    else:
        features["diversity"] = 0.0

    # performance: fitness score
    features["performance"] = score
    return features


def _hash_embedding(code: str, dim=64) -> np.ndarray:
    """Hash-based embedding (consistent with our modified vectorizer)."""
    import hashlib
    h = int(hashlib.sha256(code.encode()).hexdigest()[:8], 16) % (2**31 - 1)
    rng = np.random.RandomState(h)
    emb = rng.randn(dim).astype(np.float64)
    return emb / np.linalg.norm(emb)


def load_population(db_path: str) -> pd.DataFrame:
    """Load all evaluated programs from evolution.db."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT program_id, code, combined_score, generation, "
        "json_extract(metadata, '$.model_used') as model "
        "FROM programs WHERE combined_score IS NOT NULL AND combined_score > -1e9",
        conn,
    )
    conn.close()
    return df


def discretize(value, all_values, bins=BINS):
    """Bin a continuous value into 0..bins-1."""
    lo, hi = np.min(all_values), np.max(all_values)
    if hi <= lo:
        hi = lo + 1.0
    normalized = (value - lo) / (hi - lo)
    return min(int(normalized * bins), bins - 1)


def build_grid(df: pd.DataFrame) -> dict:
    """Build MAP-Elites grid from program data."""
    all_codes = df["code"].tolist()

    # Compute features for all programs
    features_list = []
    for _, row in df.iterrows():
        feats = compute_features(row["code"], row["combined_score"], all_codes)
        features_list.append(feats)

    # Build grid (each cell keeps the best program)
    grid = {}
    all_vals = {dim: [f[dim] for f in features_list] for dim in DIMENSIONS}

    for i, feats in enumerate(features_list):
        coords = tuple(discretize(feats[dim], all_vals[dim]) for dim in DIMENSIONS)
        score = df.iloc[i]["combined_score"]
        if coords not in grid or score > grid[coords]["score"]:
            grid[coords] = {
                "score": score,
                "features": feats,
                "generation": df.iloc[i]["generation"],
                "program_id": df.iloc[i]["program_id"],
            }
    return grid, all_vals


def plot_grid(grid, all_vals, dim_x, dim_y, cmap, out_path, title_suffix=""):
    """Plot a 2D slice of the MAP-Elites grid."""
    heatmap = np.full((BINS, BINS), np.nan)
    count_map = np.zeros((BINS, BINS), dtype=int)

    for coords, entry in grid.items():
        xi = coords[DIMENSIONS.index(dim_x)]
        yi = coords[DIMENSIONS.index(dim_y)]
        if np.isnan(heatmap[yi, xi]) or entry["score"] > heatmap[yi, xi]:
            heatmap[yi, xi] = entry["score"]
        count_map[yi, xi] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, origin="lower", cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Best Score (Impact-Adj PnL)", shrink=0.8)

    # Axis labels with actual value ranges
    x_vals = np.linspace(np.min(all_vals[dim_x]), np.max(all_vals[dim_x]), BINS + 1)
    y_vals = np.linspace(np.min(all_vals[dim_y]), np.max(all_vals[dim_y]), BINS + 1)

    ax.set_xticks(range(BINS))
    ax.set_xticklabels([f"{x_vals[i]:.0f}" if dim_x == "complexity" else f"{x_vals[i]:.3f}"
                        for i in range(BINS)], rotation=45, fontsize=7)
    ax.set_yticks(range(BINS))
    ax.set_yticklabels([f"{y_vals[i]:.0f}" if dim_y == "complexity" else f"{y_vals[i]:.3f}"
                        for i in range(BINS)], fontsize=7)

    # Annotate cell count for filled cells
    for yi in range(BINS):
        for xi in range(BINS):
            if count_map[yi, xi] > 0:
                ax.text(xi, yi, str(count_map[yi, xi]), ha="center", va="center",
                        fontsize=8, color="white" if not np.isnan(heatmap[yi, xi]) else "gray",
                        fontweight="bold")

    ax.set_xlabel(DIM_LABELS[dim_x])
    ax.set_ylabel(DIM_LABELS[dim_y])
    coverage = np.sum(~np.isnan(heatmap)) / (BINS * BINS) * 100
    ax.set_title(f"MAP-Elites Grid: {dim_x} vs {dim_y}{title_suffix}\n"
                 f"Coverage: {coverage:.1f}% ({int(np.sum(~np.isnan(heatmap)))}/{BINS*BINS} cells filled)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def main():
    parser = argparse.ArgumentParser(description="MAP-Elites Grid Visualizer")
    parser.add_argument("results_dir", help="Path to evolution results directory")
    parser.add_argument("--run-name", required=True, help="Run identifier (run1/run2)")
    args = parser.parse_args()

    db_path = os.path.join(args.results_dir, "evolution.db")
    if not os.path.exists(db_path):
        print(f"ERROR: {db_path} not found")
        sys.exit(1)

    print(f"Loading programs from {db_path}...")
    df = load_population(db_path)
    print(f"  {len(df)} programs, gen {df['generation'].min()}-{df['generation'].max()}")

    if len(df) < 2:
        print("  Not enough data for grid visualization")
        sys.exit(1)

    print("Building MAP-Elites grid...")
    grid, all_vals = build_grid(df)
    print(f"  {len(grid)} cells filled")

    fig_dir = os.path.join(PROJECT_ROOT, "fig", args.run_name)
    for dim_x, dim_y, cmap in DIM_PAIRS:
        out_path = os.path.join(fig_dir, f"mapelites_{dim_x}_{dim_y}.png")
        plot_grid(grid, all_vals, dim_x, dim_y, cmap, out_path)

    print(f"\nDone! Grids saved to {fig_dir}/")


if __name__ == "__main__":
    main()
