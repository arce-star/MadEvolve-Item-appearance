#!/usr/bin/env python3
"""
论文级分析报告生成器 — 读取 MadEvolve results 目录, 输出:
  1. Cumulative PnL 曲线 (baseline vs best, val + test)
  2. 进化进度图 (IS vs OOS)
  3. Sizing Decomposition 柱状图
  4. Sharpe/Calmar 对比图
  5. 模型贡献统计
  6. 指标汇总表
"""

import argparse, json, os, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_simulator import BacktestSimulator, AlphaForecaster

# ============================================================
# Config
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_baseline_results(results_dir):
    """加载基线评估结果."""
    db_path = os.path.join(results_dir, "evolution.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM programs WHERE generation=0 ORDER BY combined_score DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "code": row["code"],
        "combined_score": row["combined_score"],
        "public_metrics": json.loads(row["public_metrics"] or "{}"),
        "text_feedback": row["text_feedback"] or "",
    }


def load_best_results(results_dir):
    """加载最优评估结果."""
    best_dir = Path(results_dir) / "best"
    result_path = best_dir / "result.json"
    code_path = best_dir / "best.py"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        data = json.load(f)
    code = ""
    if code_path.exists():
        code = code_path.read_text()
    return {"code": code, **data}


def load_history(results_dir):
    """加载进化历史."""
    hp = os.path.join(results_dir, "history.json")
    if os.path.exists(hp):
        with open(hp) as f:
            return json.load(f)
    # fallback: build from db
    db_path = os.path.join(results_dir, "evolution.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT generation, MAX(combined_score) as best_score FROM programs "
        "WHERE combined_score IS NOT NULL GROUP BY generation ORDER BY generation",
        conn,
    )
    conn.close()
    return {
        "generations": [{"generation": r.generation, "best_score": r.best_score}
                        for r in df.itertuples()],
    }


def load_model_stats(results_dir):
    """统计每个模型的表现."""
    db_path = os.path.join(results_dir, "evolution.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT json_extract(metadata, '$.model_used') as model, "
        "combined_score, parent_id, generation "
        "FROM programs WHERE generation > 0",
        conn,
    )
    conn.close()
    if df.empty:
        return None

    # 获取 parent scores
    conn = sqlite3.connect(db_path)
    parent_scores = {}
    for row in conn.execute("SELECT program_id, combined_score FROM programs"):
        parent_scores[row[0]] = row[1]
    conn.close()

    df["parent_score"] = df["parent_id"].map(parent_scores)
    df["improvement"] = df["combined_score"] - df["parent_score"]
    df["improved"] = df["improvement"] > 0

    stats = df.groupby("model").agg(
        total=("model", "count"),
        improved=("improved", "sum"),
        avg_score=("combined_score", "mean"),
        max_score=("combined_score", "max"),
        avg_improvement=("improvement", "mean"),
    ).reset_index()
    stats["impr_rate"] = stats["improved"] / stats["total"] * 100
    stats = stats.sort_values("max_score", ascending=False)
    return stats, df


def run_backtest(code_str, data_period="val", use_tiny=False):
    """对给定代码跑回测."""
    if use_tiny:
        if data_period == "val":
            data_path = os.path.join(DATA_DIR, "btcusdt_1m_val_tiny.parquet")
        else:
            data_path = os.path.join(DATA_DIR, "btcusdt_1m_test_tiny.parquet")
            if not os.path.exists(data_path):
                # Create tiny test: first week of 2025
                test = pd.read_parquet(os.path.join(DATA_DIR, "btcusdt_1m_test.parquet"))
                test[:10080].to_parquet(data_path)  # first week: 7*24*60=10080 rows
    else:
        period_map = {
            "val": os.path.join(DATA_DIR, "btcusdt_1m_val.parquet"),
            "test": os.path.join(DATA_DIR, "btcusdt_1m_test.parquet"),
        }
        data_path = period_map.get(data_period)
    if not os.path.exists(data_path):
        return None

    ohlcv = pd.read_parquet(data_path)
    forecaster = AlphaForecaster()
    f_path = os.path.join(DATA_DIR, "forecaster.pkl")
    if os.path.exists(f_path):
        forecaster.load(f_path)
    else:
        forecaster.fit(pd.read_parquet(os.path.join(DATA_DIR, "btcusdt_1m_train.parquet")))

    # Import strategy class from code string
    import tempfile, importlib.util
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code_str)
        tmp_path = f.name
    spec = importlib.util.spec_from_file_location("tmp_strategy", tmp_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        os.unlink(tmp_path)
    cls = getattr(mod, "DefaultPassiveExecutor")
    strategy = cls()

    sim = BacktestSimulator(ohlcv, forecaster)
    return sim.run(strategy)


def sizing_decomposition(baseline_code, best_code, use_tiny=False):
    """按论文 Fig.2 算 sizing decomposition."""
    tag = "tiny" if use_tiny else "full"
    print(f"  Running 4 backtests ({tag} data)...", flush=True)
    bl_val = run_backtest(baseline_code, "val", use_tiny); print("    [1/4] baseline val ✓", flush=True)
    ev_val = run_backtest(best_code, "val", use_tiny);    print("    [2/4] evolved  val ✓", flush=True)
    bl_test = run_backtest(baseline_code, "test", use_tiny); print("    [3/4] baseline test ✓", flush=True)
    ev_test = run_backtest(best_code, "test", use_tiny);   print("    [4/4] evolved  test ✓", flush=True)

    def _decomp(bl, ev):
        if bl is None or ev is None:
            return None
        Fb = bl["public_metrics"]["pnl_frictionless"]
        Ib = bl["public_metrics"]["total_impact_cost"]
        Vb = bl["public_metrics"]["total_volume"]
        Ve = ev["public_metrics"]["total_volume"]
        k = Ve / Vb if Vb > 0 else 1.0
        pnl_sized = k * Fb - (k ** 1.5) * Ib
        pnl_evolved = ev["combined_score"]
        ratio = pnl_evolved / pnl_sized if pnl_sized > 0 else 1.0
        return {
            "pnl_baseline": bl["combined_score"],
            "pnl_sized_up": pnl_sized,
            "pnl_evolved": pnl_evolved,
            "ratio": ratio,
            "volume_ratio": k,
            "sharpe_bl": bl["public_metrics"]["sharpe_ratio"],
            "sharpe_ev": ev["public_metrics"]["sharpe_ratio"],
            "calmar_bl": bl["public_metrics"]["calmar_ratio"],
            "calmar_ev": ev["public_metrics"]["calmar_ratio"],
        }

    return {
        "val": _decomp(bl_val, ev_val),
        "test": _decomp(bl_test, ev_test),
    }


# ============================================================
# Plotting
# ============================================================
def plot_cumulative_pnl(results_dir, baseline_code, best_code, out_dir, use_tiny=False):
    """Fig.3 风格: 累积 PnL 曲线."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, period in zip(axes, ["val", "test"]):
        bl = run_backtest(baseline_code, period, use_tiny)
        ev = run_backtest(best_code, period, use_tiny)
        if bl is None or ev is None:
            ax.set_title(f"{period} — no data")
            continue
        # Reconstruct equity from PnL components
        for label, data in [("Baseline", bl), ("Evolved", ev)]:
            eq = np.array(data.get("_equity_curve", []))
            if len(eq) == 0:
                pnl_arr = data.get("_pnl_adj_series")
                if pnl_arr is not None and len(pnl_arr) > 0:
                    eq = np.cumsum(pnl_arr)
            if len(eq) > 0:
                ax.plot(eq, linewidth=1.0, label=label)
        ax.set_title(f"{period.upper()} (2024)" if period == "val" else f"{period.upper()} (2025, OOS)")
        ax.set_xlabel("Minute")
        ax.set_ylabel("Cumulative Impact-Adj PnL (USD)")
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cumulative Impact-Adjusted PnL", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "cumulative_pnl.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_evolution_progress(results_dir, out_dir):
    """进化进度图: IS vs OOS."""
    history = load_history(results_dir)
    best_dir = Path(results_dir) / "best"
    baseline_code = ""
    best_code = ""
    baseline = load_baseline_results(results_dir)
    if baseline:
        baseline_code = baseline["code"]
    best = load_best_results(results_dir)
    if best:
        best_code = best.get("code", "")

    gens = [g["generation"] for g in history.get("generations", [])]
    scores = [g["best_score"] for g in history.get("generations", [])]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(scores)), scores, linewidth=2, color="#2196F3", label="IS (Validation)")
    ax.set_xlabel("Programs Evaluated")
    ax.set_ylabel("Best Impact-Adj PnL (USD)")
    ax.set_title("Evolution Progress", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "evolution_progress.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_sizing_decomposition(results_dir, baseline_code, best_code, out_dir, use_tiny=False):
    """Sizing Decomposition 柱状图."""
    decomp = sizing_decomposition(baseline_code, best_code, use_tiny)
    if decomp is None:
        print("  ⚠ Skipping sizing (need val + test data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, period in zip(axes, ["val", "test"]):
        d = decomp.get(period)
        if d is None:
            ax.set_title(f"{period} — N/A")
            continue
        bars = [d["pnl_baseline"], d["pnl_sized_up"], d["pnl_evolved"]]
        labels = ["Baseline", "Sized-up\nBaseline", "Evolved"]
        colors = ["#B0BEC5", "#90A4AE", "#1B5E20"]
        x = np.arange(3)
        ax.bar(x, bars, color=colors, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{period.upper()}")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        if d["ratio"] > 0:
            ax.annotate(f"{d['ratio']:.1f}×", xy=(2, d["pnl_evolved"]),
                        ha="center", va="bottom", fontweight="bold", fontsize=12, color="#1B5E20")

    fig.suptitle("Sizing Decomposition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "sizing_decomposition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_sharpe_calmar(results_dir, baseline_code, best_code, out_dir, use_tiny=False):
    """Sharpe/Calmar 柱状对比."""
    decomp = sizing_decomposition(baseline_code, best_code, use_tiny)
    if decomp is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = [
        ("Sharpe Ratio", "sharpe", "Sharpe Ratio (scale-invariant)"),
        ("Calmar Ratio", "calmar", "Calmar Ratio (scale-invariant)"),
    ]
    for ax, (title, key, full_title) in zip(axes, metrics):
        categories = ["Val BL", "Val EV", "Test BL", "Test EV"]
        vals = [
            decomp["val"][f"{key}_bl"],
            decomp["val"][f"{key}_ev"],
            decomp["test"][f"{key}_bl"],
            decomp["test"][f"{key}_ev"],
        ]
        colors = ["#B0BEC5", "#1B5E20", "#B0BEC5", "#1B5E20"]
        ax.bar(categories, vals, color=colors)
        ax.set_title(full_title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "sharpe_calmar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def print_model_stats(results_dir):
    """打印模型贡献表."""
    stats, _ = load_model_stats(results_dir)
    if stats is None:
        print("  ⚠ No model stats available")
        return
    print("\n  === Model Contribution ===")
    print(f"  {'Model':<30} {'Count':>6} {'Impr%':>8} {'MaxScore':>12} {'AvgImpr':>12}")
    print("  " + "-" * 68)
    for _, r in stats.iterrows():
        model = r["model"] or "unknown"
        print(f"  {model:<30} {int(r['total']):>6} {r['impr_rate']:>7.1f}% "
              f"${r['max_score']:>10,.0f} ${r['avg_improvement']:>10,.0f}")


def print_metrics_table(baseline, best):
    """打印 Table 1 风格指标表."""
    if baseline is None or best is None:
        return
    bl_m = baseline["public_metrics"]
    be_m = best.get("public_metrics", best)

    def _ratio(a, b):
        if b == 0:
            return "—"
        return f"{a/b:.2f}×"

    print("\n  === Metrics Summary (Table 1 style) ===")
    print(f"  {'Metric':<22} {'Baseline':>14} {'Evolved':>14} {'Ratio':>10}")
    print("  " + "-" * 60)
    rows = [
        ("Impact-Adj PnL", bl_m.get("pnl_impact_adj", 0), be_m.get("pnl_impact_adj", 0)),
        ("Sharpe Ratio", bl_m.get("sharpe_ratio", 0), be_m.get("sharpe_ratio", 0)),
        ("Calmar Ratio", bl_m.get("calmar_ratio", 0), be_m.get("calmar_ratio", 0)),
        ("Win Rate", bl_m.get("win_rate", 0), be_m.get("win_rate", 0)),
        ("Max Drawdown", bl_m.get("max_drawdown", 0), be_m.get("max_drawdown", 0)),
        ("Total Volume", bl_m.get("total_volume", 0), be_m.get("total_volume", 0)),
        ("Num Trades", bl_m.get("num_trades", 0), be_m.get("num_trades", 0)),
        ("Impact (bps)", bl_m.get("impact_bps", 0), be_m.get("impact_bps", 0)),
    ]
    for name, bl_v, be_v in rows:
        ratio = _ratio(be_v, bl_v) if isinstance(bl_v, (int, float)) and bl_v != 0 else "—"
        if isinstance(bl_v, float):
            print(f"  {name:<22} {bl_v:>14,.4f} {be_v:>14,.4f} {ratio:>10}")
        else:
            print(f"  {name:<22} {bl_v:>14,} {be_v:>14,} {ratio:>10}")


def main():
    parser = argparse.ArgumentParser(description="MadEvolve Results Analyzer")
    parser.add_argument("results_dir", help="Path to evolution results directory")
    parser.add_argument("--run-name", required=True,
                        help="Run identifier: 'run1', 'run2', or 'summary'")
    parser.add_argument("--fig-dir", default=None,
                        help="Base fig directory (default: PROJECT_ROOT/fig)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow backtest plots")
    parser.add_argument("--tiny", action="store_true",
                        help="Use tiny dataset (1 week) for fast backtest plots")
    args = parser.parse_args()

    results_dir = args.results_dir
    fig_base = args.fig_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig"
    )
    out_dir = os.path.join(fig_base, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Analyzing: {results_dir}")

    # Load data
    baseline = load_baseline_results(results_dir)
    best = load_best_results(results_dir)
    baseline_code = baseline["code"] if baseline else ""
    best_code = best.get("code", "") if best else ""

    # Print metrics
    print_metrics_table(baseline, best)
    print_model_stats(results_dir)

    # Plot
    print("\n=== Generating plots ===")
    if args.quick:
        print("  (--quick: skipping slow backtest plots)")
    if not args.quick and baseline_code and best_code:
        ut = args.tiny
        plot_cumulative_pnl(results_dir, baseline_code, best_code, out_dir, ut)
        plot_sizing_decomposition(results_dir, baseline_code, best_code, out_dir, ut)
        plot_sharpe_calmar(results_dir, baseline_code, best_code, out_dir, ut)
    plot_evolution_progress(results_dir, out_dir)

    print(f"\nDone! Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
