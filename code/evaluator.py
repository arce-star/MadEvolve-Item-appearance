#!/usr/bin/env python3
"""
MadEvolve Evaluator — 量化交易策略评估脚本

被 MadEvolve 以子进程调用:
    python evaluator.py <candidate.py> --results_dir <dir>

必须在 <results_dir>/result.json 输出结果.

使用方法:
    独立测试:  python evaluator.py baseline_run1.py --results_dir /tmp/test_eval
    进化调用:  由 MadEvolve dispatcher 自动调用

速度优化:
    - forecaster 预训练并缓存到磁盘
    - 数据预加载到内存
    - 单次评估目标在 5-30 秒内完成
"""

import argparse
import json
import os
import sys
import time
import importlib.util
from pathlib import Path

# 将项目根目录加入 path，确保可以导入 quant_simulator
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from quant_simulator import BacktestSimulator, AlphaForecaster


# ============================================================
# 路径配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "data")
FORECASTER_PATH = os.path.join(DATA_DIR, "forecaster.pkl")
VAL_DATA_PATH = os.path.join(DATA_DIR, "btcusdt_1m_val.parquet")


def load_strategy_class(candidate_path: str):
    """
    从 candidate.py 加载策略类。
    候选文件包含 DefaultPassiveExecutor 类（可能被 LLM 修改过）。
    """
    spec = importlib.util.spec_from_file_location("candidate_module", candidate_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find strategy class (DefaultPassiveExecutor or subclass)
    for name in ["DefaultPassiveExecutor"]:
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError(f"candidate.py 中未找到 DefaultPassiveExecutor 类: {candidate_path}")


def main():
    parser = argparse.ArgumentParser(description="MadEvolve Trading Strategy Evaluator")
    parser.add_argument("candidate", help="Path to candidate.py")
    parser.add_argument("--results_dir", required=True, help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Show backtest progress")
    parser.add_argument("--val-data", default=None, help="Override validation data path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / "result.json"

    t_start = time.time()

    try:
        # ---- 1. Load data ----
        val_path = args.val_data or VAL_DATA_PATH
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"验证数据不存在: {val_path}")
        ohlcv_val = pd.read_parquet(val_path)

        # ---- 2. Load or train forecaster ----
        forecaster = AlphaForecaster()
        if os.path.exists(FORECASTER_PATH):
            forecaster.load(FORECASTER_PATH)
        else:
            # Auto-train if not cached
            train_path = os.path.join(DATA_DIR, "btcusdt_1m_train.parquet")
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"训练数据不存在: {train_path}")
            print(f"[evaluator] 训练 forecaster (缓存到 {FORECASTER_PATH})...")
            ohlcv_train = pd.read_parquet(train_path)
            forecaster.fit(ohlcv_train)
            os.makedirs(DATA_DIR, exist_ok=True)
            forecaster.save(FORECASTER_PATH)

        # ---- 3. Load candidate strategy ----
        StrategyClass = load_strategy_class(args.candidate)
        strategy = StrategyClass()

        # ---- 4. Run backtest ----
        simulator = BacktestSimulator(ohlcv_val, forecaster, verbose=args.verbose)
        metrics = simulator.run(strategy)

        # ---- 5. Write result ----
        # Strip large raw data before JSON serialization (keeps scoring fields only)
        for k in ("_pnl_components", "_pnl_adj_series", "_equity_curve"):
            metrics.pop(k, None)
        elapsed = time.time() - t_start
        metrics["execution_time_sec"] = elapsed
        metrics["success"] = True

        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        pnl = metrics["combined_score"]
        sharpe = metrics["public_metrics"]["sharpe_ratio"]
        print(f"[evaluator] ✓ PnL(adj)=${pnl:,.0f} Sharpe={sharpe:.2f} ({elapsed:.1f}s)")

    except Exception as e:
        import traceback
        error_result = {
            "success": False,
            "combined_score": -1e10,
            "public_metrics": {},
            "private_metrics": {},
            "text_feedback": "",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "execution_time_sec": time.time() - t_start,
        }
        with open(result_path, "w") as f:
            json.dump(error_result, f, indent=2, default=str)
        print(f"[evaluator] ✗ ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
