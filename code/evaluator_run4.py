#!/usr/bin/env python3
"""
Run4 Evaluator — 评估特征集预测质量 (R² + IC + ICIR)
每次评估 ~5秒, 不需要回测
"""

import argparse, importlib.util, json, os, sys, time
from pathlib import Path

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "btcusdt_1m_train.parquet")
VAL_PATH   = os.path.join(DATA_DIR, "btcusdt_1m_val.parquet")

# 论文 Eq.1: fitness = weighted combination
# Paper: "0.4*R²_10min + 0.3*mean_IC + 0.3*ICIR"
W_R2   = 0.4
W_IC   = 0.3
W_ICIR = 0.3 / 5.0   # paper: ICIR clamped to [-5,5], then scaled by 1/5


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.DatetimeIndex):
    """Compute R², mean daily IC, ICIR for 10-min horizon (column 1)."""
    # R² for 10-min horizon (index 1) — "no-intercept" per paper
    ss_res = np.sum((y_true[:, 1] - y_pred[:, 1]) ** 2)
    ss_tot = np.sum(y_true[:, 1] ** 2)  # no-intercept: Σy², not Σ(y-ȳ)²
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    # Daily IC (Spearman rank correlation between prediction and realized)
    unique_dates = np.unique(dates.date)
    daily_ics = []
    for di, date in enumerate(unique_dates):
        if di % 30 == 0:
            sys.stderr.write(f"\r  IC计算: {di}/{len(unique_dates)} 天"); sys.stderr.flush()
        mask = dates.date == date
        if mask.sum() < 10:
            continue
        # Use all-horizon composite for IC
        pred_comp = y_pred[mask].sum(axis=1)
        true_comp = y_true[mask][:, 1]  # 10-min return as target
        if np.std(pred_comp) < 1e-12 or np.std(true_comp) < 1e-12:
            continue
        ic, _ = spearmanr(pred_comp, true_comp)
        if not np.isnan(ic):
            daily_ics.append(ic)
    sys.stderr.write(f"\r  IC计算: {len(unique_dates)}/{len(unique_dates)} 天 ✓\n"); sys.stderr.flush()

    if len(daily_ics) == 0:
        return r2, 0.0, 0.0

    mean_ic = np.mean(daily_ics)
    icir = mean_ic / max(np.std(daily_ics, ddof=1), 1e-10)
    return r2, mean_ic, icir


def load_calcset(candidate_path: str):
    """Import default_calcset from candidate.py."""
    spec = importlib.util.spec_from_file_location("candidate", candidate_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "default_calcset"):
        raise ValueError("candidate.py must define default_calcset()")
    return getattr(mod, "default_calcset")


def build_targets(ohlcv: pd.DataFrame) -> np.ndarray:
    """Multi-horizon forward returns (same as paper Appendix B.1)."""
    close = ohlcv["close"].astype(float)
    y = pd.DataFrame({
        "ret_1m":    close.pct_change(1).shift(-1),
        "ret_10m":   close.pct_change(10).shift(-10),
        "ret_100m":  close.pct_change(100).shift(-100),
        "ret_1000m": close.pct_change(1000).shift(-1000),
    }).fillna(0.0)
    return y.to_numpy()


def main():
    parser = argparse.ArgumentParser(description="Run4 Evaluator — Forecast Quality")
    parser.add_argument("candidate", help="Path to candidate.py")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--verbose", action="store_true", help="Ignored (compat with dispatcher)")
    parser.add_argument("--val-data", default=None, help="Ignored (compat)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    try:
        # Save candidate source for debugging
        from datetime import datetime
        candidate_code = Path(args.candidate).read_text()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = results_dir / f"candidate_source_{ts}.py"
        log_file.write_text(candidate_code)
        sys.stderr.write(f"\n  [log] 候选代码: {log_file}\n"); sys.stderr.flush()

        calcset = load_calcset(args.candidate)

        # Train: compute features → fit Ridge
        val_path = args.val_data or VAL_PATH
        train_df = pd.read_parquet(TRAIN_PATH)
        X_train = np.nan_to_num(calcset(train_df).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        if X_train.shape[1] == 0 or np.all(np.std(X_train, axis=0) < 1e-15):
            raise ValueError(f"Invalid features: {X_train.shape[1]} cols, all constant or zero")
        y_train = build_targets(train_df)
        model = Ridge(alpha=0.5)
        model.fit(X_train, y_train)

        # Val: compute features → predict → score
        val_df = pd.read_parquet(val_path)
        X_val = np.nan_to_num(calcset(val_df).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        y_val = build_targets(val_df)
        y_pred = model.predict(X_val)

        r2, mean_ic, icir = compute_metrics(y_val, y_pred, val_df.index)
        # Paper: clamp each component, then weight
        r2_clamped = max(-1.0, min(1.0, r2))
        ic_clamped = max(-1.0, min(1.0, mean_ic))
        icir_clamped = max(-5.0, min(5.0, icir))
        combined = W_R2 * r2_clamped + W_IC * ic_clamped + W_ICIR * icir_clamped

        metrics = {
            "success": True,
            "combined_score": float(combined),
            "public_metrics": {
                "r2_10min":  float(r2),
                "mean_ic":   float(mean_ic),
                "icir":      float(icir),
                "num_features": X_train.shape[1],
            },
            "private_metrics": {},
            "text_feedback": (
                f"R²(10min)={r2:.4f}, IC={mean_ic:.4f}, ICIR={icir:.2f}, "
                f"Features={X_train.shape[1]}, Combined={combined:.4f}"
            ),
            "execution_time_sec": time.time() - t_start,
        }

        # Strip large data before JSON
        for k in ("_pnl_components", "_pnl_adj_series", "_equity_curve"):
            metrics.pop(k, None)

        with open(results_dir / "result.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[evaluator_run4] ✓ Score={combined:.4f} (R²={r2:.4f}, IC={mean_ic:.4f}, ICIR={icir:.2f})")

    except Exception as e:
        import traceback
        err = {
            "success": False,
            "combined_score": -1e10,
            "public_metrics": {},
            "private_metrics": {},
            "text_feedback": "",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "execution_time_sec": time.time() - t_start,
        }
        with open(results_dir / "result.json", "w") as f:
            json.dump(err, f, indent=2, default=str)
        print(f"[evaluator_run4] ✗ ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
