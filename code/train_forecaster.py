#!/usr/bin/env python3
"""
预训练 Alpha Forecaster (Ridge Regression)
在 train (2022-2023) 上拟合，保存到 data/forecaster.pkl
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from quant_simulator import AlphaForecaster

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "btcusdt_1m_train.parquet")
FORECASTER_PATH = os.path.join(DATA_DIR, "forecaster.pkl")


def main():
    if not os.path.exists(TRAIN_PATH):
        print(f"训练数据不存在: {TRAIN_PATH}")
        print("请先运行 download_data.py 下载数据")
        sys.exit(1)

    print(f"加载训练数据: {TRAIN_PATH}")
    ohlcv_train = pd.read_parquet(TRAIN_PATH)
    print(f"  数据范围: {ohlcv_train.index[0]} ~ {ohlcv_train.index[-1]}")
    print(f"  共 {len(ohlcv_train):,} 分钟")

    forecaster = AlphaForecaster()
    forecaster.fit(ohlcv_train)

    print(f"\n训练完成:")
    print(f"  alpha_mean = {forecaster.alpha_mean:.6f}")
    print(f"  alpha_std  = {forecaster.alpha_std:.6f}")
    print(f"  R^2 (train) = {forecaster.model.score(forecaster.compute_features(ohlcv_train).to_numpy(), forecaster.model.predict(forecaster.compute_features(ohlcv_train).to_numpy())):.4f}")  # misleading but indicative

    # Compute R2 on validation for sanity check
    val_path = os.path.join(DATA_DIR, "btcusdt_1m_val.parquet")
    if os.path.exists(val_path):
        val = pd.read_parquet(val_path)
        X_val = forecaster.compute_features(val).to_numpy()
        close = val["close"].astype(float)
        y_val = pd.DataFrame({
            "ret_1m":    close.pct_change(1).shift(-1),
            "ret_10m":   close.pct_change(10).shift(-10),
            "ret_100m":  close.pct_change(100).shift(-100),
            "ret_1000m": close.pct_change(1000).shift(-1000),
        }).fillna(0.0).to_numpy()
        r2_val = forecaster.model.score(X_val, y_val)
        print(f"  R^2 (val)   = {r2_val:.6f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    forecaster.save(FORECASTER_PATH)
    print(f"\n已保存: {FORECASTER_PATH}")


if __name__ == "__main__":
    main()
