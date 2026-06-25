"""
基线特征集 — Run 4 版本
EVOLVE-BLOCK 包裹 default_calcset(), 其他所有代码不可变.

Run4 进化特征工程: 只改 default_calcset 函数体.
Ridge 模型固定 (alpha=0.5), 每次评估重训.
"""

import numpy as np
import pandas as pd


# === EVOLVE-BLOCK-START ===
def default_calcset(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV -> feature matrix (baseline: 3 EMAs)."""
    close = ohlcv["close"]
    returns = close.pct_change()
    return pd.DataFrame({
        "ema_ret_1":  returns.ewm(span=1).mean(),
        "ema_ret_5":  returns.ewm(span=5).mean(),
        "ema_ret_10": returns.ewm(span=10).mean(),
    }).fillna(0)
# === EVOLVE-BLOCK-END ===


def train_forecaster(ohlcv: pd.DataFrame):
    """Fit Ridge on features to predict multi-horizon future returns."""
    X = default_calcset(ohlcv).to_numpy()
    from sklearn.linear_model import Ridge
    y = pd.DataFrame({
        "return_1m":    ohlcv["close"].pct_change(1).shift(-1),
        "return_10m":   ohlcv["close"].pct_change(10).shift(-10),
        "return_100m":  ohlcv["close"].pct_change(100).shift(-100),
        "return_1000m": ohlcv["close"].pct_change(1000).shift(-1000),
    }).fillna(0)
    model = Ridge(alpha=0.5)
    model.fit(X, y.to_numpy())
    return model


def forecast_alpha(ohlcv: pd.DataFrame, model):
    """Fresh OHLCV -> features -> alpha predictions at each horizon."""
    X = default_calcset(ohlcv).to_numpy()
    return model.predict(X)
