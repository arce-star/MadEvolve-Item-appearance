"""
基线特征集 — Run 4 版本
EVOLVE-BLOCK 包裹 default_calcset(), 其他所有代码不可变.

Run4 进化特征工程: 只改 default_calcset 函数体.
Ridge 模型固定 (alpha=0.5), 每次评估重训.
"""

import numpy as np
import pandas as pd


# === EVOLVE-BLOCK-START ===
# Tunable constants
SPAN_SHORT = 1
SPAN_MED = 5
SPAN_LONG = 10
VOL_SPAN = 20
MOMENTUM_SPAN = 12
VOLUME_SPAN = 10
VOL_EMA_SPAN = 20
ACCEL_SPAN = 3
VWR_SPAN = 10          # span for volume-weighted return EMA
ATR_SPAN = 14          # span for Average True Range
EMA_LONG_SPAN = 20     # longer return EMA span
VOL_OF_VOL_SPAN = 20   # span for volatility of volatility
HL_ROLL_SPAN = 10      # span for rolling average of high-low range
LAG3 = 3               # additional lag for returns
SHORT_VOL_SPAN = 3     # short-term volatility window
RET_ZSCORE_SPAN = 5    # window for return z-score
LAG5 = 5               # 5-minute lag for returns
LAG10 = 10             # 10-minute lag for returns
SHORT_PRICE_SPAN = 3   # short EMA span for price crossover
LONG_PRICE_SPAN = 10   # long EMA span for price crossover
EPS = 1e-12

def default_calcset(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV -> feature matrix with enhanced short-term features."""
    close = ohlcv["close"]
    volume = ohlcv["volume"].astype(float)
    high = ohlcv["high"]
    low = ohlcv["low"]
    open_ = ohlcv["open"]
    returns = close.pct_change()
    eps = EPS

    # Return EMAs (multiple time scales)
    ema_ret_1 = returns.ewm(span=SPAN_SHORT).mean()
    ema_ret_5 = returns.ewm(span=SPAN_MED).mean()
    ema_ret_10 = returns.ewm(span=SPAN_LONG).mean()
    ema_ret_20 = returns.ewm(span=EMA_LONG_SPAN).mean()

    # Volatility (rolling std of returns) - main window
    vol = returns.rolling(VOL_SPAN).std()

    # Short-term volatility
    vol_short = returns.rolling(SHORT_VOL_SPAN).std()

    # Volatility of volatility (rolling std of vol)
    vol_of_vol = vol.rolling(VOL_OF_VOL_SPAN).std()

    # Relative position: close relative to EMA
    close_ema = close / close.ewm(span=MOMENTUM_SPAN).mean() - 1.0

    # Volume relative to its EMA
    volume_ema = volume / volume.ewm(span=VOLUME_SPAN).mean() - 1.0

    # High-low range normalized by close
    hl_range = (high - low) / close

    # Smoothed high-low range (rolling mean)
    hl_smooth = hl_range.rolling(HL_ROLL_SPAN).mean()

    # Lagged returns (captures short-term reversal)
    ret_lag1 = returns.shift(1)
    ret_lag3 = returns.shift(LAG3)
    ret_lag5 = returns.shift(LAG5)
    ret_lag10 = returns.shift(LAG10)

    # Volatility relative to its own EMA (volatility regime)
    vol_ema = vol.ewm(span=VOL_EMA_SPAN).mean()
    vol_ema_ratio = vol / (vol_ema + eps) - 1.0

    # Intra-period return (close vs open)
    close_open = (close - open_) / (open_ + eps)

    # Return acceleration: fast EMA minus medium EMA
    ret_accel = returns.ewm(span=ACCEL_SPAN).mean() - ema_ret_5

    # Volume-weighted return EMA (captures volume-confirmed momentum)
    vwr = (returns * volume).ewm(span=VWR_SPAN).mean() / (volume.ewm(span=VWR_SPAN).mean() + eps)

    # Average True Range normalized by close (volatility from high-low)
    prev_close = close.shift(1)
    tr = np.maximum(high - low,
                    np.maximum(abs(high - prev_close),
                               abs(low - prev_close)))
    atr = tr.ewm(span=ATR_SPAN).mean() / (close + eps)

    # Return z-score: recent return normalized by short-term volatility
    ret_zscore = returns.rolling(RET_ZSCORE_SPAN).mean() / (vol_short + eps)

    # Price crossover: short EMA relative to long EMA
    price_crossover = (close.ewm(span=SHORT_PRICE_SPAN).mean() /
                       close.ewm(span=LONG_PRICE_SPAN).mean()) - 1.0

    return pd.DataFrame({
        "ema_ret_1": ema_ret_1,
        "ema_ret_5": ema_ret_5,
        "ema_ret_10": ema_ret_10,
        "ema_ret_20": ema_ret_20,
        "volatility_10": vol,
        "vol_short": vol_short,
        "vol_of_vol": vol_of_vol,
        "close_mom_10": close_ema,
        "volume_rel_10": volume_ema,
        "hl_range": hl_range,
        "hl_smooth": hl_smooth,
        "ret_lag1": ret_lag1,
        "ret_lag3": ret_lag3,
        "ret_lag5": ret_lag5,
        "ret_lag10": ret_lag10,
        "vol_ema_ratio": vol_ema_ratio,
        "close_open": close_open,
        "ret_accel": ret_accel,
        "vwr": vwr,
        "atr": atr,
        "ret_zscore": ret_zscore,
        "price_crossover": price_crossover,
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
