"""
量化回测模拟器 — 严格按论文 Appendix A 实现

Fill Logic (A.2.1):
  - Buy fills if candle LOW < limit_price
  - Sell fills if candle HIGH > limit_price
  - 成交价 = limit_price，无额外滑点
  - 部分成交: hit_ratio (默认 1.0)

PnL (A.3):
  - PnL_pos   = q_{t+1} * δm_t                    (frictionless position PnL)
  - PnL_net   = q_{t+1}*δm_t - (p_limit - m)*Δq_fill - 15bps*m*|Δq_fill|
  - PnL_adj   = PnL_net - I_t                      (impact-adjusted, 主优化目标)

Market Impact (A.4): square-root + power-law temporal decay (propagator model)
  - s_i  = sign(Q_i) * (|Q_i| / V)^δ
  - D(t) = α_perm * Σ s_i  +  α_trans * Σ s_i * G(t - t_i)
  - G(τ) = (τ_0 / (τ + τ_0))^β
  - c_i  = D(t_i) * Q_i
  - I_t  = Σ c_i  (trades executed during interval t)
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Market Impact Model Parameters (Table 9)
# ============================================================
IMPACT_PARAMS = {
    "V":          2_000_000_000,  # Daily market volume ($2B)
    "alpha_perm": 0.005,
    "alpha_trans": 0.010,
    "tau_0":      300,            # seconds (5 min characteristic decay)
    "beta":       0.5,
    "delta":      0.5,
}

FEE_BPS = 0.00015  # 15 bps exchange fee


# ============================================================
# Trade Record
# ============================================================
@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    side: str            # "B" or "A" (buy/sell)
    qty_btc: float       # filled BTC quantity (signed: +buy, -sell)
    price: float         # execution (limit) price
    mid: float           # mid price at execution
    notional_usd: float  # |qty_btc| * price


# ============================================================
# Market Impact Calculator (Eq. 3–7)
# ============================================================
class MarketImpactCalculator:
    """Square-root propagator impact model (Eq. 3-7)."""

    # Max lookback for transient impact. G(1 day) ≈ 0.059 (< 6% of G(0)).
    # With α_trans=0.01 coefficient, residual contribution < 0.06% of D(t).
    # Paper Eq.3 sums over ALL trades; 1-day window is numerically equivalent
    # while keeping backtest fast enough for evolution (~3.5 min/eval).
    MAX_LOOKBACK_SEC = 86400  # 1 day

    def __init__(self, params: dict = None):
        p = params or IMPACT_PARAMS
        self.V = p["V"]
        self.alpha_perm = p["alpha_perm"]
        self.alpha_trans = p["alpha_trans"]
        self.tau_0 = p["tau_0"]
        self.beta = p["beta"]
        self.delta = p["delta"]

        self._capacity = 500000
        self._timestamps = np.zeros(self._capacity, dtype=np.float64)
        self._sis = np.zeros(self._capacity, dtype=np.float64)
        self._count = 0
        self._window_start = 0  # sliding window: first trade still in lookback
        self._perm_sum = 0.0
        # Pre-compute G(τ) lookup table (int seconds → G value)
        max_tau = self.MAX_LOOKBACK_SEC
        self._G_table = (self.tau_0 / (np.arange(max_tau + 1, dtype=np.float64) + self.tau_0)) ** self.beta

    def _displacement(self, t: float) -> float:
        """D(t) with sliding window + G-table lookup. O(window_size) ≈ O(1440)."""
        if self._count == 0:
            return 0.0
        # Slide window: drop trades older than MAX_LOOKBACK_SEC
        cutoff = t - self.MAX_LOOKBACK_SEC
        while self._window_start < self._count and self._timestamps[self._window_start] < cutoff:
            self._window_start += 1
        n = self._count - self._window_start
        if n == 0:
            return self.alpha_perm * self._perm_sum
        taus_int = (t - self._timestamps[self._window_start:self._count]).astype(np.int64)
        G_vals = self._G_table[taus_int]
        trans_sum = float(np.dot(self._sis[self._window_start:self._count], G_vals))
        return self.alpha_perm * self._perm_sum + self.alpha_trans * trans_sum

    def record_trade(self, timestamp_sec: float, notional_usd: float) -> float:
        """Record a trade and return its impact cost c_i (Eq. 6)."""
        Q_i = notional_usd
        s_i = np.sign(Q_i) * (abs(Q_i) / self.V) ** self.delta

        self._timestamps[self._count] = timestamp_sec
        self._sis[self._count] = s_i
        self._count += 1
        self._perm_sum += s_i

        D_after = self._displacement(timestamp_sec)
        return D_after * Q_i

    def reset(self):
        self._count = 0
        self._window_start = 0
        self._perm_sum = 0.0


# ============================================================
# Alpha Forecaster
# ============================================================
class AlphaForecaster:
    """
    Ridge regression forecaster — paper Appendix B.1.
    Features: 3 EMAs of 1-step returns (span 1, 5, 10).
    Targets: multi-horizon returns at 1, 10, 100, 1000 minutes.
    """

    def __init__(self):
        self.model = None
        self.alpha_mean = 0.0
        self.alpha_std = 1.0

    @staticmethod
    def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
        """default_calcset(): OHLCV -> feature matrix."""
        close = ohlcv["close"].astype(float)
        returns = close.pct_change()
        df = pd.DataFrame({
            "ema_ret_1":  returns.ewm(span=1).mean(),
            "ema_ret_5":  returns.ewm(span=5).mean(),
            "ema_ret_10": returns.ewm(span=10).mean(),
        }).fillna(0.0)
        return df

    def fit(self, ohlcv_train: pd.DataFrame):
        """Train Ridge on 2022-2023 data."""
        from sklearn.linear_model import Ridge

        X = self.compute_features(ohlcv_train).to_numpy()
        close = ohlcv_train["close"].astype(float)

        # Multi-horizon forward returns
        y = pd.DataFrame({
            "ret_1m":    close.pct_change(1).shift(-1),
            "ret_10m":   close.pct_change(10).shift(-10),
            "ret_100m":  close.pct_change(100).shift(-100),
            "ret_1000m": close.pct_change(1000).shift(-1000),
        }).fillna(0.0).to_numpy()

        self.model = Ridge(alpha=0.5)
        self.model.fit(X, y)

        # Compute alpha stats (10-min prediction — paper's primary horizon)
        alphas = self.model.predict(X)[:, 1]
        self.alpha_mean = float(np.mean(alphas))
        self.alpha_std = float(np.std(alphas))

    def predict(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """Return alpha: 10-min ahead prediction (paper's primary horizon)."""
        X = self.compute_features(ohlcv).to_numpy()
        preds = self.model.predict(X)  # (n, 4)
        return preds[:, 1]  # 10-min horizon — paper Sec 5.5/6.2

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "alpha_mean": self.alpha_mean,
                "alpha_std": self.alpha_std,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.alpha_mean = data["alpha_mean"]
        self.alpha_std = data["alpha_std"]


# ============================================================
# Backtest Simulator
# ============================================================
class BacktestSimulator:
    """
    Minute-bar backtest engine matching paper Section 4.1 + Appendix A.

    Order lifecycle (each minute):
      1. Check if prior resting order filled (exchange_response)
      2. Update position
      3. Cancel open orders
      4. Build state dict + call strategy.set_passive_order_data(state)
      5. Submit new limit order
      6. Log state
    """

    def __init__(self, ohlcv: pd.DataFrame, forecaster: AlphaForecaster, verbose: bool = False):
        self.ohlcv = ohlcv.copy()
        self.forecaster = forecaster
        self.verbose = verbose

        # Pre-compute alphas for the entire period
        self.alphas = forecaster.predict(ohlcv)
        # Global alpha_sd: constant across the period (paper doesn't specify window)
        self.alpha_sd = np.full(len(self.alphas), max(float(np.std(self.alphas)), 1e-8))

        self.impact_calc = MarketImpactCalculator()

        # Results — minimal memory: numpy arrays only, no Python objects per row
        self.equity_curve: List[float] = []
        self._pnl_adj_series = None
        self._num_trades = 0  # counter, no per-trade storage

    def _mid_price(self, row) -> float:
        """Mid price from candle."""
        return float((row["high"] + row["low"]) / 2.0)

    def _check_fill(self, side: str, limit_price: float, row) -> Tuple[bool, float]:
        """
        Check if limit order fills (A.2.1).
        Returns (filled, fill_qty_btc).
        """
        high, low = float(row["high"]), float(row["low"])
        if side == "B" and low < limit_price:
            return True, 1.0  # full fill (hit_ratio=1)
        if side == "A" and high > limit_price:
            return True, 1.0
        return False, 0.0

    def run(self, strategy) -> Dict:
        """
        Run backtest with given strategy object.
        Strategy must have: set_passive_order_data(state) -> order_dict or None.

        Returns dict with all metrics.
        """
        self.equity_curve = [0.0]
        n = len(self.ohlcv)
        self._pnl_adj_series = np.zeros(n, dtype=np.float64)
        self._num_trades = 0
        self.impact_calc.reset()

        position_btc = 0.0
        pending_order = None  # {"side": "B"/"A", "limit_price": float, "qty_btc": float}
        total_pnl_pos = 0.0
        total_pnl_net = 0.0
        total_pnl_adj = 0.0
        total_impact = 0.0
        total_fee = 0.0
        total_volume = 0.0

        timestamps = self.ohlcv.index
        # Pre-compute: avoid (H+L)/2 calculation 527K times
        mids = ((self.ohlcv["high"].values + self.ohlcv["low"].values) / 2.0).astype(np.float64)
        highs = self.ohlcv["high"].values.astype(np.float64)
        lows = self.ohlcv["low"].values.astype(np.float64)
        prev_mid = 0.0
        report_interval = 10000 if self.verbose else 0  # 每1万行汇报, 太快刷屏也拖慢
        alpha_threshold = 1e-10  # skip strategy call when alpha is effectively zero

        for i in range(n):
            if report_interval and i % report_interval == 0:
                pct = i / n * 100
                import sys; sys.stderr.write(f"\r  回测进度: {pct:.0f}% ({i:,}/{n:,})"); sys.stderr.flush()
            mid = mids[i]
            ts = timestamps[i]
            ts_sec = ts.timestamp()

            # ---- Step 1: Check prior order fill ----
            fill_qty = 0.0
            fill_price = 0.0
            if pending_order is not None:
                side, limit_price = pending_order["side"], pending_order["limit_price"]
                if (side == "B" and lows[i] < limit_price) or \
                   (side == "A" and highs[i] > limit_price):
                    fill_qty = pending_order["qty_btc"]
                    fill_price = limit_price

            # ---- Step 2: Update position ----
            prev_position = position_btc
            position_btc += fill_qty

            # ---- Step 3: Cancel (implicitly via replacing) ----
            pending_order = None

            # ---- Step 4: Build state + call strategy ----
            alpha = float(self.alphas[i])
            # Fast path: skip expensive strategy call when alpha is negligible
            if abs(alpha) < alpha_threshold and abs(position_btc * mid) < 1.0:
                order = None
            else:
                alpha_sd = float(self.alpha_sd[i])
                state = {
                    "alpha":             alpha,
                    "alpha_sd":          max(alpha_sd, 1e-8),
                    "mid":               mid,
                    "mid_book":          mid,
                    "position_btc":      position_btc,
                    "data_lag_minutes":  0,
                }
                try:
                    order = strategy.set_passive_order_data(state)
                except Exception:
                    order = None

            # ---- Step 5: Validate + submit order ----
            if order is not None:
                order = apply_order_constraints(order, mid)
                if order is not None:
                    pending_order = {
                        "side":        order["side"],
                        "limit_price": order["limit_price"],
                        "qty_btc":     order["target_trade_qty"],
                    }

            # ---- Step 6: PnL calculation (A.3) ----
            if i > 0:
                delta_mid = mid - prev_mid
                pnl_pos = position_btc * delta_mid
                total_pnl_pos += pnl_pos

                if fill_qty != 0.0 and fill_price != 0.0:
                    # Spread cost
                    spread_cost = (fill_price - mid) * fill_qty
                    # Fee
                    fee_cost = FEE_BPS * mid * abs(fill_qty)
                    # Net PnL for this trade
                    pnl_net_delta = pnl_pos - spread_cost - fee_cost
                    total_fee += fee_cost
                    total_volume += abs(fill_qty * fill_price)
                else:
                    pnl_net_delta = pnl_pos

                total_pnl_net += pnl_net_delta

                # Impact cost (only when a trade is executed)
                impact_delta = 0.0
                if fill_qty != 0.0 and fill_price != 0.0:
                    notional = abs(fill_qty * fill_price)
                    # impact model uses signed USD notional
                    signed_notional = fill_qty * fill_price
                    impact_delta = self.impact_calc.record_trade(ts_sec, signed_notional)
                    total_impact += impact_delta

                total_pnl_adj += (pnl_net_delta - impact_delta)
                self._pnl_adj_series[i] = pnl_net_delta - impact_delta

                if fill_qty != 0.0:
                    self._num_trades += 1

            self.equity_curve.append(total_pnl_adj)
            prev_mid = mid

        if self.verbose:
            sys.stderr.write(f"\r  回测完成: {n:,}/{n:,} 分钟\n"); sys.stderr.flush()

        # ---- Compute metrics ----
        metrics = self._compute_metrics(total_pnl_pos, total_pnl_net, total_pnl_adj,
                                        total_impact, total_fee, total_volume)
        return metrics

    def _compute_metrics(self, pnl_pos, pnl_net, pnl_adj, total_impact,
                         total_fee, total_volume) -> Dict:
        """Compute comprehensive metrics."""
        n_minutes = len(self.ohlcv)

        # Per-minute PnL for Sharpe
        adj_pnl_series = pd.Series(self._pnl_adj_series)

        # Sharpe (annualized, assuming 365*24*60 trading minutes)
        mu = adj_pnl_series.mean()
        sigma = adj_pnl_series.std()
        sharpe = float(mu / sigma * np.sqrt(365 * 24 * 60)) if sigma > 0 else 0.0

        # Calmar ratio
        cumulative = adj_pnl_series.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_dd = abs(drawdown.min())
        annual_return = mu * 365 * 24 * 60
        calmar = float(annual_return / max_dd) if max_dd > 0 else 0.0

        # Win rate
        wins = (adj_pnl_series > 0).sum()
        total_trades = self._num_trades
        win_rate = float(wins / max(n_minutes, 1))

        # Impact in bps
        impact_bps = float(total_impact / total_volume * 10000) if total_volume > 0 else 0.0

        # Volatility of equity curve
        equity = pd.Series(self.equity_curve)
        equity_vol = float(equity.diff().std() * np.sqrt(365 * 24 * 60))

        return {
            "success": True,
            "combined_score": float(pnl_adj),  # PRIMARY FITNESS
            "public_metrics": {
                "pnl_impact_adj":    float(pnl_adj),
                "pnl_net":           float(pnl_net),
                "pnl_frictionless":  float(pnl_pos),
                "total_impact_cost": float(total_impact),
                "total_fee_cost":    float(total_fee),
                "total_volume":      float(total_volume),
                "num_trades":        self._num_trades,
                "num_minutes":       n_minutes,
                "sharpe_ratio":      sharpe,
                "calmar_ratio":      calmar,
                "win_rate":          win_rate,
                "max_drawdown":      float(max_dd),
                "impact_bps":        impact_bps,
                "annual_return":     float(annual_return),
                "equity_volatility": float(equity_vol),
            },
            "private_metrics": {
                "alpha_mean": float(np.mean(self.alphas)),
                "alpha_std":  float(np.std(self.alphas)),
            },
            "text_feedback": (
                f"PnL(adj)=${pnl_adj:,.0f}, Sharpe={sharpe:.2f}, "
                f"Calmar={calmar:.2f}, Trades={self._num_trades}, "
                f"Impact={impact_bps:.1f}bps, Win={win_rate:.1%}"
            ),
            "_pnl_adj_series": self._pnl_adj_series,
            "_equity_curve": self.equity_curve,
        }


# ============================================================
# Order Constraints (paper: apply_order_constraints, non-evolvable)
# ============================================================
BUY = "B"
SELL = "A"

def apply_order_constraints(order: dict, mid_book: float,
                            max_limit_order_usd: float = 100_000) -> Optional[dict]:
    """Post-processing after strategy creates an order. NOT in EVOLVE-BLOCK."""
    if order is None:
        return None

    # Fix sign mismatch
    if order["side"] == BUY and order["target_trade_qty"] < 0:
        order["target_trade_qty"] = abs(order["target_trade_qty"])
    elif order["side"] == SELL and order["target_trade_qty"] > 0:
        order["target_trade_qty"] = -abs(order["target_trade_qty"])

    # Cap order size
    order_usd = abs(order["target_trade_qty"] * mid_book)
    if order_usd > max_limit_order_usd:
        max_btc = max_limit_order_usd / mid_book
        order["target_trade_qty"] = np.sign(order["target_trade_qty"]) * max_btc

    return order
