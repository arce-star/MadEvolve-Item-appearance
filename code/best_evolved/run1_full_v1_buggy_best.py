"""
基线交易策略 — Run 1 版本
EVOLVE-BLOCK 只包裹 set_target(), set_limit_order 固定不变
"""

import numpy as np

BUY = "B"
SELL = "A"


class DefaultPassiveExecutor:
    def __init__(
        self,
        sizing_factor=10_000,
        q_max=200_000,
        max_trade_frac=0.2,
        min_trade_size_usd=0,
        alpha_adjustment_knob=0.5,
        risk_reduction_factor=0.6,
        zp=0.0001,
        zp_riskoff=0.00003,
        fast_flat_minutes=10,
        std=1,
    ):
        self.sizing_factor = sizing_factor
        self.q_max = q_max
        self.max_trade_frac = max_trade_frac
        self.min_trade_size_usd = min_trade_size_usd
        self.alpha_adjustment_knob = alpha_adjustment_knob
        self.risk_reduction_factor = risk_reduction_factor
        self.zp = zp
        self.zp_riskoff = zp_riskoff
        self.fast_flat_minutes = fast_flat_minutes
        self.std = std
        self.context_correction_factor = 0

    # === EVOLVE-BLOCK-START ===
    def set_target(self, state):
            """Alpha + market state -> signed target trade side."""
            # ----- Tunable constants (all UPPER_CASE) -----
            ZP = 0.0001                    # base limit order depth (bps)
            STD = 1.0                      # standard deviation scaling
            SIZING_FACTOR = 10000.0        # base risk budget per sigma
            Q_MAX = 200000.0               # maximum position in USD
            MAX_TRADE_FRAC = 0.2           # max single trade as fraction of Q_MAX
            MIN_TRADE_SIZE_USD = 2000.0    # minimum dollar change to trigger trade
            LAG_DECAY_MINUTES = 15.0       # time for stale signal to decay
            SMOOTH_FACTOR = 0.05           # target smoothing (lower = slower)
            MIN_ALPHA_THRESHOLD = 0.0005   # minimum alpha to consider trade
            FEE_MAKER = 0.015 / 100        # 15 bps maker fee
            FEE_MIN = 0.005 / 100          # 5 bps min fee after depth rebate
            CONTEXT_CORRECTION = 0.0       # weight for realized alpha correction
            RISK_ADJUST_KNOB = 0.8         # risk reduction aggressiveness
            # -------------------------------------------------

            # Extract market state
            alpha = state["alpha"]
            sigma = state["alpha_sd"]
            mid = state["mid"]
            mid_book = state["mid_book"]
            q_btc = state["position_btc"]
            lag = state.get("data_lag_minutes", 0)

            q_usd = q_btc * mid_book

            # Realized alpha for optional context correction
            realized_alpha = np.log(mid_book / mid) if mid > 0 else 0.0
            alpha_corrected = alpha - CONTEXT_CORRECTION * realized_alpha

            # Expected fee after limit order depth (maker rebate)
            depth = STD * ZP
            expected_fee = max(FEE_MAKER - depth, FEE_MIN)

            # Hysteresis band: only adjust if signal pushes outside current position
            long_target_usd = SIZING_FACTOR * (alpha_corrected - expected_fee) / sigma
            short_target_usd = SIZING_FACTOR * (alpha_corrected + expected_fee) / sigma

            if long_target_usd > q_usd:
                desired_usd = long_target_usd
            elif short_target_usd < q_usd:
                desired_usd = short_target_usd
            else:
                desired_usd = q_usd

            # Risk adjustment: reduce exposure when inventory is large
            risk_adj = 1.0 - RISK_ADJUST_KNOB * np.tanh(abs(q_usd) / Q_MAX)
            desired_usd *= risk_adj

            # Stale‑signal decay
            lag_factor = max(0.0, 1.0 - lag / LAG_DECAY_MINUTES)
            desired_usd *= lag_factor

            # Smoothing to reduce overtrading
            if "_target_usd" not in state:
                state["_target_usd"] = desired_usd
            prev_target = state["_target_usd"]
            smoothed_target = (1.0 - SMOOTH_FACTOR) * prev_target + SMOOTH_FACTOR * desired_usd
            state["_target_usd"] = smoothed_target

            # Clip to maximum position
            target_pos_usd = np.clip(smoothed_target, -Q_MAX, Q_MAX)

            # ----- Convert USD target to BTC trade quantity -----
            target_pos_btc = target_pos_usd / mid_book
            raw_trade_btc = target_pos_btc - q_btc
            max_trade_btc = MAX_TRADE_FRAC * Q_MAX / mid_book
            trade_btc = np.clip(raw_trade_btc, -max_trade_btc, max_trade_btc)
            trade_usd = abs(trade_btc) * mid_book

            # Minimum trade size and alpha significance check
            if trade_usd < MIN_TRADE_SIZE_USD or abs(alpha_corrected) < MIN_ALPHA_THRESHOLD:
                side = None
                trade_btc = 0.0
            else:
                side = BUY if trade_btc > 0 else SELL

            # Risk reduction mode – tighten depth when stuck on wrong side with weak alpha
            wrong_direction = np.sign(q_btc * alpha) < 0
            small_alpha = abs(alpha) < expected_fee
            risk_reduction_mode = small_alpha and wrong_direction

            limit_order_depth = STD * ZP

            return {
                "side": side,
                "target_trade_qty": trade_btc,
                "risk_reduction_mode": risk_reduction_mode,
                "limit_order_depth": limit_order_depth,
            }
    # === EVOLVE-BLOCK-END ===

    def set_limit_order(self, state, target):
        """Signed target trade -> passive limit order. (FIXED in Run 1)"""
        if target["side"] is None:
            return None

        mid_book = state["mid_book"]
        side_multiplier = np.sign(target["target_trade_qty"])

        if target["risk_reduction_mode"]:
            depth = self.zp_riskoff * self.std
        else:
            depth = target["limit_order_depth"]

        limit_price = mid_book * np.exp(-side_multiplier * depth)
        return {
            "side": target["side"],
            "limit_price": limit_price,
            "target_trade_qty": target["target_trade_qty"],
        }

    def set_passive_order_data(self, state):
        target = self.set_target(state)
        return self.set_limit_order(state, target)


def apply_order_constraints(order, mid_book, max_limit_order_usd=100_000):
    """Non-evolvable post-processing."""
    if order is None:
        return None

    if order["side"] == BUY and order["target_trade_qty"] < 0:
        order["target_trade_qty"] = abs(order["target_trade_qty"])
    elif order["side"] == SELL and order["target_trade_qty"] > 0:
        order["target_trade_qty"] = -abs(order["target_trade_qty"])

    order_usd = abs(order["target_trade_qty"] * mid_book)
    if order_usd > max_limit_order_usd:
        max_btc = max_limit_order_usd / mid_book
        order["target_trade_qty"] = max_btc * np.sign(order["target_trade_qty"])

    return order
