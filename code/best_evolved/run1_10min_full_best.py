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
            # --- tunable parameters (UPPER_CASE) ---
            SIZING_FACTOR = 20000.0            # Base risk scaling
            TOTAL_COST_RETURN = 0.0003         # Expected one-way cost (fee+spread) in return (3 bps)
            POSITION_LIMIT = 200000.0          # Max absolute position in USD
            MAX_TRADE_FRAC = 0.15              # Max trade as fraction of position limit
            INVENTORY_DECAY = 0.02             # Mean‑reversion strength on inventory
            LAG_DECAY_RATE = 0.1               # Exponential decay per minute for stale signals
            RISK_REDUCTION_FACTOR = 0.8        # Scale target toward zero in risk mode
            RISK_ALPHA_THRESH_MULT = 1.5       # Multiple of cost for risk‑trigger alpha threshold
            RISK_POS_THRESH_FRAC = 0.5         # Fraction of position limit for risk trigger
            MIN_TRADE_USD = 500.0              # Minimum trade size in USD
            SMOOTHING = 0.85                   # EMA weight for new target (0=no smoothing)
            # --- end of tunable parameters ---

            alpha = state["alpha"]
            alpha_sd = max(state["alpha_sd"], 1e-12)
            mid = state["mid"]
            mid_book = state["mid_book"]
            q_x = state["position_btc"]
            data_lag_minutes = state.get("data_lag_minutes", 0)

            # Passive limit depth (kept from init)
            limit_order_depth = self.std * self.zp

            # Realised alpha for mean‑reversion (set CONTEXT_CORRECTION=0 to disable)
            CONTEXT_CORRECTION = 0.0
            realized_alpha = np.log(mid_book / mid) if mid > 0 and mid_book > 0 else 0.0
            alpha_corrected = alpha - CONTEXT_CORRECTION * realized_alpha

            # Current position in USD
            pos_usd = np.nan_to_num(q_x * mid_book)

            # Signal strength in return space
            signal = alpha_corrected

            # Deadband: skip trade when signal is too weak to cover costs
            if abs(signal) < TOTAL_COST_RETURN:
                return {
                    "side": None,
                    "target_trade_qty": 0.0,
                    "risk_reduction_mode": False,
                    "limit_order_depth": limit_order_depth,
                }

            # Raw desired position (cost‑adjusted to avoid adverse selection)
            raw_target_usd = SIZING_FACTOR * (signal - np.sign(signal) * TOTAL_COST_RETURN) / alpha_sd

            # Inventory mean‑reversion
            target_usd = raw_target_usd - INVENTORY_DECAY * pos_usd

            # Lag penalty
            lag_factor = np.exp(-LAG_DECAY_RATE * data_lag_minutes)
            target_usd *= lag_factor

            # Risk reduction mode: when signal is weak and position is large opposite
            risk_thresh = RISK_ALPHA_THRESH_MULT * TOTAL_COST_RETURN
            risk_reduction_mode = (abs(signal) < risk_thresh) and (abs(pos_usd) > RISK_POS_THRESH_FRAC * POSITION_LIMIT)

            if risk_reduction_mode:
                # Scale position toward zero
                target_usd = pos_usd * RISK_REDUCTION_FACTOR

            # Clamp to position limits
            target_usd = np.clip(target_usd, -POSITION_LIMIT, POSITION_LIMIT)

            # Smooth target to reduce turnover (EMA)
            if not hasattr(self, "_prev_target_usd"):
                self._prev_target_usd = pos_usd
            target_usd = SMOOTHING * target_usd + (1.0 - SMOOTHING) * self._prev_target_usd
            self._prev_target_usd = target_usd

            # Convert to trade quantity
            target_btc = target_usd / mid_book
            raw_trade_qty = target_btc - q_x

            # Apply max trade fraction
            max_trade_btc = MAX_TRADE_FRAC * POSITION_LIMIT / mid_book
            trade_qty = np.clip(raw_trade_qty, -max_trade_btc, max_trade_btc)

            # Final check: minimum trade size in USD
            delta_usd = abs(trade_qty * mid_book)
            if delta_usd < MIN_TRADE_USD:
                side = None
                trade_qty = 0.0
            else:
                side = BUY if trade_qty > 0 else SELL

            return {
                "side": side,
                "target_trade_qty": trade_qty,
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
