"""
基线交易策略 — Run 2 版本
EVOLVE-BLOCK 只包裹 set_limit_order(), set_target 固定不变
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

    def set_target(self, state):
        """Alpha + market state -> signed target trade side. (FIXED in Run 2)"""
        alpha = state["alpha"]
        alpha_sd = state["alpha_sd"]
        mid = state["mid"]
        mid_book = state["mid_book"]
        q_x = state["position_btc"]
        data_lag_minutes = state.get("data_lag_minutes", 0)

        limit_order_depth = self.std * self.zp
        expected_fee = max(0.015 / 100 - limit_order_depth, 0.005 / 100)

        realized_alpha = np.log(mid_book / mid)
        alpha_corrected = alpha - self.context_correction_factor * realized_alpha
        q_usd = np.nan_to_num(q_x * mid_book)
        k = self.sizing_factor / alpha_sd

        small_alpha = abs(alpha_corrected - q_usd / k) < expected_fee
        wrong_direction = np.sign(q_x * alpha_corrected) < 0
        risk_reduction_mode = small_alpha and wrong_direction

        if risk_reduction_mode:
            target_pos_usd = q_usd * self.risk_reduction_factor
        elif (
            abs(realized_alpha) * self.context_correction_factor > abs(alpha)
            and np.sign(realized_alpha * alpha) > 0
        ):
            target_pos_usd = q_usd
        else:
            long_target_usd = self.sizing_factor * (alpha_corrected - expected_fee) / alpha_sd
            short_target_usd = self.sizing_factor * (alpha_corrected + expected_fee) / alpha_sd

            if long_target_usd > q_usd:
                target_pos_usd = long_target_usd
            elif short_target_usd < q_usd:
                target_pos_usd = short_target_usd
            else:
                target_pos_usd = q_usd

        lag_adjustment = 1 - min(data_lag_minutes, self.fast_flat_minutes) / self.fast_flat_minutes

        if risk_reduction_mode:
            correction_factor = lag_adjustment
        else:
            risk_adjustment = 1 - np.tanh(abs(q_usd) / self.q_max) * self.alpha_adjustment_knob
            correction_factor = risk_adjustment * lag_adjustment

        target_pos_usd = np.clip(target_pos_usd * correction_factor, -self.q_max, self.q_max)
        target_pos_btc = target_pos_usd / mid_book
        raw_trade_qty = target_pos_btc - q_x
        max_trade_btc = self.max_trade_frac * self.q_max / mid_book
        target_trade_qty = np.clip(raw_trade_qty, -max_trade_btc, max_trade_btc)

        delta_usd = abs(target_pos_btc - q_x) * mid_book
        if target_pos_btc > q_x and delta_usd > self.min_trade_size_usd:
            side = BUY
        elif target_pos_btc < q_x and delta_usd > self.min_trade_size_usd:
            side = SELL
        else:
            side = None

        return {
            "side": side,
            "target_trade_qty": target_trade_qty,
            "risk_reduction_mode": risk_reduction_mode,
            "limit_order_depth": limit_order_depth,
        }

    # === EVOLVE-BLOCK-START ===
    def set_limit_order(self, state, target):
        """Signed target trade -> passive limit order."""
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
    # === EVOLVE-BLOCK-END ===

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
