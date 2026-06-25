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
            # ── tunable constants (UPPER_CASE, ≤15) ──
            SIZING_FACTOR = 22_000               # Base alpha-to-USD scaling
            Q_MAX = 220_000                      # USD position cap
            MAX_TRADE_FRAC = 0.28                # Max fraction of Q_MAX per trade
            MIN_TRADE_SIZE_USD = 0.0             # Minimum trade size (USD)
            ALPHA_ADJ_KNOB = 0.35                # Inventory aversion strength
            RISK_REDUCTION_FACTOR = 0.45         # Position cut in risk-off mode
            ZP = 0.00010                         # Normal limit depth (log‑return)
            ZP_RISKOFF = 0.00002                 # Risk‑off limit depth (log‑return)
            STD = 1.0                            # Volatility scaling for depth
            CONTEXT_CORRECTION_FACTOR = 0.10     # Stale‑signal correction weight (higher)
            STALE_ALPHA_DECAY = 0.60             # Attenuation when realized contradicts (stronger)
            SMALL_ALPHA_THRESHOLD_FACTOR = 0.60  # Tighter threshold for risk reduction (more sensitive)
            LAG_DECAY_RATE = 0.30                # Exponential lag decay per minute (faster)
            SIGMA_FEE_MIN = 0.005 / 100          # Floor expected fee (0.5 bps)

            # ── Sync instance attributes so set_limit_order uses evolved values ──
            self.sizing_factor = SIZING_FACTOR
            self.q_max = Q_MAX
            self.max_trade_frac = MAX_TRADE_FRAC
            self.min_trade_size_usd = MIN_TRADE_SIZE_USD
            self.alpha_adjustment_knob = ALPHA_ADJ_KNOB
            self.risk_reduction_factor = RISK_REDUCTION_FACTOR
            self.zp = ZP
            self.zp_riskoff = ZP_RISKOFF
            self.std = STD
            self.context_correction_factor = CONTEXT_CORRECTION_FACTOR
            self.fast_flat_minutes = 1  # unused, set to avoid errors

            # ── Read state ──
            alpha = state["alpha"]
            alpha_sd = state["alpha_sd"]
            mid = state["mid"]
            mid_book = state["mid_book"]
            q_x = state["position_btc"]
            data_lag_minutes = state.get("data_lag_minutes", 0)

            # ── Fee computation (consistent with set_limit_order normal mode) ──
            limit_order_depth = STD * ZP
            expected_fee = max(0.015 / 100 - limit_order_depth, SIGMA_FEE_MIN)

            # ── Stale‑signal correction ──
            epsilon = 1e-12
            realized_alpha = np.log(mid_book / max(mid, epsilon))
            alpha_corrected = alpha - CONTEXT_CORRECTION_FACTOR * realized_alpha

            # Additional stale‑signal decay when realized moves opposite
            if (realized_alpha * alpha_corrected < 0
                and abs(realized_alpha) > 0.5 * abs(alpha_corrected)):
                alpha_corrected *= (1.0 - STALE_ALPHA_DECAY)

            # ── Inventory & scaling ──
            q_usd = np.nan_to_num(q_x * mid_book)
            k = SIZING_FACTOR / max(alpha_sd, epsilon)

            # ── Risk‑reduction detection ──
            small_alpha = abs(alpha_corrected - q_usd / k) < expected_fee * SMALL_ALPHA_THRESHOLD_FACTOR
            wrong_direction = np.sign(q_x * alpha_corrected) < 0
            risk_reduction_mode = small_alpha and wrong_direction

            # ── Target position from long/short bands ──
            if risk_reduction_mode:
                target_pos_usd = q_usd * RISK_REDUCTION_FACTOR
            elif (
                abs(realized_alpha) * CONTEXT_CORRECTION_FACTOR > abs(alpha)
                and np.sign(realized_alpha * alpha) > 0
            ):
                # Realized move validates current position – hold
                target_pos_usd = q_usd
            else:
                long_target_usd = SIZING_FACTOR * (alpha_corrected - expected_fee) / max(alpha_sd, epsilon)
                short_target_usd = SIZING_FACTOR * (alpha_corrected + expected_fee) / max(alpha_sd, epsilon)

                if long_target_usd > q_usd:
                    target_pos_usd = long_target_usd
                elif short_target_usd < q_usd:
                    target_pos_usd = short_target_usd
                else:
                    target_pos_usd = q_usd

            # ── Correction factors ──
            lag_adjustment = np.exp(-LAG_DECAY_RATE * data_lag_minutes)

            if risk_reduction_mode:
                correction_factor = lag_adjustment
            else:
                risk_adjustment = 1.0 - np.tanh(abs(q_usd) / Q_MAX) * ALPHA_ADJ_KNOB
                correction_factor = risk_adjustment * lag_adjustment

            target_pos_usd = np.clip(target_pos_usd * correction_factor, -Q_MAX, Q_MAX)
            target_pos_btc = target_pos_usd / mid_book
            raw_trade_qty = target_pos_btc - q_x
            max_trade_btc = MAX_TRADE_FRAC * Q_MAX / mid_book
            target_trade_qty = np.clip(raw_trade_qty, -max_trade_btc, max_trade_btc)

            delta_usd = abs(target_pos_btc - q_x) * mid_book
            if target_pos_btc > q_x and delta_usd > MIN_TRADE_SIZE_USD:
                side = BUY
            elif target_pos_btc < q_x and delta_usd > MIN_TRADE_SIZE_USD:
                side = SELL
            else:
                side = None

            return {
                "side": side,
                "target_trade_qty": target_trade_qty,
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
