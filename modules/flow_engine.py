"""
flow_engine.py
──────────────
0DTE Options Flow Detection Module.

Detects and flags in real-time:
  • Unusual volume spikes (volume >> OI)
  • Aggressive ATM call / put buying
  • IV expansion bursts
  • Pinning risk zones
  • Trend continuation vs reversal probability
  • Liquidity heatmap
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.data_fetcher import ChainSnapshot, OptionRow

logger = logging.getLogger(__name__)

# ─── Thresholds (from env with defaults) ──────────────────────────────────────
UNUSUAL_VOL_MULT     = float(os.getenv("UNUSUAL_VOLUME_MULTIPLIER", 3.0))
IV_EXPANSION_THRESH  = float(os.getenv("IV_EXPANSION_THRESHOLD", 0.15))
GAMMA_FLIP_SENS      = float(os.getenv("GAMMA_FLIP_SENSITIVITY", 0.8))


# ─── Signal types ─────────────────────────────────────────────────────────────
class FlowType(str, Enum):
    UNUSUAL_CALL_BUY  = "Unusual Call Buying"
    UNUSUAL_PUT_BUY   = "Unusual Put Buying"
    IV_EXPANSION      = "IV Expansion Burst"
    PIN_RISK          = "Pin Risk Zone"
    CALL_SWEEP        = "Call Sweep (aggressive)"
    PUT_SWEEP         = "Put Sweep (aggressive)"
    OI_CLUSTER        = "OI Concentration"
    REVERSAL_SIGNAL   = "Potential Reversal Signal"
    CONTINUATION      = "Trend Continuation Signal"


@dataclass
class FlowSignal:
    type: FlowType
    strike: float
    severity: str        # "⚠️  Low" | "🔶 Medium" | "🚨 High"
    detail: str
    volume: int
    oi: int
    iv: float
    vol_oi_ratio: float


@dataclass
class FlowResult:
    ticker: str
    spot: float
    signals: List[FlowSignal]
    trend_bias: str            # "Continuation" | "Reversal" | "Undecided"
    trend_probability: float   # 0–1
    iv_regime: str             # "Expanding" | "Contracting" | "Stable"
    avg_atm_iv: float
    top_call_strike: Optional[float]
    top_put_strike: Optional[float]
    liquidity_heatmap: Dict[float, float]   # strike → 0–100


# ─── Engine ───────────────────────────────────────────────────────────────────
class FlowEngine:
    """
    Detects unusual options activity in 0DTE chains.
    """

    def compute(self, snapshot: ChainSnapshot) -> FlowResult:
        options = snapshot.zero_dte()
        if not options:
            options = snapshot.options

        spot   = snapshot.underlying.price
        calls  = [o for o in options if o.right == "C"]
        puts   = [o for o in options if o.right == "P"]

        signals: List[FlowSignal] = []

        signals += self._detect_unusual_volume(calls, "C")
        signals += self._detect_unusual_volume(puts,  "P")
        signals += self._detect_iv_expansion(options, spot)
        signals += self._detect_pin_risk(options, spot)
        signals += self._detect_sweeps(calls, puts, spot)
        signals += self._detect_oi_clusters(options, spot)

        # Sort by severity
        sev_order = {"🚨 High": 0, "🔶 Medium": 1, "⚠️  Low": 2}
        signals.sort(key=lambda s: sev_order.get(s.severity, 3))

        trend_bias, trend_prob = self._trend_signal(calls, puts, spot)
        iv_regime  = self._iv_regime(options, spot)
        avg_atm_iv = self._avg_atm_iv(options, spot)
        top_call   = self._top_strike(calls, spot)
        top_put    = self._top_strike(puts, spot)
        heatmap    = self._liquidity_heatmap(options)

        return FlowResult(
            ticker=snapshot.ticker,
            spot=spot,
            signals=signals,
            trend_bias=trend_bias,
            trend_probability=trend_prob,
            iv_regime=iv_regime,
            avg_atm_iv=avg_atm_iv,
            top_call_strike=top_call,
            top_put_strike=top_put,
            liquidity_heatmap=heatmap,
        )

    # ── Unusual Volume ─────────────────────────────────────────────────────────
    def _detect_unusual_volume(
        self, options: List[OptionRow], right: str
    ) -> List[FlowSignal]:
        label = "Call" if right == "C" else "Put"
        signals = []
        for o in options:
            if o.open_interest == 0:
                continue
            ratio = o.volume / max(o.open_interest, 1)
            if ratio >= UNUSUAL_VOL_MULT:
                severity = (
                    "🚨 High"   if ratio >= UNUSUAL_VOL_MULT * 2 else
                    "🔶 Medium" if ratio >= UNUSUAL_VOL_MULT * 1.5 else
                    "⚠️  Low"
                )
                signals.append(FlowSignal(
                    type     = FlowType.UNUSUAL_CALL_BUY if right == "C" else FlowType.UNUSUAL_PUT_BUY,
                    strike   = o.strike,
                    severity = severity,
                    detail   = (f"Vol/OI ratio {ratio:.1f}x at {label} {o.strike:.0f} "
                                f"(vol {o.volume:,} vs OI {o.open_interest:,})"),
                    volume   = o.volume,
                    oi       = o.open_interest,
                    iv       = o.iv,
                    vol_oi_ratio = round(ratio, 2),
                ))
        return signals[:4]   # cap per side

    # ── IV Expansion ───────────────────────────────────────────────────────────
    def _detect_iv_expansion(
        self, options: List[OptionRow], spot: float
    ) -> List[FlowSignal]:
        """
        Flag options where IV is significantly above the ATM baseline.
        Indicates sudden demand / potential catalyst awareness.
        """
        atm_ivs = [o.iv for o in options
                   if abs(o.strike - spot) / spot < 0.01 and o.iv > 0]
        if not atm_ivs:
            return []

        atm_baseline = float(np.median(atm_ivs))
        signals = []

        for o in options:
            if o.iv <= 0:
                continue
            expansion = (o.iv - atm_baseline) / max(atm_baseline, 0.01)
            if expansion >= IV_EXPANSION_THRESH:
                severity = (
                    "🚨 High"   if expansion >= 0.35 else
                    "🔶 Medium" if expansion >= 0.22 else
                    "⚠️  Low"
                )
                signals.append(FlowSignal(
                    type     = FlowType.IV_EXPANSION,
                    strike   = o.strike,
                    severity = severity,
                    detail   = (f"IV {o.iv:.1%} at {o.right} {o.strike:.0f} "
                                f"({expansion:+.0%} above ATM baseline {atm_baseline:.1%})"),
                    volume   = o.volume,
                    oi       = o.open_interest,
                    iv       = o.iv,
                    vol_oi_ratio = o.volume / max(o.open_interest, 1),
                ))
        return signals[:3]

    # ── Sweeps ─────────────────────────────────────────────────────────────────
    def _detect_sweeps(
        self, calls: List[OptionRow], puts: List[OptionRow], spot: float
    ) -> List[FlowSignal]:
        """
        Sweep = paying the ask on a large single-strike volume print.
        Proxy: volume > threshold AND ask > mid (i.e. tight spread, aggressive buy).
        """
        signals = []
        atm_range = spot * 0.015

        for options, ftype in [(calls, FlowType.CALL_SWEEP), (puts, FlowType.PUT_SWEEP)]:
            # Focus on ATM sweeps (most impactful)
            atm_opts = [o for o in options if abs(o.strike - spot) <= atm_range]
            # Rank by volume
            atm_opts.sort(key=lambda o: o.volume, reverse=True)
            for o in atm_opts[:2]:
                if o.volume < 500:
                    continue
                spread_tight = (o.ask - o.bid) < o.mid * 0.10
                if spread_tight and o.volume > o.open_interest * 0.5:
                    label = "call" if ftype == FlowType.CALL_SWEEP else "put"
                    signals.append(FlowSignal(
                        type     = ftype,
                        strike   = o.strike,
                        severity = "🚨 High" if o.volume > 2000 else "🔶 Medium",
                        detail   = (f"Aggressive {label} sweep at {o.strike:.0f}: "
                                    f"{o.volume:,} contracts (spread ${o.ask-o.bid:.2f})"),
                        volume   = o.volume,
                        oi       = o.open_interest,
                        iv       = o.iv,
                        vol_oi_ratio = o.volume / max(o.open_interest, 1),
                    ))
        return signals

    # ── Pin Risk ───────────────────────────────────────────────────────────────
    def _detect_pin_risk(
        self, options: List[OptionRow], spot: float
    ) -> List[FlowSignal]:
        """
        High combined OI near current spot = magnetic pin zone.
        """
        nearby = [o for o in options if abs(o.strike - spot) / spot < 0.008]
        if not nearby:
            return []

        oi_by_strike: Dict[float, int] = {}
        for o in nearby:
            oi_by_strike[o.strike] = oi_by_strike.get(o.strike, 0) + o.open_interest

        top_pin = max(oi_by_strike, key=oi_by_strike.get)
        top_oi  = oi_by_strike[top_pin]

        if top_oi < 1000:
            return []

        return [FlowSignal(
            type     = FlowType.PIN_RISK,
            strike   = top_pin,
            severity = "🔶 Medium" if top_oi < 10_000 else "🚨 High",
            detail   = (f"Pin magnet at {top_pin:.0f} with {top_oi:,} combined OI "
                        f"({abs(top_pin - spot):.2f} pts from spot)"),
            volume   = 0,
            oi       = top_oi,
            iv       = 0.0,
            vol_oi_ratio = 0.0,
        )]

    # ── OI Cluster ────────────────────────────────────────────────────────────
    def _detect_oi_clusters(
        self, options: List[OptionRow], spot: float
    ) -> List[FlowSignal]:
        """Flag strikes with outsized open interest — key dealer hedging levels."""
        oi_by_strike: Dict[float, int] = {}
        for o in options:
            oi_by_strike[o.strike] = oi_by_strike.get(o.strike, 0) + o.open_interest

        if not oi_by_strike:
            return []

        median_oi = float(np.median(list(oi_by_strike.values())))
        threshold  = median_oi * 3.5

        signals = []
        for strike, oi in sorted(oi_by_strike.items(), key=lambda x: -x[1])[:3]:
            if oi < threshold:
                continue
            signals.append(FlowSignal(
                type     = FlowType.OI_CLUSTER,
                strike   = strike,
                severity = "🔶 Medium",
                detail   = (f"OI cluster at {strike:.0f}: {oi:,} contracts "
                            f"({oi / max(median_oi, 1):.1f}x median)"),
                volume   = 0,
                oi       = oi,
                iv       = 0.0,
                vol_oi_ratio = 0.0,
            ))
        return signals

    # ── Trend Signal ──────────────────────────────────────────────────────────
    def _trend_signal(
        self, calls: List[OptionRow], puts: List[OptionRow], spot: float
    ) -> Tuple[str, float]:
        """
        Heuristic: if call volume is accelerating near ATM and PCR < 0.8 → continuation.
        High PCR + put sweeps near ATM → reversal.
        """
        call_vol = sum(o.volume for o in calls)
        put_vol  = sum(o.volume for o in puts)
        pcr = put_vol / max(call_vol, 1)

        atm_calls = [o for o in calls if abs(o.strike - spot) / spot < 0.015]
        atm_puts  = [o for o in puts  if abs(o.strike - spot) / spot < 0.015]
        atm_call_vol = sum(o.volume for o in atm_calls)
        atm_put_vol  = sum(o.volume for o in atm_puts)

        if atm_call_vol + atm_put_vol == 0:
            return "Undecided", 0.5

        atm_bias = (atm_call_vol - atm_put_vol) / (atm_call_vol + atm_put_vol)

        if atm_bias > 0.2 and pcr < 0.9:
            prob = min(0.5 + atm_bias * 0.5, 0.95)
            return "Continuation", round(prob, 2)
        elif atm_bias < -0.2 or pcr > 1.3:
            prob = min(0.5 + abs(atm_bias) * 0.5, 0.90)
            return "Reversal", round(prob, 2)
        else:
            return "Undecided", 0.50

    # ── IV Regime ─────────────────────────────────────────────────────────────
    def _iv_regime(self, options: List[OptionRow], spot: float) -> str:
        """Classify IV regime based on OTM IV skew vs ATM."""
        atm_iv = np.mean([o.iv for o in options
                          if abs(o.strike - spot) / spot < 0.01 and o.iv > 0] or [0])
        otm_iv = np.mean([o.iv for o in options
                          if abs(o.strike - spot) / spot > 0.02 and o.iv > 0] or [0])

        if otm_iv == 0 or atm_iv == 0:
            return "Stable"
        ratio = otm_iv / atm_iv
        if ratio > 1.15:
            return "Expanding"
        elif ratio < 0.90:
            return "Contracting"
        return "Stable"

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _avg_atm_iv(self, options: List[OptionRow], spot: float) -> float:
        ivs = [o.iv for o in options if abs(o.strike - spot) / spot < 0.01 and o.iv > 0]
        return round(float(np.mean(ivs)), 4) if ivs else 0.0

    def _top_strike(self, options: List[OptionRow], spot: float) -> Optional[float]:
        if not options:
            return None
        ranked = sorted(options, key=lambda o: o.volume, reverse=True)
        return ranked[0].strike if ranked else None

    def _liquidity_heatmap(self, options: List[OptionRow]) -> Dict[float, float]:
        score: Dict[float, float] = {}
        for o in options:
            combined = o.open_interest + o.volume * 5   # vol weighted more
            score[o.strike] = score.get(o.strike, 0) + combined
        if not score:
            return {}
        mx = max(score.values())
        return {k: round(v / mx * 100, 1) for k, v in sorted(score.items())}
