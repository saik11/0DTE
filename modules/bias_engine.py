"""
bias_engine.py
──────────────
Daily Market Bias Engine for 0DTE options.

Combines 5 independent signals into a weighted confidence score:
  1. Call/Put volume imbalance (0DTE)
  2. ATM strike pressure (±1% from spot)
  3. IV expansion vs contraction
  4. Price position vs max pain
  5. Strike concentration cluster asymmetry

Output:
  • Bullish / Bearish / Neutral
  • Confidence score 0–100
  • 1–2 line reasoning
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.data_fetcher import ChainSnapshot, OptionRow
from modules.gamma_engine import GammaEngine, GammaResult

logger = logging.getLogger(__name__)

# ─── Signal weights (must sum to 1.0) ────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "volume_imbalance":  0.30,
    "atm_pressure":      0.25,
    "iv_structure":      0.20,
    "max_pain_position": 0.15,
    "cluster_asymmetry": 0.10,
}


# ─── Output container ─────────────────────────────────────────────────────────
@dataclass
class SignalBreakdown:
    name: str
    raw_value: float     # -1 (bearish) to +1 (bullish)
    weight: float
    contribution: float  # weighted contribution
    detail: str          # human readable


@dataclass
class BiasResult:
    ticker: str
    spot: float
    direction: str           # "Bullish" | "Bearish" | "Neutral"
    confidence: int          # 0–100
    reasoning: str           # 1–2 line summary
    signals: List[SignalBreakdown]
    call_volume: int
    put_volume: int
    put_call_ratio: float
    atm_iv: float


# ─── Engine ───────────────────────────────────────────────────────────────────
class BiasEngine:
    """
    Computes directional market bias from 0DTE options flow.
    """

    def __init__(self, gamma_engine: Optional[GammaEngine] = None):
        self._gamma_engine = gamma_engine

    def compute(
        self, snapshot: ChainSnapshot, gamma_result: Optional[GammaResult] = None
    ) -> BiasResult:
        options = snapshot.zero_dte()
        if not options:
            options = snapshot.options   # fallback

        spot = snapshot.underlying.price
        calls = [o for o in options if o.right == "C"]
        puts  = [o for o in options if o.right == "P"]

        # Individual signals
        sig_vol  = self._signal_volume_imbalance(calls, puts)
        sig_atm  = self._signal_atm_pressure(calls, puts, spot)
        sig_iv   = self._signal_iv_structure(calls, puts, spot)
        sig_pain = self._signal_max_pain_position(spot, gamma_result)
        sig_clus = self._signal_cluster_asymmetry(calls, puts, spot)

        signals = [sig_vol, sig_atm, sig_iv, sig_pain, sig_clus]

        # Weighted composite score [-1, +1]
        composite = sum(s.contribution for s in signals)
        composite = max(-1.0, min(1.0, composite))

        # Map to direction + confidence
        direction, confidence = self._classify(composite)
        reasoning = self._build_reasoning(signals, direction, spot)

        # Aggregate stats
        call_vol = sum(o.volume for o in calls)
        put_vol  = sum(o.volume for o in puts)
        pcr      = round(put_vol / max(call_vol, 1), 3)
        atm_iv   = self._atm_iv(options, spot)

        return BiasResult(
            ticker=snapshot.ticker,
            spot=spot,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            signals=signals,
            call_volume=call_vol,
            put_volume=put_vol,
            put_call_ratio=pcr,
            atm_iv=atm_iv,
        )

    # ── Signal 1: Volume Imbalance ─────────────────────────────────────────────
    def _signal_volume_imbalance(
        self, calls: List[OptionRow], puts: List[OptionRow]
    ) -> SignalBreakdown:
        w = SIGNAL_WEIGHTS["volume_imbalance"]

        call_vol = sum(o.volume for o in calls)
        put_vol  = sum(o.volume for o in puts)
        total    = call_vol + put_vol

        if total == 0:
            raw = 0.0
            detail = "No volume data"
        else:
            # +1 = all calls, -1 = all puts
            raw = (call_vol - put_vol) / total
            pcr = put_vol / max(call_vol, 1)
            detail = (f"C/P Vol ratio {call_vol:,}/{put_vol:,} "
                      f"(PCR={pcr:.2f})")

        return SignalBreakdown(
            name="Volume Imbalance",
            raw_value=round(raw, 4),
            weight=w,
            contribution=round(raw * w, 4),
            detail=detail,
        )

    # ── Signal 2: ATM Pressure ────────────────────────────────────────────────
    def _signal_atm_pressure(
        self, calls: List[OptionRow], puts: List[OptionRow], spot: float
    ) -> SignalBreakdown:
        w = SIGNAL_WEIGHTS["atm_pressure"]
        threshold = spot * 0.01   # ±1%

        atm_calls = [o for o in calls if abs(o.strike - spot) <= threshold]
        atm_puts  = [o for o in puts  if abs(o.strike - spot) <= threshold]

        call_vol = sum(o.volume for o in atm_calls)
        put_vol  = sum(o.volume for o in atm_puts)
        total    = call_vol + put_vol

        if total == 0:
            raw = 0.0
            detail = "No ATM activity"
        else:
            raw = (call_vol - put_vol) / total
            detail = (f"ATM ±1%: calls {call_vol:,} vs puts {put_vol:,}")

        return SignalBreakdown(
            name="ATM Pressure",
            raw_value=round(raw, 4),
            weight=w,
            contribution=round(raw * w, 4),
            detail=detail,
        )

    # ── Signal 3: IV Structure ────────────────────────────────────────────────
    def _signal_iv_structure(
        self, calls: List[OptionRow], puts: List[OptionRow], spot: float
    ) -> SignalBreakdown:
        """
        Compare OTM call IV vs OTM put IV.
        Skew toward OTM calls = bullish demand; toward OTM puts = bearish hedging.
        """
        w = SIGNAL_WEIGHTS["iv_structure"]

        otm_calls = [o for o in calls if o.strike > spot * 1.005 and o.iv > 0]
        otm_puts  = [o for o in puts  if o.strike < spot * 0.995 and o.iv > 0]

        call_iv_avg = np.mean([o.iv for o in otm_calls]) if otm_calls else 0.0
        put_iv_avg  = np.mean([o.iv for o in otm_puts])  if otm_puts  else 0.0

        if call_iv_avg + put_iv_avg == 0:
            raw = 0.0
            detail = "No IV data"
        else:
            # Negative skew (puts > calls IV) = bearish hedging demand → negative signal
            skew = (call_iv_avg - put_iv_avg) / (call_iv_avg + put_iv_avg)
            # Invert: high put skew = bearish = negative
            raw = -skew
            detail = (f"OTM Call IV {call_iv_avg:.1%} vs "
                      f"Put IV {put_iv_avg:.1%} "
                      f"(skew={skew:+.2f})")

        return SignalBreakdown(
            name="IV Structure",
            raw_value=round(raw, 4),
            weight=w,
            contribution=round(raw * w, 4),
            detail=detail,
        )

    # ── Signal 4: Max Pain Position ───────────────────────────────────────────
    def _signal_max_pain_position(
        self, spot: float, gamma_result: Optional[GammaResult]
    ) -> SignalBreakdown:
        """
        If spot is above max pain → bearish pull (reversion to max pain).
        If spot is below max pain → bullish pull.
        """
        w = SIGNAL_WEIGHTS["max_pain_position"]

        if not gamma_result or gamma_result.max_pain == 0:
            return SignalBreakdown(
                name="Max Pain Position", raw_value=0.0, weight=w,
                contribution=0.0, detail="Max pain unavailable"
            )

        mp = gamma_result.max_pain
        deviation = (mp - spot) / spot   # positive = below max pain → bullish pull

        # Clamp
        raw = max(-1.0, min(1.0, deviation * 10))
        detail = f"Max pain {mp:.1f} vs spot {spot:.1f} (Δ{deviation:+.2%})"

        return SignalBreakdown(
            name="Max Pain Position",
            raw_value=round(raw, 4),
            weight=w,
            contribution=round(raw * w, 4),
            detail=detail,
        )

    # ── Signal 5: Strike Cluster Asymmetry ────────────────────────────────────
    def _signal_cluster_asymmetry(
        self, calls: List[OptionRow], puts: List[OptionRow], spot: float
    ) -> SignalBreakdown:
        """
        Compare concentration of OI above vs below spot.
        Heavy call OI above = possible resistance (bearish pin).
        Heavy put OI below = possible support (bullish floor).
        """
        w = SIGNAL_WEIGHTS["cluster_asymmetry"]

        above_calls_oi = sum(o.open_interest for o in calls if o.strike > spot)
        below_puts_oi  = sum(o.open_interest for o in puts  if o.strike < spot)
        total          = above_calls_oi + below_puts_oi

        if total == 0:
            raw = 0.0
            detail = "No OI cluster data"
        else:
            # More put OI below = dealer short put → buy on dip → bullish
            raw = (below_puts_oi - above_calls_oi) / total
            detail = (f"Put OI below {below_puts_oi:,} vs "
                      f"Call OI above {above_calls_oi:,}")

        return SignalBreakdown(
            name="Cluster Asymmetry",
            raw_value=round(raw, 4),
            weight=w,
            contribution=round(raw * w, 4),
            detail=detail,
        )

    # ── Classification ────────────────────────────────────────────────────────
    def _classify(self, composite: float) -> Tuple[str, int]:
        """Map [-1, 1] composite to direction and 0–100 confidence."""
        confidence = int(abs(composite) * 100)
        if composite > 0.10:
            direction = "Bullish"
        elif composite < -0.10:
            direction = "Bearish"
        else:
            direction = "Neutral"
            confidence = max(confidence, 5)   # min 5 even for neutral
        return direction, min(confidence, 100)

    # ── Reasoning ─────────────────────────────────────────────────────────────
    def _build_reasoning(
        self, signals: List[SignalBreakdown], direction: str, spot: float
    ) -> str:
        """Pick the top 2 contributing signals and write a natural sentence."""
        top = sorted(signals, key=lambda s: abs(s.contribution), reverse=True)[:2]
        parts = [f"{s.name}: {s.detail}" for s in top]
        return f"{direction} bias driven by {' | '.join(parts)}"

    # ── ATM IV ────────────────────────────────────────────────────────────────
    def _atm_iv(self, options: List[OptionRow], spot: float) -> float:
        atm = [o for o in options if abs(o.strike - spot) / spot < 0.01 and o.iv > 0]
        return round(float(np.mean([o.iv for o in atm])), 4) if atm else 0.0
