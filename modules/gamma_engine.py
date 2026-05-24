"""
gamma_engine.py
───────────────
Synthetic gamma exposure model for 0DTE options.

Core formula:
    GammaScore(strike) = OI × IV × Volume × exp(-|strike - spot| / ATR)

Strikes are then clustered into zones and ranked to identify:
  • Call Gamma Walls (resistance levels)
  • Put Gamma Walls (support levels)
  • Gamma Flip Zone (where net gamma changes sign)
  • Max Pain estimate
  • Pin probability zones
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.data_fetcher import ChainSnapshot, OptionRow

logger = logging.getLogger(__name__)

# ─── Output containers ────────────────────────────────────────────────────────
@dataclass
class GammaLevel:
    strike: float
    strength: float        # 0–100 normalised
    raw_score: float
    reason: str            # human-readable driver
    oi: int
    volume: int
    iv: float


@dataclass
class GammaResult:
    ticker: str
    spot: float
    call_walls: List[GammaLevel]    # top 3 resistance
    put_walls: List[GammaLevel]     # top 3 support
    gamma_flip: Optional[float]     # strike where net gamma flips sign
    max_pain: float
    pin_zones: List[Tuple[float, float]]   # (low, high) ranges
    net_gamma_profile: Dict[float, float]  # strike → net gamma exposure
    atr: float


# ─── Engine ───────────────────────────────────────────────────────────────────
class GammaEngine:
    """
    Computes all gamma-related outputs from a ChainSnapshot.
    """

    def __init__(self, atr: float = 5.0):
        self.atr = atr

    # ── Main entry point ──────────────────────────────────────────────────────
    def compute(self, snapshot: ChainSnapshot) -> GammaResult:
        options = snapshot.zero_dte()
        spot    = snapshot.underlying.price

        if not options:
            logger.warning(f"No 0DTE options found for {snapshot.ticker}")
            options = snapshot.options   # fallback to nearest expiry

        df = self._to_df(options, spot)

        call_walls  = self._compute_walls(df[df["right"] == "C"], spot, side="call")
        put_walls   = self._compute_walls(df[df["right"] == "P"], spot, side="put")
        max_pain    = self._max_pain(df)
        flip_zone   = self._gamma_flip(df, spot)
        pin_zones   = self._pin_zones(df, spot)
        net_profile = self._net_gamma_profile(df)

        return GammaResult(
            ticker=snapshot.ticker,
            spot=spot,
            call_walls=call_walls,
            put_walls=put_walls,
            gamma_flip=flip_zone,
            max_pain=max_pain,
            pin_zones=pin_zones,
            net_gamma_profile=net_profile,
            atr=self.atr,
        )

    # ── DataFrame builder ─────────────────────────────────────────────────────
    def _to_df(self, options: List[OptionRow], spot: float) -> pd.DataFrame:
        rows = []
        for o in options:
            distance = abs(o.strike - spot)
            # Decay weight — exponential falloff with ATR as scale
            decay = math.exp(-distance / max(self.atr, 1.0))

            # Core gamma score (synthetic proxy)
            oi     = max(o.open_interest, 1)
            vol    = max(o.volume, 1)
            iv     = max(o.iv, 0.01)
            gamma_score = oi * iv * vol * decay

            rows.append({
                "strike":      o.strike,
                "right":       o.right,
                "oi":          o.open_interest,
                "volume":      o.volume,
                "iv":          o.iv,
                "mid":         o.mid,
                "gamma":       o.gamma,
                "distance":    distance,
                "decay":       decay,
                "gamma_score": gamma_score,
            })

        return pd.DataFrame(rows)

    # ── Wall detection ────────────────────────────────────────────────────────
    def _compute_walls(
        self, df: pd.DataFrame, spot: float, side: str
    ) -> List[GammaLevel]:
        """
        Cluster strikes and rank by gamma score.
        Returns top 3 levels with reason tags.
        """
        if df.empty:
            return []

        # Aggregate by strike
        agg = df.groupby("strike").agg(
            total_score=("gamma_score", "sum"),
            total_oi   =("oi",          "sum"),
            total_vol  =("volume",      "sum"),
            avg_iv     =("iv",          "mean"),
        ).reset_index()

        # Only consider relevant side
        if side == "call":
            # Call walls are resistance — above or near spot
            agg = agg[agg["strike"] >= spot * 0.99]
        else:
            # Put walls are support — below or near spot
            agg = agg[agg["strike"] <= spot * 1.01]

        if agg.empty:
            return []

        # Normalise score 0–100
        max_score = agg["total_score"].max()
        agg["strength"] = (agg["total_score"] / max_score * 100).clip(0, 100)
        agg = agg.sort_values("strength", ascending=False).head(3)

        levels = []
        for _, row in agg.iterrows():
            reason = self._reason_tag(row)
            levels.append(GammaLevel(
                strike   = round(row["strike"], 2),
                strength = round(row["strength"], 1),
                raw_score= row["total_score"],
                reason   = reason,
                oi       = int(row["total_oi"]),
                volume   = int(row["total_vol"]),
                iv       = round(row["avg_iv"], 4),
            ))
        return levels

    def _reason_tag(self, row: pd.Series) -> str:
        """Generate a human-readable reason string from dominant signal."""
        tags = []
        if row["total_oi"] > 10_000:
            tags.append("OI cluster")
        if row["avg_iv"] > 0.25:
            tags.append("IV spike")
        if row["total_vol"] > 5_000:
            tags.append("volume surge")
        if row["strength"] > 80:
            tags.append("dominant level")
        return " + ".join(tags) if tags else "moderate concentration"

    # ── Max Pain ──────────────────────────────────────────────────────────────
    def _max_pain(self, df: pd.DataFrame) -> float:
        """
        Classic max pain: strike where total option value at expiry
        (for all other strikes) is minimised.
        """
        if df.empty:
            return 0.0

        strikes = sorted(df["strike"].unique())
        if not strikes:
            return 0.0

        pain: Dict[float, float] = {}
        for expiry_price in strikes:
            total = 0.0
            for _, row in df.iterrows():
                if row["right"] == "C":
                    total += max(expiry_price - row["strike"], 0) * row["oi"]
                else:
                    total += max(row["strike"] - expiry_price, 0) * row["oi"]
            pain[expiry_price] = total

        return min(pain, key=pain.get)

    # ── Gamma Flip ────────────────────────────────────────────────────────────
    def _gamma_flip(self, df: pd.DataFrame, spot: float) -> Optional[float]:
        """
        Find the strike where net dealer gamma flips from positive to negative.
        Dealers are long gamma above strike (short call → hedge by buying),
        short gamma below strike (short put → hedge by selling).

        Net gamma ≈ Call_GEX - Put_GEX per strike.
        Flip occurs where cumulative net GEX crosses zero.
        """
        net = self._net_gamma_profile(df)
        if not net:
            return None

        strikes = sorted(net.keys())
        cumulative = 0.0
        flip = None
        for s in strikes:
            prev = cumulative
            cumulative += net[s]
            if prev < 0 <= cumulative or prev >= 0 > cumulative:
                flip = s
                break

        return flip

    # ── Net Gamma Profile ─────────────────────────────────────────────────────
    def _net_gamma_profile(self, df: pd.DataFrame) -> Dict[float, float]:
        """Strike-level net gamma exposure (calls positive, puts negative)."""
        profile: Dict[float, float] = {}
        for _, row in df.iterrows():
            s = row["strike"]
            g = row["gamma_score"]
            if row["right"] == "C":
                profile[s] = profile.get(s, 0) + g
            else:
                profile[s] = profile.get(s, 0) - g
        return dict(sorted(profile.items()))

    # ── Pin Zones ─────────────────────────────────────────────────────────────
    def _pin_zones(
        self, df: pd.DataFrame, spot: float, top_n: int = 2
    ) -> List[Tuple[float, float]]:
        """
        Identify price magnet zones where both call and put OI clusters
        overlap — high pinning probability near expiry.
        """
        if df.empty:
            return []

        # Strikes within 1 ATR of spot
        nearby = df[df["strike"].between(spot - self.atr, spot + self.atr)]
        if nearby.empty:
            return []

        # Combined OI heatmap
        oi_map = nearby.groupby("strike")["oi"].sum()
        if oi_map.empty:
            return []

        top_strikes = oi_map.nlargest(top_n).index.tolist()
        half_width = max(self.atr * 0.1, 0.5)

        zones = [(round(s - half_width, 2), round(s + half_width, 2))
                 for s in sorted(top_strikes)]
        return zones

    # ── Liquidity Heatmap ─────────────────────────────────────────────────────
    def liquidity_heatmap(self, snapshot: ChainSnapshot) -> Dict[float, float]:
        """
        Strike clustering strength — combined OI + volume normalised.
        Used for detecting crowded strikes.
        """
        options = snapshot.zero_dte() or snapshot.options
        df = self._to_df(options, snapshot.underlying.price)
        if df.empty:
            return {}
        agg = df.groupby("strike").agg(
            score=("gamma_score", "sum")
        )
        max_s = agg["score"].max()
        return {float(k): round(v / max_s * 100, 1) for k, v in agg["score"].items()}
