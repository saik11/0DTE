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
class IVWall:
    strike: float
    iv: float
    oi: int
    volume: int
    score: float


@dataclass
class ExpectedMoveRange:
    expected_move: float
    upper_bound: float
    lower_bound: float
    closest_call_strike: float
    closest_put_strike: float


@dataclass
class GammaResult:
    ticker: str
    spot: float
    call_walls: List[GammaLevel]    # top 3 resistance
    put_walls: List[GammaLevel]     # top 3 support
    gamma_flip: Optional[float]     # strike where net gamma flips sign
    max_pain: float
    net_gamma_profile: Dict[float, float]  # strike → net gamma exposure
    atr: float
    iv_call_wall: Optional[IVWall] = None
    iv_put_wall: Optional[IVWall] = None
    expected_move: Optional[ExpectedMoveRange] = None


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
        net_profile = self._net_gamma_profile(df)

        iv_call_wall, iv_put_wall = self._compute_iv_walls(df, spot)
        expected_move = self._compute_expected_move_bounds(df, spot)

        return GammaResult(
            ticker=snapshot.ticker,
            spot=spot,
            call_walls=call_walls,
            put_walls=put_walls,
            gamma_flip=flip_zone,
            max_pain=max_pain,
            net_gamma_profile=net_profile,
            atr=self.atr,
            iv_call_wall=iv_call_wall,
            iv_put_wall=iv_put_wall,
            expected_move=expected_move,
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



    # ── IV Wall detection (Option 1) ──────────────────────────────────────────
    def _compute_iv_walls(
        self, df: pd.DataFrame, spot: float
    ) -> Tuple[Optional[IVWall], Optional[IVWall]]:
        """
        Calculates IV Walls based on volatility-weighted exposure.
        Score = iv * (oi + volume)
        Filter out strikes with low open interest to avoid illiquid options.
        """
        if df.empty:
            return None, None

        # Filter out rows with low OI (e.g. < 50) to ignore noise
        df_filtered = df[df["oi"] >= 50].copy()
        if df_filtered.empty:
            df_filtered = df.copy()

        df_filtered["iv_score"] = df_filtered["iv"] * (df_filtered["oi"] + df_filtered["volume"])

        calls = df_filtered[(df_filtered["right"] == "C") & (df_filtered["strike"] >= spot)]
        puts = df_filtered[(df_filtered["right"] == "P") & (df_filtered["strike"] <= spot)]

        call_wall = None
        put_wall = None

        if not calls.empty:
            top_call = calls.sort_values("iv_score", ascending=False).iloc[0]
            call_wall = IVWall(
                strike=float(top_call["strike"]),
                iv=float(top_call["iv"]),
                oi=int(top_call["oi"]),
                volume=int(top_call["volume"]),
                score=float(top_call["iv_score"]),
            )

        if not puts.empty:
            top_put = puts.sort_values("iv_score", ascending=False).iloc[0]
            put_wall = IVWall(
                strike=float(top_put["strike"]),
                iv=float(top_put["iv"]),
                oi=int(top_put["oi"]),
                volume=int(top_put["volume"]),
                score=float(top_put["iv_score"]),
            )

        return call_wall, put_wall

    # ── Expected Move Range (Option 2) ────────────────────────────────────────
    def _compute_expected_move_bounds(
        self, df: pd.DataFrame, spot: float
    ) -> Optional[ExpectedMoveRange]:
        """
        Calculates Daily Expected Move using ATM IV, and finds the closest strikes with highest OI.
        """
        if df.empty:
            return None

        # ATM IV is average IV within 1% of spot
        atm_options = df[df["strike"].between(spot * 0.99, spot * 1.01)]
        if atm_options.empty:
            atm_options = df.copy()

        atm_iv = atm_options["iv"].mean()
        if not atm_iv or atm_iv <= 0:
            atm_iv = 0.15

        # Expected Move (daily) = Spot * ATM_IV * sqrt(1/252)
        expected_move = spot * atm_iv * math.sqrt(1 / 252)
        upper_bound = spot + expected_move
        lower_bound = spot - expected_move

        # Call Expected Move Wall (strike >= spot closest to upper_bound with highest OI among 5 closest)
        calls = df[(df["right"] == "C") & (df["strike"] >= spot)]
        closest_call_strike = spot
        if not calls.empty:
            calls = calls.copy()
            calls["proximity"] = (calls["strike"] - upper_bound).abs()
            top_proximity_calls = calls.sort_values("proximity").head(5)
            if not top_proximity_calls.empty:
                closest_call_strike = float(top_proximity_calls.sort_values("oi", ascending=False).iloc[0]["strike"])

        # Put Expected Move Wall (strike <= spot closest to lower_bound with highest OI among 5 closest)
        puts = df[(df["right"] == "P") & (df["strike"] <= spot)]
        closest_put_strike = spot
        if not puts.empty:
            puts = puts.copy()
            puts["proximity"] = (puts["strike"] - lower_bound).abs()
            top_proximity_puts = puts.sort_values("proximity").head(5)
            if not top_proximity_puts.empty:
                closest_put_strike = float(top_proximity_puts.sort_values("oi", ascending=False).iloc[0]["strike"])

        return ExpectedMoveRange(
            expected_move=round(expected_move, 2),
            upper_bound=round(upper_bound, 2),
            lower_bound=round(lower_bound, 2),
            closest_call_strike=closest_call_strike,
            closest_put_strike=closest_put_strike,
        )

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
