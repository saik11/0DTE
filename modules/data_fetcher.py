"""
data_fetcher.py  (yfinance edition)
────────────────────────────────────
Uses yfinance engine.
All dataclass interfaces (OptionRow, UnderlyingSnapshot, ChainSnapshot)
are identical — gamma_engine, bias_engine, flow_engine, bot.py unchanged.

yfinance notes:
  • Options chains: spy.option_chain(date) → .calls / .puts DataFrames
  • Columns: contractSymbol, strike, bid, ask, volume, openInterest,
             impliedVolatility, lastPrice, inTheMoney
  • Greeks (delta/gamma) are NOT provided by yfinance — we derive
    them from a fast Black-Scholes approximation using the IV yfinance gives us.
  • ATR: from spy.history(period="10d")
  • Real-time price: spy.fast_info.last_price (or .regularMarketPrice)
  • Rate limits: ~2000 req/hour — our 45-sec cache keeps us well within that.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from scipy.stats import norm

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_TTL = int(os.getenv("CACHE_TTL", 45))


# ─── Data containers ─────────────────────────────────────────────────────────
@dataclass
class OptionRow:
    strike: float
    right: str            # "C" or "P"
    expiry: str           # YYYYMMDD
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    iv: float             # decimal e.g. 0.25
    delta: float
    gamma: float
    underlying_price: float
    timestamp: float = field(default_factory=time.time)

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else 0.0


@dataclass
class UnderlyingSnapshot:
    ticker: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChainSnapshot:
    ticker: str
    underlying: UnderlyingSnapshot
    options: List[OptionRow]
    fetched_at: float = field(default_factory=time.time)

    def zero_dte(self) -> List[OptionRow]:
        today = date.today().strftime("%Y%m%d")
        return [o for o in self.options if o.expiry == today]

    def calls(self, zero_dte_only: bool = True) -> List[OptionRow]:
        pool = self.zero_dte() if zero_dte_only else self.options
        return [o for o in pool if o.right == "C"]

    def puts(self, zero_dte_only: bool = True) -> List[OptionRow]:
        pool = self.zero_dte() if zero_dte_only else self.options
        return [o for o in pool if o.right == "P"]


# ─── TTL Cache ────────────────────────────────────────────────────────────────
class TTLCache:
    def __init__(self, ttl: int = CACHE_TTL):
        self._store: Dict[str, Tuple[float, object]] = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[object]:
        if key in self._store:
            ts, val = self._store[key]
            if time.time() - ts < self.ttl:
                return val
        return None

    def set(self, key: str, value: object) -> None:
        self._store[key] = (time.time(), value)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


# ─── Black-Scholes greeks (fast approximation) ────────────────────────────────
def _bs_greeks(
    spot: float, strike: float, iv: float,
    t_years: float, right: str
) -> Tuple[float, float]:
    """
    Returns (delta, gamma) using Black-Scholes.
    t_years: time to expiry in years (e.g. 0.5/252 for ~half a trading day)
    """
    if iv <= 0 or t_years <= 0:
        return (0.5 if right == "C" else -0.5), 0.0
    try:
        d1 = (math.log(spot / strike) + 0.5 * iv ** 2 * t_years) / (iv * math.sqrt(t_years))
        delta = norm.cdf(d1) if right == "C" else norm.cdf(d1) - 1.0
        gamma = norm.pdf(d1) / (spot * iv * math.sqrt(t_years))
        return round(delta, 4), round(gamma, 6)
    except Exception:
        return 0.0, 0.0


def _t_years(expiry_str: str) -> float:
    """
    Fraction of a trading year remaining until expiry.
    expiry_str: YYYYMMDD
    Minimum: 5 minutes expressed as a fraction.
    """
    try:
        exp_date = datetime.strptime(expiry_str, "%Y%m%d").date()
        today    = date.today()
        cal_days = (exp_date - today).days
        t = max(cal_days / 365.0, 5 / (252 * 390))   # floor at 5 min
        return t
    except Exception:
        return 1 / 252


# ─── yfinance fetcher ─────────────────────────────────────────────────────────
class YFinanceFetcher:
    """
    Drop-in replacement for the data fetcher.
    Uses yfinance for underlying price, options chains, and ATR.
    No connection setup required — works anywhere Python runs.
    """

    def __init__(self):
        self._cache   = TTLCache(ttl=CACHE_TTL)
        self._lock    = asyncio.Lock()
        # yfinance Ticker objects are cheap to create; cache them by symbol
        self._tickers: Dict[str, yf.Ticker] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """No-op for compatibility — yfinance needs no connection."""
        logger.info("✅ yfinance fetcher ready (no connection required)")
        return True

    async def disconnect(self):
        pass

    @property
    def is_connected(self) -> bool:
        return True   # yfinance is always "connected"

    def force_refresh(self, ticker: str):
        for suffix in ("0dte", "full"):
            self._cache.invalidate(f"chain:{ticker.upper()}:{suffix}")

    # ── Underlying ────────────────────────────────────────────────────────────
    async def get_underlying(self, ticker: str) -> Optional[UnderlyingSnapshot]:
        ticker = ticker.upper()
        try:
            t   = self._get_yf(ticker)
            fi  = t.fast_info
            price = fi.last_price or fi.regular_market_price
            if not price:
                raise ValueError("No price returned")
            return UnderlyingSnapshot(
                ticker=ticker,
                price=round(float(price), 2),
                bid=round(float(price) - 0.01, 2),   # yfinance doesn't give bid/ask for ETFs reliably
                ask=round(float(price) + 0.01, 2),
                volume=int(fi.three_month_average_volume or 0),
            )
        except Exception as e:
            logger.error(f"get_underlying({ticker}) failed: {e}")
            return None

    # ── Options chain ─────────────────────────────────────────────────────────
    async def get_chain(
        self, ticker: str, zero_dte_only: bool = True
    ) -> Optional[ChainSnapshot]:
        ticker    = ticker.upper()
        cache_key = f"chain:{ticker}:{'0dte' if zero_dte_only else 'full'}"

        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit: {cache_key}")
            return cached

        async with self._lock:
            cached = self._cache.get(cache_key)
            if cached:
                return cached
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._fetch_chain_sync, ticker, zero_dte_only
                )
                if result:
                    self._cache.set(cache_key, result)
                return result
            except Exception as e:
                logger.error(f"get_chain({ticker}) failed: {e}")
                return None

    def _fetch_chain_sync(
        self, ticker: str, zero_dte_only: bool
    ) -> Optional[ChainSnapshot]:
        """Blocking yfinance call — run in executor to keep bot async."""
        t = self._get_yf(ticker)

        # 1. Underlying price
        try:
            fi    = t.fast_info
            price = fi.last_price or fi.regular_market_price
            if not price:
                raise ValueError("No price")
            price = float(price)
        except Exception as e:
            logger.error(f"Price fetch failed for {ticker}: {e}")
            return None

        underlying = UnderlyingSnapshot(
            ticker=ticker, price=round(price, 2),
            bid=round(price - 0.01, 2), ask=round(price + 0.01, 2),
            volume=int(t.fast_info.three_month_average_volume or 0),
        )

        # 2. Available expiry dates
        try:
            all_dates = t.options          # tuple of "YYYY-MM-DD" strings
        except Exception as e:
            logger.error(f"No option dates for {ticker}: {e}")
            return None

        if not all_dates:
            logger.warning(f"Empty option dates for {ticker}")
            return None

        today_iso = date.today().strftime("%Y-%m-%d")

        if zero_dte_only:
            # Use today if available, else nearest next date
            if today_iso in all_dates:
                target_dates = [today_iso]
            else:
                # Closest upcoming date
                future = [d for d in all_dates if d >= today_iso]
                target_dates = [future[0]] if future else [all_dates[0]]
        else:
            # Grab the nearest 2 expirations (covers weekly + next weekly)
            future = [d for d in all_dates if d >= today_iso]
            target_dates = future[:2] if future else list(all_dates[:2])

        # 3. Fetch each expiry and parse rows
        rows: List[OptionRow] = []
        for exp_iso in target_dates:
            exp_yyyymmdd = exp_iso.replace("-", "")
            t_yr         = _t_years(exp_yyyymmdd)

            try:
                chain = t.option_chain(exp_iso)
            except Exception as e:
                logger.warning(f"option_chain({exp_iso}) failed: {e}")
                continue

            for df, right in [(chain.calls, "C"), (chain.puts, "P")]:
                if df is None or df.empty:
                    continue

                # Filter strikes ±5% of spot (±3% for 0DTE)
                pct = 0.03 if zero_dte_only else 0.05
                df  = df[df["strike"].between(price * (1 - pct), price * (1 + pct))].copy()

                for _, row in df.iterrows():
                    strike = float(row["strike"])
                    bid    = float(row.get("bid",       0) or 0)
                    ask    = float(row.get("ask",       0) or 0)
                    last   = float(row.get("lastPrice", 0) or 0)
                    mid    = (bid + ask) / 2 if bid + ask > 0 else last
                    vol    = int(row.get("volume",       0) or 0)
                    oi     = int(row.get("openInterest", 0) or 0)
                    iv_raw = float(row.get("impliedVolatility", 0) or 0)

                    # yfinance gives IV as a decimal already (e.g. 0.25 = 25%)
                    # but sometimes returns values > 5 — clamp to sane range
                    iv = max(0.01, min(iv_raw, 5.0))

                    delta, gamma = _bs_greeks(price, strike, iv, t_yr, right)

                    rows.append(OptionRow(
                        strike=strike, right=right, expiry=exp_yyyymmdd,
                        bid=round(bid, 2), ask=round(ask, 2), mid=round(mid, 2),
                        volume=vol, open_interest=oi,
                        iv=round(iv, 4), delta=delta, gamma=gamma,
                        underlying_price=price,
                    ))

        if not rows:
            logger.warning(f"No option rows parsed for {ticker}")
            return None

        logger.info(f"yfinance: {len(rows)} option rows for {ticker} @ ${price:.2f}")
        return ChainSnapshot(ticker=ticker, underlying=underlying, options=rows)

    # ── ATR ───────────────────────────────────────────────────────────────────
    async def atr(self, ticker: str, days: int = 5) -> float:
        """True Average Range from yfinance daily bars."""
        cache_key = f"atr:{ticker}:{days}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            t    = self._get_yf(ticker.upper())
            hist = await asyncio.get_event_loop().run_in_executor(
                None, lambda: t.history(period=f"{days + 3}d")
            )
            if hist.empty or len(hist) < 2:
                raise ValueError("Insufficient history")

            trs = []
            closes = hist["Close"].values
            highs  = hist["High"].values
            lows   = hist["Low"].values
            for i in range(1, len(hist)):
                tr = max(
                    highs[i]  - lows[i],
                    abs(highs[i]  - closes[i - 1]),
                    abs(lows[i]   - closes[i - 1]),
                )
                trs.append(tr)

            atr_val = round(float(np.mean(trs[-days:])), 2)
            self._cache.set(cache_key, atr_val)
            return atr_val
        except Exception as e:
            logger.warning(f"ATR fallback for {ticker}: {e}")
            # 1% of last known spot
            cached_chain = self._cache.get(f"chain:{ticker}:0dte")
            spot = cached_chain.underlying.price if cached_chain else 500.0
            return round(spot * 0.01, 2)

    # ── Internal ──────────────────────────────────────────────────────────────
    def _get_yf(self, ticker: str) -> yf.Ticker:
        if ticker not in self._tickers:
            self._tickers[ticker] = yf.Ticker(ticker)
        return self._tickers[ticker]


# ─── Singleton ────────────────────────────────────────────────────────────────
_fetcher: Optional[YFinanceFetcher] = None


def get_fetcher() -> YFinanceFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = YFinanceFetcher()
    return _fetcher
