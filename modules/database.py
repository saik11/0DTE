"""
database.py
───────────
Async SQLite logging for all signals.
Stores bias, gamma walls, and flow signals for backtesting.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Optional

import aiosqlite

logger = logging.getLogger(__name__)
DB_PATH = os.getenv("DB_PATH", "data/signals.db")


CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS bias_signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    spot        REAL,
    direction   TEXT,
    confidence  INTEGER,
    put_call_ratio REAL,
    atm_iv      REAL,
    reasoning   TEXT
);

CREATE TABLE IF NOT EXISTS gamma_walls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    spot        REAL,
    side        TEXT,      -- 'call' or 'put'
    strike      REAL,
    strength    REAL,
    reason      TEXT,
    oi          INTEGER,
    volume      INTEGER
);

CREATE TABLE IF NOT EXISTS flow_signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    spot        REAL,
    signal_type TEXT,
    strike      REAL,
    severity    TEXT,
    detail      TEXT,
    vol_oi_ratio REAL
);

CREATE TABLE IF NOT EXISTS daily_outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    open_price  REAL,
    close_price REAL,
    move_pct    REAL,
    correct_bias INTEGER  -- 1 if bias matched direction, 0 otherwise
);

CREATE INDEX IF NOT EXISTS idx_bias_ticker_ts ON bias_signals(ticker, ts);
CREATE INDEX IF NOT EXISTS idx_flow_ticker_ts ON flow_signals(ticker, ts);
"""


class SignalDB:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(CREATE_TABLES)
            await db.commit()
        logger.info(f"Database initialised at {self.db_path}")

    async def log_bias(self, ticker: str, spot: float, direction: str,
                       confidence: int, pcr: float, atm_iv: float, reasoning: str):
        ts = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO bias_signals
                   (ts, ticker, spot, direction, confidence, put_call_ratio, atm_iv, reasoning)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (ts, ticker, spot, direction, confidence, pcr, atm_iv, reasoning)
            )
            await db.commit()

    async def log_gamma_walls(self, ticker: str, spot: float,
                               call_walls: list, put_walls: list):
        ts = datetime.utcnow().isoformat()
        rows = []
        for w in call_walls:
            rows.append((ts, ticker, spot, "call", w.strike, w.strength, w.reason, w.oi, w.volume))
        for w in put_walls:
            rows.append((ts, ticker, spot, "put",  w.strike, w.strength, w.reason, w.oi, w.volume))

        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """INSERT INTO gamma_walls
                   (ts, ticker, spot, side, strike, strength, reason, oi, volume)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows
            )
            await db.commit()

    async def log_flow_signals(self, ticker: str, spot: float, signals: list):
        ts = datetime.utcnow().isoformat()
        rows = [(ts, ticker, spot, s.type.value, s.strike, s.severity, s.detail, s.vol_oi_ratio)
                for s in signals]
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """INSERT INTO flow_signals
                   (ts, ticker, spot, signal_type, strike, severity, detail, vol_oi_ratio)
                   VALUES (?,?,?,?,?,?,?,?)""",
                rows
            )
            await db.commit()

    async def backtest_accuracy(self, ticker: str, days: int = 30) -> dict:
        """Return bias accuracy stats for backtesting."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT b.direction, d.move_pct, d.correct_bias
                FROM bias_signals b
                JOIN daily_outcomes d ON b.ticker = d.ticker
                  AND DATE(b.ts) = d.date
                WHERE b.ticker = ? AND d.date >= DATE('now', ?)
                ORDER BY d.date DESC
                """,
                (ticker, f"-{days} days")
            )
            rows = await cursor.fetchall()

        if not rows:
            return {"error": "No data"}

        correct = sum(1 for r in rows if r[2] == 1)
        total   = len(rows)
        accuracy = correct / total * 100

        return {
            "ticker": ticker,
            "days": days,
            "total_signals": total,
            "correct": correct,
            "accuracy_pct": round(accuracy, 1),
        }

    async def record_daily_outcome(self, ticker: str, date: str,
                                    open_price: float, close_price: float):
        """Record actual daily move for bias accuracy backtest."""
        move_pct = (close_price - open_price) / open_price * 100

        # Find morning bias
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT direction FROM bias_signals
                   WHERE ticker = ? AND DATE(ts) = ?
                   ORDER BY ts ASC LIMIT 1""",
                (ticker, date)
            )
            row = await cursor.fetchone()
            bias = row[0] if row else None

        if bias is None:
            return

        correct = 0
        if bias == "Bullish" and move_pct > 0:
            correct = 1
        elif bias == "Bearish" and move_pct < 0:
            correct = 1
        elif bias == "Neutral" and abs(move_pct) < 0.3:
            correct = 1

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO daily_outcomes
                   (date, ticker, open_price, close_price, move_pct, correct_bias)
                   VALUES (?,?,?,?,?,?)""",
                (date, ticker, open_price, close_price, round(move_pct, 3), correct)
            )
            await db.commit()


_db: Optional[SignalDB] = None


def get_db() -> SignalDB:
    global _db
    if _db is None:
        _db = SignalDB()
    return _db
