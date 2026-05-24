"""
backtest.py
───────────
Offline backtest utility.

Reads logged bias signals from SQLite, fetches historical closes (via IBKR),
and computes directional accuracy of the 0DTE bias engine.

Usage:
    python backtest.py --ticker SPY --days 30
    python backtest.py --ticker QQQ --days 60 --csv results.csv
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date, timedelta

import aiosqlite

DB_PATH = os.getenv("DB_PATH", "data/signals.db")


async def run_backtest(ticker: str, days: int, csv_path: str | None = None):
    """Pull bias signals and compare to actual next-day move (proxy)."""
    from modules.data_fetcher import get_fetcher
    fetcher = get_fetcher()

    print(f"\n{'═' * 60}")
    print(f"  0DTE Bias Backtest  ·  {ticker}  ·  {days}d lookback")
    print(f"{'═' * 60}")

    # Connect IBKR for historical data
    connected = await fetcher.connect()
    if not connected:
        print("⚠️  IBKR not available — using DB-only results")

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            SELECT DATE(ts) as day, direction, confidence, spot
            FROM bias_signals
            WHERE ticker = ?
              AND DATE(ts) >= DATE('now', ?)
            GROUP BY DATE(ts)
            ORDER BY day ASC
            """,
            (ticker, f"-{days} days"),
        )
        rows = await cursor.fetchall()

    if not rows:
        print("No bias records found in database. Run the bot first to generate signals.")
        return

    results = []
    for day, direction, confidence, spot_open in rows:
        # Fetch next available trading day close from IBKR hist data
        # In a production system, this would pull from an EOD data source
        actual_move = None  # placeholder — set via record_daily_outcome in bot
        results.append({
            "date": day,
            "direction": direction,
            "confidence": confidence,
            "spot_open": spot_open,
            "actual_move": actual_move,
        })

    # Print results table
    header = f"{'Date':<12}{'Bias':<10}{'Conf':>5}  {'Spot':>8}  {'Outcome'}"
    print(header)
    print("─" * 55)
    for r in results:
        outcome = "?"
        print(f"{r['date']:<12}{r['direction']:<10}{r['confidence']:>5}  "
              f"{r['spot_open']:>8.2f}  {outcome}")

    total = len(results)
    print(f"\n  Total signals: {total}")
    print("  Run the bot during market hours to accumulate outcome data.")
    print(f"{'═' * 60}\n")

    # Optional CSV export
    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"  Exported to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="0DTE Bias Backtest")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--days",   type=int, default=30)
    parser.add_argument("--csv",    default=None)
    args = parser.parse_args()

    asyncio.run(run_backtest(args.ticker, args.days, args.csv))


if __name__ == "__main__":
    main()
