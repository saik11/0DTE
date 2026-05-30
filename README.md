# ⚡ 0DTE Discord Bot — Options Intelligence Dashboard

A Discord bot that pulls real-time 0DTE options data via yfinance and delivers
gamma walls, directional bias, and flow signals directly in Discord.

---

## 🗂️ File Structure

```
odte-bot/
├── bot.py                  # Discord bot + all commands
├── backtest.py             # Bias accuracy backtest CLI
├── requirements.txt
├── Procfile                # Railway: worker: python bot.py
├── runtime.txt             # Railway: python-3.11.0
├── .env.example            # Copy to .env and add your Discord token
└── modules/
    ├── data_fetcher.py     # yfinance data engine + caching
    ├── gamma_engine.py     # Gamma walls, IV walls, Expected Move, Max Pain, Flip Zone, Vanna/Charm exposure
    ├── bias_engine.py      # 6-signal directional bias score (inc. Vanna/Charm drift)
    ├── flow_engine.py      # Unusual volume, IV bursts, sweeps
    └── database.py         # SQLite signal logger
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.11 recommended |
| Discord bot token | From [discord.dev](https://discord.dev) |

No broker account needed — uses yfinance (free, no API key required).

### Install

```bash
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Add your DISCORD_TOKEN to .env
```

### Run locally

```bash
python bot.py
```

### Deploy to Railway

1. Push this repo to GitHub
2. New project on [railway.app](https://railway.app) → Deploy from GitHub
3. Add `DISCORD_TOKEN` in the Variables tab
4. Done — bot goes live automatically

---

## 🤖 Commands

| Command | Description |
|---|---|
| `!ping` | Health check |
| `!bias SPY` | Bullish / Bearish / Neutral + confidence score |
| `!gamma SPY` | 3 call walls (resistance) + 3 put walls (support) |
| `!flow SPY` | Full dashboard — bias + gamma + flow signals |
| `!0dte SPY` | Live 0DTE snapshot, always fresh |
| `!stats SPY 30` | Backtest accuracy over last N days |

Supported tickers: `SPY`, `QQQ`, `SPX`

---

## 📊 How it works

### Data (yfinance)
- Underlying price from `fast_info.last_price`
- Options chain from `ticker.option_chain(date)`
- IV, volume, OI direct from yfinance
- Delta/Gamma/Vanna/Charm computed via Black-Scholes using yfinance IV
- ATR from 5-day daily history

### Bias Engine (6 signals)

| Signal | Weight |
|---|---|
| Call/Put volume imbalance | 25% |
| ATM strike pressure ±1% | 20% |
| Vanna/Charm hedging drift | 15% |
| IV skew structure | 15% |
| Price vs max pain | 15% |
| OI cluster asymmetry | 10% |

### Gamma & Volatility Engine

- **Gamma score**: `GammaScore = OI × IV × Volume × exp(−distance / ATR)`
- **IV Wall score**: `IVWallScore = IV × (OI + Volume)`
- **Expected Move**: `Expected Move = Spot * ATM IV * sqrt(1/252)`
- **Dealer Vanna/Charm exposure**: Cumulative Greek exposures modeled assuming standard short options dealer position.

Produces: Call/Put Gamma Walls, Call/Put IV Skew Walls, Expected Move corridor, Max Pain, Gamma Flip zone, and Net Vanna/Charm exposure.

### Flow Engine
Detects: unusual volume spikes, IV expansion bursts, ATM sweeps, and OI clusters.

---

## ⚙️ Environment Variables

```env
DISCORD_TOKEN=               # Required — from discord.dev
DEFAULT_TICKERS=SPY,QQQ
CACHE_TTL=45                 # seconds between data refreshes
POLL_INTERVAL=60
LOG_LEVEL=INFO
DB_PATH=data/signals.db
AUTO_REPORT_CHANNEL_ID=      # optional — Discord channel ID for daily reports
UNUSUAL_VOLUME_MULTIPLIER=3.0
IV_EXPANSION_THRESHOLD=0.15
```

---

## ⚠️ Disclaimer

For educational purposes only. Not financial advice. Options trading involves substantial risk.
