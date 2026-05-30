"""
bot.py
──────
Production Discord bot for 0DTE options intelligence.

Commands:
  !bias  TICKER  — Daily directional bias + confidence
  !gamma TICKER  — Gamma walls (3 call + 3 put levels)
  !flow  TICKER  — Full dashboard (bias + gamma + flow signals)
  !0dte  TICKER  — Pure 0DTE snapshot with real-time gamma + flow
  !stats TICKER  — Backtest accuracy
  !ping          — Health check
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, time as dtime
from typing import Dict, Optional

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from modules.data_fetcher import YFinanceFetcher, get_fetcher
from modules.gamma_engine import GammaEngine, GammaResult
from modules.bias_engine import BiasEngine, BiasResult
from modules.flow_engine import FlowEngine, FlowResult
from modules.database import SignalDB, get_db

load_dotenv()

# ─── Logging ─────────────────────────────────────────────────────────────────
log_dir = os.getenv("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "bot.log")),
    ],
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
DISCORD_TOKEN        = os.getenv("DISCORD_TOKEN", "")
PREFIX               = os.getenv("COMMAND_PREFIX", "!")
POLL_INTERVAL        = int(os.getenv("POLL_INTERVAL", 60))
AUTO_CHANNEL_ID      = int(os.getenv("AUTO_REPORT_CHANNEL_ID", 0) or 0)
DEFAULT_TICKERS      = [t.strip().upper() for t in os.getenv("DEFAULT_TICKERS", "SPY,QQQ").split(",")]

# ─── Bot setup ────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# ─── Services ─────────────────────────────────────────────────────────────────
fetcher       = get_fetcher()
gamma_engine  = GammaEngine()
bias_engine   = BiasEngine(gamma_engine)
flow_engine   = FlowEngine()
db            = get_db()

# Last gamma result cache for cross-engine use
_gamma_cache: Dict[str, GammaResult] = {}


# ─── Colour constants ─────────────────────────────────────────────────────────
C_BULL   = 0x00FF87   # green
C_BEAR   = 0xFF4444   # red
C_NEUT   = 0xFFCC00   # yellow
C_GAMMA  = 0x7B68EE   # purple
C_FLOW   = 0x00BFFF   # cyan
C_ERROR  = 0x808080


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def direction_emoji(direction: str) -> str:
    return {"Bullish": "📈", "Bearish": "📉", "Neutral": "⚖️"}.get(direction, "❓")


def confidence_bar(score: int, width: int = 10) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def fmt_strike(s: float) -> str:
    return f"{s:.0f}" if s == int(s) else f"{s:.2f}"


async def _fetch_all(ticker: str, zero_dte: bool = True):
    """Fetch chain + compute all engines. Returns (bias, gamma, flow) or raises."""
    ticker = ticker.upper().strip()
    if ticker not in ("SPY", "QQQ", "SPX"):
        raise ValueError(f"Unsupported ticker `{ticker}`. Use SPY, QQQ, or SPX.")

    snapshot = await fetcher.get_chain(ticker, zero_dte_only=zero_dte)
    if not snapshot:
        raise RuntimeError(f"Could not fetch data for {ticker}. The data source may be offline.")

    atr = await fetcher.atr(ticker)
    gamma_engine.atr = atr

    gamma_result = gamma_engine.compute(snapshot)
    _gamma_cache[ticker] = gamma_result

    bias_result  = bias_engine.compute(snapshot, gamma_result)
    flow_result  = flow_engine.compute(snapshot)

    return snapshot, bias_result, gamma_result, flow_result


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBED BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_bias_embed(bias: BiasResult) -> discord.Embed:
    colour = {"Bullish": C_BULL, "Bearish": C_BEAR}.get(bias.direction, C_NEUT)
    emoji  = direction_emoji(bias.direction)

    embed = discord.Embed(
        title=f"{bias.ticker} · 0DTE Bias  {emoji}",
        color=colour,
        timestamp=datetime.utcnow(),
    )
    embed.add_field(
        name="Direction",
        value=f"**{bias.direction}** {emoji}",
        inline=True,
    )
    embed.add_field(
        name="Confidence",
        value=f"`{confidence_bar(bias.confidence)}` {bias.confidence}/100",
        inline=True,
    )
    embed.add_field(
        name="ATM IV",
        value=f"{bias.atm_iv:.1%}",
        inline=True,
    )
    embed.add_field(
        name="Volume (Calls / Puts)",
        value=f"{bias.call_volume:,} / {bias.put_volume:,}  · PCR {bias.put_call_ratio:.2f}",
        inline=False,
    )
    embed.add_field(
        name="📋 Reasoning",
        value=bias.reasoning[:1000],
        inline=False,
    )

    # Signal breakdown
    breakdown = "\n".join(
        f"`{s.name:<20}` {s.raw_value:+.2f} (×{s.weight}) = {s.contribution:+.3f}"
        for s in bias.signals
    )
    embed.add_field(name="Signal Breakdown", value=f"```{breakdown}```", inline=False)
    embed.set_footer(text=f"Spot ${bias.spot:.2f}")
    return embed


def build_gamma_embed(gamma: GammaResult) -> discord.Embed:
    embed = discord.Embed(
        title=f"{gamma.ticker} · Gamma Walls",
        color=C_GAMMA,
        timestamp=datetime.utcnow(),
    )

    # Call walls
    if gamma.call_walls:
        lines = []
        for w in gamma.call_walls:
            bar = confidence_bar(int(w.strength), 8)
            lines.append(f"**{fmt_strike(w.strike)}**  `{bar}` {w.strength:.0f}/100  ·  {w.reason}")
        embed.add_field(
            name="🟢 Call Walls (Resistance)",
            value="\n".join(lines),
            inline=False,
        )

    # Put walls
    if gamma.put_walls:
        lines = []
        for w in gamma.put_walls:
            bar = confidence_bar(int(w.strength), 8)
            lines.append(f"**{fmt_strike(w.strike)}**  `{bar}` {w.strength:.0f}/100  ·  {w.reason}")
        embed.add_field(
            name="🔴 Put Walls (Support)",
            value="\n".join(lines),
            inline=False,
        )

    # Summary stats
    embed.add_field(name="Max Pain",    value=f"${fmt_strike(gamma.max_pain)}", inline=True)
    embed.add_field(name="Gamma Flip",
                    value=f"${fmt_strike(gamma.gamma_flip)}" if gamma.gamma_flip else "N/A",
                    inline=True)
    embed.add_field(name="ATR",         value=f"${gamma.atr:.2f}", inline=True)



    # IV Walls and Expected Range
    if gamma.expected_move:
        em = gamma.expected_move
        embed.add_field(
            name="📐 1-Day Expected Move Range",
            value=f"Expected Move: **±${em.expected_move:.2f}**\n"
                  f"Range: **${em.lower_bound:.2f}** – **${em.upper_bound:.2f}**\n"
                  f"Expected Move Walls: Put **${fmt_strike(em.closest_put_strike)}** | Call **${fmt_strike(em.closest_call_strike)}**",
            inline=False
        )

    if gamma.iv_call_wall or gamma.iv_put_wall or gamma.dealer_vanna != 0:
        iv_lines = []
        if gamma.iv_call_wall:
            cw = gamma.iv_call_wall
            iv_lines.append(f"🟢 **Call IV Wall**: **${fmt_strike(cw.strike)}** (IV: {cw.iv:.1%}, OI: {cw.oi:,})")
        if gamma.iv_put_wall:
            pw = gamma.iv_put_wall
            iv_lines.append(f"🔴 **Put IV Wall**: **${fmt_strike(pw.strike)}** (IV: {pw.iv:.1%}, OI: {pw.oi:,})")
        
        vanna_fmt = f"{gamma.dealer_vanna/1e3:+.1f}k" if abs(gamma.dealer_vanna) < 1e6 else f"{gamma.dealer_vanna/1e6:+.1f}M"
        charm_fmt = f"{gamma.dealer_charm/1e3:+.1f}k" if abs(gamma.dealer_charm) < 1e6 else f"{gamma.dealer_charm/1e6:+.1f}M"
        iv_lines.append(f"🛡️ **Dealer Vanna Exposure**: **{vanna_fmt}** shares / 1% vol change")
        iv_lines.append(f"⏳ **Dealer Charm Exposure**: **{charm_fmt}** shares decay / day")
        
        embed.add_field(
            name="⚡ Volatility Skew & Greeks Exposure",
            value="\n".join(iv_lines),
            inline=False
        )

    embed.set_footer(text=f"Spot ${gamma.spot:.2f}")
    return embed


def build_flow_embed(flow: FlowResult) -> discord.Embed:
    embed = discord.Embed(
        title=f"{flow.ticker} · Flow Intelligence",
        color=C_FLOW,
        timestamp=datetime.utcnow(),
    )

    # IV Regime
    regime_emoji = {"Expanding": "🔥", "Contracting": "❄️", "Stable": "➡️"}.get(flow.iv_regime, "")
    embed.add_field(name="IV Regime",   value=f"{regime_emoji} {flow.iv_regime}", inline=True)
    embed.add_field(name="ATM IV",      value=f"{flow.avg_atm_iv:.1%}", inline=True)
    embed.add_field(name="Trend Bias",
                    value=f"{flow.trend_bias} ({flow.trend_probability:.0%})", inline=True)

    # Flow signals
    if flow.signals:
        lines = []
        for sig in flow.signals[:8]:
            lines.append(f"{sig.severity}  **{sig.strike:.0f}** — {sig.detail[:80]}")
        embed.add_field(
            name="⚡ Active Signals",
            value="\n".join(lines) or "None detected",
            inline=False,
        )
    else:
        embed.add_field(name="⚡ Active Signals", value="No unusual activity detected", inline=False)

    # Top strikes
    if flow.top_call_strike:
        embed.add_field(name="Hottest Call Strike", value=f"${flow.top_call_strike:.0f}", inline=True)
    if flow.top_put_strike:
        embed.add_field(name="Hottest Put Strike",  value=f"${flow.top_put_strike:.0f}",  inline=True)

    # Liquidity heatmap (top 5)
    hm = flow.liquidity_heatmap
    if hm:
        top5 = sorted(hm.items(), key=lambda x: -x[1])[:5]
        hm_str = "  ".join(f"**{s:.0f}** `{v:.0f}`" for s, v in top5)
        embed.add_field(name="🌡️ Liquidity Heatmap (top strikes)", value=hm_str, inline=False)

    embed.set_footer(text=f"Spot ${flow.spot:.2f}")
    return embed


def build_full_embed(
    ticker: str, bias: BiasResult, gamma: GammaResult, flow: FlowResult
) -> discord.Embed:
    """Combine all engines into one rich dashboard embed."""
    colour = {"Bullish": C_BULL, "Bearish": C_BEAR}.get(bias.direction, C_NEUT)
    emoji  = direction_emoji(bias.direction)

    embed = discord.Embed(
        title=f"📊 {ticker} 0DTE Full Dashboard",
        description=(
            f"Spot **${bias.spot:.2f}**  ·  "
            f"Bias **{bias.direction}** {emoji} {bias.confidence}/100  ·  "
            f"IV **{bias.atm_iv:.1%}** ({flow.iv_regime})"
        ),
        color=colour,
        timestamp=datetime.utcnow(),
    )

    # Bias
    embed.add_field(name="📈 Bias Reasoning", value=bias.reasoning[:400], inline=False)

    # Gamma walls inline
    if gamma.call_walls:
        cw = "\n".join(f"`{fmt_strike(w.strike)}` {w.strength:.0f}/100 — {w.reason}" for w in gamma.call_walls)
        embed.add_field(name="🟢 Call Walls", value=cw, inline=True)
    if gamma.put_walls:
        pw = "\n".join(f"`{fmt_strike(w.strike)}` {w.strength:.0f}/100 — {w.reason}" for w in gamma.put_walls)
        embed.add_field(name="🔴 Put Walls", value=pw, inline=True)

    # Key levels
    levels = []
    if gamma.max_pain:   levels.append(f"Max Pain: **${fmt_strike(gamma.max_pain)}**")
    if gamma.gamma_flip: levels.append(f"Gamma Flip: **${fmt_strike(gamma.gamma_flip)}**")

    if levels:
        embed.add_field(name="🎯 Key Levels", value="  ·  ".join(levels), inline=False)

    # Flow signals (top 5)
    if flow.signals:
        sigs = "\n".join(f"{s.severity} `{s.strike:.0f}` {s.detail[:70]}" for s in flow.signals[:5])
        embed.add_field(name="⚡ Flow Signals", value=sigs, inline=False)

    embed.add_field(name="PCR",          value=f"{bias.put_call_ratio:.2f}", inline=True)
    embed.add_field(name="Trend",        value=f"{flow.trend_bias} {flow.trend_probability:.0%}", inline=True)
    embed.add_field(name="ATR",          value=f"${gamma.atr:.2f}", inline=True)

    # Volatility / Expected Move Boundaries
    if gamma.expected_move or gamma.iv_call_wall or gamma.iv_put_wall or gamma.dealer_vanna != 0:
        v_lines = []
        if gamma.expected_move:
            em = gamma.expected_move
            v_lines.append(f"📐 **Expected Move**: ±${em.expected_move:.2f} (${em.lower_bound:.2f} – ${em.upper_bound:.2f})")
            v_lines.append(f"   ↳ Range Boundary Walls: Put **${fmt_strike(em.closest_put_strike)}** | Call **${fmt_strike(em.closest_call_strike)}**")
        if gamma.iv_call_wall or gamma.iv_put_wall:
            iv_w = []
            if gamma.iv_call_wall:
                iv_w.append(f"🟢 Call **${fmt_strike(gamma.iv_call_wall.strike)}**")
            if gamma.iv_put_wall:
                iv_w.append(f"🔴 Put **${fmt_strike(gamma.iv_put_wall.strike)}**")
            v_lines.append(f"⚡ **IV Skew Boundaries**: " + " | ".join(iv_w))
        
        vanna_fmt = f"{gamma.dealer_vanna/1e3:+.1f}k" if abs(gamma.dealer_vanna) < 1e6 else f"{gamma.dealer_vanna/1e6:+.1f}M"
        charm_fmt = f"{gamma.dealer_charm/1e3:+.1f}k" if abs(gamma.dealer_charm) < 1e6 else f"{gamma.dealer_charm/1e6:+.1f}M"
        v_lines.append(f"🛡️ **Dealer Vanna**: **{vanna_fmt}** shares / 1% vol change")
        v_lines.append(f"⏳ **Dealer Charm**: **{charm_fmt}** shares decay / day")
        
        embed.add_field(name="📐 Volatility & Expected Range", value="\n".join(v_lines), inline=False)

    embed.set_footer(text="0DTE Dashboard")
    return embed


def _error_embed(title: str, description: str) -> discord.Embed:
    return discord.Embed(title=f"❌ {title}", description=description, color=C_ERROR)


# ═══════════════════════════════════════════════════════════════════════════════
#  COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@bot.command(name="bias")
async def cmd_bias(ctx: commands.Context, ticker: str = "SPY"):
    """!bias [TICKER] — Daily directional bias"""
    async with ctx.typing():
        try:
            _, bias, gamma, _ = await _fetch_all(ticker)
            embed = build_bias_embed(bias)
            await ctx.send(embed=embed)
            # Log to DB (non-blocking)
            asyncio.create_task(
                db.log_bias(bias.ticker, bias.spot, bias.direction,
                            bias.confidence, bias.put_call_ratio,
                            bias.atm_iv, bias.reasoning)
            )
        except Exception as e:
            await ctx.send(embed=_error_embed("Bias Error", str(e)))
            logger.exception(f"!bias error for {ticker}")


@bot.command(name="gamma")
async def cmd_gamma(ctx: commands.Context, ticker: str = "SPY"):
    """!gamma [TICKER] — Gamma walls"""
    async with ctx.typing():
        try:
            _, _, gamma, _ = await _fetch_all(ticker)
            embed = build_gamma_embed(gamma)
            await ctx.send(embed=embed)
            asyncio.create_task(
                db.log_gamma_walls(gamma.ticker, gamma.spot,
                                   gamma.call_walls, gamma.put_walls)
            )
        except Exception as e:
            await ctx.send(embed=_error_embed("Gamma Error", str(e)))
            logger.exception(f"!gamma error for {ticker}")


@bot.command(name="flow")
async def cmd_flow(ctx: commands.Context, ticker: str = "SPY"):
    """!flow [TICKER] — Full options flow dashboard"""
    async with ctx.typing():
        try:
            snapshot, bias, gamma, flow = await _fetch_all(ticker)
            embed = build_full_embed(ticker.upper(), bias, gamma, flow)
            await ctx.send(embed=embed)
            asyncio.create_task(
                db.log_flow_signals(flow.ticker, flow.spot, flow.signals)
            )
        except Exception as e:
            await ctx.send(embed=_error_embed("Flow Error", str(e)))
            logger.exception(f"!flow error for {ticker}")


@bot.command(name="0dte")
async def cmd_0dte(ctx: commands.Context, ticker: str = "SPY"):
    """!0dte [TICKER] — Pure 0DTE real-time snapshot"""
    async with ctx.typing():
        try:
            fetcher.force_refresh(ticker.upper())   # always fresh
            snapshot, bias, gamma, flow = await _fetch_all(ticker, zero_dte=True)

            # Build specialised 0DTE embed
            colour = {"Bullish": C_BULL, "Bearish": C_BEAR}.get(bias.direction, C_NEUT)
            embed = discord.Embed(
                title=f"⚡ {ticker.upper()} 0DTE Live Snapshot",
                description=f"**{len(snapshot.zero_dte())}** same-day contracts loaded",
                color=colour,
                timestamp=datetime.utcnow(),
            )
            embed.add_field(
                name=f"Bias {direction_emoji(bias.direction)}",
                value=f"**{bias.direction}** · {bias.confidence}/100 confidence",
                inline=True,
            )
            embed.add_field(
                name="ATM IV",
                value=f"{bias.atm_iv:.1%} ({flow.iv_regime})",
                inline=True,
            )
            embed.add_field(name="PCR", value=f"{bias.put_call_ratio:.2f}", inline=True)

            if gamma.call_walls:
                top_c = gamma.call_walls[0]
                embed.add_field(
                    name="🟢 #1 Call Wall",
                    value=f"**${fmt_strike(top_c.strike)}** · strength {top_c.strength:.0f}/100\n_{top_c.reason}_",
                    inline=True,
                )
            if gamma.put_walls:
                top_p = gamma.put_walls[0]
                embed.add_field(
                    name="🔴 #1 Put Wall",
                    value=f"**${fmt_strike(top_p.strike)}** · strength {top_p.strength:.0f}/100\n_{top_p.reason}_",
                    inline=True,
                )

            if gamma.gamma_flip:
                embed.add_field(
                    name="🔀 Gamma Flip",
                    value=f"${fmt_strike(gamma.gamma_flip)}",
                    inline=True,
                )

            high_sev = [s for s in flow.signals if "High" in s.severity][:4]
            if high_sev:
                lines = [f"{s.severity} `{s.strike:.0f}` — {s.detail[:60]}" for s in high_sev]
                embed.add_field(name="🚨 High-Sev Signals", value="\n".join(lines), inline=False)



            if gamma.expected_move:
                em = gamma.expected_move
                embed.add_field(
                    name="📐 Expected Move Range",
                    value=f"**±${em.expected_move:.2f}** (${em.lower_bound:.2f} – ${em.upper_bound:.2f})\n"
                          f"Range Walls: Put **${fmt_strike(em.closest_put_strike)}** | Call **${fmt_strike(em.closest_call_strike)}**",
                    inline=False
                )

            if gamma.iv_call_wall or gamma.iv_put_wall or gamma.dealer_vanna != 0:
                iv_w = []
                if gamma.iv_call_wall:
                    iv_w.append(f"🟢 Call **${fmt_strike(gamma.iv_call_wall.strike)}** (IV {gamma.iv_call_wall.iv:.1%})")
                if gamma.iv_put_wall:
                    iv_w.append(f"🔴 Put **${fmt_strike(gamma.iv_put_wall.strike)}** (IV {gamma.iv_put_wall.iv:.1%})")
                
                vanna_fmt = f"{gamma.dealer_vanna/1e3:+.1f}k" if abs(gamma.dealer_vanna) < 1e6 else f"{gamma.dealer_vanna/1e6:+.1f}M"
                charm_fmt = f"{gamma.dealer_charm/1e3:+.1f}k" if abs(gamma.dealer_charm) < 1e6 else f"{gamma.dealer_charm/1e6:+.1f}M"
                
                embed.add_field(
                    name="⚡ IV Skew & Greeks Exposure",
                    value="  ·  ".join(iv_w) + f"\n🛡️ Dealer Vanna: **{vanna_fmt}** | ⏳ Charm: **{charm_fmt}**",
                    inline=False
                )

            embed.add_field(name="Max Pain", value=f"${fmt_strike(gamma.max_pain)}", inline=True)
            embed.add_field(name="ATR",      value=f"${gamma.atr:.2f}",             inline=True)
            embed.add_field(name="Trend",    value=f"{flow.trend_bias} {flow.trend_probability:.0%}", inline=True)
            embed.set_footer(text="⚡ Real-time 0DTE")
            await ctx.send(embed=embed)

        except Exception as e:
            await ctx.send(embed=_error_embed("0DTE Error", str(e)))
            logger.exception(f"!0dte error for {ticker}")


@bot.command(name="stats")
async def cmd_stats(ctx: commands.Context, ticker: str = "SPY", days: int = 30):
    """!stats [TICKER] [DAYS] — Backtest accuracy"""
    async with ctx.typing():
        try:
            result = await db.backtest_accuracy(ticker.upper(), days)
            if "error" in result:
                await ctx.send(embed=_error_embed("No Data", result["error"]))
                return
            embed = discord.Embed(
                title=f"📊 {ticker.upper()} Bias Backtest ({days}d)",
                color=C_NEUT,
            )
            embed.add_field(name="Accuracy", value=f"{result['accuracy_pct']}%", inline=True)
            embed.add_field(name="Correct",  value=f"{result['correct']}/{result['total_signals']}", inline=True)
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(embed=_error_embed("Stats Error", str(e)))


@bot.command(name="ping")
async def cmd_ping(ctx: commands.Context):
    """!ping — Health check"""
    latency = round(bot.latency * 1000)
    embed = discord.Embed(
        title="🏓 Pong!",
        description=f"Latency: **{latency}ms**",
        color=C_BULL if fetcher.is_connected else C_NEUT,
    )
    await ctx.send(embed=embed)


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND TASKS
# ═══════════════════════════════════════════════════════════════════════════════

@tasks.loop(seconds=POLL_INTERVAL)
async def auto_poll():
    """Background polling — pre-warms cache for default tickers."""
    if not bot.is_ready():
        return
    for ticker in DEFAULT_TICKERS:
        try:
            await fetcher.get_chain(ticker, zero_dte_only=True)
            logger.debug(f"Auto-polled {ticker}")
        except Exception as e:
            logger.warning(f"Auto-poll failed for {ticker}: {e}")
        await asyncio.sleep(2)


@tasks.loop(time=dtime(hour=13, minute=30))   # 9:30 ET = 13:30 UTC
async def daily_open_report():
    """Auto daily market open report."""
    if not AUTO_CHANNEL_ID:
        return
    channel = bot.get_channel(AUTO_CHANNEL_ID)
    if not channel:
        return

    for ticker in DEFAULT_TICKERS:
        try:
            _, bias, gamma, flow = await _fetch_all(ticker)
            embed = build_full_embed(ticker, bias, gamma, flow)
            embed.title = f"🌅 Morning Report — {embed.title}"
            await channel.send(embed=embed)
            await asyncio.sleep(3)
        except Exception as e:
            logger.error(f"Daily report failed for {ticker}: {e}")


@tasks.loop(minutes=10)
async def gamma_flip_alert():
    """
    Alert when gamma flip zone moves significantly — implies dealer regime change.
    """
    if not AUTO_CHANNEL_ID:
        return

    prev_flips: Dict[str, Optional[float]] = {}
    channel = bot.get_channel(AUTO_CHANNEL_ID)
    if not channel:
        return

    for ticker in DEFAULT_TICKERS:
        try:
            snapshot = await fetcher.get_chain(ticker)
            if not snapshot:
                continue

            atr = await fetcher.atr(ticker)
            gamma_engine.atr = atr
            gamma_result = gamma_engine.compute(snapshot)
            new_flip = gamma_result.gamma_flip

            old_flip = prev_flips.get(ticker)
            if old_flip and new_flip and abs(new_flip - old_flip) > atr * 0.3:
                embed = discord.Embed(
                    title=f"🔀 Gamma Flip Alert — {ticker}",
                    description=(
                        f"Gamma flip shifted from **${old_flip:.1f}** → **${new_flip:.1f}**\n"
                        f"Spot: **${snapshot.underlying.price:.2f}**  ·  ATR: ${atr:.2f}\n"
                        f"Dealer hedging regime may be changing."
                    ),
                    color=0xFF8C00,
                    timestamp=datetime.utcnow(),
                )
                await channel.send(embed=embed)

            prev_flips[ticker] = new_flip
        except Exception as e:
            logger.warning(f"Gamma flip monitor error {ticker}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  BOT EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

@bot.event
async def on_ready():
    logger.info(f"✅ Bot logged in as {bot.user} ({bot.user.id})")

    # Init database
    await db.init()

    # Initialize the data fetcher (non-blocking)
    asyncio.create_task(fetcher.connect())

    # Start background tasks
    auto_poll.start()
    daily_open_report.start()
    gamma_flip_alert.start()

    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="0DTE flow 📊"
        )
    )
    logger.info("🚀 All systems go.")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(embed=_error_embed("Missing Argument", str(error)))
        return
    logger.error(f"Unhandled command error: {error}", exc_info=True)
    await ctx.send(embed=_error_embed("Unexpected Error", str(error)))


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not DISCORD_TOKEN:
        raise RuntimeError(
            "DISCORD_TOKEN not set. Copy .env.example → .env and fill in your token."
        )
    bot.run(DISCORD_TOKEN, log_handler=None)   # We manage logging ourselves


if __name__ == "__main__":
    main()
