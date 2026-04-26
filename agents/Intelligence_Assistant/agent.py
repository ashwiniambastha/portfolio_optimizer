"""
agent.py — Portfolio Intelligence Assistant: All Specialist Agents
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
import os
from groq import Groq

# ─── Groq client (no key needed in Artifacts; set GROQ_API_KEY in env) ───
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


DISCLAIMER = "\n\n⚠️ *This is not financial advice. Always do your own research.*"

# ════════════════════════════════════════════════════════════
# 1. MARKET DATA AGENT
# ════════════════════════════════════════════════════════════

def market_data_agent(ticker: str) -> dict:
    """Fetch real-time price, RSI, SMA, volume for a ticker."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        hist = tk.history(period="3mo", interval="1d")

        if hist.empty:
            return {"error": f"No data found for {ticker}"}

        close = hist["Close"]
        volume = hist["Volume"]

        # RSI (14-period)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = round(float(rsi.iloc[-1]), 2)

        # SMA
        sma20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
        sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None

        # SMA signal
        if sma50:
            sma_signal = "BUY" if sma20 > sma50 else ("SELL" if sma20 < sma50 else "NEUTRAL")
        else:
            sma_signal = "NEUTRAL (insufficient data)"

        # Volume anomaly
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        cur_vol = float(volume.iloc[-1])
        vol_ratio = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 1.0

        current_price = round(float(info.last_price), 2)
        prev_close = round(float(info.previous_close), 2)
        day_change = round(((current_price - prev_close) / prev_close) * 100, 2)

        return {
            "ticker": ticker.upper(),
            "price": current_price,
            "prev_close": prev_close,
            "day_change_pct": day_change,
            "volume": int(cur_vol),
            "avg_volume": int(avg_vol),
            "volume_ratio": vol_ratio,
            "rsi": rsi_val,
            "rsi_signal": "OVERBOUGHT" if rsi_val > 70 else ("OVERSOLD" if rsi_val < 30 else "NEUTRAL"),
            "sma20": sma20,
            "sma50": sma50,
            "sma_signal": sma_signal,
            "hist_close": close.tolist(),
            "hist_dates": [str(d.date()) for d in close.index],
        }
    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════
# 2. ALPHA SIGNAL AGENT
# ════════════════════════════════════════════════════════════

def alpha_signal_agent(market_data: dict) -> dict:
    """Generate BUY/SELL/NEUTRAL signal with reasoning."""
    if "error" in market_data:
        return {"signal": "UNKNOWN", "reason": market_data["error"]}

    signals = []
    reasons = []

    # SMA crossover
    sma_sig = market_data.get("sma_signal", "NEUTRAL")
    if sma_sig == "BUY":
        signals.append(1)
        reasons.append(f"SMA20 ({market_data['sma20']}) crossed above SMA50 ({market_data['sma50']}) — bullish crossover")
    elif sma_sig == "SELL":
        signals.append(-1)
        reasons.append(f"SMA20 ({market_data['sma20']}) crossed below SMA50 ({market_data['sma50']}) — bearish crossover")
    else:
        signals.append(0)
        reasons.append("SMA crossover is NEUTRAL")

    # RSI signal
    rsi = market_data.get("rsi", 50)
    if rsi < 30:
        signals.append(1)
        reasons.append(f"RSI at {rsi} — OVERSOLD, potential reversal upward")
    elif rsi > 70:
        signals.append(-1)
        reasons.append(f"RSI at {rsi} — OVERBOUGHT, potential pullback risk")
    else:
        signals.append(0)
        reasons.append(f"RSI at {rsi} — in neutral zone")

    # Day change
    change = market_data.get("day_change_pct", 0)
    if change > 3:
        signals.append(1)
        reasons.append(f"Strong positive day change: +{change}%")
    elif change < -3:
        signals.append(-1)
        reasons.append(f"Sharp negative day change: {change}%")

    # Volume anomaly
    vol_ratio = market_data.get("volume_ratio", 1.0)
    if vol_ratio > 2:
        reasons.append(f"⚡ Unusual volume spike: {vol_ratio}x average — possible breakout or news event")

    avg_signal = np.mean(signals) if signals else 0
    if avg_signal > 0.3:
        final_signal = "BUY"
    elif avg_signal < -0.3:
        final_signal = "SELL"
    else:
        final_signal = "NEUTRAL"

    return {"signal": final_signal, "score": round(avg_signal, 2), "reasons": reasons}


# ════════════════════════════════════════════════════════════
# 3. RISK MANAGEMENT AGENT
# ════════════════════════════════════════════════════════════

def risk_management_agent(market_data: dict) -> dict:
    """Calculate Sharpe ratio, max drawdown, risk level."""
    if "error" in market_data or not market_data.get("hist_close"):
        return {"error": "Insufficient data for risk analysis"}

    prices = pd.Series(market_data["hist_close"])
    returns = prices.pct_change().dropna()

    # Sharpe ratio (annualized, assuming 0% risk-free rate for simplicity)
    sharpe = round(float(returns.mean() / returns.std() * np.sqrt(252)), 2) if returns.std() > 0 else 0

    # Max drawdown
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = round(float(drawdown.min()) * 100, 2)

    # Volatility (annualized)
    volatility = round(float(returns.std() * np.sqrt(252)) * 100, 2)

    # Risk level
    if volatility > 40 or max_dd < -25:
        risk_level = "HIGH"
    elif volatility > 20 or max_dd < -15:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "annualized_volatility_pct": volatility,
        "risk_level": risk_level,
    }


# ════════════════════════════════════════════════════════════
# 4. NEWS FETCHER AGENT
# ════════════════════════════════════════════════════════════

def news_fetcher_agent(ticker: str, max_news: int = 5) -> list[dict]:
    """Fetch latest news headlines from yfinance."""
    try:
        tk = yf.Ticker(ticker)
        news = tk.news or []
        results = []
        for item in news[:max_news]:
            content = item.get("content", {})
            title = content.get("title", item.get("title", "No title"))
            url = content.get("canonicalUrl", {}).get("url", "") or item.get("link", "")
            publisher = content.get("provider", {}).get("displayName", item.get("publisher", "Unknown"))
            results.append({"title": title, "url": url, "publisher": publisher})
        return results
    except Exception as e:
        return [{"title": f"Could not fetch news: {e}", "url": "", "publisher": ""}]


def sentiment_score(headline: str) -> str:
    """Simple keyword-based sentiment classifier."""
    positive_words = ["surge", "gain", "profit", "beat", "record", "growth", "strong", "rally", "upgrade", "buy"]
    negative_words = ["fall", "drop", "loss", "miss", "decline", "weak", "sell", "downgrade", "crash", "risk"]
    headline_lower = headline.lower()
    pos = sum(1 for w in positive_words if w in headline_lower)
    neg = sum(1 for w in negative_words if w in headline_lower)
    if pos > neg:
        return "🟢 POSITIVE"
    elif neg > pos:
        return "🔴 NEGATIVE"
    return "⚪ NEUTRAL"


# ════════════════════════════════════════════════════════════
# 5. ROBO-ADVISOR AGENT
# ════════════════════════════════════════════════════════════

ROBO_PROFILES = {
    "Conservative": {
        "allocation": {"Large-cap Stocks": 60, "Bonds/ETFs": 25, "Gold/Commodities": 15},
        "tickers": {
            "Large-cap Stocks": ["JNJ", "PG", "KO", "VZ", "WMT"],
            "Bonds/ETFs": ["BND", "AGG", "TLT", "VBTLX", "IEF"],
            "Gold/Commodities": ["GLD", "IAU", "PDBC", "DJP", "GSG"],
        },
        "expected_return": "5–8% annually",
        "risk_level": "Low",
    },
    "Moderate": {
        "allocation": {"Diversified Stocks": 50, "ETFs": 30, "Mid-caps": 15, "Crypto": 5},
        "tickers": {
            "Diversified Stocks": ["AAPL", "MSFT", "AMZN", "GOOGL", "BRK-B"],
            "ETFs": ["SPY", "QQQ", "VTI", "IVV", "VOO"],
            "Mid-caps": ["MELI", "DXCM", "GNRC", "PAYC", "POOL"],
            "Crypto": ["GBTC", "ETHE"],
        },
        "expected_return": "8–12% annually",
        "risk_level": "Medium",
    },
    "Aggressive": {
        "allocation": {"Growth Stocks": 60, "Small/Mid-caps": 20, "Sector ETFs": 15, "Crypto": 5},
        "tickers": {
            "Growth Stocks": ["NVDA", "TSLA", "META", "AMD", "PLTR"],
            "Small/Mid-caps": ["ENPH", "CRSP", "ASTS", "RXRX", "LUNR"],
            "Sector ETFs": ["ARKK", "SOXX", "BOTZ", "MCHI", "IYW"],
            "Crypto": ["GBTC", "IBIT"],
        },
        "expected_return": "12–20%+ annually (high variance)",
        "risk_level": "High",
    },
}

ROBO_QUESTIONS = [
    "How old are you? (This helps assess your risk capacity and investment horizon)",
    "What is your monthly income range? (Low: <$2k / Medium: $2k–$10k / High: >$10k)",
    "What is your risk tolerance? (Conservative / Moderate / Aggressive)",
    "What is your investment goal? (Wealth Building / Retirement / Short-term Gains / Passive Income)",
    "What is your investment time horizon? (Less than 1 year / 1–3 years / 3–10 years / 10+ years)",
]

def classify_risk_profile(answers: dict) -> str:
    """Derive risk profile from user answers."""
    risk_explicit = str(answers.get("q3", "")).lower()
    if "conservative" in risk_explicit:
        return "Conservative"
    if "aggressive" in risk_explicit:
        return "Aggressive"

    # Age-based heuristic
    age_str = str(answers.get("q1", "30"))
    age = int(re.search(r"\d+", age_str).group()) if re.search(r"\d+", age_str) else 30
    horizon = str(answers.get("q5", "")).lower()

    if age < 35 and ("10+" in horizon or "3–10" in horizon):
        return "Aggressive"
    if age > 55 or "less than 1" in horizon:
        return "Conservative"
    return "Moderate"

def robo_advisor_agent(answers: dict) -> dict:
    profile = classify_risk_profile(answers)
    data = ROBO_PROFILES[profile]
    return {
        "profile": profile,
        "allocation": data["allocation"],
        "tickers": data["tickers"],
        "expected_return": data["expected_return"],
        "risk_level": data["risk_level"],
        "user_answers": answers,
    }


# ════════════════════════════════════════════════════════════
# 6. WHAT-IF SIMULATOR AGENT
# ════════════════════════════════════════════════════════════

def whatif_simulator_agent(ticker: str, amount: float, start_date: str) -> dict:
    """Historical investment simulation with S&P500 benchmark."""
    try:
        start = pd.to_datetime(start_date)
        end = datetime.today()

        tk = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        sp = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)

        if tk.empty or sp.empty:
            return {"error": f"No historical data for {ticker} from {start_date}"}

        tk_close = tk["Close"].squeeze()
        sp_close = sp["Close"].squeeze()

        buy_price = float(tk_close.iloc[0])
        current_price = float(tk_close.iloc[-1])
        shares = amount / buy_price
        current_value = shares * current_price
        total_return_pct = round(((current_value - amount) / amount) * 100, 2)

        years = (end - start).days / 365.25
        annualized_return = round(((current_value / amount) ** (1 / years) - 1) * 100, 2) if years > 0 else 0

        sp_start = float(sp_close.iloc[0])
        sp_end = float(sp_close.iloc[-1])
        sp_return_pct = round(((sp_end - sp_start) / sp_start) * 100, 2)

        # Peak & dip
        peak_price = float(tk_close.max())
        dip_price = float(tk_close.min())
        best_value = round((amount / dip_price) * current_price, 2)
        worst_value = round((amount / peak_price) * current_price, 2)

        return {
            "ticker": ticker.upper(),
            "amount_invested": amount,
            "start_date": start_date,
            "shares_bought": round(shares, 4),
            "buy_price": round(buy_price, 2),
            "current_price": round(current_price, 2),
            "current_value": round(current_value, 2),
            "profit_loss": round(current_value - amount, 2),
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return,
            "sp500_return_pct": sp_return_pct,
            "best_case_value": best_value,
            "worst_case_value": worst_value,
            "hist_prices": tk_close.tolist(),
            "hist_dates": [str(d.date()) for d in tk_close.index],
        }
    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════
# 7. PORTFOLIO MANAGEMENT AGENT
# ════════════════════════════════════════════════════════════

def portfolio_agent(holdings: list[dict]) -> dict:
    """
    holdings: [{"ticker": "AAPL", "shares": 10, "avg_cost": 150}, ...]
    """
    results = []
    total_invested = 0
    total_value = 0
    returns_list = []

    for h in holdings:
        md = market_data_agent(h["ticker"])
        if "error" in md:
            results.append({**h, "error": md["error"]})
            continue
        price = md["price"]
        shares = h.get("shares", 0)
        avg_cost = h.get("avg_cost", price)
        invested = shares * avg_cost
        current = shares * price
        pnl = current - invested
        pnl_pct = round((pnl / invested) * 100, 2) if invested > 0 else 0

        total_invested += invested
        total_value += current

        # daily returns for Sharpe
        hist = pd.Series(md["hist_close"]).pct_change().dropna()
        returns_list.append(hist.values)

        results.append({
            "ticker": h["ticker"],
            "shares": shares,
            "avg_cost": avg_cost,
            "current_price": price,
            "invested": round(invested, 2),
            "current_value": round(current, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": pnl_pct,
            "day_change_pct": md.get("day_change_pct", 0),
        })

    # Portfolio-level Sharpe
    if returns_list:
        combined = np.array([np.mean(r) for r in returns_list])
        mean_ret = np.mean(combined)
        std_ret = np.std(combined)
        sharpe = round(mean_ret / std_ret * np.sqrt(252), 2) if std_ret > 0 else 0
    else:
        sharpe = 0

    total_pnl = total_value - total_invested
    total_pnl_pct = round((total_pnl / total_invested) * 100, 2) if total_invested > 0 else 0

    sorted_results = sorted(results, key=lambda x: x.get("pnl_pct", 0), reverse=True)

    return {
        "holdings": results,
        "total_invested": round(total_invested, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": total_pnl_pct,
        "portfolio_sharpe": sharpe,
        "best_performer": sorted_results[0]["ticker"] if sorted_results else None,
        "worst_performer": sorted_results[-1]["ticker"] if sorted_results else None,
    }


# ════════════════════════════════════════════════════════════
# 8. INTELLIGENCE AGENT (Master Orchestrator)
# ════════════════════════════════════════════════════════════

def intelligence_agent(ticker: str) -> dict:
    """Orchestrate all agents and synthesize into one output."""
    md = market_data_agent(ticker)
    if "error" in md:
        return {"error": md["error"]}

    alpha = alpha_signal_agent(md)
    risk = risk_management_agent(md)
    news = news_fetcher_agent(ticker)

    # Sentiment on news
    for item in news:
        item["sentiment"] = sentiment_score(item["title"])

    # AI verdict via Claude
    summary_prompt = f"""
You are a financial analyst. Given this data for {ticker}, write a concise 3-bullet BULLISH/BEARISH/NEUTRAL verdict.

Price: ${md['price']} | Day Change: {md['day_change_pct']}%
RSI: {md['rsi']} ({md['rsi_signal']}) | SMA Signal: {md['sma_signal']}
Alpha Signal: {alpha['signal']} | Sharpe: {risk.get('sharpe_ratio', 'N/A')} | Risk: {risk.get('risk_level', 'N/A')}
Max Drawdown: {risk.get('max_drawdown_pct', 'N/A')}%
Top News: {'; '.join([n['title'] for n in news[:3]])}

Respond in exactly this format:
VERDICT: [BULLISH/BEARISH/NEUTRAL]
• Reason 1
• Reason 2
• Reason 3
"""
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        verdict_text = resp.content[0].text
    except Exception:
        verdict_text = f"VERDICT: {alpha['signal']}\n• Based on SMA crossover and RSI analysis\n• Risk level is {risk.get('risk_level', 'unknown')}\n• Monitor news closely"

    return {
        "market_data": md,
        "alpha": alpha,
        "risk": risk,
        "news": news,
        "verdict": verdict_text,
    }


# ════════════════════════════════════════════════════════════
# 9. CHAT AGENT (Conversational)
# ════════════════════════════════════════════════════════════

def chat_agent(message: str, history: list) -> str:
    try:
        system_prompt = """You are a Portfolio Intelligence Assistant..."""

        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # ✅ Define it before use
        summary_prompt = f"User asked: {message}\nPlease provide a helpful financial analysis."
        messages.append({"role": "user", "content": summary_prompt})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
            max_tokens=1000,
            messages=[{"role": "system", "content": system_prompt}] + messages,
                )
        return response.choices[0].message.content

    except Exception as e:
        return f"Chat error: {e}"

# ════════════════════════════════════════════════════════════
# 10. PORTFOLIO COMPARISON AGENT
# ════════════════════════════════════════════════════════════

def compare_stocks_agent(tickers: list[str]) -> list[dict]:
    """Compare multiple tickers on key metrics."""
    results = []
    for ticker in tickers:
        md = market_data_agent(ticker)
        if "error" in md:
            results.append({"ticker": ticker, "error": md["error"]})
            continue
        alpha = alpha_signal_agent(md)
        risk = risk_management_agent(md)
        results.append({
            "ticker": ticker,
            "price": md["price"],
            "day_change_pct": md["day_change_pct"],
            "rsi": md["rsi"],
            "sma_signal": md["sma_signal"],
            "signal": alpha["signal"],
            "sharpe": risk.get("sharpe_ratio", "N/A"),
            "risk_level": risk.get("risk_level", "N/A"),
            "max_drawdown": risk.get("max_drawdown_pct", "N/A"),
        })
    return results


# ════════════════════════════════════════════════════════════
# 11. BOT MESSAGE FORMATTER
# ════════════════════════════════════════════════════════════

def format_bot_message(data: dict, ticker: str) -> str:
    """Format analysis output for Telegram/WhatsApp (mobile-friendly)."""
    if "error" in data:
        return f"❌ Error for {ticker}: {data['error']}"

    md = data.get("market_data", {})
    alpha = data.get("alpha", {})
    risk = data.get("risk", {})
    news = data.get("news", [])[:3]
    verdict = data.get("verdict", "")

    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}.get(alpha.get("signal", ""), "⚪")

    msg = f"""📊 *{ticker}* Analysis
Price: ${md.get('price')} ({md.get('day_change_pct', 0):+.2f}%)
RSI: {md.get('rsi')} | {md.get('rsi_signal')}
SMA Signal: {md.get('sma_signal')}
{signal_emoji} Alpha Signal: *{alpha.get('signal')}*

⚖️ Risk: {risk.get('risk_level')} | Sharpe: {risk.get('sharpe_ratio')}
Max Drawdown: {risk.get('max_drawdown_pct')}%

📰 News:
"""
    for n in news:
        msg += f"• {n['title'][:80]}... {n.get('sentiment', '')}\n"

    msg += f"\n🎯 {verdict[:300]}"
    msg += DISCLAIMER
    return msg