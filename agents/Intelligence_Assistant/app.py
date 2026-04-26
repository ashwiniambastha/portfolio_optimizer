"""
app.py — Portfolio Intelligence Assistant: Gradio UI
Run: python app.py
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
from datetime import datetime, timedelta

from agent import (
    intelligence_agent,
    market_data_agent,
    alpha_signal_agent,
    risk_management_agent,
    news_fetcher_agent,
    sentiment_score,
    robo_advisor_agent,
    classify_risk_profile,
    whatif_simulator_agent,
    portfolio_agent,
    compare_stocks_agent,
    chat_agent,
    format_bot_message,
    ROBO_QUESTIONS,
    DISCLAIMER,
)

# ── Persistent state ─────────────────────────────────────────
USER_PROFILE_FILE = "user_profile.json"
WATCHLIST_FILE = "watchlist.json"
PORTFOLIO_FILE = "portfolio.json"

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

user_profile = load_json(USER_PROFILE_FILE)
watchlist = load_json(WATCHLIST_FILE).get("tickers", [])
portfolio_holdings = load_json(PORTFOLIO_FILE).get("holdings", [])

# ── Chart helpers ─────────────────────────────────────────────

def candlestick_chart(ticker: str):
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo", interval="1d")
        if hist.empty:
            return go.Figure()
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"],
            name=ticker,
        )])
        # SMA overlays
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Close"].rolling(20).mean(),
            name="SMA 20", line=dict(color="#00d4aa", width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Close"].rolling(50).mean(),
            name="SMA 50", line=dict(color="#ff6b6b", width=1.5)
        ))
        fig.update_layout(
            title=f"{ticker} — Candlestick Chart (3M)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig
    except Exception as e:
        return go.Figure()

def rsi_chart(market_data: dict):
    if not market_data or "hist_dates" not in market_data:
        return go.Figure()
    import yfinance as yf
    import pandas as pd
    ticker = market_data.get("ticker", "")
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo", interval="1d")
        close = hist["Close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=rsi, name="RSI", line=dict(color="#a78bfa")))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought 70")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold 30")
        fig.update_layout(
            title=f"{ticker} — RSI (14-period)",
            template="plotly_dark", height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(range=[0, 100]),
        )
        return fig
    except Exception:
        return go.Figure()

def pie_chart(allocation: dict, title="Portfolio Allocation"):
    colors = ["#00d4aa", "#a78bfa", "#fbbf24", "#60a5fa", "#f87171", "#34d399"]
    fig = go.Figure(data=[go.Pie(
        labels=list(allocation.keys()),
        values=list(allocation.values()),
        hole=0.45,
        marker_colors=colors[:len(allocation)],
        textinfo="label+percent",
    )])
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=True,
    )
    return fig

def line_chart(dates, values, title, label="Value ($)", benchmark_dates=None, benchmark_values=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, name=label, line=dict(color="#00d4aa", width=2)))
    if benchmark_dates and benchmark_values:
        fig.add_trace(go.Scatter(
            x=benchmark_dates, y=benchmark_values,
            name="S&P 500 Benchmark", line=dict(color="#fbbf24", width=1.5, dash="dash")
        ))
    fig.update_layout(
        title=title, template="plotly_dark", height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE STOCK
# ════════════════════════════════════════════════════════════

def analyze_stock(ticker: str):
    ticker = ticker.strip().upper()
    if not ticker:
        return "Please enter a ticker symbol.", go.Figure(), go.Figure()

    data = intelligence_agent(ticker)
    if "error" in data:
        return f"❌ Error: {data['error']}", go.Figure(), go.Figure()

    md = data["market_data"]
    alpha = data["alpha"]
    risk = data["risk"]
    news = data["news"]
    verdict = data["verdict"]

    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}.get(alpha.get("signal", ""), "⚪")
    risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk.get("risk_level", ""), "⚪")

    news_lines = "\n".join([
        f"  • {n['title'][:100]} — {n.get('sentiment', '')} | {n['publisher']}"
        for n in news
    ])

    report = f"""
📊 **Market Snapshot** — {ticker}
─────────────────────────────────────
💵 Price: **${md['price']}**  |  Day Change: **{md['day_change_pct']:+.2f}%**
📦 Volume: {md['volume']:,}  |  Avg Volume: {md['avg_volume']:,}  |  Volume Ratio: {md['volume_ratio']}x
RSI (14): **{md['rsi']}** — {md['rsi_signal']}
SMA 20: {md['sma20']}  |  SMA 50: {md.get('sma50', 'N/A')}  |  Crossover: **{md['sma_signal']}**

{signal_emoji} **Alpha Signal: {alpha['signal']}**  (Score: {alpha['score']})
{'  '.join([f'↳ {r}' for r in alpha['reasons']])}

⚖️ **Risk Metrics**
─────────────────────────────────────
Sharpe Ratio: **{risk.get('sharpe_ratio', 'N/A')}**
Max Drawdown: **{risk.get('max_drawdown_pct', 'N/A')}%**
Annual Volatility: **{risk.get('annualized_volatility_pct', 'N/A')}%**
Risk Level: {risk_emoji} **{risk.get('risk_level', 'N/A')}**

📰 **Latest News** (Source: yfinance)
─────────────────────────────────────
{news_lines}

🎯 **AI Verdict**
─────────────────────────────────────
{verdict}
{DISCLAIMER}
"""
    return report.strip(), candlestick_chart(ticker), rsi_chart(md)


# ════════════════════════════════════════════════════════════
# TAB 2 — COMPARE STOCKS
# ════════════════════════════════════════════════════════════

def compare_stocks(tickers_input: str):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        return pd.DataFrame(), go.Figure()

    results = compare_stocks_agent(tickers)
    df = pd.DataFrame(results)

    # Bar chart — Sharpe ratios
    fig = go.Figure()
    valid = [r for r in results if "error" not in r]
    if valid:
        fig.add_trace(go.Bar(
            x=[r["ticker"] for r in valid],
            y=[r["sharpe"] for r in valid],
            marker_color=["#00d4aa" if r["signal"] == "BUY" else ("#ff6b6b" if r["signal"] == "SELL" else "#a78bfa") for r in valid],
            name="Sharpe Ratio",
        ))
        fig.update_layout(
            title="Sharpe Ratio Comparison",
            template="plotly_dark", height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )

    return df, fig


# ════════════════════════════════════════════════════════════
# TAB 3 — MY PORTFOLIO
# ════════════════════════════════════════════════════════════

def load_portfolio():
    data = portfolio_agent(portfolio_holdings)
    if "error" in data:
        return pd.DataFrame(), go.Figure(), f"❌ {data['error']}"

    holdings = data["holdings"]
    df = pd.DataFrame([{
        "Ticker": h["ticker"],
        "Shares": h["shares"],
        "Avg Cost": h["avg_cost"],
        "Current": h["current_price"],
        "Invested": h["invested"],
        "Value": h["current_value"],
        "P&L ($)": h["pnl"],
        "P&L (%)": h["pnl_pct"],
        "Day Change": f"{h.get('day_change_pct', 0):+.2f}%",
    } for h in holdings if "error" not in h])

    alloc = {h["ticker"]: h["current_value"] for h in holdings if "error" not in h}
    chart = pie_chart(alloc, "Portfolio Allocation by Value") if alloc else go.Figure()

    summary = f"""
📁 **Portfolio Summary**
─────────────────────────────────────
💰 Total Invested: **${data['total_invested']:,.2f}**
📈 Total Value: **${data['total_value']:,.2f}**
💵 Total P&L: **${data['total_pnl']:+,.2f}** ({data['total_pnl_pct']:+.2f}%)
📊 Portfolio Sharpe: **{data['portfolio_sharpe']}**
🏆 Best Performer: **{data.get('best_performer', 'N/A')}**
⚠️ Worst Performer: **{data.get('worst_performer', 'N/A')}**
{DISCLAIMER}
"""
    return df, chart, summary.strip()

def add_holding(ticker, shares, avg_cost):
    global portfolio_holdings
    ticker = ticker.strip().upper()
    if not ticker:
        return "Please enter a ticker."
    portfolio_holdings.append({
        "ticker": ticker,
        "shares": float(shares),
        "avg_cost": float(avg_cost),
    })
    save_json(PORTFOLIO_FILE, {"holdings": portfolio_holdings})
    return f"✅ Added {shares} shares of {ticker} at ${avg_cost}"

def clear_portfolio():
    global portfolio_holdings
    portfolio_holdings = []
    save_json(PORTFOLIO_FILE, {"holdings": []})
    return "Portfolio cleared."


# ════════════════════════════════════════════════════════════
# TAB 4 — NEWS & EVENTS
# ════════════════════════════════════════════════════════════

def fetch_news(ticker: str):
    ticker = ticker.strip().upper()
    if not ticker:
        return "Please enter a ticker."
    news = news_fetcher_agent(ticker, max_news=10)
    lines = []
    for n in news:
        sent = sentiment_score(n["title"])
        lines.append(f"{sent} **{n['title']}**\n   📎 {n['publisher']} | [Link]({n['url']})\n")
    return "\n".join(lines) if lines else "No news found."


# ════════════════════════════════════════════════════════════
# TAB 5 — ROBO-ADVISOR
# ════════════════════════════════════════════════════════════

robo_state = {"answers": {}, "step": 0}

def robo_next(user_input: str):
    step = robo_state["step"]
    if user_input.strip():
        robo_state["answers"][f"q{step + 1}"] = user_input.strip()
        robo_state["step"] += 1
        step = robo_state["step"]

    if step < len(ROBO_QUESTIONS):
        q_num = step + 1
        return f"**Question {q_num} of 5:** {ROBO_QUESTIONS[step]}", go.Figure(), ""

    # All 5 answered → generate portfolio
    result = robo_advisor_agent(robo_state["answers"])
    profile = result["profile"]
    allocation = result["allocation"]
    tickers = result["tickers"]

    summary = f"""
✅ **Robo-Advisor Profile: {profile}**
─────────────────────────────────────
Expected Return: **{result['expected_return']}**
Risk Level: **{result['risk_level']}**

**Recommended Portfolio:**
"""
    for category, pct in allocation.items():
        t_list = ", ".join(tickers.get(category, [])[:3])
        summary += f"\n• {category} ({pct}%): {t_list}"

    summary += f"\n{DISCLAIMER}"

    # Reset for next run
    robo_state["answers"] = {}
    robo_state["step"] = 0

    # Save profile
    save_json(USER_PROFILE_FILE, result)

    return summary, pie_chart(allocation, f"{profile} Portfolio Allocation"), ""

def robo_reset():
    robo_state["answers"] = {}
    robo_state["step"] = 0
    return f"**Question 1 of 5:** {ROBO_QUESTIONS[0]}", go.Figure(), ""


# ════════════════════════════════════════════════════════════
# TAB 6 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════

def run_whatif(ticker: str, amount: float, start_date: str):
    ticker = ticker.strip().upper()
    result = whatif_simulator_agent(ticker, amount, start_date)
    if "error" in result:
        return f"❌ {result['error']}", go.Figure()

    vs_sp = result["sp500_return_pct"]
    outperform = "✅ Outperformed" if result["total_return_pct"] > vs_sp else "❌ Underperformed"

    report = f"""
💡 **What-If Simulation — {ticker}**
─────────────────────────────────────
💵 Amount Invested: **${amount:,.2f}** on {start_date}
📈 Buy Price: ${result['buy_price']} | Shares Bought: {result['shares_bought']}
📊 Current Price: ${result['current_price']}
💰 Current Value: **${result['current_value']:,.2f}**
🎯 Profit / Loss: **${result['profit_loss']:+,.2f}**
📈 Total Return: **{result['total_return_pct']:+.2f}%**
📅 Annualized Return: **{result['annualized_return_pct']:+.2f}%**

📊 S&P 500 Return (same period): **{vs_sp:+.2f}%**
{outperform} vs S&P 500

🔍 Best Case (bought at dip): ${result['best_case_value']:,.2f}
🔍 Worst Case (bought at peak): ${result['worst_case_value']:,.2f}

⚠️ *Past performance does not guarantee future results.*
{DISCLAIMER}
"""

    # Growth line chart
    dates = result["hist_dates"]
    prices = result["hist_prices"]
    base = prices[0] if prices else 1
    growth = [result["shares_bought"] * p for p in prices]

    import yfinance as yf
    try:
        sp = yf.download("^GSPC", start=start_date, auto_adjust=True, progress=False)["Close"].squeeze()
        sp_base = float(sp.iloc[0])
        sp_growth = [(amount / sp_base) * float(p) for p in sp.values]
        sp_dates = [str(d.date()) for d in sp.index]
    except Exception:
        sp_growth = None
        sp_dates = None

    chart = line_chart(
        dates, growth,
        title=f"${amount:,.0f} in {ticker} from {start_date}",
        label=f"{ticker} Value",
        benchmark_dates=sp_dates,
        benchmark_values=sp_growth,
    )

    return report.strip(), chart


# ════════════════════════════════════════════════════════════
# TAB 7 — CHAT ASSISTANT
# ════════════════════════════════════════════════════════════

chat_history = []

def chat_respond(message: str, history):
    claude_history = []
    for msg in history:
        if isinstance(msg, dict):
            claude_history.append({"role": msg["role"], "content": msg["content"]})
        else:
            user_msg, bot_msg = msg[0], msg[1]
            claude_history.append({"role": "user", "content": user_msg})
            if bot_msg:
                claude_history.append({"role": "assistant", "content": bot_msg})

    response = chat_agent(message, claude_history)
    return response


# ════════════════════════════════════════════════════════════
# TAB 8 — REPORTS
# ════════════════════════════════════════════════════════════

def generate_report(ticker: str):
    ticker = ticker.strip().upper()
    data = intelligence_agent(ticker)
    if "error" in data:
        return f"❌ {data['error']}", None

    text = f"""PORTFOLIO INTELLIGENCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Ticker: {ticker}

"""
    md = data["market_data"]
    alpha = data["alpha"]
    risk = data["risk"]
    news = data["news"]
    verdict = data["verdict"]

    text += f"PRICE SNAPSHOT\nPrice: ${md['price']} | Day Change: {md['day_change_pct']:+.2f}%\nRSI: {md['rsi']} ({md['rsi_signal']})\nSMA Signal: {md['sma_signal']}\n\n"
    text += f"ALPHA SIGNAL\n{alpha['signal']} (Score: {alpha['score']})\nReasons: {'; '.join(alpha['reasons'])}\n\n"
    text += f"RISK METRICS\nSharpe: {risk.get('sharpe_ratio')} | Max Drawdown: {risk.get('max_drawdown_pct')}% | Risk: {risk.get('risk_level')}\n\n"
    text += f"LATEST NEWS\n" + "\n".join([f"- {n['title']}" for n in news]) + "\n\n"
    text += f"VERDICT\n{verdict}\n\nDISCLAIMER\nThis is not financial advice. Always do your own research."

    path = f"/tmp/{ticker}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(path, "w") as f:
        f.write(text)

    return text, path


# ════════════════════════════════════════════════════════════
# TAB 9 — WATCHLIST
# ════════════════════════════════════════════════════════════

def add_to_watchlist(ticker: str):
    global watchlist
    ticker = ticker.strip().upper()
    if ticker and ticker not in watchlist:
        watchlist.append(ticker)
        save_json(WATCHLIST_FILE, {"tickers": watchlist})
    return get_watchlist()

def get_watchlist():
    if not watchlist:
        return "Watchlist is empty. Add tickers above."
    rows = []
    for tk in watchlist:
        md = market_data_agent(tk)
        if "error" not in md:
            rows.append(f"**{tk}**: ${md['price']} ({md['day_change_pct']:+.2f}%) | RSI: {md['rsi']} | Signal: {md['sma_signal']}")
        else:
            rows.append(f"**{tk}**: Error fetching data")
    return "\n".join(rows)

def remove_from_watchlist(ticker: str):
    global watchlist
    ticker = ticker.strip().upper()
    watchlist = [t for t in watchlist if t != ticker]
    save_json(WATCHLIST_FILE, {"tickers": watchlist})
    return get_watchlist()


# ════════════════════════════════════════════════════════════
# GRADIO APP
# ════════════════════════════════════════════════════════════

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0f1117",
    body_background_fill_dark="#0f1117",
    block_background_fill="#1a1d2e",
    block_border_color="#2d3748",
    button_primary_background_fill="#00d4aa",
    button_primary_background_fill_hover="#00b894",
    button_primary_text_color="#0f1117",
)

CSS = """
:root {
    --accent: #00d4aa;
    --accent2: #a78bfa;
}
.gradio-container { max-width: 1200px !important; }
.tab-nav button { font-size: 0.85rem !important; font-weight: 600 !important; }
.tab-nav button.selected { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }
.metric-box { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 8px; padding: 12px; }
textarea { font-family: 'JetBrains Mono', monospace !important; }
"""

with gr.Blocks(title="📈 Portfolio Intelligence Assistant") as demo:
    gr.HTML("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <h1 style="font-size:2rem; font-weight:800; color:#00d4aa; letter-spacing:-0.5px; margin:0;">
            📈 Portfolio Intelligence Assistant
        </h1>
        <p style="color:#94a3b8; margin:6px 0 0; font-size:0.95rem;">
            AI-powered stock analysis · Robo-advisor · What-If simulator · Real-time data
        </p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Analyze Stock ─────────────────────────────
        with gr.Tab("🔍 Analyze Stock"):
            with gr.Row():
                ticker_in = gr.Textbox(label="Ticker Symbol", placeholder="e.g. AAPL, NVDA, RELIANCE.NS", scale=4)
                analyze_btn = gr.Button("Analyze", variant="primary", scale=1)
            analysis_out = gr.Markdown()
            with gr.Row():
                candle_chart = gr.Plot(label="Candlestick + SMA")
                rsi_plot = gr.Plot(label="RSI Chart")
            analyze_btn.click(analyze_stock, inputs=ticker_in, outputs=[analysis_out, candle_chart, rsi_plot])
            ticker_in.submit(analyze_stock, inputs=ticker_in, outputs=[analysis_out, candle_chart, rsi_plot])

        # ── Tab 2: Compare Stocks ─────────────────────────────
        with gr.Tab("📊 Compare Stocks"):
            compare_in = gr.Textbox(label="Tickers (comma-separated)", placeholder="AAPL, MSFT, NVDA, TSLA")
            compare_btn = gr.Button("Compare", variant="primary")
            compare_df = gr.Dataframe(label="Comparison Table")
            compare_chart = gr.Plot(label="Sharpe Ratio Chart")
            compare_btn.click(compare_stocks, inputs=compare_in, outputs=[compare_df, compare_chart])

        # ── Tab 3: My Portfolio ───────────────────────────────
        with gr.Tab("💼 My Portfolio"):
            gr.Markdown("### Add Holding")
            with gr.Row():
                port_ticker = gr.Textbox(label="Ticker", placeholder="AAPL")
                port_shares = gr.Number(label="Shares", value=1)
                port_cost = gr.Number(label="Avg Cost ($)", value=100)
                add_btn = gr.Button("Add", variant="primary")
            add_status = gr.Markdown()
            add_btn.click(add_holding, inputs=[port_ticker, port_shares, port_cost], outputs=add_status)

            with gr.Row():
                load_btn = gr.Button("🔄 Refresh Portfolio", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear Portfolio", variant="stop")

            port_df = gr.Dataframe(label="Holdings")
            port_chart = gr.Plot(label="Allocation Pie Chart")
            port_summary = gr.Markdown()

            load_btn.click(load_portfolio, outputs=[port_df, port_chart, port_summary])
            clear_btn.click(clear_portfolio, outputs=add_status)

        # ── Tab 4: News & Events ──────────────────────────────
        with gr.Tab("📰 News & Events"):
            with gr.Row():
                news_ticker = gr.Textbox(label="Ticker", placeholder="TSLA")
                news_btn = gr.Button("Fetch News", variant="primary")
            news_out = gr.Markdown()
            news_btn.click(fetch_news, inputs=news_ticker, outputs=news_out)
            news_ticker.submit(fetch_news, inputs=news_ticker, outputs=news_out)

        # ── Tab 5: Robo-Advisor ───────────────────────────────
        with gr.Tab("🤖 Robo-Advisor"):
            gr.Markdown("### Build Your Personalized Portfolio\nAnswer 5 quick questions to get a custom portfolio recommendation.")
            robo_q = gr.Markdown(f"**Question 1 of 5:** {ROBO_QUESTIONS[0]}")
            robo_answer = gr.Textbox(label="Your Answer", placeholder="Type your answer here...")
            with gr.Row():
                robo_next_btn = gr.Button("Next →", variant="primary")
                robo_reset_btn = gr.Button("↩️ Restart", variant="secondary")
            robo_result = gr.Markdown()
            robo_pie = gr.Plot()

            robo_next_btn.click(robo_next, inputs=robo_answer, outputs=[robo_q, robo_pie, robo_answer])
            robo_reset_btn.click(robo_reset, outputs=[robo_q, robo_pie, robo_answer])

        # ── Tab 6: What-If Simulator ──────────────────────────
        with gr.Tab("⏳ What-If Simulator"):
            gr.Markdown("### Historical Investment Simulator\n*What if you had invested $X in Y on date Z?*")
            with gr.Row():
                wi_ticker = gr.Textbox(label="Ticker", placeholder="NVDA", scale=2)
                wi_amount = gr.Number(label="Investment ($)", value=10000, scale=2)
                wi_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", placeholder="2022-01-01", scale=2)
                wi_btn = gr.Button("Simulate", variant="primary", scale=1)
            wi_result = gr.Markdown()
            wi_chart = gr.Plot()
            wi_btn.click(run_whatif, inputs=[wi_ticker, wi_amount, wi_date], outputs=[wi_result, wi_chart])

        # ── Tab 7: Chat ────────────────────────────────────────
        with gr.Tab("💬 Chat Assistant"):
            gr.Markdown("### Ask me anything about stocks, markets, or your portfolio.")
            chat_ui = gr.ChatInterface(
                fn=chat_respond,
                
                chatbot=gr.Chatbot(height=450, placeholder="Ask: *Analyze NVDA* | *Best tech stocks for 2025?* | *Build me a portfolio*"),
                textbox=gr.Textbox(placeholder="Type your question...", show_label=False),
                )

        # ── Tab 8: Reports ─────────────────────────────────────
        with gr.Tab("📄 Reports"):
            gr.Markdown("### Generate Full Stock Report")
            with gr.Row():
                report_ticker = gr.Textbox(label="Ticker", placeholder="AAPL")
                report_btn = gr.Button("Generate Report", variant="primary")
            report_out = gr.Textbox(label="Report Preview", lines=25)
            report_file = gr.File(label="Download Report")
            report_btn.click(generate_report, inputs=report_ticker, outputs=[report_out, report_file])

        # ── Tab 9: Watchlist ────────────────────────────────────
        with gr.Tab("⭐ Watchlist"):
            gr.Markdown("### My Watchlist")
            with gr.Row():
                wl_ticker = gr.Textbox(label="Add Ticker", placeholder="MSFT")
                wl_add_btn = gr.Button("➕ Add", variant="primary")
                wl_remove_btn = gr.Button("➖ Remove", variant="secondary")
            wl_refresh_btn = gr.Button("🔄 Refresh Prices", variant="secondary")
            wl_out = gr.Markdown()

            wl_add_btn.click(add_to_watchlist, inputs=wl_ticker, outputs=wl_out)
            wl_remove_btn.click(remove_from_watchlist, inputs=wl_ticker, outputs=wl_out)
            wl_refresh_btn.click(get_watchlist, outputs=wl_out)

    gr.HTML("""
    <div style="text-align:center; padding:12px 0; color:#475569; font-size:0.8rem;">
        ⚠️ Portfolio Intelligence Assistant is for informational purposes only.
        This is not financial advice. Data powered by yfinance. AI powered by Claude.
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        
        show_error=True,
    )