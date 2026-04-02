"""
Alpha Signal Agent - Gradio UI
Week 4: Interactive Trading Signal Dashboard
"""

import gradio as gr
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# API endpoints
MARKET_DATA_API = "http://localhost:8000"
RISK_API = "http://localhost:8001"
ALPHA_API = "http://localhost:8002"

# Stock symbols
SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]


# ─────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────
def fmt(val, decimals=2):
    """Safe number formatter — returns N/A if val is None"""
    if val is None:
        return "N/A"
    try:
        return f"{val:.{decimals}f}"
    except:
        return "N/A"


def fetch_price_data(symbol, period="1y"):
    """Fetch historical price data from Market Data API"""
    try:
        response = requests.get(
            f"{MARKET_DATA_API}/historical/{symbol}",
            params={"period": period},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)

            # Handle date whether it's a column or index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif 'index' in df.columns:
                df['Date'] = pd.to_datetime(df['index'])
                df = df.drop(columns=['index'])
            else:
                df = df.reset_index()
                df.columns = ['Date'] + list(df.columns[1:])
                df['Date'] = pd.to_datetime(df['Date'])

            return df
        return None
    except Exception as e:
        print(f"fetch_price_data error: {e}")
        return None


# ─────────────────────────────────────────────
#  Tab 1 — Generate Signal
# ─────────────────────────────────────────────
def generate_signal_ui(symbol, portfolio_value):
    """Generate trading signal and display results"""
    try:
        # Call Alpha Signal API
        response = requests.get(
            f"{ALPHA_API}/signal/{symbol}",
            params={"portfolio_value": portfolio_value},
            timeout=15
        )

        if response.status_code != 200:
            return "❌ Error fetching signal", None, ""

        signal = response.json()

        if signal.get('status') == 'error':
            return f"❌ Error: {signal.get('message','Unknown error')}", None, ""

        # ── Signal result HTML ──────────────────────────────
        action      = signal.get('action', 'HOLD')
        rec         = signal.get('recommendation', 'HOLD')
        confidence  = signal.get('confidence', 0)
        price       = signal.get('current_price', 0)
        ts          = signal.get('timestamp', '')[:19]

        bg_color = (
            '#d4edda' if action == 'BUY'  else
            '#f8d7da' if action == 'SELL' else
            '#fff3cd'
        )

        result_html = f"""
        <div style="font-family: Arial, sans-serif; padding: 10px;">
            <h2 style="color: #1f77b4;">📊 Trading Signal for {signal.get('symbol','')}</h2>
            <p><strong>Current Price:</strong> ${fmt(price)}</p>
            <p><strong>Timestamp:</strong> {ts}</p>

            <div style="background:{bg_color}; padding:15px; border-radius:8px; margin:10px 0;">
                <h3 style="margin:0;">🎯 {rec}</h3>
                <p style="font-size:24px; margin:5px 0;"><strong>Action: {action}</strong></p>
                <p><strong>Confidence:</strong> {fmt(confidence, 1)}%</p>
            </div>

            <h3>📈 Individual Signals</h3>
            <ul>
        """

        signals      = signal.get('signals', {})
        explanations = signal.get('explanations', {})
        for indicator, value in signals.items():
            emoji = "🟢" if value > 0 else "🔴" if value < 0 else "⚪"
            result_html += (
                f"<li>{emoji} <strong>{indicator}:</strong> "
                f"{explanations.get(indicator, '')}</li>"
            )

        result_html += "</ul><h3>💰 Position Sizing</h3>"

        ps = signal.get('position_sizing', {})
        result_html += f"""
            <p><strong>Portfolio Value:</strong> ${fmt(ps.get('portfolio_value'))}</p>
            <p><strong>Recommended Position:</strong>
               {fmt(ps.get('position_size_pct'), 1)}%
               (${fmt(ps.get('actual_position_usd'))})</p>
            <p><strong>Shares to Buy:</strong> {ps.get('recommended_shares', 0)} shares</p>
        """

        rm = signal.get('risk_metrics', {})
        vol = rm.get('volatility')
        if vol is not None:
            result_html += f"""
            <p><strong>Volatility:</strong> {fmt(vol * 100, 2)}%</p>
            <p><strong>Risk Adjusted:</strong>
               {'Yes ✓' if rm.get('risk_adjusted') else 'No'}</p>
            """

        result_html += "</div>"

        # ── Technical indicator chart ───────────────────────
        df = fetch_price_data(symbol, "6mo")
        if df is not None and 'Close' in df.columns and 'Date' in df.columns:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
                row_heights=[0.5, 0.25, 0.25]
            )

            close_series = pd.Series(df['Close'].values)
            dates        = df['Date']

            # Price + MAs
            fig.add_trace(
                go.Scatter(x=dates, y=df['Close'],
                           name='Close', line=dict(color='blue')),
                row=1, col=1
            )
            sma_20 = close_series.rolling(20).mean()
            sma_50 = close_series.rolling(50).mean()
            fig.add_trace(
                go.Scatter(x=dates, y=sma_20,
                           name='SMA(20)', line=dict(color='orange', dash='dot')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=sma_50,
                           name='SMA(50)', line=dict(color='green', dash='dot')),
                row=1, col=1
            )

            # RSI
            delta    = close_series.diff()
            gain     = delta.where(delta > 0, 0).rolling(14).mean()
            loss     = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs       = gain / loss
            rsi      = 100 - (100 / (1 + rs))
            fig.add_trace(
                go.Scatter(x=dates, y=rsi,
                           name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # MACD
            ema_12      = close_series.ewm(span=12, adjust=False).mean()
            ema_26      = close_series.ewm(span=26, adjust=False).mean()
            macd        = ema_12 - ema_26
            signal_line = macd.ewm(span=9, adjust=False).mean()
            fig.add_trace(
                go.Scatter(x=dates, y=macd,
                           name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=signal_line,
                           name='Signal', line=dict(color='red')),
                row=3, col=1
            )

            fig.update_layout(
                height=800, showlegend=True,
                title_text=f"{symbol} — Technical Indicators"
            )
            fig.update_xaxes(title_text="Date",     row=3, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI",       row=2, col=1)
            fig.update_yaxes(title_text="MACD",      row=3, col=1)

            return result_html, fig, ""

        return result_html, None, ""

    except Exception as e:
        return f"❌ Error: {str(e)}", None, ""


# ─────────────────────────────────────────────
#  Tab 2 — Backtest
# ─────────────────────────────────────────────
def run_backtest_ui(symbol, initial_capital, period):
    """Run backtest and display results"""
    try:
        response = requests.get(
            f"{ALPHA_API}/backtest/{symbol}",
            params={"initial_capital": initial_capital, "period": period},
            timeout=20
        )

        if response.status_code != 200:
            return "❌ Error running backtest", None

        backtest = response.json()

        if backtest.get('status') == 'error':
            return f"❌ Error: {backtest.get('message','')}", None

        total_ret  = backtest.get('total_return_pct', 0)
        bh_ret     = backtest.get('buy_hold_return_pct', 0)
        outperf    = backtest.get('outperformance', 0)
        final_val  = backtest.get('final_value', 0)
        init_cap   = backtest.get('initial_capital', initial_capital)
        num_trades = backtest.get('num_trades', 0)
        sym        = backtest.get('symbol', symbol)
        per        = backtest.get('period', period)

        ret_color  = 'green' if total_ret  >= 0 else 'red'
        out_color  = 'green' if outperf    >= 0 else 'red'

        result_html = f"""
        <div style="font-family: Arial, sans-serif; padding:10px;">
            <h2 style="color: #1f77b4;">📊 Backtest Results: {sym}</h2>
            <p><strong>Period:</strong> {per}</p>
            <p><strong>Number of Trades:</strong> {num_trades}</p>

            <h3>💰 Performance</h3>
            <p><strong>Initial Capital:</strong> ${fmt(init_cap)}</p>
            <p><strong>Final Value:</strong> ${fmt(final_val)}</p>
            <p style="font-size:24px; color:{ret_color};">
                <strong>Total Return: {fmt(total_ret, 2)}%</strong>
            </p>

            <h3>📊 Comparison</h3>
            <p><strong>Buy &amp; Hold Return:</strong> {fmt(bh_ret, 2)}%</p>
            <p style="font-size:20px; color:{out_color};">
                <strong>Outperformance: {'+' if outperf >= 0 else ''}{fmt(outperf, 2)}%</strong>
            </p>
        </div>
        """

        # Equity curve chart
        equity_curve = backtest.get('equity_curve', [])
        if equity_curve:
            equity_df         = pd.DataFrame(equity_curve)
            equity_df['date'] = pd.to_datetime(equity_df['date'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['date'], y=equity_df['value'],
                mode='lines', name='Strategy Equity',
                line=dict(color='blue', width=2)
            ))

            bh_value = init_cap * (1 + bh_ret / 100)
            fig.add_hline(
                y=bh_value, line_dash="dash", line_color="green",
                annotation_text=f"Buy & Hold: ${bh_value:,.0f}"
            )
            fig.add_hline(
                y=init_cap, line_dash="dash", line_color="gray",
                annotation_text=f"Initial: ${init_cap:,.0f}"
            )

            fig.update_layout(
                title=f"{sym} — Strategy Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                showlegend=True
            )
            return result_html, fig

        return result_html, None

    except Exception as e:
        return f"❌ Error: {str(e)}", None


# ─────────────────────────────────────────────
#  Tab 3 — Raw Indicators
# ─────────────────────────────────────────────
def get_indicators_ui(symbol, period):
    """Get and display raw indicator values"""
    try:
        response = requests.get(
            f"{ALPHA_API}/indicators/{symbol}",
            params={"period": period},
            timeout=10
        )

        if response.status_code != 200:
            return "❌ Error fetching indicators"

        indicators = response.json()

        if indicators.get('status') == 'error':
            return f"❌ Error: {indicators.get('message','')}"

        ma  = indicators.get('moving_averages', {})
        rsi = indicators.get('rsi', {})
        mac = indicators.get('macd', {})
        bb  = indicators.get('bollinger_bands', {})

        result_html = f"""
        <div style="font-family: Arial, sans-serif; padding:10px;">
            <h2 style="color: #1f77b4;">📊 Technical Indicators: {indicators.get('symbol','')}</h2>
            <p><strong>Current Price:</strong> ${fmt(indicators.get('current_price'))}</p>

            <h3>📈 Moving Averages</h3>
            <table style="width:100%; border-collapse:collapse;">
                <tr><td><strong>SMA(20):</strong></td>
                    <td>${fmt(ma.get('sma_20'))}</td></tr>
                <tr><td><strong>SMA(50):</strong></td>
                    <td>${fmt(ma.get('sma_50'))}</td></tr>
                <tr><td><strong>SMA(200):</strong></td>
                    <td>${fmt(ma.get('sma_200'))}</td></tr>
                <tr><td><strong>EMA(12):</strong></td>
                    <td>${fmt(ma.get('ema_12'))}</td></tr>
                <tr><td><strong>EMA(26):</strong></td>
                    <td>${fmt(ma.get('ema_26'))}</td></tr>
            </table>

            <h3>📉 RSI</h3>
            <p><strong>Value:</strong> {fmt(rsi.get('value'))}</p>
            <p><strong>Interpretation:</strong> {rsi.get('interpretation','N/A')}</p>

            <h3>📊 MACD</h3>
            <p><strong>MACD Line:</strong>   {fmt(mac.get('macd_line'),   4)}</p>
            <p><strong>Signal Line:</strong>  {fmt(mac.get('signal_line'), 4)}</p>
            <p><strong>Histogram:</strong>    {fmt(mac.get('histogram'),   4)}</p>
            <p><strong>Signal:</strong>       {mac.get('signal','N/A')}</p>

            <h3>📏 Bollinger Bands</h3>
            <p><strong>Upper Band:</strong>  ${fmt(bb.get('upper'))}</p>
            <p><strong>Middle Band:</strong> ${fmt(bb.get('middle'))}</p>
            <p><strong>Lower Band:</strong>  ${fmt(bb.get('lower'))}</p>
            <p><strong>%B:</strong>           {fmt(bb.get('percent_b'))}</p>
            <p><strong>Bandwidth:</strong>    {fmt(bb.get('bandwidth'), 4)}</p>
        </div>
        """

        return result_html

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ─────────────────────────────────────────────
#  Gradio UI Layout
# ─────────────────────────────────────────────
with gr.Blocks(title="Alpha Signal Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 Alpha Signal Agent - Week 4")
    gr.Markdown("### Trading Signal Generation using Technical Analysis")
    gr.Markdown("*Integrated with Market Data Agent & Risk Management Agent*")

    with gr.Tabs():

        # ── Tab 1: Signal Generation ───────────────────────
        with gr.Tab("📊 Generate Signal"):
            with gr.Row():
                with gr.Column(scale=1):
                    symbol_input = gr.Dropdown(
                        choices=SYMBOLS, value="AAPL",
                        label="Stock Symbol"
                    )
                    portfolio_input = gr.Number(
                        value=100000, label="Portfolio Value ($)"
                    )
                    generate_btn = gr.Button(
                        "🎯 Generate Signal", variant="primary"
                    )

                with gr.Column(scale=2):
                    signal_output = gr.HTML(label="Signal Results")

            signal_chart  = gr.Plot(label="Technical Indicators")
            signal_status = gr.Textbox(label="Status", visible=False)

            generate_btn.click(
                fn=generate_signal_ui,
                inputs=[symbol_input, portfolio_input],
                outputs=[signal_output, signal_chart, signal_status]
            )

        # ── Tab 2: Backtest ────────────────────────────────
        with gr.Tab("📈 Backtest Strategy"):
            with gr.Row():
                with gr.Column(scale=1):
                    bt_symbol = gr.Dropdown(
                        choices=SYMBOLS, value="AAPL",
                        label="Stock Symbol"
                    )
                    bt_capital = gr.Number(
                        value=10000, label="Initial Capital ($)"
                    )
                    bt_period = gr.Dropdown(
                        choices=["1mo", "3mo", "6mo", "1y", "2y"],
                        value="1y", label="Period"
                    )
                    backtest_btn = gr.Button(
                        "🚀 Run Backtest", variant="primary"
                    )

                with gr.Column(scale=2):
                    backtest_output = gr.HTML(label="Backtest Results")

            backtest_chart = gr.Plot(label="Equity Curve")

            backtest_btn.click(
                fn=run_backtest_ui,
                inputs=[bt_symbol, bt_capital, bt_period],
                outputs=[backtest_output, backtest_chart]
            )

        # ── Tab 3: Raw Indicators ──────────────────────────
        with gr.Tab("🔍 Raw Indicators"):
            with gr.Row():
                with gr.Column(scale=1):
                    ind_symbol = gr.Dropdown(
                        choices=SYMBOLS, value="AAPL",
                        label="Stock Symbol"
                    )
                    ind_period = gr.Dropdown(
                        choices=["1mo", "3mo", "6mo", "1y", "2y"],
                        value="6mo", label="Period"
                    )
                    indicators_btn = gr.Button(
                        "📊 Get Indicators", variant="primary"
                    )

                with gr.Column(scale=2):
                    indicators_output = gr.HTML(label="Indicator Values")

            indicators_btn.click(
                fn=get_indicators_ui,
                inputs=[ind_symbol, ind_period],
                outputs=indicators_output
            )

    gr.Markdown("---")
    gr.Markdown(
        "**Note:** Make sure Market Data API (port 8000), "
        "Risk API (port 8001), and Alpha Signal API (port 8002) are running."
    )


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Alpha Signal Agent UI on port 7862...")
    print("Make sure these services are running:")
    print("  - Market Data API: http://localhost:8000")
    print("  - Risk Management API: http://localhost:8001")
    print("  - Alpha Signal API: http://localhost:8002")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        theme=gr.themes.Soft()
    )