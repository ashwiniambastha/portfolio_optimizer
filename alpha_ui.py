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


def fetch_price_data(symbol, period="1y"):
    """Fetch historical price data"""
    try:
        response = requests.get(
            f"{MARKET_DATA_API}/historical/{symbol}",
            params={"period": period},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        return None
    except:
        return None


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
            return "❌ Error fetching signal", "", ""
        
        signal = response.json()
        
        if signal['status'] == 'error':
            return f"❌ Error: {signal['message']}", "", ""
        
        # Build result display
        result_html = f"""
        <div style="font-family: Arial, sans-serif;">
            <h2 style="color: #1f77b4;">📊 Trading Signal for {signal['symbol']}</h2>
            <p><strong>Current Price:</strong> ${signal['current_price']:.2f}</p>
            <p><strong>Timestamp:</strong> {signal['timestamp'][:19]}</p>
            
            <div style="background: {'#d4edda' if signal['action'] == 'BUY' else '#f8d7da' if signal['action'] == 'SELL' else '#fff3cd'}; 
                        padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h3 style="margin: 0;">🎯 {signal['recommendation']}</h3>
                <p style="font-size: 24px; margin: 5px 0;"><strong>Action: {signal['action']}</strong></p>
                <p><strong>Confidence:</strong> {signal['confidence']:.1f}%</p>
            </div>
            
            <h3>📈 Individual Signals</h3>
            <ul>
        """
        
        for indicator, value in signal['signals'].items():
            emoji = "🟢" if value > 0 else "🔴" if value < 0 else "⚪"
            result_html += f"<li>{emoji} <strong>{indicator}:</strong> {signal['explanations'][indicator]}</li>"
        
        result_html += """
            </ul>
            
            <h3>💰 Position Sizing</h3>
        """
        
        ps = signal['position_sizing']
        result_html += f"""
            <p><strong>Portfolio Value:</strong> ${ps['portfolio_value']:,.2f}</p>
            <p><strong>Recommended Position:</strong> {ps['position_size_pct']:.1f}% (${ps['actual_position_usd']:,.2f})</p>
            <p><strong>Shares to Buy:</strong> {ps['recommended_shares']} shares</p>
        """
        
        if signal['risk_metrics']['volatility']:
            result_html += f"""
            <p><strong>Volatility:</strong> {signal['risk_metrics']['volatility']*100:.2f}%</p>
            <p><strong>Risk Adjusted:</strong> {'Yes ✓' if signal['risk_metrics']['risk_adjusted'] else 'No'}</p>
            """
        
        result_html += "</div>"
        
        # Create indicator chart
        df = fetch_price_data(symbol, "6mo")
        if df is not None:
            df['Date'] = pd.to_datetime(df['Date'])
            
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price and MAs
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Calculate indicators for visualization
            close_series = pd.Series(df['Close'].values)
            
            # SMA
            sma_20 = close_series.rolling(20).mean()
            sma_50 = close_series.rolling(50).mean()
            
            fig.add_trace(
                go.Scatter(x=df['Date'], y=sma_20, name='SMA(20)', line=dict(color='orange', dash='dot')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['Date'], y=sma_50, name='SMA(50)', line=dict(color='green', dash='dot')),
                row=1, col=1
            )
            
            # RSI
            delta = close_series.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(x=df['Date'], y=rsi, name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            ema_12 = close_series.ewm(span=12, adjust=False).mean()
            ema_26 = close_series.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal_line = macd.ewm(span=9, adjust=False).mean()
            
            fig.add_trace(
                go.Scatter(x=df['Date'], y=macd, name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['Date'], y=signal_line, name='Signal', line=dict(color='red')),
                row=3, col=1
            )
            
            fig.update_layout(height=800, showlegend=True, title_text=f"{symbol} - Technical Indicators")
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            
            return result_html, fig, ""
        
        return result_html, None, ""
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None, ""


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
            return f"❌ Error: {backtest['message']}", None
        
        # Build result display
        result_html = f"""
        <div style="font-family: Arial, sans-serif;">
            <h2 style="color: #1f77b4;">📊 Backtest Results: {backtest['symbol']}</h2>
            <p><strong>Period:</strong> {backtest['period']}</p>
            <p><strong>Number of Trades:</strong> {backtest['num_trades']}</p>
            
            <h3>💰 Performance</h3>
            <p><strong>Initial Capital:</strong> ${backtest['initial_capital']:,.2f}</p>
            <p><strong>Final Value:</strong> ${backtest['final_value']:,.2f}</p>
            <p style="font-size: 24px; color: {'green' if backtest['total_return_pct'] > 0 else 'red'};">
                <strong>Total Return: {backtest['total_return_pct']:.2f}%</strong>
            </p>
            
            <h3>📊 Comparison</h3>
            <p><strong>Buy & Hold Return:</strong> {backtest['buy_hold_return_pct']:.2f}%</p>
            <p style="font-size: 20px; color: {'green' if backtest['outperformance'] > 0 else 'red'};">
                <strong>Outperformance: {backtest['outperformance']:+.2f}%</strong>
            </p>
        </div>
        """
        
        # Create equity curve chart
        if backtest.get('equity_curve'):
            equity_df = pd.DataFrame(backtest['equity_curve'])
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=equity_df['date'],
                y=equity_df['value'],
                mode='lines',
                name='Strategy Equity',
                line=dict(color='blue', width=2)
            ))
            
            # Add buy & hold comparison
            if backtest.get('buy_hold_return_pct'):
                bh_value = initial_capital * (1 + backtest['buy_hold_return_pct'] / 100)
                fig.add_hline(y=bh_value, line_dash="dash", line_color="green", 
                             annotation_text=f"Buy & Hold: ${bh_value:,.0f}")
            
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                         annotation_text=f"Initial: ${initial_capital:,.0f}")
            
            fig.update_layout(
                title=f"{backtest['symbol']} - Strategy Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                showlegend=True
            )
            
            return result_html, fig
        
        return result_html, None
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None


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
        
        result_html = f"""
        <div style="font-family: Arial, sans-serif;">
            <h2 style="color: #1f77b4;">📊 Technical Indicators: {indicators['symbol']}</h2>
            <p><strong>Current Price:</strong> ${indicators['current_price']:.2f}</p>
            
            <h3>📈 Moving Averages</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><strong>SMA(20):</strong></td><td>${indicators['moving_averages']['sma_20']:.2f}</td></tr>
                <tr><td><strong>SMA(50):</strong></td><td>${indicators['moving_averages']['sma_50']:.2f}</td></tr>
                <tr><td><strong>SMA(200):</strong></td><td>${indicators['moving_averages']['sma_200']:.2f}</td></tr>
                <tr><td><strong>EMA(12):</strong></td><td>${indicators['moving_averages']['ema_12']:.2f}</td></tr>
                <tr><td><strong>EMA(26):</strong></td><td>${indicators['moving_averages']['ema_26']:.2f}</td></tr>
            </table>
            
            <h3>📉 RSI</h3>
            <p><strong>Value:</strong> {indicators['rsi']['value']:.2f}</p>
            <p><strong>Interpretation:</strong> {indicators['rsi']['interpretation']}</p>
            
            <h3>📊 MACD</h3>
            <p><strong>MACD Line:</strong> {indicators['macd']['macd_line']:.4f}</p>
            <p><strong>Signal Line:</strong> {indicators['macd']['signal_line']:.4f}</p>
            <p><strong>Histogram:</strong> {indicators['macd']['histogram']:.4f}</p>
            <p><strong>Signal:</strong> {indicators['macd']['signal']}</p>
            
            <h3>📏 Bollinger Bands</h3>
            <p><strong>Upper Band:</strong> ${indicators['bollinger_bands']['upper']:.2f}</p>
            <p><strong>Middle Band:</strong> ${indicators['bollinger_bands']['middle']:.2f}</p>
            <p><strong>Lower Band:</strong> ${indicators['bollinger_bands']['lower']:.2f}</p>
            <p><strong>%B:</strong> {indicators['bollinger_bands']['percent_b']:.2f}</p>
            <p><strong>Bandwidth:</strong> {indicators['bollinger_bands']['bandwidth']:.4f}</p>
        </div>
        """
        
        return result_html
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Alpha Signal Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 Alpha Signal Agent - Week 4")
    gr.Markdown("### Trading Signal Generation using Technical Analysis")
    gr.Markdown("*Integrated with Market Data Agent & Risk Management Agent*")
    
    with gr.Tabs():
        # Tab 1: Signal Generation
        with gr.Tab("📊 Generate Signal"):
            with gr.Row():
                with gr.Column(scale=1):
                    symbol_input = gr.Dropdown(
                        choices=SYMBOLS,
                        value="AAPL",
                        label="Stock Symbol"
                    )
                    portfolio_input = gr.Number(
                        value=100000,
                        label="Portfolio Value ($)"
                    )
                    generate_btn = gr.Button("🎯 Generate Signal", variant="primary")
                
                with gr.Column(scale=2):
                    signal_output = gr.HTML(label="Signal Results")
            
            signal_chart = gr.Plot(label="Technical Indicators")
            signal_status = gr.Textbox(label="Status", visible=False)
            
            generate_btn.click(
                fn=generate_signal_ui,
                inputs=[symbol_input, portfolio_input],
                outputs=[signal_output, signal_chart, signal_status]
            )
        
        # Tab 2: Backtest
        with gr.Tab("📈 Backtest Strategy"):
            with gr.Row():
                with gr.Column(scale=1):
                    bt_symbol = gr.Dropdown(
                        choices=SYMBOLS,
                        value="AAPL",
                        label="Stock Symbol"
                    )
                    bt_capital = gr.Number(
                        value=10000,
                        label="Initial Capital ($)"
                    )
                    bt_period = gr.Dropdown(
                        choices=["1mo", "3mo", "6mo", "1y", "2y"],
                        value="1y",
                        label="Period"
                    )
                    backtest_btn = gr.Button("🚀 Run Backtest", variant="primary")
                
                with gr.Column(scale=2):
                    backtest_output = gr.HTML(label="Backtest Results")
            
            backtest_chart = gr.Plot(label="Equity Curve")
            
            backtest_btn.click(
                fn=run_backtest_ui,
                inputs=[bt_symbol, bt_capital, bt_period],
                outputs=[backtest_output, backtest_chart]
            )
        
        # Tab 3: Raw Indicators
        with gr.Tab("🔍 Raw Indicators"):
            with gr.Row():
                with gr.Column(scale=1):
                    ind_symbol = gr.Dropdown(
                        choices=SYMBOLS,
                        value="AAPL",
                        label="Stock Symbol"
                    )
                    ind_period = gr.Dropdown(
                        choices=["1mo", "3mo", "6mo", "1y", "2y"],
                        value="6mo",
                        label="Period"
                    )
                    indicators_btn = gr.Button("📊 Get Indicators", variant="primary")
                
                with gr.Column(scale=2):
                    indicators_output = gr.HTML(label="Indicator Values")
            
            indicators_btn.click(
                fn=get_indicators_ui,
                inputs=[ind_symbol, ind_period],
                outputs=indicators_output
            )
    
    gr.Markdown("---")
    gr.Markdown("**Note:** Make sure Market Data API (port 8000), Risk API (port 8001), and Alpha Signal API (port 8002) are running.")


if __name__ == "__main__":
    print("Starting Alpha Signal Agent UI on port 7862...")
    print("Make sure these services are running:")
    print("  - Market Data API: http://localhost:8000")
    print("  - Risk Management API: http://localhost:8001")
    print("  - Alpha Signal API: http://localhost:8002")
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)