import gradio as gr
import pandas as pd
from agents.market_data.agent import MarketDataAgent
from agents.market_data.storage import MarketDataStorage
import plotly.graph_objects as go
from datetime import datetime

# Initialize
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
agent = MarketDataAgent(symbols)
storage = MarketDataStorage()

def fetch_current_prices():
    """Fetch and display current prices"""
    data = agent.fetch_all_symbols()
    
    # Save to database
    for symbol, info in data.items():
        storage.save_realtime_data(info)
    
    # Create DataFrame for display
    df = pd.DataFrame(data).T
    df = df[['symbol', 'price', 'open', 'high', 'low', 'volume', 'pe_ratio']]
    df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
    df['open'] = df['open'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    df['high'] = df['high'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    df['low'] = df['low'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    df['volume'] = df['volume'].apply(lambda x: f"{x:,}" if pd.notna(x) else "N/A")
    df['pe_ratio'] = df['pe_ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    return df

def get_historical_data(symbol, period):
    """Fetch historical data and create chart"""
    df = agent.fetch_historical_data(symbol, period)
    
    if df is None or df.empty:
        return None, "No data available"
    
    # Save to database
    storage.save_historical_data(symbol, df)
    
    # Create price chart
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        title=f'{symbol} Price History ({period})',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=500
    )
    
    # Create volume chart
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    fig_volume.update_layout(
        title=f'{symbol} Trading Volume ({period})',
        yaxis_title='Volume',
        xaxis_title='Date',
        template='plotly_white',
        height=300
    )
    
    # Create summary table
    summary = pd.DataFrame({
        'Metric': ['Current Price', 'Period High', 'Period Low', 'Average Volume', 'Total Days'],
        'Value': [
            f"${df['Close'].iloc[-1]:.2f}",
            f"${df['High'].max():.2f}",
            f"${df['Low'].min():.2f}",
            f"{df['Volume'].mean():,.0f}",
            f"{len(df)}"
        ]
    })
    
    return fig, fig_volume, summary

def get_database_stats():
    """Get statistics from database"""
    latest = storage.get_latest_prices()
    
    if latest.empty:
        return "No data in database yet. Fetch some prices first!"
    
    stats = f"""
    📊 **Database Statistics**
    
    - Total Symbols Tracked: {len(latest)}
    - Last Update: {latest['timestamp'].max()}
    - Symbols: {', '.join(latest['symbol'].tolist())}
    """
    
    return stats, latest[['symbol', 'price', 'timestamp']]

# Create Gradio Interface
with gr.Blocks(title="Market Data Agent - Demo", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 📈 Market Data Agent - Live Demo
    ### Week 2: Multi-Agent Portfolio Optimization System
    **Team:** Ashwini, Dibyendu Sarkar, Jyoti Ranjan Sethi
    """)
    
    with gr.Tab("💰 Real-Time Prices"):
        gr.Markdown("### Fetch current market data for tracked stocks")
        
        with gr.Row():
            fetch_btn = gr.Button("🔄 Fetch Current Prices", variant="primary", size="lg")
        
        price_output = gr.Dataframe(
            headers=['Symbol', 'Price', 'Open', 'High', 'Low', 'Volume', 'P/E Ratio'],
            label="Current Market Data"
        )
        
        fetch_btn.click(
            fn=fetch_current_prices,
            outputs=price_output
        )
    
    with gr.Tab("📊 Historical Data"):
        gr.Markdown("### View historical price charts and analysis")
        
        with gr.Row():
            symbol_input = gr.Dropdown(
                choices=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                value='AAPL',
                label="Select Stock"
            )
            period_input = gr.Dropdown(
                choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
                value='1mo',
                label="Select Period"
            )
        
        fetch_hist_btn = gr.Button("📈 Fetch Historical Data", variant="primary")
        
        with gr.Row():
            price_chart = gr.Plot(label="Price Chart")
        
        with gr.Row():
            volume_chart = gr.Plot(label="Volume Chart")
        
        summary_table = gr.Dataframe(label="Summary Statistics")
        
        fetch_hist_btn.click(
            fn=get_historical_data,
            inputs=[symbol_input, period_input],
            outputs=[price_chart, volume_chart, summary_table]
        )
    
    with gr.Tab("💾 Database Stats"):
        gr.Markdown("### View data stored in local SQLite database")
        
        refresh_btn = gr.Button("🔄 Refresh Database Stats", variant="primary")
        
        db_stats = gr.Markdown()
        db_data = gr.Dataframe(label="Latest Prices in Database")
        
        refresh_btn.click(
            fn=get_database_stats,
            outputs=[db_stats, db_data]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## Market Data Agent
        
        This is the **foundation agent** of our Multi-Agent Portfolio Optimization System.
        
        ### Features Implemented:
        1. ✅ Real-time price fetching from Yahoo Finance
        2. ✅ Historical data retrieval (1 day to 10 years)
        3. ✅ Data validation and quality checks
        4. ✅ SQLite database storage (thread-safe)
        5. ✅ REST API for other agents
        
        ### Technologies Used:
        - **Data Source:** yfinance (Yahoo Finance API)
        - **Database:** SQLite with thread-safe connections
        - **Charts:** Plotly for interactive visualizations
        - **UI:** Gradio for web interface
        - **Backend:** Python with FastAPI
        
        ### Next Week:
        We will build the **Risk Management Agent** that will use this Market Data Agent
        to calculate VaR, CVaR, and portfolio risk metrics.
        
        ---
        **Project:** Intelligent Multi-Agent Portfolio Optimization System  
        **Week:** 2 of 16  
        **Date:** February 2026
        """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates public link you can share
        server_name="0.0.0.0",
        server_port=7860
    )