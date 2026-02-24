import streamlit as st
import pandas as pd
from agents.market_data.agent import MarketDataAgent
from agents.market_data.storage import MarketDataStorage
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Market Data Agent",
    page_icon="📈",
    layout="wide"
)

# Initialize
@st.cache_resource
def get_agent():
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    return MarketDataAgent(symbols), MarketDataStorage()

agent, storage = get_agent()

# Title
st.title("📈 Market Data Agent - Live Demo")
st.markdown("### Week 2: Multi-Agent Portfolio Optimization System")
st.markdown("**Team:** Ashwini, Dibyendu Sarkar, Jyoti Ranjan Sethi")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Tracked Symbols")
    st.code("AAPL, GOOGL, MSFT, TSLA, AMZN")
    
    st.subheader("Features")
    st.markdown("""
    ✅ Real-time prices  
    ✅ Historical data  
    ✅ Price charts  
    ✅ Database storage  
    ✅ REST API
    """)
    
    st.divider()
    st.markdown("**Project:** Multi-Agent Portfolio Optimization")
    st.markdown("**Week:** 2 of 16")

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💰 Real-Time Prices", "📊 Historical Data", "💾 Database", "ℹ️ About"])

# Tab 1: Real-Time Prices
with tab1:
    st.header("Real-Time Market Data")
    
    if st.button("🔄 Fetch Current Prices", type="primary"):
        with st.spinner("Fetching data..."):
            data = agent.fetch_all_symbols()
            
            # Save to database
            for symbol, info in data.items():
                storage.save_realtime_data(info)
            
            # Display as metrics
            cols = st.columns(5)
            for i, (symbol, info) in enumerate(data.items()):
                with cols[i]:
                    change = ((info['price'] - info.get('open', info['price'])) / info.get('open', info['price']) * 100) if info.get('open') else 0
                    st.metric(
                        label=symbol,
                        value=f"${info['price']:.2f}",
                        delta=f"{change:.2f}%"
                    )
            
            # Display as table
            st.subheader("Detailed View")
            df = pd.DataFrame(data).T
            df = df[['symbol', 'price', 'open', 'high', 'low', 'volume', 'pe_ratio', 'market_cap']]
            st.dataframe(df, use_container_width=True)

# Tab 2: Historical Data
with tab2:
    st.header("Historical Price Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Select Stock", ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
    with col2:
        period = st.selectbox("Select Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'])
    
    if st.button("📈 Fetch Historical Data", type="primary"):
        with st.spinner(f"Fetching {period} of data for {symbol}..."):
            df = agent.fetch_historical_data(symbol, period)
            
            if df is not None and not df.empty:
                # Save to database
                storage.save_historical_data(symbol, df)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                with col2:
                    st.metric("Period High", f"${df['High'].max():.2f}")
                with col3:
                    st.metric("Period Low", f"${df['Low'].min():.2f}")
                with col4:
                    st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")
                
                # Price chart
                st.subheader("Price Chart")
                fig = go.Figure()
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
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                st.subheader("Volume Chart")
                fig_volume = go.Figure()
                fig_volume.add_trace(go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    marker_color='lightblue'
                ))
                fig_volume.update_layout(
                    title=f'{symbol} Trading Volume ({period})',
                    yaxis_title='Volume',
                    height=300
                )
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Data table
                with st.expander("📋 View Raw Data"):
                    st.dataframe(df, use_container_width=True)

# Tab 3: Database
with tab3:
    st.header("Database Statistics")
    
    if st.button("🔄 Refresh Database Stats", type="primary"):
        latest = storage.get_latest_prices()
        
        if not latest.empty:
            st.success(f"✅ Database contains {len(latest)} symbols")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Symbols", len(latest))
            with col2:
                st.metric("Last Update", latest['timestamp'].max())
            
            st.subheader("Latest Prices")
            st.dataframe(latest, use_container_width=True)
        else:
            st.warning("⚠️ No data in database yet. Fetch some prices first!")

# Tab 4: About
with tab4:
    st.header("About Market Data Agent")
    
    st.markdown("""
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
    - **UI:** Streamlit for web interface
    - **Backend:** Python with FastAPI
    
    ### Architecture:
    """)
    
    st.code("""
    yfinance API 
        ↓
    Market Data Agent 
        ↓
    Data Validator 
        ↓
    SQLite Database 
        ↓
    REST API → Other Agents
    """)
    
    st.markdown("""
    ### Next Week:
    We will build the **Risk Management Agent** that will use this Market Data Agent
    to calculate VaR, CVaR, and portfolio risk metrics.
    
    ---
    **Project:** Intelligent Multi-Agent Portfolio Optimization System  
    **Week:** 2 of 16  
    **Date:** February 2026
    """)

# Footer
st.divider()
st.caption("Multi-Agent Portfolio Optimization System | IIT Madras | 2026")