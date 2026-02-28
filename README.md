# ⬡ Portfolio Intelligence System 
**https://9b7a49b4b6f6a4764e.gradio.live/**
<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.0-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-API-6001D2?style=for-the-badge&logo=yahoo&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)

<br/>


> **A multi-agent AI-powered platform for comprehensive portfolio risk analysis and market intelligence.**  
> Built as part of the IIT Madras curriculum — Week 3 of 16.

<br/>

```
╔══════════════════════════════════════════════════════════════╗
║   📈 Market Overview  |  🛡️ Risk Assessment  |  🎲 Monte Carlo  ║
║   📊 Historical Data  |  💥 Stress Testing  |  🔗 Correlation   ║
╚══════════════════════════════════════════════════════════════╝
```

</div>

---

## 👥 Team

<div align="center">

| Member | Role |
|:---:|:---:|
| **Ashwini** | Market Data Agent & Storage |
| **Dibyendu Sarkar** | Risk Management Agent & API |
| **Jyoti Ranjan Sethi** | UI/UX & Integration |

**Institution:** IIT Madras &nbsp;·&nbsp; **Program:** Multi-Agent Systems &nbsp;·&nbsp; **Year:** 2026

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Modules](#-modules)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [API Reference](#-api-reference)
- [Risk Metrics Explained](#-risk-metrics-explained)
- [Project Structure](#-project-structure)
- [Disclaimer](#️-disclaimer)

---

## 🧠 Overview

The **Portfolio Intelligence System** is a full-stack, multi-agent financial analytics platform. It combines real-time market data fetching, advanced risk calculations, stress testing, and Monte Carlo simulations — all accessible through a beautiful, interactive web dashboard.

The system is built on two cooperating agents:

- 🟦 **Market Data Agent** — fetches, validates, and stores real-time & historical price data
- 🟥 **Risk Management Agent (RiskIQ)** — computes portfolio risk metrics using the market data

---

## ✨ Features

### 📡 Market Overview
- Real-time price fetching for **AAPL, GOOGL, MSFT, TSLA, AMZN**
- Color-coded KPI cards (green = gain, red = loss)
- Interactive bar charts for prices and trading volume
- Sortable detailed price table with market cap and P/E ratios

### 📈 Historical Analysis
- Candlestick charts with **20-day Moving Average** overlay
- Cumulative return area chart (color changes with direction)
- Daily volume chart (green = up day, red = down day)
- Supports periods: `1mo` `3mo` `6mo` `1y` `2y` `5y`

### 🛡️ Risk Assessment
- **Value at Risk (VaR)** at 95% and 99% confidence levels
- **Conditional VaR (CVaR / Expected Shortfall)**
- **Maximum Drawdown** with underwater chart
- **Rolling 21-day Volatility** with risk limit line
- **Sharpe Ratio** with performance rating
- **Risk Score Gauge** (0–100 composite risk dial)
- Automated alerts when limits are breached

### 💥 Stress Testing
- **8 pre-defined market crash scenarios**
- Loss % and dollar loss visualized side by side
- Estimated recovery time per scenario
- Scenarios include: 2008 Crisis, COVID Crash, Bear Market, Flash Crash

### 🔗 Correlation Analysis
- Correlation matrix heatmap (red = negative, blue = positive)
- Diversification assessment badge
- Side-by-side cumulative return comparison chart
- Supports any user-defined set of symbols

### 🎲 Monte Carlo Simulation
- Up to **1000 simulated portfolio paths**
- 90% confidence band visualization
- Median, best-case (95th), and worst-case (5th) outcome cards
- Configurable: 21–504 simulation days, 100–1000 paths

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Yahoo Finance API                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Market Data Agent                          │
│   ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│   │ Data Fetcher│  │  Validator   │  │  SQLite Storage   │ │
│   └─────────────┘  └──────────────┘  └───────────────────┘ │
│                    REST API (Port 8000)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Risk Management Agent (RiskIQ)                 │
│   ┌──────┐ ┌──────┐ ┌────────┐ ┌────────┐ ┌─────────────┐ │
│   │ VaR  │ │ CVaR │ │Sharpe  │ │Max DD  │ │Stress Tests │ │
│   └──────┘ └──────┘ └────────┘ └────────┘ └─────────────┘ │
│                    REST API (Port 8001)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Gradio Dashboard UI                         │
│         Blue-White Professional Theme · Port 7862            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Modules

### `agents/market_data/`

| File | Description |
|---|---|
| `agent.py` | Core data fetching logic using `yfinance` |
| `storage.py` | Thread-safe SQLite storage for prices |
| `validator.py` | Data quality checks and cleaning |
| `api.py` | FastAPI REST endpoints (port 8000) |

### `agents/risk_management/`

| File | Description |
|---|---|
| `agent.py` | All risk metric calculations |
| `api.py` | FastAPI REST endpoints (port 8001) |

### Root Files

| File | Description |
|---|---|
| `risk_ui_enhanced.py` | Main Gradio dashboard (run this) |
| `app.py` | Original Gradio app (Week 2) |
| `streamlit_app.py` | Alternative Streamlit UI |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Data Source** | Yahoo Finance (`yfinance`) |
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Database** | SQLite (thread-safe) |
| **Risk Engine** | NumPy, Pandas |
| **Charts** | Plotly |
| **UI Framework** | Gradio 6.0 |
| **Fonts** | Inter, Space Grotesk (Google Fonts) |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-team/portfolio-intelligence-system.git
cd portfolio-intelligence-system
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r agents/requirement.txt
```

**Full dependency list:**

```
yfinance
pandas
numpy
gradio>=6.0
plotly
fastapi
uvicorn
requests
python-dotenv
streamlit
```

---

## 🚀 Running the App

### Option A — Dashboard only (recommended)

```bash
python risk_ui_enhanced.py
```

Open your browser at: **https://9b7a49b4b6f6a4764e.gradio.live/**  
A public share link will also be printed in the terminal.

---

### Option B — Full multi-agent setup

Run each agent in a separate terminal:

```bash
# Terminal 1 — Market Data Agent API
cd agents/market_data
python api.py
# Runs on http://localhost:8000](https://ad98868533aa94b755.gradio.live/
```

```bash
# Terminal 2 — Risk Management Agent API
cd agents/risk_management
python api.py
# Runs on http://localhost:8001
```

```bash
# Terminal 3 — Dashboard
python risk_ui_enhanced.py
# Runs on http://localhost:7862](https://9b7a49b4b6f6a4764e.gradio.live/
```

---

### Option C — Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## 🔌 API Reference

### Market Data Agent (`localhost:8000`)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/price/{symbol}` | GET | Current price for a symbol |
| `/prices` | GET | Prices for all tracked symbols |
| `/historical/{symbol}?period=1mo` | GET | Historical OHLCV data |
| `/latest` | GET | Latest prices from database |

### Risk Management Agent (`localhost:8001`)

| Endpoint | Method | Description |
|---|---|---|
| `/var/{symbol}?confidence=0.95` | GET | Value at Risk |
| `/cvar/{symbol}` | GET | Conditional VaR |
| `/volatility/{symbol}` | GET | Annual & daily volatility |
| `/sharpe/{symbol}` | GET | Sharpe Ratio |
| `/beta/{symbol}` | GET | Beta vs S&P 500 |
| `/drawdown/{symbol}` | GET | Maximum Drawdown |
| `/stress/{symbol}` | GET | Stress test results |
| `/assess` | POST | Full risk assessment |
| `/correlation?symbols=AAPL,MSFT` | GET | Correlation matrix |

---

## 📐 Risk Metrics Explained

| Metric | What it answers | Good value |
|---|---|---|
| **VaR 95%** | Max loss in a day with 95% confidence | < 2% |
| **VaR 99%** | Max loss in a day with 99% confidence | < 4% |
| **CVaR 95%** | Average loss when VaR is exceeded | < 3% |
| **Annual Volatility** | How much returns fluctuate per year | < 25% |
| **Max Drawdown** | Worst peak-to-trough decline | < 20% |
| **Sharpe Ratio** | Return earned per unit of risk | > 1.0 |
| **Beta** | Sensitivity to market movements | 0.8 – 1.2 |

### Sharpe Ratio Interpretation

```
> 3.0  →  🟢 Exceptional
2.0–3.0 →  🟢 Very Good
1.0–2.0 →  🔵 Good
0.5–1.0 →  🟡 Acceptable
< 0.5  →  🔴 Poor
< 0.0  →  🔴 Losing Money
```

---

## 📁 Project Structure

```
portfolio-intelligence-system/
│
├── 📂 agents/
│   ├── 📂 market_data/
│   │   ├── agent.py          # Core data fetching
│   │   ├── storage.py        # SQLite thread-safe storage
│   │   ├── validator.py      # Data quality checks
│   │   ├── api.py            # FastAPI REST endpoints
│   │   └── __init__.py
│   │
│   ├── 📂 risk_management/
│   │   ├── agent.py          # Risk calculations
│   │   ├── api.py            # FastAPI REST endpoints
│   │   └── __init__.py
│   │
│   ├── requirement.txt       # Dependencies
│   └── __init__.py
│
├── risk_ui_enhanced.py       # ⭐ Main dashboard (run this)
├── app.py                    # Gradio app (Week 2)
├── streamlit_app.py          # Streamlit alternative
├── market_data.db            # Auto-generated SQLite DB
└── README.md                 # You are here
```

---

## 📅 Project Timeline

| Week | Milestone | Status |
|---|---|---|
| Week 1 | Project Setup & Planning | ✅ Done |
| Week 2 | Market Data Agent | ✅ Done |
| Week 3 | Risk Management Agent + Dashboard | ✅ Done |
| Week 4–8 | Portfolio Optimizer Agent | 🔄 Upcoming |
| Week 9–12 | Sentiment & News Agent | 🔄 Upcoming |
| Week 13–16 | Full Integration & Deployment | 🔄 Upcoming |

---

## ⚠️ Disclaimer

> This platform is built **for educational purposes** as part of the IIT Madras curriculum on Multi-Agent Systems.
>
> It is **not financial advice**. All data is sourced from Yahoo Finance and may be delayed or inaccurate. Always consult a qualified financial advisor before making any investment decisions.
>
> Past performance is not indicative of future results.

---

<div align="center">

**⬡ Portfolio Intelligence System**  
IIT Madras · Multi-Agent Systems · 2026

Made with ❤️ by Ashwini, Dibyendu Sarkar & Jyoti Ranjan Sethi

</div>
