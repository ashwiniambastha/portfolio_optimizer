"""
Portfolio Optimization Intelligence System
Unified Dashboard — Week 4
Combines: Market Data + Risk Management + Alpha Signal Agent
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf
import requests

# ─────────────────────────────────────────────
#  API endpoints (for Alpha Signal tab)
# ─────────────────────────────────────────────
MARKET_DATA_API = "http://localhost:8000"
RISK_API        = "http://localhost:8001"
ALPHA_API       = "http://localhost:8002"

SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

# ─────────────────────────────────────────────
#  Color Palette
# ─────────────────────────────────────────────
BG_PAGE      = "#f0f4fb"
BG_CARD      = "#ffffff"
BG_HEADER    = "#1a3a6b"
BORDER       = "#d0dff0"
BLUE_PRIMARY = "#1a5fca"
BLUE_DARK    = "#0d2d6b"
BLUE_LIGHT   = "#e8f0fe"
GREEN        = "#0d9c5b"
RED          = "#e03131"
GOLD         = "#d4940a"
TEXT_DARK    = "#0d1f3c"
TEXT_MED     = "#3a5080"
TEXT_LIGHT   = "#6b83a8"
WHITE        = "#ffffff"

# ─────────────────────────────────────────────
#  Plotly theme
# ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor="#f8faff",
    font=dict(family="Inter, sans-serif", color=TEXT_DARK, size=12),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER,
                borderwidth=1, font=dict(color=TEXT_DARK)),
    margin=dict(l=55, r=30, t=55, b=45),
)

AXIS_STYLE = dict(
    gridcolor="#e2eaf5",
    zerolinecolor="#c5d5ea",
    tickfont=dict(color=TEXT_LIGHT),
    linecolor=BORDER,
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #f0f4fb !important;
    font-family: 'Inter', sans-serif !important;
    color: #0d1f3c !important;
}

.app-header {
    background: linear-gradient(135deg, #0d2d6b 0%, #1a5fca 60%, #2979e8 100%);
    padding: 36px 40px 28px;
    text-align: center;
    border-bottom: 4px solid #d4940a;
    box-shadow: 0 4px 20px rgba(26,60,107,0.18);
}
.app-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.app-subtitle {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.75) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.app-team {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.55) !important;
    margin-top: 10px;
    letter-spacing: 1.5px;
}
.header-accent {
    width: 60px;
    height: 3px;
    background: #d4940a;
    margin: 12px auto;
    border-radius: 2px;
}

div[role="tablist"] {
    background: #ffffff !important;
    border-bottom: 2px solid #d0dff0 !important;
    padding: 0 16px !important;
    box-shadow: 0 2px 8px rgba(26,60,107,0.06) !important;
}
div[role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #6b83a8 !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 13px 16px !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
div[role="tab"]:hover { color: #1a5fca !important; background: #e8f0fe !important; }
div[role="tab"][aria-selected="true"] {
    color: #1a5fca !important;
    border-bottom: 3px solid #1a5fca !important;
    background: #e8f0fe !important;
    font-weight: 600 !important;
}

.metric-grid   { display: grid; gap: 14px; margin-bottom: 22px; }
.metric-grid-5 { grid-template-columns: repeat(5, 1fr); }
.metric-grid-4 { grid-template-columns: repeat(4, 1fr); }
.metric-grid-3 { grid-template-columns: repeat(3, 1fr); }
.metric-grid-2 { grid-template-columns: repeat(2, 1fr); }

.kpi-card {
    background: #ffffff;
    border: 1px solid #d0dff0;
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(26,60,107,0.07);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1a5fca, #2979e8);
}
.kpi-card.green::before { background: linear-gradient(90deg, #0d9c5b, #22c55e); }
.kpi-card.red::before   { background: linear-gradient(90deg, #e03131, #f87171); }
.kpi-card.gold::before  { background: linear-gradient(90deg, #d4940a, #f59e0b); }
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(26,60,107,0.12); }

.kpi-label {
    font-size: 0.70rem;
    font-weight: 600;
    color: #6b83a8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 9px;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0d1f3c;
    line-height: 1.1;
}
.kpi-value.up   { color: #0d9c5b; }
.kpi-value.down { color: #e03131; }
.kpi-value.warn { color: #d4940a; }
.kpi-delta { font-size: 0.74rem; font-weight: 500; margin-top: 5px; color: #6b83a8; }
.kpi-delta.up   { color: #0d9c5b; }
.kpi-delta.down { color: #e03131; }

.section-header { margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid #d0dff0; }
.section-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem; font-weight: 600; color: #0d2d6b;
}
.section-sub { font-size: 0.76rem; color: #6b83a8; margin-top: 3px; }

.badge { display:inline-block; padding:4px 14px; border-radius:20px; font-size:0.71rem; font-weight:600; letter-spacing:.8px; text-transform:uppercase; }
.badge-ok    { background:#d1fae5; color:#065f46; border:1px solid #6ee7b7; }
.badge-alert { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }
.badge-warn  { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }
.badge-info  { background:#e8f0fe; color:#1a5fca; border:1px solid #a8c4f8; }
.badge-buy   { background:#d1fae5; color:#065f46; border:1px solid #6ee7b7; font-size:1rem; padding:8px 24px; }
.badge-sell  { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; font-size:1rem; padding:8px 24px; }
.badge-hold  { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; font-size:1rem; padding:8px 24px; }

.info-banner    { background:#e8f0fe; border:1px solid #a8c4f8; border-left:4px solid #1a5fca; border-radius:8px; padding:12px 18px; font-size:.82rem; color:#1a3a6b; margin-bottom:14px; }
.alert-banner   { background:#fee2e2; border:1px solid #fca5a5; border-left:4px solid #e03131; border-radius:8px; padding:12px 18px; font-size:.82rem; color:#7f1d1d; margin-bottom:14px; }
.success-banner { background:#d1fae5; border:1px solid #6ee7b7; border-left:4px solid #0d9c5b; border-radius:8px; padding:12px 18px; font-size:.82rem; color:#064e3b; margin-bottom:14px; }

.signal-box {
    border-radius: 12px;
    padding: 20px 24px;
    margin: 14px 0;
    border: 1px solid #d0dff0;
}
.signal-box.buy  { background: #d1fae5; border-left: 5px solid #0d9c5b; }
.signal-box.sell { background: #fee2e2; border-left: 5px solid #e03131; }
.signal-box.hold { background: #fef3c7; border-left: 5px solid #d4940a; }

button, .gr-button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #1a5fca, #2979e8) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 3px 12px rgba(26,95,202,0.28) !important;
}
.gr-button-primary:hover, button.primary:hover {
    background: linear-gradient(135deg, #0d4fb5, #1a5fca) !important;
    box-shadow: 0 5px 18px rgba(26,95,202,0.38) !important;
    transform: translateY(-1px) !important;
}

input, select, textarea {
    background: #ffffff !important;
    border: 1.5px solid #c5d5ea !important;
    border-radius: 8px !important;
    color: #0d1f3c !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
input:focus, select:focus {
    border-color: #1a5fca !important;
    box-shadow: 0 0 0 3px rgba(26,95,202,0.12) !important;
    outline: none !important;
}

label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: #3a5080 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    background: #ffffff !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #d0dff0 !important;
    font-size: 0.83rem !important;
    box-shadow: 0 2px 10px rgba(26,60,107,0.06) !important;
}
th {
    background: #1a3a6b !important;
    color: #ffffff !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 12px 16px !important;
}
td { color: #0d1f3c !important; padding: 10px 16px !important; border-bottom: 1px solid #edf2fb !important; background: #ffffff !important; }
tr:nth-child(even) td { background: #f5f8fe !important; }
tr:hover td { background: #e8f0fe !important; }

.gr-accordion { background: #ffffff !important; border: 1px solid #d0dff0 !important; border-radius: 10px !important; }

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { font-family: 'Space Grotesk', sans-serif !important; color: #0d2d6b !important; }
.gr-markdown p, .gr-markdown li { color: #0d1f3c !important; line-height: 1.7 !important; }
.gr-markdown strong { color: #1a5fca !important; }
.gr-markdown code { background: #e8f0fe !important; color: #1a5fca !important; border-radius: 4px !important; padding: 2px 7px !important; }

input[type="range"] { accent-color: #1a5fca !important; }

.app-footer {
    background: #0d2d6b;
    color: rgba(255,255,255,0.6);
    text-align: center;
    padding: 18px;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 40px;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0f4fb; }
::-webkit-scrollbar-thumb { background: #c5d5ea; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #1a5fca; }
"""

# ─────────────────────────────────────────────
#  HTML helpers
# ─────────────────────────────────────────────
def kpi_card(label, value, delta="", color="blue"):
    cls_map  = {"green": "up", "red": "down", "gold": "warn"}
    val_cls  = cls_map.get(color, "")
    card_cls = color if color in ["green", "red", "gold"] else ""
    dlt_cls  = "up" if ("▲" in delta or "+" in delta) else "down" if ("▼" in delta or "-" in delta) else ""
    d_html   = f'<div class="kpi-delta {dlt_cls}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card {card_cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {val_cls}">{value}</div>
        {d_html}
    </div>"""

def section_header(icon, title, subtitle=""):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section-header"><div class="section-title">{icon} {title}</div>{sub}</div>'

def banner(msg, kind="info"):
    return f'<div class="{kind}-banner">{msg}</div>'

def fmt(val, decimals=2):
    if val is None:
        return "N/A"
    try:
        return f"{val:.{decimals}f}"
    except:
        return "N/A"

# ─────────────────────────────────────────────
#  Data helpers — direct Yahoo Finance
# ─────────────────────────────────────────────
def get_realtime(symbols):
    results = {}
    for sym in symbols:
        try:
            info  = yf.Ticker(sym).info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
            open_ = info.get('regularMarketOpen') or price
            results[sym] = {
                'symbol': sym, 'price': price, 'open': open_,
                'high': info.get('dayHigh', 0), 'low': info.get('dayLow', 0),
                'volume': info.get('volume', 0), 'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'change_pct': ((price - open_) / open_ * 100) if open_ else 0,
            }
        except Exception:
            results[sym] = {k: 0 for k in ['price','open','high','low','volume','market_cap','pe_ratio','change_pct']}
            results[sym]['symbol'] = sym
    return results

def get_historical(symbol, period="1y"):
    try:
        df = yf.Ticker(symbol).history(period=period)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def get_returns(symbol, period="1y"):
    df = get_historical(symbol, period)
    if df.empty:
        return pd.Series(dtype=float)
    return df['Close'].pct_change().dropna()

def compute_risk(returns, portfolio_value, rf=0.04):
    if returns.empty:
        return {}
    daily_vol  = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    total_ret  = (1 + returns).prod() - 1
    years      = max(len(returns) / 252, 0.01)
    ann_ret    = (1 + total_ret) ** (1 / years) - 1
    sharpe     = (ann_ret - rf) / annual_vol if annual_vol else 0
    var_95     = abs(np.percentile(returns, 5))
    var_99     = abs(np.percentile(returns, 1))
    tail       = returns[returns <= -var_95]
    cvar_95    = abs(tail.mean()) if len(tail) else var_95
    cum        = (1 + returns).cumprod()
    peak       = cum.cummax()
    dd         = (cum - peak) / peak
    max_dd     = abs(dd.min())
    return dict(
        annual_vol=annual_vol, daily_vol=daily_vol, ann_ret=ann_ret, sharpe=sharpe,
        var_95=var_95, var_99=var_99, cvar_95=cvar_95,
        var_95_usd=var_95*portfolio_value, var_99_usd=var_99*portfolio_value,
        cvar_95_usd=cvar_95*portfolio_value, max_dd=max_dd, drawdown=dd, returns=returns,
    )

def sharpe_label(s):
    if s > 3:   return "Exceptional", "green"
    if s > 2:   return "Very Good",   "green"
    if s > 1:   return "Good",        "blue"
    if s > 0.5: return "Acceptable",  "gold"
    if s > 0:   return "Poor",        "gold"
    return "Losing Money", "red"

def apply_theme(fig, title_text=None, yaxis_title=None, xaxis_title=None, extra=None):
    layout = dict(**PLOTLY_THEME)
    layout['xaxis'] = dict(**AXIS_STYLE)
    layout['yaxis'] = dict(**AXIS_STYLE)
    if title_text:
        layout['title'] = dict(text=title_text, font=dict(color="#1a3a6b", size=14, family="Inter, sans-serif"))
    if yaxis_title:
        layout['yaxis']['title'] = yaxis_title
    if xaxis_title:
        layout['xaxis']['title'] = xaxis_title
    if extra:
        layout.update(extra)
    fig.update_layout(**layout)
    return fig

# ─────────────────────────────────────────────
#  Alpha Signal helpers — via API
# ─────────────────────────────────────────────
def fetch_alpha_signal(symbol, portfolio_value):
    try:
        r = requests.get(f"{ALPHA_API}/signal/{symbol}",
                         params={"portfolio_value": portfolio_value}, timeout=15)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def fetch_alpha_backtest(symbol, initial_capital, period):
    try:
        r = requests.get(f"{ALPHA_API}/backtest/{symbol}",
                         params={"initial_capital": initial_capital, "period": period},
                         timeout=20)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def fetch_alpha_indicators(symbol, period):
    try:
        r = requests.get(f"{ALPHA_API}/indicators/{symbol}",
                         params={"period": period}, timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# ═══════════════════════════════════════════════
#  TAB 1 — Market Overview
# ═══════════════════════════════════════════════
def render_market_overview():
    data = get_realtime(SYMBOLS)
    ts   = datetime.now().strftime('%d %b %Y  %H:%M:%S')

    cards = '<div class="metric-grid metric-grid-5">'
    for sym, d in data.items():
        chg  = d.get('change_pct', 0)
        sign = "▲" if chg >= 0 else "▼"
        col  = "green" if chg >= 0 else "red"
        cards += kpi_card(sym, f"${d['price']:.2f}" if d['price'] else "—",
                          f"{sign} {abs(chg):.2f}%", col)
    cards += "</div>"

    prices  = {s: d['price']      for s, d in data.items() if d['price']}
    changes = {s: d['change_pct'] for s, d in data.items()}
    bcolors = [GREEN if changes.get(s, 0) >= 0 else RED for s in prices]

    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(
        x=list(prices.keys()), y=list(prices.values()),
        marker=dict(color=bcolors, line=dict(color='white', width=1)),
        text=[f"${v:.2f}" for v in prices.values()],
        textposition='outside', textfont=dict(size=11, color=TEXT_DARK),
        hovertemplate="<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>",
    ))
    apply_theme(fig_p, title_text="Current Stock Prices (USD)", yaxis_title="Price ($)",
                extra={"showlegend": False})

    vols = {s: d.get('volume', 0) for s, d in data.items()}
    fig_v = go.Figure()
    fig_v.add_trace(go.Bar(
        x=list(vols.keys()), y=list(vols.values()),
        marker=dict(color=list(vols.values()),
                    colorscale=[[0, BLUE_LIGHT], [1, BLUE_PRIMARY]],
                    showscale=False, line=dict(color='white', width=1)),
        text=[f"{v/1e6:.1f}M" for v in vols.values()],
        textposition='outside', textfont=dict(size=11, color=TEXT_DARK),
        hovertemplate="<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>",
    ))
    apply_theme(fig_v, title_text="Trading Volume", yaxis_title="Volume",
                extra={"showlegend": False})

    rows = []
    for s, d in data.items():
        chg = d.get('change_pct', 0)
        rows.append({
            'Symbol': s,
            'Price ($)': f"${d['price']:.2f}" if d['price'] else "—",
            'Open ($)':  f"${d['open']:.2f}"  if d['open']  else "—",
            'High ($)':  f"${d['high']:.2f}"  if d['high']  else "—",
            'Low ($)':   f"${d['low']:.2f}"   if d['low']   else "—",
            'Volume':    f"{d['volume']/1e6:.1f}M" if d['volume'] else "—",
            'Mkt Cap':   f"${d['market_cap']/1e12:.2f}T" if d.get('market_cap') else "—",
            'P/E':       f"{d['pe_ratio']:.1f}" if d.get('pe_ratio') else "—",
            'Change':    f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%",
        })

    return (cards, fig_p, fig_v, pd.DataFrame(rows),
            banner(f"✅ Data refreshed at {ts}", "success"))


# ═══════════════════════════════════════════════
#  TAB 2 — Historical Analysis
# ═══════════════════════════════════════════════
def render_historical(symbol, period):
    df = get_historical(symbol, period)
    if df.empty:
        return None, None, None, banner("⚠ No data available.", "alert")

    fig_c = go.Figure()
    fig_c.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing=dict(line=dict(color=GREEN), fillcolor="rgba(13,156,91,0.22)"),
        decreasing=dict(line=dict(color=RED),   fillcolor="rgba(224,49,49,0.22)"),
        name="Price",
    ))
    ma20 = df['Close'].rolling(20).mean()
    fig_c.add_trace(go.Scatter(x=df.index, y=ma20, name="MA 20",
                               line=dict(color=BLUE_PRIMARY, width=1.8, dash='dot')))
    apply_theme(fig_c, title_text=f"{symbol} — Candlestick Chart ({period})",
                yaxis_title="Price (USD)", extra={"xaxis_rangeslider_visible": False})

    returns = df['Close'].pct_change().dropna()
    cum_ret = (1 + returns).cumprod() - 1
    col_ret = GREEN if cum_ret.iloc[-1] >= 0 else RED
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=cum_ret.index, y=cum_ret * 100, fill='tozeroy',
        fillcolor="rgba(13,156,91,0.10)" if cum_ret.iloc[-1] >= 0 else "rgba(224,49,49,0.10)",
        line=dict(color=col_ret, width=2.2), name="Cumulative Return",
        hovertemplate="%{x|%b %d, %Y}<br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig_r.add_hline(y=0, line=dict(color=TEXT_LIGHT, dash='dash', width=1))
    apply_theme(fig_r, title_text="Cumulative Return (%)", yaxis_title="Return (%)")

    vcols = [GREEN if c >= o else RED for c, o in zip(df['Close'], df['Open'])]
    fig_v = go.Figure()
    fig_v.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vcols, name="Volume",
                           hovertemplate="%{x|%b %d}<br>Vol: %{y:,.0f}<extra></extra>"))
    apply_theme(fig_v, title_text="Volume (Green = Up Day, Red = Down Day)", yaxis_title="Volume")

    total = cum_ret.iloc[-1] * 100
    sign  = "▲" if total >= 0 else "▼"
    col   = "green" if total >= 0 else "red"
    stats = f"""<div class="metric-grid metric-grid-4">
        {kpi_card("Current Price", f"${df['Close'].iloc[-1]:.2f}")}
        {kpi_card("Period High",   f"${df['High'].max():.2f}", color="green")}
        {kpi_card("Period Low",    f"${df['Low'].min():.2f}",  color="red")}
        {kpi_card("Total Return",  f"{sign} {abs(total):.2f}%", color=col)}
    </div>"""

    return fig_c, fig_r, fig_v, stats


# ═══════════════════════════════════════════════
#  TAB 3 — Risk Assessment
# ═══════════════════════════════════════════════
def render_risk(symbol, portfolio_value):
    returns = get_returns(symbol, "1y")
    if returns.empty:
        return banner("⚠ Could not fetch data.", "alert"), None, None, None, None

    m = compute_risk(returns, portfolio_value)
    slabel, scol = sharpe_label(m['sharpe'])
    risk_ok = m['annual_vol'] < 0.30 and m['max_dd'] < 0.20 and m['sharpe'] > 1.0
    badge   = ('<span class="badge badge-ok">✓ Within Limits</span>' if risk_ok else
               '<span class="badge badge-alert">⚠ Risk Alert</span>')

    kpi_html = f"""
    {section_header("🛡️", f"Risk Assessment — {symbol}",
                    f"Portfolio Value: ${portfolio_value:,.0f}  |  {len(returns)} trading days")}
    <div style="margin-bottom:14px">{badge}</div>
    <div class="metric-grid metric-grid-4" style="margin-bottom:14px">
        {kpi_card("VaR 95%",    f"{m['var_95']:.2%}",    f"−${m['var_95_usd']:,.0f}/day", "red")}
        {kpi_card("VaR 99%",    f"{m['var_99']:.2%}",    f"−${m['var_99_usd']:,.0f}/day", "red")}
        {kpi_card("CVaR 95%",   f"{m['cvar_95']:.2%}",   "Expected Shortfall",            "red")}
        {kpi_card("Annual Vol", f"{m['annual_vol']:.2%}", f"Daily: {m['daily_vol']:.2%}",
                  "gold" if m['annual_vol'] > 0.25 else "blue")}
    </div>
    <div class="metric-grid metric-grid-4">
        {kpi_card("Max Drawdown",  f"{m['max_dd']:.2%}", "Peak-to-Trough",
                  "red" if m['max_dd'] > 0.20 else "gold")}
        {kpi_card("Sharpe Ratio",  f"{m['sharpe']:.2f}", slabel, scol)}
        {kpi_card("Annual Return", f"{m['ann_ret']:.2%}",
                  "▲ Positive" if m['ann_ret'] >= 0 else "▼ Negative",
                  "green" if m['ann_ret'] >= 0 else "red")}
        {kpi_card("Data Points",   str(len(returns)), "Trading Days")}
    </div>"""

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=returns * 100, nbinsx=55,
        marker=dict(color=BLUE_PRIMARY, opacity=0.75, line=dict(color='white', width=0.5)),
        name="Daily Returns",
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    fig_dist.add_vline(x=-m['var_95']*100, line=dict(color=GOLD, dash='dash', width=2),
                       annotation=dict(text="VaR 95%", font=dict(color=GOLD, size=10)))
    fig_dist.add_vline(x=-m['var_99']*100, line=dict(color=RED, dash='dash', width=2),
                       annotation=dict(text="VaR 99%", font=dict(color=RED, size=10)))
    apply_theme(fig_dist, title_text="Return Distribution with VaR Lines",
                xaxis_title="Daily Return (%)", yaxis_title="Frequency")

    dd = m['drawdown']
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd * 100, fill='tozeroy',
        fillcolor="rgba(224,49,49,0.13)",
        line=dict(color=RED, width=1.8), name="Drawdown %",
        hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig_dd.add_hline(y=-m['max_dd']*100, line=dict(color=GOLD, dash='dot', width=1.5),
                     annotation=dict(text=f"Max DD {m['max_dd']:.2%}", font=dict(color=GOLD, size=10)))
    apply_theme(fig_dd, title_text="Underwater Drawdown Chart", yaxis_title="Drawdown (%)")

    rv = returns.rolling(21).std() * np.sqrt(252) * 100
    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(
        x=rv.index, y=rv, fill='tozeroy', fillcolor="rgba(26,95,202,0.09)",
        line=dict(color=BLUE_PRIMARY, width=2), name="21-day Vol",
        hovertemplate="%{x|%b %d, %Y}<br>Vol: %{y:.2f}%<extra></extra>",
    ))
    fig_rv.add_hline(y=30, line=dict(color=RED, dash='dash', width=1.3),
                     annotation=dict(text="Risk Limit 30%", font=dict(color=RED, size=10)))
    apply_theme(fig_rv, title_text="Rolling 21-Day Annualised Volatility", yaxis_title="Volatility (%)")

    risk_score = min(100, m['annual_vol']/0.5*40 + m['max_dd']/0.5*40 + max(0, 1-m['sharpe'])*20)
    gcol = GREEN if risk_score < 40 else GOLD if risk_score < 70 else RED
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title=dict(text="RISK SCORE", font=dict(family="Inter", size=13, color=TEXT_MED)),
        number=dict(font=dict(family="Space Grotesk", size=38, color=gcol)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=TEXT_LIGHT,
                      tickfont=dict(family="Inter", size=10, color=TEXT_LIGHT)),
            bar=dict(color=gcol, thickness=0.25),
            bgcolor=BG_CARD, borderwidth=1, bordercolor=BORDER,
            steps=[dict(range=[0, 40],   color="rgba(13,156,91,0.08)"),
                   dict(range=[40, 70],  color="rgba(212,148,10,0.08)"),
                   dict(range=[70, 100], color="rgba(224,49,49,0.08)")],
            threshold=dict(line=dict(color=gcol, width=3), thickness=0.75, value=risk_score),
        ),
    ))
    fig_g.update_layout(paper_bgcolor=BG_CARD, font=dict(family="Inter", color=TEXT_DARK),
                        height=260, margin=dict(l=30, r=30, t=60, b=20))

    return kpi_html, fig_dist, fig_dd, fig_rv, fig_g


# ═══════════════════════════════════════════════
#  TAB 4 — Stress Testing
# ═══════════════════════════════════════════════
SCENARIOS = {
    "Moderate −5%":     -0.05,
    "Correction −10%":  -0.10,
    "Bear Market −20%": -0.20,
    "Severe −30%":      -0.30,
    "2008 Crisis −50%": -0.50,
    "COVID −35%":       -0.35,
    "Flash Crash −10%": -0.10,
    "Rate Shock −15%":  -0.15,
}

def render_stress(symbol, portfolio_value):
    returns   = get_returns(symbol, "1y")
    avg_daily = returns.mean() if not returns.empty else 0.0003

    rows, pcts, dloss, labels = [], [], [], []
    for name, shock in SCENARIOS.items():
        shocked  = portfolio_value * (1 + shock)
        loss     = portfolio_value - shocked
        days_rec = abs(shock) / avg_daily if avg_daily > 0 else float('inf')
        yrs_rec  = round(days_rec/252, 1) if days_rec != float('inf') else None
        rows.append({'Scenario': name, 'Market Shock': f"{shock:.0%}",
                     'Portfolio After': f"${shocked:,.0f}", 'Loss Amount': f"${loss:,.0f}",
                     'Recovery (yrs)': str(yrs_rec) if yrs_rec else "N/A"})
        pcts.append(shock * 100)
        dloss.append(loss)
        labels.append(name)

    def sev(l):
        if l < -30: return RED
        if l < -15: return GOLD
        return BLUE_PRIMARY

    fig_pct = go.Figure()
    fig_pct.add_trace(go.Bar(
        x=labels, y=pcts,
        marker=dict(color=[sev(l) for l in pcts], line=dict(color='white', width=1)),
        text=[f"{l:.0f}%" for l in pcts], textposition='outside',
        textfont=dict(size=10, color=TEXT_DARK),
        hovertemplate="<b>%{x}</b><br>Loss: %{y:.1f}%<extra></extra>",
    ))
    apply_theme(fig_pct, title_text="Portfolio Loss % by Scenario", yaxis_title="Loss (%)",
                extra={"yaxis": dict(**AXIS_STYLE, range=[min(pcts)*1.3, 5])})

    fig_usd = go.Figure()
    fig_usd.add_trace(go.Bar(
        x=labels, y=dloss,
        marker=dict(color=dloss, colorscale=[[0, BLUE_LIGHT], [0.5, GOLD], [1, RED]],
                    showscale=False, line=dict(color='white', width=1)),
        text=[f"${l:,.0f}" for l in dloss], textposition='outside',
        textfont=dict(size=10, color=TEXT_DARK),
        hovertemplate="<b>%{x}</b><br>Loss: $%{y:,.0f}<extra></extra>",
    ))
    apply_theme(fig_usd, title_text="Dollar Loss by Scenario", yaxis_title="Loss ($)")

    return fig_pct, fig_usd, pd.DataFrame(rows)


# ═══════════════════════════════════════════════
#  TAB 5 — Correlation
# ═══════════════════════════════════════════════
def render_correlation(symbols_str):
    syms = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
    if len(syms) < 2:
        return None, banner("⚠ Enter at least 2 comma-separated symbols.", "alert"), None

    all_ret = {}
    for s in syms:
        r = get_returns(s, "1y")
        if not r.empty:
            all_ret[s] = r

    if len(all_ret) < 2:
        return None, banner("⚠ Could not fetch data for enough symbols.", "alert"), None

    df_ret = pd.DataFrame(all_ret).dropna()
    corr   = df_ret.corr()

    fig_h = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0, RED], [0.5, "#f0f4fb"], [1, BLUE_PRIMARY]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(family="Inter", size=13, color=TEXT_DARK),
        hovertemplate="<b>%{x} vs %{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(family="Inter", color=TEXT_MED),
                      title=dict(text="r", font=dict(color=TEXT_MED))),
    ))
    apply_theme(fig_h, title_text="Correlation Matrix — 1 Year Daily Returns")

    cum_df  = (1 + df_ret).cumprod() - 1
    palette = [BLUE_PRIMARY, GREEN, GOLD, RED, "#7c3aed", "#db2777", "#0891b2"]
    fig_cr  = go.Figure()
    for i, col in enumerate(cum_df.columns):
        fig_cr.add_trace(go.Scatter(
            x=cum_df.index, y=cum_df[col]*100, name=col,
            line=dict(color=palette[i % len(palette)], width=2.2),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %d}}<br>Return: %{{y:.2f}}%<extra></extra>",
        ))
    apply_theme(fig_cr, title_text="Cumulative Returns Comparison (%)", yaxis_title="Return (%)")

    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    if avg_corr < 0.5:
        msg, kind = f"✅ Well Diversified — avg correlation: {avg_corr:.3f}", "success"
    elif avg_corr < 0.7:
        msg, kind = f"⚠ Moderately Correlated — avg correlation: {avg_corr:.3f}", "info"
    else:
        msg, kind = f"🔴 Highly Correlated — Low Diversification Benefit (r={avg_corr:.3f})", "alert"

    return fig_h, banner(msg, kind), fig_cr


# ═══════════════════════════════════════════════
#  TAB 6 — Monte Carlo
# ═══════════════════════════════════════════════
def render_monte_carlo(symbol, portfolio_value, days, sims):
    days, sims = int(days), int(sims)
    returns = get_returns(symbol, "1y")
    if returns.empty:
        return None, banner("⚠ Could not fetch data.", "alert")

    mu, sigma = returns.mean(), returns.std()
    np.random.seed(42)
    sim_rets  = np.random.normal(mu, sigma, (days, sims))
    sim_paths = portfolio_value * np.exp(np.cumsum(np.log(1 + sim_rets), axis=0))
    final_vals = sim_paths[-1]

    fig  = go.Figure()
    x_ax = list(range(days))
    for i in range(min(200, sims)):
        col = ("rgba(13,156,91,0.13)" if sim_paths[-1, i] >= portfolio_value
               else "rgba(224,49,49,0.10)")
        fig.add_trace(go.Scatter(x=x_ax, y=sim_paths[:, i], mode='lines',
                                 line=dict(color=col, width=0.5),
                                 showlegend=False, hoverinfo='skip'))

    med_path = np.median(sim_paths, axis=1)
    fig.add_trace(go.Scatter(x=x_ax, y=med_path, mode='lines',
                             line=dict(color=BLUE_PRIMARY, width=2.8), name="Median Path"))

    p5  = np.percentile(sim_paths, 5,  axis=1)
    p95 = np.percentile(sim_paths, 95, axis=1)
    fig.add_trace(go.Scatter(
        x=x_ax + x_ax[::-1], y=list(p95) + list(p5[::-1]),
        fill='toself', fillcolor="rgba(26,95,202,0.07)",
        line=dict(color='rgba(0,0,0,0)'), name="90% Confidence Band",
    ))
    fig.add_hline(y=portfolio_value, line=dict(color=TEXT_LIGHT, dash='dash', width=1.5),
                  annotation=dict(text="Initial Value", font=dict(color=TEXT_LIGHT, size=10)))
    apply_theme(fig, title_text=f"Monte Carlo — {sims} Paths over {days} Trading Days",
                yaxis_title="Portfolio Value ($)", xaxis_title="Trading Day")

    med_fin    = np.median(final_vals)
    p5_fin     = np.percentile(final_vals, 5)
    p95_fin    = np.percentile(final_vals, 95)
    pct_profit = (final_vals >= portfolio_value).mean() * 100
    med_ret    = (med_fin / portfolio_value - 1) * 100
    sign       = "▲" if med_ret >= 0 else "▼"

    stats = f"""<div class="metric-grid metric-grid-4" style="margin-top:16px">
        {kpi_card("Median Outcome",   f"${med_fin:,.0f}",
                  f"{sign} {abs(med_ret):.1f}%", "green" if med_ret >= 0 else "red")}
        {kpi_card("Best Case (95th)", f"${p95_fin:,.0f}",
                  f"+{(p95_fin/portfolio_value-1)*100:.1f}%", "green")}
        {kpi_card("Worst Case (5th)", f"${p5_fin:,.0f}",
                  f"{(p5_fin/portfolio_value-1)*100:.1f}%", "red")}
        {kpi_card("% Profitable",     f"{pct_profit:.1f}%",
                  f"of {sims} simulations", "green" if pct_profit >= 50 else "red")}
    </div>"""

    return fig, stats


# ═══════════════════════════════════════════════
#  TAB 7 — Alpha Signal
# ═══════════════════════════════════════════════
def render_alpha_signal(symbol, portfolio_value):
    signal = fetch_alpha_signal(symbol, portfolio_value)

    if signal is None or signal.get('status') == 'error':
        msg = signal.get('message', 'Alpha Signal API not reachable') if signal else \
              'Alpha Signal API not reachable. Make sure port 8002 is running.'
        return banner(f"⚠ {msg}", "alert"), None

    action     = signal.get('action', 'HOLD')
    rec        = signal.get('recommendation', 'HOLD')
    confidence = signal.get('confidence', 0)
    price      = signal.get('current_price', 0)
    ts         = signal.get('timestamp', '')[:19]

    box_cls = 'buy' if action == 'BUY' else 'sell' if action == 'SELL' else 'hold'
    badge_cls = f'badge-{"buy" if action=="BUY" else "sell" if action=="SELL" else "warn"}'

    signals      = signal.get('signals', {})
    explanations = signal.get('explanations', {})

    sig_rows = ""
    for ind, val in signals.items():
        emoji = "🟢" if val > 0 else "🔴" if val < 0 else "⚪"
        sig_rows += f"""
        <tr>
            <td>{emoji} <strong>{ind}</strong></td>
            <td>{explanations.get(ind, '')}</td>
        </tr>"""

    ps  = signal.get('position_sizing', {})
    rm  = signal.get('risk_metrics', {})
    vol = rm.get('volatility')

    kpi_html = f"""
    {section_header("🎯", f"Alpha Signal — {symbol}",
                    f"Generated at {ts}  |  Price: ${fmt(price)}")}

    <div class="metric-grid metric-grid-4" style="margin-bottom:16px">
        {kpi_card("Current Price",   f"${fmt(price)}")}
        {kpi_card("Confidence",      f"{fmt(confidence, 1)}%",
                  "▲ Bullish" if confidence > 0 else "▼ Bearish",
                  "green" if confidence > 0 else "red")}
        {kpi_card("Position Size",   f"{fmt(ps.get('position_size_pct'), 1)}%",
                  f"${fmt(ps.get('actual_position_usd'))}")}
        {kpi_card("Shares",          str(ps.get('recommended_shares', 0)),
                  "Recommended")}
    </div>

    <div class="signal-box {box_cls}">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem;
                    font-weight:700; color:#0d1f3c; margin-bottom:6px;">
            🎯 {rec}
        </div>
        <span class="badge {badge_cls}">{action}</span>
        {'<p style="margin-top:8px; font-size:0.82rem; color:#3a5080;">Volatility: ' +
         fmt(vol*100, 2) + '% &nbsp;|&nbsp; Risk Adjusted: ' +
         ('Yes ✓' if rm.get('risk_adjusted') else 'No') + '</p>'
         if vol is not None else ''}
    </div>

    <h3 style="font-family:'Space Grotesk',sans-serif; color:#0d2d6b; margin-top:20px;">
        📈 Individual Signals
    </h3>
    <table style="width:100%">
        <thead><tr><th>Indicator</th><th>Analysis</th></tr></thead>
        <tbody>{sig_rows}</tbody>
    </table>
    """

    # Technical chart
    df = get_historical(symbol, "6mo")
    if df.empty:
        return kpi_html, None

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
        row_heights=[0.5, 0.25, 0.25]
    )

    close_s = pd.Series(df['Close'].values)
    dates   = df.index

    fig.add_trace(go.Scatter(x=dates, y=df['Close'], name='Close',
                             line=dict(color=BLUE_PRIMARY, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=close_s.rolling(20).mean(), name='SMA(20)',
                             line=dict(color=GOLD, width=1.5, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=close_s.rolling(50).mean(), name='SMA(50)',
                             line=dict(color=GREEN, width=1.5, dash='dot')), row=1, col=1)

    delta = close_s.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss))
    fig.add_trace(go.Scatter(x=dates, y=rsi, name='RSI',
                             line=dict(color="#7c3aed", width=1.8)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=RED,   row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN, row=2, col=1)

    ema12       = close_s.ewm(span=12, adjust=False).mean()
    ema26       = close_s.ewm(span=26, adjust=False).mean()
    macd        = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    fig.add_trace(go.Scatter(x=dates, y=macd,        name='MACD',
                             line=dict(color=BLUE_PRIMARY, width=1.8)), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=signal_line, name='Signal',
                             line=dict(color=RED, width=1.5)), row=3, col=1)

    fig.update_layout(
        **PLOTLY_THEME,
        height=750,
        title=dict(text=f"{symbol} — Technical Indicator Dashboard",
                   font=dict(color="#1a3a6b", size=14, family="Inter, sans-serif")),
    )
    for row in [1, 2, 3]:
        fig.update_xaxes(**AXIS_STYLE, row=row, col=1)
        fig.update_yaxes(**AXIS_STYLE, row=row, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI",       row=2, col=1)
    fig.update_yaxes(title_text="MACD",      row=3, col=1)

    return kpi_html, fig


# ═══════════════════════════════════════════════
#  TAB 8 — Backtest
# ═══════════════════════════════════════════════
def render_backtest(symbol, initial_capital, period):
    bt = fetch_alpha_backtest(symbol, initial_capital, period)

    if bt is None or bt.get('status') == 'error':
        msg = bt.get('message', 'Alpha Signal API not reachable') if bt else \
              'Alpha Signal API not reachable. Make sure port 8002 is running.'
        return banner(f"⚠ {msg}", "alert"), None

    total_ret  = bt.get('total_return_pct', 0)
    bh_ret     = bt.get('buy_hold_return_pct', 0)
    outperf    = bt.get('outperformance', 0)
    final_val  = bt.get('final_value', 0)
    init_cap   = bt.get('initial_capital', initial_capital)
    num_trades = bt.get('num_trades', 0)
    sym        = bt.get('symbol', symbol)
    per        = bt.get('period', period)

    kpi_html = f"""
    {section_header("📊", f"Backtest Results — {sym}",
                    f"Period: {per}  |  Trades: {num_trades}")}
    <div class="metric-grid metric-grid-4">
        {kpi_card("Initial Capital", f"${init_cap:,.0f}")}
        {kpi_card("Final Value",     f"${final_val:,.0f}",
                  "▲ Gain" if final_val >= init_cap else "▼ Loss",
                  "green" if final_val >= init_cap else "red")}
        {kpi_card("Total Return",    f"{fmt(total_ret, 2)}%",
                  "▲ Profit" if total_ret >= 0 else "▼ Loss",
                  "green" if total_ret >= 0 else "red")}
        {kpi_card("Outperformance",  f"{'+' if outperf >= 0 else ''}{fmt(outperf, 2)}%",
                  f"vs Buy & Hold {fmt(bh_ret, 2)}%",
                  "green" if outperf >= 0 else "red")}
    </div>"""

    equity_curve = bt.get('equity_curve', [])
    if not equity_curve:
        return kpi_html, None

    equity_df         = pd.DataFrame(equity_curve)
    equity_df['date'] = pd.to_datetime(equity_df['date'])

    bh_value = init_cap * (1 + bh_ret / 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df['date'], y=equity_df['value'],
        mode='lines', name='Strategy Equity',
        fill='tozeroy', fillcolor="rgba(26,95,202,0.07)",
        line=dict(color=BLUE_PRIMARY, width=2.5),
        hovertemplate="%{x|%b %d, %Y}<br>Value: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=bh_value, line=dict(color=GREEN, dash='dash', width=1.8),
                  annotation=dict(text=f"Buy & Hold: ${bh_value:,.0f}",
                                  font=dict(color=GREEN, size=10)))
    fig.add_hline(y=init_cap, line=dict(color=TEXT_LIGHT, dash='dot', width=1.3),
                  annotation=dict(text=f"Initial: ${init_cap:,.0f}",
                                  font=dict(color=TEXT_LIGHT, size=10)))
    apply_theme(fig, title_text=f"{sym} — Strategy Equity Curve",
                xaxis_title="Date", yaxis_title="Portfolio Value ($)")

    return kpi_html, fig


# ═══════════════════════════════════════════════
#  TAB 9 — Raw Indicators
# ═══════════════════════════════════════════════
def render_indicators(symbol, period):
    data = fetch_alpha_indicators(symbol, period)

    if data is None or data.get('status') == 'error':
        msg = data.get('message', 'Alpha Signal API not reachable') if data else \
              'Alpha Signal API not reachable. Make sure port 8002 is running.'
        return banner(f"⚠ {msg}", "alert"), None

    ma  = data.get('moving_averages', {})
    rsi = data.get('rsi', {})
    mac = data.get('macd', {})
    bb  = data.get('bollinger_bands', {})
    cur = data.get('current_price', 0)

    rsi_val = rsi.get('value', 0) or 0
    rsi_col = "red" if rsi_val > 70 else "green" if rsi_val < 30 else "blue"
    mac_col = "green" if (mac.get('histogram') or 0) > 0 else "red"

    kpi_html = f"""
    {section_header("🔍", f"Raw Indicators — {symbol}",
                    f"Period: {period}  |  Price: ${fmt(cur)}")}

    <div class="metric-grid metric-grid-4" style="margin-bottom:16px">
        {kpi_card("RSI (14)",    fmt(rsi_val), rsi.get('interpretation',''), rsi_col)}
        {kpi_card("MACD Line",   fmt(mac.get('macd_line'),   4), "", mac_col)}
        {kpi_card("MACD Signal", fmt(mac.get('signal_line'), 4), mac.get('signal',''))}
        {kpi_card("BB %B",       fmt(bb.get('percent_b')),
                  "Overbought" if (bb.get('percent_b') or 0) > 0.8 else
                  "Oversold"   if (bb.get('percent_b') or 0) < 0.2 else "Mid-range")}
    </div>

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
        <div>
            <h3 style="font-family:'Space Grotesk',sans-serif; color:#0d2d6b;">
                📈 Moving Averages
            </h3>
            <table style="width:100%">
                <thead><tr><th>Indicator</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>SMA (20)</td> <td>${fmt(ma.get('sma_20'))}</td></tr>
                    <tr><td>SMA (50)</td> <td>${fmt(ma.get('sma_50'))}</td></tr>
                    <tr><td>SMA (200)</td><td>${fmt(ma.get('sma_200'))}</td></tr>
                    <tr><td>EMA (12)</td> <td>${fmt(ma.get('ema_12'))}</td></tr>
                    <tr><td>EMA (26)</td> <td>${fmt(ma.get('ema_26'))}</td></tr>
                </tbody>
            </table>
        </div>
        <div>
            <h3 style="font-family:'Space Grotesk',sans-serif; color:#0d2d6b;">
                📏 Bollinger Bands
            </h3>
            <table style="width:100%">
                <thead><tr><th>Band</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>Upper</td>    <td>${fmt(bb.get('upper'))}</td></tr>
                    <tr><td>Middle</td>   <td>${fmt(bb.get('middle'))}</td></tr>
                    <tr><td>Lower</td>    <td>${fmt(bb.get('lower'))}</td></tr>
                    <tr><td>%B</td>       <td>{fmt(bb.get('percent_b'))}</td></tr>
                    <tr><td>Bandwidth</td><td>{fmt(bb.get('bandwidth'), 4)}</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    """

    # Bollinger band chart
    df = get_historical(symbol, period)
    if df.empty:
        return kpi_html, None

    close_s = pd.Series(df['Close'].values, index=df.index)
    mid     = close_s.rolling(20).mean()
    std     = close_s.rolling(20).std()
    upper   = mid + 2 * std
    lower   = mid - 2 * std

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=upper, name='Upper Band',
        line=dict(color=RED, width=1, dash='dot'),
        hovertemplate="%{x|%b %d}<br>Upper: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=lower, name='Lower Band',
        line=dict(color=GREEN, width=1, dash='dot'),
        fill='tonexty', fillcolor="rgba(26,95,202,0.06)",
        hovertemplate="%{x|%b %d}<br>Lower: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=mid, name='Middle (SMA 20)',
        line=dict(color=GOLD, width=1.5, dash='dash'),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name='Price',
        line=dict(color=BLUE_PRIMARY, width=2),
        hovertemplate="%{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>",
    ))
    apply_theme(fig, title_text=f"{symbol} — Bollinger Bands ({period})",
                yaxis_title="Price ($)")

    return kpi_html, fig


# ═══════════════════════════════════════════════
#  BUILD APP
# ═══════════════════════════════════════════════
HEADER_HTML = """
<div class="app-header">
    <div class="app-title">⬡ Portfolio Optimization Intelligence System</div>
    <div class="header-accent"></div>
    <div class="app-subtitle">Multi-Agent Risk &amp; Market Analytics Platform &nbsp;·&nbsp; v4.0</div>
    <div class="app-team">Team: Ashwini &nbsp;|&nbsp; Dibyendu Sarkar &nbsp;|&nbsp; Jyoti Ranjan Sethi &nbsp;|&nbsp; IIT Madras 2026</div>
</div>
"""

FOOTER_HTML = """
<div class="app-footer">
    ⬡ Portfolio Intelligence System &nbsp;|&nbsp; IIT Madras 2026 &nbsp;|&nbsp;
    Data: Yahoo Finance &nbsp;|&nbsp; For Educational Use Only &nbsp;|&nbsp; Week 4 of 16
</div>
"""

with gr.Blocks(title="Portfolio Optimization Intelligence System") as demo:

    gr.HTML(HEADER_HTML)

    # ── Shared controls ────────────────────────────────────
    with gr.Row():
        shared_symbol    = gr.Dropdown(choices=SYMBOLS, value="AAPL",
                                       label="🚀 Stock Symbol", scale=2)
        shared_period    = gr.Dropdown(choices=["1mo","3mo","6mo","1y","2y","5y"],
                                       value="1y", label="📅 Period", scale=1)
        shared_portfolio = gr.Number(value=100_000, label="💰 Portfolio Value ($)",
                                     minimum=1000, scale=2)

    with gr.Tabs():

        # ── Tab 1: Market Overview ─────────────────────────
        with gr.Tab("📡 Market Overview"):
            overview_btn = gr.Button("🔄 Refresh Market Data", variant="primary", size="lg")
            status_out   = gr.HTML()
            cards_out    = gr.HTML()
            with gr.Row():
                price_out = gr.Plot(label="Stock Prices")
                vol_out   = gr.Plot(label="Trading Volume")
            with gr.Accordion("📋 Detailed Price Table", open=False):
                table_out = gr.Dataframe(interactive=False)
            overview_btn.click(
                fn=render_market_overview,
                outputs=[cards_out, price_out, vol_out, table_out, status_out]
            )

        # ── Tab 2: Historical Analysis ─────────────────────
        with gr.Tab("📈 Historical Analysis"):
            hist_btn  = gr.Button("📈 Load Historical Data", variant="primary", size="lg")
            hist_stat = gr.HTML()
            candle    = gr.Plot(label="Candlestick + MA20")
            ret_chart = gr.Plot(label="Cumulative Return")
            vol_hist  = gr.Plot(label="Volume")
            hist_btn.click(
                fn=render_historical,
                inputs=[shared_symbol, shared_period],
                outputs=[candle, ret_chart, vol_hist, hist_stat]
            )

        # ── Tab 3: Risk Assessment ─────────────────────────
        with gr.Tab("🛡️ Risk Assessment"):
            risk_btn = gr.Button("🔍 Calculate Risk Metrics", variant="primary", size="lg")
            risk_kpi = gr.HTML()
            with gr.Row():
                gauge_out = gr.Plot(label="Risk Score Gauge")
                dist_out  = gr.Plot(label="Return Distribution")
            dd_out = gr.Plot(label="Drawdown Chart")
            rv_out = gr.Plot(label="Rolling Volatility")
            risk_btn.click(
                fn=render_risk,
                inputs=[shared_symbol, shared_portfolio],
                outputs=[risk_kpi, dist_out, dd_out, rv_out, gauge_out]
            )

        # ── Tab 4: Stress Testing ──────────────────────────
        with gr.Tab("💥 Stress Testing"):
            stress_btn = gr.Button("💥 Run Stress Tests", variant="primary", size="lg")
            with gr.Row():
                spct = gr.Plot(label="Loss % by Scenario")
                susd = gr.Plot(label="Dollar Loss by Scenario")
            with gr.Accordion("📋 Full Stress Test Table", open=True):
                stbl = gr.Dataframe(interactive=False)
            stress_btn.click(
                fn=render_stress,
                inputs=[shared_symbol, shared_portfolio],
                outputs=[spct, susd, stbl]
            )

        # ── Tab 5: Correlation ─────────────────────────────
        with gr.Tab("🔗 Correlation"):
            sym_in   = gr.Textbox(value="AAPL,GOOGL,MSFT,TSLA,AMZN",
                                  label="Symbols (comma-separated)")
            corr_btn = gr.Button("🔗 Compute Correlations", variant="primary", size="lg")
            corr_inf = gr.HTML()
            heat_out = gr.Plot(label="Correlation Heatmap")
            cmp_out  = gr.Plot(label="Cumulative Return Comparison")
            corr_btn.click(
                fn=render_correlation,
                inputs=[sym_in],
                outputs=[heat_out, corr_inf, cmp_out]
            )

        # ── Tab 6: Monte Carlo ─────────────────────────────
        with gr.Tab("🎲 Monte Carlo"):
            with gr.Row():
                mc_days = gr.Slider(21, 504, value=252, step=21, label="Simulation Days")
                mc_sims = gr.Slider(100, 1000, value=500, step=100, label="Simulations")
            mc_btn   = gr.Button("🎲 Run Monte Carlo Simulation", variant="primary", size="lg")
            mc_stats = gr.HTML()
            mc_chart = gr.Plot(label="Simulation Paths")
            mc_btn.click(
                fn=render_monte_carlo,
                inputs=[shared_symbol, shared_portfolio, mc_days, mc_sims],
                outputs=[mc_chart, mc_stats]
            )

        # ── Tab 7: Alpha Signal ────────────────────────────
        with gr.Tab("🎯 Alpha Signal"):
            signal_btn = gr.Button("🎯 Generate Trading Signal", variant="primary", size="lg")
            signal_kpi = gr.HTML()
            signal_chart = gr.Plot(label="Technical Indicator Dashboard")
            signal_btn.click(
                fn=render_alpha_signal,
                inputs=[shared_symbol, shared_portfolio],
                outputs=[signal_kpi, signal_chart]
            )

        # ── Tab 8: Backtest ────────────────────────────────
        with gr.Tab("📊 Backtest"):
            with gr.Row():
                bt_capital = gr.Number(value=10_000, label="💵 Initial Capital ($)",
                                       minimum=1000, scale=2)
                bt_period  = gr.Dropdown(
                    choices=["1mo","3mo","6mo","1y","2y"],
                    value="1y", label="📅 Backtest Period", scale=1
                )
            bt_btn   = gr.Button("🚀 Run Backtest", variant="primary", size="lg")
            bt_kpi   = gr.HTML()
            bt_chart = gr.Plot(label="Equity Curve")
            bt_btn.click(
                fn=render_backtest,
                inputs=[shared_symbol, bt_capital, bt_period],
                outputs=[bt_kpi, bt_chart]
            )

        # ── Tab 9: Raw Indicators ──────────────────────────
        with gr.Tab("🔍 Raw Indicators"):
            ind_btn   = gr.Button("📊 Get Indicators", variant="primary", size="lg")
            ind_kpi   = gr.HTML()
            ind_chart = gr.Plot(label="Bollinger Bands Chart")
            ind_btn.click(
                fn=render_indicators,
                inputs=[shared_symbol, shared_period],
                outputs=[ind_kpi, ind_chart]
            )

        # ── Tab 10: About ──────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## ⬡ Portfolio Optimization Intelligence System — v4.0

            A **unified multi-agent AI-powered platform** combining market data,
            risk management, and alpha signal generation.

            ---

            ### 🧩 Modules

            | Tab | Description |
            |---|---|
            | 📡 **Market Overview** | Real-time prices, volume, KPI cards |
            | 📈 **Historical Analysis** | Candlestick, MA20, cumulative returns, volume |
            | 🛡️ **Risk Assessment** | VaR, CVaR, Sharpe, Drawdown, Volatility, Gauge |
            | 💥 **Stress Testing** | 8 crash scenarios — loss % and dollar loss |
            | 🔗 **Correlation** | Heatmap + cumulative return comparison |
            | 🎲 **Monte Carlo** | Up to 1000-path simulation with confidence bands |
            | 🎯 **Alpha Signal** | Buy/Sell/Hold signal via technical analysis |
            | 📊 **Backtest** | Strategy equity curve vs Buy & Hold |
            | 🔍 **Raw Indicators** | RSI, MACD, Bollinger Bands, Moving Averages |

            ---

            ### 🏗️ Architecture
```
            Yahoo Finance API ──► Market Data Agent (port 8000)
                                         │
                                  Risk Agent (port 8001)
                                         │
                              Alpha Signal Agent (port 8002)
                                         │
                              Gradio Unified Dashboard (port 7862)
```

            ---

            ### ⚠️ Notes
            - Tabs 1–6 fetch **directly from Yahoo Finance** (no API needed)
            - Tabs 7–9 require **Alpha Signal API on port 8002** to be running
            - All data is for **educational purposes only**

            ---
            **Team:** Ashwini · Dibyendu Sarkar · Jyoti Ranjan Sethi
            **Week:** 4 of 16 · IIT Madras · 2026
            """)

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    print("Starting Portfolio Optimization Intelligence System...")
    print("Tabs 1-6: Direct Yahoo Finance (no API needed)")
    print("Tabs 7-9: Require Alpha Signal API on port 8002")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=True,
        show_error=True,
        css=CUSTOM_CSS,
    )