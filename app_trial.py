"""
Portfolio Optimization Intelligence System
Unified Dashboard — Week 5
Combines: Market Data + Risk Management + Alpha Signal Agent + Portfolio Optimization
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf
import requests

MARKET_DATA_API = "http://localhost:8000"
RISK_API        = "http://localhost:8001"
ALPHA_API       = "http://localhost:8002"

SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

BG_CARD      = "#ffffff"
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

PLOTLY_THEME = dict(
    paper_bgcolor=BG_CARD, plot_bgcolor="#f8faff",
    font=dict(family="Inter, sans-serif", color=TEXT_DARK, size=12),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT_DARK)),
    margin=dict(l=55, r=30, t=55, b=45),
)
AXIS_STYLE = dict(gridcolor="#e2eaf5", zerolinecolor="#c5d5ea", tickfont=dict(color=TEXT_LIGHT), linecolor=BORDER)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container { background: #f0f4fb !important; font-family: 'Inter', sans-serif !important; color: #0d1f3c !important; }

.app-header { background: linear-gradient(135deg, #0d2d6b 0%, #1a5fca 60%, #2979e8 100%); padding: 36px 40px 28px; text-align: center; border-bottom: 4px solid #d4940a; box-shadow: 0 4px 20px rgba(26,60,107,0.18); }
.app-title { font-family: 'Space Grotesk', sans-serif !important; font-size: 2rem; font-weight: 700; color: #ffffff !important; letter-spacing: 1px; margin-bottom: 6px; }
.app-subtitle { font-size: 0.82rem; color: rgba(255,255,255,0.75) !important; letter-spacing: 2px; text-transform: uppercase; }
.app-team { font-size: 0.75rem; color: rgba(255,255,255,0.55) !important; margin-top: 10px; letter-spacing: 1.5px; }
.header-accent { width: 60px; height: 3px; background: #d4940a; margin: 12px auto; border-radius: 2px; }

/* ─── TAB BAR: wrap instead of overflow/ellipsis ─── */
div[role="tablist"] {
    background: #ffffff !important;
    border-bottom: 2px solid #d0dff0 !important;
    padding: 0 8px !important;
    box-shadow: 0 2px 8px rgba(26,60,107,0.06) !important;
    display: flex !important;
    flex-wrap: wrap !important;
    overflow: visible !important;
}
div[role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important; font-weight: 500 !important;
    color: #6b83a8 !important; border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 11px 12px !important; background: transparent !important;
    transition: all 0.2s !important; white-space: nowrap !important; flex-shrink: 0 !important;
}
div[role="tab"]:hover { color: #1a5fca !important; background: #e8f0fe !important; }
div[role="tab"][aria-selected="true"] { color: #1a5fca !important; border-bottom: 3px solid #1a5fca !important; background: #e8f0fe !important; font-weight: 600 !important; }

/* ─── METRIC GRIDS ─── */
.metric-grid   { display: grid; gap: 14px; margin-bottom: 22px; }
.metric-grid-5 { grid-template-columns: repeat(5, 1fr); }
.metric-grid-4 { grid-template-columns: repeat(4, 1fr); }
.metric-grid-3 { grid-template-columns: repeat(3, 1fr); }
.metric-grid-2 { grid-template-columns: repeat(2, 1fr); }

.kpi-card { background: #ffffff; border: 1px solid #d0dff0; border-radius: 12px; padding: 18px 20px; position: relative; overflow: hidden; box-shadow: 0 2px 10px rgba(26,60,107,0.07); transition: transform 0.2s, box-shadow 0.2s; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #1a5fca, #2979e8); }
.kpi-card.green::before { background: linear-gradient(90deg, #0d9c5b, #22c55e); }
.kpi-card.red::before   { background: linear-gradient(90deg, #e03131, #f87171); }
.kpi-card.gold::before  { background: linear-gradient(90deg, #d4940a, #f59e0b); }
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(26,60,107,0.12); }
.kpi-label { font-size: 0.70rem; font-weight: 600; color: #6b83a8; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 9px; }
.kpi-value { font-family: 'Space Grotesk', sans-serif !important; font-size: 1.6rem; font-weight: 700; color: #0d1f3c; line-height: 1.1; }
.kpi-value.up { color: #0d9c5b; } .kpi-value.down { color: #e03131; } .kpi-value.warn { color: #d4940a; }
.kpi-delta { font-size: 0.74rem; font-weight: 500; margin-top: 5px; color: #6b83a8; }
.kpi-delta.up { color: #0d9c5b; } .kpi-delta.down { color: #e03131; }

.section-header { margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid #d0dff0; }
.section-title { font-family: 'Space Grotesk', sans-serif !important; font-size: 1rem; font-weight: 600; color: #0d2d6b; }
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

.signal-box { border-radius: 12px; padding: 20px 24px; margin: 14px 0; border: 1px solid #d0dff0; }
.signal-box.buy  { background: #d1fae5; border-left: 5px solid #0d9c5b; }
.signal-box.sell { background: #fee2e2; border-left: 5px solid #e03131; }
.signal-box.hold { background: #fef3c7; border-left: 5px solid #d4940a; }

/* ─── PORTFOLIO OPTIMIZATION HERO ─── */
.po-hero {
    background: linear-gradient(135deg, #0d2d6b 0%, #1a5fca 65%, #2979e8 100%);
    border-radius: 16px; padding: 28px 32px 22px; margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 8px 32px rgba(26,60,107,0.22);
    position: relative; overflow: hidden;
}
.po-hero::after {
    content: '⬡'; position: absolute; right: 28px; top: 50%; transform: translateY(-50%);
    font-size: 7rem; opacity: 0.07; color: white; pointer-events: none; line-height: 1;
}
.po-hero-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 6px; }
.po-hero-sub   { font-size: 0.84rem; color: rgba(255,255,255,0.78); line-height: 1.65; max-width: 720px; }
.po-hero-accent { width: 40px; height: 3px; background: #d4940a; border-radius: 2px; margin: 10px 0; }
.po-pill {
    display: inline-block; background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.28); border-radius: 20px;
    padding: 4px 14px; font-size: 0.72rem; font-weight: 600;
    color: rgba(255,255,255,0.92); letter-spacing: 0.8px;
    margin: 6px 4px 0 0; text-transform: uppercase;
}

button, .gr-button { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 0.83rem !important; border-radius: 8px !important; transition: all 0.2s !important; }
.gr-button-primary, button.primary { background: linear-gradient(135deg, #1a5fca, #2979e8) !important; color: #ffffff !important; border: none !important; box-shadow: 0 3px 12px rgba(26,95,202,0.28) !important; }
.gr-button-primary:hover, button.primary:hover { background: linear-gradient(135deg, #0d4fb5, #1a5fca) !important; box-shadow: 0 5px 18px rgba(26,95,202,0.38) !important; transform: translateY(-1px) !important; }

input, select, textarea { background: #ffffff !important; border: 1.5px solid #c5d5ea !important; border-radius: 8px !important; color: #0d1f3c !important; font-family: 'Inter', sans-serif !important; font-size: 0.88rem !important; }
input:focus, select:focus { border-color: #1a5fca !important; box-shadow: 0 0 0 3px rgba(26,95,202,0.12) !important; outline: none !important; }
label { font-family: 'Inter', sans-serif !important; font-size: 0.76rem !important; font-weight: 600 !important; color: #3a5080 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }
input[type="range"] { accent-color: #1a5fca !important; }

table { border-collapse: separate !important; border-spacing: 0 !important; background: #ffffff !important; border-radius: 10px !important; overflow: hidden !important; border: 1px solid #d0dff0 !important; font-size: 0.83rem !important; box-shadow: 0 2px 10px rgba(26,60,107,0.06) !important; }
th { background: #1a3a6b !important; color: #ffffff !important; font-size: 0.72rem !important; font-weight: 600 !important; letter-spacing: 1px !important; text-transform: uppercase !important; padding: 12px 16px !important; }
td { color: #0d1f3c !important; padding: 10px 16px !important; border-bottom: 1px solid #edf2fb !important; background: #ffffff !important; }
tr:nth-child(even) td { background: #f5f8fe !important; }
tr:hover td { background: #e8f0fe !important; }

.gr-accordion { background: #ffffff !important; border: 1px solid #d0dff0 !important; border-radius: 10px !important; }
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { font-family: 'Space Grotesk', sans-serif !important; color: #0d2d6b !important; }
.gr-markdown p, .gr-markdown li { color: #0d1f3c !important; line-height: 1.7 !important; }
.gr-markdown strong { color: #1a5fca !important; }
.gr-markdown code { background: #e8f0fe !important; color: #1a5fca !important; border-radius: 4px !important; padding: 2px 7px !important; }

.app-footer { background: #0d2d6b; color: rgba(255,255,255,0.6); text-align: center; padding: 18px; font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 40px; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0f4fb; }
::-webkit-scrollbar-thumb { background: #c5d5ea; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #1a5fca; }
"""

# ─── HTML helpers ───────────────────────────────────────────────────────────
def kpi_card(label, value, delta="", color="blue"):
    cls_map = {"green":"up","red":"down","gold":"warn"}
    val_cls = cls_map.get(color,""); card_cls = color if color in ["green","red","gold"] else ""
    dlt_cls = "up" if ("▲" in delta or "+" in delta) else "down" if ("▼" in delta or "-" in delta) else ""
    d_html  = f'<div class="kpi-delta {dlt_cls}">{delta}</div>' if delta else ""
    return f'<div class="kpi-card {card_cls}"><div class="kpi-label">{label}</div><div class="kpi-value {val_cls}">{value}</div>{d_html}</div>'

def section_header(icon, title, subtitle=""):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section-header"><div class="section-title">{icon} {title}</div>{sub}</div>'

def banner(msg, kind="info"):
    return f'<div class="{kind}-banner">{msg}</div>'

def fmt(val, decimals=2):
    if val is None: return "N/A"
    try:    return f"{val:.{decimals}f}"
    except: return "N/A"

# ─── Data helpers ───────────────────────────────────────────────────────────
def get_realtime(symbols):
    results = {}
    for sym in symbols:
        try:
            info  = yf.Ticker(sym).info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
            open_ = info.get('regularMarketOpen') or price
            results[sym] = {'symbol':sym,'price':price,'open':open_,'high':info.get('dayHigh',0),'low':info.get('dayLow',0),'volume':info.get('volume',0),'market_cap':info.get('marketCap',0),'pe_ratio':info.get('trailingPE',0),'change_pct':((price-open_)/open_*100) if open_ else 0}
        except:
            results[sym] = {k:0 for k in ['price','open','high','low','volume','market_cap','pe_ratio','change_pct']}; results[sym]['symbol'] = sym
    return results

def get_historical(symbol, period="1y"):
    try:
        df = yf.Ticker(symbol).history(period=period); return df if not df.empty else pd.DataFrame()
    except: return pd.DataFrame()

def get_returns(symbol, period="1y"):
    df = get_historical(symbol, period)
    if df.empty: return pd.Series(dtype=float)
    return df['Close'].pct_change().dropna()

def compute_risk(returns, portfolio_value, rf=0.04):
    if returns.empty: return {}
    daily_vol=returns.std(); annual_vol=daily_vol*np.sqrt(252)
    total_ret=(1+returns).prod()-1; years=max(len(returns)/252,0.01); ann_ret=(1+total_ret)**(1/years)-1
    sharpe=(ann_ret-rf)/annual_vol if annual_vol else 0
    var_95=abs(np.percentile(returns,5)); var_99=abs(np.percentile(returns,1))
    tail=returns[returns<=-var_95]; cvar_95=abs(tail.mean()) if len(tail) else var_95
    cum=(1+returns).cumprod(); peak=cum.cummax(); dd=(cum-peak)/peak; max_dd=abs(dd.min())
    return dict(annual_vol=annual_vol,daily_vol=daily_vol,ann_ret=ann_ret,sharpe=sharpe,var_95=var_95,var_99=var_99,cvar_95=cvar_95,var_95_usd=var_95*portfolio_value,var_99_usd=var_99*portfolio_value,cvar_95_usd=cvar_95*portfolio_value,max_dd=max_dd,drawdown=dd,returns=returns)

def sharpe_label(s):
    if s>3: return "Exceptional","green"
    if s>2: return "Very Good","green"
    if s>1: return "Good","blue"
    if s>0.5: return "Acceptable","gold"
    if s>0: return "Poor","gold"
    return "Losing Money","red"

def apply_theme(fig, title_text=None, yaxis_title=None, xaxis_title=None, extra=None):
    layout=dict(**PLOTLY_THEME); layout['xaxis']=dict(**AXIS_STYLE); layout['yaxis']=dict(**AXIS_STYLE)
    if title_text:  layout['title']=dict(text=title_text,font=dict(color="#1a3a6b",size=14,family="Inter, sans-serif"))
    if yaxis_title: layout['yaxis']['title']=yaxis_title
    if xaxis_title: layout['xaxis']['title']=xaxis_title
    if extra:       layout.update(extra)
    fig.update_layout(**layout); return fig

def fetch_alpha_signal(symbol, portfolio_value):
    try: r=requests.get(f"{ALPHA_API}/signal/{symbol}",params={"portfolio_value":portfolio_value},timeout=15); return r.json() if r.status_code==200 else None
    except: return None

def fetch_alpha_backtest(symbol, initial_capital, period):
    try: r=requests.get(f"{ALPHA_API}/backtest/{symbol}",params={"initial_capital":initial_capital,"period":period},timeout=20); return r.json() if r.status_code==200 else None
    except: return None

def fetch_alpha_indicators(symbol, period):
    try: r=requests.get(f"{ALPHA_API}/indicators/{symbol}",params={"period":period},timeout=10); return r.json() if r.status_code==200 else None
    except: return None

# ─── Tab render functions ───────────────────────────────────────────────────
def render_market_overview():
    data=get_realtime(SYMBOLS); ts=datetime.now().strftime('%d %b %Y  %H:%M:%S')
    cards='<div class="metric-grid metric-grid-5">'
    for sym,d in data.items():
        chg=d.get('change_pct',0); sign="▲" if chg>=0 else "▼"; col="green" if chg>=0 else "red"
        cards+=kpi_card(sym,f"${d['price']:.2f}" if d['price'] else "—",f"{sign} {abs(chg):.2f}%",col)
    cards+="</div>"
    prices={s:d['price'] for s,d in data.items() if d['price']}; changes={s:d['change_pct'] for s,d in data.items()}
    bcolors=[GREEN if changes.get(s,0)>=0 else RED for s in prices]
    fig_p=go.Figure(); fig_p.add_trace(go.Bar(x=list(prices.keys()),y=list(prices.values()),marker=dict(color=bcolors,line=dict(color='white',width=1)),text=[f"${v:.2f}" for v in prices.values()],textposition='outside',textfont=dict(size=11,color=TEXT_DARK),hovertemplate="<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>"))
    apply_theme(fig_p,title_text="Current Stock Prices (USD)",yaxis_title="Price ($)",extra={"showlegend":False})
    vols={s:d.get('volume',0) for s,d in data.items()}
    fig_v=go.Figure(); fig_v.add_trace(go.Bar(x=list(vols.keys()),y=list(vols.values()),marker=dict(color=list(vols.values()),colorscale=[[0,BLUE_LIGHT],[1,BLUE_PRIMARY]],showscale=False,line=dict(color='white',width=1)),text=[f"{v/1e6:.1f}M" for v in vols.values()],textposition='outside',textfont=dict(size=11,color=TEXT_DARK),hovertemplate="<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>"))
    apply_theme(fig_v,title_text="Trading Volume",yaxis_title="Volume",extra={"showlegend":False})
    rows=[]
    for s,d in data.items():
        chg=d.get('change_pct',0); rows.append({'Symbol':s,'Price ($)':f"${d['price']:.2f}" if d['price'] else "—",'Open ($)':f"${d['open']:.2f}" if d['open'] else "—",'High ($)':f"${d['high']:.2f}" if d['high'] else "—",'Low ($)':f"${d['low']:.2f}" if d['low'] else "—",'Volume':f"{d['volume']/1e6:.1f}M" if d['volume'] else "—",'Mkt Cap':f"${d['market_cap']/1e12:.2f}T" if d.get('market_cap') else "—",'P/E':f"{d['pe_ratio']:.1f}" if d.get('pe_ratio') else "—",'Change':f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%"})
    return cards,fig_p,fig_v,pd.DataFrame(rows),banner(f"✅ Data refreshed at {ts}","success")

def render_historical(symbol,period):
    df=get_historical(symbol,period)
    if df.empty: return None,None,None,banner("⚠ No data available.","alert")
    fig_c=go.Figure(); fig_c.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],increasing=dict(line=dict(color=GREEN),fillcolor="rgba(13,156,91,0.22)"),decreasing=dict(line=dict(color=RED),fillcolor="rgba(224,49,49,0.22)"),name="Price"))
    fig_c.add_trace(go.Scatter(x=df.index,y=df['Close'].rolling(20).mean(),name="MA 20",line=dict(color=BLUE_PRIMARY,width=1.8,dash='dot')))
    apply_theme(fig_c,title_text=f"{symbol} — Candlestick Chart ({period})",yaxis_title="Price (USD)",extra={"xaxis_rangeslider_visible":False})
    returns=df['Close'].pct_change().dropna(); cum_ret=(1+returns).cumprod()-1; col_ret=GREEN if cum_ret.iloc[-1]>=0 else RED
    fig_r=go.Figure(); fig_r.add_trace(go.Scatter(x=cum_ret.index,y=cum_ret*100,fill='tozeroy',fillcolor="rgba(13,156,91,0.10)" if cum_ret.iloc[-1]>=0 else "rgba(224,49,49,0.10)",line=dict(color=col_ret,width=2.2),name="Cumulative Return",hovertemplate="%{x|%b %d, %Y}<br>Return: %{y:.2f}%<extra></extra>"))
    fig_r.add_hline(y=0,line=dict(color=TEXT_LIGHT,dash='dash',width=1)); apply_theme(fig_r,title_text="Cumulative Return (%)",yaxis_title="Return (%)")
    vcols=[GREEN if c>=o else RED for c,o in zip(df['Close'],df['Open'])]
    fig_v=go.Figure(); fig_v.add_trace(go.Bar(x=df.index,y=df['Volume'],marker_color=vcols,name="Volume",hovertemplate="%{x|%b %d}<br>Vol: %{y:,.0f}<extra></extra>"))
    apply_theme(fig_v,title_text="Volume (Green = Up Day, Red = Down Day)",yaxis_title="Volume")
    total=cum_ret.iloc[-1]*100; sign="▲" if total>=0 else "▼"; col="green" if total>=0 else "red"
    stats=f"""<div class="metric-grid metric-grid-4">{kpi_card("Current Price",f"${df['Close'].iloc[-1]:.2f}")}{kpi_card("Period High",f"${df['High'].max():.2f}",color="green")}{kpi_card("Period Low",f"${df['Low'].min():.2f}",color="red")}{kpi_card("Total Return",f"{sign} {abs(total):.2f}%",color=col)}</div>"""
    return fig_c,fig_r,fig_v,stats

def render_risk(symbol,portfolio_value):
    returns=get_returns(symbol,"1y")
    if returns.empty: return banner("⚠ Could not fetch data.","alert"),None,None,None,None
    m=compute_risk(returns,portfolio_value); slabel,scol=sharpe_label(m['sharpe'])
    risk_ok=m['annual_vol']<0.30 and m['max_dd']<0.20 and m['sharpe']>1.0
    badge='<span class="badge badge-ok">✓ Within Limits</span>' if risk_ok else '<span class="badge badge-alert">⚠ Risk Alert</span>'
    kpi_html=f"""{section_header("🛡️",f"Risk Assessment — {symbol}",f"Portfolio Value: ${portfolio_value:,.0f}  |  {len(returns)} trading days")}<div style="margin-bottom:14px">{badge}</div><div class="metric-grid metric-grid-4" style="margin-bottom:14px">{kpi_card("VaR 95%",f"{m['var_95']:.2%}",f"−${m['var_95_usd']:,.0f}/day","red")}{kpi_card("VaR 99%",f"{m['var_99']:.2%}",f"−${m['var_99_usd']:,.0f}/day","red")}{kpi_card("CVaR 95%",f"{m['cvar_95']:.2%}","Expected Shortfall","red")}{kpi_card("Annual Vol",f"{m['annual_vol']:.2%}",f"Daily: {m['daily_vol']:.2%}","gold" if m['annual_vol']>0.25 else "blue")}</div><div class="metric-grid metric-grid-4">{kpi_card("Max Drawdown",f"{m['max_dd']:.2%}","Peak-to-Trough","red" if m['max_dd']>0.20 else "gold")}{kpi_card("Sharpe Ratio",f"{m['sharpe']:.2f}",slabel,scol)}{kpi_card("Annual Return",f"{m['ann_ret']:.2%}","▲ Positive" if m['ann_ret']>=0 else "▼ Negative","green" if m['ann_ret']>=0 else "red")}{kpi_card("Data Points",str(len(returns)),"Trading Days")}</div>"""
    fig_dist=go.Figure(); fig_dist.add_trace(go.Histogram(x=returns*100,nbinsx=55,marker=dict(color=BLUE_PRIMARY,opacity=0.75,line=dict(color='white',width=0.5)),name="Daily Returns",hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>"))
    fig_dist.add_vline(x=-m['var_95']*100,line=dict(color=GOLD,dash='dash',width=2),annotation=dict(text="VaR 95%",font=dict(color=GOLD,size=10))); fig_dist.add_vline(x=-m['var_99']*100,line=dict(color=RED,dash='dash',width=2),annotation=dict(text="VaR 99%",font=dict(color=RED,size=10)))
    apply_theme(fig_dist,title_text="Return Distribution with VaR Lines",xaxis_title="Daily Return (%)",yaxis_title="Frequency")
    dd=m['drawdown']; fig_dd=go.Figure(); fig_dd.add_trace(go.Scatter(x=dd.index,y=dd*100,fill='tozeroy',fillcolor="rgba(224,49,49,0.13)",line=dict(color=RED,width=1.8),name="Drawdown %",hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2f}%<extra></extra>"))
    fig_dd.add_hline(y=-m['max_dd']*100,line=dict(color=GOLD,dash='dot',width=1.5),annotation=dict(text=f"Max DD {m['max_dd']:.2%}",font=dict(color=GOLD,size=10))); apply_theme(fig_dd,title_text="Underwater Drawdown Chart",yaxis_title="Drawdown (%)")
    rv=returns.rolling(21).std()*np.sqrt(252)*100; fig_rv=go.Figure(); fig_rv.add_trace(go.Scatter(x=rv.index,y=rv,fill='tozeroy',fillcolor="rgba(26,95,202,0.09)",line=dict(color=BLUE_PRIMARY,width=2),name="21-day Vol",hovertemplate="%{x|%b %d, %Y}<br>Vol: %{y:.2f}%<extra></extra>"))
    fig_rv.add_hline(y=30,line=dict(color=RED,dash='dash',width=1.3),annotation=dict(text="Risk Limit 30%",font=dict(color=RED,size=10))); apply_theme(fig_rv,title_text="Rolling 21-Day Annualised Volatility",yaxis_title="Volatility (%)")
    risk_score=min(100,m['annual_vol']/0.5*40+m['max_dd']/0.5*40+max(0,1-m['sharpe'])*20); gcol=GREEN if risk_score<40 else GOLD if risk_score<70 else RED
    fig_g=go.Figure(go.Indicator(mode="gauge+number",value=risk_score,title=dict(text="RISK SCORE",font=dict(family="Inter",size=13,color=TEXT_MED)),number=dict(font=dict(family="Space Grotesk",size=38,color=gcol)),gauge=dict(axis=dict(range=[0,100],tickwidth=1,tickcolor=TEXT_LIGHT,tickfont=dict(family="Inter",size=10,color=TEXT_LIGHT)),bar=dict(color=gcol,thickness=0.25),bgcolor=BG_CARD,borderwidth=1,bordercolor=BORDER,steps=[dict(range=[0,40],color="rgba(13,156,91,0.08)"),dict(range=[40,70],color="rgba(212,148,10,0.08)"),dict(range=[70,100],color="rgba(224,49,49,0.08)")],threshold=dict(line=dict(color=gcol,width=3),thickness=0.75,value=risk_score))))
    fig_g.update_layout(paper_bgcolor=BG_CARD,font=dict(family="Inter",color=TEXT_DARK),height=260,margin=dict(l=30,r=30,t=60,b=20))
    return kpi_html,fig_dist,fig_dd,fig_rv,fig_g

SCENARIOS={"Moderate −5%":-0.05,"Correction −10%":-0.10,"Bear Market −20%":-0.20,"Severe −30%":-0.30,"2008 Crisis −50%":-0.50,"COVID −35%":-0.35,"Flash Crash −10%":-0.10,"Rate Shock −15%":-0.15}

def render_stress(symbol,portfolio_value):
    returns=get_returns(symbol,"1y"); avg_daily=returns.mean() if not returns.empty else 0.0003
    rows,pcts,dloss,labels=[],[],[],[]
    for name,shock in SCENARIOS.items():
        shocked=portfolio_value*(1+shock); loss=portfolio_value-shocked; days_rec=abs(shock)/avg_daily if avg_daily>0 else float('inf'); yrs_rec=round(days_rec/252,1) if days_rec!=float('inf') else None
        rows.append({'Scenario':name,'Market Shock':f"{shock:.0%}",'Portfolio After':f"${shocked:,.0f}",'Loss Amount':f"${loss:,.0f}",'Recovery (yrs)':str(yrs_rec) if yrs_rec else "N/A"}); pcts.append(shock*100); dloss.append(loss); labels.append(name)
    def sev(l): return RED if l<-30 else GOLD if l<-15 else BLUE_PRIMARY
    fig_pct=go.Figure(); fig_pct.add_trace(go.Bar(x=labels,y=pcts,marker=dict(color=[sev(l) for l in pcts],line=dict(color='white',width=1)),text=[f"{l:.0f}%" for l in pcts],textposition='outside',textfont=dict(size=10,color=TEXT_DARK),hovertemplate="<b>%{x}</b><br>Loss: %{y:.1f}%<extra></extra>"))
    apply_theme(fig_pct,title_text="Portfolio Loss % by Scenario",yaxis_title="Loss (%)",extra={"yaxis":dict(**AXIS_STYLE,range=[min(pcts)*1.3,5])})
    fig_usd=go.Figure(); fig_usd.add_trace(go.Bar(x=labels,y=dloss,marker=dict(color=dloss,colorscale=[[0,BLUE_LIGHT],[0.5,GOLD],[1,RED]],showscale=False,line=dict(color='white',width=1)),text=[f"${l:,.0f}" for l in dloss],textposition='outside',textfont=dict(size=10,color=TEXT_DARK),hovertemplate="<b>%{x}</b><br>Loss: $%{y:,.0f}<extra></extra>"))
    apply_theme(fig_usd,title_text="Dollar Loss by Scenario",yaxis_title="Loss ($)")
    return fig_pct,fig_usd,pd.DataFrame(rows)

def render_correlation(symbols_str):
    syms=[s.strip().upper() for s in symbols_str.split(',') if s.strip()]
    if len(syms)<2: return None,banner("⚠ Enter at least 2 comma-separated symbols.","alert"),None
    all_ret={}
    for s in syms:
        r=get_returns(s,"1y")
        if not r.empty: all_ret[s]=r
    if len(all_ret)<2: return None,banner("⚠ Could not fetch data for enough symbols.","alert"),None
    df_ret=pd.DataFrame(all_ret).dropna(); corr=df_ret.corr()
    fig_h=go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.index.tolist(),colorscale=[[0,RED],[0.5,"#f0f4fb"],[1,BLUE_PRIMARY]],zmid=0,zmin=-1,zmax=1,text=corr.values.round(2),texttemplate="%{text}",textfont=dict(family="Inter",size=13,color=TEXT_DARK),hovertemplate="<b>%{x} vs %{y}</b><br>r = %{z:.3f}<extra></extra>",colorbar=dict(tickfont=dict(family="Inter",color=TEXT_MED),title=dict(text="r",font=dict(color=TEXT_MED)))))
    apply_theme(fig_h,title_text="Correlation Matrix — 1 Year Daily Returns")
    cum_df=(1+df_ret).cumprod()-1; palette=[BLUE_PRIMARY,GREEN,GOLD,RED,"#7c3aed","#db2777","#0891b2"]; fig_cr=go.Figure()
    for i,col in enumerate(cum_df.columns): fig_cr.add_trace(go.Scatter(x=cum_df.index,y=cum_df[col]*100,name=col,line=dict(color=palette[i%len(palette)],width=2.2),hovertemplate=f"<b>{col}</b><br>%{{x|%b %d}}<br>Return: %{{y:.2f}}%<extra></extra>"))
    apply_theme(fig_cr,title_text="Cumulative Returns Comparison (%)",yaxis_title="Return (%)")
    avg_corr=corr.values[np.triu_indices_from(corr.values,k=1)].mean()
    if avg_corr<0.5:   msg,kind=f"✅ Well Diversified — avg correlation: {avg_corr:.3f}","success"
    elif avg_corr<0.7: msg,kind=f"⚠ Moderately Correlated — avg correlation: {avg_corr:.3f}","info"
    else:              msg,kind=f"🔴 Highly Correlated — Low Diversification Benefit (r={avg_corr:.3f})","alert"
    return fig_h,banner(msg,kind),fig_cr

def render_monte_carlo(symbol,portfolio_value,days,sims):
    days,sims=int(days),int(sims); returns=get_returns(symbol,"1y")
    if returns.empty: return None,banner("⚠ Could not fetch data.","alert")
    mu,sigma=returns.mean(),returns.std(); np.random.seed(42)
    sim_rets=np.random.normal(mu,sigma,(days,sims)); sim_paths=portfolio_value*np.exp(np.cumsum(np.log(1+sim_rets),axis=0)); final_vals=sim_paths[-1]
    fig=go.Figure(); x_ax=list(range(days))
    for i in range(min(200,sims)):
        col="rgba(13,156,91,0.13)" if sim_paths[-1,i]>=portfolio_value else "rgba(224,49,49,0.10)"
        fig.add_trace(go.Scatter(x=x_ax,y=sim_paths[:,i],mode='lines',line=dict(color=col,width=0.5),showlegend=False,hoverinfo='skip'))
    med_path=np.median(sim_paths,axis=1); fig.add_trace(go.Scatter(x=x_ax,y=med_path,mode='lines',line=dict(color=BLUE_PRIMARY,width=2.8),name="Median Path"))
    p5=np.percentile(sim_paths,5,axis=1); p95=np.percentile(sim_paths,95,axis=1)
    fig.add_trace(go.Scatter(x=x_ax+x_ax[::-1],y=list(p95)+list(p5[::-1]),fill='toself',fillcolor="rgba(26,95,202,0.07)",line=dict(color='rgba(0,0,0,0)'),name="90% Confidence Band"))
    fig.add_hline(y=portfolio_value,line=dict(color=TEXT_LIGHT,dash='dash',width=1.5),annotation=dict(text="Initial Value",font=dict(color=TEXT_LIGHT,size=10)))
    apply_theme(fig,title_text=f"Monte Carlo — {sims} Paths over {days} Trading Days",yaxis_title="Portfolio Value ($)",xaxis_title="Trading Day")
    med_fin=np.median(final_vals); p5_fin=np.percentile(final_vals,5); p95_fin=np.percentile(final_vals,95); pct_profit=(final_vals>=portfolio_value).mean()*100; med_ret=(med_fin/portfolio_value-1)*100; sign="▲" if med_ret>=0 else "▼"
    stats=f"""<div class="metric-grid metric-grid-4" style="margin-top:16px">{kpi_card("Median Outcome",f"${med_fin:,.0f}",f"{sign} {abs(med_ret):.1f}%","green" if med_ret>=0 else "red")}{kpi_card("Best Case (95th)",f"${p95_fin:,.0f}",f"+{(p95_fin/portfolio_value-1)*100:.1f}%","green")}{kpi_card("Worst Case (5th)",f"${p5_fin:,.0f}",f"{(p5_fin/portfolio_value-1)*100:.1f}%","red")}{kpi_card("% Profitable",f"{pct_profit:.1f}%",f"of {sims} simulations","green" if pct_profit>=50 else "red")}</div>"""
    return fig,stats

def render_alpha_signal(symbol,portfolio_value):
    signal=fetch_alpha_signal(symbol,portfolio_value)
    if signal is None or signal.get('status')=='error':
        msg=signal.get('message','Alpha Signal API not reachable') if signal else 'Alpha Signal API not reachable. Make sure port 8002 is running.'
        return banner(f"⚠ {msg}","alert"),None
    action=signal.get('action','HOLD'); rec=signal.get('recommendation','HOLD'); confidence=signal.get('confidence',0); price=signal.get('current_price',0); ts=signal.get('timestamp','')[:19]
    box_cls='buy' if action=='BUY' else 'sell' if action=='SELL' else 'hold'; badge_cls=f'badge-{"buy" if action=="BUY" else "sell" if action=="SELL" else "warn"}'
    signals=signal.get('signals',{}); explanations=signal.get('explanations',{})
    sig_rows="".join(f"<tr><td>{'🟢' if val>0 else '🔴' if val<0 else '⚪'} <strong>{ind}</strong></td><td>{explanations.get(ind,'')}</td></tr>" for ind,val in signals.items())
    ps=signal.get('position_sizing',{}); rm=signal.get('risk_metrics',{}); vol=rm.get('volatility')
    kpi_html=f"""{section_header("🎯",f"Alpha Signal — {symbol}",f"Generated at {ts}  |  Price: ${fmt(price)}")}<div class="metric-grid metric-grid-4" style="margin-bottom:16px">{kpi_card("Current Price",f"${fmt(price)}")}{kpi_card("Confidence",f"{fmt(confidence,1)}%","▲ Bullish" if confidence>0 else "▼ Bearish","green" if confidence>0 else "red")}{kpi_card("Position Size",f"{fmt(ps.get('position_size_pct'),1)}%",f"${fmt(ps.get('actual_position_usd'))}")}{kpi_card("Shares",str(ps.get('recommended_shares',0)),"Recommended")}</div><div class="signal-box {box_cls}"><div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;color:#0d1f3c;margin-bottom:6px;">🎯 {rec}</div><span class="badge {badge_cls}">{action}</span>{'<p style="margin-top:8px;font-size:0.82rem;color:#3a5080;">Volatility: '+fmt(vol*100,2)+'% &nbsp;|&nbsp; Risk Adjusted: '+('Yes ✓' if rm.get('risk_adjusted') else 'No')+'</p>' if vol is not None else ''}</div><h3 style="font-family:'Space Grotesk',sans-serif;color:#0d2d6b;margin-top:20px;">📈 Individual Signals</h3><table style="width:100%"><thead><tr><th>Indicator</th><th>Analysis</th></tr></thead><tbody>{sig_rows}</tbody></table>"""
    df=get_historical(symbol,"6mo")
    if df.empty: return kpi_html,None
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.05,subplot_titles=('Price & Moving Averages','RSI','MACD'),row_heights=[0.5,0.25,0.25])
    close_s=pd.Series(df['Close'].values); dates=df.index
    fig.add_trace(go.Scatter(x=dates,y=df['Close'],name='Close',line=dict(color=BLUE_PRIMARY,width=2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=dates,y=close_s.rolling(20).mean(),name='SMA(20)',line=dict(color=GOLD,width=1.5,dash='dot')),row=1,col=1)
    fig.add_trace(go.Scatter(x=dates,y=close_s.rolling(50).mean(),name='SMA(50)',line=dict(color=GREEN,width=1.5,dash='dot')),row=1,col=1)
    delta=close_s.diff(); gain=delta.where(delta>0,0).rolling(14).mean(); loss=(-delta.where(delta<0,0)).rolling(14).mean(); rsi=100-(100/(1+gain/loss))
    fig.add_trace(go.Scatter(x=dates,y=rsi,name='RSI',line=dict(color="#7c3aed",width=1.8)),row=2,col=1)
    fig.add_hline(y=70,line_dash="dash",line_color=RED,row=2,col=1); fig.add_hline(y=30,line_dash="dash",line_color=GREEN,row=2,col=1)
    ema12=close_s.ewm(span=12,adjust=False).mean(); ema26=close_s.ewm(span=26,adjust=False).mean(); macd=ema12-ema26; signal_line=macd.ewm(span=9,adjust=False).mean()
    fig.add_trace(go.Scatter(x=dates,y=macd,name='MACD',line=dict(color=BLUE_PRIMARY,width=1.8)),row=3,col=1)
    fig.add_trace(go.Scatter(x=dates,y=signal_line,name='Signal',line=dict(color=RED,width=1.5)),row=3,col=1)
    fig.update_layout(**PLOTLY_THEME,height=750,title=dict(text=f"{symbol} — Technical Indicator Dashboard",font=dict(color="#1a3a6b",size=14,family="Inter, sans-serif")))
    for row in [1,2,3]: fig.update_xaxes(**AXIS_STYLE,row=row,col=1); fig.update_yaxes(**AXIS_STYLE,row=row,col=1)
    fig.update_yaxes(title_text="Price ($)",row=1,col=1); fig.update_yaxes(title_text="RSI",row=2,col=1); fig.update_yaxes(title_text="MACD",row=3,col=1)
    return kpi_html,fig

def render_backtest(symbol,initial_capital,period):
    bt=fetch_alpha_backtest(symbol,initial_capital,period)
    if bt is None or bt.get('status')=='error':
        msg=bt.get('message','Alpha Signal API not reachable') if bt else 'Alpha Signal API not reachable. Make sure port 8002 is running.'
        return banner(f"⚠ {msg}","alert"),None
    total_ret=bt.get('total_return_pct',0); bh_ret=bt.get('buy_hold_return_pct',0); outperf=bt.get('outperformance',0); final_val=bt.get('final_value',0); init_cap=bt.get('initial_capital',initial_capital); num_trades=bt.get('num_trades',0); sym=bt.get('symbol',symbol); per=bt.get('period',period)
    kpi_html=f"""{section_header("📊",f"Backtest Results — {sym}",f"Period: {per}  |  Trades: {num_trades}")}<div class="metric-grid metric-grid-4">{kpi_card("Initial Capital",f"${init_cap:,.0f}")}{kpi_card("Final Value",f"${final_val:,.0f}","▲ Gain" if final_val>=init_cap else "▼ Loss","green" if final_val>=init_cap else "red")}{kpi_card("Total Return",f"{fmt(total_ret,2)}%","▲ Profit" if total_ret>=0 else "▼ Loss","green" if total_ret>=0 else "red")}{kpi_card("Outperformance",f"{'+' if outperf>=0 else ''}{fmt(outperf,2)}%",f"vs Buy & Hold {fmt(bh_ret,2)}%","green" if outperf>=0 else "red")}</div>"""
    equity_curve=bt.get('equity_curve',[])
    if not equity_curve: return kpi_html,None
    equity_df=pd.DataFrame(equity_curve); equity_df['date']=pd.to_datetime(equity_df['date']); bh_value=init_cap*(1+bh_ret/100)
    fig=go.Figure(); fig.add_trace(go.Scatter(x=equity_df['date'],y=equity_df['value'],mode='lines',name='Strategy Equity',fill='tozeroy',fillcolor="rgba(26,95,202,0.07)",line=dict(color=BLUE_PRIMARY,width=2.5),hovertemplate="%{x|%b %d, %Y}<br>Value: $%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=bh_value,line=dict(color=GREEN,dash='dash',width=1.8),annotation=dict(text=f"Buy & Hold: ${bh_value:,.0f}",font=dict(color=GREEN,size=10)))
    fig.add_hline(y=init_cap,line=dict(color=TEXT_LIGHT,dash='dot',width=1.3),annotation=dict(text=f"Initial: ${init_cap:,.0f}",font=dict(color=TEXT_LIGHT,size=10)))
    apply_theme(fig,title_text=f"{sym} — Strategy Equity Curve",xaxis_title="Date",yaxis_title="Portfolio Value ($)")
    return kpi_html,fig

def render_indicators(symbol,period):
    data=fetch_alpha_indicators(symbol,period)
    if data is None or data.get('status')=='error':
        msg=data.get('message','Alpha Signal API not reachable') if data else 'Alpha Signal API not reachable. Make sure port 8002 is running.'
        return banner(f"⚠ {msg}","alert"),None
    ma=data.get('moving_averages',{}); rsi=data.get('rsi',{}); mac=data.get('macd',{}); bb=data.get('bollinger_bands',{}); cur=data.get('current_price',0)
    rsi_val=rsi.get('value',0) or 0; rsi_col="red" if rsi_val>70 else "green" if rsi_val<30 else "blue"; mac_col="green" if (mac.get('histogram') or 0)>0 else "red"
    kpi_html=f"""{section_header("🔍",f"Raw Indicators — {symbol}",f"Period: {period}  |  Price: ${fmt(cur)}")}<div class="metric-grid metric-grid-4" style="margin-bottom:16px">{kpi_card("RSI (14)",fmt(rsi_val),rsi.get('interpretation',''),rsi_col)}{kpi_card("MACD Line",fmt(mac.get('macd_line'),4),"",mac_col)}{kpi_card("MACD Signal",fmt(mac.get('signal_line'),4),mac.get('signal',''))}{kpi_card("BB %B",fmt(bb.get('percent_b')),"Overbought" if (bb.get('percent_b') or 0)>0.8 else "Oversold" if (bb.get('percent_b') or 0)<0.2 else "Mid-range")}</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;"><div><h3 style="font-family:'Space Grotesk',sans-serif;color:#0d2d6b;">📈 Moving Averages</h3><table style="width:100%"><thead><tr><th>Indicator</th><th>Value</th></tr></thead><tbody><tr><td>SMA (20)</td><td>${fmt(ma.get('sma_20'))}</td></tr><tr><td>SMA (50)</td><td>${fmt(ma.get('sma_50'))}</td></tr><tr><td>SMA (200)</td><td>${fmt(ma.get('sma_200'))}</td></tr><tr><td>EMA (12)</td><td>${fmt(ma.get('ema_12'))}</td></tr><tr><td>EMA (26)</td><td>${fmt(ma.get('ema_26'))}</td></tr></tbody></table></div><div><h3 style="font-family:'Space Grotesk',sans-serif;color:#0d2d6b;">📏 Bollinger Bands</h3><table style="width:100%"><thead><tr><th>Band</th><th>Value</th></tr></thead><tbody><tr><td>Upper</td><td>${fmt(bb.get('upper'))}</td></tr><tr><td>Middle</td><td>${fmt(bb.get('middle'))}</td></tr><tr><td>Lower</td><td>${fmt(bb.get('lower'))}</td></tr><tr><td>%B</td><td>{fmt(bb.get('percent_b'))}</td></tr><tr><td>Bandwidth</td><td>{fmt(bb.get('bandwidth'),4)}</td></tr></tbody></table></div></div>"""
    df=get_historical(symbol,period)
    if df.empty: return kpi_html,None
    close_s=pd.Series(df['Close'].values,index=df.index); mid=close_s.rolling(20).mean(); std=close_s.rolling(20).std(); upper=mid+2*std; lower=mid-2*std
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=upper,name='Upper Band',line=dict(color=RED,width=1,dash='dot'),hovertemplate="%{x|%b %d}<br>Upper: $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df.index,y=lower,name='Lower Band',line=dict(color=GREEN,width=1,dash='dot'),fill='tonexty',fillcolor="rgba(26,95,202,0.06)",hovertemplate="%{x|%b %d}<br>Lower: $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df.index,y=mid,name='Middle (SMA 20)',line=dict(color=GOLD,width=1.5,dash='dash')))
    fig.add_trace(go.Scatter(x=df.index,y=df['Close'],name='Price',line=dict(color=BLUE_PRIMARY,width=2),hovertemplate="%{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>"))
    apply_theme(fig,title_text=f"{symbol} — Bollinger Bands ({period})",yaxis_title="Price ($)")
    return kpi_html,fig

# ─── Portfolio Optimization helpers ────────────────────────────────────────
PALETTE_PO=[BLUE_PRIMARY,GREEN,GOLD,RED,"#7c3aed"]

def _po_weights_table(weights,portfolio_value):
    rows=""
    for sym,w in sorted(weights.items(),key=lambda x:-x[1]):
        usd=w*portfolio_value; bar=int(w*220)
        rows+=f'<tr><td style="font-weight:600;color:{TEXT_DARK}">{sym}</td><td>{w*100:.2f}%</td><td>${usd:,.0f}</td><td><div style="background:{BLUE_PRIMARY};height:12px;width:{bar}px;border-radius:4px;"></div></td></tr>'
    return f'<table style="width:100%;border-collapse:collapse;font-size:14px;margin-top:8px"><thead><tr style="background:{BLUE_DARK};color:#fff"><th style="padding:9px 14px;text-align:left">Symbol</th><th style="padding:9px 14px;text-align:left">Weight</th><th style="padding:9px 14px;text-align:left">Amount ($)</th><th style="padding:9px 14px;text-align:left">Allocation Bar</th></tr></thead><tbody>{rows}</tbody></table>'

def _po_metric_cards(ret,vol,sharpe):
    return f'<div class="metric-grid metric-grid-3" style="margin-bottom:14px">{kpi_card("Expected Return",f"{ret*100:.2f}%","▲ Annualised" if ret>=0 else "▼ Annualised","green" if ret>=0 else "red")}{kpi_card("Volatility",f"{vol*100:.2f}%","Annualised","gold")}{kpi_card("Sharpe Ratio",f"{sharpe:.3f}","Exceptional" if sharpe>2 else "Good" if sharpe>1 else "Acceptable","green" if sharpe>1 else "gold")}</div>'

def _po_pie(weights,title):
    fig=go.Figure(go.Pie(labels=list(weights.keys()),values=[round(v*100,2) for v in weights.values()],marker=dict(colors=PALETTE_PO,line=dict(color="white",width=2)),textinfo="label+percent",hole=0.38,hovertemplate="%{label}: %{value:.2f}%<extra></extra>"))
    fig.update_layout(**PLOTLY_THEME,title=dict(text=title,font=dict(color=BLUE_DARK,size=14,family="Inter, sans-serif")),showlegend=True,height=400); return fig

def _parse_syms(s): return [x.strip().upper() for x in s.split(",") if x.strip()]

_po_agent=None
def _po_agent_instance():
    global _po_agent
    if _po_agent is None:
        try:
            from agents.portfolio_optimization.agent import PortfolioOptimizationAgent
            _po_agent=PortfolioOptimizationAgent()
        except ImportError:
            import sys,os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
            from agents.portfolio_optimization.agent import PortfolioOptimizationAgent
            _po_agent=PortfolioOptimizationAgent()
    return _po_agent

def po_max_sharpe(symbols_str,max_weight_pct,period,portfolio_value):
    syms=_parse_syms(symbols_str); max_w=max_weight_pct/100
    try: agent=_po_agent_instance(); df_ret=agent.fetch_returns_data(syms,period); res=agent.optimize_max_sharpe(df_ret,max_weight=max_w)
    except Exception as exc: return banner(f"⚠ Error: {exc}","alert"),None
    status_txt="✅ Optimisation Successful" if res["optimization_success"] else "⚠ Converged with warnings"
    html=f"""{section_header("⭐","Maximum Sharpe Ratio Portfolio",f"Period: {period} | Symbols: {', '.join(syms)}")}{banner(status_txt,"success" if res["optimization_success"] else "info")}{_po_metric_cards(res["expected_return"],res["volatility"],res["sharpe_ratio"])}<div class="section-header"><div class="section-title">📊 Optimal Weight Allocation</div></div>{_po_weights_table(res["weights"],portfolio_value)}"""
    return html,_po_pie(res["weights"],"⭐ Max Sharpe — Weight Allocation")

def po_min_variance(symbols_str,max_weight_pct,period,portfolio_value):
    syms=_parse_syms(symbols_str); max_w=max_weight_pct/100
    try: agent=_po_agent_instance(); df_ret=agent.fetch_returns_data(syms,period); res=agent.optimize_min_variance(df_ret,max_weight=max_w)
    except Exception as exc: return banner(f"⚠ Error: {exc}","alert"),None
    html=f"""{section_header("🛡️","Minimum Variance Portfolio",f"Period: {period} | Symbols: {', '.join(syms)}")}{banner("🛡️ Conservative allocation — lowest achievable risk","info")}{_po_metric_cards(res["expected_return"],res["volatility"],res["sharpe_ratio"])}<div class="section-header"><div class="section-title">📊 Optimal Weight Allocation</div></div>{_po_weights_table(res["weights"],portfolio_value)}"""
    return html,_po_pie(res["weights"],"🛡️ Min Variance — Weight Allocation")

def po_frontier(symbols_str,max_weight_pct,period,n_pts):
    syms=_parse_syms(symbols_str); max_w=max_weight_pct/100
    try: agent=_po_agent_instance(); df_ret=agent.fetch_returns_data(syms,period); fdf=agent.generate_efficient_frontier(df_ret,int(n_pts),max_w)
    except Exception as exc: return None,banner(f"⚠ Error: {exc}","alert")
    ms_pt=fdf.loc[fdf["sharpe"].idxmax()]; mv_pt=fdf.loc[fdf["volatility"].idxmin()]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=fdf["volatility"]*100,y=fdf["return"]*100,mode="markers+lines",marker=dict(color=fdf["sharpe"],colorscale="Blues",size=8,showscale=True,colorbar=dict(title="Sharpe Ratio",tickfont=dict(family="Inter",color=TEXT_MED)),line=dict(color="white",width=0.5)),line=dict(color="rgba(26,95,202,0.25)",width=1.5),name="Efficient Frontier",hovertemplate="<b>Efficient Portfolio</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<br>Sharpe: %{marker.color:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[ms_pt["volatility"]*100],y=[ms_pt["return"]*100],mode="markers+text",text=["⭐ Max Sharpe"],textposition="top center",textfont=dict(color=GOLD,size=11),marker=dict(color=GOLD,size=16,symbol="star",line=dict(color="white",width=2)),name=f"Max Sharpe ({ms_pt['sharpe']:.3f})"))
    fig.add_trace(go.Scatter(x=[mv_pt["volatility"]*100],y=[mv_pt["return"]*100],mode="markers+text",text=["🛡 Min Vol"],textposition="bottom center",textfont=dict(color=GREEN,size=11),marker=dict(color=GREEN,size=14,symbol="diamond",line=dict(color="white",width=2)),name="Min Variance"))
    fig.update_layout(**PLOTLY_THEME,title=dict(text=f"⬡ Markowitz Efficient Frontier — {' · '.join(syms)}",font=dict(color=BLUE_DARK,size=14,family="Inter, sans-serif")),xaxis=dict(**AXIS_STYLE,title="Annualised Volatility (%)"),yaxis=dict(**AXIS_STYLE,title="Annualised Return (%)"),height=530,hovermode="closest")
    info_html=f'<div class="metric-grid metric-grid-2" style="margin-top:14px"><div class="kpi-card gold"><div class="kpi-label">⭐ Max Sharpe Portfolio</div><div style="font-size:.9rem;color:{TEXT_DARK};margin-top:6px">Return: <strong>{ms_pt["return"]*100:.2f}%</strong> &nbsp;|&nbsp; Vol: <strong>{ms_pt["volatility"]*100:.2f}%</strong> &nbsp;|&nbsp; Sharpe: <strong>{ms_pt["sharpe"]:.3f}</strong></div></div><div class="kpi-card green"><div class="kpi-label">🛡️ Min Variance Portfolio</div><div style="font-size:.9rem;color:{TEXT_DARK};margin-top:6px">Return: <strong>{mv_pt["return"]*100:.2f}%</strong> &nbsp;|&nbsp; Vol: <strong>{mv_pt["volatility"]*100:.2f}%</strong> &nbsp;|&nbsp; Sharpe: <strong>{mv_pt["sharpe"]:.3f}</strong></div></div></div>'
    return fig,info_html

def po_signal_adjusted(symbols_str,max_weight_pct,portfolio_value):
    syms=_parse_syms(symbols_str); max_w=max_weight_pct/100
    try: agent=_po_agent_instance(); res=agent.optimize_with_signals(syms,portfolio_value,max_w)
    except Exception as exc: return banner(f"⚠ Error: {exc}","alert"),None
    if res["status"]!="success": return banner(f"⚠ {res.get('message','Unknown error')}","alert"),None
    sig_rows="".join(f'<tr><td style="font-weight:600">{sym}</td><td>{"🟢" if s>0 else "🔴" if s<0 else "⚪"}</td><td style="color:{"#0d9c5b" if s>0 else "#e03131" if s<0 else "#6b83a8"}">{s*100:+.1f}%</td></tr>' for sym,s in res["signals"].items())
    html=f"""{section_header("🎯","Signal-Adjusted Portfolio",f"Integrates Week 4 alpha signals | Portfolio: ${portfolio_value:,.0f}")}{_po_metric_cards(res["expected_return"],res["volatility"],res["sharpe_ratio"])}<div class="section-header" style="margin-top:18px"><div class="section-title">📡 Alpha Signals Used</div><div class="section-sub">Signal confidence drives ±20% weight adjustments from Max Sharpe base</div></div><table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:18px"><thead><tr style="background:{BLUE_DARK};color:#fff"><th style="padding:9px 14px;text-align:left">Symbol</th><th style="padding:9px 14px;text-align:left">Direction</th><th style="padding:9px 14px;text-align:left">Confidence</th></tr></thead><tbody>{sig_rows}</tbody></table><div class="section-header"><div class="section-title">📊 Adjusted Weight Allocation</div></div>{_po_weights_table(res["adjusted_weights"],portfolio_value)}"""
    fig=make_subplots(rows=1,cols=2,specs=[[{"type":"pie"},{"type":"pie"}]],subplot_titles=["Base Weights (Max Sharpe)","Signal-Adjusted Weights"])
    for col_idx,(w_dict) in enumerate([res["base_weights"],res["adjusted_weights"]],1):
        fig.add_trace(go.Pie(labels=list(w_dict.keys()),values=[round(v*100,2) for v in w_dict.values()],marker=dict(colors=PALETTE_PO,line=dict(color="white",width=2)),textinfo="label+percent",hole=0.35,hovertemplate="%{label}: %{value:.2f}%<extra></extra>"),row=1,col=col_idx)
    fig.update_layout(**PLOTLY_THEME,title=dict(text="🎯 Base vs Signal-Adjusted Weights",font=dict(color=BLUE_DARK,size=14,family="Inter, sans-serif")),showlegend=False,height=420)
    return html,fig

# ═══════════════════════════════════════════════════════════════════════════
#  APP — 10 top-level tabs, all visible (no ··· overflow)
#  Strategy: merge "Backtest" + "Raw Indicators" into one tab to keep
#  total count at 10, which wraps cleanly to 2 rows on any screen.
# ═══════════════════════════════════════════════════════════════════════════
HEADER_HTML="""
<div class="app-header">
    <div class="app-title">⬡ Portfolio Optimization Intelligence System</div>
    <div class="header-accent"></div>
    <div class="app-subtitle">Multi-Agent Risk &amp; Market Analytics Platform &nbsp;·&nbsp; v5.0</div>
    <div class="app-team">Team: Ashwini &nbsp;|&nbsp; Dibyendu Sarkar &nbsp;|&nbsp; Jyoti Ranjan Sethi &nbsp;|&nbsp; IIT Madras 2026</div>
</div>"""

FOOTER_HTML="""
<div class="app-footer">
    ⬡ Portfolio Intelligence System &nbsp;|&nbsp; IIT Madras 2026 &nbsp;|&nbsp;
    Data: Yahoo Finance &nbsp;|&nbsp; For Educational Use Only &nbsp;|&nbsp; Week 5 of 16
</div>"""

PO_HERO_HTML="""
<div class="po-hero">
    <div class="po-hero-title">⬡ Markowitz Mean-Variance Portfolio Optimization</div>
    <div class="po-hero-accent"></div>
    <div class="po-hero-sub">Construct mathematically optimal portfolios from your chosen asset universe. Pick a strategy, set your parameters, and optimise.</div>
    <div style="margin-top:14px">
        <span class="po-pill">⭐ Max Sharpe</span>
        <span class="po-pill">🛡️ Min Variance</span>
        <span class="po-pill">📈 Efficient Frontier</span>
        <span class="po-pill">🎯 Signal-Adjusted</span>
    </div>
</div>"""

with gr.Blocks(title="Portfolio Optimization Intelligence System — v5.0", css=CUSTOM_CSS) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        shared_symbol    = gr.Dropdown(choices=SYMBOLS, value="AAPL", label="🚀 Stock Symbol", scale=2)
        shared_period    = gr.Dropdown(choices=["1mo","3mo","6mo","1y","2y","5y"], value="1y", label="📅 Period", scale=1)
        shared_portfolio = gr.Number(value=100_000, label="💰 Portfolio Value ($)", minimum=1000, scale=2)

    with gr.Tabs():

        # 1 ── Market Overview
        with gr.Tab("📡 Market Overview"):
            overview_btn=gr.Button("🔄 Refresh Market Data",variant="primary",size="lg")
            status_out=gr.HTML(); cards_out=gr.HTML()
            with gr.Row(): price_out=gr.Plot(label="Stock Prices"); vol_out=gr.Plot(label="Trading Volume")
            with gr.Accordion("📋 Detailed Price Table",open=False): table_out=gr.Dataframe(interactive=False)
            overview_btn.click(fn=render_market_overview,outputs=[cards_out,price_out,vol_out,table_out,status_out])

        # 2 ── Historical Analysis
        with gr.Tab("📈 Historical Analysis"):
            hist_btn=gr.Button("📈 Load Historical Data",variant="primary",size="lg")
            hist_stat=gr.HTML(); candle=gr.Plot(label="Candlestick + MA20"); ret_chart=gr.Plot(label="Cumulative Return"); vol_hist=gr.Plot(label="Volume")
            hist_btn.click(fn=render_historical,inputs=[shared_symbol,shared_period],outputs=[candle,ret_chart,vol_hist,hist_stat])

        # 3 ── Risk Assessment
        with gr.Tab("🛡️ Risk Assessment"):
            risk_btn=gr.Button("🔍 Calculate Risk Metrics",variant="primary",size="lg"); risk_kpi=gr.HTML()
            with gr.Row(): gauge_out=gr.Plot(label="Risk Score Gauge"); dist_out=gr.Plot(label="Return Distribution")
            dd_out=gr.Plot(label="Drawdown Chart"); rv_out=gr.Plot(label="Rolling Volatility")
            risk_btn.click(fn=render_risk,inputs=[shared_symbol,shared_portfolio],outputs=[risk_kpi,dist_out,dd_out,rv_out,gauge_out])

        # 4 ── Stress Testing
        with gr.Tab("💥 Stress Testing"):
            stress_btn=gr.Button("💥 Run Stress Tests",variant="primary",size="lg")
            with gr.Row(): spct=gr.Plot(label="Loss % by Scenario"); susd=gr.Plot(label="Dollar Loss by Scenario")
            with gr.Accordion("📋 Full Stress Test Table",open=True): stbl=gr.Dataframe(interactive=False)
            stress_btn.click(fn=render_stress,inputs=[shared_symbol,shared_portfolio],outputs=[spct,susd,stbl])

        # 5 ── Correlation
        with gr.Tab("🔗 Correlation"):
            sym_in=gr.Textbox(value="AAPL,GOOGL,MSFT,TSLA,AMZN",label="Symbols (comma-separated)")
            corr_btn=gr.Button("🔗 Compute Correlations",variant="primary",size="lg"); corr_inf=gr.HTML()
            heat_out=gr.Plot(label="Correlation Heatmap"); cmp_out=gr.Plot(label="Cumulative Return Comparison")
            corr_btn.click(fn=render_correlation,inputs=[sym_in],outputs=[heat_out,corr_inf,cmp_out])

        # 6 ── Monte Carlo
        with gr.Tab("🎲 Monte Carlo"):
            with gr.Row(): mc_days=gr.Slider(21,504,value=252,step=21,label="Simulation Days"); mc_sims=gr.Slider(100,1000,value=500,step=100,label="Simulations")
            mc_btn=gr.Button("🎲 Run Monte Carlo Simulation",variant="primary",size="lg"); mc_stats=gr.HTML(); mc_chart=gr.Plot(label="Simulation Paths")
            mc_btn.click(fn=render_monte_carlo,inputs=[shared_symbol,shared_portfolio,mc_days,mc_sims],outputs=[mc_chart,mc_stats])

        # 7 ── Alpha Signal
        with gr.Tab("🎯 Alpha Signal"):
            signal_btn=gr.Button("🎯 Generate Trading Signal",variant="primary",size="lg"); signal_kpi=gr.HTML(); signal_chart=gr.Plot(label="Technical Indicator Dashboard")
            signal_btn.click(fn=render_alpha_signal,inputs=[shared_symbol,shared_portfolio],outputs=[signal_kpi,signal_chart])

        # 8 ── Backtest & Indicators  (merged — saves one tab slot)
        with gr.Tab("📊 Backtest & Indicators"):
            with gr.Tabs():
                with gr.Tab("📊 Backtest"):
                    with gr.Row(): bt_capital=gr.Number(value=10_000,label="💵 Initial Capital ($)",minimum=1000,scale=2); bt_period=gr.Dropdown(choices=["1mo","3mo","6mo","1y","2y"],value="1y",label="📅 Backtest Period",scale=1)
                    bt_btn=gr.Button("🚀 Run Backtest",variant="primary",size="lg"); bt_kpi=gr.HTML(); bt_chart=gr.Plot(label="Equity Curve")
                    bt_btn.click(fn=render_backtest,inputs=[shared_symbol,bt_capital,bt_period],outputs=[bt_kpi,bt_chart])
                with gr.Tab("🔍 Raw Indicators"):
                    ind_btn=gr.Button("📊 Get Indicators",variant="primary",size="lg"); ind_kpi=gr.HTML(); ind_chart=gr.Plot(label="Bollinger Bands Chart")
                    ind_btn.click(fn=render_indicators,inputs=[shared_symbol,shared_period],outputs=[ind_kpi,ind_chart])

        # 9 ── Portfolio Optimization  ← ALWAYS VISIBLE, polished hero layout
        with gr.Tab("⬡ Portfolio Optimization"):
            gr.HTML(PO_HERO_HTML)
            with gr.Row():
                po_symbols    = gr.Textbox(value="AAPL,GOOGL,MSFT,TSLA,AMZN", label="📌 Symbols (comma-separated)", scale=3)
                po_max_weight = gr.Slider(10, 100, value=40, step=5, label="⚖️ Max Weight per Asset (%)", scale=2)
            with gr.Tabs():
                with gr.Tab("⭐ Max Sharpe Ratio"):
                    gr.Markdown("**Maximise risk-adjusted returns** — optimal portfolio on the efficient frontier by Sharpe ratio.")
                    ms_btn=gr.Button("🚀 Optimize Max Sharpe",variant="primary",size="lg"); ms_html=gr.HTML(); ms_chart=gr.Plot(label="Weight Allocation")
                    ms_btn.click(fn=po_max_sharpe,inputs=[po_symbols,po_max_weight,shared_period,shared_portfolio],outputs=[ms_html,ms_chart])
                with gr.Tab("🛡️ Min Variance"):
                    gr.Markdown("**Minimise portfolio risk** — ideal for conservative investors seeking the lowest achievable volatility.")
                    mv_btn=gr.Button("🚀 Optimize Min Variance",variant="primary",size="lg"); mv_html=gr.HTML(); mv_chart=gr.Plot(label="Weight Allocation")
                    mv_btn.click(fn=po_min_variance,inputs=[po_symbols,po_max_weight,shared_period,shared_portfolio],outputs=[mv_html,mv_chart])
                with gr.Tab("📈 Efficient Frontier"):
                    gr.Markdown("**Visualise all optimal portfolios** across the risk-return spectrum.")
                    ef_n_pts=gr.Slider(10,100,value=40,step=10,label="Number of Frontier Points")
                    ef_btn=gr.Button("🚀 Generate Frontier",variant="primary",size="lg"); ef_chart=gr.Plot(label="Efficient Frontier"); ef_info=gr.HTML()
                    ef_btn.click(fn=po_frontier,inputs=[po_symbols,po_max_weight,shared_period,ef_n_pts],outputs=[ef_chart,ef_info])
                with gr.Tab("🎯 Signal-Adjusted"):
                    gr.Markdown("**Integrate Week 4 alpha signals** to tilt Max-Sharpe base weights by signal confidence.")
                    sa_btn=gr.Button("🚀 Signal-Adjusted Optimize",variant="primary",size="lg"); sa_html=gr.HTML(); sa_chart=gr.Plot(label="Base vs Adjusted Weights")
                    sa_btn.click(fn=po_signal_adjusted,inputs=[po_symbols,po_max_weight,shared_portfolio],outputs=[sa_html,sa_chart])

        # 10 ── About  ← ALWAYS VISIBLE
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## ⬡ Portfolio Optimization Intelligence System — v5.0

A **unified multi-agent AI-powered platform** combining market data, risk management, alpha signal generation, and Markowitz portfolio optimization.

---
### 🧩 Modules

| Tab | Description |
|---|---|
| 📡 **Market Overview** | Real-time prices, volume, KPI cards |
| 📈 **Historical Analysis** | Candlestick · MA20 · cumulative returns · volume |
| 🛡️ **Risk Assessment** | VaR · CVaR · Sharpe · Drawdown · Volatility · Gauge |
| 💥 **Stress Testing** | 8 crash scenarios — loss % and dollar loss |
| 🔗 **Correlation** | Heatmap + cumulative return comparison |
| 🎲 **Monte Carlo** | Up to 1000-path simulation with confidence bands |
| 🎯 **Alpha Signal** | Buy/Sell/Hold signal via technical analysis |
| 📊 **Backtest & Indicators** | Strategy equity curve · RSI · MACD · Bollinger Bands |
| ⬡ **Portfolio Optimization** | Max Sharpe · Min Variance · Efficient Frontier · Signal-Adjusted |

---
### ⚠️ Notes
- Tabs 1–6 & 9 fetch **directly from Yahoo Finance** (no extra API needed)
- Tabs 7–8 require **Alpha Signal API on port 8002**
- All data is for **educational purposes only**

**Team:** Ashwini · Dibyendu Sarkar · Jyoti Ranjan Sethi · IIT Madras · Week 5 of 16
            """)

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    print("Starting Portfolio Optimization Intelligence System v5.0 ...")
    print("Tabs 1-6 & 9: Direct Yahoo Finance (no API needed)")
    print("Tabs 7-8:     Require Alpha Signal API on port 8002")
    demo.launch(server_name="0.0.0.0", server_port=7865, share=True, show_error=True)