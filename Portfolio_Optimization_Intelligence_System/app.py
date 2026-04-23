import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

# ─────────────────────────────────────────────
#  Color Palette
# ─────────────────────────────────────────────
BG_CARD      = "#ffffff"
BORDER       = "#e2e8f0"
BLUE_PRIMARY = "#2563eb"
BLUE_DARK    = "#1e3a8a"
BLUE_LIGHT   = "#eff6ff"
GREEN        = "#059669"
RED          = "#dc2626"
GOLD         = "#d97706"
TEXT_DARK    = "#0f172a"
TEXT_MED     = "#475569"
TEXT_LIGHT   = "#94a3b8"

PLOTLY_THEME = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8faff",
    font=dict(family="DM Sans, sans-serif", color=TEXT_DARK, size=12),
    legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor=BORDER,
                borderwidth=1, font=dict(color=TEXT_DARK)),
    margin=dict(l=55, r=30, t=55, b=45),
)
AXIS_STYLE = dict(
    gridcolor="#e8f0fb", zerolinecolor="#cbd5e1",
    tickfont=dict(color=TEXT_LIGHT), linecolor=BORDER,
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container, .gradio-container * {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Page background ── */
.gradio-container {
    background: #f1f5f9 !important;
    max-width: 100% !important;
    padding: 0 !important;
}

/* ── Hero header ── */
.pf-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #1d4ed8 100%);
    padding: 40px 48px 32px;
    position: relative;
    overflow: hidden;
    border-bottom: 3px solid #f59e0b;
}
.pf-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.pf-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 10%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(251,191,36,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.pf-logo {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.pf-logo-hex {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 800; color: #0f172a;
}
.pf-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.pf-title span { color: #fbbf24; }
.pf-sub {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.55) !important;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 10px;
}
.pf-pills {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.pf-pill {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.75) !important;
    background: rgba(255,255,255,0.07);
}

/* ── Control bar ── */
.pf-controls {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 14px 24px;
    box-shadow: 0 2px 8px rgba(15,23,42,0.06);
}

/* ── Tabs ── */
div[role="tablist"] {
    background: #ffffff !important;
    border-bottom: 2px solid #e2e8f0 !important;
    padding: 0 20px !important;
    gap: 0 !important;
}
div[role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.81rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 14px 18px !important;
    background: transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}
div[role="tab"]:hover {
    color: #2563eb !important;
    background: #eff6ff !important;
}
div[role="tab"][aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 3px solid #2563eb !important;
    font-weight: 700 !important;
    background: #eff6ff !important;
}

/* ── Tab content wrapper ── */
.tab-content-wrap {
    padding: 24px;
    background: #f1f5f9;
    min-height: 400px;
}

/* ── KPI cards ── */
.kpi-row {
    display: grid;
    gap: 14px;
    margin-bottom: 20px;
}
.kpi-row-5 { grid-template-columns: repeat(5, 1fr); }
.kpi-row-4 { grid-template-columns: repeat(4, 1fr); }
.kpi-row-3 { grid-template-columns: repeat(3, 1fr); }

.kpi {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 20px 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 6px rgba(15,23,42,0.05);
    transition: box-shadow 0.2s, transform 0.2s;
}
.kpi:hover {
    box-shadow: 0 6px 20px rgba(15,23,42,0.10);
    transform: translateY(-2px);
}
.kpi-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #2563eb, #60a5fa);
    border-radius: 14px 14px 0 0;
}
.kpi-accent.g { background: linear-gradient(90deg, #059669, #34d399); }
.kpi-accent.r { background: linear-gradient(90deg, #dc2626, #f87171); }
.kpi-accent.o { background: linear-gradient(90deg, #d97706, #fbbf24); }

.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
}
.kpi-val {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1;
}
.kpi-val.g { color: #059669; }
.kpi-val.r { color: #dc2626; }
.kpi-val.o { color: #d97706; }
.kpi-sub {
    font-size: 0.71rem;
    color: #94a3b8;
    margin-top: 5px;
    font-weight: 500;
}
.kpi-sub.g { color: #059669; }
.kpi-sub.r { color: #dc2626; }

/* ── Section header ── */
.sec-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #e2e8f0;
}
.sec-hdr-icon {
    width: 34px; height: 34px;
    background: #eff6ff;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.sec-hdr-text { flex: 1; }
.sec-hdr-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
}
.sec-hdr-sub {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* ── Banners ── */
.bn {
    border-radius: 10px;
    padding: 11px 16px;
    font-size: 0.81rem;
    font-weight: 500;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.bn-ok    { background: #ecfdf5; border: 1px solid #6ee7b7; color: #065f46; border-left: 4px solid #059669; }
.bn-warn  { background: #fffbeb; border: 1px solid #fcd34d; color: #78350f; border-left: 4px solid #d97706; }
.bn-err   { background: #fef2f2; border: 1px solid #fca5a5; color: #7f1d1d; border-left: 4px solid #dc2626; }
.bn-info  { background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a; border-left: 4px solid #2563eb; }

/* ── Badge ── */
.bdg {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.6px; text-transform: uppercase;
    margin-bottom: 14px;
}
.bdg-ok   { background: #ecfdf5; color: #065f46; border: 1px solid #6ee7b7; }
.bdg-warn { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }

/* ── Chart wrapper ── */
.chart-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 1px 6px rgba(15,23,42,0.04);
    margin-bottom: 16px;
}

/* ── Buttons ── */
button.primary, .gr-button-primary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.3px !important;
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.30) !important;
    transition: all 0.2s !important;
    padding: 10px 24px !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.40) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs / dropdowns ── */
input, select, textarea, .gr-dropdown {
    font-family: 'DM Sans', sans-serif !important;
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
input:focus, select:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.10) !important;
    outline: none !important;
}
label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.73rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
}

/* ── Tables ── */
table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    background: #ffffff !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 0.82rem !important;
    box-shadow: 0 1px 6px rgba(15,23,42,0.04) !important;
}
th {
    background: #1e3a8a !important;
    color: #ffffff !important;
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 12px 16px !important;
    white-space: nowrap !important;
}
td {
    color: #0f172a !important;
    padding: 10px 16px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    background: #ffffff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.80rem !important;
}
tr:nth-child(even) td { background: #f8fafc !important; }
tr:hover td { background: #eff6ff !important; }

/* ── Slider ── */
input[type="range"] { accent-color: #2563eb !important; }

/* ── Footer ── */
.pf-footer {
    background: #0f172a;
    color: rgba(255,255,255,0.4);
    text-align: center;
    padding: 20px;
    font-size: 0.70rem;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-top: 32px;
    border-top: 2px solid #1e3a8a;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2563eb; }

/* ── Accordion ── */
.gr-accordion {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .kpi-row-5 { grid-template-columns: repeat(2, 1fr) !important; }
    .kpi-row-4 { grid-template-columns: repeat(2, 1fr) !important; }
    .pf-title  { font-size: 1.4rem !important; }
    .pf-hero   { padding: 24px 20px !important; }
}
"""

# ─────────────────────────────────────────────
#  HTML helpers
# ─────────────────────────────────────────────
def kpi(label, value, sub="", accent="b"):
    ac = {"g": "g", "r": "r", "o": "o"}.get(accent, "")
    vc = ac
    sc = "g" if ("▲" in sub or "+" in sub) else "r" if ("▼" in sub or (sub.startswith("-") and sub != "-")) else ""
    s_html = f'<div class="kpi-sub {sc}">{sub}</div>' if sub else ""
    return f"""<div class="kpi">
        <div class="kpi-accent {ac}"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-val {vc}">{value}</div>
        {s_html}
    </div>"""

def sec(icon, title, sub=""):
    s = f'<div class="sec-hdr-sub">{sub}</div>' if sub else ""
    return f"""<div class="sec-hdr">
        <div class="sec-hdr-icon">{icon}</div>
        <div class="sec-hdr-text">
            <div class="sec-hdr-title">{title}</div>
            {s}
        </div>
    </div>"""

def banner(msg, kind="info"):
    icons = {"ok": "✅", "warn": "⚠️", "err": "🚫", "info": "ℹ️"}
    css   = {"ok": "bn-ok", "warn": "bn-warn", "err": "bn-err", "info": "bn-info"}
    return f'<div class="bn {css.get(kind,"bn-info")}">{icons.get(kind,"ℹ️")} {msg}</div>'

SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'RELIANCE.NS']

# ─────────────────────────────────────────────
#  Data helpers
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
    if s > 3:   return "Exceptional", "g"
    if s > 2:   return "Very Good",   "g"
    if s > 1:   return "Good",        "b"
    if s > 0.5: return "Acceptable",  "o"
    if s > 0:   return "Poor",        "o"
    return "Losing Money", "r"

def apply_theme(fig, title_text=None, yaxis_title=None, xaxis_title=None, extra=None):
    layout = dict(**PLOTLY_THEME)
    layout['xaxis'] = dict(**AXIS_STYLE)
    layout['yaxis'] = dict(**AXIS_STYLE)
    if title_text:
        layout['title'] = dict(text=title_text, font=dict(color="#0f172a", size=13, family="Syne, sans-serif"))
    if yaxis_title: layout['yaxis']['title'] = dict(text=yaxis_title, font=dict(color=TEXT_MED))
    if xaxis_title: layout['xaxis']['title'] = dict(text=xaxis_title, font=dict(color=TEXT_MED))
    if extra: layout.update(extra)
    fig.update_layout(**layout)
    return fig

# ═══════════════════════════════════════════════
#  TAB RENDER FUNCTIONS
# ═══════════════════════════════════════════════

def render_market_overview():
    data = get_realtime(SYMBOLS)
    ts   = datetime.now().strftime('%d %b %Y  %H:%M:%S')

    # KPI cards
    cards = '<div class="kpi-row kpi-row-5" style="margin-bottom:20px">'
    for sym, d in data.items():
        chg  = d.get('change_pct', 0)
        sign = "▲" if chg >= 0 else "▼"
        acc  = "g" if chg >= 0 else "r"
        display_sym = sym.replace('.NS', '')
        cards += kpi(display_sym, f"${d['price']:.2f}" if d['price'] else "—",
                     f"{sign} {abs(chg):.2f}%", acc)
    cards += "</div>"

    prices  = {s.replace('.NS',''): d['price'] for s, d in data.items() if d['price']}
    changes = {s.replace('.NS',''): d['change_pct'] for s, d in data.items()}
    bcolors = [GREEN if changes.get(s,0) >= 0 else RED for s in prices]

    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(
        x=list(prices.keys()), y=list(prices.values()),
        marker=dict(color=bcolors, line=dict(color='white', width=1.5),
                    opacity=0.9),
        text=[f"${v:.2f}" for v in prices.values()],
        textposition='outside', textfont=dict(size=11, color=TEXT_DARK, family="DM Mono"),
        hovertemplate="<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>",
    ))
    apply_theme(fig_p, title_text="Current Stock Prices (USD)", yaxis_title="Price ($)",
                extra={"showlegend": False, "bargap": 0.3})

    vols = {s.replace('.NS',''): d.get('volume', 0) for s, d in data.items()}
    fig_v = go.Figure()
    fig_v.add_trace(go.Bar(
        x=list(vols.keys()), y=list(vols.values()),
        marker=dict(color=list(vols.values()),
                    colorscale=[[0,"#bfdbfe"],[1,"#1d4ed8"]],
                    showscale=False, line=dict(color='white', width=1.5), opacity=0.9),
        text=[f"{v/1e6:.1f}M" for v in vols.values()],
        textposition='outside', textfont=dict(size=11, color=TEXT_DARK, family="DM Mono"),
        hovertemplate="<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>",
    ))
    apply_theme(fig_v, title_text="Trading Volume", yaxis_title="Volume",
                extra={"showlegend": False, "bargap": 0.3})

    rows = []
    for s, d in data.items():
        chg = d.get('change_pct', 0)
        rows.append({
            'Symbol': s.replace('.NS',''),
            'Price ($)': f"${d['price']:.2f}" if d['price'] else "—",
            'Open ($)':  f"${d['open']:.2f}"  if d['open']  else "—",
            'High ($)':  f"${d['high']:.2f}"  if d['high']  else "—",
            'Low ($)':   f"${d['low']:.2f}"   if d['low']   else "—",
            'Volume':    f"{d['volume']/1e6:.1f}M" if d['volume'] else "—",
            'Mkt Cap':   f"${d['market_cap']/1e12:.2f}T" if d.get('market_cap') else "—",
            'P/E':       f"{d['pe_ratio']:.1f}" if d.get('pe_ratio') else "—",
            'Change':    f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%",
        })

    return (
        cards, fig_p, fig_v, pd.DataFrame(rows),
        banner(f"Data refreshed at {ts}", "ok")
    )


def render_historical(symbol, period):
    df = get_historical(symbol, period)
    if df.empty:
        return None, None, None, banner("No data available for this symbol/period.", "err")

    fig_c = go.Figure()
    fig_c.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing=dict(line=dict(color=GREEN, width=1.2), fillcolor="rgba(5,150,105,0.20)"),
        decreasing=dict(line=dict(color=RED,   width=1.2), fillcolor="rgba(220,38,38,0.20)"),
        name="OHLC",
    ))
    ma20 = df['Close'].rolling(20).mean()
    fig_c.add_trace(go.Scatter(
        x=df.index, y=ma20, name="MA 20",
        line=dict(color=BLUE_PRIMARY, width=1.8, dash='dot'),
    ))
    apply_theme(fig_c, title_text=f"{symbol} — Price Chart with MA20 ({period})",
                yaxis_title="Price (USD)", extra={"xaxis_rangeslider_visible": False})

    returns = df['Close'].pct_change().dropna()
    cum_ret = (1 + returns).cumprod() - 1
    col_ret = GREEN if cum_ret.iloc[-1] >= 0 else RED
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=cum_ret.index, y=cum_ret * 100, fill='tozeroy',
        fillcolor="rgba(5,150,105,0.08)" if cum_ret.iloc[-1] >= 0 else "rgba(220,38,38,0.08)",
        line=dict(color=col_ret, width=2.2), name="Cumulative Return",
        hovertemplate="%{x|%b %d, %Y}<br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig_r.add_hline(y=0, line=dict(color=TEXT_LIGHT, dash='dash', width=1))
    apply_theme(fig_r, title_text="Cumulative Return (%)", yaxis_title="Return (%)")

    vcols = [GREEN if c >= o else RED for c, o in zip(df['Close'], df['Open'])]
    fig_v = go.Figure()
    fig_v.add_trace(go.Bar(
        x=df.index, y=df['Volume'], marker_color=vcols, name="Volume",
        opacity=0.8,
        hovertemplate="%{x|%b %d}<br>Vol: %{y:,.0f}<extra></extra>",
    ))
    apply_theme(fig_v, title_text="Volume (Green = Up Day · Red = Down Day)", yaxis_title="Volume")

    total = cum_ret.iloc[-1] * 100
    sign  = "▲" if total >= 0 else "▼"
    acc   = "g" if total >= 0 else "r"
    stats = f"""
    {sec("📈", f"Historical Analysis — {symbol}", f"Period: {period}  ·  {len(df)} trading days")}
    <div class="kpi-row kpi-row-4">
        {kpi("Current Price", f"${df['Close'].iloc[-1]:.2f}", "", "b")}
        {kpi("Period High",   f"${df['High'].max():.2f}",    "", "g")}
        {kpi("Period Low",    f"${df['Low'].min():.2f}",     "", "r")}
        {kpi("Total Return",  f"{sign} {abs(total):.2f}%",  "", acc)}
    </div>"""

    return fig_c, fig_r, fig_v, stats


def render_risk(symbol, portfolio_value):
    returns = get_returns(symbol, "1y")
    if returns.empty:
        return banner("Could not fetch data for this symbol.", "err"), None, None, None, None

    m = compute_risk(returns, portfolio_value)
    slabel, scol = sharpe_label(m['sharpe'])
    risk_ok = m['annual_vol'] < 0.30 and m['max_dd'] < 0.20 and m['sharpe'] > 1.0
    badge = f'<span class="bdg {"bdg-ok" if risk_ok else "bdg-warn"}">{"✓ Within Limits" if risk_ok else "⚠ Risk Alert"}</span>'

    kpi_html = f"""
    {sec("🛡️", f"Risk Assessment — {symbol}", f"Portfolio: ${portfolio_value:,.0f}  ·  {len(returns)} trading days")}
    {badge}
    <div class="kpi-row kpi-row-4" style="margin-top:12px">
        {kpi("VaR 95%",    f"{m['var_95']:.2%}",    f"−${m['var_95_usd']:,.0f} / day", "r")}
        {kpi("VaR 99%",    f"{m['var_99']:.2%}",    f"−${m['var_99_usd']:,.0f} / day", "r")}
        {kpi("CVaR 95%",   f"{m['cvar_95']:.2%}",   "Expected Shortfall",               "r")}
        {kpi("Annual Vol", f"{m['annual_vol']:.2%}", f"Daily: {m['daily_vol']:.2%}",
             "o" if m['annual_vol'] > 0.25 else "b")}
    </div>
    <div class="kpi-row kpi-row-4">
        {kpi("Max Drawdown",  f"{m['max_dd']:.2%}",  "Peak-to-Trough",
             "r" if m['max_dd'] > 0.20 else "o")}
        {kpi("Sharpe Ratio",  f"{m['sharpe']:.2f}",  slabel, scol)}
        {kpi("Annual Return", f"{m['ann_ret']:.2%}",
             "▲ Positive" if m['ann_ret'] >= 0 else "▼ Negative",
             "g" if m['ann_ret'] >= 0 else "r")}
        {kpi("Data Points",   str(len(returns)),     "Trading Days", "b")}
    </div>"""

    # Distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=returns * 100, nbinsx=55,
        marker=dict(color=BLUE_PRIMARY, opacity=0.75, line=dict(color='white', width=0.5)),
        name="Daily Returns",
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    fig_dist.add_vline(x=-m['var_95']*100, line=dict(color=GOLD, dash='dash', width=2),
                       annotation=dict(text="VaR 95%", font=dict(color=GOLD, size=10)))
    fig_dist.add_vline(x=-m['var_99']*100, line=dict(color=RED,  dash='dash', width=2),
                       annotation=dict(text="VaR 99%", font=dict(color=RED,  size=10)))
    apply_theme(fig_dist, title_text="Return Distribution + VaR Lines",
                xaxis_title="Daily Return (%)", yaxis_title="Frequency")

    # Drawdown
    dd = m['drawdown']
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd * 100, fill='tozeroy',
        fillcolor="rgba(220,38,38,0.10)",
        line=dict(color=RED, width=1.8), name="Drawdown %",
        hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig_dd.add_hline(y=-m['max_dd']*100, line=dict(color=GOLD, dash='dot', width=1.5),
                     annotation=dict(text=f"Max DD {m['max_dd']:.2%}", font=dict(color=GOLD, size=10)))
    apply_theme(fig_dd, title_text="Underwater Drawdown Chart", yaxis_title="Drawdown (%)")

    # Rolling vol
    rv = returns.rolling(21).std() * np.sqrt(252) * 100
    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(
        x=rv.index, y=rv, fill='tozeroy', fillcolor="rgba(37,99,235,0.08)",
        line=dict(color=BLUE_PRIMARY, width=2), name="21-day Vol",
        hovertemplate="%{x|%b %d, %Y}<br>Vol: %{y:.2f}%<extra></extra>",
    ))
    fig_rv.add_hline(y=30, line=dict(color=RED, dash='dash', width=1.3),
                     annotation=dict(text="Risk Limit 30%", font=dict(color=RED, size=10)))
    apply_theme(fig_rv, title_text="Rolling 21-Day Annualised Volatility", yaxis_title="Volatility (%)")

    # Gauge
    risk_score = min(100, m['annual_vol']/0.5*40 + m['max_dd']/0.5*40 + max(0,1-m['sharpe'])*20)
    gcol = GREEN if risk_score < 40 else GOLD if risk_score < 70 else RED
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title=dict(text="COMPOSITE RISK SCORE", font=dict(family="DM Sans", size=11, color=TEXT_MED)),
        number=dict(font=dict(family="Syne", size=36, color=gcol)),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor=TEXT_LIGHT,
                      tickfont=dict(family="DM Sans", size=10, color=TEXT_LIGHT)),
            bar=dict(color=gcol, thickness=0.22),
            bgcolor="#ffffff", borderwidth=1, bordercolor=BORDER,
            steps=[dict(range=[0,40],  color="rgba(5,150,105,0.07)"),
                   dict(range=[40,70], color="rgba(217,119,6,0.07)"),
                   dict(range=[70,100],color="rgba(220,38,38,0.07)")],
            threshold=dict(line=dict(color=gcol, width=3), thickness=0.75, value=risk_score),
        ),
    ))
    fig_g.update_layout(paper_bgcolor="#ffffff", font=dict(family="DM Sans", color=TEXT_DARK),
                        height=260, margin=dict(l=30,r=30,t=60,b=20))

    return kpi_html, fig_dist, fig_dd, fig_rv, fig_g


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
                     'Portfolio After': f"${shocked:,.0f}",
                     'Loss Amount': f"${loss:,.0f}",
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
        marker=dict(color=[sev(l) for l in pcts], line=dict(color='white', width=1),
                    opacity=0.85),
        text=[f"{l:.0f}%" for l in pcts], textposition='outside',
        textfont=dict(size=10, color=TEXT_DARK, family="DM Mono"),
        hovertemplate="<b>%{x}</b><br>Loss: %{y:.1f}%<extra></extra>",
    ))
    apply_theme(fig_pct, title_text="Portfolio Loss % by Scenario", yaxis_title="Loss (%)",
                extra={"yaxis": dict(**AXIS_STYLE, range=[min(pcts)*1.3, 5]), "bargap": 0.3})

    fig_usd = go.Figure()
    fig_usd.add_trace(go.Bar(
        x=labels, y=dloss,
        marker=dict(color=dloss, colorscale=[[0,"#bfdbfe"],[0.5,GOLD],[1,RED]],
                    showscale=False, line=dict(color='white', width=1), opacity=0.85),
        text=[f"${l:,.0f}" for l in dloss], textposition='outside',
        textfont=dict(size=10, color=TEXT_DARK, family="DM Mono"),
        hovertemplate="<b>%{x}</b><br>Loss: $%{y:,.0f}<extra></extra>",
    ))
    apply_theme(fig_usd, title_text="Dollar Loss by Scenario", yaxis_title="Loss ($)",
                extra={"bargap": 0.3})

    return fig_pct, fig_usd, pd.DataFrame(rows)


def render_correlation(symbols_str):
    syms = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
    if len(syms) < 2:
        return None, banner("Enter at least 2 comma-separated symbols.", "warn"), None

    all_ret = {}
    for s in syms:
        r = get_returns(s, "1y")
        if not r.empty:
            all_ret[s] = r

    if len(all_ret) < 2:
        return None, banner("Could not fetch data for enough symbols.", "err"), None

    df_ret = pd.DataFrame(all_ret).dropna()
    corr   = df_ret.corr()

    fig_h = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0, RED],[0.5,"#f8fafc"],[1, BLUE_PRIMARY]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(family="DM Mono", size=12, color=TEXT_DARK),
        hovertemplate="<b>%{x} vs %{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(family="DM Sans", color=TEXT_MED),
                      title=dict(text="r", font=dict(color=TEXT_MED))),
    ))
    apply_theme(fig_h, title_text="Correlation Matrix — 1Y Daily Returns")

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
        msg, kind = f"Well Diversified — avg pairwise correlation: {avg_corr:.3f}", "ok"
    elif avg_corr < 0.7:
        msg, kind = f"Moderately Correlated — avg correlation: {avg_corr:.3f}", "warn"
    else:
        msg, kind = f"Highly Correlated — low diversification benefit (r = {avg_corr:.3f})", "err"

    return fig_h, banner(msg, kind), fig_cr


def render_monte_carlo(symbol, portfolio_value, days, sims):
    days, sims = int(days), int(sims)
    returns = get_returns(symbol, "1y")
    if returns.empty:
        return None, banner("Could not fetch data.", "err")

    mu, sigma = returns.mean(), returns.std()
    np.random.seed(42)
    sim_rets   = np.random.normal(mu, sigma, (days, sims))
    sim_paths  = portfolio_value * np.exp(np.cumsum(np.log(1 + sim_rets), axis=0))
    final_vals = sim_paths[-1]

    fig = go.Figure()
    x_ax = list(range(days))
    for i in range(min(200, sims)):
        col = "rgba(5,150,105,0.10)" if sim_paths[-1,i] >= portfolio_value else "rgba(220,38,38,0.08)"
        fig.add_trace(go.Scatter(x=x_ax, y=sim_paths[:,i], mode='lines',
                                 line=dict(color=col, width=0.5),
                                 showlegend=False, hoverinfo='skip'))

    med_path = np.median(sim_paths, axis=1)
    fig.add_trace(go.Scatter(x=x_ax, y=med_path, mode='lines',
                             line=dict(color=BLUE_PRIMARY, width=2.8), name="Median Path"))

    p5  = np.percentile(sim_paths, 5,  axis=1)
    p95 = np.percentile(sim_paths, 95, axis=1)
    fig.add_trace(go.Scatter(
        x=x_ax + x_ax[::-1], y=list(p95)+list(p5[::-1]),
        fill='toself', fillcolor="rgba(37,99,235,0.06)",
        line=dict(color='rgba(0,0,0,0)'), name="90% Confidence Band",
    ))
    fig.add_hline(y=portfolio_value, line=dict(color=TEXT_LIGHT, dash='dash', width=1.5),
                  annotation=dict(text="Initial Value", font=dict(color=TEXT_LIGHT, size=10)))
    apply_theme(fig, title_text=f"Monte Carlo Simulation — {sims:,} Paths · {days} Trading Days",
                yaxis_title="Portfolio Value ($)", xaxis_title="Trading Day")

    med_fin    = np.median(final_vals)
    p5_fin     = np.percentile(final_vals, 5)
    p95_fin    = np.percentile(final_vals, 95)
    pct_profit = (final_vals >= portfolio_value).mean() * 100
    med_ret    = (med_fin / portfolio_value - 1) * 100
    sign       = "▲" if med_ret >= 0 else "▼"

    stats = f"""
    {sec("🎲", f"Monte Carlo Results — {symbol}", f"{sims:,} simulations · {days} trading days")}
    <div class="kpi-row kpi-row-4">
        {kpi("Median Outcome",   f"${med_fin:,.0f}",
             f"{sign} {abs(med_ret):.1f}%", "g" if med_ret >= 0 else "r")}
        {kpi("Best Case (95th)", f"${p95_fin:,.0f}",
             f"+{(p95_fin/portfolio_value-1)*100:.1f}%", "g")}
        {kpi("Worst Case (5th)", f"${p5_fin:,.0f}",
             f"{(p5_fin/portfolio_value-1)*100:.1f}%", "r")}
        {kpi("% Profitable",     f"{pct_profit:.1f}%",
             f"of {sims:,} simulations", "g" if pct_profit >= 50 else "r")}
    </div>"""

    return fig, stats


# ═══════════════════════════════════════════════
#  BUILD APP
# ═══════════════════════════════════════════════

HEADER_HTML = """
<div class="pf-hero">
    <div class="pf-logo">
        <div class="pf-logo-hex">⬡</div>
        <div>
            <div class="pf-title">Portfolio <span>Intelligence</span> System</div>
        </div>
    </div>
    <div class="pf-sub">Multi-Agent Risk &amp; Market Analytics Platform · v3.0</div>
    <div class="pf-pills">
        <span class="pf-pill">📡 Real-Time Data</span>
        <span class="pf-pill">🛡️ Risk Metrics</span>
        <span class="pf-pill">🎲 Monte Carlo</span>
        <span class="pf-pill">📊 Stress Testing</span>
        <span class="pf-pill">🔗 Correlation</span>
        <span class="pf-pill">IIT Madras 2026</span>
    </div>
</div>
"""

FOOTER_HTML = """
<div class="pf-footer">
    ⬡ Portfolio Intelligence System &nbsp;·&nbsp; IIT Madras 2026 &nbsp;·&nbsp;
    Ashwini · Dibyendu Sarkar · Jyoti Ranjan Sethi &nbsp;·&nbsp;
    Data: Yahoo Finance &nbsp;·&nbsp; Educational Use Only
</div>
"""

with gr.Blocks(title="Portfolio Intelligence System", css=CUSTOM_CSS) as demo:

    gr.HTML(HEADER_HTML)

    # ── Global controls ──
    with gr.Row(elem_classes=["pf-controls"]):
        shared_symbol = gr.Dropdown(
            choices=['AAPL','GOOGL','MSFT','TSLA','AMZN','RELIANCE.NS'],
            value="AAPL", label="📌 Stock Symbol", scale=2,
        )
        shared_period = gr.Dropdown(
            choices=["1mo","3mo","6mo","1y","2y","5y"],
            value="1y", label="📅 Period", scale=1,
        )
        shared_portfolio = gr.Number(
            value=100_000, label="💰 Portfolio Value ($)",
            minimum=1000, scale=2,
        )

    with gr.Tabs():

        # ── Tab 1: Market Overview ──
        with gr.Tab("📡 Market Overview"):
            with gr.Row():
                overview_btn = gr.Button("🔄 Refresh Market Data", variant="primary", size="lg", scale=1)
            status_out = gr.HTML()
            cards_out  = gr.HTML()
            with gr.Row():
                price_out = gr.Plot(label="Stock Prices", show_label=False)
                vol_out   = gr.Plot(label="Trading Volume", show_label=False)
            with gr.Accordion("📋 Full Price Table", open=False):
                table_out = gr.Dataframe(interactive=False)
            overview_btn.click(
                fn=render_market_overview,
                outputs=[cards_out, price_out, vol_out, table_out, status_out],
            )

        # ── Tab 2: Historical Analysis ──
        with gr.Tab("📈 Historical Analysis"):
            hist_btn  = gr.Button("📈 Load Historical Data", variant="primary", size="lg")
            hist_stat = gr.HTML()
            with gr.Row():
                candle    = gr.Plot(label="Candlestick", show_label=False)
                ret_chart = gr.Plot(label="Cumulative Return", show_label=False)
            vol_hist = gr.Plot(label="Volume", show_label=False)
            hist_btn.click(
                fn=render_historical,
                inputs=[shared_symbol, shared_period],
                outputs=[candle, ret_chart, vol_hist, hist_stat],
            )

        # ── Tab 3: Risk Assessment ──
        with gr.Tab("🛡️ Risk Assessment"):
            risk_btn = gr.Button("🔍 Calculate Risk Metrics", variant="primary", size="lg")
            risk_kpi = gr.HTML()
            with gr.Row():
                gauge_out = gr.Plot(label="Risk Gauge", show_label=False)
                dist_out  = gr.Plot(label="Distribution", show_label=False)
            with gr.Row():
                dd_out = gr.Plot(label="Drawdown", show_label=False)
                rv_out = gr.Plot(label="Rolling Vol", show_label=False)
            risk_btn.click(
                fn=render_risk,
                inputs=[shared_symbol, shared_portfolio],
                outputs=[risk_kpi, dist_out, dd_out, rv_out, gauge_out],
            )

        # ── Tab 4: Stress Testing ──
        with gr.Tab("💥 Stress Testing"):
            stress_btn = gr.Button("💥 Run Stress Tests", variant="primary", size="lg")
            with gr.Row():
                spct = gr.Plot(label="Loss %", show_label=False)
                susd = gr.Plot(label="Dollar Loss", show_label=False)
            with gr.Accordion("📋 Full Stress Test Table", open=True):
                stbl = gr.Dataframe(interactive=False)
            stress_btn.click(
                fn=render_stress,
                inputs=[shared_symbol, shared_portfolio],
                outputs=[spct, susd, stbl],
            )

        # ── Tab 5: Correlation ──
        with gr.Tab("🔗 Correlation"):
            with gr.Row():
                sym_in   = gr.Textbox(
                    value="AAPL,GOOGL,MSFT,TSLA,AMZN",
                    label="Symbols (comma-separated)", scale=4,
                )
                corr_btn = gr.Button("🔗 Compute", variant="primary", scale=1)
            corr_inf = gr.HTML()
            with gr.Row():
                heat_out = gr.Plot(label="Heatmap", show_label=False)
                cmp_out  = gr.Plot(label="Returns Comparison", show_label=False)
            corr_btn.click(
                fn=render_correlation,
                inputs=[sym_in],
                outputs=[heat_out, corr_inf, cmp_out],
            )

        # ── Tab 6: Monte Carlo ──
        with gr.Tab("🎲 Monte Carlo"):
            with gr.Row():
                mc_days = gr.Slider(21, 504, value=252, step=21, label="📅 Simulation Days")
                mc_sims = gr.Slider(100, 1000, value=500, step=100, label="🔢 Simulations")
            mc_btn   = gr.Button("🎲 Run Simulation", variant="primary", size="lg")
            mc_stats = gr.HTML()
            mc_chart = gr.Plot(label="Simulation Paths", show_label=False)
            mc_btn.click(
                fn=render_monte_carlo,
                inputs=[shared_symbol, shared_portfolio, mc_days, mc_sims],
                outputs=[mc_chart, mc_stats],
            )

        # ── Tab 7: About ──
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## ⬡ Portfolio Intelligence System

A **multi-agent AI-powered platform** for comprehensive portfolio risk analysis and market intelligence. Built as part of the IIT Madras Multi-Agent Systems curriculum.

---

### 🧩 Modules

| Module | Description |
|---|---|
| **📡 Market Overview** | Real-time prices, volume, P/E, market cap for 6 stocks |
| **📈 Historical Analysis** | Candlestick + MA20, cumulative returns, volume |
| **🛡️ Risk Assessment** | VaR, CVaR, Sharpe, Max Drawdown, Rolling Vol, Risk Gauge |
| **💥 Stress Testing** | 8 crash scenarios — loss % and dollar impact |
| **🔗 Correlation** | Correlation heatmap + cumulative return comparison |
| **🎲 Monte Carlo** | Up to 1000 simulation paths with confidence bands |

---

### 🏗️ Architecture

```
Yahoo Finance API
       ↓
Market Data Agent  →  SQLite DB
       ↓
Risk Management Agent (RiskIQ)
       ↓
Gradio Dashboard UI
```

---

### 📐 Risk Metrics Reference

| Metric | Good Value | Description |
|---|---|---|
| VaR 95% | < 2% | Max 1-day loss with 95% confidence |
| CVaR 95% | < 3% | Avg loss when VaR is exceeded |
| Sharpe Ratio | > 1.0 | Return per unit of risk |
| Max Drawdown | < 20% | Worst peak-to-trough decline |
| Annual Volatility | < 25% | Annualised return fluctuation |

---

**Team:** Ashwini · Dibyendu Sarkar · Jyoti Ranjan Sethi
**Program:** Multi-Agent Systems · IIT Madras · Week 3 of 16 · 2026

> ⚠️ *For educational purposes only. Not financial advice. Data sourced from Yahoo Finance.*
            """)

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch()