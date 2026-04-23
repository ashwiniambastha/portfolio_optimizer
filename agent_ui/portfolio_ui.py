"""
Portfolio Optimization Agent UI
agent_ui/portfolio_ui.py

Gradio tab definitions for the Portfolio Optimization Agent (Week 5).
Uses the same design system as risk_ui.py, alpha_ui.py, and streamlit_app.py.

Registers four tabs:
  ⭐  Max Sharpe Ratio
  🛡️  Min Variance
  📈  Efficient Frontier
  🎯  Signal-Adjusted
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict

# ---------------------------------------------------------------------------
# Colour palette — identical to app_trial.py
# ---------------------------------------------------------------------------
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

PALETTE = [BLUE_PRIMARY, GREEN, GOLD, RED, "#7c3aed"]

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

# ---------------------------------------------------------------------------
# Re-usable HTML component helpers (shared style with other agent UIs)
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, delta: str = "", color: str = "blue") -> str:
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


def section_header(icon: str, title: str, subtitle: str = "") -> str:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section-header"><div class="section-title">{icon} {title}</div>{sub}</div>'


def banner(msg: str, kind: str = "info") -> str:
    return f'<div class="{kind}-banner">{msg}</div>'


def _weights_table(weights: Dict[str, float], portfolio_value: float) -> str:
    """Render a styled weight allocation table."""
    rows = ""
    for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
        usd = w * portfolio_value
        bar = int(w * 220)
        rows += f"""
        <tr>
          <td style="font-weight:600;color:{TEXT_DARK}">{sym}</td>
          <td>{w * 100:.2f}%</td>
          <td>${usd:,.0f}</td>
          <td>
            <div style="background:{BLUE_PRIMARY};height:12px;width:{bar}px;
                        border-radius:4px;"></div>
          </td>
        </tr>"""
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:14px;margin-top:8px">
      <thead>
        <tr style="background:{BLUE_DARK};color:#fff">
          <th style="padding:9px 14px;text-align:left">Symbol</th>
          <th style="padding:9px 14px;text-align:left">Weight</th>
          <th style="padding:9px 14px;text-align:left">Amount ($)</th>
          <th style="padding:9px 14px;text-align:left">Allocation Bar</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _metric_cards_3(ret: float, vol: float, sharpe: float) -> str:
    return f"""
    <div class="metric-grid metric-grid-3" style="margin-bottom:14px">
        {kpi_card("Expected Return", f"{ret * 100:.2f}%",
                  "▲ Annualised" if ret >= 0 else "▼ Annualised",
                  "green" if ret >= 0 else "red")}
        {kpi_card("Volatility", f"{vol * 100:.2f}%", "Annualised", "gold")}
        {kpi_card("Sharpe Ratio", f"{sharpe:.3f}",
                  "Exceptional" if sharpe > 2 else "Good" if sharpe > 1 else "Acceptable",
                  "green" if sharpe > 1 else "gold")}
    </div>"""


def _pie_chart(weights: Dict[str, float], title: str) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=list(weights.keys()),
        values=[round(v * 100, 2) for v in weights.values()],
        marker=dict(colors=PALETTE, line=dict(color="white", width=2)),
        textinfo="label+percent",
        hole=0.38,
        hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=title, font=dict(color=BLUE_DARK, size=14, family="Inter, sans-serif")),
        showlegend=True,
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Lazy import of the agent (avoids hard dependency when viewing the file)
# ---------------------------------------------------------------------------

def _get_agent():
    """Import PortfolioOptimizationAgent lazily so the module can be viewed
    even if the agents package is not installed."""
    try:
        from agents.portfolio_optimization.agent import PortfolioOptimizationAgent
        return PortfolioOptimizationAgent()
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from agents.portfolio_optimization.agent import PortfolioOptimizationAgent
        return PortfolioOptimizationAgent()


_agent = None


def _agent_instance():
    global _agent
    if _agent is None:
        _agent = _get_agent()
    return _agent


def _parse_symbols(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Tab render functions
# ---------------------------------------------------------------------------

SYMS_DEFAULT = "AAPL,GOOGL,MSFT,TSLA,AMZN"


def render_max_sharpe(symbols_str: str, max_weight_pct: float,
                      period: str, portfolio_value: float):
    """Render Maximum Sharpe Ratio optimisation."""
    syms  = _parse_symbols(symbols_str)
    max_w = max_weight_pct / 100
    try:
        agent  = _agent_instance()
        df_ret = agent.fetch_returns_data(syms, period)
        res    = agent.optimize_max_sharpe(df_ret, max_weight=max_w)
    except Exception as exc:
        return banner(f"⚠ Error: {exc}", "alert"), None

    status_txt = "✅ Optimisation Successful" if res["optimization_success"] else "⚠ Converged with warnings"

    html = f"""
    {section_header("⭐", "Maximum Sharpe Ratio Portfolio",
                    f"Period: {period} | Symbols: {', '.join(syms)}")}
    {banner(status_txt, "success" if res["optimization_success"] else "info")}
    {_metric_cards_3(res["expected_return"], res["volatility"], res["sharpe_ratio"])}
    <div class="section-header">
        <div class="section-title">📊 Optimal Weight Allocation</div>
    </div>
    {_weights_table(res["weights"], portfolio_value)}
    """

    fig = _pie_chart(res["weights"], "⭐ Max Sharpe — Weight Allocation")
    return html, fig


def render_min_variance(symbols_str: str, max_weight_pct: float,
                        period: str, portfolio_value: float):
    """Render Minimum Variance optimisation."""
    syms  = _parse_symbols(symbols_str)
    max_w = max_weight_pct / 100
    try:
        agent  = _agent_instance()
        df_ret = agent.fetch_returns_data(syms, period)
        res    = agent.optimize_min_variance(df_ret, max_weight=max_w)
    except Exception as exc:
        return banner(f"⚠ Error: {exc}", "alert"), None

    html = f"""
    {section_header("🛡️", "Minimum Variance Portfolio",
                    f"Period: {period} | Symbols: {', '.join(syms)}")}
    {banner("🛡️ Conservative allocation — lowest achievable risk", "info")}
    {_metric_cards_3(res["expected_return"], res["volatility"], res["sharpe_ratio"])}
    <div class="section-header">
        <div class="section-title">📊 Optimal Weight Allocation</div>
    </div>
    {_weights_table(res["weights"], portfolio_value)}
    """

    fig = _pie_chart(res["weights"], "🛡️ Min Variance — Weight Allocation")
    return html, fig


def render_efficient_frontier(symbols_str: str, max_weight_pct: float,
                               period: str, n_pts: int):
    """Render the Markowitz Efficient Frontier."""
    syms  = _parse_symbols(symbols_str)
    max_w = max_weight_pct / 100
    try:
        agent  = _agent_instance()
        df_ret = agent.fetch_returns_data(syms, period)
        fdf    = agent.generate_efficient_frontier(df_ret, int(n_pts), max_w)
    except Exception as exc:
        return None, banner(f"⚠ Error: {exc}", "alert")

    ms_idx = fdf["sharpe"].idxmax()
    mv_idx = fdf["volatility"].idxmin()
    ms_pt  = fdf.loc[ms_idx]
    mv_pt  = fdf.loc[mv_idx]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fdf["volatility"] * 100,
        y=fdf["return"] * 100,
        mode="markers+lines",
        marker=dict(
            color=fdf["sharpe"],
            colorscale="Blues",
            size=8,
            showscale=True,
            colorbar=dict(title="Sharpe Ratio", tickfont=dict(family="Inter", color=TEXT_MED)),
            line=dict(color="white", width=0.5),
        ),
        line=dict(color="rgba(26,95,202,0.25)", width=1.5),
        name="Efficient Frontier",
        hovertemplate=(
            "<b>Efficient Portfolio</b><br>"
            "Return: %{y:.2f}%<br>"
            "Volatility: %{x:.2f}%<br>"
            "Sharpe: %{marker.color:.3f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=[ms_pt["volatility"] * 100],
        y=[ms_pt["return"] * 100],
        mode="markers+text",
        text=["⭐ Max Sharpe"],
        textposition="top center",
        textfont=dict(color=GOLD, size=11),
        marker=dict(color=GOLD, size=16, symbol="star",
                    line=dict(color="white", width=2)),
        name=f"Max Sharpe ({ms_pt['sharpe']:.3f})",
        hovertemplate=(
            f"<b>Maximum Sharpe Portfolio</b><br>"
            f"Return: {ms_pt['return'] * 100:.2f}%<br>"
            f"Volatility: {ms_pt['volatility'] * 100:.2f}%<br>"
            f"Sharpe: {ms_pt['sharpe']:.3f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=[mv_pt["volatility"] * 100],
        y=[mv_pt["return"] * 100],
        mode="markers+text",
        text=["🛡 Min Vol"],
        textposition="bottom center",
        textfont=dict(color=GREEN, size=11),
        marker=dict(color=GREEN, size=14, symbol="diamond",
                    line=dict(color="white", width=2)),
        name="Min Variance",
        hovertemplate=(
            f"<b>Minimum Variance Portfolio</b><br>"
            f"Return: {mv_pt['return'] * 100:.2f}%<br>"
            f"Volatility: {mv_pt['volatility'] * 100:.2f}%<br>"
            f"Sharpe: {mv_pt['sharpe']:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(
            text=f"⬡ Markowitz Efficient Frontier — {' · '.join(syms)}",
            font=dict(color=BLUE_DARK, size=14, family="Inter, sans-serif"),
        ),
        xaxis=dict(**AXIS_STYLE, title="Annualised Volatility (%)"),
        yaxis=dict(**AXIS_STYLE, title="Annualised Return (%)"),
        height=530,
        hovermode="closest",
    )

    info_html = f"""
    <div class="metric-grid metric-grid-2" style="margin-top:14px">
        <div class="kpi-card gold">
            <div class="kpi-label">⭐ Max Sharpe Portfolio</div>
            <div style="font-size:.9rem;color:{TEXT_DARK};margin-top:6px">
                Return: <strong>{ms_pt['return'] * 100:.2f}%</strong> &nbsp;|&nbsp;
                Vol: <strong>{ms_pt['volatility'] * 100:.2f}%</strong> &nbsp;|&nbsp;
                Sharpe: <strong>{ms_pt['sharpe']:.3f}</strong>
            </div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">🛡️ Min Variance Portfolio</div>
            <div style="font-size:.9rem;color:{TEXT_DARK};margin-top:6px">
                Return: <strong>{mv_pt['return'] * 100:.2f}%</strong> &nbsp;|&nbsp;
                Vol: <strong>{mv_pt['volatility'] * 100:.2f}%</strong> &nbsp;|&nbsp;
                Sharpe: <strong>{mv_pt['sharpe']:.3f}</strong>
            </div>
        </div>
    </div>
    """
    return fig, info_html


def render_signal_adjusted(symbols_str: str, max_weight_pct: float,
                            portfolio_value: float):
    """Render Signal-Adjusted optimisation (integrates Week 4 alpha signals)."""
    syms  = _parse_symbols(symbols_str)
    max_w = max_weight_pct / 100
    try:
        agent = _agent_instance()
        res   = agent.optimize_with_signals(syms, portfolio_value, max_w)
    except Exception as exc:
        return banner(f"⚠ Error: {exc}", "alert"), None

    if res["status"] != "success":
        return banner(f"⚠ {res.get('message', 'Unknown error')}", "alert"), None

    sig_rows = "".join(
        f"""<tr>
              <td style="font-weight:600">{sym}</td>
              <td>{"🟢" if s > 0 else "🔴" if s < 0 else "⚪"}</td>
              <td style="color:{"#0d9c5b" if s > 0 else "#e03131" if s < 0 else "#6b83a8"}">{s * 100:+.1f}%</td>
            </tr>"""
        for sym, s in res["signals"].items()
    )

    html = f"""
    {section_header("🎯", "Signal-Adjusted Portfolio",
                    f"Integrates Week 4 alpha signals | Portfolio: ${portfolio_value:,.0f}")}
    {_metric_cards_3(res["expected_return"], res["volatility"], res["sharpe_ratio"])}

    <div class="section-header" style="margin-top:18px">
        <div class="section-title">📡 Alpha Signals Used</div>
        <div class="section-sub">Signal confidence drives ±20% weight adjustments from Max Sharpe base</div>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:18px">
      <thead>
        <tr style="background:{BLUE_DARK};color:#fff">
          <th style="padding:9px 14px;text-align:left">Symbol</th>
          <th style="padding:9px 14px;text-align:left">Direction</th>
          <th style="padding:9px 14px;text-align:left">Confidence</th>
        </tr>
      </thead>
      <tbody>{sig_rows}</tbody>
    </table>

    <div class="section-header">
        <div class="section-title">📊 Adjusted Weight Allocation</div>
    </div>
    {_weights_table(res["adjusted_weights"], portfolio_value)}
    """

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Base Weights (Max Sharpe)", "Signal-Adjusted Weights"],
    )
    fig.add_trace(go.Pie(
        labels=list(res["base_weights"].keys()),
        values=[round(v * 100, 2) for v in res["base_weights"].values()],
        marker=dict(colors=PALETTE, line=dict(color="white", width=2)),
        textinfo="label+percent", hole=0.35,
        hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Pie(
        labels=list(res["adjusted_weights"].keys()),
        values=[round(v * 100, 2) for v in res["adjusted_weights"].values()],
        marker=dict(colors=PALETTE, line=dict(color="white", width=2)),
        textinfo="label+percent", hole=0.35,
        hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(
            text="🎯 Base vs Signal-Adjusted Weights",
            font=dict(color=BLUE_DARK, size=14, family="Inter, sans-serif"),
        ),
        showlegend=False,
        height=420,
    )
    return html, fig


# ---------------------------------------------------------------------------
# Gradio tab builder  —  call this from app_trial.py / main app
#
# IMPORTANT: Call this function INSIDE an existing gr.Tab() block.
# The function creates the shared input row + four sub-tabs internally.
# Do NOT wrap in another gr.Tabs() at the call site.
# ---------------------------------------------------------------------------

def build_portfolio_tabs(shared_symbol, shared_period, shared_portfolio):
    """
    Register the Portfolio Optimization sub-tabs.

    Call this inside a  `with gr.Tab("⬡ Portfolio Optimization"):` block.

    Parameters
    ----------
    shared_symbol    : gr.Dropdown  — selected stock symbol (kept for API parity)
    shared_period    : gr.Dropdown  — selected period
    shared_portfolio : gr.Number    — portfolio value in USD
    """

    # Shared inputs specific to this agent
    with gr.Row():
        po_symbols = gr.Textbox(
            value=SYMS_DEFAULT,
            label="📌 Symbols (comma-separated)",
            scale=3,
        )
        po_max_weight = gr.Slider(
            10, 100, value=40, step=5,
            label="⚖️ Max Weight per Asset (%)",
            scale=2,
        )

    # ── Sub-tabs (wrapped in their OWN gr.Tabs so they don't pollute the
    #    parent tab-bar and never duplicate)
    with gr.Tabs():

        with gr.Tab("⭐ Max Sharpe Ratio"):
            gr.Markdown(
                "**Maximise risk-adjusted returns** — optimal portfolio on the "
                "efficient frontier by Sharpe ratio."
            )
            ms_btn   = gr.Button("🚀 Optimize Max Sharpe", variant="primary", size="lg")
            ms_html  = gr.HTML()
            ms_chart = gr.Plot(label="Weight Allocation")
            ms_btn.click(
                fn=render_max_sharpe,
                inputs=[po_symbols, po_max_weight, shared_period, shared_portfolio],
                outputs=[ms_html, ms_chart],
            )

        with gr.Tab("🛡️ Min Variance"):
            gr.Markdown(
                "**Minimise portfolio risk** — ideal for conservative investors "
                "seeking the lowest achievable volatility."
            )
            mv_btn   = gr.Button("🚀 Optimize Min Variance", variant="primary", size="lg")
            mv_html  = gr.HTML()
            mv_chart = gr.Plot(label="Weight Allocation")
            mv_btn.click(
                fn=render_min_variance,
                inputs=[po_symbols, po_max_weight, shared_period, shared_portfolio],
                outputs=[mv_html, mv_chart],
            )

        with gr.Tab("📈 Efficient Frontier"):
            gr.Markdown(
                "**Visualise all optimal portfolios** across the risk-return spectrum."
            )
            ef_n_pts = gr.Slider(
                10, 100, value=40, step=10,
                label="Number of Frontier Points",
            )
            ef_btn   = gr.Button("🚀 Generate Frontier", variant="primary", size="lg")
            ef_chart = gr.Plot(label="Efficient Frontier")
            ef_info  = gr.HTML()
            ef_btn.click(
                fn=render_efficient_frontier,
                inputs=[po_symbols, po_max_weight, shared_period, ef_n_pts],
                outputs=[ef_chart, ef_info],
            )

        with gr.Tab("🎯 Signal-Adjusted"):
            gr.Markdown(
                "**Integrate Week 4 alpha signals** to tilt Max-Sharpe base weights "
                "by signal confidence."
            )
            sa_btn   = gr.Button("🚀 Signal-Adjusted Optimize", variant="primary", size="lg")
            sa_html  = gr.HTML()
            sa_chart = gr.Plot(label="Base vs Adjusted Weights")
            sa_btn.click(
                fn=render_signal_adjusted,
                inputs=[po_symbols, po_max_weight, shared_portfolio],
                outputs=[sa_html, sa_chart],
            )


# ---------------------------------------------------------------------------
# Stand-alone launch (for testing this UI in isolation)
# ---------------------------------------------------------------------------

STANDALONE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container { background: #f0f4fb !important; font-family: 'Inter', sans-serif !important; color: #0d1f3c !important; }
.app-header { background: linear-gradient(135deg, #0d2d6b 0%, #1a5fca 60%, #2979e8 100%); padding: 36px 40px 28px; text-align: center; border-bottom: 4px solid #d4940a; box-shadow: 0 4px 20px rgba(26,60,107,0.18); }
.app-title { font-family: 'Space Grotesk', sans-serif !important; font-size: 2rem; font-weight: 700; color: #ffffff !important; letter-spacing: 1px; margin-bottom: 6px; }
.app-subtitle { font-size: 0.82rem; color: rgba(255,255,255,0.75) !important; letter-spacing: 2px; text-transform: uppercase; }
.app-team { font-size: 0.75rem; color: rgba(255,255,255,0.55) !important; margin-top: 10px; letter-spacing: 1.5px; }
.header-accent { width: 60px; height: 3px; background: #d4940a; margin: 12px auto; border-radius: 2px; }
div[role="tablist"] { background: #ffffff !important; border-bottom: 2px solid #d0dff0 !important; padding: 0 16px !important; }
div[role="tab"] { font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important; font-weight: 500 !important; color: #6b83a8 !important; border: none !important; border-bottom: 3px solid transparent !important; padding: 13px 16px !important; background: transparent !important; }
div[role="tab"]:hover { color: #1a5fca !important; background: #e8f0fe !important; }
div[role="tab"][aria-selected="true"] { color: #1a5fca !important; border-bottom: 3px solid #1a5fca !important; background: #e8f0fe !important; font-weight: 600 !important; }
.metric-grid { display: grid; gap: 14px; margin-bottom: 22px; }
.metric-grid-3 { grid-template-columns: repeat(3, 1fr); }
.metric-grid-2 { grid-template-columns: repeat(2, 1fr); }
.kpi-card { background: #ffffff; border: 1px solid #d0dff0; border-radius: 12px; padding: 18px 20px; position: relative; overflow: hidden; box-shadow: 0 2px 10px rgba(26,60,107,0.07); }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #1a5fca, #2979e8); }
.kpi-card.green::before { background: linear-gradient(90deg, #0d9c5b, #22c55e); }
.kpi-card.red::before { background: linear-gradient(90deg, #e03131, #f87171); }
.kpi-card.gold::before { background: linear-gradient(90deg, #d4940a, #f59e0b); }
.kpi-label { font-size: 0.70rem; font-weight: 600; color: #6b83a8; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 9px; }
.kpi-value { font-family: 'Space Grotesk', sans-serif !important; font-size: 1.6rem; font-weight: 700; color: #0d1f3c; line-height: 1.1; }
.kpi-value.up { color: #0d9c5b; } .kpi-value.down { color: #e03131; } .kpi-value.warn { color: #d4940a; }
.kpi-delta { font-size: 0.74rem; font-weight: 500; margin-top: 5px; color: #6b83a8; }
.kpi-delta.up { color: #0d9c5b; } .kpi-delta.down { color: #e03131; }
.section-header { margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid #d0dff0; }
.section-title { font-family: 'Space Grotesk', sans-serif !important; font-size: 1rem; font-weight: 600; color: #0d2d6b; }
.section-sub { font-size: 0.76rem; color: #6b83a8; margin-top: 3px; }
.info-banner { background:#e8f0fe; border:1px solid #a8c4f8; border-left:4px solid #1a5fca; border-radius:8px; padding:12px 18px; font-size:.82rem; color:#1a3a6b; margin-bottom:14px; }
.alert-banner { background:#fee2e2; border:1px solid #fca5a5; border-left:4px solid #e03131; border-radius:8px; padding:12px 18px; font-size:.82rem; color:#7f1d1d; margin-bottom:14px; }
.success-banner { background:#d1fae5; border:1px solid #6ee7b7; border-left:4px solid #0d9c5b; border-radius:8px; padding:12px 18px; font-size:.82rem; color:#064e3b; margin-bottom:14px; }
button, .gr-button { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 0.83rem !important; border-radius: 8px !important; }
.gr-button-primary, button.primary { background: linear-gradient(135deg, #1a5fca, #2979e8) !important; color: #ffffff !important; border: none !important; }
input, select, textarea { background: #ffffff !important; border: 1.5px solid #c5d5ea !important; border-radius: 8px !important; color: #0d1f3c !important; }
label { font-family: 'Inter', sans-serif !important; font-size: 0.76rem !important; font-weight: 600 !important; color: #3a5080 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }
input[type="range"] { accent-color: #1a5fca !important; }
.app-footer { background: #0d2d6b; color: rgba(255,255,255,0.6); text-align: center; padding: 18px; font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 40px; }
"""

HEADER_HTML = """
<div class="app-header">
    <div class="app-title">⬡ Portfolio Optimization Agent</div>
    <div class="header-accent"></div>
    <div class="app-subtitle">Markowitz Mean-Variance Optimization &nbsp;·&nbsp; Week 5</div>
    <div class="app-team">Team: Ashwini &nbsp;|&nbsp; Dibyendu Sarkar &nbsp;|&nbsp; Jyoti Ranjan Sethi &nbsp;|&nbsp; IIT Madras 2026</div>
</div>
"""

FOOTER_HTML = """
<div class="app-footer">
    ⬡ Portfolio Intelligence System &nbsp;|&nbsp; IIT Madras 2026 &nbsp;|&nbsp;
    Data: Yahoo Finance &nbsp;|&nbsp; For Educational Use Only &nbsp;|&nbsp; Week 5 of 16
</div>
"""


if __name__ == "__main__":
    import gradio as gr

    with gr.Blocks(title="Portfolio Optimization Agent — Week 5", css=STANDALONE_CSS) as demo:
        gr.HTML(HEADER_HTML)

        with gr.Row():
            shared_symbol    = gr.Dropdown(
                choices=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
                value="AAPL",
                label="🚀 Stock Symbol",
                scale=2,
            )
            shared_period    = gr.Dropdown(
                choices=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                value="1y",
                label="📅 Period",
                scale=1,
            )
            shared_portfolio = gr.Number(
                value=100_000,
                label="💰 Portfolio Value ($)",
                minimum=1000,
                scale=2,
            )

        # Stand-alone: use the helper directly (it creates its own sub-tabs)
        build_portfolio_tabs(shared_symbol, shared_period, shared_portfolio)

        gr.HTML(FOOTER_HTML)

    print("Starting Portfolio Optimization UI ...")
    demo.launch(server_name="0.0.0.0", server_port=7864, share=True, show_error=True)