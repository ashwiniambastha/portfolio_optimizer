import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from agents.risk_management.agent import RiskManagementAgent

# Initialize
risk_agent = RiskManagementAgent()

def assess_single_stock(symbol, portfolio_value):
    """Comprehensive risk assessment"""
    assessment = risk_agent.assess_risk(symbol, portfolio_value)
    
    if 'error' in assessment:
        return f"Error: {assessment['error']}", None, None, None
    
    # Format summary
    summary = f"""
    # Risk Assessment: {symbol}
    
    **Portfolio Value:** ${portfolio_value:,.0f}
    **Status:** {'⚠️ ALERT' if assessment['risk_status'] == 'ALERT' else '✅ OK'}
    
    ## Key Metrics
    
    - **VaR (95%):** {assessment['var_95']['var_pct']:.2%} (${assessment['var_95']['var_dollar']:,.0f})
    - **CVaR (95%):** {assessment['cvar_95']['cvar_pct']:.2%} (${assessment['cvar_95']['cvar_dollar']:,.0f})
    - **Annual Volatility:** {assessment['volatility']['annual_volatility']:.2%}
    - **Max Drawdown:** {assessment['max_drawdown']['max_drawdown']:.2%}
    - **Sharpe Ratio:** {assessment['sharpe_ratio']['sharpe_ratio']:.2f} ({assessment['sharpe_ratio']['rating']})
    """
    
    if assessment['beta']['beta'] is not None:
        summary += f"- **Beta:** {assessment['beta']['beta']:.2f} ({assessment['beta']['interpretation']})\n"
    
    # Alerts
    if assessment['alerts']:
        summary += "\n## 🚨 Risk Alerts\n"
        for alert in assessment['alerts']:
            summary += f"- {alert}\n"
    
    # Create stress test chart
    stress_df = pd.DataFrame(assessment['stress_test']).T
    fig_stress = go.Figure()
    fig_stress.add_trace(go.Bar(
        x=stress_df.index,
        y=stress_df['loss_pct'] * 100,
        text=[f"{x:.1f}%" for x in stress_df['loss_pct'] * 100],
        textposition='auto',
    ))
    fig_stress.update_layout(
        title="Stress Test Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Loss (%)",
        height=400
    )
    
    # Create detailed metrics table
    metrics_data = {
        'Metric': ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'Volatility', 'Max Drawdown', 'Sharpe Ratio'],
        'Value': [
            f"{assessment['var_95']['var_pct']:.2%}",
            f"{assessment['var_99']['var_pct']:.2%}",
            f"{assessment['cvar_95']['cvar_pct']:.2%}",
            f"{assessment['volatility']['annual_volatility']:.2%}",
            f"{assessment['max_drawdown']['max_drawdown']:.2%}",
            f"{assessment['sharpe_ratio']['sharpe_ratio']:.2f}"
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # Stress test table
    stress_table = stress_df[['shock_pct', 'loss_amount', 'years_to_recover']].copy()
    stress_table['shock_pct'] = stress_table['shock_pct'].apply(lambda x: f"{x:.2%}")
    stress_table['loss_amount'] = stress_table['loss_amount'].apply(lambda x: f"${x:,.0f}")
    stress_table.columns = ['Shock %', 'Loss $', 'Recovery (years)']
    
    return summary, fig_stress, metrics_df, stress_table

def calculate_correlation(symbols_str):
    """Calculate correlation matrix"""
    symbols = [s.strip() for s in symbols_str.split(',')]
    
    corr_matrix = risk_agent.calculate_correlation_matrix(symbols)
    
    if corr_matrix is None:
        return "Error: Could not fetch data", None
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12},
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        height=500,
        xaxis_title="Symbol",
        yaxis_title="Symbol"
    )
    
    return corr_matrix.round(3), fig

# Create Gradio Interface
with gr.Blocks(title="Risk Management Agent", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🛡️ Risk Management Agent (RiskIQ)
    ### Week 3: Multi-Agent Portfolio Optimization System
    **Team:** Ashwini, Dibyendu Sarkar, Jyoti Ranjan Sethi
    """)
    
    with gr.Tab("📊 Risk Assessment"):
        gr.Markdown("### Comprehensive Risk Analysis for Individual Assets")
        
        with gr.Row():
            symbol_input = gr.Dropdown(
                choices=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                value='AAPL',
                label="Select Stock"
            )
            portfolio_input = gr.Number(
                value=100000,
                label="Portfolio Value ($)"
            )
        
        assess_btn = gr.Button("🔍 Assess Risk", variant="primary", size="lg")
        
        with gr.Row():
            summary_output = gr.Markdown()
        
        with gr.Row():
            stress_chart = gr.Plot(label="Stress Test Results")
        
        with gr.Row():
            metrics_table = gr.Dataframe(label="Risk Metrics Summary")
            stress_table = gr.Dataframe(label="Stress Test Details")
        
        assess_btn.click(
            fn=assess_single_stock,
            inputs=[symbol_input, portfolio_input],
            outputs=[summary_output, stress_chart, metrics_table, stress_table]
        )
    
    with gr.Tab("🔗 Correlation Analysis"):
        gr.Markdown("### Portfolio Diversification Analysis")
        
        symbols_input = gr.Textbox(
            value="AAPL,GOOGL,MSFT,TSLA,AMZN",
            label="Enter symbols (comma-separated)"
        )
        
        corr_btn = gr.Button("📈 Calculate Correlations", variant="primary")
        
        with gr.Row():
            corr_matrix_output = gr.Dataframe(label="Correlation Matrix")
        
        with gr.Row():
            corr_heatmap = gr.Plot(label="Correlation Heatmap")
        
        corr_btn.click(
            fn=calculate_correlation,
            inputs=symbols_input,
            outputs=[corr_matrix_output, corr_heatmap]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## Risk Management Agent (RiskIQ)
        
        This agent calculates comprehensive risk metrics for portfolio management.
        
        ### Features Implemented:
        
        1. **Value at Risk (VaR)** - Maximum expected loss at confidence levels
        2. **Conditional VaR (CVaR)** - Expected loss in worst-case scenarios
        3. **Volatility Analysis** - Historical and rolling volatility
        4. **Maximum Drawdown** - Largest peak-to-trough decline
        5. **Sharpe Ratio** - Risk-adjusted return metric
        6. **Beta Calculation** - Market sensitivity analysis
        7. **Stress Testing** - Performance under extreme scenarios
        8. **Correlation Matrix** - Portfolio diversification analysis
        
        ### Integration:
        
        This agent connects to the **Market Data Agent** (Week 2) via REST API to fetch
        historical price data for risk calculations.
        
        ### Technologies:
        - NumPy, Pandas for calculations
        - Plotly for visualizations
        - Gradio for web interface
        - Requests for API integration
        
        ---
        **Week:** 3 of 16  
        **Project:** Multi-Agent Portfolio Optimization System
        """)

if __name__ == "__main__":
    demo.launch(share=True, server_port=7861)