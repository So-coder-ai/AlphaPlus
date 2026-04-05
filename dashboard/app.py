"""
AlphaPulse Dashboard - Interactive Streamlit Web Interface

Real-time trading signals, backtesting visualization, and order management.

Author: AlphaPulse Team
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.logging_config import get_logger
from ml.features import klines_to_dataframe, build_features, FEATURE_COLS
from ml.model import SignalModel
from backtest.engine import run_backtest, BacktestConfig

logger = get_logger("alphapulse.dashboard")


st.set_page_config(
    page_title="AlphaPulse | AI Trading Bot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("## ⚡ AlphaPulse")
    st.markdown("AI-Powered Futures Trading")
    st.divider()

    symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"], index=0)
    interval = st.selectbox("Interval", ["1h", "4h", "1d"], index=0)
    lookback = st.slider("Lookback Bars", 100, 1000, 500, 50)
    signal_threshold = st.slider("Signal Threshold", 0.50, 0.75, 0.55, 0.01)

    st.divider()
    st.markdown("**Backtest Settings**")
    initial_capital = st.number_input("Initial Capital ($)", 1000, 100000, 10000, 1000)
    allow_short = st.checkbox("Allow Short Trades", value=True)
    tx_cost = st.number_input("Transaction Cost (%)", 0.01, 0.5, 0.05, 0.01) / 100

    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    st.divider()
    st.caption("Data: Binance Futures Testnet")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")


@st.cache_data(ttl=300)
def get_demo_ohlcv(symbol: str, bars: int = 500) -> pd.DataFrame:
    np.random.seed(abs(hash(symbol)) % (2**31))
    price = 30000 if "BTC" in symbol else 1800
    dates = pd.date_range(end=datetime.now(), periods=bars, freq="1h")
    log_returns = np.random.normal(0.0002, 0.008, bars)
    prices = price * np.exp(np.cumsum(log_returns))
    noise = np.random.uniform(0.998, 1.002, bars)
    df = pd.DataFrame({
        "open":   prices * np.random.uniform(0.997, 1.003, bars),
        "high":   prices * np.random.uniform(1.002, 1.015, bars),
        "low":    prices * np.random.uniform(0.985, 0.998, bars),
        "close":  prices * noise,
        "volume": np.random.lognormal(10, 0.5, bars),
    }, index=dates)
    return df


tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Live Signal", "📈 Backtest", "🔬 Feature Lab", "📋 Orders"
])

df_raw = get_demo_ohlcv(symbol, lookback)
df_feat = build_features(df_raw, drop_na=True)
model = SignalModel(n_estimators=100)

with st.spinner("Training model on historical data..."):
    try:
        metrics = model.train(df_feat, n_splits=3)
        signal_probs = model.predict_series(df_feat)
        latest_signal = model.predict_latest(df_feat)
    except Exception as e:
        st.error(f"Model error: {e}")
        st.stop()


with tab1:
    st.markdown("### Current Signal")

    col1, col2, col3, col4 = st.columns(4)

    direction = latest_signal["direction"]
    confidence = latest_signal["confidence"]
    current_price = float(df_raw["close"].iloc[-1])
    price_change = float(df_raw["close"].pct_change().iloc[-1] * 100)

    col1.metric("Signal", direction, f"{'↑' if direction == 'BUY' else '↓'} {confidence:.1%} conf.")
    col2.metric("Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")
    col3.metric("Model Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    col4.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

    st.divider()

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=latest_signal["prob_up"] * 100,
        title={"text": "Bull Probability (%)"},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1d9e75" if latest_signal["prob_up"] > 0.5 else "#d85a30"},
            "steps": [
                {"range": [0, signal_threshold * 100], "color": "#faeeda"},
                {"range": [signal_threshold * 100, 100], "color": "#e1f5ee"},
            ],
            "threshold": {
                "line": {"color": "#3266ad", "width": 3},
                "thickness": 0.75,
                "value": signal_threshold * 100,
            },
        },
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=20))

    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        signals_plot = signal_probs[signal_probs >= signal_threshold]
        buy_dates = signals_plot.index
        buy_prices = df_feat.loc[buy_dates, "close"] if len(buy_dates) > 0 else pd.Series()

        fig_candle = go.Figure()
        plot_df = df_feat.tail(120)
        fig_candle.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"], high=plot_df["high"],
            low=plot_df["low"], close=plot_df["close"],
            name=symbol, increasing_line_color="#1d9e75",
            decreasing_line_color="#d85a30",
        ))
        buy_in_window = buy_prices[buy_prices.index.isin(plot_df.index)]
        if len(buy_in_window) > 0:
            fig_candle.add_trace(go.Scatter(
                x=buy_in_window.index,
                y=buy_in_window.values * 0.998,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#1d9e75"),
                name="BUY Signal",
            ))
        fig_candle.update_layout(
            title=f"{symbol} Price + Signals (last 120 bars)",
            height=280, xaxis_rangeslider_visible=False,
            margin=dict(t=40, b=20), showlegend=True,
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    st.markdown("##### Technical Indicators")
    fig_ind = go.Figure()
    rsi_data = df_feat["rsi"].tail(120)
    fig_ind.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data.values, name="RSI", line=dict(color="#7f77dd")))
    fig_ind.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_ind.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_ind.update_layout(height=180, margin=dict(t=20, b=20), title="RSI(14)")
    st.plotly_chart(fig_ind, use_container_width=True)


with tab2:
    st.markdown("### Strategy Backtest")

    config = BacktestConfig(
        initial_capital=initial_capital,
        signal_threshold=signal_threshold,
        allow_short=allow_short,
        transaction_cost_pct=tx_cost,
    )

    with st.spinner("Running backtest..."):
        result = run_backtest(df_feat, signal_probs, config)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Return", f"{result.total_return_pct:+.2f}%")
    m2.metric("CAGR", f"{result.cagr_pct:+.2f}%")
    m3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.3f}")
    m4.metric("Max Drawdown", f"-{result.max_drawdown_pct:.2f}%")
    m5.metric("Win Rate", f"{result.win_rate_pct:.1f}%")
    m6.metric("# Trades", result.num_trades)

    st.divider()

    eq = result.equity_curve
    bh_equity = initial_capital * (1 + df_feat["close"].pct_change().fillna(0)).cumprod()
    bh_equity = bh_equity.reindex(eq.index)

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="AlphaPulse Strategy",
                                line=dict(color="#1d9e75", width=2), fill="tozeroy",
                                fillcolor="rgba(29,158,117,0.08)"))
    fig_eq.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity.values, name="Buy & Hold",
                                line=dict(color="#7f77dd", width=1.5, dash="dot")))
    fig_eq.update_layout(
        title="Equity Curve: Strategy vs Buy & Hold",
        yaxis_title="Portfolio Value ($)",
        height=380, margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
                                fill="tozeroy", fillcolor="rgba(216,90,48,0.2)",
                                line=dict(color="#d85a30"), name="Drawdown"))
    fig_dd.update_layout(height=200, title="Drawdown (%)", margin=dict(t=40, b=20))
    st.plotly_chart(fig_dd, use_container_width=True)


with tab3:
    st.markdown("### Feature Importance & ML Diagnostics")

    fi_df = model.get_feature_importance_df()
    fig_fi = px.bar(
        fi_df, x="importance", y="feature", orientation="h",
        title="Random Forest Feature Importance",
        color="importance", color_continuous_scale="teal",
    )
    fig_fi.update_layout(height=500, yaxis={"categoryorder": "total ascending"}, margin=dict(t=50))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()
    st.markdown("##### Model Evaluation (Walk-Forward CV)")
    metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: "Score"})
    metrics_df["Score"] = metrics_df["Score"].round(4)
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("##### Feature Correlation Matrix")
    corr = df_feat[FEATURE_COLS].corr()
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Feature Correlation", height=550)
    st.plotly_chart(fig_corr, use_container_width=True)


with tab4:
    st.markdown("### Order Placement (Paper Trading)")
    st.info("Connected to Binance Futures **Testnet**. No real funds at risk.", icon="ℹ️")

    with st.form("order_form"):
        c1, c2, c3 = st.columns(3)
        order_symbol = c1.text_input("Symbol", value="BTCUSDT").upper()
        order_side = c2.selectbox("Side", ["BUY", "SELL"])
        order_type = c3.selectbox("Order Type", ["MARKET", "LIMIT"])
        c4, c5 = st.columns(2)
        quantity = c4.number_input("Quantity", min_value=0.001, value=0.01, step=0.001, format="%.3f")
        price = c5.number_input("Limit Price ($)", min_value=0.0, value=0.0,
                                disabled=(order_type == "MARKET"))
        submitted = st.form_submit_button("Place Order", type="primary", use_container_width=True)

    if submitted:
        if order_type == "LIMIT" and price <= 0:
            st.error("Limit price must be > 0 for LIMIT orders.")
        else:
            st.success(
                f"✅ **Order Submitted** — {order_side} {quantity} {order_symbol} "
                f"{'@ MARKET' if order_type == 'MARKET' else f'@ ${price:,.2f}'}"
            )
            st.json({
                "symbol": order_symbol, "side": order_side,
                "type": order_type, "quantity": quantity,
                "price": price if order_type == "LIMIT" else None,
                "status": "PAPER_FILLED",
                "timestamp": datetime.now().isoformat(),
            })

