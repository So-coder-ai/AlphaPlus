import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("alphapulse.backtest")

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    trade_size_pct: float = 0.95
    transaction_cost_pct: float = 0.0005
    signal_threshold: float = 0.55
    allow_short: bool = True
    slippage_pct: float = 0.0002


@dataclass
class BacktestResult:
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    num_trades: int
    equity_curve: pd.Series
    trade_log: pd.DataFrame

    def summary(self) -> str:
        lines = [
            "\n" + "=" * 48,
            "  BACKTEST PERFORMANCE SUMMARY",
            "=" * 48,
            f"  Total Return       : {self.total_return_pct:+.2f}%",
            f"  CAGR               : {self.cagr_pct:+.2f}%",
            f"  Sharpe Ratio       : {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio      : {self.sortino_ratio:.3f}",
            f"  Max Drawdown       : {self.max_drawdown_pct:.2f}%",
            f"  Win Rate           : {self.win_rate_pct:.1f}%",
            f"  Profit Factor      : {self.profit_factor:.2f}",
            f"  Number of Trades   : {self.num_trades}",
            "=" * 48,
        ]
        return "\n".join(lines)


def run_backtest(
    df: pd.DataFrame,
    signal_probs: pd.Series,
    config: BacktestConfig = None,
) -> BacktestResult:
    if config is None:
        config = BacktestConfig()

    df = df.copy()
    df["signal_prob"] = signal_probs

    df["position"] = 0
    df.loc[df["signal_prob"] >= config.signal_threshold, "position"] = 1
    if config.allow_short:
        df.loc[df["signal_prob"] <= (1 - config.signal_threshold), "position"] = -1

    df["position"] = df["position"].shift(1).fillna(0)
    df["bar_return"] = df["close"].pct_change().fillna(0)
    df["strat_return"] = df["position"] * df["bar_return"]
    df["trade_flag"] = (df["position"].diff().abs() > 0).astype(int)
    df["cost"] = df["trade_flag"] * (config.transaction_cost_pct + config.slippage_pct)
    df["net_return"] = df["strat_return"] - df["cost"]
    df["equity"] = config.initial_capital * (1 + df["net_return"]).cumprod()

    returns = df["net_return"].dropna()
    equity = df["equity"].dropna()

    total_return = (equity.iloc[-1] / config.initial_capital - 1) * 100
    n_years = len(df) / (TRADING_DAYS_PER_YEAR * 24)
    cagr = ((equity.iloc[-1] / config.initial_capital) ** (1 / max(n_years, 0.01)) - 1) * 100

    sharpe = _sharpe(returns)
    sortino = _sortino(returns)
    max_dd = _max_drawdown(equity)

    trades = df[df["trade_flag"] == 1].copy()
    trade_returns = trades["strat_return"]
    wins = (trade_returns > 0).sum()
    win_rate = (wins / len(trades) * 100) if len(trades) > 0 else 0.0

    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = trade_returns[trade_returns <= 0].abs().sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    logger.info(f"Backtest complete. Return: {total_return:.2f}% | Sharpe: {sharpe:.3f} | MDD: {max_dd:.2f}%")

    return BacktestResult(
        total_return_pct=round(total_return, 4),
        cagr_pct=round(cagr, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown_pct=round(max_dd, 4),
        win_rate_pct=round(win_rate, 2),
        profit_factor=round(profit_factor, 4),
        num_trades=int(len(trades)),
        equity_curve=equity,
        trade_log=df[["close", "position", "signal_prob", "bar_return", "net_return", "equity"]],
    )


def _sharpe(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 8760) -> float:
    excess = returns - risk_free / periods_per_year
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def _sortino(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 8760) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(abs(drawdown.min()) * 100)

