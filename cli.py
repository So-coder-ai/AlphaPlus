"""
AlphaPulse - AI-Powered Crypto Futures Trading Bot

Main CLI interface for signal generation, backtesting, and order execution.

Author: AlphaPulse Team
License: MIT
"""

import argparse
import sys
from bot.logging_config import get_logger
from bot.orders import OrderExecutor, RiskManager
from bot.validators import parse_side, parse_order_type, parse_quantity, parse_price, parse_symbol

logger = get_logger()


def cmd_order(args):
    try:
        symbol = parse_symbol(args.symbol)
        side = parse_side(args.side)
        order_type = parse_order_type(args.type)
        quantity = parse_quantity(args.quantity)
        price = parse_price(args.price, order_type)

        executor = OrderExecutor(risk_manager=RiskManager(max_quantity=10.0))
        result = executor.place(symbol, side, order_type, quantity, price)

        logger.info(f"Order placed: {result}")
        print(result)
        print("\nOrder executed successfully.")

    except Exception as e:
        logger.error(str(e))
        print(f"\nOrder failed: {e}")
        sys.exit(1)


def cmd_signal(args):
    try:
        from bot.client import FuturesClient
        from ml.features import klines_to_dataframe, build_features
        from ml.model import SignalModel

        print(f"\nFetching {args.lookback} bars of {args.symbol} {args.interval}...")
        client = FuturesClient()
        klines = client.get_klines(args.symbol, args.interval, args.lookback)
        df_raw = klines_to_dataframe(klines)
        df_feat = build_features(df_raw)

        print("Training signal model...")
        model = SignalModel()
        metrics = model.train(df_feat)
        signal = model.predict_latest(df_feat)

        print(f"\n{'='*40}")
        print(f"  Symbol      : {args.symbol}")
        print(f"  Interval    : {args.interval}")
        print(f"  Signal      : {signal['direction']}")
        print(f"  Confidence  : {signal['confidence']:.1%}")
        print(f"  P(UP)       : {signal['prob_up']:.3f}")
        print(f"  P(DOWN)     : {signal['prob_down']:.3f}")
        print(f"  Model Acc   : {metrics.get('accuracy', 0):.1%}")
        print(f"{'='*40}\n")

    except Exception as e:
        logger.error(str(e))
        print(f"Signal fetch failed: {e}")
        sys.exit(1)


def cmd_backtest(args):
    try:
        from bot.client import FuturesClient
        from ml.features import klines_to_dataframe, build_features
        from ml.model import SignalModel
        from backtest.engine import run_backtest, BacktestConfig

        print(f"\nRunning backtest for {args.symbol} ({args.interval})...")
        client = FuturesClient()
        klines = client.get_klines(args.symbol, args.interval, limit=1000)
        df_raw = klines_to_dataframe(klines)
        df_feat = build_features(df_raw)

        model = SignalModel()
        model.train(df_feat)
        signal_probs = model.predict_series(df_feat)

        config = BacktestConfig(initial_capital=float(args.capital))
        result = run_backtest(df_feat, signal_probs, config)
        print(result.summary())

    except Exception as e:
        logger.error(str(e))
        print(f"Backtest failed: {e}")
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="alphapulse",
        description="AlphaPulse — AI-powered crypto futures trading bot",
    )
    subparsers = parser.add_subparsers(dest="command")

    order_p = subparsers.add_parser("order", help="Place a futures order")
    order_p.add_argument("--symbol", required=True)
    order_p.add_argument("--side", required=True)
    order_p.add_argument("--type", required=True)
    order_p.add_argument("--quantity", required=True)
    order_p.add_argument("--price", default=None)

    signal_p = subparsers.add_parser("signal", help="Get ML trading signal")
    signal_p.add_argument("--symbol", default="BTCUSDT")
    signal_p.add_argument("--interval", default="1h")
    signal_p.add_argument("--lookback", type=int, default=500)

    bt_p = subparsers.add_parser("backtest", help="Run strategy backtest")
    bt_p.add_argument("--symbol", default="BTCUSDT")
    bt_p.add_argument("--interval", default="1h")
    bt_p.add_argument("--capital", default=10000)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "order":
        cmd_order(args)
    elif args.command == "signal":
        cmd_signal(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
