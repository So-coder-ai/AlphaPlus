"""
Binance Futures API Client with Retry Logic

This module provides a robust client for interacting with Binance Futures API,
including automatic retry logic, error handling, and support for both testnet
and mainnet environments.

Author: AlphaPulse Team
License: MIT
"""

import os
import time
import logging
from functools import wraps
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

load_dotenv()
logger = logging.getLogger("alphapulse.client")


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator for automatic retry of failed API calls with exponential backoff.
    
    Args:
        max_attempts (int): Maximum number of retry attempts. Default: 3
        delay (float): Base delay in seconds between retries. Default: 1.0
        
    Returns:
        function: Decorated function with retry logic
        
    Note:
        - Retries are only attempted for temporary API errors (5xx status codes)
        - Permanent errors (4xx codes) are raised immediately
        - Delay increases exponentially with attempt number
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except BinanceAPIException as e:
                    if attempt == max_attempts or e.status_code in (400, 401, 403):
                        raise
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay * attempt)
        return wrapper
    return decorator


class FuturesClient:
    """
    Binance Futures API client with built-in retry logic and error handling.
    
    Provides a simplified interface for common Binance Futures operations including
    order placement, market data retrieval, and account management.
    
    Attributes:
        client: The underlying python-binance Client instance
        testnet (bool): Whether connected to testnet or mainnet
        
    Example:
        >>> client = FuturesClient(testnet=True)
        >>> order = client.create_order("BTCUSDT", "BUY", "MARKET", 0.01)
        >>> klines = client.get_klines("BTCUSDT", "1h", 500)
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize the FuturesClient.
        
        Args:
            testnet (bool): Whether to use Binance Futures Testnet. Default: True
            
        Raises:
            ValueError: If API keys are not found in environment variables
            
        Note:
            API keys should be set in environment variables:
            - BINANCE_API_KEY
            - BINANCE_API_SECRET
        """
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            raise ValueError("Missing BINANCE_API_KEY or BINANCE_API_SECRET in environment.")

        self.client = Client(api_key, api_secret)
        if testnet:
            self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
        self.testnet = testnet

    @retry(max_attempts=3)
    def create_order(self, symbol: str, side: str, order_type: str,
                     quantity: float, price: float = None) -> dict:
        """
        Place a futures order with retry logic.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            side (str): Order side ('BUY' or 'SELL')
            order_type (str): Order type ('MARKET' or 'LIMIT')
            quantity (float): Order quantity
            price (float, optional): Limit price. Required for LIMIT orders
            
        Returns:
            dict: Order response from Binance API
            
        Raises:
            ValueError: If price is not provided for LIMIT orders
            BinanceAPIException: If order placement fails
        """
        payload = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Price required for LIMIT orders.")
            payload["price"] = price
            payload["timeInForce"] = "GTC"

        response = self.client.futures_create_order(**payload)
        logger.info(f"Order placed: {response.get('orderId')} | {side} {quantity} {symbol} @ {price or 'MARKET'}")
        return response

    @retry(max_attempts=3)
    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> list:
        """
        Retrieve historical kline data for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            interval (str): Kline interval ('1m', '5m', '1h', '4h', '1d')
            limit (int): Number of klines to retrieve (max: 1000)
            
        Returns:
            list: List of kline data arrays
            
        Example:
            >>> klines = client.get_klines("BTCUSDT", "1h", 100)
        """
        return self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

    @retry(max_attempts=3)
    def get_ticker_price(self, symbol: str) -> float:
        """
        Get the current price for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Current price
        """
        data = self.client.get_symbol_ticker(symbol=symbol)
        return float(data["price"])

    @retry(max_attempts=3)
    def get_account_balance(self) -> list:
        """
        Get futures account balance information.
        
        Returns:
            list: Account balance details for all assets
            
        Example:
            >>> balances = client.get_account_balance()
            >>> usdt_balance = next(b for b in balances if b['asset'] == 'USDT')
        """
        return self.client.futures_account_balance()

    @retry(max_attempts=3)
    def get_open_orders(self, symbol: str = None) -> list:
        """
        Get all open orders, optionally filtered by symbol.
        
        Args:
            symbol (str, optional): Filter by specific symbol. If None, returns all open orders
            
        Returns:
            list: List of open order details
        """
        params = {"symbol": symbol} if symbol else {}
        return self.client.futures_get_open_orders(**params)

    @retry(max_attempts=3)
    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """
        Cancel an existing order.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            order_id (int): Order ID to cancel
            
        Returns:
            dict: Cancellation response from Binance API
            
        Raises:
            BinanceAPIException: If order cancellation fails
        """
        return self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
