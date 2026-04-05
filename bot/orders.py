import logging
from dataclasses import dataclass, field
from typing import Optional
from .client import FuturesClient

logger = logging.getLogger("alphapulse.orders")


@dataclass
class OrderResult:
    order_id: Optional[int]
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    executed_qty: float
    avg_price: float
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_response(cls, response: dict, symbol: str, side: str,
                      order_type: str, quantity: float, price: float = None):
        return cls(
            order_id=response.get("orderId"),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=response.get("status", "UNKNOWN"),
            executed_qty=float(response.get("executedQty", 0)),
            avg_price=float(response.get("avgPrice", 0)),
            raw=response,
        )

    def __str__(self):
        lines = [
            f"\n{'='*40}",
            f"  Order ID    : {self.order_id}",
            f"  Symbol      : {self.symbol}",
            f"  Side        : {self.side}",
            f"  Type        : {self.order_type}",
            f"  Quantity    : {self.quantity}",
        ]
        if self.order_type == "LIMIT":
            lines.append(f"  Limit Price : {self.price}")
        lines += [
            f"  Status      : {self.status}",
            f"  Executed    : {self.executed_qty}",
            f"  Avg Price   : {self.avg_price}",
            f"{'='*40}",
        ]
        return "\n".join(lines)


class RiskManager:

    def __init__(self, max_quantity: float = 10.0, allowed_symbols: list = None):
        self.max_quantity = max_quantity
        self.allowed_symbols = {s.upper() for s in allowed_symbols} if allowed_symbols else None

    def validate(self, symbol: str, quantity: float):
        if self.allowed_symbols and symbol not in self.allowed_symbols:
            raise ValueError(f"Symbol {symbol} not in allowed list: {self.allowed_symbols}")
        if quantity > self.max_quantity:
            raise ValueError(
                f"Quantity {quantity} exceeds risk limit of {self.max_quantity}. "
                f"Reduce position size."
            )


class OrderExecutor:

    def __init__(self, risk_manager: RiskManager = None):
        self.client = FuturesClient()
        self.risk = risk_manager or RiskManager()

    def place(self, symbol: str, side: str, order_type: str,
              quantity: float, price: float = None) -> OrderResult:
        self.risk.validate(symbol, quantity)

        response = self.client.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )

        result = OrderResult.from_response(
            response, symbol, side, order_type, quantity, price
        )
        logger.info(f"OrderResult: {result}")
        return result
