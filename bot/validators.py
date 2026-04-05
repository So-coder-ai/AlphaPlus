VALID_SIDES = {"BUY", "SELL"}
VALID_ORDER_TYPES = {"MARKET", "LIMIT"}
VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}


def parse_side(value: str) -> str:
    side = value.strip().upper()
    if side not in VALID_SIDES:
        raise ValueError(f"Invalid side '{value}'. Must be one of: {', '.join(VALID_SIDES)}")
    return side


def parse_order_type(value: str) -> str:
    order_type = value.strip().upper()
    if order_type not in VALID_ORDER_TYPES:
        raise ValueError(f"Invalid order type '{value}'. Must be one of: {', '.join(VALID_ORDER_TYPES)}")
    return order_type


def parse_quantity(value: str | float) -> float:
    try:
        quantity = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Quantity must be a number, got: '{value}'")
    if quantity <= 0:
        raise ValueError(f"Quantity must be > 0, got: {quantity}")
    return quantity


def parse_price(value, order_type: str) -> float | None:
    if order_type == "LIMIT":
        if value is None:
            raise ValueError("Price is required for LIMIT orders.")
        try:
            price = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Price must be a number, got: '{value}'")
        if price <= 0:
            raise ValueError(f"Price must be > 0, got: {price}")
        return price
    return None


def parse_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol.isalpha() or len(symbol) < 5:
        raise ValueError(f"Invalid symbol format: '{value}'. Expected e.g. BTCUSDT.")
    return symbol


def parse_interval(value: str) -> str:
    interval = value.strip().lower()
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval '{value}'. Must be one of: {', '.join(sorted(VALID_INTERVALS))}")
    return interval
