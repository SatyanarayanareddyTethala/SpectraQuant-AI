"""Crypto exchange adapters for market data and order execution."""

from __future__ import annotations

from spectraquant.crypto.exchange.coinbase_ws import CoinbaseWSClient
from spectraquant.crypto.exchange.coinbase_exec import (
    CoinbaseExecutor,
    OrderResult,
    AccountBalance,
)

__all__ = [
    "CoinbaseWSClient",
    "CoinbaseExecutor",
    "OrderResult",
    "AccountBalance",
]

# CCXTExecutor is optional — imported only when ccxt is installed.
try:
    from spectraquant.crypto.exchange.ccxt_exec import CCXTExecutor  # noqa: F401

    __all__.append("CCXTExecutor")
except ImportError:
    pass
