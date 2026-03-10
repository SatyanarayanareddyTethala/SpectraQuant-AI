"""Crypto pipeline sub-package for SpectraQuant-AI-V3.

Modules here must NOT import anything from ``spectraquant_v3.equities``.
Cross-asset imports are a hard forbidden pattern.

Supported data sources (to be implemented):
- Primary OHLCV: CCXT (Binance, Coinbase, Kraken)
- Fallback:      CoinGecko, CryptoCompare
- Funding / OI:  Binance Futures, Bybit, dYdX
- On-chain:      Glassnode
- News:          CryptoPanic, RSS
"""
