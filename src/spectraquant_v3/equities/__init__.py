"""Equities pipeline sub-package for SpectraQuant-AI-V3.

Modules here must NOT import anything from ``spectraquant_v3.crypto``.
Cross-asset imports are a hard forbidden pattern.

Supported data sources (to be implemented):
- Primary OHLCV: yfinance (research mode; replaceable via provider abstraction)
- News:          RSS / generic news provider abstraction
"""
