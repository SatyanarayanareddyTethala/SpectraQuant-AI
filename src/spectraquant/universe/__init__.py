"""Utilities for loading large ticker universes from configuration."""
from __future__ import annotations

import json
import logging
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
from urllib.request import Request, urlopen

from spectraquant.universe.loader import load_nse_universe

logger = logging.getLogger(__name__)
_PLACEHOLDER_TICKERS = {"", "-", "na", "n/a", "none", "null", "nan", "tbd", "placeholder"}
_TICKER_SUFFIX_PATTERN = re.compile(r"\.[A-Za-z]{1,4}$")
UNIVERSE_SOURCES = {
    "nifty50": {
        "exchange": "NSE",
        "url": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
        "cache": "nifty_50",
        "suffix": ".NS",
        "format": "json",
    },
    "next50": {
        "exchange": "NSE",
        "url": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20NEXT%2050",
        "cache": "nifty_next_50",
        "suffix": ".NS",
        "format": "json",
    },
    "midcap100": {
        "exchange": "NSE",
        "url": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MIDCAP%20100",
        "cache": "nifty_midcap_100",
        "suffix": ".NS",
        "format": "json",
    },
    "nse_all": {
        "exchange": "NSE",
        "url": "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "cache": "nse_all",
        "suffix": ".NS",
        "format": "csv",
    },
    "ftse100": {
        "exchange": "LSE",
        "url": "https://www.ftserussell.com/sites/default/files/ftse_uk_index_series_constituents.csv",
        "cache": "ftse_100",
        "filter": "FTSE 100",
        "suffix": ".L",
        "format": "csv",
    },
    "ftse250": {
        "exchange": "LSE",
        "url": "https://www.ftserussell.com/sites/default/files/ftse_uk_index_series_constituents.csv",
        "cache": "ftse_250",
        "filter": "FTSE 250",
        "suffix": ".L",
        "format": "csv",
    },
    "custom": {
        "exchange": "CUSTOM",
        "cache": "custom",
    },
}


def _fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as response:  # noqa: S310 - intentional public download
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _fetch_csv(url: str) -> pd.DataFrame:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as response:  # noqa: S310 - intentional public download
        return pd.read_csv(response)


def _normalize_index_payload(payload: dict, exchange: str, suffix: str | None) -> list[str]:
    data = payload.get("data") or payload.get("records", {}).get("data")
    if not data:
        return []
    df = pd.DataFrame(data)
    symbol_col = None
    for candidate in ("symbol", "ticker", "code", "identifier"):
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col is None:
        return []
    symbols = df[symbol_col].astype(str).str.strip()
    symbols = symbols[symbols != ""]
    tickers = symbols.tolist()
    if suffix:
        tickers = [t if t.endswith(suffix) else f"{t}{suffix}" for t in tickers]
    logger.info("Fetched %s tickers for %s", len(tickers), exchange)
    return tickers


def _normalize_ftse_payload(df: pd.DataFrame, filter_label: str | None, suffix: str | None) -> list[str]:
    if filter_label:
        candidates = [c for c in df.columns if "index" in c.lower()]
        if candidates:
            df = df[df[candidates[0]].astype(str).str.contains(filter_label, na=False)]
    ticker_col = None
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in ("ticker", "epic", "symbol", "code", "instrument"):
        if candidate in lower_cols:
            ticker_col = lower_cols[candidate]
            break
    if ticker_col is None:
        return []
    tickers = df[ticker_col].astype(str).str.strip().tolist()
    tickers = [t for t in tickers if t]
    if suffix:
        tickers = [t if t.endswith(suffix) else f"{t}{suffix}" for t in tickers]
    logger.info("Fetched %s tickers for FTSE filter %s", len(tickers), filter_label or "all")
    return tickers


def _normalize_equity_list_payload(df: pd.DataFrame, suffix: str | None) -> list[str]:
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    ticker_col = lower_cols.get("symbol") or lower_cols.get("ticker")
    if ticker_col is None:
        ticker_col = df.columns[0]
    tickers = df[ticker_col].astype(str).str.strip().tolist()
    tickers = [t for t in tickers if t]
    if suffix:
        tickers = [t if t.endswith(suffix) else f"{t}{suffix}" for t in tickers]
    logger.info("Fetched %s tickers from equity list", len(tickers))
    return tickers


def _load_cached_universe(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read cached universe file %s: %s", path, exc)
        return []
    lower_cols = {c.lower(): c for c in df.columns}
    ticker_col = lower_cols.get("ticker") or lower_cols.get("symbol")
    if ticker_col is None:
        return []
    tickers = df[ticker_col].astype(str).str.strip().tolist()
    return [t for t in tickers if t]


def _fetch_universe_set(name: str, cache_dir: Path) -> list[str]:
    source = UNIVERSE_SOURCES.get(name)
    if not source:
        return []
    if name == "custom":
        return []
    cache_path = cache_dir / f"{source['cache']}.csv"
    cached = _load_cached_universe(cache_path)
    if cached:
        return cached
    exchange = source["exchange"]
    url = source["url"]
    payload_format = source.get("format", "json" if exchange == "NSE" else "csv")
    if payload_format == "json":
        payload = _fetch_json(url)
        tickers = _normalize_index_payload(payload, exchange=exchange, suffix=source.get("suffix"))
    elif exchange == "NSE":
        df = _fetch_csv(url)
        tickers = _normalize_equity_list_payload(df, suffix=source.get("suffix"))
    else:
        df = _fetch_csv(url)
        tickers = _normalize_ftse_payload(df, filter_label=source.get("filter"), suffix=source.get("suffix"))
    if tickers:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
        logger.info("Cached %s tickers to %s", len(tickers), cache_path)
    return tickers


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _read_csv_from_source(source: str | Path, header: int | None = 0) -> pd.DataFrame:
    if isinstance(source, Path):
        return pd.read_csv(source, header=header)
    if _is_url(str(source)):
        req = Request(str(source), headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as response:  # noqa: S310 - intentional public download
            payload = response.read()
        return pd.read_csv(BytesIO(payload), header=header)
    return pd.read_csv(Path(source), header=header)


def _read_ticker_file(path: str | Path) -> tuple[list[object], int]:
    source = str(path)
    if not _is_url(source):
        file_path = Path(source)
        if not file_path.exists():
            logger.warning("Universe file %s not found; skipping.", file_path)
            return [], 0

    try:
        df = _read_csv_from_source(source, header=0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read universe file %s: %s", source, exc)
        return [], 0

    if df.empty:
        return [], 0

    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    ticker_col = lower_cols.get("ticker") or lower_cols.get("symbol")
    if ticker_col is not None:
        tickers = df[ticker_col].tolist()
        return tickers, int(df.shape[0])

    try:
        df = _read_csv_from_source(source, header=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read universe file %s: %s", source, exc)
        return [], 0
    if df.empty:
        return [], 0

    first_row = df.iloc[0].astype(str).str.strip().str.lower().tolist()
    if any(value in {"ticker", "symbol"} for value in first_row):
        df = df.iloc[1:]
    tickers = df.iloc[:, 0].tolist()
    return tickers, int(df.shape[0])


def _resolve_tickers_file(region: str, region_cfg: dict | None) -> str | None:
    if not isinstance(region_cfg, dict):
        return None
    tickers_file = region_cfg.get("tickers_file")
    legacy_path = region_cfg.get("path")
    if tickers_file and legacy_path and str(tickers_file) != str(legacy_path):
        logger.warning(
            "Universe config for %s has both tickers_file=%s and path=%s; using tickers_file.",
            region,
            tickers_file,
            legacy_path,
        )
    return str(tickers_file or legacy_path) if (tickers_file or legacy_path) else None


def _normalize_ticker_input(tickers: object) -> list[object]:
    if tickers is None:
        return []
    if isinstance(tickers, (list, tuple, set)):
        return list(tickers)
    return [tickers]


def _infer_region_suffix(region: str, region_cfg: dict | None) -> str | None:
    if isinstance(region_cfg, dict):
        explicit = region_cfg.get("suffix")
        if explicit:
            return str(explicit)
        source = str(region_cfg.get("source", "")).lower()
        if source == "nse":
            return ".NS"
        if source in {"lse", "uk"}:
            return ".L"
    if region == "india":
        return ".NS"
    if region == "uk":
        return ".L"
    return None


def _apply_suffixes(
    tickers: Iterable[object],
    suffix: str | None,
    allowed_suffixes: Iterable[str],
) -> list[object]:
    if not suffix:
        return list(tickers)
    allowed = tuple(str(suf) for suf in allowed_suffixes)
    normalized: list[object] = []
    for ticker in tickers:
        if ticker is None:
            normalized.append(ticker)
            continue
        if isinstance(ticker, float) and math.isnan(ticker):
            normalized.append(ticker)
            continue
        value = str(ticker).strip()
        if not value:
            normalized.append(ticker)
            continue
        if value.endswith(allowed):
            normalized.append(value)
            continue
        if "." in value:
            normalized.append(value)
            continue
        normalized.append(f"{value}{suffix}")
    return normalized


def _collect_csv_tickers(config: dict) -> tuple[list[object], list[str], int, dict | None]:
    tickers: list[object] = []
    raw_count = 0
    sources: list[str] = []
    india_meta: dict | None = None
    universe_cfg = config.get("universe", {}) if isinstance(config, dict) else {}
    universe_default_path = universe_cfg.get("path") if isinstance(universe_cfg, dict) else None
    allowed_suffixes = _resolve_allowed_suffixes(config)
    for region in ("india", "uk"):
        region_cfg = universe_cfg.get(region, {}) or {}
        if region == "india" and str(region_cfg.get("source", "")).lower() == "csv":
            file_path = _resolve_tickers_file(region, region_cfg)
            if not file_path and universe_default_path:
                file_path = str(universe_default_path)
            if file_path:
                nse_tickers, meta, _ = load_nse_universe(
                    file_path,
                    symbol_column=region_cfg.get("symbol_column", "SYMBOL"),
                    suffix=region_cfg.get("suffix", ".NS"),
                    filter_series_eq=bool(region_cfg.get("filter_series_eq", True)),
                )
                if nse_tickers:
                    sources.append(f"india:{Path(file_path).stem}")
                tickers.extend(nse_tickers)
                raw_count += int(meta.get("raw_count", 0))
                india_meta = meta
            continue
        file_path = _resolve_tickers_file(region, region_cfg)
        if file_path:
            file_tickers, file_rows = _read_ticker_file(file_path)
            suffix = _infer_region_suffix(region, region_cfg)
            file_tickers = _apply_suffixes(file_tickers, suffix, allowed_suffixes)
            if file_tickers:
                sources.append(f"csv:{Path(file_path).stem}")
            tickers.extend(file_tickers)
            raw_count += file_rows
    return tickers, sources, raw_count, india_meta


def _resolve_tickerset_config(config: dict) -> list[str]:
    universe_cfg = config.get("universe", {}) if isinstance(config, dict) else {}
    selected_sets: list[str] = []
    explicit = universe_cfg.get("selected_sets") or universe_cfg.get("tickerset")
    if explicit:
        if isinstance(explicit, str):
            selected_sets.extend([item.strip().lower() for item in explicit.split(",") if item.strip()])
        else:
            selected_sets.extend([str(item).strip().lower() for item in explicit if str(item).strip()])

    for region in ("india", "uk"):
        region_cfg = universe_cfg.get(region, {}) or {}
        region_set = region_cfg.get("tickerset") or region_cfg.get("tickersets")
        if not region_set:
            continue
        if isinstance(region_set, str):
            selected_sets.extend([item.strip().lower() for item in region_set.split(",") if item.strip()])
        else:
            selected_sets.extend([str(item).strip().lower() for item in region_set if str(item).strip()])

    return list(dict.fromkeys(selected_sets))


def _load_tickerset_tickers(config: dict) -> tuple[list[object], list[str]]:
    sets = _resolve_tickerset_config(config)
    if not sets:
        return [], []
    tickers: list[object] = []
    sources: list[str] = []
    cache_dir = Path("data/universe")
    for set_name in sets:
        if set_name == "custom":
            custom_tickers, custom_sources, _, _ = _collect_csv_tickers(config)
            tickers.extend(custom_tickers)
            sources.extend(custom_sources if custom_sources else ["custom"])
            continue
        fetched = _fetch_universe_set(set_name, cache_dir)
        if fetched:
            tickers.extend(fetched)
            sources.append(f"set:{set_name}")
        else:
            logger.warning("No tickers resolved for tickerset '%s'", set_name)
    return tickers, sources


def _dedupe_preserve(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _resolve_allowed_suffixes(config: dict) -> list[str]:
    universe_cfg = config.get("universe", {}) if isinstance(config, dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    allowed_suffixes = data_cfg.get("allowed_ticker_suffixes")
    if not allowed_suffixes:
        allowed_suffixes = universe_cfg.get("allowed_ticker_suffixes")
    if not allowed_suffixes:
        allowed_suffixes = [".NS", ".L"]
    return list(allowed_suffixes)


def _looks_like_ticker(value: str, allowed_suffixes: Iterable[str]) -> bool:
    if not value:
        return False
    cleaned = value.strip()
    if not cleaned:
        return False
    allowed = tuple(str(suf).upper() for suf in allowed_suffixes)
    upper = cleaned.upper()
    if upper.endswith(allowed):
        return True
    if "." in cleaned and _TICKER_SUFFIX_PATTERN.search(cleaned):
        return True
    return False


def parse_universe_override(value: str, config: dict) -> tuple[str, list[str]]:
    """Parse a universe override into tickers or set names."""

    tokens = [item.strip() for item in str(value).split(",") if item.strip()]
    if not tokens:
        return "sets", []
    allowed_suffixes = _resolve_allowed_suffixes(config)
    ticker_flags = [_looks_like_ticker(token, allowed_suffixes) for token in tokens]
    if any(ticker_flags) and not all(ticker_flags):
        raise ValueError(
            "Universe override mixes tickers and set names. "
            "Provide either comma-separated tickers (e.g., RELIANCE.NS,TCS.NS) "
            "or universe set names (e.g., nifty50,ftse100)."
        )
    if all(ticker_flags):
        return "tickers", tokens
    return "sets", [token.lower() for token in tokens]


def _resolve_test_mode_limit(config: dict) -> int:
    if not isinstance(config, dict):
        return 0
    raw = config.get("test_mode", False)
    if isinstance(raw, dict):
        enabled = bool(raw.get("enabled", False))
        limit = int(raw.get("limit_tickers", 0) or 0)
        return limit if enabled else 0
    if raw:
        return 0
    return 0


def resolve_universe(config: dict) -> tuple[list[str], dict]:
    """Resolve tickers from config with cleaning metadata."""

    universe_cfg = config.get("universe", {}) if isinstance(config, dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    allowed_suffixes = _resolve_allowed_suffixes(config)

    source = "unknown"
    raw_tickers: list[object] = []
    raw_count = 0
    sources: list[str] = []
    selected_sets: list[str] = []
    india_meta: dict | None = None

    selected_sets = _resolve_tickerset_config(config)
    # Special universe set: "news" -> load latest news candidates tickers
    if "news" in selected_sets:
        p = Path("data/news_cache/news_universe_latest.json")
        if p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
            tickers = payload.get("tickers", []) or []
            tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
            if tickers:
                meta = {
                    "source": "news_universe_latest.json",
                    "selected_sets": selected_sets,
                    "raw_count": len(tickers),
                    "cleaned_count": len(tickers),
                    "dropped_empty": 0,
                    "dropped_placeholders": 0,
                }
                return tickers, meta
    if selected_sets:
        for set_name in selected_sets:
            if set_name in {"india", "uk"}:
                region_cfg = universe_cfg.get(set_name, {}) if isinstance(universe_cfg, dict) else {}
                file_path = _resolve_tickers_file(set_name, region_cfg)
                if not file_path:
                    logger.warning("Selected universe set '%s' missing tickers_file/path.", set_name)
                    continue
                if set_name == "india" and str(region_cfg.get("source", "")).lower() == "csv":
                    nse_tickers, meta, _ = load_nse_universe(
                        file_path,
                        symbol_column=region_cfg.get("symbol_column", "SYMBOL"),
                        suffix=region_cfg.get("suffix", ".NS"),
                        filter_series_eq=bool(region_cfg.get("filter_series_eq", True)),
                    )
                    raw_tickers.extend(nse_tickers)
                    raw_count += int(meta.get("raw_count", 0))
                    sources.append(f"{set_name}:{file_path}")
                    india_meta = meta
                else:
                    file_tickers, file_rows = _read_ticker_file(file_path)
                    suffix = _infer_region_suffix(set_name, region_cfg)
                    file_tickers = _apply_suffixes(file_tickers, suffix, allowed_suffixes)
                    raw_tickers.extend(file_tickers)
                    raw_count += file_rows
                    sources.append(f"{set_name}:{file_path}")
            elif set_name == "custom":
                custom_tickers, custom_sources, custom_rows, custom_meta = _collect_csv_tickers(config)
                if custom_tickers:
                    raw_tickers.extend(custom_tickers)
                    raw_count += custom_rows
                    sources.extend(custom_sources if custom_sources else ["custom"])
                if custom_meta is not None:
                    india_meta = custom_meta
            else:
                fetched = _fetch_universe_set(set_name, Path("data/universe"))
                if fetched:
                    raw_tickers.extend(fetched)
                    raw_count += len(fetched)
                    sources.append(f"set:{set_name}")
                else:
                    logger.warning("No tickers resolved for selected universe set '%s'.", set_name)
        source = ",".join(sources) if sources else "selected_sets"
    else:
        data_tickers = data_cfg.get("tickers") if isinstance(data_cfg, dict) else None
        if isinstance(data_tickers, (list, tuple, set)) and list(data_tickers):
            source = "data.tickers"
            sources = ["data.tickers"]
            raw_tickers = list(data_tickers)
            raw_count = len(raw_tickers)
        else:
            legacy_file = data_cfg.get("tickers_file") if isinstance(data_cfg, dict) else None
            if legacy_file:
                raw_tickers, raw_count = _read_ticker_file(legacy_file)
                source = f"data.tickers_file:{legacy_file}"
                sources = [source]
            else:
                universe_raw = _normalize_ticker_input(universe_cfg.get("tickers")) if isinstance(universe_cfg, dict) else []
                if universe_raw:
                    raw_tickers = universe_raw
                    raw_count = len(universe_raw)
                    source = "universe.tickers"
                    sources = ["universe.tickers"]
                else:
                    csv_tickers, csv_sources, csv_raw_count, india_meta = _collect_csv_tickers(config)
                    raw_tickers = csv_tickers
                    raw_count = csv_raw_count
                    source = ",".join(csv_sources) if csv_sources else "csv"
                    sources = csv_sources

    cleaned: list[str] = []
    dropped_empty = 0
    dropped_placeholders = 0
    for ticker in raw_tickers:
        if ticker is None:
            dropped_empty += 1
            continue
        if isinstance(ticker, float) and math.isnan(ticker):
            dropped_empty += 1
            continue
        value = str(ticker).strip()
        if not value:
            dropped_empty += 1
            continue
        if value.lower() in _PLACEHOLDER_TICKERS:
            dropped_placeholders += 1
            continue
        cleaned.append(value)

    deduped = _dedupe_preserve(cleaned)
    duplicates = len(cleaned) - len(deduped)

    invalid = [t for t in deduped if not any(t.endswith(s) for s in allowed_suffixes)]
    invalid_sample = invalid[:10]
    valid = [t for t in deduped if t not in set(invalid)]

    max_per_run = int(data_cfg.get("max_tickers_per_run", 0) or 0)
    test_mode_limit = _resolve_test_mode_limit(config)
    if source == "data.tickers":
        cap_limit = test_mode_limit
    else:
        cap_limit = test_mode_limit or max_per_run
    cap_reason = None
    if cap_limit > 0 and len(valid) > cap_limit:
        cap_reason = "test_mode" if test_mode_limit else "max_tickers_per_run"
    capped = valid[:cap_limit] if cap_limit > 0 else valid
    capped_count = len(capped)

    meta = {
        "source": source,
        "sources": sources,
        "raw_count": raw_count,
        "cleaned_count": len(cleaned),
        "dropped_empty": dropped_empty,
        "dropped_placeholders": dropped_placeholders,
        "deduped_count": len(deduped),
        "duplicates": duplicates,
        "allowed_suffixes": allowed_suffixes,
        "dropped_invalid_suffix": invalid_sample,
        "invalid_suffix_count": len(invalid),
        "max_tickers_per_run": max_per_run,
        "test_mode_limit": test_mode_limit,
        "capped_count": capped_count,
        "cap_reason": cap_reason,
        "selected_sets": selected_sets,
        "india_universe": india_meta,
    }
    return capped, meta


def _clean_tickers(tickers: Iterable[object]) -> tuple[list[str], int]:
    cleaned: list[str] = []
    skipped = 0
    for t in tickers:
        if t is None:
            skipped += 1
            continue
        if isinstance(t, float) and math.isnan(t):
            skipped += 1
            continue
        ticker = str(t).strip()
        if not ticker or ticker.lower() in _PLACEHOLDER_TICKERS:
            skipped += 1
            continue
        cleaned.append(ticker)
    return cleaned, skipped


def load_universe_from_config(config: dict) -> list[str]:
    """Load tickers from configured universe files with safe fallbacks."""
    tickers, meta = resolve_universe(config)
    if not tickers:
        raise ValueError(
            "Universe is empty after cleaning. Provide a non-empty external CSV universe "
            "or config tickers and remove placeholder/blank rows."
        )
    logger.info(
        "Loaded tickers: raw=%s cleaned=%s deduped=%s duplicates=%s dropped_empty=%s capped=%s cap_reason=%s source=%s",
        meta.get("raw_count"),
        meta.get("cleaned_count"),
        len(tickers),
        meta.get("duplicates"),
        meta.get("dropped_empty"),
        meta.get("capped_count"),
        meta.get("cap_reason"),
        meta.get("source"),
    )
    return tickers


def load_universe_set(config: dict, set_name: str) -> tuple[list[str], dict]:
    """Load a specific regional universe set for stats or diagnostics."""
    universe_cfg = config.get("universe", {}) if isinstance(config, dict) else {}
    region_cfg = universe_cfg.get(set_name, {}) if isinstance(universe_cfg, dict) else {}
    if set_name == "india" and str(region_cfg.get("source", "")).lower() == "csv":
        file_path = _resolve_tickers_file(set_name, region_cfg)
        if not file_path and isinstance(universe_cfg, dict):
            file_path = universe_cfg.get("path")
        if not file_path:
            return [], {"source": set_name, "raw_count": 0, "cleaned_count": 0, "missing_file": True}
        tickers, meta, diagnostics = load_nse_universe(
            file_path,
            symbol_column=region_cfg.get("symbol_column", "SYMBOL"),
            suffix=region_cfg.get("suffix", ".NS"),
            filter_series_eq=bool(region_cfg.get("filter_series_eq", True)),
        )
        meta = {
            "source": str(file_path),
            "raw_count": meta.get("raw_count", 0),
            "cleaned_count": meta.get("cleaned_count", 0),
            "missing_file": False,
            "diagnostics": [diag.code for diag in diagnostics],
        }
        return tickers, meta
    file_path = _resolve_tickers_file(set_name, region_cfg)
    if not file_path:
        return [], {"source": set_name, "raw_count": 0, "cleaned_count": 0, "missing_file": True}
    raw_tickers, raw_count = _read_ticker_file(file_path)
    allowed_suffixes = _resolve_allowed_suffixes(config)
    suffix = _infer_region_suffix(set_name, region_cfg)
    raw_tickers = _apply_suffixes(raw_tickers, suffix, allowed_suffixes)
    cleaned, skipped_blank = _clean_tickers(raw_tickers)
    deduped = _dedupe_preserve(cleaned)
    invalid = [t for t in deduped if not any(t.endswith(s) for s in allowed_suffixes)]
    valid = [t for t in deduped if t not in set(invalid)]
    meta = {
        "source": str(file_path),
        "raw_count": raw_count,
        "cleaned_count": len(cleaned),
        "skipped_blank": skipped_blank,
        "deduped": len(deduped),
        "invalid_suffix_count": len(invalid),
        "missing_file": False,
    }
    return valid, meta
