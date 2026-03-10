"""CLI commands for the SpectraQuant-AI-V3 Feature Store.

Provides:
  sqv3 feature-store list     List all persisted feature sets.
  sqv3 feature-store query    Query feature data with filters.
"""

from __future__ import annotations

from typing import Optional

import typer
from typing_extensions import Annotated

feature_store_app = typer.Typer(
    name="feature-store",
    help="Feature store operations — list and query persisted feature sets.",
    no_args_is_help=True,
)


@feature_store_app.command("list")
def feature_store_list(
    store_root: Annotated[
        str,
        typer.Option("--store-root", help="Feature store root directory"),
    ] = "data/feature_store",
    asset_class: Annotated[
        Optional[str],
        typer.Option("--asset-class", help="Filter by asset class (crypto/equity)"),
    ] = None,
    feature_name: Annotated[
        Optional[str],
        typer.Option("--feature-name", help="Filter by feature name"),
    ] = None,
) -> None:
    """List all persisted feature sets in the store.

    Displays a table of feature sets with metadata including symbol,
    version, date range, and row count.
    """
    from spectraquant_v3.feature_store import FeatureStore

    store = FeatureStore(store_root)
    sets = store.list_feature_sets(
        asset_class=asset_class,
        feature_name=feature_name,
    )

    if not sets:
        typer.echo("No feature sets found.")
        return

    typer.echo(
        f"{'FEATURE':<25} {'VERSION':<10} {'SYMBOL':<12} "
        f"{'ASSET':<8} {'DATE_START':<12} {'DATE_END':<12} {'ROWS':>6}"
    )
    typer.echo("-" * 90)
    for meta in sets:
        typer.echo(
            f"{meta.feature_name:<25} {meta.feature_version:<10} {meta.symbol:<12} "
            f"{meta.asset_class:<8} {meta.date_start:<12} {meta.date_end:<12} "
            f"{meta.row_count:>6}"
        )
    typer.echo(f"\n{len(sets)} feature set(s) found.")


@feature_store_app.command("query")
def feature_store_query(
    store_root: Annotated[
        str,
        typer.Option("--store-root", help="Feature store root directory"),
    ] = "data/feature_store",
    feature_name: Annotated[
        Optional[str],
        typer.Option("--feature-name", help="Feature name to query"),
    ] = None,
    symbol: Annotated[
        Optional[str],
        typer.Option("--symbol", help="Symbol to filter"),
    ] = None,
    asset_class: Annotated[
        Optional[str],
        typer.Option("--asset-class", help="Asset class filter"),
    ] = None,
    feature_version: Annotated[
        Optional[str],
        typer.Option("--feature-version", help="Feature version filter"),
    ] = None,
    date_start: Annotated[
        Optional[str],
        typer.Option("--date-start", help="Start date filter (YYYY-MM-DD)"),
    ] = None,
    date_end: Annotated[
        Optional[str],
        typer.Option("--date-end", help="End date filter (YYYY-MM-DD)"),
    ] = None,
    head: Annotated[
        int,
        typer.Option("--head", help="Number of rows to display"),
    ] = 10,
) -> None:
    """Query feature data with optional filters.

    Returns the first --head rows of matching feature data.
    """
    from spectraquant_v3.feature_store import FeatureStore

    store = FeatureStore(store_root)
    df = store.query_feature_data(
        feature_name=feature_name,
        symbol=symbol,
        asset_class=asset_class,
        feature_version=feature_version,
        date_start=date_start,
        date_end=date_end,
    )

    if df.empty:
        typer.echo("No data found matching the given filters.")
        return

    typer.echo(f"Found {len(df)} rows.  Showing first {head}:")
    typer.echo(df.head(head).to_string())
