# Universe sourcing

SpectraQuant does not ship with embedded tickers. Universes are external inputs
that must be sourced from official exchange-maintained data.

You can also reference index presets in `config.yaml` (or via CLI) to pull
and cache constituents on demand:

```yaml
universe:
  india:
    tickerset: nifty50
  uk:
    tickerset: ftse100
```

CLI override example:

```bash
python -m src.spectraquant.cli.main refresh --universe "nifty50,ftse100"
```

## Official sources
- NSE index constituents: https://www.nseindia.com/market-data/indices-constituents
- FTSE Russell index data: https://www.ftserussell.com/resources

## Regenerating universes

```bash
python scripts/download_universe.py --only nifty_500
python scripts/download_universe.py --only nifty_50 nifty_100 nifty_200
python scripts/download_universe.py --only ftse_100 ftse_250 ftse_all_share
```

Outputs are stored under `data/universe/` with the schema:

- `ticker`
- `exchange`
- `name`
