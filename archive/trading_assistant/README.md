# Trading Assistant Archive

This directory holds archival material from the standalone trading assistant
application that was built alongside SpectraQuant-V3.

**This module is not actively developed or maintained.**

## Directory structure

```
archive/trading_assistant/
├── README.md                     ← this file
└── trading_assistant_runner.py   ← orphaned root runner, moved here 2026-03-10
```

The full application source code lives at `trading_assistant/` in the repository
root and is labelled archived there too (`trading_assistant/ARCHIVED.md`).

## Status

- Application: `trading_assistant/` at repo root — **inactive, archived in place**
- Runner script: `archive/trading_assistant/trading_assistant_runner.py` — **archived**
- Not reintegrated into V3.
- For V3 execution capabilities see `src/spectraquant_v3/execution/`.

## History

Originally the full `trading_assistant/` directory was planned to be moved here as
part of the SpectraQuant-V3 platform consolidation (ref: docs/design/ARCHITECTURE_REVIEW.md).
The full move was deferred; instead the directory was labelled archived in place and the
orphaned root scripts were relocated here.
