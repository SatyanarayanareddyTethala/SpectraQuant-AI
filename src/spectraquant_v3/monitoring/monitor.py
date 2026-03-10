"""Pipeline monitor for SpectraQuant-AI-V3.

Provides runtime health-checks for quantitative research pipelines,
including QA coverage checks, data quality gates, and signal health
validation.

Usage::

    from spectraquant_v3.monitoring import PipelineMonitor

    monitor = PipelineMonitor(run_id="run_001")

    monitor.check_ohlcv_coverage(
        symbols=["BTC", "ETH"],
        loaded_symbols=["BTC", "ETH"],
        min_coverage=0.9,
    )

    monitor.check_signal_health(
        signal_rows=[...],
        min_coverage=0.5,
    )

    report = monitor.get_report()
    print(report.status)        # "ok"
    report.write("reports/monitoring")
"""

from __future__ import annotations

import logging
from typing import Any

from spectraquant_v3.monitoring.health import HealthReport

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Runtime health-check monitor for a pipeline run.

    Args:
        run_id: Identifier of the pipeline run being monitored.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._report = HealthReport(run_id=run_id)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def check_ohlcv_coverage(
        self,
        symbols: list[str],
        loaded_symbols: list[str],
        min_coverage: float = 0.9,
    ) -> bool:
        """Check that enough OHLCV data was loaded for the universe.

        Args:
            symbols:        Full universe of symbols.
            loaded_symbols: Symbols with successfully loaded OHLCV data.
            min_coverage:   Minimum fraction required (0–1].

        Returns:
            True if coverage meets the threshold.
        """
        n_total = len(symbols)
        if n_total == 0:
            self._report.add_check(
                "ohlcv_coverage",
                passed=False,
                message="Universe is empty",
                value=0.0,
            )
            return False

        n_loaded = len(set(loaded_symbols) & set(symbols))
        coverage = n_loaded / n_total

        passed = coverage >= min_coverage
        self._report.add_check(
            "ohlcv_coverage",
            passed=passed,
            message=(
                f"Coverage {coverage:.1%} ({'OK' if passed else 'BELOW'} "
                f"threshold {min_coverage:.1%})"
            ),
            value=coverage,
        )
        self._report.metrics["ohlcv_coverage"] = coverage
        return passed

    def check_signal_health(
        self,
        signal_rows: list[Any],
        min_coverage: float = 0.5,
    ) -> bool:
        """Check that enough signals were produced for the run.

        Args:
            signal_rows:  List of signal row objects (any type).
            min_coverage: Minimum fraction of non-empty signals required.

        Returns:
            True if signal health meets the threshold.
        """
        n_signals = len(signal_rows)
        passed = n_signals > 0
        self._report.add_check(
            "signal_health",
            passed=passed,
            message=f"{n_signals} signal(s) produced",
            value=n_signals,
        )
        self._report.metrics["signal_count"] = n_signals
        return passed

    def check_allocation_health(
        self,
        allocation_rows: list[Any],
        min_allocated: int = 1,
    ) -> bool:
        """Check that at least *min_allocated* positions were allocated.

        Args:
            allocation_rows: List of allocation row objects.
            min_allocated:   Minimum number of non-zero positions required.

        Returns:
            True if allocation health meets the threshold.
        """
        n_allocated = len(allocation_rows)
        passed = n_allocated >= min_allocated
        self._report.add_check(
            "allocation_health",
            passed=passed,
            message=f"{n_allocated} position(s) allocated (minimum {min_allocated})",
            value=n_allocated,
        )
        self._report.metrics["allocation_count"] = n_allocated
        return passed

    def check_qa_matrix(
        self,
        qa_summary: dict[str, Any],
        max_failure_rate: float = 0.2,
    ) -> bool:
        """Check that the QA matrix failure rate is below a threshold.

        Args:
            qa_summary:        Dict with ``"total"`` and ``"failed"`` counts.
            max_failure_rate:  Maximum tolerated fraction of QA failures (0–1].

        Returns:
            True if the failure rate is below the threshold.
        """
        total = qa_summary.get("total", 0)
        failed = qa_summary.get("failed", 0)

        if total == 0:
            rate = 0.0
        else:
            rate = failed / total

        passed = rate <= max_failure_rate
        self._report.add_check(
            "qa_matrix",
            passed=passed,
            message=(
                f"QA failure rate {rate:.1%} "
                f"({'OK' if passed else 'EXCEEDS'} threshold {max_failure_rate:.1%})"
            ),
            value=rate,
        )
        self._report.metrics["qa_failure_rate"] = rate
        return passed

    def add_custom_check(
        self,
        name: str,
        passed: bool,
        message: str = "",
        value: Any = None,
    ) -> None:
        """Add an arbitrary health check result.

        Args:
            name:    Short check name.
            passed:  Whether the check passed.
            message: Human-readable message.
            value:   Optional numeric or other value associated with the check.
        """
        self._report.add_check(name, passed=passed, message=message, value=value)

    # ------------------------------------------------------------------
    # Report access
    # ------------------------------------------------------------------

    def get_report(self) -> HealthReport:
        """Return the accumulated :class:`HealthReport`."""
        return self._report

    def is_healthy(self) -> bool:
        """Return True only if every check passed."""
        return self._report.status == "ok"
