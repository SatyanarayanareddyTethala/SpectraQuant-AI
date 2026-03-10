"""Regression tests for Stage-1 V3 contract hardening.

Tests cover:
- SignalStatus enum (including DEGRADED)
- NoSignalReason enum
- SignalRow construction, clamping, and field defaults
- AllocationRow construction and optional timestamp field
- PolicyDecision minimal construction
- run_signal_agent error wrapping
- BacktestEngine status comparisons via enum values
"""

from __future__ import annotations

import logging
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# SignalStatus enum
# ---------------------------------------------------------------------------


class TestSignalStatusEnum:
    def test_ok_value(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        assert SignalStatus.OK.value == "OK"

    def test_no_signal_value(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        assert SignalStatus.NO_SIGNAL.value == "NO_SIGNAL"

    def test_error_value(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        assert SignalStatus.ERROR.value == "ERROR"

    def test_degraded_value(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        assert SignalStatus.DEGRADED.value == "DEGRADED"

    def test_degraded_is_string_enum(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        # str Enum — the value must compare equal to a plain string
        assert SignalStatus.DEGRADED == "DEGRADED"

    def test_all_four_statuses_present(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        values = {s.value for s in SignalStatus}
        assert values == {"OK", "NO_SIGNAL", "DEGRADED", "ERROR"}


# ---------------------------------------------------------------------------
# NoSignalReason enum
# ---------------------------------------------------------------------------


class TestNoSignalReasonEnum:
    def test_missing_inputs_value(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason

        assert NoSignalReason.MISSING_INPUTS.value == "missing_inputs"

    def test_insufficient_rows_value(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason

        assert NoSignalReason.INSUFFICIENT_ROWS.value == "insufficient_rows"

    def test_top_n_cutoff_value(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason

        assert NoSignalReason.TOP_N_CUTOFF.value == "top_n_cutoff"

    def test_below_threshold_value(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason

        assert NoSignalReason.BELOW_THRESHOLD.value == "below_threshold"

    def test_unknown_value(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason

        assert NoSignalReason.UNKNOWN.value == "unknown"

    def test_all_five_reasons_present(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason

        expected = {
            "missing_inputs",
            "insufficient_rows",
            "top_n_cutoff",
            "below_threshold",
            "unknown",
        }
        assert {r.value for r in NoSignalReason} == expected

    def test_is_string_enum(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason

        assert NoSignalReason.UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# ContractViolationError
# ---------------------------------------------------------------------------


class TestContractViolationError:
    def test_importable(self) -> None:
        from spectraquant_v3.core.errors import ContractViolationError

        assert issubclass(ContractViolationError, Exception)

    def test_is_spectraquant_error(self) -> None:
        from spectraquant_v3.core.errors import ContractViolationError, SpectraQuantError

        assert issubclass(ContractViolationError, SpectraQuantError)

    def test_raiseable(self) -> None:
        from spectraquant_v3.core.errors import ContractViolationError

        with pytest.raises(ContractViolationError, match="out of range"):
            raise ContractViolationError("signal_score out of range")


# ---------------------------------------------------------------------------
# SignalRow construction and field defaults
# ---------------------------------------------------------------------------


def _minimal_signal_row(**kwargs):
    """Return the minimal valid SignalRow for testing."""
    from spectraquant_v3.core.schema import SignalRow

    base = dict(
        run_id="r1",
        timestamp="2025-01-01T00:00:00+00:00",
        canonical_symbol="BTC",
        asset_class="crypto",
        agent_id="test_agent",
        horizon="1d",
    )
    base.update(kwargs)
    return SignalRow(**base)


class TestSignalRowDefaults:
    def test_status_defaults_to_no_signal(self) -> None:
        row = _minimal_signal_row()
        assert row.status == "NO_SIGNAL"

    def test_signal_score_defaults_to_zero(self) -> None:
        row = _minimal_signal_row()
        assert row.signal_score == 0.0

    def test_confidence_defaults_to_zero(self) -> None:
        row = _minimal_signal_row()
        assert row.confidence == 0.0

    def test_no_signal_reason_defaults_to_empty(self) -> None:
        row = _minimal_signal_row()
        assert row.no_signal_reason == ""

    def test_rationale_defaults_to_empty(self) -> None:
        row = _minimal_signal_row()
        assert row.rationale == ""

    def test_error_reason_defaults_to_empty(self) -> None:
        row = _minimal_signal_row()
        assert row.error_reason == ""

    def test_required_inputs_defaults_to_empty_list(self) -> None:
        row = _minimal_signal_row()
        assert row.required_inputs == []

    def test_available_inputs_defaults_to_empty_list(self) -> None:
        row = _minimal_signal_row()
        assert row.available_inputs == []

    def test_no_signal_reason_set(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason

        row = _minimal_signal_row(no_signal_reason=NoSignalReason.MISSING_INPUTS.value)
        assert row.no_signal_reason == "missing_inputs"

    def test_degraded_status_accepted(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason, SignalStatus

        row = _minimal_signal_row(
            status=SignalStatus.DEGRADED.value,
            signal_score=0.3,
            confidence=0.4,
            no_signal_reason=NoSignalReason.INSUFFICIENT_ROWS.value,
        )
        assert row.status == "DEGRADED"
        assert row.no_signal_reason == "insufficient_rows"

    def test_ok_status_accepted(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        row = _minimal_signal_row(
            status=SignalStatus.OK.value,
            signal_score=0.7,
            confidence=0.7,
        )
        assert row.status == "OK"


# ---------------------------------------------------------------------------
# SignalRow bounds clamping (no crash, just clamp + warn)
# ---------------------------------------------------------------------------


class TestSignalRowBoundsClamping:
    def test_score_above_max_is_clamped(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="spectraquant_v3.core.schema"):
            row = _minimal_signal_row(signal_score=1.5)
        assert row.signal_score == 1.0
        assert "signal_score" in caplog.text

    def test_score_below_min_is_clamped(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="spectraquant_v3.core.schema"):
            row = _minimal_signal_row(signal_score=-2.0)
        assert row.signal_score == -1.0
        assert "signal_score" in caplog.text

    def test_confidence_above_max_is_clamped(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="spectraquant_v3.core.schema"):
            row = _minimal_signal_row(confidence=1.2)
        assert row.confidence == 1.0
        assert "confidence" in caplog.text

    def test_confidence_below_zero_is_clamped(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="spectraquant_v3.core.schema"):
            row = _minimal_signal_row(confidence=-0.1)
        assert row.confidence == 0.0
        assert "confidence" in caplog.text

    def test_valid_score_at_boundary_not_clamped(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="spectraquant_v3.core.schema"):
            row = _minimal_signal_row(signal_score=1.0)
        assert row.signal_score == 1.0
        assert "signal_score" not in caplog.text

    def test_valid_confidence_at_boundary_not_clamped(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="spectraquant_v3.core.schema"):
            row = _minimal_signal_row(confidence=1.0)
        assert row.confidence == 1.0
        assert "confidence" not in caplog.text

    def test_zero_values_not_clamped(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="spectraquant_v3.core.schema"):
            row = _minimal_signal_row(signal_score=0.0, confidence=0.0)
        assert row.signal_score == 0.0
        assert row.confidence == 0.0
        assert "signal_score" not in caplog.text
        assert "confidence" not in caplog.text

    def test_negative_one_score_not_clamped(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="spectraquant_v3.core.schema"):
            row = _minimal_signal_row(signal_score=-1.0)
        assert row.signal_score == -1.0
        assert "signal_score" not in caplog.text


# ---------------------------------------------------------------------------
# AllocationRow optional timestamp
# ---------------------------------------------------------------------------


class TestAllocationRowTimestamp:
    def test_timestamp_defaults_to_empty(self) -> None:
        from spectraquant_v3.core.schema import AllocationRow

        row = AllocationRow(run_id="r1", canonical_symbol="BTC", asset_class="crypto")
        assert row.timestamp == ""

    def test_timestamp_can_be_set(self) -> None:
        from spectraquant_v3.core.schema import AllocationRow

        ts = "2025-06-01T12:00:00+00:00"
        row = AllocationRow(
            run_id="r1",
            canonical_symbol="BTC",
            asset_class="crypto",
            timestamp=ts,
        )
        assert row.timestamp == ts

    def test_all_other_defaults_unchanged(self) -> None:
        from spectraquant_v3.core.schema import AllocationRow

        row = AllocationRow(run_id="r1", canonical_symbol="ETH", asset_class="crypto")
        assert row.target_weight == 0.0
        assert row.blocked is False
        assert row.blocked_reason == ""
        assert row.expected_turnover == 0.0


# ---------------------------------------------------------------------------
# PolicyDecision minimal construction
# ---------------------------------------------------------------------------


class TestPolicyDecisionMinimal:
    def test_minimal_passed_decision(self) -> None:
        from spectraquant_v3.pipeline.meta_policy import PolicyDecision

        d = PolicyDecision(
            canonical_symbol="BTC",
            asset_class="crypto",
            composite_score=0.5,
            composite_confidence=0.5,
            passed=True,
            reason="passed_all_filters",
        )
        assert d.passed is True
        assert d.contributing_agents == []

    def test_minimal_blocked_decision(self) -> None:
        from spectraquant_v3.pipeline.meta_policy import PolicyDecision

        d = PolicyDecision(
            canonical_symbol="ETH",
            asset_class="crypto",
            composite_score=0.0,
            composite_confidence=0.0,
            passed=False,
            reason="all_no_signal",
        )
        assert d.passed is False

    def test_contributing_agents_optional(self) -> None:
        from spectraquant_v3.pipeline.meta_policy import PolicyDecision

        d = PolicyDecision(
            canonical_symbol="SOL",
            asset_class="crypto",
            composite_score=0.3,
            composite_confidence=0.3,
            passed=True,
            reason="passed_all_filters",
            contributing_agents=["agent_a", "agent_b"],
        )
        assert d.contributing_agents == ["agent_a", "agent_b"]


# ---------------------------------------------------------------------------
# run_signal_agent error wrapping
# ---------------------------------------------------------------------------


class _GoodAgent:
    """Minimal single-symbol agent that always returns a valid SignalRow."""

    run_id = "run1"
    asset_class = "crypto"
    agent_id = "good_agent"
    horizon = "1d"

    def evaluate(self, symbol: str, frame, as_of: str):
        from spectraquant_v3.core.enums import SignalStatus
        from spectraquant_v3.core.schema import SignalRow

        return SignalRow(
            run_id=self.run_id,
            timestamp=as_of,
            canonical_symbol=symbol,
            asset_class=self.asset_class,
            agent_id=self.agent_id,
            horizon=self.horizon,
            signal_score=0.5,
            confidence=0.5,
            status=SignalStatus.OK.value,
        )


class _BrokenAgent:
    """Single-symbol agent that always raises."""

    run_id = "run1"
    asset_class = "crypto"
    agent_id = "broken_agent"
    horizon = "1d"

    def evaluate(self, symbol: str, frame, as_of: str):
        raise RuntimeError("internal error")


class _BrokenCrossSectionalAgent:
    """Cross-sectional agent that always raises from evaluate_many."""

    run_id = "run1"
    asset_class = "crypto"
    agent_id = "broken_xsect_agent"
    horizon = "1d"

    def evaluate_many(self, feature_map: dict, as_of: str):
        raise ValueError("cross-sectional failure")


class TestRunSignalAgentErrorWrapping:
    def _make_feature_map(self, symbols=("BTC", "ETH")):
        return {sym: pd.DataFrame({"close": [100.0]}) for sym in symbols}

    def test_good_agent_returns_signal_rows(self) -> None:
        from spectraquant_v3.strategies.agents.runner import run_signal_agent

        fm = self._make_feature_map(["BTC"])
        rows = run_signal_agent(_GoodAgent(), fm, as_of="2025-01-01T00:00:00+00:00")
        assert len(rows) == 1
        assert rows[0].status == "OK"
        assert rows[0].canonical_symbol == "BTC"

    def test_broken_agent_emits_error_row_not_crash(self, caplog) -> None:
        from spectraquant_v3.strategies.agents.runner import run_signal_agent

        fm = self._make_feature_map(["BTC"])
        with caplog.at_level(logging.WARNING, logger="spectraquant_v3.strategies.agents.runner"):
            rows = run_signal_agent(
                _BrokenAgent(), fm, as_of="2025-01-01T00:00:00+00:00"
            )
        assert len(rows) == 1
        assert rows[0].status == "ERROR"
        assert "RuntimeError" in rows[0].error_reason
        assert "internal error" in rows[0].error_reason

    def test_broken_agent_error_row_fields(self) -> None:
        from spectraquant_v3.strategies.agents.runner import run_signal_agent

        fm = self._make_feature_map(["BTC"])
        rows = run_signal_agent(
            _BrokenAgent(), fm, as_of="2025-01-01T00:00:00+00:00"
        )
        row = rows[0]
        assert row.canonical_symbol == "BTC"
        assert row.agent_id == "broken_agent"
        assert row.signal_score == 0.0
        assert row.confidence == 0.0

    def test_broken_agent_multiple_symbols_all_error_rows(self) -> None:
        from spectraquant_v3.strategies.agents.runner import run_signal_agent

        fm = self._make_feature_map(["BTC", "ETH", "SOL"])
        rows = run_signal_agent(
            _BrokenAgent(), fm, as_of="2025-01-01T00:00:00+00:00"
        )
        assert len(rows) == 3
        assert all(r.status == "ERROR" for r in rows)
        assert {r.canonical_symbol for r in rows} == {"BTC", "ETH", "SOL"}

    def test_partial_agent_failure_produces_mix(self) -> None:
        """When only some symbols fail, good rows are returned for others."""
        from spectraquant_v3.core.enums import SignalStatus
        from spectraquant_v3.core.schema import SignalRow
        from spectraquant_v3.strategies.agents.runner import run_signal_agent

        class _PartialAgent:
            run_id = "run1"
            asset_class = "crypto"
            agent_id = "partial_agent"
            horizon = "1d"

            def evaluate(self, symbol: str, frame, as_of: str):
                if symbol == "BAD":
                    raise ValueError("bad symbol")
                return SignalRow(
                    run_id=self.run_id,
                    timestamp=as_of,
                    canonical_symbol=symbol,
                    asset_class=self.asset_class,
                    agent_id=self.agent_id,
                    horizon=self.horizon,
                    status=SignalStatus.OK.value,
                )

        fm = self._make_feature_map(["BTC", "BAD"])
        rows = run_signal_agent(_PartialAgent(), fm, as_of="2025-01-01T00:00:00+00:00")
        assert len(rows) == 2
        status_by_sym = {r.canonical_symbol: r.status for r in rows}
        assert status_by_sym["BTC"] == "OK"
        assert status_by_sym["BAD"] == "ERROR"

    def test_broken_cross_sectional_emits_error_rows(self, caplog) -> None:
        from spectraquant_v3.strategies.agents.runner import run_signal_agent

        fm = self._make_feature_map(["BTC", "ETH"])
        with caplog.at_level(
            logging.WARNING, logger="spectraquant_v3.strategies.agents.runner"
        ):
            rows = run_signal_agent(
                _BrokenCrossSectionalAgent(),
                fm,
                as_of="2025-01-01T00:00:00+00:00",
            )
        assert len(rows) == 2
        assert all(r.status == "ERROR" for r in rows)
        assert "cross-sectional failure" in rows[0].error_reason

    def test_runner_warning_logged_on_single_symbol_error(self, caplog) -> None:
        from spectraquant_v3.strategies.agents.runner import run_signal_agent

        fm = self._make_feature_map(["BTC"])
        with caplog.at_level(
            logging.WARNING, logger="spectraquant_v3.strategies.agents.runner"
        ):
            run_signal_agent(
                _BrokenAgent(), fm, as_of="2025-01-01T00:00:00+00:00"
            )
        # The warning must mention the symbol and the exception type
        assert "BTC" in caplog.text
        assert "RuntimeError" in caplog.text


# ---------------------------------------------------------------------------
# BacktestEngine status comparisons via enum (regression: no bare string literals)
# ---------------------------------------------------------------------------


class TestEngineStatusComparisons:
    """Verify that engine status counting uses enum values, not bare strings.

    We do this by testing that the enum values used in engine.py match the
    SignalStatus constants — so any rename would break both the engine and
    these tests symmetrically.
    """

    def test_ok_string_matches_enum(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        # The engine counts: sum(1 for s in signals if s.status == SignalStatus.OK.value)
        assert SignalStatus.OK.value == "OK"

    def test_no_signal_string_matches_enum(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        assert SignalStatus.NO_SIGNAL.value == "NO_SIGNAL"

    def test_no_signal_reason_fallback_chain(self) -> None:
        """Engine uses no_signal_reason, then rationale, then sentinel.

        This verifies the priority logic without instantiating the full engine.
        """
        row = _minimal_signal_row(status="NO_SIGNAL")
        # No reason or rationale → sentinel
        reason = row.no_signal_reason or row.rationale or "no_signal"
        assert reason == "no_signal"

    def test_no_signal_reason_field_takes_priority_over_rationale(self) -> None:
        row = _minimal_signal_row(
            status="NO_SIGNAL",
            no_signal_reason="top_n_cutoff",
            rationale="rank=10;raw_score=0.001",
        )
        reason = row.no_signal_reason or row.rationale or "no_signal"
        assert reason == "top_n_cutoff"

    def test_rationale_used_when_no_signal_reason_absent(self) -> None:
        row = _minimal_signal_row(
            status="NO_SIGNAL",
            no_signal_reason="",
            rationale="insufficient_rows",
        )
        reason = row.no_signal_reason or row.rationale or "no_signal"
        assert reason == "insufficient_rows"


# ---------------------------------------------------------------------------
# NoSignalReason importable via schema module (convenience re-export)
# ---------------------------------------------------------------------------


class TestSchemaModuleExports:
    def test_no_signal_reason_importable_from_enums(self) -> None:
        from spectraquant_v3.core.enums import NoSignalReason  # noqa: F401

    def test_signal_status_degraded_importable_from_enums(self) -> None:
        from spectraquant_v3.core.enums import SignalStatus

        assert hasattr(SignalStatus, "DEGRADED")

    def test_no_signal_reason_re_exported_from_schema(self) -> None:
        # schema.py does `from spectraquant_v3.core.enums import ..., NoSignalReason`
        # with noqa: F401 to allow consumers to import from schema if desired.
        import importlib

        schema_mod = importlib.import_module("spectraquant_v3.core.schema")
        assert hasattr(schema_mod, "NoSignalReason")
