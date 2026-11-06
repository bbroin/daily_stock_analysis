# -*- coding: utf-8 -*-
"""Tests for the whole-portfolio review (PORTFOLIO_REVIEW_HOLDINGS)."""

import logging
import os
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.portfolio_review_service import (  # noqa: E402
    build_portfolio_snapshot,
    build_review_prompt,
    parse_portfolio_review_holdings,
    render_snapshot_markdown,
    run_portfolio_review,
)

MARKET = {
    "VAS.AX": {
        "name": "Vanguard Australian Shares Index ETF", "quote_type": "ETF", "currency": "AUD",
        "price": 100.0, "sector_weightings": {"financial_services": 0.3, "basic_materials": 0.2, "healthcare": 0.1},
    },
    "PDN.AX": {
        "name": "Paladin Energy Ltd", "quote_type": "EQUITY", "currency": "AUD",
        "price": 10.0, "sector": "Energy", "industry": "Uranium", "country": "Australia",
    },
    "AAPL": {
        "name": "Apple Inc.", "quote_type": "EQUITY", "currency": "USD",
        "price": 200.0, "sector": "Technology", "country": "United States",
    },
}


def _market(code):
    return dict(MARKET.get(code, {"name": code, "price": None}))


def _fx(src, dst):
    return {("USD", "AUD"): 1.5}.get((src, dst), 1.0 if src == dst else None)


def test_parse_units_values_names_and_duplicates():
    holdings, warnings = parse_portfolio_review_holdings(
        "vas.ax:100; PDN.AX:300\nVanguard Diversified Growth:$5000, PDN.AX:200, bad entry, X:-1"
    )
    by_label = {h.label: h for h in holdings}
    assert by_label["VAS.AX"].units == 100
    assert by_label["PDN.AX"].units == 500  # duplicates are summed
    assert by_label["Vanguard Diversified Growth"].value == 5000
    assert by_label["Vanguard Diversified Growth"].units is None
    assert len(warnings) == 2


def test_parse_warnings_never_echo_entry_text():
    _, warnings = parse_portfolio_review_holdings("SECRETCODE123;My Fund:12")
    assert warnings
    assert all("SECRETCODE123" not in w and "My Fund" not in w for w in warnings)


def test_parse_empty_is_disabled():
    assert parse_portfolio_review_holdings("") == ([], [])
    assert parse_portfolio_review_holdings(None) == ([], [])


def test_snapshot_weights_fx_lookthrough_and_unpriced():
    holdings, _ = parse_portfolio_review_holdings("VAS.AX:100,PDN.AX:300,AAPL:10,UNKNOWN.AX:5,My Fund:$7000")
    snap = build_portfolio_snapshot(holdings, base_currency="AUD", market_data_fn=_market, fx_fn=_fx)

    values = {h.label: h.value for h in snap.holdings}
    assert values["VAS.AX"] == pytest.approx(10000)
    assert values["PDN.AX"] == pytest.approx(3000)
    assert values["AAPL"] == pytest.approx(3000)  # 10 * 200 USD * 1.5
    assert values["My Fund"] == pytest.approx(7000)  # user value, never priced
    assert snap.unpriced == ["UNKNOWN.AX"]
    assert snap.total_value == pytest.approx(23000)
    assert sum(h.weight_pct for h in snap.holdings if h.weight_pct) == pytest.approx(100)
    assert snap.top1_pct == pytest.approx(10000 / 23000 * 100)
    # ETF look-through: 30% of VAS weight is financials
    assert snap.sector_exposure_pct["Financial Services"] == pytest.approx(10000 / 23000 * 100 * 0.3)
    assert snap.sector_exposure_pct["Energy"] == pytest.approx(3000 / 23000 * 100)

    table = render_snapshot_markdown(snap)
    assert "AUD 23,000" in table
    assert "Could not price: UNKNOWN.AX" in table


def test_value_only_label_is_not_looked_up():
    calls = []
    holdings, _ = parse_portfolio_review_holdings("My Managed Fund:$1000")
    build_portfolio_snapshot(holdings, market_data_fn=lambda c: calls.append(c) or {}, fx_fn=_fx)
    assert calls == []


def test_prompt_contains_code_computed_weights_and_language():
    holdings, _ = parse_portfolio_review_holdings("VAS.AX:100,PDN.AX:300")
    snap = build_portfolio_snapshot(holdings, market_data_fn=_market, fx_fn=_fx)
    result = SimpleNamespace(code="PDN.AX", name="Paladin", operation_advice="Hold", sentiment_score=62,
                             dashboard={"core_conclusion": {"one_sentence": "Ramp-up risk remains"}},
                             analysis_summary="")
    prompt = build_review_prompt(snap, [result], report_language="en")
    assert "weight=76.9%" in prompt and "weight=23.1%" in prompt
    assert "Ramp-up risk remains" in prompt
    assert "Write the review in English" in prompt
    assert "## Sectors and themes to consider for long-term growth" in prompt


def test_run_review_uses_llm_and_keeps_snapshot_when_llm_fails():
    analyzer = MagicMock()
    analyzer.is_available.return_value = True
    analyzer.generate_text.return_value = "## Strengths\n- Diversified core"
    content = run_portfolio_review("VAS.AX:100", analyzer=analyzer, market_data_fn=_market, fx_fn=_fx)
    assert "## Portfolio snapshot" in content and "Diversified core" in content

    analyzer.generate_text.side_effect = RuntimeError("boom")
    content = run_portfolio_review("VAS.AX:100", analyzer=analyzer, market_data_fn=_market, fx_fn=_fx)
    assert "## Portfolio snapshot" in content
    assert "could not be generated" in content


def test_run_review_skips_when_nothing_priced():
    assert run_portfolio_review("UNKNOWN.AX:5", analyzer=None, market_data_fn=_market, fx_fn=_fx) is None
    assert run_portfolio_review("garbage", analyzer=None, market_data_fn=_market, fx_fn=_fx) is None


def test_run_review_does_not_log_holdings(caplog):
    caplog.set_level(logging.DEBUG)
    run_portfolio_review("PDN.AX:300;Secret Fund:12", analyzer=None, market_data_fn=_market, fx_fn=_fx)
    assert "PDN.AX" not in caplog.text and "Secret Fund" not in caplog.text and "300" not in caplog.text


class TestMainHook:
    def _pipeline(self, available=True):
        notifier = MagicMock()
        notifier.is_available.return_value = available
        notifier.send.return_value = True
        return SimpleNamespace(analyzer=None, notifier=notifier)

    def _config(self, holdings="VAS.AX:100"):
        return SimpleNamespace(portfolio_review_holdings=holdings, portfolio_review_currency="AUD",
                               report_language="en")

    def test_disabled_when_unset(self, monkeypatch):
        import main
        called = MagicMock()
        monkeypatch.setattr("src.services.portfolio_review_service.run_portfolio_review", called)
        main._run_portfolio_review(self._config(""), self._pipeline(), [], Namespace(no_notify=False))
        called.assert_not_called()

    def test_sends_as_separate_message(self, monkeypatch):
        import main
        monkeypatch.setattr("src.services.portfolio_review_service.run_portfolio_review",
                            lambda *a, **k: "# review")
        pipeline = self._pipeline()
        main._run_portfolio_review(self._config(), pipeline, [], Namespace(no_notify=False))
        pipeline.notifier.send.assert_called_once_with("# review", email_send_to_all=True, route_type="report")
        pipeline.notifier.save_report_to_file.assert_not_called()

    def test_never_saved_to_artifact_dir_on_actions(self, monkeypatch):
        import main
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setattr("src.services.portfolio_review_service.run_portfolio_review",
                            lambda *a, **k: "# review")
        pipeline = self._pipeline(available=False)
        main._run_portfolio_review(self._config(), pipeline, [], Namespace(no_notify=False))
        pipeline.notifier.save_report_to_file.assert_not_called()
        pipeline.notifier.send.assert_not_called()

    def test_saved_locally_when_no_notify_outside_actions(self, monkeypatch):
        import main
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.setattr("src.services.portfolio_review_service.run_portfolio_review",
                            lambda *a, **k: "# review")
        pipeline = self._pipeline()
        main._run_portfolio_review(self._config(), pipeline, [], Namespace(no_notify=True))
        pipeline.notifier.send.assert_not_called()
        assert pipeline.notifier.save_report_to_file.call_args.kwargs["filename"].startswith("portfolio_review_")

    def test_failure_does_not_raise(self, monkeypatch):
        import main

        def boom(*a, **k):
            raise RuntimeError("x")

        monkeypatch.setattr("src.services.portfolio_review_service.run_portfolio_review", boom)
        main._run_portfolio_review(self._config(), self._pipeline(), [], Namespace(no_notify=False))
