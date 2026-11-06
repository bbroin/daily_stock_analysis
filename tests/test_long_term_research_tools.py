# -*- coding: utf-8 -*-
"""Tests for long-term research tools: company history search and annual statements."""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.yfinance_fundamental_adapter import build_annual_statements  # noqa: E402
from src.search_service import SearchResponse, SearchResult, SearchService  # noqa: E402


class _RecordingProvider:
    name = "Fake"
    is_available = True

    def __init__(self):
        self.calls = []

    def search(self, query, max_results=5, days=7, **kwargs):
        self.calls.append({"query": query, "days": days, "kwargs": kwargs})
        old_date = "2023-03-15"  # far outside any NEWS_MAX_AGE_DAYS window
        return SearchResponse(
            query=query,
            results=[
                SearchResult(
                    title="Paladin Energy appoints new CEO",
                    snippet="Paladin Energy announced ...",
                    url="https://example.com/paladin-ceo",
                    source="example.com",
                    published_date=old_date,
                )
            ],
            provider=self.name,
            success=True,
        )


def _service(provider):
    service = SearchService(searxng_public_instances_enabled=False, news_max_age_days=3)
    service._providers = [provider]
    return service


@pytest.mark.parametrize("code", ["PDN.AX", "7203.T", "005930.KS", "AAPL", "00700.HK"])
def test_suffix_markets_are_foreign_for_english_queries(code):
    assert SearchService._is_foreign_stock(code) is True


@pytest.mark.parametrize("code", ["600519", "600519.SH", "000001.SZ"])
def test_a_shares_stay_domestic(code):
    assert SearchService._is_foreign_stock(code) is False


def test_company_history_uses_english_long_lookback_and_keeps_old_results():
    provider = _RecordingProvider()
    with patch("src.search_service.time.sleep"):
        results = _service(provider).search_company_history("PDN.AX", "Paladin Energy")

    assert set(results) == {"current_leadership", "leadership", "capital_allocation", "execution", "strategy"}
    assert all(call["days"] == SearchService.COMPANY_HISTORY_LOOKBACK_DAYS for call in provider.calls)
    assert all("Paladin Energy" in call["query"] and "最新" not in call["query"] for call in provider.calls)
    # Old (2023) articles are not dropped by the news freshness window.
    assert results["leadership"].results
    assert results["leadership"].results[0].published_date.startswith("2023-03-15")


def test_company_history_respects_max_searches_and_skips_etfs():
    provider = _RecordingProvider()
    service = _service(provider)
    with patch("src.search_service.time.sleep"):
        assert len(service.search_company_history("PDN.AX", "Paladin Energy", max_searches=2)) == 2
        assert service.search_company_history("SPY", "SPDR S&P 500 ETF Trust") == {}


def test_company_history_tool_formats_dimensions():
    from src.agent.tools import search_tools

    provider = _RecordingProvider()
    service = _service(provider)
    with patch.object(search_tools, "_get_search_service", return_value=service), \
            patch("src.search_service.time.sleep"):
        payload = search_tools._handle_search_company_history("PDN.AX", "Paladin Energy")

    assert "lookback_note" in payload
    assert payload["dimensions"]["leadership"]["results"][0]["title"] == "Paladin Energy appoints new CEO"


class _FakeTicker:
    def __init__(self):
        cols = pd.to_datetime(["2022-06-30", "2025-06-30", "2023-06-30", "2024-06-30"])
        self.income_stmt = pd.DataFrame(
            {
                cols[0]: [100.0, -10.0],
                cols[1]: [200.0, 30.0],
                cols[2]: [120.0, -5.0],
                cols[3]: [160.0, 12.0],
            },
            index=["Total Revenue", "Net Income"],
        )
        self.balance_sheet = pd.DataFrame(
            {
                cols[1]: [1500.0, 500.0, 300.0],
                cols[0]: [1000.0, 200.0, 100.0],
            },
            index=["Ordinary Shares Number", "Total Debt", "Cash And Cash Equivalents"],
        )
        self.cashflow = pd.DataFrame()


def test_annual_statements_sorted_newest_first_with_derived_signals():
    result = build_annual_statements(_FakeTicker(), max_years=4)

    assert result["status"] == "ok"
    assert result["periods"] == ["2025-06-30", "2024-06-30", "2023-06-30", "2022-06-30"]
    revenue = result["income_statement"]["revenue"]
    assert list(revenue) == result["periods"]
    derived = result["derived"]
    assert derived["revenue_cagr_pct"] == pytest.approx(((200 / 100) ** (1 / 3) - 1) * 100, abs=0.01)
    assert derived["share_count_change_pct"] == pytest.approx(50.0)
    assert derived["net_debt_latest"] == pytest.approx(200.0)
    assert derived["loss_making_years"] == 2
    assert "cash_flow" in result and result["cash_flow"] == {}


def test_annual_statements_not_supported_when_empty():
    class Empty:
        income_stmt = pd.DataFrame()
        balance_sheet = pd.DataFrame()
        cashflow = pd.DataFrame()

    assert build_annual_statements(Empty())["status"] == "not_supported"


def test_annual_financials_tool_marks_unavailable_data():
    from src.agent.tools import data_tools

    with patch(
        "data_provider.yfinance_fundamental_adapter.get_annual_statements",
        return_value={"status": "not_supported", "periods": [], "errors": []},
    ):
        payload = data_tools._handle_get_annual_financials("PDN.AX", years="bad")
    assert payload["code"] == "PDN.AX"
    assert "Do not invent figures" in payload["note"]


def test_new_tools_registered():
    from src.agent.factory import get_tool_registry

    registry = get_tool_registry()
    for name in ("get_annual_financials", "search_company_history"):
        assert registry.resolve(name) is not None


def test_current_leadership_query_names_this_and_last_year():
    from datetime import datetime

    provider = _RecordingProvider()
    with patch("src.search_service.time.sleep"):
        _service(provider).search_company_history("PDN.AX", "Paladin Energy")
    year = datetime.now().year
    first = provider.calls[0]["query"]
    assert "CEO" in first and str(year) in first and str(year - 1) in first


def test_history_summary_mode_is_compact_and_correct():
    from src.agent.tools.data_tools import _summarize_history_records

    records = [
        {"date": f"2026-01-{(i % 28) + 1:02d}", "close": 10.0 + i, "high": 10.5 + i, "low": 9.5 + i}
        for i in range(260)
    ]
    records[100]["high"] = 999.0
    summary = _summarize_history_records(records)
    assert summary["last_close"] == 269.0
    assert summary["high_52w"] == 999.0
    assert summary["low_52w"] == records[8]["low"]  # window is the last 252 rows
    assert summary["drawdown_from_52w_high_pct"] == round((269.0 / 999.0 - 1) * 100, 2)
    assert summary["return_pct"]["1m"] == round((269.0 / 248.0 - 1) * 100, 2)
    assert len(summary["recent_closes"]) == 20
    assert _summarize_history_records([]) == {}


def test_daily_history_summary_flag_replaces_rows():
    from src.agent.tools import data_tools

    df = pd.DataFrame(
        {"date": pd.date_range("2025-01-01", periods=30), "close": range(30), "high": range(30), "low": range(30)}
    )
    with patch("src.services.history_loader.load_history_df", return_value=(df, "db_cache")):
        full = data_tools._handle_get_daily_history("PDN.AX", days=30)
        compact = data_tools._handle_get_daily_history("PDN.AX", days=30, summary=True)
    assert len(full["data"]) == 30 and "summary" not in full
    assert "data" not in compact and compact["summary"]["trading_days"] == 30


def test_skill_allowlist_restricts_tool_surface_only_when_all_skills_declare_it():
    from types import SimpleNamespace

    from src.agent.factory import _restrict_registry_to_skill_allowlist, get_tool_registry

    registry = get_tool_registry()
    skills = {
        "lt": SimpleNamespace(agent_tool_allowlist=["get_annual_financials", "search_company_history", "no_such_tool"]),
        "open": SimpleNamespace(agent_tool_allowlist=[]),
        # SKILL.md allowed-tools is metadata only and must not restrict tools.
        "bundle": SimpleNamespace(allowed_tools=["get_annual_financials"], agent_tool_allowlist=[]),
    }
    manager = SimpleNamespace(get=skills.get)

    restricted = _restrict_registry_to_skill_allowlist(registry, manager, ["lt"])
    assert sorted(restricted.list_names()) == ["get_annual_financials", "search_company_history"]
    assert restricted.category_timeout_map == registry.category_timeout_map
    # Mixed selection (one skill without an allowlist) keeps the full surface.
    assert _restrict_registry_to_skill_allowlist(registry, manager, ["lt", "open"]) is registry
    assert _restrict_registry_to_skill_allowlist(registry, manager, []) is registry
    assert _restrict_registry_to_skill_allowlist(registry, manager, ["bundle"]) is registry


def test_long_term_skill_executor_gets_small_tool_surface():
    from types import SimpleNamespace

    from src.agent.factory import build_agent_executor

    config = SimpleNamespace(
        agent_skills=["long_term_investor"], agent_skill_dir=None, agent_arch="single",
        agent_max_steps=10, agent_orchestrator_timeout_s=0,
    )
    with patch("src.agent.llm_adapter.LLMToolAdapter.__init__", return_value=None):
        try:
            executor = build_agent_executor(config, ["long_term_investor"])
        except Exception as exc:  # pragma: no cover - environment-specific config needs
            pytest.skip(f"executor build needs full config: {exc}")
    names = set(executor.tool_registry.list_names())
    assert names == {
        "get_realtime_quote", "get_daily_history", "calculate_ma", "get_stock_info",
        "get_annual_financials", "search_company_history", "search_stock_news",
    }
