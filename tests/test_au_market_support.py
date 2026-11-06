# -*- coding: utf-8 -*-
"""ASX (Australia, Yahoo `.AX` suffix) market support."""

from datetime import date

import pytest

from data_provider.base import _market_tag, normalize_stock_code
from data_provider.yfinance_fetcher import YfinanceFetcher
from src.core.trading_calendar import MARKET_EXCHANGE, MARKET_TIMEZONE, get_market_for_stock, is_market_open
from src.market_context import detect_market, get_market_guidelines, get_market_role
from src.services.market_symbol_utils import get_suffix_market, is_au_suffix_symbol
from src.services.portfolio_service import VALID_MARKETS
from src.services.stock_code_utils import resolve_daily_stock_identity
from src.services.stock_list_parser import parse_stock_list


@pytest.mark.parametrize("code", ["BHP.AX", "bhp.ax", "PDN.AX", "4DX.AX", "VGS.AX", "CBAPK.AX"])
def test_asx_codes_detected_as_au(code):
    assert get_suffix_market(code) == "au"
    assert is_au_suffix_symbol(code)
    assert detect_market(code) == "au"
    assert _market_tag(code) == "au"
    assert get_market_for_stock(code) == "au"


@pytest.mark.parametrize("code", ["AB.AX", "TOOLONGX.AX", "BH-P.AX", ".AX"])
def test_invalid_asx_shapes_rejected(code):
    assert get_suffix_market(code) is None


def test_existing_markets_unchanged():
    assert detect_market("AAPL") == "us"
    assert detect_market("BRK.B") == "us"
    assert detect_market("7203.T") == "jp"
    assert detect_market("2330.TW") == "tw"
    assert detect_market("600519") == "cn"
    assert detect_market("hk00700") == "hk"


def test_normalize_keeps_ax_suffix_uppercase():
    assert normalize_stock_code("bhp.ax") == "BHP.AX"
    assert normalize_stock_code("BHP.AX") == "BHP.AX"


def test_stock_list_parser_assigns_ax_exchange():
    targets = parse_stock_list("BHP.AX,pdn.ax,AAPL")
    assert [(t.canonical_id, t.exchange) for t in targets] == [
        ("BHP.AX", "AX"),
        ("PDN.AX", "AX"),
        ("AAPL", "US"),
    ]


def test_daily_identity_is_au():
    identity = resolve_daily_stock_identity("BHP.AX")
    assert identity is not None
    assert identity.market == "au"
    assert identity.normalized_code == "BHP.AX"


def test_calendar_uses_asx():
    assert MARKET_EXCHANGE["au"] == "XASX"
    assert MARKET_TIMEZONE["au"] == "Australia/Sydney"
    assert is_market_open("au", date(2026, 12, 25)) is False
    assert is_market_open("au", date(2026, 11, 7)) is False  # Saturday


def test_yfinance_passes_ax_symbol_through():
    assert YfinanceFetcher()._convert_stock_code("BHP.AX") == "BHP.AX"


def test_prompt_context_and_valid_market():
    assert "ASX" in get_market_role("BHP.AX", "en")
    assert "AUD" in get_market_guidelines("BHP.AX", "en")
    assert "au" in VALID_MARKETS
