# -*- coding: utf-8 -*-
"""
Portfolio review (long-term, whole-portfolio view).

Reads a holdings list from ``PORTFOLIO_REVIEW_HOLDINGS`` (``CODE:units`` or
``CODE:$value``), prices it, aggregates weights / sectors / concentration
deterministically, and asks the LLM for a strengths / weaknesses /
sector-gap review. The numeric snapshot table is rendered by code, never by the
LLM, so weights and values in the report cannot be hallucinated.

Privacy: holdings are user-private. This module never logs holding codes,
units or values (GitHub only masks the full secret string, not fragments), and
the caller sends the review through notification channels instead of saving it
into ``reports/`` (which the Actions workflow uploads as an artifact).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_SPLIT_RE = re.compile(r"[,;\n]+")
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
_TICKER_RE = re.compile(r"^[A-Za-z0-9.^\-]+$")

MAX_HOLDINGS = 40


@dataclass
class ReviewHolding:
    """One parsed holding line."""

    label: str                      # ticker as written, or a free-text name for value-only lines
    units: Optional[float] = None   # number of units (priced from market data)
    value: Optional[float] = None   # user-provided value in the base currency (not priced)

    @property
    def is_ticker(self) -> bool:
        return bool(_TICKER_RE.match(self.label))


@dataclass
class HoldingSnapshot:
    label: str
    name: str
    quote_type: str = ""            # EQUITY / ETF / MUTUALFUND / ...
    sector: str = ""
    industry: str = ""
    country: str = ""
    category: str = ""              # fund category, if any
    currency: str = ""
    units: Optional[float] = None
    price: Optional[float] = None
    value: Optional[float] = None   # in base currency
    weight_pct: Optional[float] = None
    sector_weightings: Dict[str, float] = field(default_factory=dict)  # ETF look-through (0-1)
    top_holdings: List[Tuple[str, float]] = field(default_factory=list)  # ETF top holdings (name, 0-1)
    value_source: str = ""          # "market" | "user"
    error: str = ""


@dataclass
class PortfolioSnapshot:
    base_currency: str
    holdings: List[HoldingSnapshot]
    total_value: float
    unpriced: List[str]
    sector_exposure_pct: Dict[str, float]
    asset_type_pct: Dict[str, float]
    top1_pct: Optional[float]
    top3_pct: Optional[float]
    effective_positions: Optional[float]   # 1 / HHI


def parse_portfolio_review_holdings(raw: Optional[str]) -> Tuple[List[ReviewHolding], List[str]]:
    """Parse ``CODE:units`` / ``CODE:$value`` entries.

    Entries are separated by commas, semicolons or newlines, so numbers must not
    contain thousands separators (write ``5000``, not ``5,000``). Value-only
    entries (``$``) may use a free-text label, e.g. a managed fund name.

    Returns ``(holdings, warnings)``. Warnings never echo the raw entry, only
    its position, so they are safe to log.
    """
    holdings: List[ReviewHolding] = []
    warnings: List[str] = []
    if not raw or not str(raw).strip():
        return holdings, warnings

    seen: Dict[str, int] = {}
    for index, entry in enumerate(_SPLIT_RE.split(str(raw)), start=1):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            warnings.append(f"entry #{index}: missing ':' (expected CODE:units or CODE:$value)")
            continue
        label, _, amount = entry.rpartition(":")
        label = label.strip()
        amount = amount.strip().replace(" ", "")
        if not label:
            warnings.append(f"entry #{index}: empty code")
            continue

        is_value = amount.startswith("$") or amount.endswith("$")
        number_text = amount.strip("$")
        if not _NUMBER_RE.match(number_text):
            warnings.append(f"entry #{index}: amount is not a plain number")
            continue
        number = float(number_text)
        if number <= 0:
            warnings.append(f"entry #{index}: amount must be positive")
            continue

        if not is_value and not _TICKER_RE.match(label):
            warnings.append(f"entry #{index}: a name with spaces needs a $ value (unit counts need a ticker)")
            continue

        key = label.upper()
        holding = ReviewHolding(
            label=label.upper() if _TICKER_RE.match(label) else label,
            units=None if is_value else number,
            value=number if is_value else None,
        )
        if key in seen:
            existing = holdings[seen[key]]
            if (existing.units is None) != (holding.units is None):
                warnings.append(f"entry #{index}: duplicate code mixes units and $ value; kept the first")
                continue
            if holding.units is not None:
                existing.units = (existing.units or 0) + holding.units
            else:
                existing.value = (existing.value or 0) + (holding.value or 0)
            continue
        if len(holdings) >= MAX_HOLDINGS:
            warnings.append(f"more than {MAX_HOLDINGS} holdings; extra entries ignored")
            break
        seen[key] = len(holdings)
        holdings.append(holding)

    return holdings, warnings


# ---------------------------------------------------------------------------
# Market data (yfinance, best effort)
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _fast_price(fast: Any) -> Optional[float]:
    for getter in (
        lambda: getattr(fast, "last_price", None),
        lambda: fast["lastPrice"],
    ):
        try:
            price = _safe_float(getter())
        except Exception:
            price = None
        if price is not None:
            return price
    return None


def fetch_holding_market_data(code: str) -> Dict[str, Any]:
    """Fetch price + classification for one ticker via yfinance.

    Returns a plain dict (keys: name, quote_type, sector, industry, country,
    category, currency, price, sector_weightings, top_holdings, error).
    """
    out: Dict[str, Any] = {"error": ""}
    try:
        import yfinance as yf
        from data_provider.yfinance_fetcher import YfinanceFetcher
    except Exception as exc:  # pragma: no cover - import guard
        out["error"] = f"import:{type(exc).__name__}"
        return out

    symbol = YfinanceFetcher()._convert_stock_code(code)
    ticker = yf.Ticker(symbol)

    info: Dict[str, Any] = {}
    try:
        info = ticker.get_info() if hasattr(ticker, "get_info") else (ticker.info or {})
        if not isinstance(info, dict):
            info = {}
    except Exception as exc:
        out["error"] = f"info:{type(exc).__name__}"

    price = None
    try:
        price = _fast_price(ticker.fast_info)
    except Exception:
        price = None
    if price is None:
        price = _safe_float(info.get("regularMarketPrice") or info.get("previousClose") or info.get("navPrice"))
    if price is None:
        try:
            hist = ticker.history(period="5d")
            if hist is not None and not hist.empty:
                price = _safe_float(hist["Close"].iloc[-1])
        except Exception as exc:
            out["error"] = out["error"] or f"history:{type(exc).__name__}"

    out.update(
        {
            "name": info.get("longName") or info.get("shortName") or code,
            "quote_type": str(info.get("quoteType") or ""),
            "sector": str(info.get("sector") or ""),
            "industry": str(info.get("industry") or ""),
            "country": str(info.get("country") or ""),
            "category": str(info.get("category") or ""),
            "currency": str(info.get("currency") or ""),
            "price": price,
            "sector_weightings": {},
            "top_holdings": [],
        }
    )

    if out["quote_type"].upper() in {"ETF", "MUTUALFUND"}:
        try:
            funds = ticker.funds_data
            weights = getattr(funds, "sector_weightings", None) or {}
            out["sector_weightings"] = {
                str(k): float(v) for k, v in dict(weights).items() if _safe_float(v)
            }
            top = getattr(funds, "top_holdings", None)
            if top is not None and not getattr(top, "empty", True):
                weight_col = "Holding Percent" if "Holding Percent" in top.columns else top.columns[-1]
                name_col = "Name" if "Name" in top.columns else None
                rows = []
                for idx, row in top.head(10).iterrows():
                    label = str(row[name_col]) if name_col else str(idx)
                    pct = _safe_float(row[weight_col])
                    if pct is not None:
                        rows.append((label, pct))
                out["top_holdings"] = rows
        except Exception:
            # Look-through data is optional; many ASX ETFs have none on Yahoo.
            pass
    return out


def fetch_fx_rate(from_currency: str, to_currency: str) -> Optional[float]:
    """Return the FX rate to convert 1 unit of ``from_currency`` into ``to_currency``."""
    if not from_currency or not to_currency or from_currency == to_currency:
        return 1.0
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{from_currency}{to_currency}=X")
        rate = _fast_price(ticker.fast_info)
        if rate is None:
            hist = ticker.history(period="5d")
            if hist is not None and not hist.empty:
                rate = _safe_float(hist["Close"].iloc[-1])
        return rate
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Snapshot aggregation (deterministic)
# ---------------------------------------------------------------------------

def _pence_adjust(price: Optional[float], currency: str) -> Tuple[Optional[float], str]:
    """Yahoo quotes LSE prices in pence (GBp) — normalise to GBP."""
    if price is not None and currency in {"GBp", "GBX"}:
        return price / 100.0, "GBP"
    return price, (currency or "").upper()


def build_portfolio_snapshot(
    holdings: Sequence[ReviewHolding],
    *,
    base_currency: str = "AUD",
    market_data_fn: Callable[[str], Dict[str, Any]] = fetch_holding_market_data,
    fx_fn: Callable[[str, str], Optional[float]] = fetch_fx_rate,
) -> PortfolioSnapshot:
    base_currency = (base_currency or "AUD").upper()
    snapshots: List[HoldingSnapshot] = []
    fx_cache: Dict[str, Optional[float]] = {}

    for holding in holdings:
        snap = HoldingSnapshot(label=holding.label, name=holding.label, units=holding.units)
        data: Dict[str, Any] = {}
        if holding.is_ticker:
            try:
                data = market_data_fn(holding.label) or {}
            except Exception as exc:
                data = {"error": f"fetch:{type(exc).__name__}"}
        snap.name = str(data.get("name") or holding.label)
        snap.quote_type = str(data.get("quote_type") or "")
        snap.sector = str(data.get("sector") or "")
        snap.industry = str(data.get("industry") or "")
        snap.country = str(data.get("country") or "")
        snap.category = str(data.get("category") or "")
        snap.sector_weightings = dict(data.get("sector_weightings") or {})
        snap.top_holdings = list(data.get("top_holdings") or [])
        snap.error = str(data.get("error") or "")
        price, currency = _pence_adjust(_safe_float(data.get("price")), str(data.get("currency") or ""))
        snap.price = price
        snap.currency = currency or base_currency

        if holding.value is not None:
            snap.value = holding.value
            snap.value_source = "user"
            snap.currency = base_currency
        elif holding.units is not None and price is not None:
            if snap.currency not in fx_cache:
                fx_cache[snap.currency] = fx_fn(snap.currency, base_currency)
            rate = fx_cache[snap.currency]
            if rate is not None:
                snap.value = holding.units * price * rate
                snap.value_source = "market"
            else:
                snap.error = snap.error or f"fx:{snap.currency}->{base_currency}"
        snapshots.append(snap)

    total = sum(s.value for s in snapshots if s.value is not None)
    unpriced = [s.label for s in snapshots if s.value is None]

    sector_exposure: Dict[str, float] = {}
    asset_type: Dict[str, float] = {}
    weights: List[float] = []
    if total > 0:
        for s in snapshots:
            if s.value is None:
                continue
            s.weight_pct = s.value / total * 100.0
            weights.append(s.weight_pct)
            kind = _asset_kind(s)
            asset_type[kind] = asset_type.get(kind, 0.0) + s.weight_pct
            if s.sector_weightings:
                covered = sum(s.sector_weightings.values())
                for sector, frac in s.sector_weightings.items():
                    key = _pretty_sector(sector)
                    sector_exposure[key] = sector_exposure.get(key, 0.0) + s.weight_pct * frac
                if covered < 0.98:
                    sector_exposure["Unclassified"] = sector_exposure.get("Unclassified", 0.0) + s.weight_pct * max(0.0, 1 - covered)
            elif s.sector:
                sector_exposure[s.sector] = sector_exposure.get(s.sector, 0.0) + s.weight_pct
            else:
                sector_exposure["Unclassified"] = sector_exposure.get("Unclassified", 0.0) + s.weight_pct

    ordered = sorted(weights, reverse=True)
    hhi = sum((w / 100.0) ** 2 for w in weights)
    return PortfolioSnapshot(
        base_currency=base_currency,
        holdings=snapshots,
        total_value=total,
        unpriced=unpriced,
        sector_exposure_pct=dict(sorted(sector_exposure.items(), key=lambda kv: kv[1], reverse=True)),
        asset_type_pct=dict(sorted(asset_type.items(), key=lambda kv: kv[1], reverse=True)),
        top1_pct=ordered[0] if ordered else None,
        top3_pct=sum(ordered[:3]) if ordered else None,
        effective_positions=(1.0 / hhi) if hhi > 0 else None,
    )


def _asset_kind(s: HoldingSnapshot) -> str:
    qt = (s.quote_type or "").upper()
    if qt == "ETF":
        return "ETF"
    if qt == "MUTUALFUND":
        return "Managed fund"
    if qt == "EQUITY":
        return "Single stock"
    if s.value_source == "user" and not s.quote_type:
        return "Other (user-valued)"
    return "Other"


def _pretty_sector(key: str) -> str:
    return key.replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# Rendering + LLM review
# ---------------------------------------------------------------------------

def _fmt_money(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:,.0f}"


def _fmt_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.1f}%"


def render_snapshot_markdown(snapshot: PortfolioSnapshot) -> str:
    """Deterministic snapshot section (numbers come from code, not the LLM)."""
    cur = snapshot.base_currency
    lines = [
        "## Portfolio snapshot",
        "",
        f"Total value (priced holdings): **{cur} {_fmt_money(snapshot.total_value)}**",
        "",
        f"| Holding | Name | Type | Sector | Value ({cur}) | Weight |",
        "|---|---|---|---|---:|---:|",
    ]
    for s in sorted(snapshot.holdings, key=lambda h: h.weight_pct or -1, reverse=True):
        sector = s.sector or ("Multi-sector (fund)" if s.sector_weightings else (s.category or "—"))
        lines.append(
            f"| {s.label} | {s.name} | {_asset_kind(s)} | {sector} | {_fmt_money(s.value)} | {_fmt_pct(s.weight_pct)} |"
        )
    lines.append("")
    lines.append(
        f"Concentration: largest holding {_fmt_pct(snapshot.top1_pct)}, top 3 {_fmt_pct(snapshot.top3_pct)}, "
        + (
            f"effective number of positions {snapshot.effective_positions:.1f}."
            if snapshot.effective_positions
            else "effective number of positions N/A."
        )
    )
    if snapshot.sector_exposure_pct:
        lines.append("")
        lines.append("Sector exposure (ETF look-through where Yahoo provides it):")
        for sector, pct in list(snapshot.sector_exposure_pct.items())[:12]:
            lines.append(f"- {sector}: {pct:.1f}%")
    if snapshot.unpriced:
        lines.append("")
        lines.append(
            "⚠️ Could not price: " + ", ".join(snapshot.unpriced)
            + " — excluded from weights. Use CODE:$value for anything Yahoo cannot price."
        )
    return "\n".join(lines)


def _snapshot_for_prompt(snapshot: PortfolioSnapshot) -> str:
    rows = []
    for s in sorted(snapshot.holdings, key=lambda h: h.weight_pct or -1, reverse=True):
        parts = [
            f"{s.label}",
            f"name={s.name}",
            f"type={_asset_kind(s)}",
            f"weight={_fmt_pct(s.weight_pct)}",
        ]
        for key in ("sector", "industry", "country", "category"):
            val = getattr(s, key)
            if val:
                parts.append(f"{key}={val}")
        if s.sector_weightings:
            top_sectors = sorted(s.sector_weightings.items(), key=lambda kv: kv[1], reverse=True)[:5]
            parts.append("fund_sectors=" + ", ".join(f"{_pretty_sector(k)} {v * 100:.0f}%" for k, v in top_sectors))
        if s.top_holdings:
            parts.append("fund_top_holdings=" + ", ".join(f"{n} {w * 100:.1f}%" for n, w in s.top_holdings[:5]))
        rows.append("- " + "; ".join(parts))
    sectors = ", ".join(f"{k} {v:.1f}%" for k, v in snapshot.sector_exposure_pct.items())
    types = ", ".join(f"{k} {v:.1f}%" for k, v in snapshot.asset_type_pct.items())
    return (
        "\n".join(rows)
        + f"\nAsset types: {types or 'N/A'}"
        + f"\nSector exposure: {sectors or 'N/A'}"
        + f"\nLargest holding {_fmt_pct(snapshot.top1_pct)}, top 3 {_fmt_pct(snapshot.top3_pct)}"
        + (f"\nUnpriced (excluded from weights): {', '.join(snapshot.unpriced)}" if snapshot.unpriced else "")
    )


def _results_for_prompt(results: Sequence[Any]) -> str:
    lines = []
    for r in results or []:
        dashboard = getattr(r, "dashboard", None) or {}
        core = dashboard.get("core_conclusion", {}) if isinstance(dashboard, dict) else {}
        one = (core.get("one_sentence") if isinstance(core, dict) else None) or getattr(r, "analysis_summary", "") or ""
        lines.append(
            f"- {getattr(r, 'code', '?')} ({getattr(r, 'name', '')}): advice={getattr(r, 'operation_advice', '')}, "
            f"score={getattr(r, 'sentiment_score', '')}, summary={str(one)[:300]}"
        )
    return "\n".join(lines) or "(no individual stock reviews this run)"


_LANGUAGE_NAMES = {"zh": "Simplified Chinese", "en": "English", "ko": "Korean"}


def build_review_prompt(snapshot: PortfolioSnapshot, results: Sequence[Any], report_language: str = "en") -> str:
    language = _LANGUAGE_NAMES.get(report_language, "English")
    return f"""You are a long-term, business-owner style investor (1-10 year horizon) reviewing a private
investor's whole portfolio. Base currency: {snapshot.base_currency}. Write the review in {language}.

PORTFOLIO DATA (computed by code from live prices; treat these numbers as facts and do not
recompute or alter them):
{_snapshot_for_prompt(snapshot)}

THIS MONTH'S INDIVIDUAL STOCK REVIEWS (from the same run; may cover only some holdings):
{_results_for_prompt(results)}

Rules:
- Use only the numbers above for weights and values. If sector or geography data is missing for
  a fund, you may infer its exposure from its name and well-known index methodology, but label
  each such statement "(inferred)".
- Identify overlap between funds and single stocks (e.g. a bank held directly and inside an
  Australian equity ETF).
- Do not invent prices, returns, fees or holdings. Fund fees and franking are not in the data;
  if you mention them, mark them "(background, unverified)".
- Be specific and concise; no generic filler.

Write exactly these markdown sections (level-2 headings, translated into {language}):
## Strengths
## Weaknesses and risks
(cover concentration, home bias / geography, sector gaps and overweights, single-stock risk,
income vs growth tilt, currency exposure)
## Overlap check
## What to add to first
(rank the existing holdings to add to next, using this month's stock reviews where available)
## Sectors and themes to consider for long-term growth
(2-4 gaps relative to a diversified global portfolio; for each give the reason and one or two
example ASX-listed ETFs or companies, each marked "verify before buying")
## What to watch before next month

End with one line: "General information only, not financial advice."
"""


def run_portfolio_review(
    raw_holdings: str,
    *,
    analyzer: Any,
    results: Sequence[Any] = (),
    base_currency: str = "AUD",
    report_language: str = "en",
    market_data_fn: Callable[[str], Dict[str, Any]] = fetch_holding_market_data,
    fx_fn: Callable[[str, str], Optional[float]] = fetch_fx_rate,
) -> Optional[str]:
    """Build the full portfolio review markdown, or None when nothing usable."""
    holdings, warnings = parse_portfolio_review_holdings(raw_holdings)
    for warning in warnings:
        logger.warning("[PortfolioReview] %s", warning)
    if not holdings:
        logger.warning("[PortfolioReview] PORTFOLIO_REVIEW_HOLDINGS has no valid entries; skipped")
        return None

    logger.info("[PortfolioReview] reviewing %d holdings", len(holdings))
    snapshot = build_portfolio_snapshot(
        holdings,
        base_currency=base_currency,
        market_data_fn=market_data_fn,
        fx_fn=fx_fn,
    )
    if snapshot.total_value <= 0:
        logger.warning("[PortfolioReview] no holding could be priced; skipped")
        return None

    parts = ["# 🧭 Portfolio review", "", render_snapshot_markdown(snapshot)]
    if warnings:
        parts.extend(["", "⚠️ Some PORTFOLIO_REVIEW_HOLDINGS entries were skipped: " + "; ".join(warnings)])

    review_text: Optional[str] = None
    if analyzer is not None and getattr(analyzer, "is_available", lambda: True)():
        try:
            review_text = analyzer.generate_text(
                build_review_prompt(snapshot, results, report_language),
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as exc:
            logger.error("[PortfolioReview] LLM review failed: %s", type(exc).__name__)
            review_text = None
    if review_text:
        parts.extend(["", str(review_text).strip()])
    else:
        parts.extend(["", "_The AI review could not be generated this run; the snapshot above is still current._"])
    return "\n".join(parts)
