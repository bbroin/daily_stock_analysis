# -*- coding: utf-8 -*-
"""Built-in `long_term_investor` strategy: loading, prompt baseline and workflow wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from src.agent.factory import resolve_skill_prompt_state
from src.agent.skills.base import load_skill_from_yaml
from src.report_language import localize_strategy_skill

ROOT_DIR = Path(__file__).resolve().parent.parent
SKILL_PATH = ROOT_DIR / "strategies" / "long_term_investor.yaml"
WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "00-daily-analysis.yml"


def _config(**overrides):
    return SimpleNamespace(**{"agent_skill_dir": None, "agent_skills": None, **overrides})


def test_skill_file_loads_and_is_opt_in():
    skill = load_skill_from_yaml(SKILL_PATH)
    assert skill.name == "long_term_investor"
    assert skill.category == "framework"
    assert skill.default_active is False
    assert skill.default_router is False
    for tool in ("get_stock_info", "get_daily_history", "calculate_ma", "search_stock_news"):
        assert tool in skill.required_tools
    assert skill.allowed_tools == []  # must not restrict the agent's tool surface


def test_default_selection_is_unchanged():
    state = resolve_skill_prompt_state(_config())
    assert state.skills_to_activate == ["bull_trend"]
    assert state.default_skill_policy  # short-term baseline still applies by default


def test_explicit_selection_replaces_short_term_baseline():
    state = resolve_skill_prompt_state(_config(agent_skills=["long_term_investor"]))
    assert state.skills_to_activate == ["long_term_investor"]
    assert state.default_skill_policy == ""
    assert state.technical_skill_policy == ""
    assert state.use_legacy_default_prompt is False
    assert "Moat and quality" in state.skill_instructions
    assert "Thesis invalidation" in state.skill_instructions
    assert "MA5 > MA10" not in state.skill_instructions


def test_localized_label():
    assert localize_strategy_skill("long_term_investor", "en") == "Long-Term Investor"
    assert localize_strategy_skill("long_term_investor", "zh") == "长期投资"


def test_workflow_passes_strategy_and_news_settings():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["analyze"]["steps"]
    env = next(step for step in steps if step.get("name") == "执行股票分析")["env"]
    assert "vars.AGENT_SKILLS" in env["AGENT_SKILLS"]
    assert "vars.NEWS_STRATEGY_PROFILE" in env["NEWS_STRATEGY_PROFILE"]
    assert "vars.NEWS_MAX_AGE_DAYS" in env["NEWS_MAX_AGE_DAYS"]
    # Unset variables keep the previous defaults.
    assert env["AGENT_SKILLS"].endswith("|| '' }}")
    assert env["NEWS_STRATEGY_PROFILE"].endswith("|| 'short' }}")
    assert env["NEWS_MAX_AGE_DAYS"].endswith("|| '3' }}")
