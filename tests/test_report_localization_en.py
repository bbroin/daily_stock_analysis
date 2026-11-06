# -*- coding: utf-8 -*-
"""English report rendering must not leak Chinese units / enum values."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notification import NotificationService  # noqa: E402
from src.notification_sender.email_sender import EmailSender  # noqa: E402


def test_amounts_keep_chinese_units_for_zh():
    assert NotificationService._format_amount(1.5612e10, "AUD", "zh") == "156.12 亿澳元"
    assert NotificationService._format_per_share_localized(0.6, "AUD", "zh") == "0.6000 澳元"


def test_amounts_use_currency_prefix_for_en():
    assert NotificationService._format_amount(1.5612e10, "AUD", "en") == "A$15.61B"
    assert NotificationService._format_amount(-3.66e8, "AUD", "en") == "-A$366.00M"
    assert NotificationService._format_amount(1.2e12, "USD", "en") == "US$1.20T"
    assert NotificationService._format_amount(950, "EUR", "en") == "EUR 950"
    assert NotificationService._format_amount("x", "AUD", "en") == "N/A"
    assert NotificationService._format_per_share_localized(0.6, "AUD", "en") == "A$0.6000"


def test_cn_enum_values_translated_only_for_non_zh():
    assert NotificationService._localize_cn_enum("不急", "en") == "No rush"
    assert NotificationService._localize_cn_enum("盘后复盘", "en") == "After-market review"
    assert NotificationService._localize_cn_enum("无盘中动作", "en") == "No intraday action"
    assert NotificationService._localize_cn_enum("不急", "zh") == "不急"
    assert NotificationService._localize_cn_enum("Custom text", "en") == "Custom text"


def _sender(language, sender_name="daily_stock_analysis股票分析助手"):
    config = SimpleNamespace(
        email_sender="a@example.com", email_sender_name=sender_name, email_password="x",
        email_receivers=[], stock_email_groups=[], report_language=language,
    )
    return EmailSender(config)


def test_email_subject_and_sender_name_follow_report_language():
    zh = _sender("zh")
    assert zh._default_subject("# 🧭 Portfolio review").startswith("📈 股票智能分析报告 - ")
    assert zh._email_config["sender_name"] == "daily_stock_analysis股票分析助手"

    en = _sender("en")
    assert en._email_config["sender_name"] == "Stock Analysis"
    assert en._default_subject("# 🧭 Portfolio review\n\nbody").startswith("🧭 Portfolio review - ")
    assert en._default_subject("# 🎯 2026-11-06 Decision Dashboard").count("2026-11-06") <= 1
    assert en._default_subject("plain text").startswith("📈 Stock Analysis Report - ")
    # A user-configured sender name is never overridden.
    assert _sender("en", "My Reports")._email_config["sender_name"] == "My Reports"
