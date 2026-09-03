from __future__ import annotations

import pytest

from newsletter_agent.config import AppConfig


def test_topic_and_categories_are_validated_config_variables(config_dict):
    config = AppConfig.model_validate(config_dict)
    profile = config.profiles["migraine"]
    assert profile.topic == "migraine research"
    assert [category.name for category in profile.categories] == ["Mechanisms", "Treatments"]
    assert "Mechanisms: Biology" in profile.category_prompt()
    assert "Additional Research" in profile.allowed_categories


def test_unknown_prompt_placeholder_is_rejected(copied_config):
    copied_config["profiles"]["migraine"]["prompts"]["item_summary"] += " {secret}"
    with pytest.raises(ValueError, match="unsupported placeholders"):
        AppConfig.model_validate(copied_config)


def test_missing_topic_placeholder_is_rejected(copied_config):
    copied_config["profiles"]["migraine"]["prompts"]["categorization"] = (
        "Choose from {categories} for {source_text}"
    )
    with pytest.raises(ValueError, match="missing required placeholders"):
        AppConfig.model_validate(copied_config)


def test_duplicate_enabled_gmail_labels_are_rejected(copied_config):
    copied_config["profiles"]["second"] = dict(copied_config["profiles"]["migraine"])
    copied_config["profiles"]["second"]["topic"] = "another topic"
    with pytest.raises(ValueError, match="same Gmail label"):
        AppConfig.model_validate(copied_config)


def test_configured_gmail_label_id_is_rejected(copied_config):
    copied_config["profiles"]["migraine"]["gmail"]["label_id"] = "Label_123"
    with pytest.raises(ValueError, match="label_id"):
        AppConfig.model_validate(copied_config)


def test_duplicate_category_names_are_case_insensitive(copied_config):
    copied_config["profiles"]["migraine"]["categories"].append(
        {"name": "mechanisms", "guidance": "Duplicate."}
    )
    with pytest.raises(ValueError, match="category names must be unique"):
        AppConfig.model_validate(copied_config)
