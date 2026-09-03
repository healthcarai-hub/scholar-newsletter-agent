from __future__ import annotations

from copy import deepcopy

import pytest
import yaml


@pytest.fixture
def config_dict():
    return yaml.safe_load(
        """
runtime:
  timezone: Asia/Kolkata
  preview_directory: outputs/previews
profiles:
  migraine:
    enabled: true
    topic: migraine research
    gmail:
      label_name: Scholar Alerts/Migraine
    categories:
      - name: Mechanisms
        guidance: Biology and pathophysiology.
      - name: Treatments
        guidance: Trials and interventions.
    prompts:
      categorization: |
        For {topic}, choose one category from {categories}.
        Source: {source_text}
      item_summary: |
        Summarize this {topic} result: {source_text}
      issue_summary: |
        Summarize these {topic} results from {categories}: {items}
    newsletter:
      subject_template: "This Week in {topic} — {issue_date}"
      title_template: "Weekly {topic} Digest"
      recipients: [editor@example.com]
"""
    )


@pytest.fixture
def copied_config(config_dict):
    return deepcopy(config_dict)
