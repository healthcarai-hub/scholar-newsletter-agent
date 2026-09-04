from newsletter_agent.cli import _parser, _rag_indexing_requested
from newsletter_agent.config import AppConfig


def test_check_gmail_command_accepts_optional_profiles() -> None:
    args = _parser().parse_args(
        ["check-gmail", "--profile", "migraine", "--profile", "cardiology"]
    )

    assert args.command == "check-gmail"
    assert args.profiles == ["migraine", "cardiology"]


def test_run_command_accepts_keep_labels() -> None:
    args = _parser().parse_args(
        ["run", "--profile", "migraine", "--keep-labels", "--lookback-days", "10"]
    )

    assert args.command == "run"
    assert args.profiles == ["migraine"]
    assert args.keep_labels is True
    assert args.lookback_days == 10


def test_embedding_provider_is_needed_only_for_selected_indexed_profiles(config_dict) -> None:
    config_dict["profiles"]["migraine"]["rag_indexing_enabled"] = False
    config_dict["profiles"]["indexed"] = {
        **config_dict["profiles"]["migraine"],
        "gmail": {"label_name": "Scholar Alerts/Indexed"},
        "rag_indexing_enabled": True,
    }
    config = AppConfig.model_validate(config_dict)

    assert _rag_indexing_requested(config, ["migraine"]) is False
    assert _rag_indexing_requested(config, ["indexed"]) is True
    assert _rag_indexing_requested(config) is True
