from newsletter_agent.cli import _parser


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
