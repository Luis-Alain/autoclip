"""Tests for the fixes that make a local test run possible on a Mac."""

from __future__ import annotations

from autoclip import config
from autoclip.cli import app
from autoclip.providers.ollama_provider import _CTX_MAX, _CTX_MIN, context_size
from typer.testing import CliRunner

runner = CliRunner()


# -- Ollama num_ctx ---------------------------------------------------------


def test_context_size_has_a_floor_for_short_prompts() -> None:
    assert context_size("system", "short") == _CTX_MIN


def test_context_size_fits_an_eight_minute_window() -> None:
    # ~1200 words rendered as "[12345]palabra " is about 15 chars per word.
    user = "[12345]palabra " * 1200
    system = "x" * 5000
    size = context_size(system, user)
    estimated_tokens = (len(system) + len(user)) / 2.5
    assert size > estimated_tokens
    assert size % 4096 == 0


def test_context_size_is_capped() -> None:
    assert context_size("", "x" * 1_000_000) == _CTX_MAX


# -- autoclip config set ----------------------------------------------------


def test_config_set_string_and_nested_values() -> None:
    assert runner.invoke(app, ["config", "set", "active_provider", "ollama"]).exit_code == 0
    assert (
        runner.invoke(app, ["config", "set", "providers.ollama.model", "qwen2.5:7b"]).exit_code
        == 0
    )
    assert runner.invoke(app, ["config", "set", "whisper.language", "es"]).exit_code == 0

    settings = config.load()
    assert settings.active_provider == "ollama"
    assert settings.provider("ollama").model == "qwen2.5:7b"
    assert settings.whisper.language == "es"


def test_config_set_parses_numbers() -> None:
    assert runner.invoke(app, ["config", "set", "clips.max_clips", "5"]).exit_code == 0
    assert config.load().clips.max_clips == 5


def test_config_set_rejects_unknown_keys() -> None:
    result = runner.invoke(app, ["config", "set", "whisper.nope", "1"])
    assert result.exit_code == 2
    assert "Unknown setting" in result.output


def test_config_set_rejects_invalid_values() -> None:
    result = runner.invoke(app, ["config", "set", "active_provider", "not-a-provider"])
    assert result.exit_code == 2
    assert config.load().active_provider != "not-a-provider"
