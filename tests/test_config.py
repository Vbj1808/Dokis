"""Tests for Config loading helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from dokis.config import Config


def test_config_from_yaml_raises_on_yaml_extension(tmp_path: Path) -> None:
    """from_yaml() must raise ValueError when passed a .yaml path."""
    f = tmp_path / "config.yaml"
    f.write_text("min_citation_rate: 0.85\n")
    with pytest.raises(ValueError, match="no longer supports YAML"):
        Config.from_yaml(f)


def test_config_from_yaml_loads_toml(tmp_path: Path) -> None:
    """from_yaml() must load a valid TOML file correctly."""
    f = tmp_path / "config.toml"
    f.write_bytes(b"min_citation_rate = 0.90\nclaim_threshold = 0.65\n")
    config = Config.from_yaml(f)
    assert config.min_citation_rate == pytest.approx(0.90)
    assert config.claim_threshold == pytest.approx(0.65)


def test_fail_on_violation_maps_to_enforce_mode() -> None:
    """Legacy fail_on_violation=True must still enable fail-closed behavior."""
    config = Config(fail_on_violation=True)
    assert config.enforcement_mode == "enforce"
    assert config.fail_on_violation is True


def test_config_defaults_to_guardrail_mode() -> None:
    """Zero-config usage should prefer the modern non-raising guardrail mode."""
    config = Config()
    assert config.enforcement_mode == "guardrail"
    assert config.fail_on_violation is False


def test_enforcement_mode_overrides_legacy_fail_on_violation() -> None:
    """Explicit enforcement_mode wins when both old and new settings exist."""
    config = Config(
        enforcement_mode="audit",
        fail_on_violation=True,
    )
    assert config.enforcement_mode == "audit"
    assert config.fail_on_violation is False
