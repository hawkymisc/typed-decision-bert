"""Configuration loading (spec 17.2; POC_DESIGN 4.3). Startup is fail-fast."""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest
import yaml

from jevbert.config import (
    API_KEYS_ENV,
    MIN_API_KEY_LENGTH,
    ConfigurationError,
    Limits,
    Settings,
    load_api_keys,
    load_settings,
    parse_api_keys,
    validate_api_keys,
)

BASE_CONFIG: dict[str, object] = {
    "project": "JevBERT",
    "manifests_dir": "../manifests",
    "models_dir": "../models",
    "bundles": [],
    "enable_fake_bundle": True,
}
KEY_ONE = "key-one-" + "a" * 32
KEY_TWO = "key-two-" + "b" * 32
ENV = {API_KEYS_ENV: f"{KEY_ONE},{KEY_TWO}"}


def write_config(tmp_path: Path, **overrides: object) -> Path:
    data = dict(BASE_CONFIG)
    data.update(overrides)
    directory = tmp_path / "configs"
    directory.mkdir(exist_ok=True)
    path = directory / "jevbert.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestApiKeys:
    def test_comma_separated_keys_are_split_and_stripped(self) -> None:
        assert parse_api_keys(" a , b ,, c ") == ("a", "b", "c")

    def test_empty_value_yields_no_keys(self) -> None:
        assert parse_api_keys("") == ()
        assert parse_api_keys("  ,  ") == ()

    def test_keys_come_from_the_environment(self) -> None:
        assert load_api_keys(ENV) == (KEY_ONE, KEY_TWO)


class TestApiKeyStrength:
    """S-L4: a guessable key is a configuration error, not a warning."""

    def test_the_minimum_length_is_at_least_thirty_two(self) -> None:
        assert MIN_API_KEY_LENGTH >= 32

    def test_a_key_at_the_minimum_length_is_accepted(self) -> None:
        validate_api_keys(("k" * MIN_API_KEY_LENGTH,))

    def test_a_key_one_character_short_refuses_startup(self) -> None:
        with pytest.raises(ConfigurationError):
            validate_api_keys(("k" * (MIN_API_KEY_LENGTH - 1),))

    def test_a_short_key_refuses_startup_through_the_config_loader(
        self, tmp_path: Path
    ) -> None:
        path = write_config(tmp_path)
        with pytest.raises(ConfigurationError):
            load_settings(path, env={API_KEYS_ENV: "short"})

    def test_a_long_key_beside_a_short_one_does_not_excuse_it(self, tmp_path: Path) -> None:
        path = write_config(tmp_path)
        with pytest.raises(ConfigurationError):
            load_settings(path, env={API_KEYS_ENV: f"{KEY_ONE},short"})

    def test_the_key_generated_by_init_env_satisfies_the_minimum(self) -> None:
        # The generator and the gate must not drift apart (see tests/unit/test_cli.py).
        assert len(secrets.token_urlsafe(32)) >= MIN_API_KEY_LENGTH


class TestApiKeysNeverAppearInDiagnostics:
    """S-L3: a key must survive neither ``repr`` nor an error message."""

    def test_repr_hides_the_keys(self) -> None:
        settings = Settings(api_keys=(KEY_ONE, KEY_TWO))
        assert KEY_ONE not in repr(settings)
        assert KEY_TWO not in repr(settings)
        assert KEY_ONE not in str(settings)

    def test_the_model_dump_of_a_setting_dict_is_not_a_backdoor(self) -> None:
        # Any diagnostic that prints the settings must stay safe, so the guard is on
        # ``repr`` of the object rather than on one call site.
        settings = Settings(api_keys=(KEY_ONE,))
        assert KEY_ONE not in f"{settings!r}"
        assert KEY_ONE not in f"the configuration is {settings}"

    @pytest.mark.parametrize("bad", ["sekrit", "sekrit-but-still-far-too-short"])
    def test_a_rejected_key_is_not_quoted_back(self, bad: str) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            validate_api_keys((bad,))
        assert bad not in str(excinfo.value)

    def test_a_rejected_short_key_is_not_quoted_back_by_the_loader(
        self, tmp_path: Path
    ) -> None:
        path = write_config(tmp_path)
        secret = "sekrit-but-too-short"
        with pytest.raises(ConfigurationError) as excinfo:
            load_settings(path, env={API_KEYS_ENV: secret})
        assert secret not in str(excinfo.value)

    def test_keys_in_the_configuration_file_are_refused(self, tmp_path: Path) -> None:
        # Keys belong in the environment; a YAML file is far more likely to be shared.
        path = write_config(tmp_path, api_keys=[KEY_ONE])
        with pytest.raises(ConfigurationError) as excinfo:
            load_settings(path, env=ENV)
        assert KEY_ONE not in str(excinfo.value)


class TestStartupPreconditions:
    def test_missing_api_key_refuses_startup(self, tmp_path: Path) -> None:
        # POC_DESIGN 4.3: there is no unauthenticated mode.
        path = write_config(tmp_path)
        with pytest.raises(ConfigurationError, match=API_KEYS_ENV):
            load_settings(path, env={})

    def test_blank_api_key_refuses_startup(self, tmp_path: Path) -> None:
        path = write_config(tmp_path)
        with pytest.raises(ConfigurationError):
            load_settings(path, env={API_KEYS_ENV: "   "})

    def test_valid_config_loads(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path), env=ENV)
        assert settings.api_keys == (KEY_ONE, KEY_TWO)
        assert settings.enable_fake_bundle is True

    def test_relative_paths_resolve_against_the_config_file(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path), env=ENV)
        assert settings.manifests_dir == (tmp_path / "manifests").resolve()
        assert settings.models_dir == (tmp_path / "models").resolve()

    def test_unknown_key_refuses_startup(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, unknown_option=1)
        with pytest.raises(ConfigurationError):
            load_settings(path, env=ENV)

    def test_aliases_without_the_flag_refuse_startup(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path, serving={"allow_jev_aliases": False, "aliases": {"jev-latest": "x"}}
        )
        with pytest.raises(ConfigurationError, match="allow_jev_aliases"):
            load_settings(path, env=ENV)

    def test_sequence_limit_above_the_request_limit_refuses_startup(
        self, tmp_path: Path
    ) -> None:
        path = write_config(
            tmp_path, limits={"max_sequence_tokens": 4096, "max_request_tokens": 2048}
        )
        with pytest.raises(ConfigurationError):
            load_settings(path, env=ENV)

    def test_unimplemented_features_cannot_be_switched_on(self, tmp_path: Path) -> None:
        # Silently ignoring these would misrepresent what the server does.
        unimplemented = (
            "result_cache_enabled",
            "allow_remote_model_download",
            "raw_request_logging",
        )
        for option in unimplemented:
            path = write_config(tmp_path, serving={option: True})
            with pytest.raises(ConfigurationError):
                load_settings(path, env=ENV)

    def test_missing_file_is_a_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            load_settings(tmp_path / "absent.yaml", env=ENV)

    def test_invalid_yaml_is_a_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("key: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_settings(path, env=ENV)

    def test_non_mapping_yaml_is_a_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_settings(path, env=ENV)


class TestAliasResolution:
    def test_aliases_are_empty_when_disabled(self) -> None:
        settings = Settings(api_keys=("k",))
        assert settings.resolved_aliases() == {}

    def test_aliases_are_returned_when_enabled(self) -> None:
        settings = Settings(
            api_keys=("k",),
            serving={"allow_jev_aliases": True, "aliases": {"jev-latest": "bundle"}},
        )
        assert settings.resolved_aliases() == {"jev-latest": "bundle"}


class TestLimits:
    def test_defaults_match_the_specification(self) -> None:
        limits = Limits()
        assert limits.max_body_bytes == 2_097_152
        assert limits.max_json_depth == 32
        assert limits.max_questions == 256
        assert (limits.min_choice_options, limits.max_choice_options) == (2, 255)
        assert (limits.min_score_levels, limits.max_score_levels) == (2, 10)
        assert limits.overflow_policy == "reject"

    def test_overflow_policy_cannot_be_changed(self) -> None:
        # spec 6.3: v0.1 fixes overflow_policy to reject.
        with pytest.raises(ValueError):
            Limits(overflow_policy="truncate")

    def test_choice_option_ceiling_cannot_exceed_the_contract(self) -> None:
        with pytest.raises(ValueError):
            Limits(max_choice_options=256)
