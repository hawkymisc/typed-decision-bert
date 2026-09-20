"""Configuration loading and validation (spec 17.2; POC_DESIGN 4.3).

Loading is fail-fast: an unknown key, an out-of-range limit or a missing API key stops
the process before the server binds a port. There is no unauthenticated mode.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from jevbert import CONTRACT_PROFILE

#: Environment variable holding the comma separated Bearer keys.
API_KEYS_ENV = "JEVBERT_API_KEYS"


class ConfigurationError(Exception):
    """Raised when the server must refuse to start."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Limits(_Strict):
    """Structural limits (spec 4.2). Token limits may be narrowed per bundle."""

    max_body_bytes: int = Field(default=2_097_152, gt=0)
    max_json_depth: int = Field(default=32, gt=0)
    max_questions: int = Field(default=32, gt=0)
    min_choice_options: int = Field(default=2, ge=2)
    max_choice_options: int = Field(default=255, le=255)
    min_score_levels: int = Field(default=2, ge=2)
    max_score_levels: int = Field(default=10, le=10)
    max_sequence_tokens: int = Field(default=2048, gt=0)
    max_request_tokens: int = Field(default=32_768, gt=0)
    overflow_policy: Literal["reject"] = "reject"


class ServingSettings(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, gt=0, lt=65536)
    max_pending_requests: int = Field(default=8, gt=0)
    request_deadline_seconds: float = Field(default=30.0, gt=0)
    max_batch_tokens: int = Field(default=16_384, gt=0)
    max_batch_sequences: int = Field(default=64, gt=0)
    # The PoC does not implement these; they exist so that a config that switches them
    # on is refused rather than silently ignored (spec 12.4, 15.2, 18.3).
    result_cache_enabled: Literal[False] = False
    allow_remote_model_download: Literal[False] = False
    raw_request_logging: Literal[False] = False
    allow_jev_aliases: bool = False
    aliases: dict[str, str] = Field(default_factory=dict)


class Settings(_Strict):
    project: str = "JevBERT"
    contract_profile: str = CONTRACT_PROFILE
    manifests_dir: Path = Path("manifests")
    models_dir: Path = Path("models")
    #: Manifest file names (relative to ``manifests_dir``) to register at startup.
    bundles: tuple[str, ...] = ()
    #: The fake backend never registers itself implicitly (POC_DESIGN 6.2).
    enable_fake_bundle: bool = False
    limits: Limits = Field(default_factory=Limits)
    serving: ServingSettings = Field(default_factory=ServingSettings)
    api_keys: tuple[str, ...] = ()

    @field_validator("api_keys")
    @classmethod
    def _keys_are_not_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not key.strip() for key in value):
            raise ValueError("API keys must not be blank")
        return value

    def resolved_aliases(self) -> dict[str, str]:
        """Aliases are only honoured when explicitly enabled (spec 18.3)."""
        if not self.serving.allow_jev_aliases:
            return {}
        return dict(self.serving.aliases)


class _ApiKeyEnv(BaseSettings):
    """Reads ``JEVBERT_API_KEYS`` from the environment or from ``.env``."""

    model_config = SettingsConfigDict(env_prefix="JEVBERT_", env_file=".env", extra="ignore")

    api_keys: str = ""


def parse_api_keys(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_api_keys(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Read the configured Bearer keys, preferring an explicit mapping over ``.env``."""
    if env is not None:
        return parse_api_keys(env.get(API_KEYS_ENV, ""))
    return parse_api_keys(_ApiKeyEnv().api_keys)


def load_settings(
    config_path: str | os.PathLike[str], env: Mapping[str, str] | None = None
) -> Settings:
    """Load a YAML config, resolving relative paths against the config file's directory."""
    path = Path(config_path).resolve()
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Cannot read the configuration file {path}") from exc

    try:
        data: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"{path} is not valid YAML") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must contain a YAML mapping")

    data["api_keys"] = load_api_keys(env)
    try:
        settings = Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"{path} is not a valid JevBERT configuration:\n{exc}") from exc

    base = path.parent
    settings = settings.model_copy(
        update={
            "manifests_dir": _resolve(base, settings.manifests_dir),
            "models_dir": _resolve(base, settings.models_dir),
        }
    )
    _check_startup_preconditions(settings)
    return settings


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base / value).resolve()


def _check_startup_preconditions(settings: Settings) -> None:
    if not settings.api_keys:
        raise ConfigurationError(
            f"No API key configured. Set {API_KEYS_ENV} (comma separated) or run "
            "`python -m jevbert init-env`. The server has no unauthenticated mode."
        )
    if settings.limits.max_sequence_tokens > settings.limits.max_request_tokens:
        raise ConfigurationError(
            "limits.max_sequence_tokens must not exceed limits.max_request_tokens"
        )
    if settings.serving.aliases and not settings.serving.allow_jev_aliases:
        raise ConfigurationError(
            "serving.aliases is set but serving.allow_jev_aliases is false; "
            "enable it explicitly or remove the alias table"
        )
