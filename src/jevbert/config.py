"""Configuration loading and validation (spec 17.2; POC_DESIGN 4.3).

Loading is fail-fast: an unknown key, an out-of-range limit or a missing API key stops
the process before the server binds a port. There is no unauthenticated mode.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from jevbert import CONTRACT_PROFILE

#: Environment variable holding the comma separated Bearer keys.
API_KEYS_ENV = "JEVBERT_API_KEYS"

#: Shortest Bearer key the server will start with. ``init-env`` generates
#: ``secrets.token_urlsafe(32)``, which is 43 characters, so the generator is well
#: clear of the gate. A shorter key is a configuration error, not a warning (S-L4).
MIN_API_KEY_LENGTH = 32


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
    #: Cheap ceiling applied before the backend is asked to tokenize anything, so that
    #: a candidate/state amplification cannot buy minutes of tokenizer time with one
    #: small body (S-H2). ``None`` derives it from ``max_request_tokens``.
    max_request_chars: int | None = Field(default=None, gt=0)
    overflow_policy: Literal["reject"] = "reject"

    #: A token is at least this many characters' worth of budget in the worst case
    #: (one character per token); four is a deliberately generous multiplier so that
    #: the gate only ever fires on requests the token budget would refuse anyway.
    CHARS_PER_TOKEN: ClassVar[int] = 4

    @property
    def effective_max_request_chars(self) -> int:
        """The character ceiling actually enforced, configured or derived."""
        if self.max_request_chars is not None:
            return self.max_request_chars
        return self.CHARS_PER_TOKEN * self.max_request_tokens


class ServingSettings(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, gt=0, lt=65536)
    max_pending_requests: int = Field(default=8, gt=0)
    request_deadline_seconds: float = Field(default=30.0, gt=0)
    #: Reaches the backend through ``BackendContext`` (A-F5); the PoC's fake backend
    #: ignores all three, the phase 2 NLI backend uses them to cut microbatches.
    max_batch_tokens: int = Field(default=16_384, gt=0)
    max_batch_sequences: int = Field(default=64, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
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
    #: ``repr=False`` so that no diagnostic, traceback or log line that happens to
    #: render the settings can print a credential (S-L3, spec 15.2). The keys are
    #: checked by :func:`validate_api_keys` rather than by a pydantic validator,
    #: because pydantic quotes the offending input back in its error message.
    api_keys: tuple[str, ...] = Field(default=(), repr=False)

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


def validate_api_keys(keys: tuple[str, ...]) -> None:
    """Refuse to start on a missing or weak Bearer key.

    Raises ``ConfigurationError``. The message never contains a key, not even a
    rejected one: a message like "the key 'sekrit' is too short" is exactly the line
    that ends up in a terminal recording or a pasted issue (S-L3, spec 15.2).
    """
    if not keys:
        raise ConfigurationError(
            f"No API key configured. Set {API_KEYS_ENV} (comma separated) or run "
            "`python -m jevbert init-env`. The server has no unauthenticated mode."
        )
    weak = sum(1 for key in keys if len(key.strip()) < MIN_API_KEY_LENGTH)
    if weak:
        raise ConfigurationError(
            f"{weak} of the {len(keys)} configured API keys are shorter than "
            f"{MIN_API_KEY_LENGTH} characters. Generate one with "
            "`python -m jevbert init-env`."
        )


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
    if "api_keys" in data:
        raise ConfigurationError(
            f"{path} sets api_keys. Keys are read from {API_KEYS_ENV} only, so that a "
            "configuration file can be shared or committed without leaking one."
        )

    try:
        settings = Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"{path} is not a valid JevBERT configuration:\n{exc}") from exc

    # Applied after validation so that a key never passes through pydantic, whose
    # error messages quote the input value (S-L3).
    keys = load_api_keys(env)
    base = path.parent
    settings = settings.model_copy(
        update={
            "api_keys": keys,
            "manifests_dir": _resolve(base, settings.manifests_dir),
            "models_dir": _resolve(base, settings.models_dir),
        }
    )
    _check_startup_preconditions(settings)
    return settings


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base / value).resolve()


def _check_startup_preconditions(settings: Settings) -> None:
    validate_api_keys(settings.api_keys)
    if settings.limits.max_sequence_tokens > settings.limits.max_request_tokens:
        raise ConfigurationError(
            "limits.max_sequence_tokens must not exceed limits.max_request_tokens"
        )
    if settings.serving.aliases and not settings.serving.allow_jev_aliases:
        raise ConfigurationError(
            "serving.aliases is set but serving.allow_jev_aliases is false; "
            "enable it explicitly or remove the alias table"
        )
