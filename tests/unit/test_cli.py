"""``python -m jevbert`` (POC_DESIGN 4.3, 10; S-L5).

``init-env`` writes a credential. It must create the file exclusively, never clobber an
existing one, and report what it did through its exit code.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from jevbert.__main__ import main
from jevbert.config import API_KEYS_ENV, MIN_API_KEY_LENGTH, parse_api_keys


class TestInitEnv:
    def test_writes_a_usable_key_and_reports_success(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        assert main(["init-env", "--path", str(path)]) == 0

        content = path.read_text(encoding="utf-8")
        assert content.startswith("#")
        line = next(line for line in content.splitlines() if line.startswith(API_KEYS_ENV))
        keys = parse_api_keys(line.split("=", 1)[1])
        assert len(keys) == 1
        assert len(keys[0]) >= MIN_API_KEY_LENGTH

    def test_an_existing_file_is_never_overwritten(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("JEVBERT_API_KEYS=the-operators-own-key\n", encoding="utf-8")

        assert main(["init-env", "--path", str(path)]) == 1
        assert path.read_text(encoding="utf-8") == "JEVBERT_API_KEYS=the-operators-own-key\n"

    def test_an_empty_existing_file_is_never_overwritten(self, tmp_path: Path) -> None:
        # ``exists()`` and ``open("w")`` race; exclusive creation is what actually holds.
        path = tmp_path / ".env"
        path.touch()
        assert main(["init-env", "--path", str(path)]) == 1
        assert path.read_text(encoding="utf-8") == ""

    def test_two_runs_in_a_row_produce_one_file_and_one_key(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        assert main(["init-env", "--path", str(path)]) == 0
        first = path.read_text(encoding="utf-8")
        assert main(["init-env", "--path", str(path)]) == 1
        assert path.read_text(encoding="utf-8") == first

    def test_successive_keys_differ(self, tmp_path: Path) -> None:
        keys = []
        for index in range(2):
            path = tmp_path / f"{index}.env"
            assert main(["init-env", "--path", str(path)]) == 0
            keys.append(path.read_text(encoding="utf-8"))
        assert keys[0] != keys[1]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits only")
    def test_the_file_is_not_world_readable(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        assert main(["init-env", "--path", str(path)]) == 0
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode & (stat.S_IRWXG | stat.S_IRWXO) == 0


class TestOtherCommands:
    def test_fetch_model_reports_that_it_is_unimplemented(self) -> None:
        assert main(["fetch-model"]) == 3

    def test_serve_with_an_unreadable_config_exits_with_two(self, tmp_path: Path) -> None:
        assert main(["serve", "--config", str(tmp_path / "absent.yaml")]) == 2

    def test_serve_without_a_key_exits_with_two(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Run from an empty directory so that neither the environment nor a developer's
        # own ``.env`` can supply a key and let uvicorn actually start.
        monkeypatch.delenv(API_KEYS_ENV, raising=False)
        monkeypatch.chdir(tmp_path)
        config = tmp_path / "jevbert.yaml"
        config.write_text("enable_fake_bundle: true\n", encoding="utf-8")
        assert main(["serve", "--config", str(config)]) == 2

    def test_an_unknown_command_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            main(["nonsense"])
