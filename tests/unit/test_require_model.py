"""The ``model`` marker's skip, and the switch that turns it into a failure (Q-M2).

Without the weights, 59 tests skip and pytest still prints a green summary. A run whose
purpose is to verify the real model has to be able to say "the weights had better be
there", or a verification that silently measured nothing reads exactly like one that
measured everything.
"""

from __future__ import annotations

import pytest

from tests.conftest import REQUIRE_MODEL_ENV, _no_weights, model_tests_are_required


class TestTheStrictSwitch:
    @pytest.mark.parametrize("value", ["1", "yes", "true", "on"])
    def test_a_set_value_means_required(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(REQUIRE_MODEL_ENV, value)
        assert model_tests_are_required()

    @pytest.mark.parametrize("value", ["", "0", "false", "False", "   "])
    def test_an_empty_or_off_value_means_optional(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(REQUIRE_MODEL_ENV, value)
        assert not model_tests_are_required()

    def test_an_unset_variable_means_optional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(REQUIRE_MODEL_ENV, raising=False)
        assert not model_tests_are_required()


class TestWhatHappensWithoutWeights:
    def test_it_skips_when_the_switch_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(REQUIRE_MODEL_ENV, raising=False)
        with pytest.raises(pytest.skip.Exception) as raised:
            _no_weights("the weights are not there")
        assert "fetch-model" in str(raised.value)

    def test_it_fails_when_the_switch_is_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REQUIRE_MODEL_ENV, "1")
        with pytest.raises(pytest.fail.Exception) as raised:
            _no_weights("the weights are not there")
        assert REQUIRE_MODEL_ENV in str(raised.value)
        assert "fetch-model" in str(raised.value)

    def test_the_reason_is_always_carried_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A skip with an unstated reason reads like a pass; the command that fixes it
        # is part of the message in both modes.
        for value, error in (("", pytest.skip.Exception), ("1", pytest.fail.Exception)):
            monkeypatch.setenv(REQUIRE_MODEL_ENV, value)
            with pytest.raises(error) as raised:
                _no_weights("CANARY-REASON")
            assert "CANARY-REASON" in str(raised.value)
