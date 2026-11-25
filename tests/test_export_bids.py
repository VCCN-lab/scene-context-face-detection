"""Tests on how export_bids.py codes the participant answers."""

import pytest

import export_bids as X


@pytest.mark.parametrize("answer,coded", [
    ("y", "yes"), ("yes", "yes"), ("Yes", "yes"), (" YES ", "yes"),
    ("n", "no"), ("no", "no"), ("No", "no"), ("", "n/a")])
def test_yes_no(answer, coded):
    assert X.yes_no(answer) == coded


@pytest.mark.parametrize("answer", ["c", "maybe", "glasses"])
def test_yes_no_rejects_other_answers(answer):
    with pytest.raises(ValueError):
        X.yes_no(answer)
