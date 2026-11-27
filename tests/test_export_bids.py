"""Tests on how export_bids.py codes and aggregates the participant answers."""

import pandas as pd
import pytest

import export_bids as X


def person(pid, task, age, sex, handedness, vision):
    """One participant record as export_bids collects it."""
    return dict(participant_id=pid, task=task, age=age, sex=sex,
                handedness=handedness, corrected_vision=vision)


@pytest.fixture
def people():
    """Three face detection and two moving-window participants."""
    return [person("sub-001", "facedetection", 20, "f", "r", "no"),
            person("sub-002", "facedetection", 22, "m", "l", "yes"),
            person("sub-003", "facedetection", 27, "f", "r", "no"),
            person("sub-004", "movingwindow", 30, "f", "r", "yes"),
            person("sub-005", "movingwindow", "n/a", "m", "r", "n/a")]


@pytest.mark.parametrize("answer,coded", [
    ("y", "yes"), ("yes", "yes"), ("Yes", "yes"), (" YES ", "yes"),
    ("n", "no"), ("no", "no"), ("No", "no"), ("", "n/a")])
def test_yes_no(answer, coded):
    """Typed yes/no answers are coded as yes, no or n/a."""
    assert X.yes_no(answer) == coded


@pytest.mark.parametrize("answer", ["c", "maybe", "glasses"])
def test_yes_no_rejects_other_answers(answer):
    """Any other answer raises."""
    with pytest.raises(ValueError):
        X.yes_no(answer)


def test_participants_table_holds_only_id_and_task(people):
    """participants.tsv carries no individual demographics."""
    table = X.participants_table(people)
    assert list(table.columns) == ["participant_id", "task"]
    assert list(table["participant_id"]) == [p["participant_id"]
                                             for p in people]
    assert list(table["task"]) == [p["task"] for p in people]


def test_demographics_summary(people):
    """Per-task counts and age statistics; n/a ages are left out of the
    age statistics but counted in n."""
    summary = X.demographics_summary(people)
    expected = pd.DataFrame({
        "task": ["facedetection", "movingwindow"],
        "n": [3, 2],
        "age_mean": [23.0, 30.0], "age_sd": [3.6, float("nan")],
        "age_min": [20, 30], "age_max": [27, 30],
        "female": [2, 1], "male": [1, 1],
        "right_handed": [2, 2], "left_handed": [1, 0],
        "corrective_lenses": [1, 1]})
    pd.testing.assert_frame_equal(summary, expected, check_dtype=False)


@pytest.mark.parametrize("field,value", [
    ("sex", "x"), ("handedness", "a"), ("corrected_vision", "maybe")])
def test_demographics_summary_rejects_unknown_codes(people, field, value):
    """An unexpected code raises, so no participant goes uncounted."""
    people[0][field] = value
    with pytest.raises(ValueError):
        X.demographics_summary(people)
