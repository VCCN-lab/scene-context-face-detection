"""Unit tests for the trial reduction, the behavioural pooling and the figure
helpers, on synthetic events."""

import numpy as np
import pandas as pd
import pytest

from facescenes import behaviour as B
from facescenes import config as cfg
from facescenes import data as D
from facescenes import onset_gaze as G
from facescenes import plotting as P


# ---------------------------------------------------------- data.trials()
def _events():
    """Two trials for one participant.

    Trial 1: a carry-over fixation at 0.05 s (discarded), then two search
    fixations; the last one, at 0.70 s, is the detection.
    Trial 2: a timeout, which must be dropped.
    """
    def row(trial, onset, x, y, ttype, stim, timeout=0, phase="search"):
        return dict(trial=trial, onset=onset, duration=.2, phase=phase,
                    trial_type=ttype, stim_file=f"stimuli/{stim}",
                    gaze_x=x, gaze_y=y, timeout=timeout)
    return pd.DataFrame([
        row(1, -0.30, 0, 0, "preview", "b.jpg", phase="preview"),
        row(1, 0.05, 0, 0, "preview", "b.jpg"),
        row(1, 0.30, 3, 4, "preview", "b.jpg"),
        row(1, 0.70, 3, 10, "preview", "b.jpg"),
        row(2, 0.20, 0, 0, "no_preview", "a.jpg", timeout=1),
        row(2, 5.00, 9, 9, "no_preview", "a.jpg", timeout=1),
    ])


@pytest.fixture
def fake_data(monkeypatch):
    monkeypatch.setattr(D, "participant_ids", lambda task, apply_exclusions=True: ["sub-x"])
    monkeypatch.setattr(D, "stimulus_names", lambda: ("a.jpg", "b.jpg", "c.jpg"))
    monkeypatch.setattr(D, "events", lambda pid, task: _events())
    monkeypatch.setitem(cfg.EXCLUDE_STIMULI, "facedetection", [])


def test_trials_definition(fake_data):
    t = D.trials("facedetection")
    assert len(t) == 1                             # timeout dropped
    r = t.iloc[0]
    assert r.image == 1 and r.trial_type == "preview"
    assert r.latency == pytest.approx(0.70)        # onset of the LAST fixation
    assert r.n_fixations == 2                      # carry-over not counted
    # path runs from the carry-over position: 5 px then 6 px
    assert r.path_length == pytest.approx(11)


def test_trials_exclusion(fake_data, monkeypatch):
    monkeypatch.setitem(cfg.EXCLUDE_STIMULI, "facedetection", [1])
    assert D.trials("facedetection").empty
    assert len(D.trials("facedetection", apply_exclusions=False)) == 1


def test_latency_array(fake_data):
    a = D.latency_array("facedetection")
    assert a.shape == (1, 2, 3)
    assert a[0, 0, 1] == pytest.approx(.7)
    assert np.isnan(a).sum() == 5


# ------------------------------------------------------- participant pooling
def _table():
    return pd.DataFrame(dict(
        participant=["s1"] * 4 + ["s2"] * 3,
        trial_type=["preview", "preview", "no_preview", "no_preview",
                    "preview", "no_preview", "no_preview"],
        v=[1.0, 4.0, 2.0, 8.0, 3.0, 5.0, 7.0]))


def test_participant_means():
    a, b = B.participant_means(_table(), "v")
    assert np.allclose(a, [2.5, 3.0]) and np.allclose(b, [5.0, 6.0])


def test_participant_means_log_is_mean_of_logs():
    a, b = B.participant_means(_table(), "v", log=True)
    assert np.allclose(np.exp(a), [2.0, 3.0])      # geometric mean of 1 and 4
    assert np.allclose(np.exp(b), [4.0, np.sqrt(35)])


def test_participant_means_log_drops_non_positive():
    df = _table()
    df.loc[0, "v"] = 0.0
    a, _ = B.participant_means(df, "v", log=True)
    assert np.exp(a[0]) == pytest.approx(4.0)


def test_participant_table_missing_condition():
    df = _table()
    df = df[df["trial_type"] == "preview"]
    t = B.participant_table(df, "v")
    assert list(t.columns) == list(cfg.CONDITIONS)
    assert t["no_preview"].isna().all()


def test_participant_means_median():
    a, _ = B.participant_means(_table(), "v", agg="median")
    assert np.allclose(a, [2.5, 3.0])


# -------------------------------------------------------------- onset gaze
def test_image_coords_centre():
    x, y = G._to_image_coords(cfg.DISPLAY["width_px"] / 2, cfg.DISPLAY["height_px"] / 2)
    assert (x, y) == (cfg.DISPLAY["image_width_px"] / 2, cfg.DISPLAY["image_height_px"] / 2)


def test_within_participant_ols_recovers_effect():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(5):
        prev = rng.random(60) < .5
        d = rng.uniform(0, 5, 60)
        rows.append(pd.DataFrame(dict(
            participant=f"s{s}", trial_type=np.where(prev, "preview", "no_preview"),
            d_face_deg=d, latency_ms=300 + s * 50 - 20 * prev + 10 * d)))
    df = pd.concat(rows)
    assert np.allclose(G._within_participant_ols(df, [], 1)[0], -20, atol=15)
    assert np.allclose(G._within_participant_ols(df, ["d_face_deg"], 1), -20)
    assert np.allclose(G._within_participant_ols(df, ["d_face_deg"], 2), 10)
    # participants below min_trials are skipped
    assert len(G._within_participant_ols(df.iloc[:70], [], 1, min_trials=20)) == 1


# --------------------------------------------------------------- plotting
def test_significance_levels():
    assert [P.significance(p) for p in (.0005, .005, .02, .05, .5)] == \
        ["***", "**", "*", "n.s.", "n.s."]


def test_one_sided_errorbar():
    lower, upper = P.one_sided_errorbar([.3, -.2, 0], [.1, .05, .02])
    assert np.allclose(lower, [0, .05, 0]) and np.allclose(upper, [.1, 0, .02])


@pytest.mark.parametrize("lo,hi", [(-.07, .45), (0, .2), (.1, .9), (-.3, -.1)])
def test_zero_anchored_ticks(lo, hi):
    ticks, (t0, t1) = P.zero_anchored_ticks(lo, hi)
    assert np.any(np.isclose(ticks, 0))
    assert t0 <= min(lo, 0) and t1 >= hi
    assert np.allclose(np.diff(ticks), np.diff(ticks)[0])


def test_title_is_off_by_default():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    P.title(ax, "Task")
    P.title(fig, "Task")
    assert ax.get_title() == "" and fig._suptitle is None
    plt.close(fig)


def test_save_writes_rgb_png_pdf_and_tsv(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from PIL import Image
    monkeypatch.setattr(cfg, "FIGURES", tmp_path / "fig")
    monkeypatch.setattr(P, "SOURCE_DATA", tmp_path / "src")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    P.save(fig, "panel.png", pd.DataFrame(dict(x=[1 / 3], y=[np.nan])))
    plt.close(fig)
    assert Image.open(tmp_path / "fig" / "panel.png").mode == "RGB"
    assert (tmp_path / "fig" / "panel.pdf").stat().st_size > 0
    lines = (tmp_path / "src" / "panel.tsv").read_text().splitlines()
    assert lines == ["x\ty", "0.333333\tn/a"]
    assert not (tmp_path / "src" / "panel.csv").exists()
