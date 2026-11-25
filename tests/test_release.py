"""Tests on the released data and derived files.

These check that the dataset satisfies the assumptions the code makes, that the
caches, tables, figures' source data and config constants agree with each other,
and that the headline numbers of the paper are reproduced. They read the
released dataset (cfg.DATA, with cfg.DERIVATIVES) and take about a minute.
Results written to cfg.OUTPUT by a rerun are compared with the released ones.
"""

import numpy as np
import pandas as pd
import pytest

from facescenes import behaviour as B
from facescenes import config as cfg
from facescenes import data as D
from facescenes import encoding as E
from facescenes import nan
from facescenes import stats as S

TABLES = cfg.DERIVATIVES / "tables"
SOURCE = cfg.DERIVATIVES / "source_data"
CACHE = cfg.DERIVATIVES / "encoding"
FRESH = (sorted(p for p in cfg.OUTPUT.rglob("*")
                if p.suffix in {".npz", ".tsv"})
         if cfg.OUTPUT.is_dir() else [])
N_IMAGES = {"facedetection": 108, "movingwindow": 120}


def cache(task):
    return dict(np.load(CACHE / f"{task}_encoding.npz"))


# ------------------------------------------------------------- the dataset
def test_participants():
    p = D.participants()
    assert len(p) == 76 and p["participant_id"].is_unique
    assert D.participant_ids("facedetection") == [f"sub-{i:03d}" for i in range(1, 39)]
    assert D.participant_ids("movingwindow") == [f"sub-{i:03d}" for i in range(39, 77)]


def test_stimulus_order():
    names = D.stimulus_names()
    assert len(names) == 120 and len(set(names)) == 120
    assert all("Conferanceroom" in names[i] for i in range(40, 50))
    assert "Artstudio_06" in names[2] and "Restaurant_16" in names[113]


@pytest.mark.parametrize("task", cfg.TASKS)
def test_event_invariants(task):
    for pid in D.participant_ids(task):
        ev = D.events(pid, task)
        assert set(ev["trial_type"]) <= set(cfg.CONDITIONS)
        s = ev[ev["phase"] == "search"]
        # every image at most once per participant
        assert not s.groupby("trial")["stim_file"].first().duplicated().any()
        for _, g in s[s["timeout"] == 0].groupby("trial"):
            # the last row of a successful trial is the detection fixation,
            # which is what data.trials() takes the latency from
            assert g["is_detection"].iloc[-1] == 1 and g["is_detection"].sum() == 1


@pytest.mark.parametrize("task", cfg.TASKS)
def test_trials(task):
    t = D.trials(task)
    assert t["participant"].nunique() == 38
    assert not t.duplicated(["participant", "image"]).any()
    assert (t["latency"] > cfg.MIN_FIXATION_ONSET_S).all()
    assert (t["latency"] < cfg.STIM_DUR[task]).all()
    assert (t["n_fixations"] >= 1).all()
    assert not set(t["image"]) & set(cfg.EXCLUDE_STIMULI[task])


@pytest.mark.parametrize("task", cfg.TASKS)
def test_analysable_images(task):
    keep = D.analysable_images(task)
    assert len(keep) == N_IMAGES[task]
    m = D.image_means(task, log=True)
    assert not np.isnan(m[:, keep]).any()


def test_features():
    X = E._raw_features()
    assert X.shape == (120, len(cfg.FEATURE_NAMES))
    assert np.isfinite(X).all()
    for task in cfg.TASKS:
        Z, y, keep, splits = E.design(task)
        assert np.allclose(Z.mean(0), 0) and np.allclose(Z.std(0), 1)
        assert y.shape == (2, N_IMAGES[task]) and len(splits) == cfg.N_FOLDS


# ------------------------------------------------- headline numbers (paper)
@pytest.mark.parametrize("task,t,p,dz", [("facedetection", -4.14, 1.95e-4, -0.67),
                                          ("movingwindow", -7.96, 1.55e-9, -1.29)])
def test_detection_latency(task, t, p, dz):
    r = B.detection_latency(task)
    assert r["n"] == 38
    assert r["t"] == pytest.approx(t, abs=.005)
    assert r["p"] == pytest.approx(p, rel=.01)
    assert r["dz"] == pytest.approx(dz, abs=.005)


def test_figure1b_means():
    a, b = B.participant_means(B.trial_table("facedetection"), "latency_ms")
    assert a.mean() == pytest.approx(208.77, abs=.01)
    assert b.mean() == pytest.approx(221.70, abs=.01)


@pytest.mark.parametrize("task", cfg.TASKS)
def test_reliability_constants(task):
    """config.RELIABILITY is hard-coded; a fresh estimate must agree."""
    lat = D.latency_array(task)
    log = np.log(np.where(lat > 0, lat, np.nan))
    keep = D.analysable_images(task)
    rng = np.random.default_rng(7)
    for ci, cond in enumerate(cfg.CONDITIONS):
        m, _ = S.split_half_reliability(log[:, ci, :], keep, rng, 200)
        assert m == pytest.approx(cfg.RELIABILITY[task][cond], abs=.02)


# ----------------------------------------------- caches, tables, source data
@pytest.mark.parametrize("task", cfg.TASKS)
def test_cache_reproduces_observed_predictivity(task):
    """The cached point estimates follow from the current data and code."""
    res = cache(task)
    Z, y, keep, splits = E.design(task)
    assert np.array_equal(res["images"], keep)
    s, u = E.fold_scores(Z, y[0], splits, E.unique_columns())
    rel = cfg.RELIABILITY[task]["preview"]
    assert np.allclose(s.mean(0) / rel, res["single_preview"])
    assert np.allclose(nan.mean(u) / rel, res["unique_preview"], equal_nan=True)


@pytest.mark.parametrize("task", cfg.TASKS)
def test_cache_has_no_p_for_untested_features(task):
    res = cache(task)
    for kind in ("single", "unique"):
        for c in cfg.CONDITIONS:
            est, p = res[f"{kind}_{c}"], res[f"p_{kind}_{c}"]
            assert np.array_equal(np.isnan(est), np.isnan(p))
            assert np.all((p[~np.isnan(p)] > 0) & (p[~np.isnan(p)] <= 1))


@pytest.mark.parametrize("kind", ["single", "unique"])
def test_tables_match_cache(kind):
    tab = pd.read_csv(TABLES / f"encoding_{kind}.tsv", sep="\t")
    for task in cfg.TASKS:
        res = cache(task)
        t = tab[tab["task"] == task].set_index("feature")
        for k, label in enumerate(cfg.FEATURE_LABELS):
            if np.isnan(res[f"{kind}_preview"][k]):
                assert label not in t.index
                continue
            for c in cfg.CONDITIONS:
                assert t.loc[label, f"{c}_r2"] == pytest.approx(res[f"{kind}_{c}"][k])
                assert t.loc[label, f"{c}_p"] == pytest.approx(res[f"p_{kind}_{c}"][k])
                assert t.loc[label, f"{c}_p_bonferroni"] == pytest.approx(
                    min(1, 20 * res[f"p_{kind}_{c}"][k]))
                lo, hi = t.loc[label, f"{c}_r2_ci95_low"], t.loc[label, f"{c}_r2_ci95_high"]
                assert lo <= hi


@pytest.mark.parametrize("panel,task,kind", [
    ("fig3a_single_face_detection", "facedetection", "single"),
    ("fig3b_single_moving_window", "movingwindow", "single"),
    ("fig3c_unique_face_detection", "facedetection", "unique"),
    ("fig3d_unique_moving_window", "movingwindow", "unique")])
def test_figure3_source_data_match_cache(panel, task, kind):
    src = pd.read_csv(SOURCE / f"{panel}.tsv", sep="\t")
    res = cache(task)
    ok = ~np.isnan(res[f"{kind}_preview"])
    assert list(src["feature"]) == list(np.array(cfg.FEATURE_LABELS)[ok])
    for c in cfg.CONDITIONS:
        assert np.allclose(src[f"{c}_r2"], res[f"{kind}_{c}"][ok], rtol=1e-5)
        assert np.allclose(src[f"{c}_p"], res[f"p_{kind}_{c}"][ok], rtol=1e-5)


def test_source_data_complete():
    figs = {p.stem for p in cfg.FIGURES.glob("*.png")}
    assert figs == {p.stem for p in SOURCE.glob("*.tsv")}
    assert figs == {p.stem for p in cfg.FIGURES.glob("*.pdf")}


def test_figure1_source_data_has_all_participants():
    src = pd.read_csv(SOURCE / "fig1b_latency_face_detection.tsv", sep="\t")
    assert len(src) == 38 and src["participant"].is_unique
    assert src["preview_mean_latency_ms"].mean() == pytest.approx(208.77, abs=.01)


# ------------------------------------------------- rerun against release
def fresh_id(path):
    return str(path.relative_to(cfg.OUTPUT))


@pytest.mark.parametrize("path", FRESH, ids=fresh_id)
def test_fresh_result_matches_release(path):
    released = cfg.DERIVATIVES / path.relative_to(cfg.OUTPUT)
    assert released.is_file(), released
    if path.suffix == ".tsv":
        pd.testing.assert_frame_equal(pd.read_csv(path, sep="\t"),
                                      pd.read_csv(released, sep="\t"),
                                      check_exact=False, rtol=1e-6)
        return
    new, old = np.load(path), np.load(released)
    assert sorted(new.files) == sorted(old.files)
    for k in old.files:
        if old[k].dtype.kind == "f":
            assert np.allclose(new[k], old[k], rtol=1e-6, equal_nan=True), k
        else:
            assert np.array_equal(new[k], old[k]), k
