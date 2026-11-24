"""
Control analysis: where the eye was when the search display appeared.

The preview benefit could in principle be an oculomotor effect rather than a
guidance effect. If seeing the scene lets observers move their eyes toward the
upcoming face before the search display arrives, they start closer to the
target and reach it sooner, without any of the top-down expectation the
encoding models are meant to capture. This module tests that account.

The measure
-----------
Eye position at target onset is read from the fixation that was in progress
when the display appeared. Recording begins at display onset, so that
fixation's start time is clipped to the event and its mean position reflects
the oculomotor state established BEFORE the target: saccade latencies of at
least ~150 ms make a target-driven change within it impossible. Which fixation
qualifies is defined by MIN_ONSET_FIXATION_S and ONSET_GAZE_WINDOW_S in config.

This is an event-based estimate, not a continuous trace. The 500 ms
fixation-dot interval before the search display was not sampled, so the
fixation straddling onset is the only pre-target position available.

Two distances are computed from it, both in degrees of visual angle: distance
from the central fixation point, which asks whether the preview moves the eye
at all, and distance from the face centroid, which asks whether the movement is
TOWARD the target. Only the second is the confound; a displacement unrelated to
the face cannot produce a head start.

Why the tests here are on the raw millisecond scale
---------------------------------------------------
Detection latency is tested on logs everywhere else in this package, because it
is multiplicative. The regressions here are on raw milliseconds instead, and
deliberately so: the bound in orienting_bound multiplies a distance in degrees
by a slope in ms/deg, which is only meaningful if the slope is additive in
milliseconds. A log model would give a bound in ratios that cannot be compared
with the millisecond effect it is meant to bound.
"""

import numpy as np
import pandas as pd

from . import behaviour as B
from . import config as cfg
from . import data as D
from . import stats as S


def _to_image_coords(gaze_x, gaze_y):
    """Screen pixels to displayed-image pixels; the image is screen-centred."""
    return (gaze_x - cfg.DISPLAY["width_px"] / 2 + cfg.DISPLAY["image_width_px"] / 2,
            gaze_y - cfg.DISPLAY["height_px"] / 2 + cfg.DISPLAY["image_height_px"] / 2)


def onset_gaze_table(task, window=None):
    """Per-trial onset gaze geometry, joined onto the standard trial table.

    Adds d_centre_deg and d_face_deg to the columns of data.trials(), keeping
    that function as the single definition of detection latency, the fixation
    count and the exclusions. Trials with no qualifying onset fixation are
    absent, so len() against data.trials() gives the number dropped.

    `window` is the latest onset, in s after the display, at which a fixation
    still qualifies; None means config.ONSET_GAZE_WINDOW_S. Pass np.inf to drop
    the criterion, as the robustness check in run_onset_gaze_control.py does.
    """
    window = cfg.ONSET_GAZE_WINDOW_S if window is None else window
    rows = []
    for pid in D.participant_ids(task):
        ev = D.events(pid, task)
        search = ev[(ev["phase"] == "search") & (ev["timeout"] == 0)]
        for trial, g in search.groupby("trial", sort=True):
            # A fixation with no recorded position cannot supply the estimate,
            # and would otherwise reach the regressions as a NaN coefficient
            # rather than as an error. There are none in the released data.
            at_onset = g[(g["duration"] >= cfg.MIN_ONSET_FIXATION_S)
                         & (g["onset"] <= window)
                         & g["gaze_x"].notna() & g["gaze_y"].notna()]
            if at_onset.empty:
                continue
            f = at_onset.iloc[0]
            rows.append(dict(participant=pid, trial=int(trial),
                             gaze_x=float(f["gaze_x"]), gaze_y=float(f["gaze_y"])))

    df = D.trials(task).merge(pd.DataFrame(rows), on=["participant", "trial"])
    names = D.stimulus_names()
    centres = {i: B.face_centre(names[i]) for i in df["image"].unique()}
    fx, fy = np.array([centres[i] for i in df["image"]]).T
    ix, iy = _to_image_coords(df["gaze_x"].to_numpy(), df["gaze_y"].to_numpy())
    ppd = cfg.DISPLAY["px_per_degree"]

    df["latency_ms"] = df["latency"] * 1000
    df["d_centre_deg"] = np.hypot(df["gaze_x"] - cfg.DISPLAY["width_px"] / 2,
                                  df["gaze_y"] - cfg.DISPLAY["height_px"] / 2) / ppd
    df["d_face_deg"] = np.hypot(ix - fx, iy - fy) / ppd
    return df


def displacement(task, df=None):
    """Does the preview move the eye, and does it move it toward the face?

    Paired tests on per-participant MEDIAN distances. Medians rather than means
    because a handful of trials start with the eye far off centre and would
    otherwise dominate a participant's value. Returns one paired_test dict per
    distance, keyed by column name.
    """
    df = onset_gaze_table(task) if df is None else df
    return {col: S.paired_test(*B.participant_means(df, col, agg="median"))
            for col in ("d_centre_deg", "d_face_deg")}


def _within_participant_ols(df, covariates, term, min_trials=20):
    """One OLS per participant; returns that term's coefficient for each.

    Design matrix is [intercept, preview, *covariates] and the dependent
    variable is latency in ms, so term 1 is the preview effect and terms 2
    onward are the covariate slopes. Fitting within participant rather than
    pooling means the estimate cannot be carried by between-participant
    differences in either latency or resting eye position. Participants with
    fewer than min_trials usable trials are skipped.
    """
    out = []
    for _, g in df.groupby("participant"):
        if len(g) < min_trials:
            continue
        X = [np.ones(len(g)), (g["trial_type"] == "preview").to_numpy(float)]
        X += [g[c].to_numpy(float) for c in covariates]
        beta, *_ = np.linalg.lstsq(np.column_stack(X),
                                   g["latency_ms"].to_numpy(float), rcond=None)
        out.append(beta[term])
    return np.array(out)


def preview_effect(task, covariates=(), df=None):
    """Preview coefficient in ms, tested across participants against zero.

    With covariates=() this is the unadjusted effect, estimated the same way as
    the adjusted one so that the two are comparable. Comparing a median-based
    effect with a regression coefficient instead makes the adjustment appear to
    change the estimate when it has not.
    """
    df = onset_gaze_table(task) if df is None else df
    return S.one_sample_test(_within_participant_ols(df, list(covariates), 1))


def distance_slope(task, covariate="d_face_deg", df=None):
    """How much latency each degree of onset distance buys, in ms/deg.

    Both distances are always in the model, so each slope is adjusted for the
    other: the face term is not merely reporting generic eccentricity. Serves
    as the positive control -- if onset position mattered at all, the
    gaze-to-face slope must be positive -- and supplies the multiplier for
    orienting_bound.
    """
    df = onset_gaze_table(task) if df is None else df
    cols = ["d_face_deg", "d_centre_deg"]
    return S.one_sample_test(
        _within_participant_ols(df, cols, 2 + cols.index(covariate)))


def orienting_bound(task, df=None):
    """Largest slice of the preview benefit that pre-target orienting can explain.

    The most favourable end of the confidence interval for the condition
    difference in gaze-to-face distance, times the slope of latency on that
    distance. This is deliberately generous: it grants the confound the largest
    displacement toward the face the data will tolerate, then converts it to
    milliseconds at the observed exchange rate. If the result is small next to
    the benefit, the account fails even at its best.
    """
    df = onset_gaze_table(task) if df is None else df
    towards = displacement(task, df)["d_face_deg"]["diff_ci"][0]
    slope = distance_slope(task, "d_face_deg", df)["mean"]
    return dict(ms=abs(towards * slope), degrees=towards, ms_per_degree=slope)


def exclusion_by_condition(task, df=None):
    """Is the loss of trials without a qualifying onset fixation balanced?

    Per-participant percentage of successful trials dropped, compared between
    preview and no-preview with a paired t-test on the raw scale. An imbalance
    would mean the two conditions are compared on differently selected trials.
    """
    df = onset_gaze_table(task) if df is None else df
    t = D.trials(task)
    kept = t.set_index(["participant", "trial"]).index.isin(
        df.set_index(["participant", "trial"]).index)
    t = t.assign(dropped_pct=100.0 * ~kept)
    a, b = B.participant_means(t, "dropped_pct")
    return S.paired_test(a, b)
