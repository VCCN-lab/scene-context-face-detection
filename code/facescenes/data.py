"""
Reading the BIDS dataset and reducing it to the arrays the analyses use.

The dataset stores one row per fixation. Everything else -- detection latency,
fixation counts, scan path ratio -- is derived here, so there is exactly one
definition of each measure and every figure inherits it.

Detection latency
-----------------
The display was gaze-contingent: the scene disappeared as soon as the observer
fixated the face. Detection latency is therefore the onset of the LAST fixation
of the trial, measured from search-display onset. Fixations beginning within
MIN_FIXATION_ONSET_S of onset are discarded first (they carry over from the
preceding fixation-dot period), and timeout trials are dropped because the face
was never fixated. There is no upper cut-off.
"""

import functools

import numpy as np
import pandas as pd

from . import config as cfg
from . import nan


def results_path(kind, name):
    """A cached result: the rerun's in cfg.OUTPUT if present, else the
    released one in cfg.DERIVATIVES. `kind` is the derivative folder, e.g.
    "encoding"."""
    fresh = cfg.OUTPUT / kind / name
    return fresh if fresh.exists() else cfg.DERIVATIVES / kind / name


@functools.lru_cache(maxsize=None)
def participants():
    """participants.tsv as a DataFrame."""
    return pd.read_csv(cfg.DATA / "participants.tsv", sep="\t")


def participant_ids(task, apply_exclusions=True):
    """Sorted participant IDs of `task`, by default without exclusions."""
    p = participants()
    ids = sorted(p.loc[p["task"] == task, "participant_id"])
    if apply_exclusions:
        ids = [i for i in ids if i not in cfg.EXCLUDE_PARTICIPANTS[task]]
    return ids


@functools.lru_cache(maxsize=None)
def stimulus_names():
    """The 120 stimuli in canonical order, read from data/stimulus_order.tsv.

    This ordering defines the image axis of every array in the analyses and of
    every predictor file. It is stored explicitly rather than derived from the
    contents of the stimulus directory: listing a directory would silently
    reorder everything if a file were added, renamed or missing, and every
    predictor would then be attached to the wrong image.
    """
    order = pd.read_csv(cfg.DATA / "stimulus_order.tsv", sep="\t")
    return tuple(order["stimulus"])


@functools.lru_cache(maxsize=None)
def events(participant, task):
    """Fixation-level events for one participant."""
    f = (cfg.DATA / participant / "beh"
         / f"{participant}_task-{task}_events.tsv")
    return pd.read_csv(f, sep="\t")


def trials(task, apply_exclusions=True):
    """One row per usable trial, with the derived per-trial measures.

    Columns: participant, trial, image (index into stimulus_names()),
    trial_type, latency (s), n_fixations, path_length (px), timeout.
    """
    names = {n: i for i, n in enumerate(stimulus_names())}
    drop = set(cfg.EXCLUDE_STIMULI[task]) if apply_exclusions else set()
    rows = []
    for pid in participant_ids(task, apply_exclusions):
        ev = events(pid, task)
        search = ev[(ev["phase"] == "search") & (ev["timeout"] == 0)]
        for trial, g in search.groupby("trial", sort=True):
            img = names[g["stimulus"].iloc[0]]
            if img in drop:
                continue
            valid = g[g["onset"] > cfg.MIN_FIXATION_ONSET_S]
            if valid.empty:
                continue
            xy = g[["gaze_x", "gaze_y"]].to_numpy(float)
            path = float(np.nansum(np.hypot(*(xy[1:] - xy[:-1]).T))) if len(xy) > 1 else np.nan
            rows.append(dict(participant=pid, trial=int(trial), image=img,
                             trial_type=g["trial_type"].iloc[0],
                             latency=float(valid["onset"].iloc[-1]),
                             n_fixations=int(len(valid)),
                             path_length=path,
                             timeout=0))
    return pd.DataFrame(rows)


def latency_array(task, apply_exclusions=True):
    """(n_participant, 2, 120) detection latency in seconds; index 0 = preview.

    NaN where a participant has no usable trial for that image x condition. Note
    the design is counterbalanced BETWEEN participant groups, so each participant
    contributes to only one condition per image.
    """
    ids = participant_ids(task, apply_exclusions)
    pos = {p: i for i, p in enumerate(ids)}
    ci = {"preview": 0, "no_preview": 1}
    out = np.full((len(ids), 2, len(stimulus_names())), np.nan)
    for r in trials(task, apply_exclusions).itertuples():
        out[pos[r.participant], ci[r.trial_type], r.image] = r.latency
    return out


def image_means(task, log=False, apply_exclusions=True):
    """(2, 120) latency averaged across participants; NaN for unusable images.

    With log=True each latency is log-transformed FIRST and the logs are then
    averaged: the log of the per-image geometric mean. This is the dependent
    variable of the encoding models and of Fig. 1d/e. It is the same quantity
    the split-half reliability is computed on (stats.split_half_reliability),
    so the noise ceiling matches what is modelled, and unlike the log of an
    arithmetic mean it is not dominated by the occasional very slow trial
    (there is no upper latency cut-off).
    """
    lat = latency_array(task, apply_exclusions)
    if log:
        return nan.mean(np.log(np.where(lat > 0, lat, np.nan)), axis=0)
    return nan.mean(lat, axis=0)


def analysable_images(task):
    """Indices of images usable in a within-image comparison: data in BOTH
    conditions and not excluded."""
    m = image_means(task)
    ok = ~np.isnan(m[0]) & ~np.isnan(m[1])
    return np.setdiff1d(np.where(ok)[0], cfg.EXCLUDE_STIMULI[task])
