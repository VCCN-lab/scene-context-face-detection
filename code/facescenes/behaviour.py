"""
Behavioural measures derived from the fixation events.

All of these are computed per trial, then averaged within participant, so the
participant is the unit of the paired tests. Which measures are log-transformed
before testing is a deliberate choice, documented at each function: latency and
scan path ratio are multiplicative and right-skewed, counts and percentages are
not.
"""

import numpy as np
import pandas as pd
from PIL import Image

from . import config as cfg
from . import data as D
from . import stats as S

# The stimuli were presented at three times their stored size, so mask
# coordinates must be scaled to match the gaze coordinates.
STIMULUS_SCALE = 3


def face_centre(stimulus):
    """Centroid of the face region in displayed pixels, from the mask image."""
    mask = cfg.DATA / "stimuli" / "masks" / stimulus.replace("with_face", "face_mask")
    a = np.array(Image.open(mask).convert("RGB")).mean(2)
    ys, xs = np.nonzero(a < 128)
    return xs.mean() * STIMULUS_SCALE, ys.mean() * STIMULUS_SCALE


def trial_table(task):
    """Per-trial measures with the scan path ratio added.

    Scan path ratio = total distance travelled between successive fixations,
    divided by the straight-line distance from the central fixation point to the
    face centroid. A value of 1 means the observer went straight to the face.
    Because the display was gaze-contingent and ended on the first fixation on
    the face, every recorded fixation belongs to the search path -- there is no
    post-target segment to trim.
    """
    df = D.trials(task).copy()
    names = D.stimulus_names()
    centres = {}
    ratios = []
    for r in df.itertuples():
        stim = names[r.image]
        if stim not in centres:
            centres[stim] = face_centre(stim)
        fx, fy = centres[stim]
        direct = np.hypot(fx - cfg.DISPLAY["image_width_px"] / 2,
                          fy - cfg.DISPLAY["image_height_px"] / 2)
        ratios.append(r.path_length / direct if direct > 0 and not np.isnan(r.path_length)
                      else np.nan)
    df["scan_path_ratio"] = ratios
    df["latency_ms"] = df["latency"] * 1000
    return df


def participant_means(df, column, log=False, agg="mean"):
    """(preview, no_preview) arrays of per-participant means, aligned by index.

    With log=True the TRIAL values are logged before averaging. Averaging the
    logs is not the same as logging the average, and it is the former that makes
    the geometric means and the t-test consistent.

    `agg` selects how trials are pooled within a participant. Everything in this
    module uses the mean; the onset-gaze control passes "median", because the
    gaze distances it summarises are heavily right-skewed by the occasional
    trial that begins with the eye far off centre.
    """
    piv = participant_table(df, column, log, agg)
    return piv["preview"].to_numpy(float), piv["no_preview"].to_numpy(float)


def participant_table(df, column, log=False, agg="mean"):
    """participant_means as a table indexed by participant ID, one column per
    condition. Used for the figures' source data, so each row keeps its ID."""
    d = df
    if log:
        d = df.assign(**{column: np.log(df[column].where(df[column] > 0))})
    piv = d.pivot_table(index="participant", columns="trial_type",
                        values=column, aggfunc=agg)
    for c in cfg.CONDITIONS:
        if c not in piv:
            piv[c] = np.nan
    return piv[list(cfg.CONDITIONS)]


def detection_latency(task):
    """Preview vs no-preview detection latency. Tested on logs (right-skewed)."""
    df = trial_table(task)
    a, b = participant_means(df, "latency_ms")
    la, lb = participant_means(df, "latency_ms", log=True)
    return S.paired_test(a, b, la, lb)


def fixation_count(task):
    """Number of fixations needed to reach the face.

    Tested on the raw scale: these are small bounded counts, not a multiplicative
    quantity, and the floor at 1 rather than skew drives their asymmetry.
    """
    a, b = participant_means(trial_table(task), "n_fixations")
    return S.paired_test(a, b)


def scan_path_ratio(task):
    """Directness of the search path. Tested on logs: a ratio, bounded below by
    zero and unbounded above (trial-level skew above 4)."""
    df = trial_table(task)
    a, b = participant_means(df, "scan_path_ratio")
    la, lb = participant_means(df, "scan_path_ratio", log=True)
    return S.paired_test(a, b, la, lb)


def first_fixation_on_face(task):
    """Trials where the first fixation after scene onset already landed on the
    face, i.e. exactly one fixation began more than 100 ms after onset before the
    gaze-contingent check ended the trial.

    Returns the percentage per condition (tested raw) and the detection latency
    restricted to those trials (tested on logs).
    """
    df = trial_table(task)
    df["hit"] = (df["n_fixations"] == 1).astype(float) * 100
    a, b = participant_means(df, "hit")
    accuracy = S.paired_test(a, b)
    hits = df[df["n_fixations"] == 1]
    ha, hb = participant_means(hits, "latency_ms")
    hla, hlb = participant_means(hits, "latency_ms", log=True)
    return dict(accuracy=accuracy,
                latency=S.paired_test(ha, hb, hla, hlb),
                n_hit=int(len(hits)), n_total=int(len(df)),
                pct_overall=float(100 * len(hits) / len(df)))
