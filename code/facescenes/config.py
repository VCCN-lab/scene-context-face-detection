"""
Every choice that affects a reported number, in one place.

Nothing here is inferred at run time: if a value determines a figure or a
statistic, it is written down explicitly so a reader can see it and change it.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def data_root(environ=os.environ):
    """The released BIDS dataset: $FACESCENES_DATA if set, else <repo>/data."""
    return Path(environ.get("FACESCENES_DATA", ROOT / "data"))


# The released dataset is read only. Its derivatives are nested in it, as BIDS
# derivative datasets. Scripts write their results to OUTPUT, and
# readers of cached results prefer a result in OUTPUT over the released one
# (data.results_path).
DATA = data_root()
DERIVATIVES = DATA / "derivatives"
FEATURES = DERIVATIVES / "stimulus_features"
OUTPUT = ROOT / "derivatives"
FIGURES = ROOT / "figures"

# The published Zenodo record that download.fetch() unpacks into <repo>/data:
# the archive's URL (a "doi:<DOI>/<file name>" URL resolves through Zenodo) and
# its checksum ("md5:<hex digest>"). Record 24581856, DOI
# 10.5281/zenodo.24581856.
ZENODO_URL = ("doi:10.5281/zenodo.24581856/"
              "scene-context-face-detection_data.zip")
ZENODO_HASH = "md5:937db6f707ec8f6e905513871ee77678"

TASKS = ("facedetection", "movingwindow")
TASK_LABEL = {"facedetection": "Face detection task",
              "movingwindow": "Moving-window task"}

# Search display duration; a trial reaching it is a timeout (the face was never
# fixated), so no detection latency exists and the trial is dropped.
STIM_DUR = {"facedetection": 6.0, "movingwindow": 20.0}

# A fixation counts only if it begins more than this long after search-display
# onset. This removes the carry-over fixation from the preceding fixation-dot
# period, which is already on screen when the scene appears.
MIN_FIXATION_ONSET_S = 0.1

# ------------------------------------------------------------- onset-gaze
# The onset-gaze control reads eye position at the moment the search display
# appeared from the fixation that was in progress at that moment. Recording
# starts at display onset, so that fixation's start time is clipped to the
# event and its mean position reflects the oculomotor state established BEFORE
# the target: saccade latencies of at least ~150 ms rule out any target-driven
# change within it.
#
# A fixation qualifies if it lasted at least MIN_ONSET_FIXATION_S -- shorter
# events at recording start are artefacts of the clipping -- and began no later
# than ONSET_GAZE_WINDOW_S after the display, which keeps it below the saccade
# latency floor. The first qualifying fixation of the trial supplies the
# estimate; trials without one are dropped.
#
# The window matters only in the moving-window task, where the eye more often
# moves immediately and it drops 18 % of trials against 0.3 % in face
# detection. It is not a confound: the drop rate does not differ between
# conditions in either task (face detection -0.05 pp, p = .73; moving window
# -0.8 pp, p = .48), and every result of the control is unchanged when the
# window is removed.
MIN_ONSET_FIXATION_S = 0.04
ONSET_GAZE_WINDOW_S = MIN_FIXATION_ONSET_S      # same 100 ms boundary

# ---------------------------------------------------------------- exclusions
# Participants. None are excluded. The two counterbalancing groups have 19
# participants each.
EXCLUDE_PARTICIPANTS = {
    "facedetection": set(),
    "movingwindow": set(),
}

# Stimuli, by index into the alphabetical with_face list (see stimulus_names()).
#   40-49  the ten Conferanceroom scenes. A filename-spelling mismatch at
#          collection time (scene files Conferanceroom_*, preview files
#          Conferenceroom_*) meant the experiment's pairing failed and these
#          were shown with a grey preview to EVERY participant, in both
#          counterbalancing groups. They therefore have no preview data at all
#          and cannot contribute to a within-image comparison.
#   2, 113 Artstudio_06 and Restaurant_16, whose pooled mean latency lies more
#          than 3 SD above the mean in the face-detection task (z = +6.7, +6.5).
#          Neither exceeds 3 SD in the moving-window task (z = +2.1, +1.4), so
#          that task excludes no stimuli.
EXCLUDE_STIMULI = {
    "facedetection": [2, 113] + list(range(40, 50)),
    "movingwindow": [],
}

# ------------------------------------------------------------ cross-validation
# Balanced repeated two-fold splits over images: the images are shuffled, cut in
# half, and each half serves as the test set once. With 50 repetitions every
# image is held out exactly 50 times, over 100 folds in total. Preferable to 100
# independent random half-splits, where each image is held out a random number
# of times.
N_REPEATS = 50
N_FOLDS = N_REPEATS * 2
SEED = 20260805

# ---------------------------------------------------------------- predictors
FEATURE_NAMES = ["FaceExpect", "Pixel", "Crowding", "Clutter_rms",
                 "FaceSize", "FaceEcc", "BodySize", "BodyEcc",
                 "SceneFaceDet", "UntrainedCNN"]
FEATURE_LABELS = ["Face expect.", "Pixel", "Edge density", "RMS contrast",
                  "Face size", "Face ecc.", "Body size", "Body ecc.",
                  "SceneFaceDet", "Untrained"]

# The unique-contribution model omits the untrained CNN. It is a near-redundant
# operationalization of the same quantity as SceneFaceDet (Spearman r = .91,
# Pearson r = .76): including both leaves total predictivity essentially
# unchanged while splitting the shared contribution between them, which drives
# each one's unique value toward zero.
EXCLUDE_FROM_UNIQUE = {"UntrainedCNN"}

# CNN read-out layer: index 6 of 7 (the last fully connected layer).
CNN_LAYER = 6
CNN_KEY = "layer_means_resp_face_cos_f_n"

# ---------------------------------------------------------------- reliability
# Split-half reliability of the image-wise latencies, Spearman-Brown corrected.
# Both are computed on per-image means of LOG latency (geometric means), the
# exact quantity that is modelled -- data.image_means(log=True).
# Regenerate with run_reliability.py.
#
# RELIABILITY (Spearman) noise-corrects predictivity throughout, because
# predictivity is a Spearman correlation.
RELIABILITY = {
    "facedetection": {"preview": 0.739, "no_preview": 0.859},
    "movingwindow": {"preview": 0.824, "no_preview": 0.863},
}

# RELIABILITY_PEARSON exists only for the ceiling below.
RELIABILITY_PEARSON = {
    "facedetection": {"preview": 0.780, "no_preview": 0.860},
    "movingwindow": {"preview": 0.840, "no_preview": 0.887},
}

# Ceiling on R^2 when BOTH axes are noisy image-wise means, as in Fig. 1d/e,
# where the fit is least squares on log-scale means: the observed Pearson
# correlation is attenuated by sqrt(rel_x * rel_y), so R^2 cannot exceed the
# product of the two Pearson reliabilities.
R2_CEILING = {t: r["preview"] * r["no_preview"]
              for t, r in RELIABILITY_PEARSON.items()}

# ---------------------------------------------------------------- display
# From Results.Disp in the raw session files; identical in both experiments.
#
# px_per_degree is the scale at the CENTRE of the display: (3840 / 600) px per
# mm x 460 mm x tan(1 deg) = 51.38. Dividing the full width in pixels by the
# full width in degrees gives ~58 instead, but that is the average over a
# tangent-distorted axis, not the scale at fixation, and it is the wrong
# constant for distances measured from the centre. The panel width itself
# (600 mm) is the display's own EDID report and has not been measured, so the
# absolute degree values carry that uncertainty. No inference depends on it:
# the constant cancels from every paired test and scales the onset-gaze
# regression coefficients and the orienting bound by the same factor.
DISPLAY = dict(width_mm=600, height_mm=340, width_px=3840, height_px=2160,
               refresh_hz=60, viewing_distance_mm=460, px_per_degree=51.383,
               image_width_px=2400, image_height_px=1800)

CONDITIONS = ("preview", "no_preview")
COND_LABEL = {"preview": "Preview", "no_preview": "No preview"}

# Figure colours, matching the published figures.
C_PREVIEW = (239 / 255, 67 / 255, 53 / 255)
C_NO_PREVIEW = (240 / 255, 230 / 255, 140 / 255)
