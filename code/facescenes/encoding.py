"""
Image-level encoding models.

A model is fitted to one half of the images and scored on the held-out half, so
predictivity measures generalization to stimuli the model has not seen. Two
model families:

  single-feature   latency ~ one predictor           (all 10 predictors)
  unique           predictivity(full) - predictivity(full minus one predictor)
                   over the 9 predictors of the full model

Scoring is the SIGNED squared Spearman correlation between predicted and
observed held-out latencies, sign(r) * r^2. It is signed because a
cross-validated model can predict worse than chance, and reporting an unsigned
square would hide that; negative values are meaningful and appear in the
figures. Because it is rank-based, single-feature predictivity is invariant to
any monotone transform of a predictor.

Predictors are z-scored before fitting. For unpenalised least squares this is a
reparameterization that leaves predictivity exactly unchanged -- it serves only
to put regression coefficients on a common scale.
"""

import functools

import numpy as np
import scipy.io as sio
from scipy import stats

from . import config as cfg
from . import data as D


# ------------------------------------------------------------------ scoring
def signed_r2(y, yhat):
    """sign(r) * r^2 for the Spearman correlation between y and yhat."""
    ry = stats.rankdata(y)
    rh = stats.rankdata(yhat)
    ry = ry - ry.mean()
    rh = rh - rh.mean()
    denom = np.sqrt((ry @ ry) * (rh @ rh))
    if denom == 0:
        return 0.0
    r = (ry @ rh) / denom
    return float(np.sign(r) * r * r)


def ols_predict(x_train, y_train, x_test):
    """Least-squares fit on the training images, prediction for the held-out."""
    A = np.c_[np.ones(len(x_train)), x_train]
    B = np.c_[np.ones(len(x_test)), x_test]
    beta, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    return B @ beta


def make_splits(n_images, rng, n_repeats=cfg.N_REPEATS):
    """Balanced repeated two-fold splits; returns the TRAIN indices of each fold.

    Each repetition shuffles the images, cuts them in half, and lets each half
    serve as the test set once, so every image is held out exactly n_repeats
    times across the 2 * n_repeats folds.
    """
    splits = []
    for _ in range(n_repeats):
        perm = rng.permutation(n_images)
        for fold in np.array_split(perm, 2):
            splits.append(np.setdiff1d(np.arange(n_images), fold))
    return splits


# --------------------------------------------------------------- predictors
@functools.lru_cache(maxsize=None)
def _raw_features():
    F = cfg.FEATURES
    face_expect = np.load(F / "face_expectation.npy")
    pixel = np.load(F / "pixel_difference.npy")
    rms = sio.loadmat(F / "contrast_edge_density.mat")
    face = sio.loadmat(F / "face_size_eccentricity.mat")["size_locs"]
    body = sio.loadmat(F / "body_size_eccentricity.mat")["size_locs"]
    k, L = cfg.CNN_KEY, cfg.CNN_LAYER
    scenefacedet = np.load(F / "cnn_scenefacedet.npz")[k][L, :]
    untrained = np.load(F / "cnn_untrained.npz")[k][L, :]
    return np.column_stack([face_expect, pixel,
                            rms["Edge_r32"].squeeze(), rms["RMS_r32"].squeeze(),
                            face[:, 0], face[:, 1], body[:, 0], body[:, 1],
                            scenefacedet, untrained])


def features(task):
    """(n_usable_image, 10) predictor matrix and the image indices it covers."""
    keep = D.analysable_images(task)
    return _raw_features()[keep], keep


def masked_features(task):
    """Predictors recomputed with the face region occluded, for the two CNNs."""
    keep = D.analysable_images(task)
    k, L = cfg.CNN_KEY, cfg.CNN_LAYER
    out = {}
    for name, fname in (("SceneFaceDet", "cnn_scenefacedet_face_masked.npz"),
                        ("UntrainedCNN", "cnn_untrained_face_masked.npz")):
        out[name] = np.load(cfg.FEATURES / fname)[k][L, :][keep]
    return out


def design(task):
    """z-scored predictors, log latency per condition, and the fold splits."""
    X, keep = features(task)
    Z = (X - X.mean(0)) / X.std(0)
    y = D.image_means(task, log=True)[:, keep]
    rng = np.random.default_rng(cfg.SEED)
    return Z, y, keep, make_splits(len(keep), rng)


# -------------------------------------------------------------------- models
def unique_columns():
    return [i for i, n in enumerate(cfg.FEATURE_NAMES)
            if n not in cfg.EXCLUDE_FROM_UNIQUE]


def fold_scores(Z, y, splits, unique_cols):
    """Per-fold single-feature and unique predictivity, both (n_fold, n_feature).

    Unique columns not in the full model are left as NaN.
    """
    n_img, P = Z.shape
    single = np.zeros((len(splits), P))
    unique = np.full((len(splits), P), np.nan)
    Zu = Z[:, unique_cols]
    for f, train in enumerate(splits):
        test = np.setdiff1d(np.arange(n_img), train)
        full = signed_r2(y[test], ols_predict(Zu[train], y[train], Zu[test]))
        for k in range(P):
            single[f, k] = signed_r2(
                y[test], ols_predict(Z[train][:, [k]], y[train], Z[test][:, [k]]))
        for j, k in enumerate(unique_cols):
            rest = [c for c in range(len(unique_cols)) if c != j]
            reduced = signed_r2(
                y[test], ols_predict(Zu[train][:, rest], y[train], Zu[test][:, rest]))
            unique[f, k] = full - reduced
    return single, unique


def standardised_betas(Z, y, splits, unique_cols):
    """beta +/- SD across folds, for the single-feature and the full model.

    Predictors and the dependent variable are z-scored WITHIN each training
    fold, so beta is the change in log latency in SD units per 1 SD of the
    predictor. A negative beta means the predictor predicts faster detection.
    beta is not noise-corrected: the reliability correction rescales
    predictivity, it does not apply to a regression slope.
    """
    n_img, P = Z.shape
    single = np.zeros((len(splits), P))
    full = np.full((len(splits), P), np.nan)
    for f, train in enumerate(splits):
        yt = y[train]
        ys = (yt - yt.mean()) / yt.std(ddof=1)
        for k in range(P):
            x = Z[train][:, [k]]
            A = np.c_[np.ones(len(train)), (x - x.mean(0)) / x.std(0, ddof=1)]
            single[f, k] = np.linalg.lstsq(A, ys, rcond=None)[0][1]
        Xu = Z[train][:, unique_cols]
        Xu = (Xu - Xu.mean(0)) / Xu.std(0, ddof=1)
        full[f, unique_cols] = np.linalg.lstsq(
            np.c_[np.ones(len(train)), Xu], ys, rcond=None)[0][1:]
    return single, full


def noise_correct(scores, task, condition):
    """Divide predictivity by the split-half reliability of that condition."""
    return scores / cfg.RELIABILITY[task][condition]


# ---------------------------------------------------------------- CNN layers
N_CNN_LAYERS = 7


def layer_predictivity(task, npz_name):
    """Noise-corrected single-feature predictivity of each CNN read-out layer.

    `npz_name` is a file in cfg.FEATURES (cnn_scenefacedet.npz
    or cnn_untrained.npz). Each layer's face/no-face distance is used as a
    single predictor, scored over the same folds as every other model.
    Returns {condition: array of N_CNN_LAYERS values}.
    """
    _, y, keep, splits = design(task)
    n_img = len(keep)
    raw = np.load(cfg.FEATURES / npz_name)[cfg.CNN_KEY]
    out = {c: np.zeros(N_CNN_LAYERS) for c in cfg.CONDITIONS}
    for L in range(N_CNN_LAYERS):
        v = raw[L, :][keep]
        z = ((v - v.mean()) / v.std())[:, None]
        for ci, cond in enumerate(cfg.CONDITIONS):
            acc = 0.0
            for train in splits:
                test = np.setdiff1d(np.arange(n_img), train)
                acc += signed_r2(y[ci][test],
                                 ols_predict(z[train], y[ci][train], z[test]))
            out[cond][L] = acc / len(splits) / cfg.RELIABILITY[task][cond]
    return out
