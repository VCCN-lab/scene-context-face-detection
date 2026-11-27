#!/usr/bin/env python3
"""
Does CNN predictivity depend on facial information? (Supplementary Figure 6)

The CNN feature is the representational distance between a scene with a face and
the same scene without one. Here it is recomputed with the face region occluded
by a grey mask, and the masked and visible versions are compared.

Why an ANOVA over images rather than a permutation test
-------------------------------------------------------
Predictivity is defined over a SET of images, so each cell of the design yields
a single number and there are no replicates for a factorial test. An earlier
version therefore used a permutation null that exchanged the visible and masked
feature values per image. That null is invalid: a permuted predictor is a
MIXTURE of visible and masked values, systematically worse than either pure
predictor, whereas the observed statistic compares two pure predictors. The
resulting null was bimodal, with no mass at all near zero.

Squared prediction error does decompose by image. Each image is a test item in
half of the folds, so averaging its held-out squared errors gives one value per
image per cell, and images become the replicates of a 2 (preview, no preview)
x 2 (face visible, face masked) repeated-measures ANOVA.

Two details matter. Log latency is z-scored within condition so that squared
errors are comparable across conditions; this leaves predictivity (rank-based)
and beta (internally standardised) untouched. And the ANOVA runs on LOG squared
error, because the raw errors are strongly right-skewed, which inflates the
error term and makes F far too conservative.

The main effect of condition is not interpretable here: preview and no-preview
latencies differ in reliability, so that contrast is confounded. The mask
contrast is clean, because within a condition both predictors are fitted to an
identical dependent variable and reliability cancels.

Run: python3 run_masked_face_analyses.py
"""

import numpy as np
from scipy import stats

from facescenes import config as cfg
from facescenes import encoding as E
from facescenes import stats as S

NETWORKS = ("UntrainedCNN", "SceneFaceDet")
NETWORK_LABEL = {"UntrainedCNN": "Untrained", "SceneFaceDet": "SceneFaceDet"}


def run(task):
    """Print the visible-versus-masked CNN comparison of `task` and return
    one result row per network and ANOVA effect."""
    Z, y, keep, splits = E.design(task)
    n_img = len(keep)
    masked = E.masked_features(task)
    rel = cfg.RELIABILITY[task]

    # z-score log latency within condition: squared errors are scale dependent
    yz = np.stack([(row - np.nanmean(row)) / np.nanstd(row, ddof=1) for row in y])

    rows = []
    print("=" * 84)
    print(f"{cfg.TASK_LABEL[task]}   {n_img} images as the random unit")
    print("=" * 84)

    for net in NETWORKS:
        col = cfg.FEATURE_NAMES.index(net)
        zv = Z[:, [col]]
        m = masked[net]
        zm = ((m - m.mean()) / m.std())[:, None]

        err = np.zeros((n_img, 2, 2))            # image x condition x mask
        pred = {}
        for ci, cond in enumerate(cfg.CONDITIONS):
            err[:, ci, 0] = S.per_image_squared_error(zv, yz[ci], splits, n_img)
            err[:, ci, 1] = S.per_image_squared_error(zm, yz[ci], splits, n_img)
            for mi, z in ((0, zv), (1, zm)):
                acc = sum(E.signed_r2(y[ci][np.setdiff1d(np.arange(n_img), tr)],
                                      E.ols_predict(z[tr], y[ci][tr],
                                                    z[np.setdiff1d(np.arange(n_img), tr)]))
                          for tr in splits)
                pred[(cond, mi)] = acc / len(splits) / rel[cond]

        print(f"\n  --- {NETWORK_LABEL[net]} ---")
        print(f"    {'':12s} {'predictivity':>26s} | {'log squared error':>22s}")
        print(f"    {'':12s} {'visible':>12s} {'masked':>13s} | "
              f"{'visible':>10s} {'masked':>11s}")
        for ci, cond in enumerate(cfg.CONDITIONS):
            print(f"    {cfg.COND_LABEL[cond]:12s} {pred[(cond, 0)]:12.3f} "
                  f"{pred[(cond, 1)]:13.3f} | {np.log(err[:, ci, 0]).mean():10.3f} "
                  f"{np.log(err[:, ci, 1]).mean():11.3f}")

        res = S.anova_2x2(np.log(err))
        label = {"B": "main effect of mask", "A": "condition (not interpretable)",
                 "AxB": "condition x mask"}
        print(f"    {'effect':30s} {'F':>9s} {'df':>9s} {'p':>10s} {'eta2p':>7s} "
              f"{'90% CI':>16s}")
        for k in ("B", "A", "AxB"):
            F, d1, d2, p, e2 = res[k]
            lo, hi = S.eta2p_ci(F, d1, d2)
            print(f"    {label[k]:30s} {F:9.3f} {d1:3d},{d2:<5d} {p:10.2e} {e2:7.3f} "
                  f"[{lo:.3f}, {hi:.3f}]")
            rows.append(dict(task=task, network=NETWORK_LABEL[net], effect=label[k],
                             F=F, df1=d1, df2=d2, p=p, eta2p=e2,
                             eta2p_ci90_low=lo, eta2p_ci90_high=hi,
                             **{f"r2_{c}_{'masked' if mi else 'visible'}": pred[(c, mi)]
                                for c in cfg.CONDITIONS for mi in (0, 1)}))

        d = (np.log(err[:, :, 1]) - np.log(err[:, :, 0])).mean(1)
        w = stats.wilcoxon(d)
        print(f"    masked - visible, pooled: Wilcoxon W = {w.statistic:.0f}, "
              f"p = {w.pvalue:.2e}, {100 * np.mean(d > 0):.0f}% of images favour visible")
    print()
    return rows


def main():
    """Run the masked-face comparison for every task."""
    for task in cfg.TASKS:
        run(task)


if __name__ == "__main__":
    main()
