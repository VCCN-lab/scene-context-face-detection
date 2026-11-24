#!/usr/bin/env python3
"""
Split-half reliability of the image-wise detection latencies.

For each condition the participants contributing to an image are randomly split
in two, the image mean is computed within each half, and the two vectors are
correlated ACROSS IMAGES. The correlation is Spearman-Brown corrected to the full
sample and averaged over many random splits.

The half means are means of LOG latency -- the per-image geometric mean, the
same quantity as data.image_means(log=True), which the encoding models and
Figure 1d/e use. Four quantities are produced, and they answer different
questions:

  per condition   Spearman. Divides predictivity in Figure 3, correcting for
  (Spearman)      the fact that a model cannot predict noise. Spearman because
                  predictivity is a Spearman correlation.
  per condition   Pearson. Only for the ceiling below.
  (Pearson)
  ceiling         product of the two Pearson values: the ceiling on R^2 in
                  Figure 1d/e, where BOTH axes are noisy image-wise means and the
                  fit is least squares on the log scale. The observed correlation
                  is attenuated by sqrt(rel_x * rel_y), so R^2 cannot exceed
                  rel_x * rel_y.
  shared          both conditions pooled, roughly doubling the observations per
                  image; an estimate of reliability for a single latency measure

The values printed here are the ones hard-coded in config.RELIABILITY and
config.R2_CEILING. Rerun this if the data or the exclusions change.

Run: python3 run_reliability.py [--splits 2000]
"""

import argparse

import numpy as np

from facescenes import config as cfg
from facescenes import data as D
from facescenes import stats as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=2000)
    a = ap.parse_args()

    rng = np.random.default_rng(7)
    print(f"{a.splits} random splits per estimate, Spearman-Brown corrected,")
    print("per-image means of log latency\n")
    for task in cfg.TASKS:
        lat = D.latency_array(task)
        log = np.log(np.where(lat > 0, lat, np.nan))
        images = D.analysable_images(task)
        print(f"{cfg.TASK_LABEL[task]}   {len(images)} images, "
              f"{lat.shape[0]} participants")
        r, rp = {}, {}
        for ci, cond in enumerate(cfg.CONDITIONS):
            m, sd = S.split_half_reliability(log[:, ci, :], images, rng, a.splits)
            r[cond] = m
            print(f"   {cfg.COND_LABEL[cond]:12s} Spearman r = {m:.3f}  "
                  f"(SD across splits {sd:.3f})")
        for ci, cond in enumerate(cfg.CONDITIONS):
            m, sd = S.split_half_reliability(log[:, ci, :], images, rng, a.splits,
                                             method="pearson")
            rp[cond] = m
            print(f"   {cfg.COND_LABEL[cond]:12s} Pearson  r = {m:.3f}  "
                  f"(SD across splits {sd:.3f})")
        pooled = np.concatenate([log[:, 0, :], log[:, 1, :]])
        m, sd = S.split_half_reliability(pooled, images, rng, a.splits)
        print(f"   {'ceiling':12s} R2 = {rp['preview'] * rp['no_preview']:.3f}"
              f"   <- Pearson product, R^2 ceiling for Figure 1d/e")
        print(f"   {'shared':12s} r = {m:.3f}  (SD across splits {sd:.3f})")
        print(f"   config.RELIABILITY currently holds {cfg.RELIABILITY[task]}")
        print(f"   config.R2_CEILING currently holds {cfg.R2_CEILING[task]:.3f}\n")


if __name__ == "__main__":
    main()
