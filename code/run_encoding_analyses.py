#!/usr/bin/env python3
"""
Single-feature and unique predictivity for both experiments (Figure 3), with
image-level permutation tests and standardised regression coefficients.

The permutations take roughly half an hour per experiment at the default 5000,
so the results are cached. Without --force the released results in
data/derivatives/encoding/ are read, or a rerun's in derivatives/encoding/ when
one exists; with --force they are recomputed and written to
derivatives/encoding/, leaving the released copy unchanged. The notebooks read
the same cache, so the figures and this script cannot disagree.

Three tests are run.

  predictivity > 0   The latencies are shuffled across images and the entire
                     cross-validation is repeated on the shuffled data, so the
                     null reflects model fitting as well as scoring. One-sided:
                     predicting worse than chance is not of interest.

  preview vs no      The two condition means are exchanged for a random subset
                     of images and the pipeline is re-run. Two-sided.
                     The null needs a correction. After the swap both
                     pseudo-conditions are equivalent ~50/50 mixtures, yet the
                     statistic still divides them by different reliabilities,
                     which places the null at R * (1/rel_p - 1/rel_n) instead of
                     zero. The observed statistic is unbiased, because each real
                     condition is divided by its own correct reliability, so the
                     null alone is recentred. Recentring the observed value too
                     would over-correct.

Run: python3 run_encoding_analyses.py [--permutations 5000]
"""

import argparse

import numpy as np

from facescenes import config as cfg
from facescenes import data as D
from facescenes import encoding as E
from facescenes import nan
from facescenes import stats as S


def analyse(task, n_perm, force=False):
    name = f"{task}_encoding.npz"
    cached = D.results_path("encoding", name)
    if cached.exists() and not force:
        print(f"{task}: loading {cached}")
        return dict(np.load(cached))

    Z, y, keep, splits = E.design(task)
    n_img = len(keep)
    uniq = E.unique_columns()
    rel = cfg.RELIABILITY[task]
    rng = np.random.default_rng(cfg.SEED)

    obs = {}
    for ci, cond in enumerate(cfg.CONDITIONS):
        s, u = E.fold_scores(Z, y[ci], splits, uniq)
        obs[cond] = (s.mean(0) / rel[cond], nan.mean(u) / rel[cond],
                     s.std(0, ddof=1) / rel[cond], nan.std(u) / rel[cond])

    P = Z.shape[1]
    null_zero = {c: {"single": np.zeros((n_perm, P)), "unique": np.zeros((n_perm, P))}
                 for c in cfg.CONDITIONS}
    null_diff = {"single": np.zeros((n_perm, P)), "unique": np.zeros((n_perm, P))}

    print(f"{task}: {n_perm} permutations over {n_img} images")
    for i in range(n_perm):
        # (a) predictivity > 0
        for ci, cond in enumerate(cfg.CONDITIONS):
            yp = y[ci].copy()
            good = ~np.isnan(yp)
            yp[good] = rng.permutation(yp[good])
            s, u = E.fold_scores(Z, yp, splits, uniq)
            null_zero[cond]["single"][i] = s.mean(0) / rel[cond]
            null_zero[cond]["unique"][i] = nan.mean(u) / rel[cond]
        # (b) preview vs no preview
        flip = rng.random(n_img) < 0.5
        y0 = np.where(flip, y[1], y[0])
        y1 = np.where(flip, y[0], y[1])
        s0, u0 = E.fold_scores(Z, y0, splits, uniq)
        s1, u1 = E.fold_scores(Z, y1, splits, uniq)
        null_diff["single"][i] = (s0.mean(0) / rel["preview"]
                                  - s1.mean(0) / rel["no_preview"])
        null_diff["unique"][i] = (nan.mean(u0) / rel["preview"]
                                  - nan.mean(u1) / rel["no_preview"])
        if (i + 1) % 500 == 0:
            print(f"   {i + 1}/{n_perm}", flush=True)

    res = dict(features=np.array(cfg.FEATURE_NAMES), images=keep)
    for ci, cond in enumerate(cfg.CONDITIONS):
        s, u, ssd, usd = obs[cond]
        res[f"single_{cond}"] = s
        res[f"unique_{cond}"] = u
        res[f"single_sd_{cond}"] = ssd
        res[f"unique_sd_{cond}"] = usd
        for kind, o in (("single", s), ("unique", u)):
            res[f"p_{kind}_{cond}"] = np.array(
                [S.permutation_p(o[k], null_zero[cond][kind][:, k], "greater")
                 for k in range(P)])
    for kind in ("single", "unique"):
        diff = res[f"{kind}_preview"] - res[f"{kind}_no_preview"]
        centred = S.recentre_null(null_diff[kind])       # see module docstring
        res[f"delta_{kind}"] = diff
        res[f"p_delta_{kind}"] = np.array(
            [S.permutation_p(diff[k], centred[:, k]) for k in range(P)])

    bs, bf = {}, {}
    for ci, cond in enumerate(cfg.CONDITIONS):
        s, f = E.standardised_betas(Z, y[ci], splits, uniq)
        bs[cond], bf[cond] = s, f
        res[f"beta_single_{cond}"] = s.mean(0)
        res[f"beta_single_sd_{cond}"] = s.std(0, ddof=1)
        res[f"beta_full_{cond}"] = nan.mean(f)
        res[f"beta_full_sd_{cond}"] = nan.std(f)

    out = cfg.OUTPUT / "encoding"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / name, **res)
    return res


def report(task, res):
    names = cfg.FEATURE_LABELS
    n_test = len(names) * 2
    print("=" * 110)
    print(f"{cfg.TASK_LABEL[task]}")
    print("=" * 110)
    for kind, label in (("single", "SINGLE-FEATURE"), ("unique", "UNIQUE")):
        print(f"\n{label} predictivity   (beta from the "
              f"{'single-feature' if kind == 'single' else 'full 9-feature'} fit)")
        print(f"  {'feature':14s} | {'preview':>26s} | {'no preview':>26s} | {'delta':>16s}")
        print(f"  {'':14s} | {'r2':>7s} {'p':>7s} {'beta':>9s} "
              f"| {'r2':>7s} {'p':>7s} {'beta':>9s} | {'r2':>7s} {'p':>7s}")
        bkey = "beta_single" if kind == "single" else "beta_full"
        for k, nm in enumerate(names):
            if np.isnan(res[f"{kind}_preview"][k]):
                continue
            row = f"  {nm:14s} |"
            for cond in cfg.CONDITIONS:
                row += (f" {res[f'{kind}_{cond}'][k]:+7.3f}"
                        f" {res[f'p_{kind}_{cond}'][k]:7.4f}"
                        f" {res[f'{bkey}_{cond}'][k]:+9.3f} |")
            row += (f" {res[f'delta_{kind}'][k]:+7.3f}"
                    f" {res[f'p_delta_{kind}'][k]:7.4f}")
            print(row)
        print(f"  Bonferroni factor within this family: {n_test}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=5000,
                    help="at 500 the Monte Carlo SE near p = .05 is about .01, "
                         "enough for borderline features to change between runs")
    ap.add_argument("--force", action="store_true", help="ignore any cache")
    a = ap.parse_args()
    for task in cfg.TASKS:
        report(task, analyse(task, a.permutations, a.force))
        print()


if __name__ == "__main__":
    main()
