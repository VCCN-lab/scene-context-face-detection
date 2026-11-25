#!/usr/bin/env python3
"""
Full statistics for every test that the text summarises, as tables.

Writes BIDS TSV files (tab-separated, "n/a" for missing values) to
derivatives/tables/ and prints them. These are the
Supplementary Tables of the paper:

  feature_correlations.tsv   every pairwise Spearman correlation between the ten
                             predictors (Fig. 2e), n = 120 images, with exact p
                             and a 95% CI (Fisher z, Bonett-Wright SE).
  encoding_single.tsv        single-feature predictivity per task, feature and
  encoding_unique.tsv        condition (Fig. 3), and unique predictivity, with
                             the standardised beta, the noise-corrected r2, its
                             95% bootstrap CI, the one-sided permutation p
                             (uncorrected and Bonferroni x 20), and the
                             preview-minus-no-preview difference with its 95%
                             bootstrap CI and two-sided condition-swap p.
  encoding_contrasts.tsv     the planned contrasts (face expectation vs the
                             other features) with bootstrap CIs and the
                             one-sided p of run_contrast_test.py.
  masking_anova.tsv          the 2 x 2 ANOVAs of Supplementary Note 7 with 90% CIs
                             for partial eta squared.

Confidence intervals for predictivity come from a bootstrap over IMAGES, the
unit of inference everywhere in this analysis: images are resampled with
replacement and the whole cross-validation is rerun. Copies of a resampled image
are always kept in the same fold, so an image can never be in the training and
the test half at once. Noise correction uses the same reliability as the point
estimate. The p-values are not recomputed here: they come from the permutation
caches written by run_encoding_analyses.py and run_contrast_test.py, so the
tables and the figures cannot disagree.

Run: python3 run_statistics_tables.py [--bootstrap 1000]
     (needs both caches; about 15 minutes at 1000 resamples)
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from facescenes import config as cfg
from facescenes import encoding as E
from facescenes import nan
from facescenes import stats as S
import run_contrast_test as CT
import run_masked_face_analyses as MASK

CACHE = cfg.ROOT / "derivatives" / "encoding"
OUT = cfg.ROOT / "derivatives" / "tables"
N_BONF = 20       # ten features x two conditions, the family used in the text


def write_table(df, name):
    """Write `df` to OUT/<name>.tsv and return the path."""
    path = OUT / f"{name}.tsv"
    df.to_csv(path, sep="\t", index=False, na_rep="n/a")
    return path


def feature_correlations():
    X = E._raw_features()                       # all 120 images, as in Fig. 2e
    n = X.shape[0]
    rho = stats.spearmanr(X)
    rows = []
    for i in range(len(cfg.FEATURE_NAMES)):
        for j in range(i + 1, len(cfg.FEATURE_NAMES)):
            r, p = rho.statistic[i, j], rho.pvalue[i, j]
            lo, hi = S.spearman_ci(r, n)
            rows.append(dict(feature_1=cfg.FEATURE_LABELS[i],
                             feature_2=cfg.FEATURE_LABELS[j], n_images=n,
                             spearman_r=r, ci95_low=lo, ci95_high=hi, p=p))
    return pd.DataFrame(rows)


def bootstrap_splits(idx, rng, n_repeats=cfg.N_REPEATS):
    """Balanced two-fold splits over the DISTINCT images of a bootstrap sample.

    Returns training positions into the resampled arrays, so that every copy of
    an image falls on the same side of each split.
    """
    distinct = np.unique(idx)
    splits = []
    for _ in range(n_repeats):
        perm = rng.permutation(distinct)
        half = len(perm) // 2
        for part in (perm[:half], perm[half:]):
            splits.append(np.where(np.isin(idx, part))[0])
    return splits


def bootstrap(task, n_boot):
    Z, y, keep, _ = E.design(task)
    n = len(keep)
    uniq = E.unique_columns()
    rel = cfg.RELIABILITY[task]
    ref = cfg.FEATURE_NAMES.index(CT.REFERENCE)
    others = {"single": [k for k in range(Z.shape[1]) if k != ref],
              "unique": [k for k in uniq if k != ref]}
    rng = np.random.default_rng(cfg.SEED + 1)
    P = Z.shape[1]
    out = {f"{k}_{c}": np.full((n_boot, P), np.nan)
           for k in ("single", "unique") for c in cfg.CONDITIONS}
    out["contrast_single"] = np.zeros((n_boot, 3))
    out["contrast_unique"] = np.zeros((n_boot, 3))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        splits = bootstrap_splits(idx, rng)
        for ci, cond in enumerate(cfg.CONDITIONS):
            s, u = E.fold_scores(Z[idx], y[ci][idx], splits, uniq)
            out[f"single_{cond}"][b] = s.mean(0) / rel[cond]
            out[f"unique_{cond}"][b] = nan.mean(u) / rel[cond]
        for kind in ("single", "unique"):
            d = out[f"{kind}_preview"][b] - out[f"{kind}_no_preview"][b]
            o = d[others[kind]].mean()
            out[f"contrast_{kind}"][b] = (d[ref], o, d[ref] - o)
        if (b + 1) % 100 == 0:
            print(f"   {task}: {b + 1}/{n_boot} bootstrap resamples", flush=True)
    return out


def ci(a):
    lo, hi = np.nanpercentile(a, [2.5, 97.5], axis=0)
    return lo, hi


def encoding_tables(task, boot):
    res = dict(np.load(CACHE / f"{task}_encoding.npz", allow_pickle=True))
    n_img = len(res["images"])
    tables = {}
    for kind in ("single", "unique"):
        bkey = "beta_single" if kind == "single" else "beta_full"
        delta_boot = boot[f"{kind}_preview"] - boot[f"{kind}_no_preview"]
        dlo, dhi = ci(delta_boot)
        rows = []
        for k, label in enumerate(cfg.FEATURE_LABELS):
            if np.isnan(res[f"{kind}_preview"][k]):
                continue                        # not in the unique model
            row = dict(task=task, feature=label, n_images=n_img)
            for cond in cfg.CONDITIONS:
                lo, hi = ci(boot[f"{kind}_{cond}"][:, k])
                p = res[f"p_{kind}_{cond}"][k]
                row.update({
                    f"{cond}_beta": res[f"{bkey}_{cond}"][k],
                    f"{cond}_beta_sd": res[f"{bkey}_sd_{cond}"][k],
                    f"{cond}_r2": res[f"{kind}_{cond}"][k],
                    f"{cond}_r2_ci95_low": lo, f"{cond}_r2_ci95_high": hi,
                    f"{cond}_p": p, f"{cond}_p_bonferroni": min(1.0, p * N_BONF)})
            row.update(delta=res[f"delta_{kind}"][k], delta_ci95_low=dlo[k],
                       delta_ci95_high=dhi[k], delta_p=res[f"p_delta_{kind}"][k])
            rows.append(row)
        tables[kind] = pd.DataFrame(rows)
    return tables


def contrast_table(task, boot):
    c = np.load(CACHE / f"{task}_contrast.npz", allow_pickle=True)
    labels = list(c["statistics"])
    pick = {"face expectation delta > 0": (0, "Face expectation: preview - no preview"),
            "mean delta of others < 0": (1, "Mean of other features: preview - no preview"),
            "contrast vs mean of others": (2, "Contrast: face expectation - mean of others")}
    rows = []
    for kind in ("single", "unique"):
        lo, hi = ci(boot[f"contrast_{kind}"])
        for name, (j, label) in pick.items():
            s = labels.index(name)
            rows.append(dict(task=task, model=kind, statistic=label,
                             estimate=c[f"observed_{kind}"][s],
                             ci95_low=lo[j], ci95_high=hi[j],
                             p_one_sided=c[f"p_{kind}"][s]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 30,
                  "display.float_format", "{:.3f}".format)

    corr = feature_correlations()
    write_table(corr, "feature_correlations")
    print(corr.to_string(index=False), "\n")

    single, unique, contrasts = [], [], []
    for task in cfg.TASKS:
        boot = bootstrap(task, a.bootstrap)
        t = encoding_tables(task, boot)
        single.append(t["single"]); unique.append(t["unique"])
        contrasts.append(contrast_table(task, boot))
    for name, frames in (("encoding_single", single), ("encoding_unique", unique),
                         ("encoding_contrasts", contrasts)):
        df = pd.concat(frames, ignore_index=True)
        write_table(df, name)
        print(name); print(df.to_string(index=False), "\n")

    rows = []
    for task in cfg.TASKS:
        rows += MASK.run(task)
    write_table(pd.DataFrame(rows), "masking_anova")
    print(f"tables written to {OUT}")


if __name__ == "__main__":
    main()
