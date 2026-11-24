#!/usr/bin/env python3
"""
Is face expectation modulated by scene preview differently
from the remaining features?

The hypothesis here is: face expectation, and not the low-level
or geometric features, should matter more once the scene has been previewed:

    contrast = delta(face expectation) - mean(delta of the other features)

where delta is the preview minus no-preview difference in predictivity.

A positive contrast shows that preview affects face expectation differently from
the rest, but it does not distinguish a difference in SIGN from a difference in
MAGNITUDE: every feature could be reduced by preview, with face expectation
simply reduced least. The two component terms are therefore also tested
directionally -- is face expectation's own delta positive, and is the mean delta
of the others negative -- so that a sign reversal can be told apart from a
difference in degree.

Null distribution
-----------------
The same condition-swap as elsewhere: the two condition means are exchanged for
a random subset of images and the pipeline is re-run. Both permuted
pseudo-conditions are noise-corrected by the AVERAGE of the two reliabilities.
After permutation they are statistically equivalent mixtures, so giving them
different reliabilities would displace the null from zero.

This is preferable to simply recentring the null, because recentring is valid
only where the H0 expectation is zero. It is for "vs mean of others", but not
for "vs max of others": the maximum of eight or nine noisy deltas is upward
biased, so reference-minus-max is genuinely negative under H0, and that offset
is real structure the permutation must keep. Equalising the reliabilities
removes the artefact while preserving it.

Run: python3 run_contrast_test.py [--permutations 5000]
"""

import argparse

import numpy as np

from facescenes import config as cfg
from facescenes import encoding as E
from facescenes import nan
from facescenes import stats as S

CACHE = cfg.ROOT / "derivatives" / "encoding"
REFERENCE = "FaceExpect"
# (label, tail) -- the direction each statistic is predicted to take.
STATISTICS = [("contrast vs mean of others", "greater"),
              ("contrast vs max of others", "greater"),
              ("face expectation delta > 0", "greater"),
              ("mean delta of others < 0", "less")]


def deltas(Z, y, splits, uniq, rel_preview, rel_no_preview):
    s0, u0 = E.fold_scores(Z, y[0], splits, uniq)
    s1, u1 = E.fold_scores(Z, y[1], splits, uniq)
    return (s0.mean(0) / rel_preview - s1.mean(0) / rel_no_preview,
            nan.mean(u0) / rel_preview - nan.mean(u1) / rel_no_preview)


def statistics(d, ref, others):
    v = d[others]
    return np.array([d[ref] - v.mean(), d[ref] - v.max(), d[ref], v.mean()])


def run(task, n_perm):
    Z, y, keep, splits = E.design(task)
    n_img = len(keep)
    uniq = E.unique_columns()
    rel = cfg.RELIABILITY[task]
    ref = cfg.FEATURE_NAMES.index(REFERENCE)
    others = {"single": [k for k in range(Z.shape[1]) if k != ref],
              "unique": [k for k in uniq if k != ref]}

    obs_s, obs_u = deltas(Z, y, splits, uniq, rel["preview"], rel["no_preview"])
    observed = {"single": statistics(obs_s, ref, others["single"]),
                "unique": statistics(obs_u, ref, others["unique"])}

    # equalised reliability -> the null is centred by construction
    rel_mix = (rel["preview"] + rel["no_preview"]) / 2
    rng = np.random.default_rng(cfg.SEED)
    null = {k: np.zeros((n_perm, len(STATISTICS))) for k in observed}
    for i in range(n_perm):
        flip = rng.random(n_img) < 0.5
        yp = np.stack([np.where(flip, y[1], y[0]), np.where(flip, y[0], y[1])])
        ds, du = deltas(Z, yp, splits, uniq, rel_mix, rel_mix)
        null["single"][i] = statistics(ds, ref, others["single"])
        null["unique"][i] = statistics(du, ref, others["unique"])
        if (i + 1) % 500 == 0:
            print(f"   {task}: {i + 1}/{n_perm}", flush=True)

    print("=" * 84)
    print(f"{cfg.TASK_LABEL[task]}   {n_img} images, {n_perm} permutations")
    print("=" * 84)
    for kind, d in (("single-feature", obs_s), ("unique", obs_u)):
        key = "single" if kind == "single-feature" else "unique"
        incl = [ref] + others[key]
        print(f"\n  {kind} preview-minus-no-preview delta")
        order = sorted(incl, key=lambda k: -d[k])
        for k in order:
            mark = "   <- reference" if k == ref else ""
            print(f"     {cfg.FEATURE_LABELS[k]:14s} {d[k]:+8.4f}{mark}")
        print(f"\n     {'statistic':28s} {'observed':>10s} {'null mean':>10s} {'p':>8s}")
        for j, (label, tail) in enumerate(STATISTICS):
            o = observed[key][j]
            nl = null[key][:, j]
            print(f"     {label:28s} {o:10.4f} {nl.mean():10.4f} "
                  f"{S.permutation_p(o, nl, tail):8.4f}")
    print()

    # Cached so that run_statistics_tables.py reports these exact p-values.
    out = {"statistics": np.array([s for s, _ in STATISTICS])}
    for key in observed:
        out[f"observed_{key}"] = observed[key]
        out[f"p_{key}"] = np.array([S.permutation_p(observed[key][j], null[key][:, j], tail)
                                    for j, (_, tail) in enumerate(STATISTICS)])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"{task}_contrast.npz", **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=5000)
    a = ap.parse_args()
    for task in cfg.TASKS:
        run(task, a.permutations)


if __name__ == "__main__":
    main()
