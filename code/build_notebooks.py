#!/usr/bin/env python3
"""
Generate the figure notebooks in ../notebooks.

The notebooks are written from here rather than by hand so that the analysis
code they contain cannot drift from the facescenes package: each one imports the
package and calls it, and this script only assembles the cells.

Run: python3 build_notebooks.py
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks"

SETUP = '''import sys
from pathlib import Path

# The notebooks live next to the code, so the package is importable without
# installing anything. Nothing outside this release is referenced.
sys.path.insert(0, str(Path.cwd().parent / "code"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from facescenes import config as cfg
from facescenes import data as D
from facescenes import behaviour as B
from facescenes import encoding as E
from facescenes import onset_gaze as G
from facescenes import plotting as P
from facescenes import stats as S

P.use_style()
print("stimuli    :", len(D.stimulus_names()))
for t in cfg.TASKS:
    print(f"{t:14s}: {len(D.participant_ids(t))} participants, "
          f"{len(D.analysable_images(t))} images")'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


def notebook(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5}


# --------------------------------------------------------------- Figure 1
FIG1 = [
    md("""# Figure 1 — Scene previews speed face detection

**b, c** Detection latency per participant, preview versus no preview.
**d, e** Per-image mean latency in the two conditions.

Detection latency is the onset of the last fixation of the trial, measured from
search-display onset; the display was gaze-contingent and ended when the observer
fixated the face. Fixations beginning within 100 ms of onset are discarded and
timeout trials are dropped. Panels b and c show raw means; the statistics are
computed on log latency, because latencies are right-skewed and multiplicative.

Panels d and e regress the **paired** per-image means, so a point's x and y
coordinates come from the same scene. Note that independently sorting the two
vectors would turn this into a quantile-quantile plot and break the pairing."""),
    code(SETUP),
    md("## b, c — detection latency per participant"),
    code('''def panel_latency(task, fname, step, resume=None, ylim=None):
    df = B.trial_table(task)
    a, b = B.participant_means(df, "latency_ms")
    r = B.detection_latency(task)

    lo = 0 if resume is None else resume
    hi = ylim if ylim else np.ceil(max(np.nanmax(a), np.nanmax(b)) / step) * step

    if resume is None:
        fig, ax = plt.subplots(figsize=(5, 7))
        stub = None
    else:
        # The axis has to start at zero, and the effect is a few ms on a mean of
        # ~200 ms, so the axis is broken: a stub carrying the zero tick, the
        # break, then the data range from `resume` up.
        fig, (ax, stub) = plt.subplots(2, 1, sharex=True, figsize=(5, 7),
                                       gridspec_kw=dict(height_ratios=[18, 1],
                                                        hspace=.06))

    for a_ in ([ax] if stub is None else [ax, stub]):
        for i in range(len(a)):
            a_.plot([0, 1], [a[i], b[i]], color="grey", lw=.6, alpha=.5, zorder=1)
        box = a_.boxplot([a, b], positions=[0, 1], widths=.5, patch_artist=True,
                         showfliers=False, zorder=2,
                         medianprops=dict(color="k", lw=2))
        for patch, cond in zip(box["boxes"], cfg.CONDITIONS):
            patch.set_facecolor(P.condition_colours()[cond])
            patch.set_edgecolor("k")
        P.clean(a_)

    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + step / 2, step))
    ax.set_xticks([0, 1])
    P.title(ax, cfg.TASK_LABEL[task], fontsize=P.FONT - 6, pad=12)
    ax.text(.5, .96, P.significance(r["p"]), ha="center", va="top",
            transform=ax.transAxes, fontsize=P.FONT - 6)
    ax.set_ylabel("Detection latency (ms)", fontsize=P.FONT - 8)
    labels = [cfg.COND_LABEL[c] for c in cfg.CONDITIONS]

    if stub is None:
        ax.set_xticklabels(labels, fontsize=P.FONT - 8)
    else:
        stub.set_ylim(0, lo * .04)          # a sliver of the axis, so 0 is shown
        stub.set_yticks([0])
        stub.set_xticklabels(labels, fontsize=P.FONT - 8)
        ax.spines["bottom"].set_visible(False)
        stub.spines["top"].set_visible(False)
        ax.tick_params(axis="x", bottom=False)
        P.break_marker(ax, stub)
    src = B.participant_table(df, "latency_ms").rename(
        columns=lambda c: f"{c}_mean_latency_ms").reset_index()
    P.save(fig, fname, src); plt.show()

    print(f"{cfg.TASK_LABEL[task]}")
    print(f"  raw means       {r['mean_a']:.1f} vs {r['mean_b']:.1f} ms")
    print(f"  geometric means {r['geo_a']:.1f} vs {r['geo_b']:.1f} ms "
          f"({r['pct_change']:+.2f} %)")
    print(f"  test on log:    t({r['n']-1}) = {r['t']:.3f}, p = {r['p']:.4g}, "
          f"dz = {r['dz']:.2f}, 95% CI [{r['dz_ci'][0]:.2f}, {r['dz_ci'][1]:.2f}]")


panel_latency("facedetection", "fig1b_latency_face_detection.png",
              step=20, resume=170, ylim=300)'''),
    code('panel_latency("movingwindow", "fig1c_latency_moving_window.png", step=1000)'),
    md("""## d, e — per-image latency, preview against no preview

Each point is one image's **geometric mean** latency in the two conditions: the
mean of the log latencies, the same per-image quantity the encoding models
predict and the reliabilities are computed on. The axes are logarithmic and the
line is a least-squares fit in log space, so a slope below 1 means preview
shortens slow images by a larger *proportion* than fast ones.

R² is compared with a ceiling equal to the product of the two conditions'
Pearson split-half reliabilities on this same scale (config.R2_CEILING).

The slope is tested against 1. Because no-preview latency is itself measured with
error, least squares attenuates the slope: a *true* slope of 1 would be observed
as roughly the reliability of the x variable. The disattenuated slope and that
benchmark are both reported, so the compression is not read off an artefact."""),
    code('''def panel_scatter(task, fname, ticks):
    lm = D.image_means(task, log=True)          # mean of log seconds
    keep = D.analysable_images(task)
    lx = lm[1, keep] + np.log(1000)              # no preview, log ms
    ly = lm[0, keep] + np.log(1000)              # preview, log ms
    x, y = np.exp(lx), np.exp(ly)                # geometric means, ms

    lr = stats.linregress(lx, ly)
    rng = np.random.default_rng(0)
    boot = np.array([stats.linregress(*(lambda i: (lx[i], ly[i]))(
        rng.integers(0, len(lx), len(lx)))).slope for _ in range(10000)])
    ci = np.percentile(boot, [2.5, 97.5])
    rel_x = cfg.RELIABILITY_PEARSON[task]["no_preview"]

    lo, hi = ticks[0], ticks[-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([lo, hi], [lo, hi], ":", color="grey", lw=2, label="Identity (slope = 1)")
    ax.scatter(x, y, s=25, color="k", alpha=.55)
    gx = np.array([lo, hi])
    ax.plot(gx, np.exp(lr.intercept + lr.slope * np.log(gx)),
            color=cfg.C_PREVIEW, lw=2, label=f"Fit (slope = {lr.slope:.2f})")
    P.clean(ax)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks]); ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.tick_params(labelsize=P.FONT - 12)
    ax.set_xlabel("No-preview latency (ms)", fontsize=P.FONT - 8)
    ax.set_ylabel("Preview latency (ms)", fontsize=P.FONT - 8)
    P.title(ax, cfg.TASK_LABEL[task], fontsize=P.FONT - 6, pad=12)
    ax.legend(fontsize=P.FONT - 14, frameon=False, loc="upper left")
    src = pd.DataFrame({"image": np.asarray(D.stimulus_names())[keep],
                        "preview_geometric_mean_ms": y,
                        "no_preview_geometric_mean_ms": x})
    P.save(fig, fname, src); plt.show()

    print(f"{cfg.TASK_LABEL[task]}   n = {len(x)} images, log-scale geometric means")
    print(f"  R2        {lr.rvalue ** 2:.3f}      ceiling {cfg.R2_CEILING[task]:.3f}")
    print(f"  slope     {lr.slope:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  p(slope >= 1)                    {S.permutation_p(0, boot - 1, 'greater'):.4g}")
    print(f"  attenuation benchmark            {rel_x:.3f}  "
          f"(a true slope of 1 reads as this)")
    print(f"  p(slope >= benchmark)            "
          f"{S.permutation_p(0, boot - rel_x, 'greater'):.4g}")
    print(f"  disattenuated slope              {lr.slope / rel_x:.3f}")
    print(f"  images below the identity line   {100 * np.mean(y < x):.0f} %")


panel_scatter("facedetection", "fig1d_scatter_face_detection.png",
              ticks=[150, 200, 300, 450])'''),
    code('panel_scatter("movingwindow", "fig1e_scatter_moving_window.png",\n'
         '              ticks=[400, 1000, 2000, 4000, 6000])'),
]

# --------------------------------------------------------------- Figure 3
FIG3 = [
    md("""# Figure 3 — Encoding models

**a, b** Predictivity of single-feature models, preview versus no preview.
**c, d** Unique predictivity: the drop when one feature is removed from the full
model.

Predictivity is the signed squared Spearman correlation between predicted and
observed held-out log latency, divided by the split-half reliability of that
condition. It is signed because a cross-validated model can do worse than
chance, so negative values are meaningful and are plotted as such. It is not the
coefficient of determination of a fitted model, which is why the axis says
predictivity rather than explained variance.

Single-feature models use all ten predictors. The unique model uses nine: the
untrained CNN is omitted because it is a near-redundant operationalization of
the same quantity as SceneFaceDet (Spearman r = .91), and including both splits
the shared contribution between them.

Error bars are the SD across the 100 folds and are descriptive only — folds are
overlapping resamples of the same images, so all inference is at the image level.
The results are read from the released cache in data/derivatives/encoding/,
or from derivatives/encoding/ after a rerun with
`python3 ../code/run_encoding_analyses.py --force`.

Each table below gives, per condition, predictivity ± SD and `p>0` — the
one-sided permutation test in which the latencies are shuffled across images and
the whole cross-validation is rerun, so the null covers model fitting as well as
scoring — followed by the preview-minus-no-preview difference and its two-sided
test.

Every p is Bonferroni-corrected in the `bonf` column beside it, over its own
family: the >0 tests by 20 (ten features x two conditions) for both models, the
family used in the text and in Supplementary Tables 2 and 3, and the delta tests
by one per feature (10 and 9). Correcting the deltas by the larger factor would
count each feature twice."""),
    code(SETUP),
    code('''import run_encoding_analyses as RUN

RES = {t: RUN.analyse(t, n_perm=5000) for t in cfg.TASKS}
print({t: len(r["images"]) for t, r in RES.items()}, "images per task")'''),
    code('''def panel(task, kind, fname, alpha=None):
    res = RES[task]
    labels, prev, nop, sd_p, sd_n, pv = [], [], [], [], [], []
    p0_p, p0_n = [], []
    for k, name in enumerate(cfg.FEATURE_NAMES):
        if np.isnan(res[f"{kind}_preview"][k]):
            continue                        # not in the unique model
        labels.append(cfg.FEATURE_LABELS[k])
        prev.append(res[f"{kind}_preview"][k])
        nop.append(res[f"{kind}_no_preview"][k])
        sd_p.append(res[f"{kind}_sd_preview"][k])
        sd_n.append(res[f"{kind}_sd_no_preview"][k])
        pv.append(res[f"p_delta_{kind}"][k])
        # Predictivity > 0: the latencies are shuffled across images and the
        # whole cross-validation is rerun, so the null covers fitting as well as
        # scoring. One-sided -- predicting worse than chance is not of interest.
        p0_p.append(res[f"p_{kind}_preview"][k])
        p0_n.append(res[f"p_{kind}_no_preview"][k])
    prev, nop = np.array(prev), np.array(nop)
    sd_p, sd_n = np.array(sd_p), np.array(sd_n)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.bar(x - .2, prev, .4, color=cfg.C_PREVIEW, label="Preview")
    ax.errorbar(x - .2, prev, yerr=P.one_sided_errorbar(prev, sd_p), fmt="None",
                color="k", lw=1.2)
    ax.bar(x + .2, nop, .4, color=cfg.C_NO_PREVIEW, label="No preview")
    ax.errorbar(x + .2, nop, yerr=P.one_sided_errorbar(nop, sd_n), fmt="None",
                color="k", lw=1.2)
    ax.axhline(0, color="k", lw=.8)

    lo = min(0, np.min(np.minimum(prev - sd_p, nop - sd_n)))
    hi = np.max(np.maximum(prev + sd_p, nop + sd_n)) * 1.18
    ticks, (t0, t1) = P.zero_anchored_ticks(lo, hi)
    ax.set_ylim(t0, t1); ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks], fontsize=P.FONT - 6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=P.FONT - 6)
    ax.set_ylabel(P.PREDICTIVITY_LABEL if kind == "single" else P.UNIQUE_LABEL,
                  fontsize=P.FONT - 4)
    P.title(ax, cfg.TASK_LABEL[task], fontsize=P.FONT - 2, pad=14)
    if (task, kind) == ("facedetection", "single"):   # one legend serves all panels
        ax.legend(fontsize=P.FONT - 8, frameon=False, loc="upper left")
    P.clean(ax)
    # Predictivity > 0, one-sided, Bonferroni x 20 (ten features x two
    # conditions) for both models, the family used in the text and in
    # Supplementary Tables 2 and 3.
    n_fam = len(cfg.FEATURE_NAMES) * 2
    for xs, vals, sds, ps in ((x - .2, prev, sd_p, p0_p), (x + .2, nop, sd_n, p0_n)):
        for xi, v, s, p in zip(xs, vals, sds, ps):
            # One level only: with 5000 permutations the smallest corrected
            # p is 20/5001 = .004, so finer star levels would be misleading.
            if p * n_fam < .05:
                ax.text(xi, max(v, 0) + s + .01 * (t1 - t0), "*", ha="center",
                        va="bottom", fontsize=P.FONT - 8)
    if alpha is not None:
        for i, p in enumerate(pv):
            if p < alpha:
                top = max(prev[i] + sd_p[i], nop[i] + sd_n[i])
                ax.text(x[i], top + .02 * (t1 - t0), "*", ha="center",
                        va="bottom", fontsize=P.FONT - 4)
    src = pd.DataFrame({
        "feature": labels,
        "preview_r2": prev, "preview_sd_across_folds": sd_p,
        "preview_p": p0_p, "preview_p_bonferroni": np.minimum(1, np.array(p0_p) * n_fam),
        "no_preview_r2": nop, "no_preview_sd_across_folds": sd_n,
        "no_preview_p": p0_n,
        "no_preview_p_bonferroni": np.minimum(1, np.array(p0_n) * n_fam)})
    P.save(fig, fname, src); plt.show()

    # Two families, two factors: the >0 tests are one per feature x condition,
    # the delta tests one per feature.
    n_p0, n_delta = len(cfg.FEATURE_NAMES) * 2, len(labels)
    print(f"{cfg.TASK_LABEL[task]}  —  {kind} predictivity")
    print(f"  {'feature':14s} {'preview':>16s} {'p>0':>7s} {'bonf':>6s}  "
          f"{'no preview':>16s} {'p>0':>7s} {'bonf':>6s}  "
          f"{'delta':>8s} {'p':>7s} {'bonf':>6s}")
    for i, lab in enumerate(labels):
        print(f"  {lab:14s} {prev[i]:+8.3f}+/-{sd_p[i]:.3f} "
              f"{p0_p[i]:7.4f} {min(1, p0_p[i]*n_p0):6.3f}  "
              f"{nop[i]:+8.3f}+/-{sd_n[i]:.3f} "
              f"{p0_n[i]:7.4f} {min(1, p0_n[i]*n_p0):6.3f}  "
              f"{prev[i]-nop[i]:+8.3f} {pv[i]:7.4f} {min(1, pv[i]*n_delta):6.3f}")
    print(f"  p>0 one-sided against shuffled images, Bonferroni x{n_p0} "
          f"({n_delta} features x 2 conditions); delta two-sided, "
          f"Bonferroni x{n_delta} (one test per feature); "
          f"smallest attainable p {1/5001:.1e}")


panel("facedetection", "single", "fig3a_single_face_detection.png")'''),
    code('panel("movingwindow", "single", "fig3b_single_moving_window.png")'),
    code('panel("facedetection", "unique", "fig3c_unique_face_detection.png")'),
    code('panel("movingwindow", "unique", "fig3d_unique_moving_window.png")'),
    md("""## Standardised regression coefficients

Predictors and the dependent variable are z-scored within each training fold, so
β is the change in log latency in SD units per 1 SD of the predictor. A negative
β means the feature predicts *faster* detection. β is not noise-corrected: the
reliability correction rescales predictivity, it does not apply to a slope."""),
    code('''for task in cfg.TASKS:
    res = RES[task]
    print(cfg.TASK_LABEL[task])
    print(f"  {'feature':14s} {'single-feature model':>26s} {'full 9-feature model':>26s}")
    print(f"  {'':14s} {'preview':>12s} {'no prev':>13s} {'preview':>12s} {'no prev':>13s}")
    for k, lab in enumerate(cfg.FEATURE_LABELS):
        f_p, f_n = res["beta_full_preview"][k], res["beta_full_no_preview"][k]
        full = ("        --      --   " if np.isnan(f_p)
                else f"{f_p:+6.3f}+/-{res['beta_full_sd_preview'][k]:.3f} "
                     f"{f_n:+6.3f}+/-{res['beta_full_sd_no_preview'][k]:.3f}")
        print(f"  {lab:14s} "
              f"{res['beta_single_preview'][k]:+6.3f}+/-{res['beta_single_sd_preview'][k]:.3f} "
              f"{res['beta_single_no_preview'][k]:+6.3f}+/-{res['beta_single_sd_no_preview'][k]:.3f} "
              f"{full}")
    print()'''),
]


# ---------------------------------------- Supplementary: raw predictivity
SUPP_ENCODING = [
    md("""# Supplementary Figure 4 — single-feature models WITHOUT noise correction

Same models, folds, features and inference as Figure 3a/b. The only difference is
that predictivity is *not* divided by each condition's split-half reliability, so
these bars are what the models actually achieve against noisy image means, and the
ceiling is the reliability rather than 1. The unique-contribution analysis is not
repeated: the point here is the scale of the raw scores, and it is the same point
in both model families.

Nothing is recomputed. The correction is one positive constant per condition, so
multiplying the cached corrected values back by it recovers the raw scores and
their fold SDs exactly:

| task | preview | no preview |
|---|---|---|
| face detection | x .739 | x .859 |
| moving window | x .824 | x .862 |

**The p-values.** `p>0` is *identical* with and without the correction: it scales
the observed statistic and its permutation null by the same constant, so the
comparison is unchanged. The preview-minus-no-preview `delta` is different — the
two conditions scale by different constants, so the uncorrected delta strictly has
its own null — and the p carried over here is the corrected one, marked `p*`. No
feature's delta survives correction for multiple comparisons under either version,
so the distinction changes no conclusion.

Standardised betas are not repeated either: they are never noise-corrected, so the
values in Figure 3 already are the uncorrected ones."""),
    code(SETUP),
    code('''import run_encoding_analyses as RUN

RES = {t: RUN.analyse(t, n_perm=5000) for t in cfg.TASKS}


def raw_panel(task, fname):
    res = RES[task]
    labels = cfg.FEATURE_LABELS
    rel = cfg.RELIABILITY[task]
    # Undo the correction: predictivity and its fold SD were both divided by rel.
    prev = res["single_preview"] * rel["preview"]
    nop = res["single_no_preview"] * rel["no_preview"]
    sd_p = res["single_sd_preview"] * rel["preview"]
    sd_n = res["single_sd_no_preview"] * rel["no_preview"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.bar(x - .2, prev, .4, color=cfg.C_PREVIEW, label="Preview")
    ax.errorbar(x - .2, prev, yerr=P.one_sided_errorbar(prev, sd_p), fmt="None",
                color="k", lw=1.2)
    ax.bar(x + .2, nop, .4, color=cfg.C_NO_PREVIEW, label="No preview")
    ax.errorbar(x + .2, nop, yerr=P.one_sided_errorbar(nop, sd_n), fmt="None",
                color="k", lw=1.2)
    ax.axhline(0, color="k", lw=.8)

    lo = min(0, np.min(np.minimum(prev - sd_p, nop - sd_n)))
    hi = np.max(np.maximum(prev + sd_p, nop + sd_n)) * 1.18
    ticks, (t0, t1) = P.zero_anchored_ticks(lo, hi)
    ax.set_ylim(t0, t1); ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks], fontsize=P.FONT - 6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=P.FONT - 6)
    # The one uncorrected predictivity axis in the release.
    ax.set_ylabel(P.RAW_PREDICTIVITY_LABEL, fontsize=P.FONT - 4)
    P.title(ax, cfg.TASK_LABEL[task], fontsize=P.FONT - 2, pad=14)
    if task == "facedetection":                       # one legend serves both panels
        ax.legend(fontsize=P.FONT - 8, frameon=False, loc="upper left")
    P.clean(ax)
    # Same asterisks as Figure 3a/b: p>0 is unchanged by the correction, so
    # one-sided, Bonferroni x 20, a single level (smallest corrected p is .004).
    n_fam = len(cfg.FEATURE_NAMES) * 2
    for xs, vals, sds, ps in ((x - .2, prev, sd_p, res["p_single_preview"]),
                              (x + .2, nop, sd_n, res["p_single_no_preview"])):
        for xi, v, s, p in zip(xs, vals, sds, ps):
            if p * n_fam < .05:
                ax.text(xi, max(v, 0) + s + .01 * (t1 - t0), "*", ha="center",
                        va="bottom", fontsize=P.FONT - 8)
    pp, pn = res["p_single_preview"], res["p_single_no_preview"]
    src = pd.DataFrame({
        "feature": labels,
        "preview_r2_uncorrected": prev, "preview_sd_across_folds": sd_p,
        "preview_p": pp, "preview_p_bonferroni": np.minimum(1, pp * n_fam),
        "no_preview_r2_uncorrected": nop, "no_preview_sd_across_folds": sd_n,
        "no_preview_p": pn, "no_preview_p_bonferroni": np.minimum(1, pn * n_fam)})
    P.save(fig, fname, src); plt.show()

    n_p0, n_delta = len(labels) * 2, len(labels)
    print(f"{cfg.TASK_LABEL[task]}  —  single-feature predictivity, NOT noise corrected")
    print(f"  {'feature':14s} {'preview':>16s} {'p>0':>7s} {'bonf':>6s}  "
          f"{'no preview':>16s} {'p>0':>7s} {'bonf':>6s}  "
          f"{'delta':>8s} {'p*':>7s} {'bonf':>6s}")
    for k, lab in enumerate(labels):
        p0p, p0n = res["p_single_preview"][k], res["p_single_no_preview"][k]
        p_d = res["p_delta_single"][k]
        print(f"  {lab:14s} {prev[k]:+8.3f}+/-{sd_p[k]:.3f} "
              f"{p0p:7.4f} {min(1, p0p*n_p0):6.3f}  "
              f"{nop[k]:+8.3f}+/-{sd_n[k]:.3f} "
              f"{p0n:7.4f} {min(1, p0n*n_p0):6.3f}  "
              f"{prev[k]-nop[k]:+8.3f} {p_d:7.4f} {min(1, p_d*n_delta):6.3f}")
    print(f"  ceiling here is the reliability, not 1: "
          f"preview {rel['preview']:.3f}, no preview {rel['no_preview']:.3f}")
    print("  p>0 identical to the corrected analysis; p* carried over from it")


raw_panel("facedetection", "suppfig4a_single_raw_face_detection.png")'''),
    code('raw_panel("movingwindow", "suppfig4b_single_raw_moving_window.png")'),
]


# ------------------------------------------------- Supplementary: behaviour
SUPP_BEHAVIOUR = [
    md("""# Supplementary Figures 1-3 — behavioural measures

**S1** Number of fixations needed to reach the face.
**S2** Detection latency on trials where the first fixation already landed on the
face, face detection task only.
**S3** Scan path ratio, moving-window task only.

Which measures are log-transformed is a deliberate choice. Latency and scan path
ratio are ratios or durations: bounded below by zero, unbounded above, and
multiplicative, so they are tested on logs. Fixation counts and percentages are
bounded and not multiplicative, so they are tested raw. Raw means are always
reported for description."""),
    code(SETUP),
    code('''def paired_panel(task, column, ylabel, fname, result, step=None, df=None,
                 resume=None):
    """One panel: per-participant means in the two conditions.

    `resume` breaks the y-axis, as in Figure 1b: a stub carrying the zero tick,
    the break, then the data range from `resume` up. Used where the effect is a
    few ms on a mean of ~190 ms and would otherwise be invisible.
    """
    if df is None:
        df = B.trial_table(task)
    a, b = B.participant_means(df, column)

    top = np.nanmax([a, b])
    if step is None:
        step = 10 ** np.floor(np.log10(top)) / 2
    # Headroom for the significance marker, except on a broken axis, where the
    # range above `resume` is already tall relative to the data.
    hi = np.ceil((top if resume else top * 1.12) / step) * step
    lo = 0 if resume is None else resume

    if resume is None:
        fig, ax = plt.subplots(figsize=(5, 7))
        stub = None
    else:
        fig, (ax, stub) = plt.subplots(2, 1, sharex=True, figsize=(5, 7),
                                       gridspec_kw=dict(height_ratios=[18, 1],
                                                        hspace=.06))

    for a_ in ([ax] if stub is None else [ax, stub]):
        for i in range(len(a)):
            a_.plot([0, 1], [a[i], b[i]], color="grey", lw=.6, alpha=.5, zorder=1)
        box = a_.boxplot([a, b], positions=[0, 1], widths=.5, patch_artist=True,
                         showfliers=False, zorder=2,
                         medianprops=dict(color="k", lw=2))
        for patch, cond in zip(box["boxes"], cfg.CONDITIONS):
            patch.set_facecolor(P.condition_colours()[cond]); patch.set_edgecolor("k")
        P.clean(a_)

    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + step / 2, step))
    ax.set_xticks([0, 1])
    P.title(ax, cfg.TASK_LABEL[task], fontsize=P.FONT - 6, pad=12)
    ax.text(.5, .96, P.significance(result["p"]), ha="center", va="top",
            transform=ax.transAxes, fontsize=P.FONT - 6)
    ax.set_ylabel(ylabel, fontsize=P.FONT - 8)
    labels = [cfg.COND_LABEL[c] for c in cfg.CONDITIONS]

    if stub is None:
        ax.set_xticklabels(labels, fontsize=P.FONT - 8)
    else:
        stub.set_ylim(0, lo * .04)          # a sliver of the axis, so 0 is shown
        stub.set_yticks([0])
        stub.set_xticklabels(labels, fontsize=P.FONT - 8)
        ax.spines["bottom"].set_visible(False)
        stub.spines["top"].set_visible(False)
        ax.tick_params(axis="x", bottom=False)
        P.break_marker(ax, stub)
    src = B.participant_table(df, column).rename(
        columns=lambda c: f"{c}_mean_{column}").reset_index()
    P.save(fig, fname, src); plt.show()

    r = result
    print(f"{cfg.TASK_LABEL[task]} — {ylabel}")
    print(f"  raw means  {r['mean_a']:.3f} vs {r['mean_b']:.3f}")
    if "geo_a" in r:
        print(f"  geometric  {r['geo_a']:.3f} vs {r['geo_b']:.3f} "
              f"({r['pct_change']:+.2f} %)")
    print(f"  t({r['n']-1}) = {r['t']:.3f}, p = {r['p']:.4g}, dz = {r['dz']:.2f}, "
          f"95% CI [{r['dz_ci'][0]:.2f}, {r['dz_ci'][1]:.2f}]")


for task, fname in (("facedetection", "suppfig1a_fixation_counts_face_detection.png"),
                    ("movingwindow", "suppfig1b_fixation_counts_moving_window.png")):
    paired_panel(task, "n_fixations", "Fixations to reach the face", fname,
                 B.fixation_count(task), step=1 if task == "facedetection" else 2)'''),
    md("""## S2 — detection latency on trials reached in a single fixation

Trials where the first fixation beginning more than 100 ms after scene onset
already landed on the face, so the search ended on one saccade. What is plotted
is the LATENCY of that fixation, not how often it happened; the hit rate is
printed alongside. Face detection task only: in the moving-window task only 5.9 %
of trials end this way, and ten participants have no such trial in one of the two
conditions, so the comparison rests on 28 of 38 participants.

Display is on the raw scale, the test on logs, as for detection latency in
Figure 1."""),
    code('''task = "facedetection"
ff = B.first_fixation_on_face(task)
print(f"{cfg.TASK_LABEL[task]}: {ff['n_hit']}/{ff['n_total']} trials "
      f"({ff['pct_overall']:.1f} %) ended on the first fixation")

df = B.trial_table(task)
hits = df[df["n_fixations"] == 1]
paired_panel(task, "latency_ms",
             "Detection latency with a\\nsingle fixation (ms)",
             "suppfig2_first_fixation_latency_face_detection.png",
             ff["latency"], step=40, resume=160, df=hits)'''),
    md("""## S3 — scan path ratio

Total distance travelled between successive fixations, divided by the
straight-line distance from the centre to the face. A value of 1 means the
observer went straight there. Reported for the moving-window task only: in the
face detection task observers reach the face in roughly one fixation, so there is
barely a path to measure."""),
    code('''paired_panel("movingwindow", "scan_path_ratio", "Scan path ratio",
             "suppfig3_scanpath_ratio_moving_window.png",
             B.scan_path_ratio("movingwindow"), step=1)'''),
]

# ------------------------------------------ Supplementary: CNN layers, masks
SUPP_CNN = [
    md("""# Supplementary Figures 5 and 6 — the CNN predictors

**S5** Predictivity across the seven read-out layers of each network.
**S6** What happens when the face is occluded before the CNN feature is computed.

The CNN feature is the representational distance between a scene with a face and
the same scene without one, so it asks how much the face changes the network's
response. The main analyses use the final layer.

The untrained network is a randomly initialised copy of the same architecture. It
is genuinely untrained: its face/no-face distance *contracts* with depth, as
random weights progressively wash out the input difference, whereas both trained
networks amplify it."""),
    code(SETUP),
    code('''# The layer-depth correlations quoted in the text: python3 run_layer_depth.py
N_LAYERS = E.N_CNN_LAYERS
NETWORKS = {"SceneFaceDet": "cnn_scenefacedet.npz", "UntrainedCNN": "cnn_untrained.npz"}

for task, fname in (("facedetection", "suppfig5a_layers_face_detection.png"),
                    ("movingwindow", "suppfig5b_layers_moving_window.png")):
    fig, ax = plt.subplots(figsize=(8, 6))
    rows = []
    for net, npz in NETWORKS.items():
        r = E.layer_predictivity(task, npz)
        for cond in cfg.CONDITIONS:
            rows += [dict(network=net, condition=cond, layer=k + 1, r2=v)
                     for k, v in enumerate(r[cond])]
            ax.plot(range(1, N_LAYERS + 1), r[cond],
                    color=P.condition_colours()[cond],
                    linestyle="-" if net == "SceneFaceDet" else ":",
                    marker="o", lw=2.5, ms=6,
                    label=f"{net}, {cfg.COND_LABEL[cond].lower()}")
    # Fixed range so the two panels match; fail loudly rather than clip a value.
    vals = [row["r2"] for row in rows]
    assert 0 <= min(vals) and max(vals) <= 0.6, (min(vals), max(vals))
    ticks, (t0, t1) = P.zero_anchored_ticks(0, 0.6, step=0.1)
    ax.set_ylim(0, 0.6); ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks], fontsize=P.FONT - 10)
    ax.set_xticks(range(1, N_LAYERS + 1))
    ax.tick_params(labelsize=P.FONT - 10)
    ax.set_xlabel("CNN layer", fontsize=P.FONT - 8)
    ax.set_ylabel(P.PREDICTIVITY_LABEL, fontsize=P.FONT - 8)
    P.title(ax, cfg.TASK_LABEL[task], fontsize=P.FONT - 6, pad=12)
    if task == "facedetection":
        # The two panels share these four curves, so the legend is drawn once,
        # inside the axes in the empty band above the lines.
        ax.legend(fontsize=P.FONT - 16, frameon=False, ncol=2, loc="upper center")
    P.clean(ax)
    P.save(fig, fname, pd.DataFrame(rows)); plt.show()'''),
    md("""## S6 — face visible against face masked

Predictivity is defined over a set of images, so each cell has a single value and
there are no replicates for a factorial test. Squared prediction error does
decompose by image, so images become the replicates of a 2 (preview) x 2 (mask)
repeated-measures ANOVA. The bracket over each network is the main effect of
masking. See `run_masked_face_analyses.py` for why an earlier permutation
approach was invalid.

Error bars are the SD across the 100 folds and are descriptive only, as in
Figure 3: the folds are overlapping resamples of the same images, so their spread
reflects resampling noise rather than sampling variability, and the inference is
the image-level ANOVA."""),
    code('''import run_masked_face_analyses as MASK

# The main effect of masking is reported in the text and printed below each
# panel; set this to True to draw it on the figure as a bracket per network.
SHOW_SIGNIFICANCE = False

EFFECTS = {"A": "preview", "B": "mask", "AxB": "preview x mask"}


def anova_line(row):
    """F(df1,df2) = ..., p = ..., eta2p = ... for one effect of anova_2x2."""
    F, df1, df2, p, eta = row
    ps = "p < .001" if p < .001 else f"p = {f'{p:.3f}'[1:]}"
    return f"F({df1},{df2}) = {F:.2f}, {ps}, \\u03b7\\u00b2\\u209a = {f'{eta:.3f}'[1:]}"

for task, fname in (("facedetection", "suppfig6a_masked_face_detection.png"),
                    ("movingwindow", "suppfig6b_masked_moving_window.png")):
    Z, y, keep, splits = E.design(task)
    n_img = len(keep)
    masked = E.masked_features(task)
    yz = np.stack([(row - np.nanmean(row)) / np.nanstd(row, ddof=1) for row in y])

    nets = list(MASK.NETWORKS)
    W, G, bars = .18, .05, []      # bar width; gap between the two condition pairs
    fig, ax = plt.subplots(figsize=(8, 6))
    tops, pvals, sds, aov = [], {}, {}, {}
    for i, net in enumerate(nets):
        col = cfg.FEATURE_NAMES.index(net)
        zv = Z[:, [col]]
        m = masked[net]; zm = ((m - m.mean()) / m.std())[:, None]
        err = np.zeros((n_img, 2, 2))
        for ci, cond in enumerate(cfg.CONDITIONS):
            err[:, ci, 0] = S.per_image_squared_error(zv, yz[ci], splits, n_img)
            err[:, ci, 1] = S.per_image_squared_error(zm, yz[ci], splits, n_img)
            for mi, z in ((0, zv), (1, zm)):
                folds = np.array(
                    [E.signed_r2(y[ci][np.setdiff1d(np.arange(n_img), tr)],
                                 E.ols_predict(z[tr], y[ci][tr],
                                               z[np.setdiff1d(np.arange(n_img), tr)]))
                     for tr in splits]) / cfg.RELIABILITY[task][cond]
                v, sd = folds.mean(), folds.std(ddof=1)
                off = (-1.5 + 2 * ci + mi) * W + (ci - .5) * G
                ax.bar(i + off, v, W,
                       color="white" if mi else P.condition_colours()[cond],
                       edgecolor=P.condition_colours()[cond] if mi else "k",
                       hatch="...." if mi else None, lw=1.4 if mi else .8,
                       label=(f"{cfg.COND_LABEL[cond]}, face "
                              f"{'masked' if mi else 'visible'}") if i == 0 else None)
                # SD across folds, drawn away from zero only, as in Figure 3.
                ax.errorbar(i + off, v, yerr=P.one_sided_errorbar([v], [sd]),
                            fmt="None", color="k", lw=1.2)
                tops.append(v + sd)
                sds[(net, cond, mi)] = (v, sd)
        aov[net] = MASK.S.anova_2x2(np.log(err))
        pvals[net] = aov[net]["B"][3]

    # Reserved whether or not the brackets are drawn, so the two panels keep the
    # same y-axis either way.
    bar_y = max(tops) * 1.15
    if SHOW_SIGNIFICANCE:
        for i, net in enumerate(nets):
            a, b = i - 1.5 * W - G / 2, i + 1.5 * W + G / 2
            ax.plot([a, a, b, b], [bar_y, bar_y + .012, bar_y + .012, bar_y],
                    lw=1.1, c="k")
            ax.text(i, bar_y + .018, P.significance(pvals[net]), ha="center",
                    va="bottom", fontsize=P.FONT - 12)

    hi = np.ceil((bar_y + .07) * 10) / 10
    # A negative bar's error bar points down, so the axis has to reach it; where
    # it does, zero is drawn explicitly.
    below = [v - sd for v, sd in sds.values() if v < 0]
    lo = np.floor(min(below + [0]) * 10) / 10
    ticks = np.arange(lo, hi + .05, .1)
    if lo < 0:
        ax.axhline(0, color="k", lw=.8)
    ax.set_ylim(lo, hi); ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks], fontsize=P.FONT - 10)
    ax.set_xticks(range(len(nets)))
    ax.set_xticklabels([MASK.NETWORK_LABEL[n] for n in nets], fontsize=P.FONT - 10)
    ax.set_ylabel(P.PREDICTIVITY_LABEL, fontsize=P.FONT - 8)
    P.title(ax, cfg.TASK_LABEL[task], fontsize=P.FONT - 6, pad=12)
    if task == "facedetection":
        # Both panels show the same four bars, so the legend is drawn once,
        # inside the axes above them.
        ax.legend(fontsize=P.FONT - 18, frameon=False, ncol=2, loc="upper center")
    P.clean(ax)
    src = pd.DataFrame([dict(network=MASK.NETWORK_LABEL[net], condition=cond,
                             face="masked" if mi else "visible", r2=v,
                             sd_across_folds=sd)
                        for (net, cond, mi), (v, sd) in sds.items()])
    P.save(fig, fname, src); plt.show()
    print(f"{cfg.TASK_LABEL[task]}   {n_img} images")
    for net in nets:
        print(f"  {MASK.NETWORK_LABEL[net]}")
        for cond in cfg.CONDITIONS:
            v0, s0 = sds[(net, cond, 0)]
            v1, s1 = sds[(net, cond, 1)]
            print(f"    {cfg.COND_LABEL[cond]:11s} visible {v0:+.3f}+/-{s0:.3f}"
                  f"   masked {v1:+.3f}+/-{s1:.3f}")
        # Repeated-measures ANOVA on log squared error, images as replicates.
        for key, label in EFFECTS.items():
            print(f"    {label:14s} {anova_line(aov[net][key])}")'''),
]


# --------------------------------------- Supplementary: onset-gaze control
SUPP_ONSET_GAZE = [
    md("""# Supplementary Note 8 — gaze position at target onset

A control for the obvious alternative reading of Figure 1. If the preview lets
observers move their eyes toward the upcoming face BEFORE the search display
appears, they start closer to the target and reach it sooner, and the benefit is
oculomotor rather than a matter of scene guidance. Earlier controls could not
address this: they were all restricted to eye movements *following* target onset.

Eye position at onset is read from the fixation that was in progress when the
display appeared. Recording begins at display onset, so that fixation's start
time is clipped to the event and its mean position reflects the oculomotor state
established BEFORE the target — saccade latencies of at least ~150 ms rule out
any target-driven change within it. The estimate is event-based, not a
continuous trace: the 500 ms fixation-dot interval was not sampled, so this is
the only pre-target position available.

Two distances are computed from that position, in degrees of visual angle:
distance from the central fixation point, which asks whether the preview moves
the eye at all, and distance from the face centroid, which asks whether the
movement is toward the target. Only the second is the confound. A displacement
unrelated to the face buys no head start.

The analysis runs on all successful trials of both experiments. Distances are
pooled within participant by the MEDIAN, since a handful of trials begin with
the eye far off centre; latency regressions are on the raw millisecond scale
rather than logs, because the bound below multiplies degrees by ms/deg and needs
an additive scale."""),
    code(SETUP),
    md("""## S7 — is the eye displaced, and is it displaced toward the face?

Left panel: distance of onset gaze from the screen centre. Right panel: distance
from the face centroid. Each line is a participant."""),
    code('''def onset_panel(ax, df, column, ylabel, result):
    """Per-participant median distance in the two conditions."""
    a, b = B.participant_means(df, column, agg="median")
    for i in range(len(a)):
        ax.plot([0, 1], [a[i], b[i]], color="grey", lw=.6, alpha=.5, zorder=1)
    box = ax.boxplot([a, b], positions=[0, 1], widths=.5, patch_artist=True,
                     showfliers=False, zorder=2, medianprops=dict(color="k", lw=2))
    for patch, cond in zip(box["boxes"], cfg.CONDITIONS):
        patch.set_facecolor(P.condition_colours()[cond]); patch.set_edgecolor("k")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([cfg.COND_LABEL[c] for c in cfg.CONDITIONS],
                       fontsize=P.FONT - 14)
    ax.set_ylabel(ylabel, fontsize=P.FONT - 14)
    ax.set_ylim(0, np.nanmax([a, b]) * 1.15)
    ax.text(.5, .99, P.significance(result["p"]), ha="center", va="top",
            transform=ax.transAxes, fontsize=P.FONT - 12)
    P.clean(ax)


TABLES = {}      # reused by the cells below, so the table is built once per task

for task, fname in (("facedetection", "suppnote8a_onset_gaze_face_detection.png"),
                    ("movingwindow", "suppnote8b_onset_gaze_moving_window.png")):
    TABLES[task] = df = G.onset_gaze_table(task)
    disp = G.displacement(task, df)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5))
    onset_panel(axes[0], df, "d_centre_deg",
                "Distance from screen\\ncentre at onset (deg)", disp["d_centre_deg"])
    onset_panel(axes[1], df, "d_face_deg",
                "Distance from face\\ncentroid at onset (deg)", disp["d_face_deg"])
    P.title(fig, cfg.TASK_LABEL[task], fontsize=P.FONT - 6, y=1.02)
    fig.tight_layout()
    src = pd.concat(
        [B.participant_table(df, c, agg="median").rename(
            columns=lambda k, c=c: f"{k}_median_{c}") for c in ("d_centre_deg", "d_face_deg")],
        axis=1).reset_index()
    P.save(fig, fname, src); plt.show()

    dropped = len(D.trials(task)) - len(df)
    print(f"{cfg.TASK_LABEL[task]} — {len(df)} trials, "
          f"{dropped} dropped with no qualifying onset fixation "
          f"({100 * dropped / (len(df) + dropped):.1f} %)")
    for col, label in (("d_centre_deg", "from screen centre"),
                       ("d_face_deg", "to face centroid")):
        r = disp[col]
        print(f"  {label:19s} {r['mean_a']:5.2f} vs {r['mean_b']:5.2f} deg   "
              f"diff {r['diff']:+.2f} deg, 95% CI "
              f"[{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}], "
              f"t({r['n']-1}) = {r['t']:.2f}, p = {r['p']:.3g}, dz = {r['dz']:+.2f}")'''),
    md("""## Can the displacement account for the benefit?

Three questions in order.

**Does onset position matter at all?** The positive control. Latency is regressed
on preview condition and both distances, separately within each participant, and
the gaze-to-face slope is tested across participants. It must be positive, or the
confound has no mechanism to work through.

**How much of the benefit could it buy?** The most favourable end of the
confidence interval for the condition difference in gaze-to-face distance, times
that slope. Deliberately generous: it grants the confound the largest
displacement toward the face the data will tolerate.

**Does the benefit survive adjustment?** The preview coefficient with and without
the distances as covariates. All four estimates are regression coefficients from
the same within-participant model, so they are comparable with each other —
setting a median-based effect against a regression coefficient makes adjustment
appear to change an estimate when it has not."""),
    code('''for task in cfg.TASKS:
    df = TABLES[task]
    print(cfg.TASK_LABEL[task])

    for col, label in (("d_face_deg", "gaze-to-face"),
                       ("d_centre_deg", "gaze-to-centre")):
        s = G.distance_slope(task, col, df)
        print(f"  latency on {label:15s} {s['mean']:+8.2f} ms/deg   "
              f"t({s['n']-1}) = {s['t']:.2f}, p = {s['p']:.3g}")

    b = G.orienting_bound(task, df)
    print(f"  maximum attributable to orienting: {b['ms']:.1f} ms "
          f"({b['degrees']:+.2f} deg x {b['ms_per_degree']:.1f} ms/deg)")

    for covs, label in (((), "unadjusted"),
                        (("d_face_deg",), "+ gaze-to-face"),
                        (("d_centre_deg",), "+ gaze-to-centre"),
                        (("d_face_deg", "d_centre_deg"), "+ both")):
        e = G.preview_effect(task, covs, df)
        print(f"  preview effect {label:17s} {e['mean']:+9.2f} ms, 95% CI "
              f"[{e['ci'][0]:+.1f}, {e['ci'][1]:+.1f}], p = {e['p']:.3g}, "
              f"dz = {e['dz']:+.2f}")
    print()'''),
    md("""## Trials reached in a single fixation

The face detection task only, as in Supplementary Figure 2. On these trials the
whole of detection latency is one saccade launched from the onset position, so
the exchange rate between distance and latency is not diluted by the extra
fixations of longer searches. It is the strictest form of the reviewer's own
account, and the version the earlier draft of this control reported for
Experiment 1. The moving-window task is not shown: only a small minority of its
trials end this way.

Reported here as a robustness check rather than as the primary analysis, so that
both experiments are analysed on the same trials."""),
    code('''task = "facedetection"
hits = TABLES[task][TABLES[task]["n_fixations"] == 1]
print(f"{cfg.TASK_LABEL[task]}: {len(hits)}/{len(TABLES[task])} trials "
      f"({100 * len(hits) / len(TABLES[task]):.1f} %) ended on the first fixation\\n")

r = G.displacement(task, hits)["d_face_deg"]
print(f"  distance to face centroid  diff {r['diff']:+.2f} deg, 95% CI "
      f"[{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}], p = {r['p']:.3g}")
s = G.distance_slope(task, "d_face_deg", hits)
print(f"  latency on gaze-to-face    {s['mean']:+.2f} ms/deg, p = {s['p']:.3g}")
b = G.orienting_bound(task, hits)
print(f"  maximum attributable to orienting: {b['ms']:.1f} ms")
for covs, label in (((), "unadjusted"),
                    (("d_face_deg", "d_centre_deg"), "+ both")):
    e = G.preview_effect(task, covs, hits)
    print(f"  preview effect {label:12s} {e['mean']:+7.2f} ms, 95% CI "
          f"[{e['ci'][0]:+.1f}, {e['ci'][1]:+.1f}], p = {e['p']:.3g}")'''),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in (("figure1_behaviour.ipynb", FIG1),
                        ("figure3_encoding.ipynb", FIG3),
                        ("supplementary_behaviour.ipynb", SUPP_BEHAVIOUR),
                        ("supplementary_encoding.ipynb", SUPP_ENCODING),
                        ("supplementary_cnn.ipynb", SUPP_CNN),
                        ("supplementary_onset_gaze.ipynb", SUPP_ONSET_GAZE)):
        with open(OUT / name, "w") as f:
            json.dump(notebook(cells), f, indent=1)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
