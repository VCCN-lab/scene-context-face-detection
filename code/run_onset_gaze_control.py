#!/usr/bin/env python3
"""
Control analysis of gaze position at target onset, for both experiments.

Tests whether the preview benefit could be an oculomotor effect: if the preview
lets observers move their eyes toward the upcoming face before the search
display appears, they start closer to it and reach it sooner, with no scene
guidance involved. Every other analysis in this release looks only at eye
movements AFTER onset and cannot address that.

Reproduces the values reported in Supplementary Note 8. See facescenes/onset_gaze
for how eye position at onset is recovered and why the regressions here are on
the raw millisecond scale rather than on logs.

Two robustness checks follow the main results, both reported in Supplementary
Note 8: whether the trials dropped for lack of a qualifying onset fixation are
balanced across conditions, and whether every result survives removing the
100 ms onset criterion that causes most of those drops.

Run: python3 run_onset_gaze_control.py
"""

import numpy as np

from facescenes import config as cfg
from facescenes import data as D
from facescenes import onset_gaze as G

DISTANCES = (("d_centre_deg", "from screen centre"),
             ("d_face_deg", "to face centroid"))

MODELS = (((), "unadjusted"),
          (("d_face_deg",), "+ gaze-to-face"),
          (("d_centre_deg",), "+ gaze-to-centre"),
          (("d_face_deg", "d_centre_deg"), "+ both"))


def report(task, df, label, models=MODELS):
    """The three questions, in order, for one set of trials."""
    disp = G.displacement(task, df)
    print(f"  {label}: {len(df)} trials\n")

    print("  1. is the eye displaced, and is it displaced toward the face?")
    for col, name in DISTANCES:
        r = disp[col]
        print(f"     {name:19s} {r['mean_a']:5.2f} vs {r['mean_b']:5.2f} deg   "
              f"diff {r['diff']:+.2f}, 95% CI "
              f"[{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}], "
              f"t({r['n'] - 1}) = {r['t']:.2f}, p = {r['p']:.4g}, "
              f"dz = {r['dz']:+.2f}")

    print("\n  2. does onset position affect latency at all? (positive control)")
    for col, name in (("d_face_deg", "gaze-to-face"),
                      ("d_centre_deg", "gaze-to-centre")):
        s = G.distance_slope(task, col, df)
        print(f"     latency on {name:15s} {s['mean']:+8.2f} ms/deg   "
              f"t({s['n'] - 1}) = {s['t']:.2f}, p = {s['p']:.4g}")
    b = G.orienting_bound(task, df)
    print(f"     most it could explain  {b['ms']:8.1f} ms   "
          f"({b['degrees']:+.2f} deg x {b['ms_per_degree']:.1f} ms/deg)")

    print("\n  3. does the benefit survive adjustment?")
    for covariates, name in models:
        e = G.preview_effect(task, covariates, df)
        print(f"     preview effect {name:17s} {e['mean']:+9.2f} ms, 95% CI "
              f"[{e['ci'][0]:+8.2f}, {e['ci'][1]:+8.2f}], p = {e['p']:.4g}, "
              f"dz = {e['dz']:+.2f}")


def main():
    for task in cfg.TASKS:
        df = G.onset_gaze_table(task)
        dropped = len(D.trials(task)) - len(df)
        print("=" * 78)
        print(f"{cfg.TASK_LABEL[task]}   {df['participant'].nunique()} participants")
        print(f"{dropped} trials dropped with no qualifying onset fixation "
              f"({100 * dropped / (len(df) + dropped):.1f} %)")
        print("=" * 78)
        report(task, df, "all successful trials")

        if task == "facedetection":
            # Latency here is a single saccade launched from the onset position,
            # so the exchange rate between distance and latency is not diluted
            # by the extra fixations of longer searches: the strictest form of
            # the account being tested. Not run for the moving-window task,
            # where few trials end this way.
            print()
            report(task, df[df["n_fixations"] == 1],
                   "trials reached in one fixation",
                   models=(MODELS[0], MODELS[-1]))
        print()

    for task in cfg.TASKS:
        print("=" * 78)
        print(f"{cfg.TASK_LABEL[task]}   robustness")
        print("=" * 78)

        # 1. Are the dropped trials balanced across conditions?
        e = G.exclusion_by_condition(task)
        print(f"  trials dropped   preview {e['mean_a']:5.2f} %   no preview "
              f"{e['mean_b']:5.2f} %   diff {e['diff']:+.2f} pp, "
              f"t({e['n'] - 1}) = {e['t']:.2f}, p = {e['p']:.3f}\n")

        # 2. Does anything change without the onset window? Only fixations
        # with no recorded position or shorter than MIN_ONSET_FIXATION_S are
        # then rejected, so almost no trials are lost.
        df = G.onset_gaze_table(task, window=np.inf)
        report(task, df, "without the onset window")
        if task == "facedetection":
            print()
            report(task, df[df["n_fixations"] == 1],
                   "without the onset window, reached in one fixation",
                   models=(MODELS[0], MODELS[-1]))
        print()


if __name__ == "__main__":
    main()
