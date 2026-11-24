#!/usr/bin/env python3
"""
Behavioural results: detection latency, fixation counts, scan path ratio, and
first-fixation accuracy, for both experiments.

Reports raw means for description and runs the statistics on the scale stated
for each measure. Reproduces the values quoted in the Results section and in
Supplementary Figures 1-3.

Run: python3 run_behavioural_analyses.py
"""

from facescenes import behaviour as B
from facescenes import config as cfg


def show(name, r, unit="", log=True):
    print(f"  {name}")
    print(f"     raw means       preview {r['mean_a']:9.3f} +/- {r['sem_a']:.3f} {unit}"
          f"   no preview {r['mean_b']:9.3f} +/- {r['sem_b']:.3f} {unit}")
    if log and "geo_a" in r:
        print(f"     geometric means preview {r['geo_a']:9.3f} {unit}"
              f"   no preview {r['geo_b']:9.3f} {unit}   ({r['pct_change']:+.2f} %)")
        print(f"     log difference  {r['diff']:+.4f}, "
              f"95% CI [{r['diff_ci'][0]:+.4f}, {r['diff_ci'][1]:+.4f}]")
    else:
        print(f"     difference      {r['diff']:+.4f}, "
              f"95% CI [{r['diff_ci'][0]:+.4f}, {r['diff_ci'][1]:+.4f}] {unit}")
    scale = "log" if (log and "geo_a" in r) else "raw"
    print(f"     test on {scale}:   t({r['n'] - 1}) = {r['t']:.3f}, p = {r['p']:.4g}, "
          f"dz = {r['dz']:.2f}, 95% CI [{r['dz_ci'][0]:.2f}, {r['dz_ci'][1]:.2f}]")


def main():
    for task in cfg.TASKS:
        df = B.trial_table(task)
        n_sub = df["participant"].nunique()
        print("=" * 78)
        print(f"{cfg.TASK_LABEL[task]}   {n_sub} participants, {len(df)} usable trials")
        print("=" * 78)

        show("detection latency", B.detection_latency(task), "ms")
        print()
        show("fixations to reach the face", B.fixation_count(task), log=False)
        print()

        ff = B.first_fixation_on_face(task)
        print(f"  first fixation landed on the face: "
              f"{ff['n_hit']}/{ff['n_total']} trials ({ff['pct_overall']:.1f} %)")
        show("   accuracy (% of trials)", ff["accuracy"], "%", log=False)
        show("   latency on those trials", ff["latency"], "ms")
        print()

        if task == "movingwindow":
            # Not meaningful in the face detection task: observers reach the face
            # in roughly one fixation, so there is barely a path to measure.
            show("scan path ratio", B.scan_path_ratio(task))
        print()


if __name__ == "__main__":
    main()
